#!/usr/bin/env python
# coding: utf-8
"""Backend-agnostic inductive (split) conformal prediction.

The calibrators here consume already-computed predictions on a held-out
calibration set, never a model object, so they apply to every prediction
backend regardless of whether that backend reports uncertainty itself.

References
----------
Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* (2005);
Papadopoulos et al., *Inductive Confidence Machines for Regression* (2002);
Lei et al., *Distribution-Free Predictive Inference for Regression* (2018).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ConformalPredictionError",
    "InductiveConformalClassifier",
    "InductiveConformalRegressor",
    "conformal_quantile_index",
    "empirical_coverage",
    "empirical_set_coverage",
    "minimum_calibration_size",
]

# Floating-point slack used only to recover exact integers from products such as
# (n + 1) * (1 - alpha), which are integral in exact arithmetic but may land a
# few ULP above the integer in binary floating point.
_INTEGER_EPS = 1e-9

# Tolerance on probability inputs, so that a backend emitting 1.0000000002 is
# accepted while a genuinely out-of-range column is rejected.
_PROBA_EPS = 1e-6


class ConformalPredictionError(ValueError):
    """Raised when conformal calibration inputs or requests are unusable."""


def minimum_calibration_size(alpha: float) -> int:
    """Return the smallest calibration set size that supports ``alpha``.

    The corrected rank ``ceil((n + 1) * (1 - alpha))`` must not exceed ``n``,
    which rearranges to ``n >= 1 / alpha - 1``; hence ``ceil(1 / alpha) - 1``.

    Parameters
    ----------
    alpha
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    int
        Minimum number of calibration points.
    """
    _validate_alpha(alpha)
    return max(1, int(math.ceil(1.0 / float(alpha) - _INTEGER_EPS)) - 1)


def conformal_quantile_index(n_calib: int, alpha: float) -> int:
    """Return the finite-sample-corrected rank ``k = ceil((n + 1) * (1 - alpha))``.

    The split-conformal guarantee ``P(y_test in C(x_test)) >= 1 - alpha`` holds
    when the threshold is the ``k``-th *smallest* calibration score with
    ``k = ceil((n + 1) * (1 - alpha))``, i.e. the ``k / n`` empirical quantile.
    Using ``ceil(n * (1 - alpha))`` or a linearly interpolated
    ``numpy.quantile(scores, 1 - alpha)`` instead is the classic off-by-one that
    silently under-covers, so the rank is computed explicitly here.

    Parameters
    ----------
    n_calib
        Number of calibration scores.
    alpha
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    int
        One-based rank of the calibration score to use as threshold.

    Raises
    ------
    ConformalPredictionError
        If ``alpha`` is out of range, ``n_calib`` is non-positive, or the
        calibration set is too small to support ``alpha``.
    """
    _validate_alpha(alpha)
    if int(n_calib) < 1:
        raise ConformalPredictionError(f"n_calib must be >= 1, got {n_calib}.")
    n = int(n_calib)
    rank = int(math.ceil((n + 1) * (1.0 - float(alpha)) - _INTEGER_EPS))
    if rank > n:
        required = minimum_calibration_size(alpha)
        raise ConformalPredictionError(
            f"alpha={alpha!r} is too small for n_calib={n}: the corrected rank "
            f"ceil((n + 1) * (1 - alpha)) = {rank} exceeds n_calib. "
            f"At least {required} calibration points are required for this alpha, "
            f"or use alpha >= {1.0 / (n + 1):.6g}."
        )
    return rank


def empirical_coverage(
    y_true: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    alpha: float = 0.1,
    tolerance: Optional[float] = None,
    n_calib: Optional[int] = None,
) -> Dict[str, Any]:
    """Return realised coverage of prediction intervals against the target.

    Parameters
    ----------
    y_true
        Observed values.
    lower, upper
        Interval bounds, as returned by :meth:`InductiveConformalRegressor.predict_interval`.
    alpha
        Miscoverage level the intervals were built for; the target is ``1 - alpha``.
    tolerance
        Absolute two-sided tolerance on ``coverage - target``. ``None`` derives one
        from the sampling noise (see ``n_calib``).
    n_calib
        Calibration set size the intervals came from, available as ``n_calib`` in
        the ``predict_interval`` result. Supplying it widens the default tolerance
        to include calibration-set variability, which usually dominates; omitting
        it gives an optimistically narrow band.

    Returns
    -------
    dict
        ``coverage``, ``target``, ``n``, ``within_tolerance``, ``mean_width``
        and the ``tolerance`` actually applied.
    """
    _validate_alpha(alpha)
    y = _as_1d_float(y_true, "y_true")
    low = _as_1d_float(lower, "lower")
    high = _as_1d_float(upper, "upper")
    _require_same_size(("y_true", y), ("lower", low), ("upper", high))
    if y.size == 0:
        raise ConformalPredictionError("empirical_coverage requires at least one observation.")

    covered = (y >= low) & (y <= high)
    coverage = float(np.mean(covered))
    target = 1.0 - float(alpha)
    applied = _coverage_tolerance(target, int(y.size), tolerance, n_calib)
    return {
        "coverage": coverage,
        "target": target,
        "n": int(y.size),
        "within_tolerance": bool(abs(coverage - target) <= applied),
        "mean_width": float(np.mean(high - low)),
        "tolerance": applied,
    }


def empirical_set_coverage(
    y_true: ArrayLike,
    sets: Sequence[Sequence[Any]],
    *,
    alpha: float = 0.1,
    tolerance: Optional[float] = None,
    n_calib: Optional[int] = None,
) -> Dict[str, Any]:
    """Return realised coverage of classification prediction sets.

    Parameters
    ----------
    y_true
        Observed labels.
    sets
        One label collection per row, as returned by
        :meth:`InductiveConformalClassifier.predict_set`.
    alpha
        Miscoverage level the sets were built for; the target is ``1 - alpha``.
    tolerance
        Absolute two-sided tolerance on ``coverage - target``. ``None`` derives one
        from the sampling noise (see ``n_calib``).
    n_calib
        Calibration set size the sets came from, available as ``n_calib`` in the
        ``predict_set`` result. Supplying it widens the default tolerance to
        include calibration-set variability, which usually dominates.

    Returns
    -------
    dict
        ``coverage``, ``target``, ``n``, ``within_tolerance``, ``mean_set_size``
        and the ``tolerance`` actually applied.
    """
    _validate_alpha(alpha)
    labels = [_py_scalar(value) for value in np.asarray(y_true).ravel()]
    rows = [{_py_scalar(value) for value in row} for row in sets]
    if len(labels) != len(rows):
        raise ConformalPredictionError(
            f"y_true has {len(labels)} entries but sets has {len(rows)}."
        )
    if not labels:
        raise ConformalPredictionError("empirical_set_coverage requires at least one observation.")

    covered = [label in row for label, row in zip(labels, rows, strict=True)]
    coverage = float(np.mean(covered))
    target = 1.0 - float(alpha)
    applied = _coverage_tolerance(target, len(labels), tolerance, n_calib)
    return {
        "coverage": coverage,
        "target": target,
        "n": len(labels),
        "within_tolerance": bool(abs(coverage - target) <= applied),
        "mean_set_size": float(np.mean([len(row) for row in rows])),
        "tolerance": applied,
    }


class InductiveConformalRegressor:
    """Inductive conformal regressor calibrated on held-out predictions.

    Nonconformity is the absolute residual ``|y - y_hat|``, optionally divided by
    a supplied per-sample difficulty estimate ``sigma`` (normalised ICP), which
    yields intervals whose width varies with predicted difficulty.
    """

    def __init__(self) -> None:
        self._scores: Optional[np.ndarray] = None
        self._normalized: bool = False

    @property
    def n_calib(self) -> int:
        """Number of stored calibration scores (0 before :meth:`fit`)."""
        return 0 if self._scores is None else int(self._scores.size)

    @property
    def normalized(self) -> bool:
        """Whether the calibrator was fitted with a difficulty estimate."""
        return self._normalized

    def fit(
        self,
        y_calib: ArrayLike,
        pred_calib: ArrayLike,
        sigma_calib: Optional[ArrayLike] = None,
    ) -> "InductiveConformalRegressor":
        """Store absolute-residual nonconformity scores from a calibration set.

        Parameters
        ----------
        y_calib
            Observed calibration values.
        pred_calib
            Model predictions for the same rows, produced without seeing them.
        sigma_calib
            Optional strictly positive difficulty estimate per row. When given,
            scores become ``|y - y_hat| / sigma`` and ``predict_interval``
            requires a matching ``sigma_test``.

        Returns
        -------
        InductiveConformalRegressor
            ``self``, for chaining.
        """
        y = _as_1d_float(y_calib, "y_calib")
        pred = _as_1d_float(pred_calib, "pred_calib")
        _require_same_size(("y_calib", y), ("pred_calib", pred))
        if y.size == 0:
            raise ConformalPredictionError("Calibration set is empty.")

        scores = np.abs(y - pred)
        normalized = sigma_calib is not None
        if normalized:
            sigma = _as_1d_float(sigma_calib, "sigma_calib")
            _require_same_size(("y_calib", y), ("sigma_calib", sigma))
            _require_positive(sigma, "sigma_calib")
            scores = scores / sigma
        if not np.all(np.isfinite(scores)):
            raise ConformalPredictionError("Nonconformity scores contain non-finite values.")

        self._scores = np.sort(scores)
        self._normalized = normalized
        return self

    def predict_interval(
        self,
        pred_test: ArrayLike,
        alpha: float = 0.1,
        sigma_test: Optional[ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Return two-sided prediction intervals with ``1 - alpha`` coverage.

        Parameters
        ----------
        pred_test
            Model predictions for the rows to bound.
        alpha
            Miscoverage level in ``(0, 1)``.
        sigma_test
            Difficulty estimate per test row. Required when, and only when, the
            calibrator was fitted with ``sigma_calib``.

        Returns
        -------
        dict
            ``lower``, ``upper``, ``width`` arrays plus ``quantile``, ``alpha``,
            ``n_calib`` and ``normalized``.
        """
        scores = self._require_fitted()
        pred = _as_1d_float(pred_test, "pred_test")
        quantile = _score_quantile(scores, alpha)

        if self._normalized:
            if sigma_test is None:
                raise ConformalPredictionError(
                    "This calibrator was fitted with sigma_calib, so sigma_test is required; "
                    "passing none would mix normalised scores with unnormalised predictions."
                )
            sigma = _as_1d_float(sigma_test, "sigma_test")
            _require_same_size(("pred_test", pred), ("sigma_test", sigma))
            _require_positive(sigma, "sigma_test")
            half_width = quantile * sigma
        else:
            if sigma_test is not None:
                raise ConformalPredictionError(
                    "This calibrator was fitted without sigma_calib, so sigma_test cannot be "
                    "applied; refit with sigma_calib to use normalised intervals."
                )
            half_width = np.full(pred.shape, quantile, dtype=float)

        lower = pred - half_width
        upper = pred + half_width
        return {
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
            "quantile": quantile,
            "alpha": float(alpha),
            "n_calib": int(scores.size),
            "normalized": self._normalized,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for storage alongside a model."""
        scores = self._require_fitted()
        return {
            "kind": "inductive_conformal_regressor",
            "normalized": self._normalized,
            "scores": [float(value) for value in scores],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InductiveConformalRegressor":
        """Rebuild a calibrator from :meth:`to_dict` output."""
        _require_kind(payload, "inductive_conformal_regressor")
        scores = _as_1d_float(payload.get("scores", []), "scores")
        if scores.size == 0:
            raise ConformalPredictionError("Serialized calibrator holds no calibration scores.")
        model = cls()
        model._scores = np.sort(scores)
        model._normalized = bool(payload.get("normalized", False))
        return model

    def _require_fitted(self) -> np.ndarray:
        if self._scores is None:
            raise ConformalPredictionError("Calibrator is not fitted; call fit() first.")
        return self._scores


class InductiveConformalClassifier:
    """Inductive conformal classifier producing label sets.

    Nonconformity is ``1 - p(true class)``. In the default marginal mode a single
    threshold is shared by all classes, which guarantees overall coverage but can
    starve a minority class. The class-conditional (Mondrian) mode calibrates one
    threshold per true class, so ``1 - alpha`` coverage holds within each class at
    the cost of larger sets.
    """

    def __init__(self, *, class_conditional: bool = False) -> None:
        self.class_conditional = bool(class_conditional)
        self._classes: List[Any] = []
        self._class_scores: Dict[Any, np.ndarray] = {}

    @property
    def classes(self) -> List[Any]:
        """Class labels in probability-column order."""
        return list(self._classes)

    @property
    def n_calib(self) -> int:
        """Total number of stored calibration scores."""
        return int(sum(scores.size for scores in self._class_scores.values()))

    def fit(
        self,
        y_calib: ArrayLike,
        proba_calib: ArrayLike,
        *,
        classes: Optional[Sequence[Any]] = None,
    ) -> "InductiveConformalClassifier":
        """Store ``1 - p(true class)`` nonconformity scores from a calibration set.

        Parameters
        ----------
        y_calib
            Observed calibration labels.
        proba_calib
            Predicted class probabilities, shape ``(n_rows, n_classes)``.
        classes
            Labels in column order. Defaults to the sorted unique labels in
            ``y_calib``; pass the model's own class order explicitly whenever the
            calibration split may not contain every class.

        Returns
        -------
        InductiveConformalClassifier
            ``self``, for chaining.
        """
        proba = _as_2d_proba(proba_calib, "proba_calib")
        y = np.asarray(y_calib).ravel()
        if y.size != proba.shape[0]:
            raise ConformalPredictionError(
                f"y_calib has {y.size} rows but proba_calib has {proba.shape[0]}."
            )
        if y.size == 0:
            raise ConformalPredictionError("Calibration set is empty.")

        if classes is None:
            labels = [_py_scalar(value) for value in np.unique(y)]
        else:
            labels = [_py_scalar(value) for value in np.asarray(classes).ravel()]
        if len(labels) != len(set(map(repr, labels))):
            raise ConformalPredictionError(f"classes contains duplicates: {labels}.")
        if len(labels) != proba.shape[1]:
            raise ConformalPredictionError(
                f"proba_calib has {proba.shape[1]} columns but {len(labels)} classes were "
                f"resolved ({labels}). Pass classes=... in the model's column order."
            )

        column_of = {label: index for index, label in enumerate(labels)}
        try:
            columns = np.array([column_of[_py_scalar(value)] for value in y], dtype=int)
        except KeyError as exc:
            raise ConformalPredictionError(
                f"Calibration label {exc.args[0]!r} is not among the classes {labels}."
            ) from exc

        scores = 1.0 - proba[np.arange(y.size), columns]
        self._classes = labels
        self._class_scores = {
            label: np.sort(scores[columns == index]) for index, label in enumerate(labels)
        }
        return self

    def predict_set(self, proba_test: ArrayLike, alpha: float = 0.1) -> Dict[str, Any]:
        """Return a prediction set per row, possibly empty or multi-label.

        Parameters
        ----------
        proba_test
            Predicted class probabilities, shape ``(n_rows, n_classes)``, in the
            same column order used for calibration.
        alpha
            Miscoverage level in ``(0, 1)``.

        Returns
        -------
        dict
            ``sets`` (list of label lists), ``set_size``, the ``quantiles`` used
            per class, ``alpha``, ``class_conditional`` and ``n_calib``.
        """
        if not self._classes:
            raise ConformalPredictionError("Calibrator is not fitted; call fit() first.")
        proba = _as_2d_proba(proba_test, "proba_test")
        if proba.shape[1] != len(self._classes):
            raise ConformalPredictionError(
                f"proba_test has {proba.shape[1]} columns but the calibrator was fitted on "
                f"{len(self._classes)} classes."
            )

        if self.class_conditional:
            quantiles = {label: self._class_quantile(label, alpha) for label in self._classes}
        else:
            pooled = np.sort(np.concatenate([self._class_scores[c] for c in self._classes]))
            shared = _score_quantile(pooled, alpha)
            quantiles = dict.fromkeys(self._classes, shared)

        thresholds = np.array([quantiles[label] for label in self._classes], dtype=float)
        # A label enters the set when its nonconformity 1 - p is no larger than the
        # calibrated threshold; `<=` keeps the guarantee conservative under ties.
        included = (1.0 - proba) <= thresholds[np.newaxis, :]
        sets = [
            [label for label, keep in zip(self._classes, row, strict=True) if keep]
            for row in included
        ]
        return {
            "sets": sets,
            "set_size": included.sum(axis=1).astype(int),
            "quantiles": quantiles,
            "alpha": float(alpha),
            "class_conditional": self.class_conditional,
            "n_calib": self.n_calib,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable snapshot for storage alongside a model."""
        if not self._classes:
            raise ConformalPredictionError("Calibrator is not fitted; call fit() first.")
        return {
            "kind": "inductive_conformal_classifier",
            "class_conditional": self.class_conditional,
            "classes": list(self._classes),
            "class_scores": [
                [float(value) for value in self._class_scores[label]] for label in self._classes
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InductiveConformalClassifier":
        """Rebuild a calibrator from :meth:`to_dict` output."""
        _require_kind(payload, "inductive_conformal_classifier")
        labels = [_py_scalar(value) for value in payload.get("classes", [])]
        raw_scores = list(payload.get("class_scores", []))
        if not labels or len(labels) != len(raw_scores):
            raise ConformalPredictionError(
                "Serialized classifier needs matching 'classes' and 'class_scores' entries."
            )
        model = cls(class_conditional=bool(payload.get("class_conditional", False)))
        model._classes = labels
        model._class_scores = {
            label: np.sort(_as_1d_float(scores, f"class_scores[{label!r}]"))
            for label, scores in zip(labels, raw_scores, strict=True)
        }
        return model

    def _class_quantile(self, label: Any, alpha: float) -> float:
        scores = self._class_scores[label]
        if scores.size == 0:
            raise ConformalPredictionError(
                f"Class {label!r} has no calibration rows, so a class-conditional threshold "
                "cannot be computed. Use the marginal mode or enlarge the calibration set."
            )
        try:
            return _score_quantile(scores, alpha)
        except ConformalPredictionError as exc:
            raise ConformalPredictionError(f"Class {label!r}: {exc}") from exc


def _score_quantile(sorted_scores: np.ndarray, alpha: float) -> float:
    """Return the conformal threshold: the k-th smallest score, k from the corrected rank."""
    rank = conformal_quantile_index(int(sorted_scores.size), alpha)
    # `rank` is one-based, so index rank - 1 into the ascending score array.
    return float(sorted_scores[rank - 1])


def _coverage_tolerance(
    target: float, n: int, tolerance: Optional[float], n_calib: Optional[int]
) -> float:
    """Return the absolute coverage tolerance, defaulting to two standard errors.

    Realised conformal coverage is noisy for two reasons: the finite evaluation
    set (binomial, variance ``p * (1 - p) / n``) and the finite calibration set,
    whose conditional coverage follows ``Beta(k, n_calib + 1 - k)`` with variance
    close to ``p * (1 - p) / (n_calib + 2)``. The second term usually dominates and
    is only included when ``n_calib`` is supplied, so a band computed without it
    is deliberately optimistic.
    """
    if tolerance is not None:
        value = float(tolerance)
        if value < 0.0:
            raise ConformalPredictionError(f"tolerance must be >= 0, got {tolerance!r}.")
        return value
    variance = target * (1.0 - target) / n
    if n_calib is not None:
        if int(n_calib) < 1:
            raise ConformalPredictionError(f"n_calib must be >= 1, got {n_calib!r}.")
        variance += target * (1.0 - target) / (int(n_calib) + 2)
    # Two percentage points is the floor below which a deviation is not actionable
    # for a QSAR report, however large the evaluation set is.
    return max(0.02, 2.0 * math.sqrt(variance))


def _validate_alpha(alpha: float) -> None:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ConformalPredictionError(f"alpha must lie strictly in (0, 1), got {alpha!r}.")


def _as_1d_float(values: ArrayLike, name: str) -> np.ndarray:
    """Return ``values`` as a finite 1-D float array."""
    try:
        array = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError) as exc:
        raise ConformalPredictionError(f"{name} is not numeric: {exc}") from exc
    if not np.all(np.isfinite(array)):
        raise ConformalPredictionError(f"{name} contains NaN or infinite values.")
    return array


def _as_2d_proba(values: ArrayLike, name: str) -> np.ndarray:
    """Return ``values`` as a finite ``(n_rows, n_classes)`` probability matrix."""
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ConformalPredictionError(f"{name} is not numeric: {exc}") from exc
    if array.ndim != 2:
        raise ConformalPredictionError(
            f"{name} must be a 2-D (n_rows, n_classes) array, got shape {array.shape}."
        )
    if array.size == 0:
        raise ConformalPredictionError(f"{name} is empty.")
    if not np.all(np.isfinite(array)):
        raise ConformalPredictionError(f"{name} contains NaN or infinite values.")
    if array.min() < -_PROBA_EPS or array.max() > 1.0 + _PROBA_EPS:
        raise ConformalPredictionError(
            f"{name} must hold probabilities in [0, 1], got range "
            f"[{array.min():.6g}, {array.max():.6g}]."
        )
    return np.clip(array, 0.0, 1.0)


def _require_same_size(*named: tuple[str, np.ndarray]) -> None:
    sizes = {name: int(array.size) for name, array in named}
    if len(set(sizes.values())) > 1:
        raise ConformalPredictionError(f"Length mismatch between inputs: {sizes}.")


def _require_positive(array: np.ndarray, name: str) -> None:
    if np.any(array <= 0.0):
        raise ConformalPredictionError(
            f"{name} must be strictly positive; got a minimum of {float(array.min()):.6g}."
        )


def _require_kind(payload: Mapping[str, Any], expected: str) -> None:
    kind = payload.get("kind")
    if kind != expected:
        raise ConformalPredictionError(f"Expected kind={expected!r} in payload, got {kind!r}.")


def _py_scalar(value: Any) -> Any:
    """Return a plain Python scalar for a numpy scalar, leaving other values as-is."""
    return value.item() if isinstance(value, np.generic) else value
