#!/usr/bin/env python
# coding: utf-8
"""Extended QSAR evaluation metrics.

Standalone companions to ``training_orchestration.compute_regression_metrics`` and
``training_orchestration.compute_classification_metrics``. Every public function here is
pure: it takes lists or numpy arrays, touches no disk and no global state, and follows
the conventions of the existing metric helpers -- plain ``dict`` results from the
aggregators, ``float`` values, and ``None`` (never ``NaN``) for a metric that is
undefined for the given input.

Scoring functions that scikit-learn already implements exactly (MCC, PR-AUC, Cohen's
kappa) delegate to it; this module adds the input validation, the missing-value policy
and the ``None``-on-undefined contract around them. The QSAR-specific statistics
(CCC, the OECD external-validation Q2 variants, the Golbraikh-Tropsha criteria) are
implemented here from their published definitions.

References
----------
Consonni, V.; Ballabio, D.; Todeschini, R. "Comments on the Definition of the Q2
Parameter for QSAR Validation." J. Chem. Inf. Model. 2009, 49, 1669-1678.
Golbraikh, A.; Tropsha, A. "Beware of q2!" J. Mol. Graph. Model. 2002, 20, 269-276.
Lin, L. I-K. "A Concordance Correlation Coefficient to Evaluate Reproducibility."
Biometrics 1989, 45, 255-268.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics as sk_metrics

__all__ = [
    "average_precision",
    "brier_score",
    "cohen_kappa",
    "concordance_correlation_coefficient",
    "expected_calibration_error",
    "extended_classification_metrics",
    "extended_regression_metrics",
    "golbraikh_tropsha_criteria",
    "matthews_corrcoef",
    "q2_f1",
    "q2_f2",
    "q2_f3",
    "specificity",
]

ArrayLike = Sequence[Any] | np.ndarray


# --------------------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_degenerate_input() -> Iterator[None]:
    """Silence scikit-learn's degenerate-input warnings.

    Those cases (a single class present, a vanishing denominator) are reported here as
    a documented ``None``/``0.0``, so the warning would describe a value this module
    never returns.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _finite_or_none(value: float) -> Optional[float]:
    """Return ``value`` as a float, or ``None`` when it is not finite."""
    number = float(value)
    return number if math.isfinite(number) else None


def _as_array(values: ArrayLike, *, name: str) -> np.ndarray:
    """Coerce ``values`` to a one-dimensional numpy array."""
    array = np.asarray(values)
    if array.ndim > 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    return np.atleast_1d(array)


def _missing_mask(values: np.ndarray) -> np.ndarray:
    """Boolean mask of entries that are ``None`` or ``NaN``/infinite."""
    if values.dtype.kind in {"f", "c"}:
        return ~np.isfinite(values)
    if values.dtype.kind == "O":
        return np.array(
            [item is None or (isinstance(item, float) and math.isnan(item)) for item in values],
            dtype=bool,
        )
    return np.zeros(values.shape, dtype=bool)


def _pair_mask(truth: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Mask of index positions where both vectors hold a usable value."""
    if truth.size != predicted.size:
        raise ValueError(
            f"y_true and y_pred must have equal length ({truth.size} != {predicted.size})"
        )
    return ~(_missing_mask(truth) | _missing_mask(predicted))


def _paired_labels(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Align two label vectors of any dtype and drop pairs with a missing entry."""
    truth = _as_array(y_true, name="y_true")
    predicted = _as_array(y_pred, name="y_pred")
    keep = _pair_mask(truth, predicted)
    return truth[keep], predicted[keep]


def _paired_values(y_true: ArrayLike, y_pred: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Align two numeric vectors as floats and drop pairs with a non-finite entry."""
    truth = _as_array(y_true, name="y_true").astype(float)
    predicted = _as_array(y_pred, name="y_pred").astype(float)
    keep = _pair_mask(truth, predicted)
    return truth[keep], predicted[keep]


def _binary_codes(values: np.ndarray) -> Optional[np.ndarray]:
    """Return 0/1 float codes when the labels already are 0/1, otherwise ``None``."""
    try:
        codes = values.astype(float)
    except (TypeError, ValueError):
        return None
    if codes.size and not bool(np.all(np.isin(codes, (0.0, 1.0)))):
        return None
    return codes


def _paired_binary(y_true: ArrayLike, y_prob: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """Align 0/1 truth with positive-class probabilities, validating both ranges."""
    truth, probability = _paired_values(y_true, y_prob)
    if truth.size and not bool(np.all(np.isin(truth, (0.0, 1.0)))):
        raise ValueError("binary metrics require y_true encoded as 0 (negative) and 1 (positive)")
    if probability.size and (probability.min() < 0.0 or probability.max() > 1.0):
        raise ValueError("y_prob must hold positive-class probabilities inside [0, 1]")
    return truth, probability


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------


def matthews_corrcoef(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """Matthews correlation coefficient for binary or multiclass labels.

    Delegates to :func:`sklearn.metrics.matthews_corrcoef`. The coefficient is the
    Pearson correlation of the one-hot encoded truth and prediction; it is ``+1`` for a
    perfect prediction, ``-1`` for a fully inverted binary prediction and ``0`` for a
    prediction uncorrelated with the truth.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted class labels of equal length. Any label dtype is
        accepted; pairs containing ``None``/``NaN`` are dropped.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains. ``0.0`` when the denominator vanishes,
        which happens whenever truth or prediction is constant -- that case is
        undefined as a correlation and is reported as "no association" by convention.
    """
    truth, predicted = _paired_labels(y_true, y_pred)
    if truth.size == 0:
        return None
    with _quiet_degenerate_input():
        return _finite_or_none(sk_metrics.matthews_corrcoef(truth, predicted))


def cohen_kappa(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """Cohen's kappa: agreement corrected for the agreement expected by chance.

    Delegates to :func:`sklearn.metrics.cohen_kappa_score`. Equals ``1`` for perfect
    agreement and ``0`` when agreement matches chance expectation.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains, or when the statistic is undefined because
        truth and prediction share a single label. ``0.0`` when observed agreement
        exactly matches the chance expectation.
    """
    truth, predicted = _paired_labels(y_true, y_pred)
    if truth.size == 0:
        return None
    with _quiet_degenerate_input():
        return _finite_or_none(sk_metrics.cohen_kappa_score(truth, predicted))


def average_precision(y_true: ArrayLike, y_prob: ArrayLike) -> Optional[float]:
    """Average precision (PR-AUC) for a binary problem.

    Delegates to :func:`sklearn.metrics.average_precision_score`, which sums
    ``(R_n - R_{n-1}) * P_n`` over the precision/recall points of the ranking rather
    than trapezoidally interpolating them.

    Parameters
    ----------
    y_true
        Observed labels encoded as 0/1.
    y_prob
        Positive-class scores or probabilities, same length as ``y_true``.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains or when only one class is present, in
        which case average precision is undefined (scikit-learn would return ``0.0``
        with a warning).
    """
    truth, probability = _paired_binary(y_true, y_prob)
    if truth.size == 0:
        return None
    if truth.min() == truth.max():
        return None
    return _finite_or_none(sk_metrics.average_precision_score(truth, probability))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> Optional[float]:
    """Brier score ``mean((p_i - y_i) ** 2)`` for binary positive-class probabilities.

    Lower is better: ``0.0`` for perfectly confident and correct probabilities, ``1.0``
    for perfectly confident and wrong ones.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains.
    """
    truth, probability = _paired_binary(y_true, y_prob)
    if truth.size == 0:
        return None
    return _finite_or_none(np.mean((probability - truth) ** 2))


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    *,
    n_bins: int = 10,
) -> Optional[float]:
    """Expected calibration error of binary positive-class probabilities.

    Probabilities are grouped into ``n_bins`` equal-width bins over ``[0, 1]`` (bin
    ``b`` covers ``[b / n_bins, (b + 1) / n_bins)``, the last bin including ``1.0``).
    The error is the sample-weighted mean gap between the predicted and the observed
    positive rate of each non-empty bin::

        ECE = sum_b (|B_b| / n) * |mean_{i in B_b}(p_i) - mean_{i in B_b}(y_i)|

    Parameters
    ----------
    y_true
        Observed labels encoded as 0/1.
    y_prob
        Positive-class probabilities inside ``[0, 1]``.
    n_bins
        Number of equal-width bins, at least 1.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains. ``0.0`` means every bin's predicted rate
        matches its observed rate, which is calibration and not accuracy.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    truth, probability = _paired_binary(y_true, y_prob)
    if truth.size == 0:
        return None

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, n_bins - 1)
    total = float(truth.size)
    error = 0.0
    for index in range(n_bins):
        in_bin = assignments == index
        count = int(in_bin.sum())
        if count == 0:
            continue
        gap = abs(float(probability[in_bin].mean()) - float(truth[in_bin].mean()))
        error += (count / total) * gap
    return _finite_or_none(error)


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """True negative rate ``TN / (TN + FP)`` for a binary problem.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted labels encoded as 0 (negative) and 1 (positive).

    Returns
    -------
    float or None
        ``None`` when no valid pair remains or when the sample contains no actual
        negative, in which case the rate is undefined.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    for values, name in ((truth, "y_true"), (predicted, "y_pred")):
        if values.size and not bool(np.all(np.isin(values, (0.0, 1.0)))):
            raise ValueError(
                f"specificity requires {name} encoded as 0 (negative) and 1 (positive)"
            )
    negatives = int((truth == 0.0).sum())
    if negatives == 0:
        return None
    true_negatives = int(((truth == 0.0) & (predicted == 0.0)).sum())
    return _finite_or_none(true_negatives / negatives)


# --------------------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------------------


def concordance_correlation_coefficient(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """Lin's concordance correlation coefficient (CCC).

    Measures agreement with the 45-degree line through the origin, penalising both
    poor correlation and systematic bias::

        CCC = 2 * cov(y, yhat) / (var(y) + var(yhat) + (mean(y) - mean(yhat)) ** 2)

    Variance and covariance use the population (``1 / n``) convention, as in Lin
    (1989). ``CCC(y, y) == 1`` for any non-constant ``y``.

    Returns
    -------
    float or None
        ``None`` when fewer than two valid pairs remain, or when the denominator
        vanishes (both vectors constant and equal), where the coefficient is undefined.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    if truth.size < 2:
        return None
    truth_mean = float(truth.mean())
    predicted_mean = float(predicted.mean())
    truth_variance = float(((truth - truth_mean) ** 2).mean())
    predicted_variance = float(((predicted - predicted_mean) ** 2).mean())
    covariance = float(((truth - truth_mean) * (predicted - predicted_mean)).mean())
    denominator = truth_variance + predicted_variance + (truth_mean - predicted_mean) ** 2
    if denominator <= 0.0:
        return None
    return _finite_or_none(2.0 * covariance / denominator)


def q2_f1(y_true: ArrayLike, y_pred: ArrayLike, y_train_mean: float) -> Optional[float]:
    """OECD external-validation statistic Q2F1 (Shi et al. / Schuurmann et al. form).

    Residuals over the external set are referenced to the *training* mean::

        Q2F1 = 1 - sum_ext (y_i - yhat_i) ** 2 / sum_ext (y_i - mean(y_train)) ** 2

    Both sums run over the external (test) set; only the centring value comes from the
    training set. ``y_train_mean`` is required rather than defaulting to the test mean,
    because substituting it silently would compute Q2F2 under the Q2F1 name.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values of the external set.
    y_train_mean
        Arithmetic mean of the training-set responses.

    Returns
    -------
    float or None
        ``None`` when no valid pair remains or when the denominator vanishes (every
        external response equals the training mean). ``1.0`` for a perfect prediction.

    References
    ----------
    Consonni, Ballabio and Todeschini, J. Chem. Inf. Model. 2009, 49, 1669-1678, eq. 6.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    if truth.size == 0:
        return None
    reference = float(y_train_mean)
    if not math.isfinite(reference):
        raise ValueError("y_train_mean must be a finite number")
    denominator = float(((truth - reference) ** 2).sum())
    if denominator <= 0.0:
        return None
    residual_sum = float(((truth - predicted) ** 2).sum())
    return _finite_or_none(1.0 - residual_sum / denominator)


def q2_f2(y_true: ArrayLike, y_pred: ArrayLike) -> Optional[float]:
    """OECD external-validation statistic Q2F2 (Schuurmann et al. form).

    Residuals over the external set are referenced to the *external* mean, which makes
    Q2F2 numerically identical to the coefficient of determination of the test set::

        Q2F2 = 1 - sum_ext (y_i - yhat_i) ** 2 / sum_ext (y_i - mean(y_ext)) ** 2

    Returns
    -------
    float or None
        ``None`` when no valid pair remains or when the external responses are
        constant, which makes the denominator vanish. ``1.0`` for a perfect prediction.

    References
    ----------
    Consonni, Ballabio and Todeschini, J. Chem. Inf. Model. 2009, 49, 1669-1678, eq. 7.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    if truth.size == 0:
        return None
    denominator = float(((truth - float(truth.mean())) ** 2).sum())
    if denominator <= 0.0:
        return None
    residual_sum = float(((truth - predicted) ** 2).sum())
    return _finite_or_none(1.0 - residual_sum / denominator)


def q2_f3(y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike) -> Optional[float]:
    """OECD external-validation statistic Q2F3 (Consonni, Ballabio and Todeschini).

    Compares the mean squared error of the external set with the mean squared deviation
    of the *training* responses about the training mean, so each sum is scaled by the
    size of the set it runs over::

        Q2F3 = 1 - [sum_ext (y_i - yhat_i) ** 2 / n_ext]
                   / [sum_train (y_j - mean(y_train)) ** 2 / n_train]

    The full ``y_train`` vector is required, not just its mean: the denominator needs
    the training variance and the training set size.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values of the external set.
    y_train
        Training-set responses.

    Returns
    -------
    float or None
        ``None`` when either set is empty after dropping non-finite values, or when the
        training responses are constant. ``1.0`` for a perfect prediction.

    References
    ----------
    Consonni, Ballabio and Todeschini, J. Chem. Inf. Model. 2009, 49, 1669-1678, eq. 8.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    training = _as_array(y_train, name="y_train").astype(float)
    training = training[np.isfinite(training)]
    if truth.size == 0 or training.size == 0:
        return None
    training_variance = float(((training - float(training.mean())) ** 2).sum()) / training.size
    if training_variance <= 0.0:
        return None
    external_mse = float(((truth - predicted) ** 2).sum()) / truth.size
    return _finite_or_none(1.0 - external_mse / training_variance)


def _pearson_r2(truth: np.ndarray, predicted: np.ndarray) -> Optional[float]:
    """Squared Pearson correlation of two aligned float vectors."""
    truth_centered = truth - float(truth.mean())
    predicted_centered = predicted - float(predicted.mean())
    denominator = math.sqrt(float((truth_centered**2).sum()) * float((predicted_centered**2).sum()))
    if denominator <= 0.0:
        return None
    correlation = float((truth_centered * predicted_centered).sum()) / denominator
    return _finite_or_none(correlation**2)


def golbraikh_tropsha_criteria(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    q2: float,
) -> Dict[str, Any]:
    """Golbraikh-Tropsha acceptability criteria for an external prediction set.

    The quantities, all computed on the external set with observed ``y`` and predicted
    ``yhat`` (regressions through the origin, as in the original paper)::

        r2     = squared Pearson correlation of y and yhat
        k      = sum(y * yhat) / sum(yhat ** 2)          slope of y on yhat
        k'     = sum(y * yhat) / sum(y ** 2)             slope of yhat on y
        r02    = 1 - sum((y - k * yhat) ** 2) / sum((y - mean(y)) ** 2)
        r02'   = 1 - sum((yhat - k' * y) ** 2) / sum((yhat - mean(yhat)) ** 2)

    The criteria, each reported as its own boolean:

    ``q2_above_0_5``
        ``q2 > 0.5`` -- the internal (cross-validated) q2 of the training set, which
        cannot be derived from the external set and therefore must be supplied.
    ``r2_above_0_6``
        ``r2 > 0.6`` -- the external set is predicted with useful correlation.
    ``r2_r02_close`` / ``r2_r02_prime_close``
        ``|r2 - r02| / r2 < 0.1`` -- the through-origin fit is nearly as good as the
        free fit, i.e. the prediction carries no large systematic offset.
    ``k_in_range`` / ``k_prime_in_range``
        ``0.85 <= k <= 1.15`` -- the through-origin slope is near unity, i.e. the
        prediction is not systematically compressed or inflated.
    ``r02_difference_small``
        ``|r02 - r02'| < 0.3`` -- the two through-origin fits agree.

    ``passes`` is ``True`` only when ``q2_above_0_5``, ``r2_above_0_6`` and
    ``r02_difference_small`` hold and at least one of the two primed/unprimed
    ``(closeness, slope)`` pairs holds, as specified by Golbraikh and Tropsha (2002).

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values of the external set.
    q2
        Internal cross-validated q2 of the model, supplied by the caller.

    Returns
    -------
    dict
        The five quantities above (``float`` or ``None`` when undefined), every
        criterion as a ``bool``, ``passes`` and ``n``. A criterion whose quantity is
        undefined is reported as ``False``: an uncomputable criterion is never met.

    References
    ----------
    Golbraikh, A.; Tropsha, A. J. Mol. Graph. Model. 2002, 20, 269-276.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    q2_value = float(q2)
    if not math.isfinite(q2_value):
        raise ValueError("q2 must be a finite number")

    r2 = _pearson_r2(truth, predicted) if truth.size else None
    predicted_square_sum = float((predicted**2).sum())
    truth_square_sum = float((truth**2).sum())
    cross_sum = float((truth * predicted).sum())
    k = cross_sum / predicted_square_sum if predicted_square_sum > 0.0 else None
    k_prime = cross_sum / truth_square_sum if truth_square_sum > 0.0 else None

    truth_spread = float(((truth - float(truth.mean())) ** 2).sum()) if truth.size else 0.0
    predicted_spread = (
        float(((predicted - float(predicted.mean())) ** 2).sum()) if predicted.size else 0.0
    )
    r02: Optional[float] = None
    if k is not None and truth_spread > 0.0:
        r02 = _finite_or_none(1.0 - float(((truth - k * predicted) ** 2).sum()) / truth_spread)
    r02_prime: Optional[float] = None
    if k_prime is not None and predicted_spread > 0.0:
        r02_prime = _finite_or_none(
            1.0 - float(((predicted - k_prime * truth) ** 2).sum()) / predicted_spread
        )

    def _relative_gap_small(reference: Optional[float], through_origin: Optional[float]) -> bool:
        if reference is None or through_origin is None or reference <= 0.0:
            return False
        return abs(reference - through_origin) / reference < 0.1

    r2_above = r2 is not None and r2 > 0.6
    q2_above = q2_value > 0.5
    r2_r02_close = _relative_gap_small(r2, r02)
    r2_r02_prime_close = _relative_gap_small(r2, r02_prime)
    k_in_range = k is not None and 0.85 <= k <= 1.15
    k_prime_in_range = k_prime is not None and 0.85 <= k_prime <= 1.15
    r02_difference_small = r02 is not None and r02_prime is not None and abs(r02 - r02_prime) < 0.3
    passes = bool(
        r2_above
        and q2_above
        and r02_difference_small
        and ((r2_r02_close and k_in_range) or (r2_r02_prime_close and k_prime_in_range))
    )
    return {
        "r2": r2,
        "q2": _finite_or_none(q2_value),
        "r02": r02,
        "r02_prime": r02_prime,
        "k": _finite_or_none(k) if k is not None else None,
        "k_prime": _finite_or_none(k_prime) if k_prime is not None else None,
        "r2_above_0_6": bool(r2_above),
        "q2_above_0_5": bool(q2_above),
        "r2_r02_close": bool(r2_r02_close),
        "r2_r02_prime_close": bool(r2_r02_prime_close),
        "k_in_range": bool(k_in_range),
        "k_prime_in_range": bool(k_prime_in_range),
        "r02_difference_small": bool(r02_difference_small),
        "passes": passes,
        "n": int(truth.size),
    }


# --------------------------------------------------------------------------------------
# aggregators
# --------------------------------------------------------------------------------------


def extended_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_prob: Optional[ArrayLike] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Extended classification metrics, ready to merge into ``compute_classification_metrics``.

    Always reports ``mcc``, ``cohen_kappa`` and ``n``. ``specificity`` is added when the
    labels are binary 0/1. ``average_precision``, ``brier_score`` and
    ``expected_calibration_error`` are added only when ``y_prob`` is supplied for a
    binary 0/1 problem; otherwise they are omitted rather than raising.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted class labels.
    y_prob
        Optional positive-class probabilities for a binary problem.
    n_bins
        Bin count for the expected calibration error.

    Returns
    -------
    dict
        Metric name to ``float``/``None`` value, plus ``n``. Returns ``{}`` when no
        valid pair remains.
    """
    truth_raw = _as_array(y_true, name="y_true")
    predicted_raw = _as_array(y_pred, name="y_pred")
    keep = _pair_mask(truth_raw, predicted_raw)
    truth = truth_raw[keep]
    predicted = predicted_raw[keep]
    if truth.size == 0:
        return {}

    metrics: Dict[str, Any] = {
        "mcc": matthews_corrcoef(truth, predicted),
        "cohen_kappa": cohen_kappa(truth, predicted),
        "n": int(truth.size),
    }

    truth_codes = _binary_codes(truth)
    predicted_codes = _binary_codes(predicted)
    if truth_codes is None or predicted_codes is None:
        return metrics

    metrics["specificity"] = specificity(truth_codes, predicted_codes)
    if y_prob is None:
        return metrics

    probability = _as_array(y_prob, name="y_prob").astype(float)
    if probability.size != truth_raw.size:
        raise ValueError(
            f"y_prob must have the same length as y_true ({probability.size} != {truth_raw.size})"
        )
    probability = probability[keep]
    valid = np.isfinite(probability)
    metrics["average_precision"] = average_precision(truth_codes[valid], probability[valid])
    metrics["brier_score"] = brier_score(truth_codes[valid], probability[valid])
    metrics["expected_calibration_error"] = expected_calibration_error(
        truth_codes[valid], probability[valid], n_bins=n_bins
    )
    return metrics


def extended_regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    y_train: Optional[ArrayLike] = None,
    q2: Optional[float] = None,
) -> Dict[str, Any]:
    """Extended regression metrics, ready to merge into ``compute_regression_metrics``.

    Always reports ``ccc``, ``q2_f2`` and ``n`` -- the statistics computable from the
    external set alone. ``q2_f1`` and ``q2_f3`` are added only when ``y_train`` is
    supplied, because their denominators reference the training set; they are omitted
    rather than approximated. ``golbraikh_tropsha`` (a nested dict) is added only when
    an internal ``q2`` is supplied.

    Parameters
    ----------
    y_true, y_pred
        Observed and predicted values of the external set.
    y_train
        Optional training-set responses.
    q2
        Optional internal cross-validated q2 of the model.

    Returns
    -------
    dict
        Metric name to value, plus ``n``. Returns ``{}`` when no valid pair remains.
    """
    truth, predicted = _paired_values(y_true, y_pred)
    if truth.size == 0:
        return {}

    metrics: Dict[str, Any] = {
        "ccc": concordance_correlation_coefficient(truth, predicted),
        "q2_f2": q2_f2(truth, predicted),
        "n": int(truth.size),
    }
    if y_train is not None:
        training = _as_array(y_train, name="y_train").astype(float)
        training = training[np.isfinite(training)]
        if training.size:
            metrics["q2_f1"] = q2_f1(truth, predicted, float(training.mean()))
            metrics["q2_f3"] = q2_f3(truth, predicted, training)
    if q2 is not None:
        metrics["golbraikh_tropsha"] = golbraikh_tropsha_criteria(truth, predicted, q2=q2)
    return metrics
