#!/usr/bin/env python
# coding: utf-8
"""Statistical validation helpers for QSAR model selection.

A metric computed on a single held-out split is a point estimate with no error
bar, and QSAR test sets are small enough that two models separated by 0.02 R2
are routinely indistinguishable.  This module supplies the missing uncertainty
layer:

* :func:`bootstrap_metric_ci` - percentile bootstrap interval for any metric.
* :func:`paired_bootstrap_comparison` - is model A really ahead of model B?
* :func:`wilcoxon_signed_rank_errors` - distribution-free per-compound test.
* :func:`y_scrambling` - chance-correlation control (Y-randomisation).
* :func:`benjamini_hochberg` - false-discovery control across a leaderboard.

Everything here is metric-agnostic and model-agnostic: the caller supplies the
metric, and for Y-scrambling the retraining closure, so no backend is imported.
Stochastic helpers take ``seed`` and are reproducible for a fixed seed.

Nothing is estimated silently.  Non-finite inputs, mismatched pair lengths and
metrics that blow up on a degenerate resample all raise
:class:`~cs_copilot.tools.prediction.backend.InvalidPredictionInputError`
rather than returning a plausible-looking number.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.stats import rankdata, wilcoxon

from .backend import InvalidPredictionInputError

ArrayLike = Union[Sequence[float], np.ndarray]
MetricFn = Callable[[np.ndarray, np.ndarray], float]
ScoreFn = Callable[[np.ndarray], float]

VERDICT_A_BETTER = "a_better"
VERDICT_B_BETTER = "b_better"
VERDICT_TIE = "tie"
VERDICT_TIE_ALL_ERRORS_EQUAL = "tie_all_errors_equal"
VERDICT_CHANCE_CORRELATION = "chance_correlation_suspected"
VERDICT_NO_CHANCE_CORRELATION = "no_chance_correlation_detected"

DEFAULT_N_RESAMPLES = 1000
DEFAULT_ALPHA = 0.05


def _as_float_array(values: ArrayLike, name: str) -> np.ndarray:
    """Return ``values`` as a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise InvalidPredictionInputError(
            f"{name} must be one-dimensional, got shape {array.shape}."
        )
    if array.size and not np.all(np.isfinite(array)):
        bad = int(np.count_nonzero(~np.isfinite(array)))
        raise InvalidPredictionInputError(
            f"{name} contains {bad} NaN or infinite value(s); drop or impute them before "
            "running statistical validation."
        )
    return array


def _check_paired(**named: ArrayLike) -> List[np.ndarray]:
    """Validate that every named array is finite, 1-D and of the same length."""
    arrays = {name: _as_float_array(values, name) for name, values in named.items()}
    sizes = {name: int(array.size) for name, array in arrays.items()}
    if len(set(sizes.values())) != 1:
        raise InvalidPredictionInputError(f"paired arrays must have equal length, got {sizes}.")
    size = next(iter(sizes.values()))
    if size < 2:
        raise InvalidPredictionInputError(
            f"at least 2 paired observations are required, got {size}."
        )
    return list(arrays.values())


def _check_resampling(n_resamples: int, alpha: float) -> None:
    """Validate the resampling budget and the two-sided interval level."""
    if int(n_resamples) < 2:
        raise InvalidPredictionInputError(
            f"n_resamples must be at least 2 for an interval, got {n_resamples}."
        )
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidPredictionInputError(f"alpha must lie strictly between 0 and 1, got {alpha}.")


def _finite_score(value: Any, context: str) -> float:
    """Return ``value`` as a float, rejecting NaN and infinities."""
    score = float(value)
    if not np.isfinite(score):
        raise InvalidPredictionInputError(f"{context} returned a non-finite score ({value!r}).")
    return score


def _reject_non_finite(scores: np.ndarray, context: str) -> None:
    """Raise if any resampled score is non-finite, since percentiles would be undefined."""
    if np.all(np.isfinite(scores)):
        return
    bad = int(np.count_nonzero(~np.isfinite(scores)))
    raise InvalidPredictionInputError(
        f"{context} returned a non-finite score on {bad}/{scores.size} bootstrap resamples. "
        "A percentile interval over non-finite scores is undefined; guard the metric (for "
        "example by requiring both classes to be present in a resample) and retry."
    )


def _resample_indices(rng: np.random.Generator, n_obs: int, n_resamples: int) -> np.ndarray:
    """Yield ``n_resamples`` index vectors drawn with replacement from ``range(n_obs)``."""
    for _ in range(int(n_resamples)):
        yield rng.integers(0, n_obs, size=n_obs)


def _percentile_bounds(scores: np.ndarray, alpha: float) -> Tuple[float, float]:
    """Return the two-sided ``alpha`` percentile bounds of ``scores``."""
    low, high = np.percentile(scores, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(low), float(high)


def bootstrap_metric_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: MetricFn,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """Percentile bootstrap confidence interval for a paired metric.

    Observations are resampled as ``(y_true[i], y_pred[i])`` pairs: the same index
    vector indexes both arrays on every draw.  Resampling the two arrays
    independently would destroy the correspondence and collapse the estimate
    towards the metric's chance level.

    Parameters
    ----------
    y_true, y_pred
        Finite, equal-length observed and predicted values.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``, called once on the observed data
        and once per resample.
    n_resamples
        Number of bootstrap draws (at least 2).
    alpha
        Two-sided level; the interval spans the ``alpha/2`` and ``1 - alpha/2``
        percentiles of the bootstrap distribution.
    seed
        Seed for :func:`numpy.random.default_rng`; fixed by default so repeated
        calls on the same data reproduce.

    Returns
    -------
    dict
        ``{"point", "ci_low", "ci_high", "n_resamples", "alpha"}``.

    Raises
    ------
    InvalidPredictionInputError
        If the inputs are not valid finite pairs, or if ``metric_fn`` returns a
        non-finite score on the observed data or on any resample.
    """
    observed_true, observed_pred = _check_paired(y_true=y_true, y_pred=y_pred)
    _check_resampling(n_resamples, alpha)

    point = _finite_score(metric_fn(observed_true, observed_pred), "metric_fn")
    rng = np.random.default_rng(seed)
    scores = np.fromiter(
        (
            float(metric_fn(observed_true[index], observed_pred[index]))
            for index in _resample_indices(rng, observed_true.size, n_resamples)
        ),
        dtype=float,
        count=int(n_resamples),
    )
    _reject_non_finite(scores, "metric_fn")

    ci_low, ci_high = _percentile_bounds(scores, alpha)
    return {
        "point": point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples": int(n_resamples),
        "alpha": float(alpha),
    }


def paired_bootstrap_comparison(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    metric_fn: MetricFn,
    higher_is_better: bool,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> Dict[str, Any]:
    """Compare two models on the same test set with a paired bootstrap.

    Every draw resamples one index vector and scores *both* models on it, so the
    shared test-set noise cancels out of the difference.  This is strictly more
    sensitive than comparing two independent confidence intervals.

    Parameters
    ----------
    y_true
        Finite observed values.
    pred_a, pred_b
        Predictions of the two models for the same compounds, in the same order.
    metric_fn
        ``metric_fn(y_true, y_pred) -> float``.
    higher_is_better
        Orientation of the metric; required, because an RMSE comparison read as
        if higher were better inverts the verdict.  ``delta`` itself is always
        reported as ``score_a - score_b`` in metric units, never sign-flipped.
    n_resamples, alpha, seed
        As in :func:`bootstrap_metric_ci`.

    Returns
    -------
    dict
        ``{"delta", "ci_low", "ci_high", "p_value", "verdict"}``, where ``verdict``
        is ``"a_better"``, ``"b_better"`` or ``"tie"``.  The verdict is ``"tie"``
        whenever the delta interval contains zero.  ``p_value`` is the two-sided
        bootstrap proportion of sign-flipped deltas with the ``(count + 1) /
        (n + 1)`` correction, so a finite number of draws can never report exactly
        zero evidence against the null.

    Raises
    ------
    InvalidPredictionInputError
        If the inputs are not valid finite triples, or ``metric_fn`` returns a
        non-finite score on the observed data or on any resample.
    """
    observed_true, observed_a, observed_b = _check_paired(
        y_true=y_true, pred_a=pred_a, pred_b=pred_b
    )
    _check_resampling(n_resamples, alpha)

    score_a = _finite_score(metric_fn(observed_true, observed_a), "metric_fn (pred_a)")
    score_b = _finite_score(metric_fn(observed_true, observed_b), "metric_fn (pred_b)")
    delta = score_a - score_b

    rng = np.random.default_rng(seed)
    deltas = np.fromiter(
        (
            float(metric_fn(observed_true[index], observed_a[index]))
            - float(metric_fn(observed_true[index], observed_b[index]))
            for index in _resample_indices(rng, observed_true.size, n_resamples)
        ),
        dtype=float,
        count=int(n_resamples),
    )
    _reject_non_finite(deltas, "metric_fn")

    ci_low, ci_high = _percentile_bounds(deltas, alpha)

    if delta == 0.0:
        p_value = 1.0
    else:
        flipped = int(np.count_nonzero(deltas <= 0.0 if delta > 0.0 else deltas >= 0.0))
        p_value = min(1.0, 2.0 * (flipped + 1) / (int(n_resamples) + 1))

    if delta == 0.0 or ci_low <= 0.0 <= ci_high:
        verdict = VERDICT_TIE
    elif (delta > 0.0) == bool(higher_is_better):
        verdict = VERDICT_A_BETTER
    else:
        verdict = VERDICT_B_BETTER

    return {
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": float(p_value),
        "verdict": verdict,
    }


def wilcoxon_signed_rank_errors(
    y_true: ArrayLike,
    pred_a: ArrayLike,
    pred_b: ArrayLike,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on per-compound absolute errors.

    Compares ``|y_true - pred_a|`` against ``|y_true - pred_b|`` compound by
    compound.  Unlike an aggregate metric this is distribution-free and is not
    dominated by a handful of large residuals, so it is the right test when the
    two models disagree mostly on a subset of the chemistry.

    Lower error wins: ``"a_better"`` means model A's absolute errors are
    significantly smaller.

    Parameters
    ----------
    y_true
        Finite observed values.
    pred_a, pred_b
        Predictions of the two models for the same compounds, in the same order.
    alpha
        Significance level below which a verdict other than ``"tie"`` is issued.

    Returns
    -------
    dict
        ``{"statistic", "p_value", "verdict", "n_pairs", "n_informative_pairs"}``.
        When every compound has identical absolute errors under both models the
        test is undefined (SciPy has no non-zero differences to rank), so the
        result is ``statistic = 0.0``, ``p_value = None`` and
        ``verdict = "tie_all_errors_equal"`` - a documented sentinel, not a
        fabricated p-value.  ``n_informative_pairs`` counts the non-zero
        differences actually ranked; with fewer than six the exact two-sided test
        cannot reach p < 0.05 at all.

    Raises
    ------
    InvalidPredictionInputError
        If the inputs are not valid finite triples.
    """
    observed_true, observed_a, observed_b = _check_paired(
        y_true=y_true, pred_a=pred_a, pred_b=pred_b
    )
    errors_a = np.abs(observed_true - observed_a)
    errors_b = np.abs(observed_true - observed_b)
    differences = errors_a - errors_b
    informative = differences[differences != 0.0]

    if informative.size == 0:
        return {
            "statistic": 0.0,
            "p_value": None,
            "verdict": VERDICT_TIE_ALL_ERRORS_EQUAL,
            "n_pairs": int(differences.size),
            "n_informative_pairs": 0,
        }

    result = wilcoxon(errors_a, errors_b)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)

    # SciPy's two-sided statistic is min(R+, R-) and carries no direction, so
    # recover the direction from the signed rank sum it is built from.
    signed_rank_sum = float(np.sum(rankdata(np.abs(informative)) * np.sign(informative)))

    if p_value >= float(alpha) or signed_rank_sum == 0.0:
        verdict = VERDICT_TIE
    elif signed_rank_sum < 0.0:
        verdict = VERDICT_A_BETTER
    else:
        verdict = VERDICT_B_BETTER

    return {
        "statistic": statistic,
        "p_value": p_value,
        "verdict": verdict,
        "n_pairs": int(differences.size),
        "n_informative_pairs": int(informative.size),
    }


def y_scrambling(
    fit_predict_fn: ScoreFn,
    y: ArrayLike,
    n_repeats: int = 10,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Y-randomisation test for chance correlation.

    The model is scored once on the real endpoint and ``n_repeats`` times on
    permuted endpoints.  If shuffling the labels barely hurts the score, the
    model is fitting the descriptor matrix rather than the chemistry.

    The caller owns retraining: ``fit_predict_fn`` receives one permuted ``y``
    array and must refit the *whole* pipeline (feature selection and
    hyperparameter choice included) on it before returning a score.  Reusing a
    model fitted on the real ``y`` makes this test meaningless.

    Parameters
    ----------
    fit_predict_fn
        ``fit_predict_fn(y) -> float``, returning a **higher-is-better** score
        such as R2, ROC AUC or Spearman rho.  For an error metric, negate it;
        otherwise the p-value is computed against the wrong tail.
    y
        Finite observed endpoint values.
    n_repeats
        Number of permutations (at least 2).  The smallest attainable p-value is
        ``1 / (n_repeats + 1)``, so ``n_repeats >= 19`` is needed for p < 0.05.
    seed
        Seed for :func:`numpy.random.default_rng`.
    alpha
        Threshold above which chance correlation is flagged.

    Returns
    -------
    dict
        ``{"observed", "null_scores", "null_mean", "null_std", "p_value",
        "z_score", "n_repeats", "verdict"}``.  ``p_value`` is
        ``(count(null >= observed) + 1) / (n_repeats + 1)``, so it is never
        exactly zero.  ``z_score`` is ``None`` when the null scores are identical
        (zero spread leaves the z-score undefined).  ``verdict`` is
        ``"chance_correlation_suspected"`` when ``p_value > alpha``.

    Raises
    ------
    InvalidPredictionInputError
        If ``y`` is not a valid finite array, ``n_repeats < 2``, or
        ``fit_predict_fn`` returns a non-finite score.
    """
    endpoint = _as_float_array(y, "y")
    if endpoint.size < 2:
        raise InvalidPredictionInputError(
            f"at least 2 endpoint values are required, got {endpoint.size}."
        )
    if int(n_repeats) < 2:
        raise InvalidPredictionInputError(
            f"n_repeats must be at least 2 to estimate a null distribution, got {n_repeats}."
        )
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidPredictionInputError(f"alpha must lie strictly between 0 and 1, got {alpha}.")

    observed = _finite_score(fit_predict_fn(endpoint.copy()), "fit_predict_fn")
    rng = np.random.default_rng(seed)
    null_scores = np.fromiter(
        (float(fit_predict_fn(rng.permutation(endpoint))) for _ in range(int(n_repeats))),
        dtype=float,
        count=int(n_repeats),
    )
    _reject_non_finite(null_scores, "fit_predict_fn")

    null_mean = float(np.mean(null_scores))
    # np.std over identical values returns ~1e-16 rather than 0, which would turn
    # the z-score into amplified rounding noise, so key the degenerate case off
    # the spread of the scores themselves.
    has_spread = bool(np.max(null_scores) != np.min(null_scores))
    null_std = float(np.std(null_scores, ddof=1)) if has_spread else 0.0
    at_least_as_good = int(np.count_nonzero(null_scores >= observed))
    p_value = (at_least_as_good + 1) / (int(n_repeats) + 1)
    z_score: Optional[float] = (observed - null_mean) / null_std if has_spread else None

    return {
        "observed": observed,
        "null_scores": null_scores.tolist(),
        "null_mean": null_mean,
        "null_std": null_std,
        "p_value": float(p_value),
        "z_score": z_score,
        "n_repeats": int(n_repeats),
        "verdict": (
            VERDICT_CHANCE_CORRELATION if p_value > float(alpha) else VERDICT_NO_CHANCE_CORRELATION
        ),
    }


def benjamini_hochberg(p_values: ArrayLike, alpha: float = DEFAULT_ALPHA) -> List[bool]:
    """Benjamini-Hochberg step-up procedure for false-discovery-rate control.

    Screening a leaderboard of ``m`` candidate models against a baseline at
    p < 0.05 each produces roughly ``0.05 * m`` spurious winners.  This controls
    the expected proportion of false positives among the rejections at ``alpha``
    instead.

    Note this is the step-*up* form: the cut is made at the largest rank ``k``
    with ``p_(k) <= k * alpha / m``, and *every* hypothesis up to that rank is
    rejected, including ones whose own p-value exceeds their own threshold.

    Parameters
    ----------
    p_values
        Finite p-values in ``[0, 1]``, one per comparison.
    alpha
        Target false-discovery rate.

    Returns
    -------
    list of bool
        ``True`` where the null hypothesis is rejected, in the input order.
        An empty input returns an empty list.

    Raises
    ------
    InvalidPredictionInputError
        If any value lies outside ``[0, 1]`` or is not finite, or ``alpha`` is
        not strictly between 0 and 1.
    """
    values = _as_float_array(p_values, "p_values")
    if not 0.0 < float(alpha) < 1.0:
        raise InvalidPredictionInputError(f"alpha must lie strictly between 0 and 1, got {alpha}.")
    if values.size == 0:
        return []
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise InvalidPredictionInputError("p_values must all lie within [0, 1].")

    n_tests = values.size
    order = np.argsort(values, kind="stable")
    thresholds = float(alpha) * np.arange(1, n_tests + 1) / n_tests
    passing = np.flatnonzero(values[order] <= thresholds)

    rejected = np.zeros(n_tests, dtype=bool)
    if passing.size:
        rejected[order[: int(passing[-1]) + 1]] = True
    return [bool(flag) for flag in rejected]
