from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cs_copilot.tools.prediction.conformal import (
    ConformalPredictionError,
    InductiveConformalClassifier,
    InductiveConformalRegressor,
    conformal_quantile_index,
    empirical_coverage,
    empirical_set_coverage,
    minimum_calibration_size,
)

SEED = 20240517


def _regression_split(
    rng: np.random.Generator, n: int, noise: float = 0.5
) -> tuple[np.ndarray, np.ndarray]:
    """Return exchangeable (y, prediction) pairs from a fixed noise model."""
    y = rng.normal(0.0, 1.0, n)
    pred = y + rng.normal(0.0, noise, n)
    return y, pred


def _imbalanced_split(
    rng: np.random.Generator, n: int, *, rate: float = 0.06, separation: float = 1.2
) -> tuple[np.ndarray, np.ndarray]:
    """Return a 2-D feature matrix and labels with a `rate` minority class."""
    y = (rng.random(n) < rate).astype(int)
    x = rng.normal(0.0, 1.0, (n, 2)) + y[:, None] * separation
    return x, y


# --------------------------------------------------------------------------------------
# Finite-sample rank: the off-by-one that silently breaks the coverage guarantee
# --------------------------------------------------------------------------------------


def test_quantile_is_the_corrected_rank_not_the_naive_one():
    # ceil((100 + 1) * 0.9) = ceil(90.9) = 91, so the threshold is the 91st smallest.
    assert conformal_quantile_index(100, 0.1) == 91
    # Both common mistakes land on 90, which under-covers.
    assert math.ceil(100 * 0.9) == 90
    assert conformal_quantile_index(100, 0.1) != math.ceil(100 * 0.9)


def test_quantile_index_is_exact_at_float_boundaries():
    # (n + 1) * (1 - alpha) is integral here; floating point must not push it up.
    assert conformal_quantile_index(999, 0.1) == 900
    assert conformal_quantile_index(1000, 0.1) == 901
    assert conformal_quantile_index(19, 0.05) == 19
    assert conformal_quantile_index(4, 0.2) == 4


@pytest.mark.parametrize(
    ("alpha", "expected_minimum"),
    [(0.2, 4), (0.15, 6), (0.1, 9), (0.05, 19), (0.01, 99)],
)
def test_minimum_calibration_size_is_the_exact_boundary(alpha, expected_minimum):
    assert minimum_calibration_size(alpha) == expected_minimum
    # Exactly the minimum works, one fewer must raise instead of under-covering.
    assert conformal_quantile_index(expected_minimum, alpha) == expected_minimum
    with pytest.raises(ConformalPredictionError, match="too small"):
        conformal_quantile_index(expected_minimum - 1, alpha)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_alpha_outside_the_open_unit_interval_raises(alpha):
    with pytest.raises(ConformalPredictionError, match=r"\(0, 1\)"):
        conformal_quantile_index(100, alpha)


# --------------------------------------------------------------------------------------
# Regression: closed form, coverage, monotonicity, persistence
# --------------------------------------------------------------------------------------


def test_regressor_quantile_matches_closed_form():
    # Residuals are exactly 1..100, so the alpha=0.1 threshold is exactly the 91st
    # smallest score, i.e. 91.0 -- and the alpha=0.01 threshold the 100th, i.e. 100.0.
    residuals = np.arange(1.0, 101.0)
    icp = InductiveConformalRegressor().fit(np.zeros(100), -residuals)

    assert icp.predict_interval(np.zeros(3), alpha=0.1)["quantile"] == pytest.approx(91.0)
    assert icp.predict_interval(np.zeros(3), alpha=0.01)["quantile"] == pytest.approx(100.0)
    # The naive alternatives would have produced these instead.
    assert float(np.quantile(residuals, 0.9)) == pytest.approx(90.1)
    assert residuals[math.ceil(100 * 0.9) - 1] == pytest.approx(90.0)


def test_regressor_interval_is_symmetric_around_the_prediction():
    residuals = np.arange(1.0, 101.0)
    icp = InductiveConformalRegressor().fit(np.zeros(100), -residuals)
    pred = np.array([-2.0, 0.0, 5.0])

    interval = icp.predict_interval(pred, alpha=0.1)

    assert interval["lower"] == pytest.approx(pred - 91.0)
    assert interval["upper"] == pytest.approx(pred + 91.0)
    assert interval["width"] == pytest.approx(np.full(3, 182.0))
    assert interval["n_calib"] == 100
    assert interval["normalized"] is False


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_empirical_coverage_matches_target_on_exchangeable_data(alpha):
    rng = np.random.default_rng(SEED)
    y_calib, pred_calib = _regression_split(rng, 1000)
    y_test, pred_test = _regression_split(rng, 4000)

    icp = InductiveConformalRegressor().fit(y_calib, pred_calib)
    interval = icp.predict_interval(pred_test, alpha=alpha)
    report = empirical_coverage(
        y_test,
        interval["lower"],
        interval["upper"],
        alpha=alpha,
        n_calib=interval["n_calib"],
    )

    target = 1.0 - alpha
    assert report["target"] == pytest.approx(target)
    assert report["n"] == 4000
    assert target - 0.04 <= report["coverage"] <= target + 0.04
    assert report["within_tolerance"] is True
    assert report["mean_width"] > 0.0


def test_empirical_coverage_at_alpha_0_1_lands_in_the_required_band():
    rng = np.random.default_rng(SEED)
    y_calib, pred_calib = _regression_split(rng, 1000)
    y_test, pred_test = _regression_split(rng, 4000)

    icp = InductiveConformalRegressor().fit(y_calib, pred_calib)
    interval = icp.predict_interval(pred_test, alpha=0.1)

    coverage = empirical_coverage(y_test, interval["lower"], interval["upper"], alpha=0.1)
    assert 0.86 <= coverage["coverage"] <= 0.94


def test_interval_width_grows_as_alpha_shrinks():
    rng = np.random.default_rng(SEED)
    y_calib, pred_calib = _regression_split(rng, 1000)
    icp = InductiveConformalRegressor().fit(y_calib, pred_calib)

    widths = [
        float(icp.predict_interval(np.zeros(5), alpha=alpha)["width"][0])
        for alpha in (0.5, 0.2, 0.1, 0.05, 0.01)
    ]

    assert widths == sorted(widths)
    assert widths[0] < widths[-1]


def test_detects_a_broken_coverage_guarantee():
    # Intervals built from a deliberately too-small quantile must be reported as
    # under-covering: empirical_coverage is what makes the maths falsifiable.
    rng = np.random.default_rng(SEED)
    y_test, pred_test = _regression_split(rng, 2000)
    half_width = 0.2

    report = empirical_coverage(y_test, pred_test - half_width, pred_test + half_width, alpha=0.1)

    assert report["coverage"] < 0.86
    assert report["within_tolerance"] is False


def test_too_small_calibration_set_raises_and_names_the_minimum():
    icp = InductiveConformalRegressor().fit(np.zeros(8), np.arange(8.0))

    with pytest.raises(ConformalPredictionError, match="At least 9 calibration points"):
        icp.predict_interval(np.zeros(3), alpha=0.1)

    # One more calibration point and the same request succeeds.
    icp = InductiveConformalRegressor().fit(np.zeros(9), np.arange(9.0))
    assert icp.predict_interval(np.zeros(3), alpha=0.1)["quantile"] == pytest.approx(8.0)


def test_round_trip_reproduces_identical_intervals():
    rng = np.random.default_rng(SEED)
    y_calib, pred_calib = _regression_split(rng, 500)
    _, pred_test = _regression_split(rng, 200)
    icp = InductiveConformalRegressor().fit(y_calib, pred_calib)

    restored = InductiveConformalRegressor.from_dict(icp.to_dict())

    for alpha in (0.05, 0.1, 0.2):
        original = icp.predict_interval(pred_test, alpha=alpha)
        copied = restored.predict_interval(pred_test, alpha=alpha)
        assert copied["quantile"] == original["quantile"]
        np.testing.assert_array_equal(copied["lower"], original["lower"])
        np.testing.assert_array_equal(copied["upper"], original["upper"])


def test_round_trip_of_a_normalized_regressor_keeps_the_normalized_contract():
    rng = np.random.default_rng(SEED)
    sigma = np.abs(rng.normal(0.0, 1.0, 400)) + 0.2
    y_calib = rng.normal(0.0, sigma)
    icp = InductiveConformalRegressor().fit(y_calib, np.zeros(400), sigma_calib=sigma)

    restored = InductiveConformalRegressor.from_dict(icp.to_dict())

    assert restored.normalized is True
    sigma_test = np.array([0.5, 2.0])
    original = icp.predict_interval(np.zeros(2), alpha=0.1, sigma_test=sigma_test)
    copied = restored.predict_interval(np.zeros(2), alpha=0.1, sigma_test=sigma_test)
    np.testing.assert_array_equal(copied["width"], original["width"])


def test_normalized_quantile_divides_by_sigma():
    # Residuals are sigma * (1..100), so the NORMALISED scores are exactly 1..100
    # and the alpha=0.1 threshold is exactly 91.0. An implementation that forgot to
    # divide by sigma would use the raw residuals and get a different number.
    scaled = np.arange(1.0, 101.0)
    sigma = 1.0 + (np.arange(100) % 3)
    residuals = scaled * sigma
    icp = InductiveConformalRegressor().fit(np.zeros(100), -residuals, sigma_calib=sigma)

    sigma_test = np.array([0.5, 4.0])
    interval = icp.predict_interval(np.zeros(2), alpha=0.1, sigma_test=sigma_test)

    assert interval["quantile"] == pytest.approx(91.0)
    assert interval["normalized"] is True
    # Half-width is the normalised quantile rescaled by each row's own difficulty.
    assert interval["upper"] == pytest.approx(91.0 * sigma_test)
    # The sigma-ignoring implementation would have produced this instead.
    assert float(np.sort(residuals)[90]) != pytest.approx(91.0)


def test_normalized_intervals_cover_and_scale_with_difficulty():
    rng = np.random.default_rng(SEED)
    sigma_calib = np.abs(rng.normal(0.0, 1.0, 1000)) + 0.2
    y_calib = rng.normal(0.0, sigma_calib)
    sigma_test = np.abs(rng.normal(0.0, 1.0, 4000)) + 0.2
    y_test = rng.normal(0.0, sigma_test)

    icp = InductiveConformalRegressor().fit(y_calib, np.zeros(1000), sigma_calib=sigma_calib)
    interval = icp.predict_interval(np.zeros(4000), alpha=0.1, sigma_test=sigma_test)
    report = empirical_coverage(y_test, interval["lower"], interval["upper"], alpha=0.1)

    assert 0.86 <= report["coverage"] <= 0.94
    # Width must track the supplied difficulty, not be constant.
    assert interval["width"][np.argmax(sigma_test)] > interval["width"][np.argmin(sigma_test)]


def test_normalized_and_marginal_modes_cannot_be_mixed():
    rng = np.random.default_rng(SEED)
    sigma = np.abs(rng.normal(0.0, 1.0, 100)) + 0.2
    normalized = InductiveConformalRegressor().fit(
        rng.normal(0.0, sigma), np.zeros(100), sigma_calib=sigma
    )
    marginal = InductiveConformalRegressor().fit(np.zeros(100), np.arange(100.0))

    with pytest.raises(ConformalPredictionError, match="sigma_test is required"):
        normalized.predict_interval(np.zeros(3), alpha=0.1)
    with pytest.raises(ConformalPredictionError, match="without sigma_calib"):
        marginal.predict_interval(np.zeros(3), alpha=0.1, sigma_test=np.ones(3))


def test_regressor_rejects_unusable_inputs():
    icp = InductiveConformalRegressor()

    with pytest.raises(ConformalPredictionError, match="not fitted"):
        icp.predict_interval(np.zeros(3))
    with pytest.raises(ConformalPredictionError, match="Length mismatch"):
        icp.fit(np.zeros(5), np.zeros(4))
    with pytest.raises(ConformalPredictionError, match="NaN"):
        icp.fit([1.0, np.nan], [0.0, 0.0])
    with pytest.raises(ConformalPredictionError, match="strictly positive"):
        icp.fit(np.zeros(3), np.zeros(3), sigma_calib=[1.0, 0.0, 1.0])
    with pytest.raises(ConformalPredictionError, match="kind"):
        InductiveConformalRegressor.from_dict({"kind": "something_else", "scores": [1.0]})


# --------------------------------------------------------------------------------------
# Classification: marginal coverage, Mondrian mode on imbalanced data, persistence
# --------------------------------------------------------------------------------------


def test_classifier_quantile_matches_closed_form():
    # Nonconformity of the true class is 1 - p, so these scores are 0.01..1.00.
    scores = np.arange(1, 101) / 100.0
    proba = np.column_stack([1.0 - scores, scores])
    labels = np.zeros(100, dtype=int)  # true class is column 0, score = 1 - (1 - s) = s

    cp = InductiveConformalClassifier().fit(labels, proba, classes=[0, 1])
    result = cp.predict_set(proba[:1], alpha=0.1)

    # 91st smallest of 0.01..1.00 is 0.91.
    assert result["quantiles"][0] == pytest.approx(0.91)
    assert result["quantiles"][1] == pytest.approx(0.91)


def test_classifier_marginal_coverage_matches_target():
    rng = np.random.default_rng(SEED)
    x_train, y_train = _imbalanced_split(rng, 4000, rate=0.5, separation=1.0)
    x_calib, y_calib = _imbalanced_split(rng, 2000, rate=0.5, separation=1.0)
    x_test, y_test = _imbalanced_split(rng, 4000, rate=0.5, separation=1.0)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)

    cp = InductiveConformalClassifier().fit(
        y_calib, model.predict_proba(x_calib), classes=model.classes_
    )
    result = cp.predict_set(model.predict_proba(x_test), alpha=0.1)
    report = empirical_set_coverage(y_test, result["sets"], alpha=0.1, n_calib=result["n_calib"])

    assert 0.86 <= report["coverage"] <= 0.94
    assert report["within_tolerance"] is True
    assert report["n"] == 4000
    np.testing.assert_array_equal(
        result["set_size"], np.array([len(row) for row in result["sets"]])
    )


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_classifier_coverage_holds_across_alphas(alpha):
    rng = np.random.default_rng(SEED)
    x_train, y_train = _imbalanced_split(rng, 4000, rate=0.5, separation=1.0)
    x_calib, y_calib = _imbalanced_split(rng, 2000, rate=0.5, separation=1.0)
    x_test, y_test = _imbalanced_split(rng, 4000, rate=0.5, separation=1.0)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)

    cp = InductiveConformalClassifier().fit(
        y_calib, model.predict_proba(x_calib), classes=model.classes_
    )
    result = cp.predict_set(model.predict_proba(x_test), alpha=alpha)
    report = empirical_set_coverage(y_test, result["sets"], alpha=alpha)

    target = 1.0 - alpha
    assert target - 0.04 <= report["coverage"] <= target + 0.04


def test_class_conditional_mode_rescues_minority_coverage_under_imbalance():
    rng = np.random.default_rng(7)
    x_train, y_train = _imbalanced_split(rng, 4000)
    x_calib, y_calib = _imbalanced_split(rng, 6000)
    x_test, y_test = _imbalanced_split(rng, 8000)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    proba_calib = model.predict_proba(x_calib)
    proba_test = model.predict_proba(x_test)
    minority = y_test == 1
    assert 100 < int(minority.sum()) < 1000  # genuinely imbalanced test set

    coverages = {}
    for class_conditional in (False, True):
        cp = InductiveConformalClassifier(class_conditional=class_conditional).fit(
            y_calib, proba_calib, classes=model.classes_
        )
        result = cp.predict_set(proba_test, alpha=0.1)
        minority_sets = [row for row, keep in zip(result["sets"], minority, strict=True) if keep]
        coverages[class_conditional] = empirical_set_coverage(
            y_test[minority], minority_sets, alpha=0.1
        )["coverage"]
        # Marginal coverage is fine in both modes; it is the per-class split that differs.
        assert empirical_set_coverage(y_test, result["sets"], alpha=0.1)["coverage"] >= 0.86

    # The marginal mode is swamped by the majority class and abandons the minority.
    assert coverages[False] < 0.5
    # The Mondrian mode restores minority coverage close to the 0.9 target.
    assert coverages[True] > 0.78
    assert coverages[True] > coverages[False] + 0.3


def test_prediction_sets_can_be_empty_or_multi_label():
    rng = np.random.default_rng(7)
    x_train, y_train = _imbalanced_split(rng, 4000)
    x_calib, y_calib = _imbalanced_split(rng, 6000)
    x_test, _ = _imbalanced_split(rng, 8000)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    proba_calib = model.predict_proba(x_calib)
    proba_test = model.predict_proba(x_test)

    marginal = InductiveConformalClassifier().fit(y_calib, proba_calib, classes=model.classes_)
    mondrian = InductiveConformalClassifier(class_conditional=True).fit(
        y_calib, proba_calib, classes=model.classes_
    )

    # A tight shared threshold leaves hard rows with no admissible label at all:
    # abstention is a legitimate conformal output, not a bug.
    assert (marginal.predict_set(proba_test, alpha=0.1)["set_size"] == 0).any()
    # Per-class thresholds instead widen ambiguous rows to both labels.
    assert (mondrian.predict_set(proba_test, alpha=0.1)["set_size"] == 2).any()


def test_classifier_round_trip_reproduces_identical_sets():
    rng = np.random.default_rng(7)
    x_train, y_train = _imbalanced_split(rng, 2000)
    x_calib, y_calib = _imbalanced_split(rng, 3000)
    x_test, _ = _imbalanced_split(rng, 500)
    model = LogisticRegression(max_iter=1000).fit(x_train, y_train)
    proba_test = model.predict_proba(x_test)

    cp = InductiveConformalClassifier(class_conditional=True).fit(
        y_calib, model.predict_proba(x_calib), classes=model.classes_
    )
    restored = InductiveConformalClassifier.from_dict(cp.to_dict())

    assert restored.class_conditional is True
    assert restored.classes == [0, 1]
    original = cp.predict_set(proba_test, alpha=0.1)
    copied = restored.predict_set(proba_test, alpha=0.1)
    assert copied["quantiles"] == original["quantiles"]
    assert copied["sets"] == original["sets"]


def test_classifier_rejects_unusable_inputs():
    proba = np.array([[0.7, 0.3], [0.2, 0.8]])

    with pytest.raises(ConformalPredictionError, match="not fitted"):
        InductiveConformalClassifier().predict_set(proba)
    with pytest.raises(ConformalPredictionError, match="columns"):
        InductiveConformalClassifier().fit([0, 1], proba, classes=[0, 1, 2])
    with pytest.raises(ConformalPredictionError, match="not among the classes"):
        InductiveConformalClassifier().fit([0, 5], proba, classes=[0, 1])
    with pytest.raises(ConformalPredictionError, match="probabilities in"):
        InductiveConformalClassifier().fit([0, 1], np.array([[1.4, -0.4], [0.2, 0.8]]))
    with pytest.raises(ConformalPredictionError, match="2-D"):
        InductiveConformalClassifier().fit([0, 1], np.array([0.3, 0.8]))

    cp = InductiveConformalClassifier().fit([0, 1], proba, classes=[0, 1])
    with pytest.raises(ConformalPredictionError, match="columns"):
        cp.predict_set(np.array([[0.3, 0.3, 0.4]]), alpha=0.1)


def test_class_conditional_mode_reports_which_class_is_too_small():
    # Class 1 has a single calibration row, far too few for alpha=0.1.
    y_calib = np.array([0] * 20 + [1])
    proba = np.column_stack([np.full(21, 0.8), np.full(21, 0.2)])

    cp = InductiveConformalClassifier(class_conditional=True).fit(y_calib, proba, classes=[0, 1])

    with pytest.raises(ConformalPredictionError, match=r"Class 1: .*At least 9"):
        cp.predict_set(proba, alpha=0.1)
    # The marginal mode pools the scores and still works on the same data.
    marginal = InductiveConformalClassifier().fit(y_calib, proba, classes=[0, 1])
    assert marginal.predict_set(proba, alpha=0.1)["quantiles"][0] > 0.0


# --------------------------------------------------------------------------------------
# Coverage reporting helpers
# --------------------------------------------------------------------------------------


def test_empirical_coverage_counts_closed_interval_endpoints():
    report = empirical_coverage(
        [0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 5.0, 2.0], [0.5, 1.0, 6.0, 4.0], alpha=0.1
    )

    assert report["coverage"] == pytest.approx(0.75)  # rows 0, 1 (endpoints) and 3
    assert report["n"] == 4
    assert report["mean_width"] == pytest.approx((0.5 + 0.5 + 1.0 + 2.0) / 4.0)


def test_empirical_set_coverage_reports_mean_set_size():
    report = empirical_set_coverage(["a", "b", "c"], [["a", "b"], ["b"], []], alpha=0.1)

    assert report["coverage"] == pytest.approx(2.0 / 3.0)
    assert report["mean_set_size"] == pytest.approx(1.0)
    assert report["target"] == pytest.approx(0.9)
    # Three rows cannot distinguish 0.67 from 0.9, and the reported tolerance says so.
    assert report["tolerance"] > 0.3
    assert report["within_tolerance"] is True


def test_explicit_tolerance_overrides_the_default_band():
    lower = np.full(100, -1.0)
    upper = np.full(100, 1.0)
    y_true = np.full(100, 0.0)
    y_true[:5] = 10.0  # 95% coverage against a 90% target

    default = empirical_coverage(y_true, lower, upper, alpha=0.1)
    assert default["coverage"] == pytest.approx(0.95)
    assert default["tolerance"] == pytest.approx(0.06)  # 2 * sqrt(0.9 * 0.1 / 100)
    assert default["within_tolerance"] is True

    strict = empirical_coverage(y_true, lower, upper, alpha=0.1, tolerance=0.01)
    assert strict["tolerance"] == pytest.approx(0.01)
    assert strict["within_tolerance"] is False


def test_supplying_n_calib_widens_the_default_band():
    # Calibration-set variability dominates at modest n_calib, so a band that
    # ignores it would flag an entirely normal deviation.
    lower = np.full(4000, -1.0)
    upper = np.full(4000, 1.0)
    y_true = np.full(4000, 0.0)
    y_true[:80] = 10.0  # 98% coverage against a 95% target

    without = empirical_coverage(y_true, lower, upper, alpha=0.05)
    with_calib = empirical_coverage(y_true, lower, upper, alpha=0.05, n_calib=150)

    assert with_calib["tolerance"] > without["tolerance"]
    assert without["within_tolerance"] is False
    assert with_calib["within_tolerance"] is True


def test_coverage_helpers_reject_mismatched_inputs():
    with pytest.raises(ConformalPredictionError, match="Length mismatch"):
        empirical_coverage([1.0, 2.0], [0.0], [3.0, 4.0])
    with pytest.raises(ConformalPredictionError, match="at least one observation"):
        empirical_coverage([], [], [])
    with pytest.raises(ConformalPredictionError, match="2 entries but sets has 1"):
        empirical_set_coverage([0, 1], [[0]])
