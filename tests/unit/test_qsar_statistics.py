"""Tests for the QSAR statistical-validation layer.

Every stochastic case passes a fixed seed and asserts an exact or closed-form
value, so a failure means the maths changed, not that the dice rolled badly.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
from sklearn.metrics import r2_score

from cs_copilot.tools.prediction.backend import InvalidPredictionInputError
from cs_copilot.tools.prediction.qsar_statistics import (
    VERDICT_A_BETTER,
    VERDICT_B_BETTER,
    VERDICT_CHANCE_CORRELATION,
    VERDICT_NO_CHANCE_CORRELATION,
    VERDICT_TIE,
    VERDICT_TIE_ALL_ERRORS_EQUAL,
    benjamini_hochberg,
    bootstrap_metric_ci,
    paired_bootstrap_comparison,
    wilcoxon_signed_rank_errors,
    y_scrambling,
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mean_prediction(_y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Metric that depends only on which rows were drawn."""
    return float(np.mean(y_pred))


def _constant_metric(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
    return 3.5


# --------------------------------------------------------------------------
# bootstrap_metric_ci
# --------------------------------------------------------------------------


def test_bootstrap_ci_of_constant_metric_is_degenerate():
    result = bootstrap_metric_ci(
        [1.0, 2.0, 3.0, 4.0], [1.1, 2.2, 2.9, 3.8], _constant_metric, n_resamples=50, seed=0
    )

    assert result["point"] == 3.5
    assert result["ci_low"] == 3.5
    assert result["ci_high"] == 3.5
    assert result["n_resamples"] == 50
    assert result["alpha"] == 0.05


def test_bootstrap_ci_matches_closed_form_percentiles_for_two_observations():
    # Resampling 2 paired rows with replacement gives exactly 4 equally likely
    # draws, so mean(y_pred) is 0 w.p. 1/4, 5 w.p. 1/2 and 10 w.p. 1/4.
    # The 2.5th percentile therefore sits in the 0s and the 97.5th in the 10s.
    result = bootstrap_metric_ci(
        [0.0, 0.0], [0.0, 10.0], _mean_prediction, n_resamples=1000, seed=7
    )

    assert result["point"] == 5.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 10.0


def test_bootstrap_resamples_pairs_not_arrays_independently():
    # y_pred == y_true, so RMSE is 0 for every *paired* resample. Resampling the
    # two arrays independently would almost surely give a positive RMSE.
    y_true = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    result = bootstrap_metric_ci(y_true, y_true, _rmse, n_resamples=500, seed=3)

    assert result["point"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(11)
    y_true = rng.normal(size=60)
    y_pred = y_true + rng.normal(scale=0.3, size=60)

    first = bootstrap_metric_ci(y_true, y_pred, r2_score, n_resamples=300, seed=42)
    second = bootstrap_metric_ci(y_true, y_pred, r2_score, n_resamples=300, seed=42)
    other_seed = bootstrap_metric_ci(y_true, y_pred, r2_score, n_resamples=300, seed=43)

    assert first == second
    assert first["ci_low"] < first["point"] < first["ci_high"]
    assert first["ci_low"] != other_seed["ci_low"]


def test_bootstrap_wider_alpha_gives_narrower_interval():
    rng = np.random.default_rng(5)
    y_true = rng.normal(size=40)
    y_pred = y_true + rng.normal(scale=0.5, size=40)

    narrow = bootstrap_metric_ci(y_true, y_pred, r2_score, n_resamples=400, alpha=0.5, seed=1)
    wide = bootstrap_metric_ci(y_true, y_pred, r2_score, n_resamples=400, alpha=0.01, seed=1)

    assert wide["ci_high"] - wide["ci_low"] > narrow["ci_high"] - narrow["ci_low"]


def test_bootstrap_rejects_mismatched_lengths():
    with pytest.raises(InvalidPredictionInputError, match="equal length"):
        bootstrap_metric_ci([1.0, 2.0, 3.0], [1.0, 2.0], _rmse, n_resamples=10)


def test_bootstrap_rejects_non_finite_inputs():
    with pytest.raises(InvalidPredictionInputError, match="NaN or infinite"):
        bootstrap_metric_ci([1.0, 2.0, np.nan], [1.0, 2.0, 3.0], _rmse, n_resamples=10)


def test_bootstrap_rejects_non_finite_metric_rather_than_guessing():
    def _broken(_y_true: np.ndarray, _y_pred: np.ndarray) -> float:
        return float("nan")

    with pytest.raises(InvalidPredictionInputError, match="non-finite"):
        bootstrap_metric_ci([1.0, 2.0], [1.0, 2.0], _broken, n_resamples=10)


def test_bootstrap_rejects_degenerate_resampling_budget():
    with pytest.raises(InvalidPredictionInputError, match="n_resamples"):
        bootstrap_metric_ci([1.0, 2.0], [1.0, 2.0], _rmse, n_resamples=1)

    with pytest.raises(InvalidPredictionInputError, match="alpha"):
        bootstrap_metric_ci([1.0, 2.0], [1.0, 2.0], _rmse, n_resamples=10, alpha=0.0)


# --------------------------------------------------------------------------
# paired_bootstrap_comparison
# --------------------------------------------------------------------------


def test_paired_comparison_of_a_model_against_itself_is_a_tie():
    rng = np.random.default_rng(2)
    y_true = rng.normal(size=50)
    y_pred = y_true + rng.normal(scale=0.4, size=50)

    result = paired_bootstrap_comparison(
        y_true, y_pred, y_pred, r2_score, higher_is_better=True, n_resamples=200, seed=0
    )

    # Identical predictions scored on identical indices cancel exactly; a
    # non-degenerate interval here would mean the two models were resampled apart.
    assert result["delta"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["p_value"] == 1.0
    assert result["verdict"] == VERDICT_TIE


def test_paired_comparison_detects_a_uniformly_better_model():
    y_true = np.arange(20, dtype=float)
    perfect = y_true.copy()
    offset = y_true + 5.0

    result = paired_bootstrap_comparison(
        y_true, perfect, offset, _rmse, higher_is_better=False, n_resamples=200, seed=0
    )

    # RMSE is 0 for every resample of the perfect model and exactly 5 for the
    # shifted one, so the delta is -5 on every draw.
    assert result["delta"] == -5.0
    assert result["ci_low"] == -5.0
    assert result["ci_high"] == -5.0
    assert result["p_value"] == pytest.approx(2.0 / 201.0)
    assert result["verdict"] == VERDICT_A_BETTER


def test_paired_comparison_verdict_respects_metric_orientation():
    y_true = np.arange(20, dtype=float)
    perfect = y_true.copy()
    offset = y_true + 5.0

    # Same data, but telling the function that a larger RMSE is better must flip
    # the verdict rather than silently reporting the same winner.
    flipped = paired_bootstrap_comparison(
        y_true, perfect, offset, _rmse, higher_is_better=True, n_resamples=200, seed=0
    )

    assert flipped["delta"] == -5.0
    assert flipped["verdict"] == VERDICT_B_BETTER


def test_paired_comparison_calls_a_tie_when_the_interval_spans_zero():
    rng = np.random.default_rng(13)
    y_true = rng.normal(size=80)
    noise = rng.normal(scale=0.5, size=80)
    pred_a = y_true + noise
    pred_b = y_true + noise[::-1]  # same error magnitudes, different compounds

    result = paired_bootstrap_comparison(
        y_true, pred_a, pred_b, _rmse, higher_is_better=False, n_resamples=400, seed=9
    )

    assert result["ci_low"] < 0.0 < result["ci_high"]
    assert result["verdict"] == VERDICT_TIE
    assert result["p_value"] > 0.05


def test_paired_comparison_p_value_is_never_exactly_zero():
    y_true = np.arange(20, dtype=float)
    result = paired_bootstrap_comparison(
        y_true,
        y_true,
        y_true + 5.0,
        _rmse,
        higher_is_better=False,
        n_resamples=1000,
        seed=0,
    )

    assert result["p_value"] > 0.0
    assert result["p_value"] == pytest.approx(2.0 / 1001.0)


def test_paired_comparison_rejects_mismatched_lengths():
    with pytest.raises(InvalidPredictionInputError, match="equal length"):
        paired_bootstrap_comparison(
            [1.0, 2.0], [1.0, 2.0], [1.0], _rmse, higher_is_better=False, n_resamples=10
        )


# --------------------------------------------------------------------------
# wilcoxon_signed_rank_errors
# --------------------------------------------------------------------------


def test_wilcoxon_matches_the_exact_closed_form_for_six_concordant_pairs():
    # Absolute-error differences are [1, 2, 3, 4, 5, 6]: all six ranks are
    # positive, so min(R+, R-) == 0 and the exact two-sided p is 2 / 2**6.
    y_true = np.zeros(6)
    pred_a = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    pred_b = np.ones(6)

    result = wilcoxon_signed_rank_errors(y_true, pred_a, pred_b)

    assert result["statistic"] == 0.0
    assert result["p_value"] == pytest.approx(2.0 / 64.0)
    assert result["p_value"] == pytest.approx(0.03125)
    assert result["verdict"] == VERDICT_B_BETTER
    assert result["n_pairs"] == 6
    assert result["n_informative_pairs"] == 6


def test_wilcoxon_direction_flips_when_model_a_has_the_smaller_errors():
    y_true = np.zeros(6)
    pred_a = np.ones(6)
    pred_b = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    result = wilcoxon_signed_rank_errors(y_true, pred_a, pred_b)

    assert result["p_value"] == pytest.approx(0.03125)
    assert result["verdict"] == VERDICT_A_BETTER


def test_wilcoxon_all_ties_returns_documented_sentinel_instead_of_a_p_value():
    # Mirror-image residuals have identical magnitudes, so no pair is informative
    # and SciPy has nothing to rank.
    y_true = np.zeros(5)
    pred_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred_b = -pred_a

    result = wilcoxon_signed_rank_errors(y_true, pred_a, pred_b)

    assert result["p_value"] is None
    assert result["statistic"] == 0.0
    assert result["verdict"] == VERDICT_TIE_ALL_ERRORS_EQUAL
    assert result["n_informative_pairs"] == 0


def test_wilcoxon_identical_models_are_all_ties():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.5, 3.5, 3.5])

    result = wilcoxon_signed_rank_errors(y_true, pred, pred)

    assert result["p_value"] is None
    assert result["verdict"] == VERDICT_TIE_ALL_ERRORS_EQUAL


def test_wilcoxon_calls_a_tie_when_the_evidence_is_weak():
    # Three pairs favour A, three favour B by the same margins.
    y_true = np.zeros(6)
    pred_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    pred_b = np.array([2.0, 3.0, 4.0, 1.0, 2.0, 3.0])

    result = wilcoxon_signed_rank_errors(y_true, pred_a, pred_b)

    assert result["p_value"] > 0.05
    assert result["verdict"] == VERDICT_TIE


def test_wilcoxon_ignores_residual_sign_and_ranks_magnitude():
    # Under-prediction and over-prediction of the same size must tie.
    y_true = np.array([10.0, 20.0, 30.0])
    result = wilcoxon_signed_rank_errors(y_true, y_true - 2.0, y_true + 2.0)

    assert result["verdict"] == VERDICT_TIE_ALL_ERRORS_EQUAL


# --------------------------------------------------------------------------
# y_scrambling
# --------------------------------------------------------------------------


def test_y_scrambling_flags_a_model_that_ignores_the_endpoint():
    def _ignores_y(_y: np.ndarray) -> float:
        return 0.7

    result = y_scrambling(_ignores_y, np.arange(20, dtype=float), n_repeats=20, seed=0)

    assert result["observed"] == 0.7
    assert result["null_scores"] == [0.7] * 20
    assert result["null_mean"] == pytest.approx(0.7)
    assert result["null_std"] == 0.0
    assert result["p_value"] == 1.0
    # np.std of identical floats is ~1e-16, so a naive `std > 0` guard would
    # report a meaningless z-score of about 2 here instead of "undefined".
    assert result["z_score"] is None
    assert result["n_repeats"] == 20
    assert result["verdict"] == VERDICT_CHANCE_CORRELATION


def test_y_scrambling_clears_a_model_that_tracks_the_real_endpoint():
    endpoint = np.arange(30, dtype=float)

    def _correlates_with_truth(y: np.ndarray) -> float:
        return float(np.corrcoef(y, endpoint)[0, 1])

    result = y_scrambling(_correlates_with_truth, endpoint, n_repeats=50, seed=0)

    assert result["observed"] == pytest.approx(1.0)
    assert result["p_value"] == pytest.approx(1.0 / 51.0)
    assert result["p_value"] < 0.05
    assert result["z_score"] > 3.0
    assert abs(result["null_mean"]) < 0.3
    assert result["verdict"] == VERDICT_NO_CHANCE_CORRELATION


def test_y_scrambling_p_value_can_never_be_zero():
    endpoint = np.arange(10, dtype=float)

    def _always_beats_the_null(y: np.ndarray) -> float:
        return 1.0 if np.array_equal(y, endpoint) else 0.0

    result = y_scrambling(_always_beats_the_null, endpoint, n_repeats=4, seed=0)

    assert result["p_value"] == pytest.approx(1.0 / 5.0)


def test_y_scrambling_passes_true_permutations_and_leaves_the_input_intact():
    endpoint = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    original = endpoint.copy()
    seen: List[np.ndarray] = []

    def _record(y: np.ndarray) -> float:
        seen.append(np.asarray(y).copy())
        return 0.5

    y_scrambling(_record, endpoint, n_repeats=5, seed=0)

    assert np.array_equal(endpoint, original)
    assert len(seen) == 6  # one observed fit plus five permuted fits
    assert np.array_equal(seen[0], original)
    for permuted in seen[1:]:
        assert np.array_equal(np.sort(permuted), np.sort(original))
    assert any(not np.array_equal(permuted, original) for permuted in seen[1:])


def test_y_scrambling_is_reproducible_for_a_fixed_seed():
    endpoint = np.arange(15, dtype=float)

    def _score(y: np.ndarray) -> float:
        return float(np.corrcoef(y, endpoint)[0, 1])

    first = y_scrambling(_score, endpoint, n_repeats=10, seed=123)
    second = y_scrambling(_score, endpoint, n_repeats=10, seed=123)

    assert first == second


def test_y_scrambling_rejects_a_null_distribution_it_cannot_estimate():
    with pytest.raises(InvalidPredictionInputError, match="n_repeats"):
        y_scrambling(lambda _y: 1.0, np.arange(5, dtype=float), n_repeats=1)


def test_y_scrambling_rejects_a_non_finite_score():
    with pytest.raises(InvalidPredictionInputError, match="non-finite"):
        y_scrambling(lambda _y: float("inf"), np.arange(5, dtype=float), n_repeats=3)


# --------------------------------------------------------------------------
# benjamini_hochberg
# --------------------------------------------------------------------------

# Benjamini & Hochberg (1995), Table 1: m = 15, alpha = 0.05 rejects the four
# smallest p-values (thresholds k * 0.05 / 15 = 0.00333, 0.00667, 0.01, 0.01333;
# 0.0201 > 0.01667 and nothing above it passes either).
BH_1995_EXAMPLE = [
    0.0001,
    0.0004,
    0.0019,
    0.0095,
    0.0201,
    0.0278,
    0.0298,
    0.0344,
    0.0459,
    0.3240,
    0.4262,
    0.5719,
    0.6528,
    0.7590,
    1.0000,
]


def test_benjamini_hochberg_matches_the_published_worked_example():
    assert benjamini_hochberg(BH_1995_EXAMPLE, alpha=0.05) == [True] * 4 + [False] * 11


def test_benjamini_hochberg_is_a_step_up_procedure():
    # Sorted thresholds are 0.0125, 0.025, 0.0375, 0.05. The second-ranked p of
    # 0.03 fails its own threshold, but the fourth passes, so all four reject.
    assert benjamini_hochberg([0.001, 0.03, 0.03, 0.03], alpha=0.05) == [True] * 4


def test_benjamini_hochberg_returns_flags_in_input_order():
    shuffled = list(reversed(BH_1995_EXAMPLE))
    flags = benjamini_hochberg(shuffled, alpha=0.05)

    assert flags == [False] * 11 + [True] * 4
    assert [p for p, keep in zip(shuffled, flags, strict=True) if keep] == sorted(
        BH_1995_EXAMPLE[:4], reverse=True
    )


def test_benjamini_hochberg_is_stricter_than_an_uncorrected_threshold():
    # Thresholds are k * 0.005. Ranks 1 and 2 clear 0.005 and 0.01; rank 3 needs
    # 0.015 but carries 0.04, and no higher rank recovers it, so the third model
    # is a false discovery that an uncorrected screen would have kept.
    leaderboard = [0.001, 0.004, 0.04, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    uncorrected = [p < 0.05 for p in leaderboard]
    corrected = benjamini_hochberg(leaderboard, alpha=0.05)

    assert uncorrected == [True, True, True] + [False] * 7
    assert corrected == [True, True] + [False] * 8


def test_benjamini_hochberg_rejects_nothing_when_nothing_is_significant():
    assert benjamini_hochberg([0.9, 0.8, 1.0], alpha=0.05) == [False, False, False]


def test_benjamini_hochberg_handles_an_empty_leaderboard():
    assert benjamini_hochberg([]) == []


def test_benjamini_hochberg_rejects_values_outside_the_unit_interval():
    with pytest.raises(InvalidPredictionInputError, match=r"\[0, 1\]"):
        benjamini_hochberg([0.1, 1.5])

    with pytest.raises(InvalidPredictionInputError, match="NaN or infinite"):
        benjamini_hochberg([0.1, float("nan")])

    with pytest.raises(InvalidPredictionInputError, match="alpha"):
        benjamini_hochberg([0.1, 0.2], alpha=1.0)
