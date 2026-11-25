"""Closed-form and oracle tests for the extended QSAR evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn import metrics as sk_metrics

from cs_copilot.tools.prediction.metrics_ext import (
    average_precision,
    brier_score,
    cohen_kappa,
    concordance_correlation_coefficient,
    expected_calibration_error,
    extended_classification_metrics,
    extended_regression_metrics,
    golbraikh_tropsha_criteria,
    matthews_corrcoef,
    q2_f1,
    q2_f2,
    q2_f3,
    specificity,
)

# --------------------------------------------------------------------------------------
# classification: closed forms
# --------------------------------------------------------------------------------------


def test_mcc_perfect_inverted_and_constant() -> None:
    assert matthews_corrcoef([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    assert matthews_corrcoef([0, 1, 0, 1], [1, 0, 1, 0]) == pytest.approx(-1.0)
    # Zero denominator: a constant prediction has no variance, reported as no association.
    assert matthews_corrcoef([0, 1, 0, 1], [1, 1, 1, 1]) == pytest.approx(0.0)
    assert matthews_corrcoef([0, 0, 0, 0], [0, 0, 0, 0]) == pytest.approx(0.0)


def test_mcc_known_confusion_matrix() -> None:
    # TP=3, FN=4, TN=2, FP=1
    # MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
    #     = (3*2 - 1*4) / sqrt(4 * 7 * 3 * 6) = 2 / sqrt(504)
    truth = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    predicted = [1, 1, 1, 0, 0, 0, 0, 0, 0, 1]
    assert matthews_corrcoef(truth, predicted) == pytest.approx(2.0 / math.sqrt(504.0))


def test_mcc_multiclass_and_string_labels() -> None:
    assert matthews_corrcoef([0, 1, 2, 2, 1], [0, 1, 2, 2, 1]) == pytest.approx(1.0)
    assert matthews_corrcoef(["a", "b", "c"], ["a", "b", "c"]) == pytest.approx(1.0)


def test_mcc_returns_none_without_valid_pairs() -> None:
    assert matthews_corrcoef([], []) is None
    assert matthews_corrcoef([np.nan, np.nan], [1.0, 0.0]) is None


def test_cohen_kappa_closed_form() -> None:
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    # p_o = 0.5, p_e = (2*2 + 2*2) / 16 = 0.5  ->  kappa = 0
    assert cohen_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)
    assert cohen_kappa([], []) is None


def test_average_precision_closed_form() -> None:
    # Perfect ranking of the positives.
    assert average_precision([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2]) == pytest.approx(1.0)
    # Ranked 0.9(+), 0.8(-), 0.7(+):  (0.5 - 0) * 1 + 0 * 0.5 + (1 - 0.5) * 2/3
    assert average_precision([1, 0, 1], [0.9, 0.8, 0.7]) == pytest.approx(0.5 + 1.0 / 3.0)


def test_average_precision_is_none_for_a_single_class() -> None:
    # sklearn would return 0.0 with a warning; an undefined metric must not look like a score.
    assert average_precision([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert average_precision([1, 1], [0.4, 0.6]) is None
    assert average_precision([], []) is None


def test_brier_score_closed_form() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)
    assert brier_score([0, 1], [1.0, 0.0]) == pytest.approx(1.0)
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert brier_score([], []) is None


def test_expected_calibration_error_closed_form() -> None:
    # Perfectly confident and correct: every bin's predicted rate equals its observed rate.
    assert expected_calibration_error([0, 0, 1, 1], [0.0, 0.0, 1.0, 1.0]) == pytest.approx(0.0)
    # One bin [0.9, 1.0]: mean p = 0.9, observed positive rate = 0.5.
    assert expected_calibration_error([0, 1], [0.9, 0.9]) == pytest.approx(0.4)
    # Bin [0.1, 0.2): weight 0.5, gap |0.1 - 0.5| = 0.4.  Bin [0.9, 1.0]: weight 0.5, gap 0.05.
    error = expected_calibration_error([0, 1, 1, 1], [0.1, 0.1, 0.95, 0.95])
    assert error == pytest.approx(0.5 * 0.4 + 0.5 * 0.05)


def test_expected_calibration_error_single_bin_is_the_global_gap() -> None:
    value = expected_calibration_error([0, 0, 1, 1], [0.2, 0.4, 0.6, 0.6], n_bins=1)
    assert value == pytest.approx(abs(0.45 - 0.5))


def test_expected_calibration_error_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.5, 0.5], n_bins=0)
    with pytest.raises(ValueError):
        expected_calibration_error([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        expected_calibration_error([0, 2], [0.5, 0.5])


def test_specificity_closed_form() -> None:
    # TN = 1, FP = 1
    assert specificity([0, 0, 1, 1], [0, 1, 1, 1]) == pytest.approx(0.5)
    assert specificity([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)
    # No actual negatives: the true negative rate is undefined.
    assert specificity([1, 1, 1], [1, 0, 1]) is None
    assert specificity([], []) is None


# --------------------------------------------------------------------------------------
# classification: scikit-learn oracle on random arrays
# --------------------------------------------------------------------------------------


@pytest.fixture(name="random_binary")
def fixture_random_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20240617)
    truth = rng.integers(0, 2, size=250)
    predicted = rng.integers(0, 2, size=250)
    probability = rng.random(250)
    return truth, predicted, probability


def test_matches_sklearn_on_random_binary_arrays(
    random_binary: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    truth, predicted, probability = random_binary
    assert matthews_corrcoef(truth, predicted) == pytest.approx(
        sk_metrics.matthews_corrcoef(truth, predicted)
    )
    assert cohen_kappa(truth, predicted) == pytest.approx(
        sk_metrics.cohen_kappa_score(truth, predicted)
    )
    assert average_precision(truth, probability) == pytest.approx(
        sk_metrics.average_precision_score(truth, probability)
    )
    assert brier_score(truth, probability) == pytest.approx(
        sk_metrics.brier_score_loss(truth, probability)
    )
    assert specificity(truth, predicted) == pytest.approx(
        sk_metrics.recall_score(truth, predicted, pos_label=0)
    )


def test_matches_sklearn_on_random_multiclass_arrays() -> None:
    rng = np.random.default_rng(99)
    truth = rng.integers(0, 4, size=300)
    predicted = rng.integers(0, 4, size=300)
    assert matthews_corrcoef(truth, predicted) == pytest.approx(
        sk_metrics.matthews_corrcoef(truth, predicted)
    )
    assert cohen_kappa(truth, predicted) == pytest.approx(
        sk_metrics.cohen_kappa_score(truth, predicted)
    )


# --------------------------------------------------------------------------------------
# regression: closed forms
# --------------------------------------------------------------------------------------


def test_ccc_of_identical_vectors_is_one() -> None:
    values = [1.0, 2.0, 3.5, 7.0, -2.0]
    assert concordance_correlation_coefficient(values, values) == pytest.approx(1.0)


def test_ccc_penalises_a_pure_offset() -> None:
    # y = [1,2,3,4], yhat = y + 1:  var = 1.25 each, cov = 1.25, mean gap = 1
    # CCC = 2 * 1.25 / (1.25 + 1.25 + 1) = 2.5 / 3.5
    value = concordance_correlation_coefficient([1, 2, 3, 4], [2, 3, 4, 5])
    assert value == pytest.approx(2.5 / 3.5)
    # Pearson correlation is still 1.0, so CCC must be the stricter statistic.
    assert value < 1.0


def test_ccc_undefined_cases() -> None:
    assert concordance_correlation_coefficient([1.0], [1.0]) is None
    assert concordance_correlation_coefficient([2.0, 2.0], [2.0, 2.0]) is None


def test_q2_variants_on_a_perfect_prediction() -> None:
    truth = [1.0, 2.0, 3.0, 4.0]
    assert q2_f1(truth, truth, 0.0) == pytest.approx(1.0)
    assert q2_f2(truth, truth) == pytest.approx(1.0)
    assert q2_f3(truth, truth, [0.0, 1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_q2_variants_differ_by_their_denominator() -> None:
    truth = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 4.0]
    train = [0.0, 2.0, 4.0]
    # residual sum of squares = 1
    # Q2F1: denominator sum((y - 0)^2) = 14  ->  1 - 1/14
    assert q2_f1(truth, predicted, 0.0) == pytest.approx(1.0 - 1.0 / 14.0)
    # Q2F2: denominator sum((y - 2)^2) = 2   ->  1 - 1/2
    assert q2_f2(truth, predicted) == pytest.approx(0.5)
    # Q2F3: (1/3) / (8/3) = 1/8  ->  1 - 1/8
    assert q2_f3(truth, predicted, train) == pytest.approx(0.875)
    # The three variants must not collapse onto one another.
    assert q2_f1(truth, predicted, 0.0) != pytest.approx(q2_f2(truth, predicted))
    assert q2_f3(truth, predicted, train) != pytest.approx(q2_f2(truth, predicted))


def test_q2_f1_uses_the_training_mean_not_the_test_mean() -> None:
    truth = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 4.0]
    # Passing the test mean as the training mean must reproduce Q2F2 exactly; any other
    # training mean must not. This pins which mean sits in the denominator.
    assert q2_f1(truth, predicted, float(np.mean(truth))) == pytest.approx(q2_f2(truth, predicted))
    assert q2_f1(truth, predicted, 10.0) != pytest.approx(q2_f2(truth, predicted))


def test_q2_f3_uses_the_training_set_size() -> None:
    truth = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 4.0]
    # Same training variance, different training set size -> same value, since the
    # denominator is a per-sample mean square, not a sum.
    small = q2_f3(truth, predicted, [0.0, 2.0, 4.0])
    duplicated = q2_f3(truth, predicted, [0.0, 2.0, 4.0, 0.0, 2.0, 4.0])
    assert small == pytest.approx(duplicated)
    # A wider training range makes the model look better.
    wide = q2_f3(truth, predicted, [-10.0, 0.0, 10.0])
    assert wide > small


def test_q2_variants_require_their_reference_data() -> None:
    truth = [1.0, 2.0, 3.0]
    with pytest.raises(TypeError):
        q2_f1(truth, truth)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        q2_f3(truth, truth)  # type: ignore[call-arg]


def test_q2_variants_return_none_when_undefined() -> None:
    assert q2_f1([2.0, 2.0], [2.0, 2.0], 2.0) is None
    assert q2_f2([2.0, 2.0], [1.0, 3.0]) is None
    assert q2_f3([1.0, 2.0], [1.0, 2.0], [5.0, 5.0]) is None
    assert q2_f1([], [], 0.0) is None
    with pytest.raises(ValueError):
        q2_f1([1.0, 2.0], [1.0, 2.0], float("nan"))


# --------------------------------------------------------------------------------------
# Golbraikh-Tropsha
# --------------------------------------------------------------------------------------


def test_golbraikh_tropsha_passes_a_perfect_prediction() -> None:
    truth = [1.0, 2.0, 3.0, 4.0, 5.0]
    report = golbraikh_tropsha_criteria(truth, truth, q2=0.9)
    assert report["r2"] == pytest.approx(1.0)
    assert report["r02"] == pytest.approx(1.0)
    assert report["r02_prime"] == pytest.approx(1.0)
    assert report["k"] == pytest.approx(1.0)
    assert report["k_prime"] == pytest.approx(1.0)
    assert report["n"] == 5
    assert report["passes"] is True
    assert all(
        report[name] is True
        for name in (
            "r2_above_0_6",
            "q2_above_0_5",
            "r2_r02_close",
            "r2_r02_prime_close",
            "k_in_range",
            "k_prime_in_range",
            "r02_difference_small",
        )
    )


def test_golbraikh_tropsha_fails_a_systematically_scaled_prediction() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    report = golbraikh_tropsha_criteria(truth, 0.5 * truth, q2=0.9)
    # Correlation is still perfect, so the criteria must catch the slope instead.
    assert report["r2"] == pytest.approx(1.0)
    assert report["k"] == pytest.approx(2.0)
    assert report["k_prime"] == pytest.approx(0.5)
    assert report["r2_above_0_6"] is True
    assert report["k_in_range"] is False
    assert report["k_prime_in_range"] is False
    assert report["passes"] is False


def test_golbraikh_tropsha_fails_a_low_internal_q2() -> None:
    truth = [1.0, 2.0, 3.0, 4.0, 5.0]
    report = golbraikh_tropsha_criteria(truth, truth, q2=0.4)
    assert report["q2_above_0_5"] is False
    assert report["passes"] is False


def test_golbraikh_tropsha_fails_an_uncorrelated_prediction() -> None:
    rng = np.random.default_rng(7)
    truth = rng.normal(size=60)
    report = golbraikh_tropsha_criteria(truth, rng.normal(size=60), q2=0.9)
    assert report["r2"] < 0.6
    assert report["r2_above_0_6"] is False
    assert report["passes"] is False


def test_golbraikh_tropsha_reports_none_for_undefined_quantities() -> None:
    report = golbraikh_tropsha_criteria([1.0, 2.0, 3.0], [2.0, 2.0, 2.0], q2=0.9)
    assert report["r2"] is None
    assert report["r02_prime"] is None
    assert report["r2_above_0_6"] is False
    assert report["passes"] is False


def test_golbraikh_tropsha_requires_q2() -> None:
    with pytest.raises(TypeError):
        golbraikh_tropsha_criteria([1.0, 2.0], [1.0, 2.0])  # type: ignore[call-arg]


# --------------------------------------------------------------------------------------
# aggregators
# --------------------------------------------------------------------------------------


def test_extended_classification_metrics_without_probabilities() -> None:
    metrics = extended_classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert set(metrics) == {"mcc", "cohen_kappa", "specificity", "n"}
    assert metrics["n"] == 4
    assert metrics["specificity"] == pytest.approx(0.5)


def test_extended_classification_metrics_with_probabilities() -> None:
    metrics = extended_classification_metrics(
        [0, 0, 1, 1],
        [0, 1, 1, 1],
        y_prob=[0.1, 0.6, 0.8, 0.9],
    )
    assert {"average_precision", "brier_score", "expected_calibration_error"} <= set(metrics)
    assert metrics["average_precision"] == pytest.approx(
        average_precision([0, 0, 1, 1], [0.1, 0.6, 0.8, 0.9])
    )


def test_extended_classification_metrics_degrades_for_non_binary_labels() -> None:
    metrics = extended_classification_metrics(
        ["low", "high", "mid"],
        ["low", "high", "high"],
        y_prob=[0.1, 0.9, 0.5],
    )
    assert set(metrics) == {"mcc", "cohen_kappa", "n"}
    assert metrics["mcc"] is not None


def test_extended_classification_metrics_drops_missing_pairs_and_realigns_probabilities() -> None:
    metrics = extended_classification_metrics(
        [0, 1, np.nan, 1],
        [0, 1, 0, 0],
        y_prob=[0.2, 0.9, 0.5, 0.3],
    )
    assert metrics["n"] == 3
    assert metrics["brier_score"] == pytest.approx(brier_score([0, 1, 1], [0.2, 0.9, 0.3]))


def test_extended_classification_metrics_empty_input() -> None:
    assert extended_classification_metrics([], []) == {}


def test_extended_classification_metrics_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        extended_classification_metrics([0, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        extended_classification_metrics([0, 1], [0, 1], y_prob=[0.5])


def test_extended_regression_metrics_without_training_data() -> None:
    metrics = extended_regression_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
    assert set(metrics) == {"ccc", "q2_f2", "n"}
    assert metrics["q2_f2"] == pytest.approx(0.5)


def test_extended_regression_metrics_with_training_data_and_q2() -> None:
    metrics = extended_regression_metrics(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.0],
        y_train=[0.0, 2.0, 4.0],
        q2=0.7,
    )
    assert metrics["q2_f1"] == pytest.approx(q2_f1([1.0, 2.0, 3.0], [1.0, 2.0, 4.0], 2.0))
    assert metrics["q2_f3"] == pytest.approx(0.875)
    assert metrics["golbraikh_tropsha"]["q2"] == pytest.approx(0.7)
    assert isinstance(metrics["golbraikh_tropsha"]["passes"], bool)


def test_extended_regression_metrics_empty_input() -> None:
    assert extended_regression_metrics([], []) == {}


def test_metric_values_are_floats_or_none_never_nan() -> None:
    metrics = {
        **extended_classification_metrics([0, 0, 0], [0, 0, 0], y_prob=[0.1, 0.2, 0.3]),
        **extended_regression_metrics([2.0, 2.0], [2.0, 2.0], y_train=[1.0, 1.0], q2=0.6),
    }
    for key, value in metrics.items():
        if key in {"n", "golbraikh_tropsha"}:
            continue
        assert value is None or isinstance(value, float), key
        assert value is None or math.isfinite(value), key
