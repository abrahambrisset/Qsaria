#!/usr/bin/env python
# coding: utf-8
"""Tests for the extended metrics wired into the live training path.

Values are hand-computed closed forms, so the assertions fail if the wiring passes the
wrong vector (predictions instead of truth, unaligned probabilities, the test mean in
place of the training mean) rather than merely if a key is missing.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cs_copilot.tools.prediction.qsar_training_policy import aggregate_split_families
from cs_copilot.tools.prediction.training_orchestration import (
    compute_classification_metrics,
    compute_regression_metrics,
)

# --------------------------------------------------------------------------------------
# regression
# --------------------------------------------------------------------------------------


def test_regression_metrics_gain_ccc_and_q2_f2() -> None:
    """A pure +1 offset gives CCC = 4/7 and Q2F2 = -0.5 in closed form.

    y = [1, 2, 3], yhat = y + 1. var_pop(y) = var_pop(yhat) = 2/3, cov_pop = 2/3,
    so CCC = 2 * (2/3) / (2/3 + 2/3 + 1^2) = (4/3) / (7/3) = 4/7 -- below the Pearson
    r of 1.0, because CCC penalises the bias that r ignores. Q2F2 is centred on the
    test mean: 1 - PRESS / TSS = 1 - 3 / 2 = -0.5.
    """
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([2.0, 3.0, 4.0])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["ccc"] == pytest.approx(4.0 / 7.0)
    assert metrics["q2_f2"] == pytest.approx(-0.5)
    assert metrics["r2"] == pytest.approx(-0.5)


def test_regression_metrics_keep_existing_keys_and_semantics() -> None:
    """The pre-existing keys keep their exact names, types and values."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.0, 2.0, 3.0, 5.0])

    metrics = compute_regression_metrics(y_true, y_pred, target_column="pIC50")

    assert metrics["mse"] == pytest.approx(0.25)
    assert metrics["mae"] == pytest.approx(0.25)
    assert metrics["rmse"] == pytest.approx(0.5)
    assert metrics["r2"] == pytest.approx(0.8)
    assert metrics["n"] == 4
    assert metrics["target_column"] == "pIC50"
    assert isinstance(metrics["spearman"], float)
    assert isinstance(metrics["kendall"], float)


def test_regression_n_counts_the_rows_the_old_contract_counted() -> None:
    """NaN pairs are dropped by both layers, so `n` is unchanged by the merge."""
    y_true = pd.Series([1.0, 2.0, float("nan"), 4.0])
    y_pred = pd.Series([1.5, 2.5, 3.5, float("nan")])

    metrics = compute_regression_metrics(y_true, y_pred)

    assert metrics["n"] == 2


def test_regression_omits_training_set_statistics() -> None:
    """Q2F1/Q2F3/Golbraikh-Tropsha need training data this function never receives."""
    metrics = compute_regression_metrics(pd.Series([1.0, 2.0, 3.0]), pd.Series([1.1, 2.1, 2.9]))

    assert "q2_f1" not in metrics
    assert "q2_f3" not in metrics
    assert "golbraikh_tropsha" not in metrics


def test_regression_empty_input_still_returns_empty_dict() -> None:
    """The documented empty-result contract is unchanged."""
    assert compute_regression_metrics(pd.Series([float("nan")]), pd.Series([1.0])) == {}


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------


def test_binary_classification_gains_mcc_kappa_and_specificity() -> None:
    """Hand-built confusion matrix TP=3, TN=2, FP=1, FN=2 on eight rows.

    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        = (3*2 - 1*2) / sqrt(4 * 5 * 3 * 4) = 4 / sqrt(240).
    Specificity = TN / (TN + FP) = 2 / 3.
    Accuracy = 5 / 8; expected agreement = (5*4 + 3*4) / 64 = 0.5, so
    kappa = (0.625 - 0.5) / 0.5 = 0.25.
    """
    truth = [1, 1, 1, 1, 1, 0, 0, 0]
    predicted = [1, 1, 1, 0, 0, 1, 0, 0]

    metrics = compute_classification_metrics(pd.Series(truth), pd.Series(predicted))

    assert metrics["mcc"] == pytest.approx(4.0 / math.sqrt(240.0))
    assert metrics["specificity"] == pytest.approx(2.0 / 3.0)
    assert metrics["cohen_kappa"] == pytest.approx(0.25)
    assert metrics["accuracy"] == pytest.approx(0.625)


def test_specificity_is_not_recall() -> None:
    """A model that never predicts the negative class has specificity 0, recall 1."""
    metrics = compute_classification_metrics(pd.Series([0, 0, 1, 1]), pd.Series([1, 1, 1, 1]))

    assert metrics["specificity"] == pytest.approx(0.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_probability_metrics_appear_only_with_positive_scores() -> None:
    """Without probabilities the probability-based keys are omitted, not faked."""
    metrics = compute_classification_metrics(pd.Series([0, 1, 0, 1]), pd.Series([0, 1, 1, 1]))

    assert "mcc" in metrics
    for key in ("average_precision", "brier_score", "expected_calibration_error"):
        assert key not in metrics


def test_probability_metrics_closed_form() -> None:
    """Perfectly confident and correct probabilities give Brier 0, ECE 0, AP 1."""
    truth = pd.Series([0, 0, 1, 1])
    scores = pd.Series([0.0, 0.0, 1.0, 1.0])

    metrics = compute_classification_metrics(truth, truth, positive_scores=scores)

    assert metrics["brier_score"] == pytest.approx(0.0)
    assert metrics["expected_calibration_error"] == pytest.approx(0.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_brier_and_ece_use_truth_not_predictions() -> None:
    """Brier = mean((p - y)^2) = (0.25 + 0.25 + 0.04 + 0.04) / 4 = 0.145.

    With ten equal-width bins the two 0.5 scores share bin [0.5, 0.6) and their observed
    rate is also 0.5, so that bin contributes nothing. The 0.2 and 0.8 scores sit alone
    in their bins with a 0.2 gap each: ECE = 0.25 * 0.2 + 0.25 * 0.2 = 0.1.
    Both are read against the truth, not the predicted labels, which disagree on two of
    the four rows.
    """
    truth = pd.Series([0, 1, 0, 1])
    predicted = pd.Series([1, 0, 0, 1])
    scores = pd.Series([0.5, 0.5, 0.2, 0.8])

    metrics = compute_classification_metrics(truth, predicted, positive_scores=scores)

    assert metrics["brier_score"] == pytest.approx(0.145)
    assert metrics["expected_calibration_error"] == pytest.approx(0.1)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_out_of_range_scores_are_ranked_but_not_reported_as_probabilities() -> None:
    """A raw decision score still yields roc_auc, but no Brier/ECE/AP."""
    truth = pd.Series([0, 0, 1, 1])
    scores = pd.Series([-2.5, -1.0, 1.0, 3.5])

    metrics = compute_classification_metrics(truth, truth, positive_scores=scores)

    assert metrics["roc_auc"] == pytest.approx(1.0)
    for key in ("average_precision", "brier_score", "expected_calibration_error"):
        assert key not in metrics


def test_probabilities_are_aligned_to_the_dropped_rows() -> None:
    """A missing label drops its row from truth and from the probability vector together.

    Rows 0 and 2 survive, carrying probabilities 0.0 and 1.0 against the negative and
    positive class, so the Brier score is 0.0. Shifting the probability vector by one
    row instead gives 0.5.
    """
    truth = pd.Series(["neg", None, "pos", None], dtype=object)
    predicted = pd.Series(["neg", "pos", "pos", "neg"], dtype=object)
    scores = pd.Series([0.0, 1.0, 1.0, 0.0])

    metrics = compute_classification_metrics(
        truth, predicted, class_labels=["neg", "pos"], positive_scores=scores
    )

    assert metrics["n"] == 2
    assert metrics["positive_class"] == "pos"
    assert metrics["brier_score"] == pytest.approx(0.0)


def test_single_class_average_precision_is_none_not_zero() -> None:
    """One class present makes average precision undefined; sklearn would say 0.0."""
    truth = pd.Series([1, 1, 1])
    metrics = compute_classification_metrics(
        truth,
        pd.Series([1, 0, 1]),
        class_labels=[0, 1],
        positive_scores=pd.Series([0.9, 0.2, 0.8]),
    )

    assert metrics["average_precision"] is None
    assert metrics["specificity"] is None


def test_multiclass_keeps_mcc_and_kappa_but_drops_binary_only_metrics() -> None:
    """Three labels: `mcc`/`cohen_kappa` stay, the binary-only keys are omitted."""
    truth = pd.Series(["a", "b", "c", "a", "b", "c"])
    predicted = pd.Series(["a", "b", "c", "a", "c", "c"])

    metrics = compute_classification_metrics(truth, predicted)

    assert metrics["num_classes"] == 3
    assert isinstance(metrics["mcc"], float)
    assert isinstance(metrics["cohen_kappa"], float)
    for key in ("specificity", "average_precision", "brier_score"):
        assert key not in metrics


def test_classification_keeps_existing_keys_and_semantics() -> None:
    """The pre-existing keys keep their exact names, types and values."""
    truth = pd.Series(["inactive", "active", "inactive", "active"])
    predicted = pd.Series(["inactive", "active", "active", "active"])

    metrics = compute_classification_metrics(truth, predicted, target_column="label")

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(2.0 / 3.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["n"] == 4
    assert metrics["num_classes"] == 2
    assert metrics["class_count"] == 2
    assert metrics["class_labels"] == ["inactive", "active"]
    assert metrics["positive_class"] == "active"
    assert metrics["negative_class"] == "inactive"
    assert metrics["target_column"] == "label"
    assert [row["class_label"] for row in metrics["per_class"]] == ["inactive", "active"]


def test_classification_metric_values_are_json_safe() -> None:
    """Merged values stay plain floats or None, never numpy scalars."""
    metrics = compute_classification_metrics(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1, 1, 1]),
        positive_scores=pd.Series([0.1, 0.9, 0.6, 0.8]),
    )

    for key in (
        "mcc",
        "cohen_kappa",
        "specificity",
        "average_precision",
        "brier_score",
        "expected_calibration_error",
    ):
        assert metrics[key] is None or type(metrics[key]) is float


# --------------------------------------------------------------------------------------
# aggregation across split families
# --------------------------------------------------------------------------------------


def test_aggregate_split_families_averages_the_new_metrics() -> None:
    """Means and stds are produced for the extended names, population std as before.

    mcc values 0.2 and 0.6 -> mean 0.4, population std 0.2.
    """
    split_results = [
        {
            "strategy_family": "random",
            "strategy_label": "random_seed_0",
            "metrics": {"test": {"n": 10, "mcc": 0.2, "ccc": 0.5, "specificity": 0.4}},
        },
        {
            "strategy_family": "random",
            "strategy_label": "random_seed_1",
            "metrics": {"test": {"n": 10, "mcc": 0.6, "ccc": 0.9, "specificity": 0.6}},
        },
    ]

    aggregated = aggregate_split_families(split_results)["random"]

    assert aggregated["mcc_mean"] == pytest.approx(0.4)
    assert aggregated["mcc_std"] == pytest.approx(0.2)
    assert aggregated["ccc_mean"] == pytest.approx(0.7)
    assert aggregated["specificity_mean"] == pytest.approx(0.5)


def test_aggregate_split_families_tolerates_missing_and_none_metrics() -> None:
    """Absent or None extended metrics are skipped, not coerced to 0.0."""
    split_results = [
        {
            "strategy_family": "scaffold",
            "strategy_label": "scaffold_seed_0",
            "metrics": {"test": {"n": 5, "r2": 0.5, "average_precision": None}},
        }
    ]

    aggregated = aggregate_split_families(split_results)["scaffold"]

    assert "average_precision_mean" not in aggregated
    assert "brier_score_mean" not in aggregated
    assert aggregated["r2"] == pytest.approx(0.5)


def test_aggregate_split_families_handles_a_full_metric_dict() -> None:
    """A real `compute_classification_metrics` output aggregates without raising.

    The nested `per_class`/`golbraikh_tropsha`-style values must not reach `float()`.
    """
    metrics = compute_classification_metrics(
        pd.Series([0, 1, 0, 1]),
        pd.Series([0, 1, 1, 1]),
        positive_scores=pd.Series([0.1, 0.9, 0.6, 0.8]),
    )
    split_results = [
        {"strategy_family": "random", "strategy_label": "run", "metrics": {"test": metrics}}
    ]

    aggregated = aggregate_split_families(split_results)["random"]

    assert aggregated["num_runs"] == 1
    assert aggregated["mcc"] == pytest.approx(metrics["mcc"])
    assert math.isclose(aggregated["brier_score"], metrics["brier_score"])
