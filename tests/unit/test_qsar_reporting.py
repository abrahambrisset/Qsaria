#!/usr/bin/env python
# coding: utf-8
"""Tests for canonical QSAR reporting handoffs."""

from cs_copilot.tools.prediction.qsar_reporting import (
    build_external_evaluation_reporting_handoff,
    build_registry_reporting_handoff,
    build_training_reporting_handoff,
)
from cs_copilot.tools.prediction.qsar_training_toolkit import _compact_training_tool_result


def test_classification_training_handoff_keeps_validation_test_ad_rows():
    result = {
        "backend_name": "chemprop",
        "task_type": "classification",
        "metrics_status": "evaluated",
        "target_columns": ["Y"],
        "applicability_domain": {
            "method": "bounding_box",
            "feature_space": "chemprop_embedding",
            "split_score_summaries": {
                "validation": {
                    "row_count": 10,
                    "n_in_domain": 8,
                    "n_out_of_domain": 2,
                    "coverage_in_domain": 0.8,
                    "metrics_all": {
                        "accuracy": 0.7,
                        "balanced_accuracy": 0.71,
                        "f1_macro": 0.69,
                        "roc_auc": 0.8,
                        "n": 10,
                    },
                    "metrics_in_domain": {"accuracy": 0.75, "n": 8},
                    "metrics_out_of_domain": {},
                },
                "test": {
                    "row_count": 12,
                    "n_in_domain": 11,
                    "n_out_of_domain": 1,
                    "coverage_in_domain": 0.9167,
                    "metrics_all": {"accuracy": 0.8, "n": 12},
                    "metrics_in_domain": {"accuracy": 0.82, "n": 11},
                    "metrics_out_of_domain": {},
                },
            },
        },
    }

    table = build_training_reporting_handoff(result)["evaluation_metrics_markdown"]

    assert table.count("| validation interne |") == 3
    assert table.count("| test interne |") == 3
    assert "| validation interne | out_of_domain | 2 | non calculable |" in table
    assert "| test interne | out_of_domain | 1 | non calculable |" in table
    assert "Commentaire :" not in table


def test_regression_training_handoff_uses_regression_columns():
    result = {
        "backend_name": "lightgbm",
        "task_type": "regression",
        "metrics_status": "evaluated",
        "target_columns": ["pEC50"],
        "applicability_domain": {
            "split_score_summaries": {
                "test": {
                    "row_count": 5,
                    "n_in_domain": 4,
                    "n_out_of_domain": 1,
                    "metrics_all": {
                        "r2": 0.5,
                        "rmse": 0.7,
                        "mae": 0.4,
                        "mse": 0.49,
                        "n": 5,
                    },
                    "metrics_in_domain": {"r2": 0.6, "n": 4},
                    "metrics_out_of_domain": {},
                },
            },
        },
    }

    table = build_training_reporting_handoff(result)["evaluation_metrics_markdown"]

    assert "| Source | AD subset | n | R2 | RMSE | MAE | MSE |" in table
    assert "| test interne | all | 5 | 0.5 | 0.7 | 0.4 | 0.49 |" in table
    assert "| test interne | out_of_domain | 1 | non calculable |" in table
    assert "Commentaire :" not in table


def test_applicability_domain_handoff_keeps_coverage_table_uninterpreted():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "classification",
            "metrics_status": "evaluated",
            "applicability_domain": {
                "method": "bounding_box",
                "feature_space": "rdkit_all",
                "feature_count": 217,
                "fit_row_count": 10,
                "split_score_summaries": {
                    "train": {
                        "row_count": 10,
                        "coverage_in_domain": 1.0,
                        "n_out_of_domain": 0,
                        "n_invalid_features": 0,
                    },
                    "test": {
                        "row_count": 4,
                        "coverage_in_domain": 0.75,
                        "n_out_of_domain": 1,
                        "n_invalid_features": 0,
                    },
                },
            },
        }
    )

    ad_markdown = handoff["applicability_domain_markdown"]

    assert "| Source | n | coverage_in_domain |" in ad_markdown
    assert "| test | 4 | 0.75 | 1 | 0 | - |" in ad_markdown
    assert "Commentaire :" not in ad_markdown


def test_full_train_handoff_states_not_evaluated():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "metrics_status": "not_evaluated",
            "evaluation_required": True,
        }
    )

    assert "metrics_status=not_evaluated" in handoff["decision_summary"]
    assert "Aucune metrique interne" in handoff["evaluation_metrics_markdown"]
    assert "evaluation externe labelisee" in handoff["governance_markdown"]


def test_external_evaluation_handoff_is_append_only_and_ad_aware():
    handoff = build_external_evaluation_reporting_handoff(
        {
            "evaluation_id": "pxr_eval_1",
            "dataset_path": "data/test.csv",
            "task_type": "regression",
            "target_columns": ["pEC50"],
            "row_count": 3,
            "ad_metrics_by_target": {
                "pEC50": {
                    "metrics_all": {"r2": 0.4, "rmse": 0.8, "mae": 0.5, "mse": 0.64, "n": 3},
                    "metrics_in_domain": {"r2": 0.5, "n": 2},
                    "metrics_out_of_domain": {},
                    "n_out_of_domain": 1,
                }
            },
        }
    )

    assert "append-only" in handoff["decision_summary"]
    assert "| evaluation externe | out_of_domain | 1 | non calculable |" in handoff[
        "evaluation_metrics_markdown"
    ]


def test_registry_handoff_reports_status_adjustment_reason():
    handoff = build_registry_reporting_handoff(
        requested_status="experimental",
        final_status="workflow_demo",
        status_reason="Requested `experimental` was adjusted to `workflow_demo`.",
        metrics_status="evaluated",
        payload={"model_path": "/tmp/model.pkl", "metadata_path": "/tmp/metadata.json"},
    )

    governance = handoff["governance_markdown"]
    assert "Statut demande: `experimental`" in governance
    assert "Statut final: `workflow_demo`" in governance
    assert "Requested `experimental` was adjusted" in governance


def test_compact_training_result_preserves_reporting_handoff():
    compact = _compact_training_tool_result(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "reporting_handoff": {"evaluation_metrics_markdown": "| Source |"},
        }
    )

    assert compact["reporting_handoff"]["evaluation_metrics_markdown"] == "| Source |"
