#!/usr/bin/env python
# coding: utf-8
"""Tests for canonical QSAR reporting handoffs."""

import json

from cs_copilot.tools.prediction.qsar_reporting import (
    build_external_evaluation_reporting_handoff,
    build_registry_reporting_handoff,
    build_training_reporting_handoff,
)
from cs_copilot.tools.prediction.qsar_response_compaction import (
    compact_model_payload_for_response,
    compact_prediction_result_for_response,
)
from cs_copilot.tools.prediction.qsar_training_toolkit import (
    _compact_registry_payload,
    _compact_training_tool_result,
)


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


def test_v2_ad_table_treats_omitted_method_status_category_as_zero():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "metrics_status": "evaluated",
            "applicability_domain": {
                "method": "combined",
                "feature_space": "rdkit_all",
                "methods": {
                    "bounding_box": {
                        "feature_space": "rdkit_all",
                        "rule": "strict_min_max_any_feature_outside_is_out_of_domain",
                    },
                    "isolation_forest": {
                        "feature_space": "rdkit_all",
                        "rule": "decision_function_less_than_zero_is_out_of_domain",
                    },
                },
                "split_score_summaries": {
                    "train": {
                        "row_count": 3718,
                        "n_in_domain": 3511,
                        "n_out_of_domain": 207,
                        "coverage_in_domain": 3511 / 3718,
                        "method_status_summaries": {
                            "bounding_box": {
                                "row_count": 3718,
                                "status_counts": {"in_domain": 3718},
                                "coverage_in_domain": 1.0,
                            },
                            "isolation_forest": {
                                "row_count": 3718,
                                "status_counts": {
                                    "in_domain": 3627,
                                    "out_of_domain": 91,
                                },
                                "coverage_in_domain": 3627 / 3718,
                            },
                        },
                    }
                },
            },
        }
    )

    table = handoff["report_tables"]["applicability_domain"]
    bounding_box = next(row for row in table["rows"] if row["method"] == "bounding_box")

    assert bounding_box["n"] == 3718
    assert bounding_box["n_in_domain"] == 3718
    assert bounding_box["n_out_of_domain"] == 0
    assert bounding_box["n_invalid_features"] == 0
    assert "| bounding_box | rdkit_all |" in table["markdown"]
    assert "| train | 3718 | 3718 | 0 | 0 | 1 |" in table["markdown"]


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
    assert (
        "| evaluation externe | out_of_domain | 1 | non calculable |"
        in handoff["evaluation_metrics_markdown"]
    )


def test_training_v2_handoff_normalizes_facts_and_limits_tables_to_three():
    def variant(name, rmse):
        return {
            "variant_id": name,
            "split_label": "holdout_1",
            "run": {
                "applicability_domain": {
                    "split_score_summaries": {
                        "test": {
                            "row_count": 5,
                            "n_in_domain": 4,
                            "n_out_of_domain": 1,
                            "coverage_in_domain": 0.8,
                            "metrics_all": {"r2": 0.5, "rmse": rmse, "mae": 0.4, "n": 5},
                            "metrics_in_domain": {"r2": 0.6, "rmse": rmse - 0.1, "n": 4},
                            "metrics_out_of_domain": {},
                        }
                    }
                }
            },
        }

    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "train_csv": "data/train.csv",
            "target_columns": ["pEC50"],
            "validation_protocol": "standard_qsar",
            "metrics_status": "evaluated",
            "curation": {
                "status": "completed",
                "ready_for_qsar": True,
                "rows_in": 100,
                "rows_out": 92,
                "retained_columns": ["SMILES", "pEC50"],
                "details": {
                    "invalid_smiles_removed": 2,
                    "curation_policy": {"duplicates": "aggregate"},
                    "curation_actions": ["standardized"],
                },
            },
            "activity_cliffs": {
                "enabled": True,
                "mode": "annotate",
                "index_name": "sali",
                "flagged_count": 4,
                "priority_counts": {"high": 1},
                "index_parameters": {"similarity_threshold": 0.7},
                "tiering_policy": "standard",
            },
            "applicability_domain": {
                "method": "bounding_box",
                "feature_space": "rdkit_all",
                "feature_count": 217,
                "fit_row_count": 80,
                "methods": {"bounding_box": {"feature_space": "rdkit_all", "threshold": 0.95}},
                "split_score_summaries": {
                    "validation": {
                        "row_count": 10,
                        "n_in_domain": 8,
                        "n_out_of_domain": 2,
                        "coverage_in_domain": 0.8,
                    },
                    "test": {
                        "row_count": 10,
                        "n_in_domain": 9,
                        "n_out_of_domain": 1,
                        "coverage_in_domain": 0.9,
                    },
                },
            },
            "selection_validation": {
                "metrics": {
                    "all": {"r2": 0.4, "rmse": 0.8, "mae": 0.5, "n": 10},
                    "in_domain": {"r2": 0.5, "rmse": 0.7, "mae": 0.4, "n": 8},
                    "out_of_domain": {"r2": 0.1, "rmse": 1.0, "mae": 0.9, "n": 2},
                }
            },
            "hyperparameter_tuning": {
                "engine": "optuna_tpe",
                "requested_trials": 50,
                "completed_trials": 50,
                "best_trial": {
                    "number": 12,
                    "params": {"num_leaves": 31},
                    "diagnostics": {"large": "not forwarded"},
                },
                "trials": [{"number": index, "large": "not forwarded"} for index in range(50)],
            },
            "outlier_analysis": {
                "enabled": True,
                "status": "completed",
                "eligible_count": 6,
                "selected_count": 1,
            },
            "outlier_model_variants": [variant("baseline", 0.7), variant("outlier_filtered", 0.6)],
        }
    )

    facts = handoff["report_facts"]
    tables = handoff["report_tables"]
    assert facts["schema_version"] == "2.0"
    assert facts["dataset"]["curation"]["retained_columns"] == ["SMILES", "pEC50"]
    assert facts["dataset"]["activity_cliffs"]["index_parameters"]["similarity_threshold"] == 0.7
    assert facts["hyperparameter_optimization"]["best_trial"]["number"] == 12
    assert "trials" not in facts["hyperparameter_optimization"]
    assert "not forwarded" not in str(facts["hyperparameter_optimization"])
    assert set(tables) == {
        "applicability_domain",
        "selection_validation_metrics",
        "final_test_comparison",
    }
    assert tables["final_test_comparison"]["markdown"].count("baseline") == 3
    assert tables["final_test_comparison"]["markdown"].count("outlier_filtered") == 3


def test_training_v2_handoff_marks_cv_without_outer_test_as_internal():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "tabicl",
            "task_type": "regression",
            "validation_protocol": "cross_validation",
            "validation_strategy_type": "cross_validation",
            "validation_strategy": {"n_folds": 5},
            "metrics_status": "evaluated",
        }
    )

    evaluation = handoff["report_facts"]["evaluation"]
    assert evaluation["scope"] == "internal"
    assert "No external test set" in evaluation["statement"]
    assert "final_test_comparison" not in handoff["report_tables"]


def test_training_v2_handoff_marks_standard_holdout_as_internal_and_separates_refit_counts():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "validation_protocol": "standard_qsar",
            "validation_strategy_type": "holdout",
            "validation_strategy": {
                "kind": "holdout",
                "split_family": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            },
            "metrics_status": "evaluated",
            "effective_train_count": 3718,
            "validation_count": 0,
            "test_count": 413,
            "final_refit": True,
            "selection_validation": {"diagnostics": {"validation_count": 413}},
        }
    )

    facts = handoff["report_facts"]
    assert facts["evaluation"]["scope"] == "internal"
    assert "No separate labelled external dataset" in facts["evaluation"]["statement"]
    assert facts["training_protocol"]["counts"]["selection_split"] == {
        "train_count": 3305,
        "validation_count": 413,
        "test_count": 413,
    }
    assert facts["training_protocol"]["counts"]["final_refit_split"] == {
        "train_count": 3718,
        "validation_count": 0,
        "test_count": 413,
    }


def test_training_v2_handoff_keeps_chemprop_native_val_loss_and_skipped_outliers():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "chemprop",
            "task_type": "regression",
            "validation_protocol": "standard_qsar",
            "metrics_status": "evaluated",
            "hyperparameter_tuning": {
                "status": "completed",
                "engine": "chemprop_hpopt_hyperopt",
                "requested_trials": 50,
                "completed_trials": 50,
                "objective": {"metric": "val_loss", "direction": "minimize"},
                "selection_protocol": {"applicability_domain_used_for_selection": False},
                "best_trial": {"number": 8, "value": 0.21, "params": {"depth": 4}},
                "trials": [{"number": 1, "checkpoint": "/tmp/heavy"}],
            },
            "outlier_analysis": {
                "enabled": False,
                "status": "skipped",
                "reason": "Disabled explicitly by outlier_analysis.enabled=false.",
            },
        }
    )

    facts = handoff["report_facts"]
    assert facts["hyperparameter_optimization"]["engine"] == "chemprop_hpopt_hyperopt"
    assert facts["hyperparameter_optimization"]["objective"]["metric"] == "val_loss"
    assert (
        facts["hyperparameter_optimization"]["selection_protocol"][
            "applicability_domain_used_for_selection"
        ]
        is False
    )
    assert "trials" not in facts["hyperparameter_optimization"]
    assert facts["outlier_analysis"]["status"] == "skipped"


def test_training_v2_handoff_marks_cv_with_outer_test_as_external():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "validation_protocol": "cross_validation",
            "validation_strategy_type": "cross_validation",
            "validation_strategy": {"n_folds": 5, "outer_test_size": 0.1},
            "metrics_status": "evaluated",
        }
    )

    assert handoff["report_facts"]["evaluation"]["scope"] == "external"


def test_training_v2_handoff_reports_automatic_development_only_ac_annotation(tmp_path):
    summary_path = tmp_path / "outlier_analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "selection_annotations": {
                    "activity_cliffs": {
                        "executed": True,
                        "flagged_count": 2,
                        "scope": "development_only",
                    },
                    "applicability_domain": {
                        "available": True,
                        "out_of_domain_count": 25,
                        "scope": "train_fit_validation_score",
                    },
                }
            }
        )
    )
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "metrics_status": "evaluated",
            "outlier_analysis": {
                "enabled": True,
                "status": "completed",
                "eligible_count": 5,
                "selected_count": 1,
                "summary_path": str(summary_path),
                "artifacts": {"summary_path": str(summary_path)},
            },
        }
    )

    facts = handoff["report_facts"]
    ac = facts["dataset"]["activity_cliffs"]
    assert ac["status"] == "selection_annotation_only"
    assert ac["flagged_count"] == 2
    assert ac["feedback_loops_requested"] == 0
    assert (
        facts["outlier_analysis"]["selection_annotations"]["applicability_domain"][
            "out_of_domain_count"
        ]
        == 25
    )


def test_training_v2_handoff_marks_full_train_as_not_internally_evaluated():
    handoff = build_training_reporting_handoff(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "validation_protocol": "full_train",
            "metrics_status": "not_evaluated",
            "evaluation_required": True,
        }
    )

    assert handoff["report_facts"]["evaluation"]["scope"] == "none"
    assert "no internal evaluation" in handoff["report_facts"]["evaluation"]["statement"]


def test_external_evaluation_v2_handoff_is_short_and_uses_only_external_facts():
    handoff = build_external_evaluation_reporting_handoff(
        {
            "evaluation_id": "pxr_eval_2",
            "dataset_path": "data/external.csv",
            "task_type": "regression",
            "target_columns": ["pEC50"],
            "row_count": 3,
            "metrics": {"r2": 0.4, "rmse": 0.8, "mae": 0.5, "mse": 0.64, "n": 3},
            "artifacts": {"evaluation_summary": "/tmp/evaluation_summary.json"},
        }
    )

    facts = handoff["report_facts"]
    assert facts["report_kind"] == "standalone_labelled_evaluation"
    assert facts["evaluation"]["dataset_path"] == "data/external.csv"
    assert "curation" not in facts
    assert set(handoff["report_tables"]) == {"external_test_results"}


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


def test_compact_training_result_omits_heavy_ad_feature_lists():
    feature_names = [f"fp_{index:04d}" for index in range(2048)]
    ad = {
        "available": True,
        "method": "combined",
        "feature_space": "morgan_only",
        "feature_count": 2048,
        "feature_names": feature_names,
        "feature_kinds": ["morgan_binary"] * 2048,
        "manifest_path": "/tmp/ad/manifest.json",
        "methods": {
            "similarity_matrix": {
                "feature_names": feature_names,
                "feature_kinds": ["morgan_binary"] * 2048,
                "feature_count": 2048,
                "subspaces": {
                    "morgan_binary": {
                        "feature_names": feature_names,
                        "feature_kinds": ["morgan_binary"] * 2048,
                        "feature_count": 2048,
                        "threshold": 0.25,
                        "matrix_all_path": "/tmp/matrix.npy",
                    }
                },
            }
        },
        "split_score_summaries": {
            "test": {
                "row_count": 10,
                "coverage_in_domain": 0.8,
                "metrics_all": {"r2": 0.4, "n": 10},
            }
        },
    }

    compact = _compact_training_tool_result(
        {
            "backend_name": "lightgbm",
            "task_type": "regression",
            "applicability_domain": ad,
            "recommended_registry_payload": {"applicability_domain": ad},
        }
    )
    registry = _compact_registry_payload({"applicability_domain": ad})
    rendered = str(compact) + str(registry)

    assert "feature_names" not in rendered
    assert "feature_kinds" not in rendered
    assert (
        compact["applicability_domain"]["methods"]["similarity_matrix"]["subspaces"][
            "morgan_binary"
        ]["threshold"]
        == 0.25
    )
    assert (
        compact["applicability_domain"]["split_score_summaries"]["test"]["metrics_all"]["r2"] == 0.4
    )


def test_common_compaction_handles_catalog_model_payloads():
    feature_columns = [f"fp_{index:04d}" for index in range(64)]
    feature_names = [f"fp_{index:04d}" for index in range(2048)]

    compact = compact_model_payload_for_response(
        {
            "model_id": "model",
            "inference_profile": {"feature_columns": feature_columns},
            "applicability_domain": {
                "feature_names": feature_names,
                "methods": {"bounding_box": {"feature_names": feature_names}},
            },
        }
    )
    rendered = str(compact)

    assert "feature_names" not in rendered
    assert "feature_columns" not in compact["inference_profile"]
    assert compact["inference_profile"]["feature_columns_count"] == 64


def test_common_compaction_handles_prediction_results():
    feature_columns = [f"fp_{index:04d}" for index in range(64)]
    feature_names = [f"fp_{index:04d}" for index in range(2048)]

    compact = compact_prediction_result_for_response(
        {
            "preds_path": "/tmp/preds.csv",
            "feature_columns": feature_columns,
            "applicability_domain": {"feature_names": feature_names, "feature_count": 2048},
        }
    )
    rendered = str(compact)

    assert "feature_names" not in rendered
    assert "feature_columns" not in compact
    assert compact["feature_columns_count"] == 64
    assert compact["applicability_domain"]["feature_count"] == 2048
