from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from cs_copilot.tools.prediction.qsar_contracts import (
    ChempropTrainingRequest,
    LightGBMTrainingRequest,
    QsariaBenchmarkRequest,
    QsariaTrainingRequest,
    TabICLTrainingRequest,
    TuningObjective,
    canonical_validation_strategy,
)


def _lightgbm_payload() -> dict:
    return {
        "schema_version": "2.0",
        "smiles_column": "smiles",
        "target_columns": ["activity"],
        "task_type": "regression",
        "representation": {"kind": "generated", "name": "rdkit_all"},
        "validation": {"kind": "standard_qsar"},
        "tuning": {"enabled": True, "n_trials": 50},
        "backend": {"name": "lightgbm"},
    }


def test_public_contract_rejects_unknown_fields_recursively():
    payload = _lightgbm_payload()
    payload["backend"]["feature_cache_dir"] = "/tmp/cache"
    with pytest.raises(ValidationError, match="feature_cache_dir"):
        QsariaTrainingRequest.model_validate(payload)

    payload = _lightgbm_payload()
    payload["workflow"] = "benchmark"
    with pytest.raises(ValidationError, match="workflow"):
        QsariaTrainingRequest.model_validate(payload)


@pytest.mark.parametrize("old_protocol", ["fast_local", "robust_qsar", "challenging_qsar"])
def test_removed_validation_protocols_are_not_accepted(old_protocol: str):
    payload = _lightgbm_payload()
    payload["validation"] = {"kind": old_protocol}
    with pytest.raises(ValidationError):
        QsariaTrainingRequest.model_validate(payload)


def test_validation_contract_uses_canonical_names_only():
    payload = _lightgbm_payload()
    payload["validation"] = {"kind": "cv", "folds": 5}
    with pytest.raises(ValidationError):
        QsariaTrainingRequest.model_validate(payload)

    payload["validation"] = {
        "kind": "cross_validation",
        "n_folds": 5,
        "n_repeats": 2,
        "outer_test_size": 0.1,
    }
    request = QsariaTrainingRequest.model_validate(payload)
    assert canonical_validation_strategy(request.validation) == {
        "type": "cross_validation",
        "split_family": "random",
        "n_folds": 5,
        "n_repeats": 2,
        "outer_test_size": 0.1,
        "final_refit": True,
    }


@pytest.mark.parametrize("kind", ["holdout", "repeated_holdout", "cross_validation"])
def test_validation_contract_rejects_removed_selection_metric(kind: str):
    payload = _lightgbm_payload()
    payload["validation"] = {"kind": kind, "selection_metric": "rmse"}
    with pytest.raises(ValidationError, match="selection_metric"):
        QsariaTrainingRequest.model_validate(payload)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("MSE", "mse"),
        ("mean squared error", "mse"),
        ("MAE", "mae"),
        ("mean-absolute-error", "mae"),
        ("RMSE", "rmse"),
        ("root_mean_squared_error", "rmse"),
        ("R2", "r2"),
        ("r²", "r2"),
        ("R^2", "r2"),
        ("r2_score", "r2"),
        ("coefficient of determination", "r2"),
        ("acc", "accuracy"),
        ("balanced accuracy", "balanced_accuracy"),
        ("balanced acc", "balanced_accuracy"),
        ("precision macro", "precision_macro"),
        ("macro precision", "precision_macro"),
        ("recall macro", "recall_macro"),
        ("macro recall", "recall_macro"),
        ("F1 macro", "f1_macro"),
        ("macro F1", "f1_macro"),
        ("ROC-AUC", "roc_auc"),
        ("ROC AUC", "roc_auc"),
        ("AUC ROC", "roc_auc"),
        ("validation loss", "val_loss"),
        ("validation_loss", "val_loss"),
    ],
)
def test_tuning_metric_aliases_are_persisted_canonically(alias: str, canonical: str):
    objective = TuningObjective(metric=alias, direction="minimize")
    assert objective.metric == canonical
    assert objective.model_dump()["metric"] == canonical


@pytest.mark.parametrize("metric", ["auc", "r2ish", "unknown metric"])
def test_ambiguous_or_unknown_tuning_metrics_are_rejected(metric: str):
    with pytest.raises(ValidationError, match="metric"):
        TuningObjective(metric=metric, direction="maximize")


def test_tuning_objective_is_validated_for_backend_task_and_direction():
    payload = _lightgbm_payload()
    payload["tuning"]["objective"] = {
        "metric": "balanced accuracy",
        "direction": "maximize",
    }
    with pytest.raises(ValidationError, match="not compatible with LightGBM regression"):
        QsariaTrainingRequest.model_validate(payload)

    payload["tuning"]["objective"] = {"metric": "R²", "direction": "minimize"}
    with pytest.raises(ValidationError, match="must use direction `maximize`"):
        QsariaTrainingRequest.model_validate(payload)

    payload["backend"] = {"name": "chemprop"}
    payload["representation"] = {"kind": "molecular_graph"}
    payload["tuning"]["objective"] = {
        "metric": "validation loss",
        "direction": "maximize",
        "subset": "all",
    }
    with pytest.raises(ValidationError, match="must use direction `minimize`"):
        QsariaTrainingRequest.model_validate(payload)


def test_backend_and_representation_must_match():
    payload = _lightgbm_payload()
    payload["representation"] = {"kind": "molecular_graph"}
    with pytest.raises(ValidationError, match="generated or precomputed"):
        QsariaTrainingRequest.model_validate(payload)

    payload["backend"] = {"name": "chemprop"}
    request = QsariaTrainingRequest.model_validate(payload)
    assert request.backend.name == "chemprop"


def test_tuning_space_is_backend_specific_and_strict():
    payload = _lightgbm_payload()
    payload["tuning"] = {
        "enabled": True,
        "parameters": ["learning_rate"],
        "search_space": {
            "backend": "lightgbm",
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        },
    }
    request = QsariaTrainingRequest.model_validate(payload)
    assert request.tuning.search_space.learning_rate.high == 0.1

    payload["tuning"]["parameters"] = ["feature_cache_dir"]
    with pytest.raises(ValidationError, match="Unsupported lightgbm tuning parameters"):
        QsariaTrainingRequest.model_validate(payload)


def test_contract_is_type_strict_and_rejects_ineligible_scientific_combinations():
    payload = _lightgbm_payload()
    payload["tuning"]["n_trials"] = "50"
    with pytest.raises(ValidationError, match="valid integer"):
        QsariaTrainingRequest.model_validate(payload)

    payload = _lightgbm_payload()
    payload["validation"] = {"kind": "full_train"}
    with pytest.raises(ValidationError, match="validation subset"):
        QsariaTrainingRequest.model_validate(payload)

    payload["tuning"] = {"enabled": False}
    with pytest.raises(ValidationError, match="Outlier analysis requires"):
        QsariaTrainingRequest.model_validate(payload)

    payload["outlier_analysis"] = {"enabled": False}
    request = QsariaTrainingRequest.model_validate(payload)
    assert request.validation.kind == "full_train"


def test_disabled_tuning_cannot_hide_a_configured_search():
    payload = _lightgbm_payload()
    payload["tuning"] = {"enabled": False, "n_trials": 10}
    with pytest.raises(ValidationError, match="Disabled tuning"):
        QsariaTrainingRequest.model_validate(payload)


def test_lightgbm_standard_qsar_rejects_disabled_tuning():
    payload = _lightgbm_payload()
    payload["tuning"] = {"enabled": False}

    with pytest.raises(ValidationError, match="standard_qsar includes the canonical 50-trial"):
        QsariaTrainingRequest.model_validate(payload)


def test_tabicl_scientific_option_enums_are_closed():
    payload = TabICLTrainingRequest(
        target_columns=["activity"], task_type="regression"
    ).model_dump()
    payload["backend"]["feat_shuffle_method"] = "invented"
    with pytest.raises(ValidationError, match="literal_error"):
        TabICLTrainingRequest.model_validate(payload)


def test_tabicl_runtime_controls_are_not_public_and_tuning_is_disabled():
    request = TabICLTrainingRequest(target_columns=["activity"], task_type="regression")
    assert request.tuning.enabled is False

    payload = request.model_dump()
    payload["backend"]["device"] = "gpu"
    with pytest.raises(ValidationError, match="device"):
        TabICLTrainingRequest.model_validate(payload)

    payload = request.model_dump()
    payload["tuning"]["enabled"] = True
    with pytest.raises(ValidationError, match="does not support hyperparameter tuning"):
        TabICLTrainingRequest.model_validate(payload)


def test_specialized_contracts_narrow_the_backend():
    lightgbm = LightGBMTrainingRequest(target_columns=["y"], task_type="regression")
    chemprop = ChempropTrainingRequest(target_columns=["y"], task_type="regression")
    tabicl = TabICLTrainingRequest(target_columns=["y"], task_type="regression")
    assert (lightgbm.backend.name, chemprop.backend.name, tabicl.backend.name) == (
        "lightgbm",
        "chemprop",
        "tabicl",
    )


def test_chemprop_warmup_must_finish_before_training_ends():
    payload = ChempropTrainingRequest(
        target_columns=["y"], task_type="regression"
    ).model_dump()
    payload["backend"]["epochs"] = 1
    payload["backend"]["warmup_epochs"] = 2

    with pytest.raises(ValidationError, match="warmup_epochs must be strictly lower"):
        ChempropTrainingRequest.model_validate(payload)


def test_json_schema_forbids_additional_properties_and_internal_fields():
    schema = QsariaTrainingRequest.model_json_schema()
    serialized = json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert '"additionalProperties": false' in serialized
    for forbidden in ("extra_args", "feature_cache_dir", "heartbeat_path", "split_payload"):
        assert forbidden not in serialized
    assert "selection_metric" not in serialized

    def assert_closed(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                assert node.get("additionalProperties") is False
            for value in node.values():
                assert_closed(value)
        elif isinstance(node, list):
            for value in node:
                assert_closed(value)

    assert_closed(schema)
    assert_closed(QsariaBenchmarkRequest.model_json_schema())
