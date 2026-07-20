import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

import cs_copilot.tools.prediction.catalog as catalog_module
import cs_copilot.tools.prediction.model_registry_toolkit as registry_module
from cs_copilot.tools.prediction.applicability_domain import (
    fit_bounding_box_domain,
    fit_modern_applicability_domain,
    score_record_applicability_domain,
)
from cs_copilot.tools.prediction.backend import (
    InvalidPredictionInputError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from cs_copilot.tools.prediction.backend_capabilities import (
    backend_requires_feature_preparation,
    backend_supports_component_orchestration,
    describe_backend_capabilities,
    get_backend_capabilities,
)
from cs_copilot.tools.prediction.backend_factory import build_default_prediction_backends
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog
from cs_copilot.tools.prediction.chemprop_adapter import materialize_chemprop_inputs
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit, _agent_storage_path
from cs_copilot.tools.prediction.lightgbm_backend import LightGBMBackend
from cs_copilot.tools.prediction.model_registry_toolkit import ModelRegistryToolkit
from cs_copilot.tools.prediction.prediction_inference_toolkit import PredictionInferenceToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit
from cs_copilot.tools.prediction.session_state import (
    bundle_artifacts,
    discover_curation_artifacts_near_dataset,
    get_prediction_state,
    latest_curation_artifacts,
    write_active_training_marker,
)
from cs_copilot.tools.prediction.tabicl_backend import (
    DEFAULT_TABICL_CLASSIFIER_CHECKPOINT,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    TabICLBackend,
)
from cs_copilot.tools.prediction.training_orchestration import (
    apply_training_profile,
    build_training_plots_if_possible,
    collect_training_bundle_files,
    compute_classification_metrics,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    write_training_summary,
)


def test_collision_safe_model_id_preserves_legacy_first_and_isolates_later_models(
    monkeypatch,
    tmp_path,
):
    internal_root = tmp_path / "internal"
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    task = PredictionTaskSpec(
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["logS"],
    )
    base_model_id = "solubility_dataset_protocol_lightgbm_v1_17072026_120000"
    existing_path = tmp_path / "existing.pkl"
    current_path = tmp_path / "current.pkl"
    existing_path.write_bytes(b"first")
    current_path.write_bytes(b"second")
    existing = PredictionModelRecord(
        model_id=base_model_id,
        backend_name="lightgbm",
        model_path=str(existing_path),
        task=task,
    )
    current = PredictionModelRecord(
        model_id=base_model_id,
        backend_name="lightgbm",
        model_path=str(current_path),
        task=task,
    )
    catalog = PredictionModelCatalog(
        records=[existing],
        source_path=tmp_path / "catalog.json",
    )

    isolated_id = registry_module._collision_safe_model_id(
        base_model_id,
        current=current,
        catalog=catalog,
    )
    assert isolated_id != base_model_id
    assert isolated_id.startswith(f"{base_model_id}_")
    assert isolated_id == registry_module._collision_safe_model_id(
        base_model_id,
        current=current,
        catalog=catalog,
    )
    free_model_id = f"{base_model_id}_free"
    assert (
        registry_module._collision_safe_model_id(
            free_model_id,
            current=current,
            catalog=catalog,
        )
        == free_model_id
    )

    idempotent = PredictionModelRecord(
        model_id=base_model_id,
        backend_name="lightgbm",
        model_path=str(existing_path),
        task=task,
    )
    assert (
        registry_module._collision_safe_model_id(
            base_model_id,
            current=idempotent,
            catalog=catalog,
        )
        == base_model_id
    )


class _FakeLightGBMPredictor:
    def predict(self, features):
        return [float(index) for index in range(len(features))]


def test_backend_capabilities_registry_core_contracts():
    chemprop = get_backend_capabilities("chemprop")
    lightgbm = get_backend_capabilities("lightgbm")
    tabicl = get_backend_capabilities("tabicl")
    ensemble = get_backend_capabilities("ensemble")

    assert chemprop.requires_feature_preparation is False
    assert backend_requires_feature_preparation("lightgbm") is True
    assert tabicl.requires_feature_preparation is True
    assert ensemble.supports_component_orchestration is True
    assert backend_supports_component_orchestration("ensemble") is True
    assert ensemble.supports_uncertainty == "component_disagreement_std"
    assert lightgbm.supports_activity_cliff_feedback_loops is True
    assert chemprop.supports_activity_cliff_feedback_loops is False
    assert "classification" in chemprop.supported_task_types
    assert "multiclass_classification" in chemprop.supported_task_types
    assert "classification" in lightgbm.supported_task_types
    assert "multiclass_classification" in lightgbm.supported_task_types
    assert "classification" in tabicl.supported_task_types
    assert "multiclass_classification" in tabicl.supported_task_types
    assert "classification" in ensemble.supported_task_types
    assert "classification" in chemprop.multi_target_task_types
    assert "multiclass_classification" in chemprop.multi_target_task_types
    assert tabicl.multi_target_task_types == ()
    assert chemprop.gpu_support == "runtime_dependent"
    assert lightgbm.gpu_support == "supported_when_available"
    assert ensemble.gpu_support == "not_applicable"
    assert tabicl.catalog_model_filename == "best.pkl"
    assert "tabicl_training_summary.json" in tabicl.training_summary_filenames


def test_backend_capabilities_unknown_backend_is_clear():
    with pytest.raises(KeyError, match="No backend capabilities registered"):
        get_backend_capabilities("unknown_backend")


def test_describe_backend_capabilities_is_serializable():
    payload = describe_backend_capabilities()

    json.dumps(payload)
    assert payload["chemprop"]["backend_name"] == "chemprop"
    assert "morgan_rdkit_all" in payload["lightgbm"]["supported_representations"]
    assert "morgan_count_only" in payload["lightgbm"]["supported_representations"]
    assert "morgan_binary_count_rdkit_all" in payload["tabicl"]["supported_representations"]
    assert payload["lightgbm"]["gpu_support"] == "supported_when_available"


def test_shared_prediction_state_helpers_initialize_backend_neutral_state(tmp_path):
    agent = SimpleNamespace(session_state={})

    state = get_prediction_state(agent)

    assert state["registered"] == {}
    assert state["prediction_history"] == []
    assert state["training_runs"] == []
    assert state["active_training_run"] is None

    marker = tmp_path / "run" / ".training_in_progress"
    write_active_training_marker(marker, {"status": "running", "backend_name": "fake"})
    assert json.loads(marker.read_text())["backend_name"] == "fake"


def test_shared_curation_artifact_helpers_are_backend_neutral(tmp_path):
    report = tmp_path / "dataset_curation_report.json"
    report.write_text("{}")
    artifacts_dir = tmp_path / "dataset_curation_artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "curation_manifest.json").write_text("{}")
    dataset = tmp_path / "dataset_curated.csv"
    dataset.write_text("smiles,pEC50\nCCO,5.0\n")

    discovered = discover_curation_artifacts_near_dataset(str(dataset))

    assert discovered["curated_dataset_path"] == str(dataset)
    assert discovered["artifacts"]["curated_dataset_csv"] == str(dataset)
    assert discovered["artifacts"]["curation_report_json"] == str(report)
    assert discovered["artifacts"]["manifest_json"] == str(artifacts_dir / "curation_manifest.json")

    agent = SimpleNamespace(
        session_state={
            "qsar_curation": {
                "last_result": {
                    "curation_backend_used": "chembl_structure_v1",
                    "curated_dataset_path": str(dataset),
                    "rows_in": 2,
                    "rows_out": 1,
                    "report_path": str(report),
                }
            }
        }
    )
    latest = latest_curation_artifacts(agent)

    assert latest["curation_backend"] == "chembl_structure_v1"
    assert latest["artifacts"]["curated_dataset_csv"] == str(dataset)
    assert latest["artifacts"]["curation_report_json"] == str(report)


def test_shared_bundle_artifacts_deduplicates_archive_names(tmp_path):
    first = tmp_path / "a" / "same.txt"
    second = tmp_path / "b" / "same.txt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text("first")
    second.write_text("second")

    bundle = bundle_artifacts(tmp_path / "bundle.zip", [first, second])

    assert bundle.exists()
    with ZipFile(bundle) as zf:
        names = zf.namelist()
    assert len(names) == 2
    assert len(set(names)) == 2


def test_shared_bundle_artifacts_expands_directories_to_files(tmp_path):
    run_dir = tmp_path / "lightgbm_standard_qsar"
    model_path = run_dir / "random_seed_1_split" / "model_0" / "best.pkl"
    predictions_path = run_dir / "scaffold_split" / "model_0" / "test_predictions.csv"
    empty_dir = run_dir / "empty"
    active_marker = run_dir / ".training_in_progress"
    model_path.parent.mkdir(parents=True)
    predictions_path.parent.mkdir(parents=True)
    empty_dir.mkdir(parents=True)
    model_path.write_text("model")
    predictions_path.write_text("prediction")
    active_marker.write_text("{}")

    bundle = bundle_artifacts(tmp_path / "bundle.zip", [run_dir])

    with ZipFile(bundle) as zf:
        names = zf.namelist()
    assert any(name.endswith("random_seed_1_split/model_0/best.pkl") for name in names)
    assert any(name.endswith("scaffold_split/model_0/test_predictions.csv") for name in names)
    assert all(not name.endswith("/") for name in names)
    assert all("empty" not in name for name in names)
    assert ".training_in_progress" not in names


def test_shared_bundle_artifacts_writes_relative_file_names(tmp_path):
    file_path = tmp_path / "app" / ".files" / "sessions" / "thread" / "run" / "summary.json"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("{}")

    bundle = bundle_artifacts(tmp_path / "bundle.zip", [file_path])

    with ZipFile(bundle) as zf:
        names = zf.namelist()
    assert names == ["summary.json"]
    assert all(not name.startswith("/") for name in names)


def test_training_orchestration_normalizes_agent_list_arguments():
    assert normalize_json_list_argument("pEC50", argument_name="target_columns") == ["pEC50"]
    assert normalize_json_list_argument('["smiles"]', argument_name="smiles_columns") == ["smiles"]
    assert normalize_json_list_argument(
        "0.8,0.1,0.1", argument_name="split_sizes", coerce_numbers=True
    ) == [
        0.8,
        0.1,
        0.1,
    ]


def test_training_orchestration_classification_metrics_binary_text_labels():
    metrics = compute_classification_metrics(
        pd.Series(["inactive", "active", "active", "inactive"]),
        pd.Series(["inactive", "active", "inactive", "inactive"]),
        positive_scores=pd.Series([0.1, 0.9, 0.4, 0.2]),
    )

    assert metrics["n"] == 4
    assert metrics["class_count"] == 2
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["f1_macro"] > 0.7
    assert metrics["roc_auc"] == 1.0
    assert metrics["positive_class"] == "active"


def test_training_orchestration_classification_metrics_multiclass_text_labels():
    metrics = compute_classification_metrics(
        pd.Series(["low", "medium", "high", "high"]),
        pd.Series(["low", "medium", "medium", "high"]),
    )

    assert metrics["class_count"] == 3
    assert metrics["balanced_accuracy"] > 0.6
    assert "roc_auc" not in metrics


def test_chemprop_adapter_encodes_binary_classification_labels(tmp_path):
    source = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN", "COC"],
            "active": ["inactive", "active", "inactive", "active"],
            "extra": [1, 2, 3, 4],
        }
    ).to_csv(source, index=False)

    result = materialize_chemprop_inputs(
        source_csv=str(source),
        output_dir=tmp_path / "chemprop_inputs",
        task=PredictionTaskSpec(
            task_type="classification", smiles_columns=["smiles"], target_columns=["active"]
        ),
        split_payload=[{"train": [0, 1], "test": [2, 3]}],
        split_label="random",
        seed=123,
    )

    clean = pd.read_csv(result["chemprop_training_input_csv"])
    manifest = json.loads(Path(result["manifest_path"]).read_text())
    assert clean.columns.tolist() == ["smiles", "active"]
    assert sorted(clean["active"].unique().tolist()) == [0, 1]
    assert manifest["classification_targets"]["active"]["class_count"] == 2
    assert json.loads(Path(result["chemprop_splits_file"]).read_text()) == [
        {"train": [0, 1], "test": [2, 3]}
    ]


def test_chemprop_adapter_rejects_multiclass_classification(tmp_path):
    source = tmp_path / "training.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC", "CCN"], "label": ["low", "medium", "high"]}).to_csv(
        source, index=False
    )

    with pytest.raises(InvalidPredictionInputError, match="exactly two classes|multiclass"):
        materialize_chemprop_inputs(
            source_csv=str(source),
            output_dir=tmp_path / "chemprop_inputs",
            task=PredictionTaskSpec(
                task_type="classification", smiles_columns=["smiles"], target_columns=["label"]
            ),
            split_payload=[{"train": [0, 1], "test": [2]}],
            split_label="random",
            seed=123,
        )


def test_chemprop_toolkit_normalizes_multi_target_binary_classification(tmp_path):
    toolkit = ChempropToolkit(register_tools=False)
    train_csv = tmp_path / "chemprop_input.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN", "COC"],
            "active": [0, 1, 0, 1],
            "toxic": [1, 0, 1, 0],
        }
    ).to_csv(train_csv, index=False)
    output_dir = tmp_path / "chemprop_run"
    (output_dir / "chemprop_inputs").mkdir(parents=True)
    (output_dir / "chemprop_inputs" / "chemprop_input_manifest.json").write_text(
        json.dumps(
            {
                "classification_targets": {
                    "active": {"class_labels": ["inactive", "active"]},
                    "toxic": {"class_labels": ["safe", "toxic"]},
                }
            }
        )
    )
    splits_file = output_dir / "splits.json"
    splits_file.write_text(json.dumps([{"train": [0, 1], "test": [2, 3]}]))
    model_dir = output_dir / "model_0"
    model_dir.mkdir(parents=True)
    (model_dir / "best.pt").write_text("model")
    pd.DataFrame(
        {
            "smiles": ["CCN", "COC"],
            "active": [0.2, 0.8],
            "toxic": [0.7, 0.1],
        }
    ).to_csv(model_dir / "test_predictions.csv", index=False)

    result = toolkit._compute_training_metrics(
        train_csv=str(train_csv),
        output_dir=str(output_dir),
        task=PredictionTaskSpec(
            task_type="classification",
            smiles_columns=["smiles"],
            target_columns=["active", "toxic"],
        ),
        splits_file=str(splits_file),
    )

    predictions = pd.read_csv(result["test_predictions_path"])
    assert predictions["active_prediction"].tolist() == ["inactive", "active"]
    assert predictions["toxic_prediction"].tolist() == ["toxic", "safe"]
    assert result["metrics"]["test"]["balanced_accuracy"] == 1.0
    assert result["target_metrics"]["toxic"]["balanced_accuracy"] == 1.0


def test_training_orchestration_applies_profile_with_backend_specific_limits():
    compute_env = {
        "cpu_count": 48,
        "memory_gb_total": 31.0,
        "gpu_available": True,
        "execution_env": "apptainer_local",
    }

    def defaults(profile: str):
        return {"n_estimators": 1000 if profile == "heavy_validation" else 300}

    def limit(profile: str, args: dict, allow_heavy: bool):
        if profile == "local_light":
            args["n_estimators"] = min(int(args["n_estimators"]), 300)
        return args

    policy = apply_training_profile(
        {"training_profile": "heavy_validation", "n_estimators": 9999},
        defaults_for_profile=defaults,
        limit_profile_args=limit,
        compute_environment=compute_env,
        protected_profiles=("heavy_validation",),
    )

    assert policy["training_profile"] == "heavy_validation"
    assert policy["extra_args"]["n_estimators"] == 9999


def _fake_lightgbm_module():
    calls = {"early_stopping": 0, "fit_kwargs": None}

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params

        def fit(self, X_train, y_train, **kwargs):
            calls["fit_kwargs"] = kwargs
            return self

        def predict(self, features):
            return [0.5 for _ in range(len(features))]

    def early_stopping(*, stopping_rounds, verbose):
        calls["early_stopping"] += 1
        return {
            "callback": "early_stopping",
            "stopping_rounds": stopping_rounds,
            "verbose": verbose,
        }

    def log_evaluation(*, period):
        return {"callback": "log_evaluation", "period": period}

    return (
        SimpleNamespace(
            LGBMRegressor=FakeRegressor,
            early_stopping=early_stopping,
            log_evaluation=log_evaluation,
        ),
        calls,
    )


def test_lightgbm_fit_without_validation_disables_early_stopping(monkeypatch):
    backend = LightGBMBackend()
    fake_lgb, calls = _fake_lightgbm_module()
    monkeypatch.setattr(backend, "_import_lightgbm", lambda: fake_lgb)

    backend._fit_regressor(
        model_params={"n_estimators": 10},
        X_train=pd.DataFrame({"x": [1.0, 2.0]}),
        y_train=pd.Series([1.0, 2.0]),
        X_val=None,
        y_val=None,
        categorical_feature_columns=[],
        early_stopping_rounds=50,
    )

    assert calls["early_stopping"] == 0
    assert "eval_set" not in calls["fit_kwargs"]
    assert calls["fit_kwargs"]["callbacks"] == [{"callback": "log_evaluation", "period": 0}]


def test_lightgbm_fit_with_validation_uses_early_stopping(monkeypatch):
    backend = LightGBMBackend()
    fake_lgb, calls = _fake_lightgbm_module()
    monkeypatch.setattr(backend, "_import_lightgbm", lambda: fake_lgb)

    backend._fit_regressor(
        model_params={"n_estimators": 10},
        X_train=pd.DataFrame({"x": [1.0, 2.0]}),
        y_train=pd.Series([1.0, 2.0]),
        X_val=pd.DataFrame({"x": [3.0]}),
        y_val=pd.Series([3.0]),
        categorical_feature_columns=[],
        early_stopping_rounds=50,
    )

    assert calls["early_stopping"] == 1
    assert "eval_set" in calls["fit_kwargs"]
    assert calls["fit_kwargs"]["callbacks"][1]["callback"] == "early_stopping"


def test_lightgbm_final_refit_preserves_explicit_external_test(monkeypatch, tmp_path):
    """Outlier-filtered refits retain their untouched external test split."""
    backend = LightGBMBackend()
    fake_lgb, _ = _fake_lightgbm_module()
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_import_lightgbm", lambda: fake_lgb)
    monkeypatch.setattr(
        "cs_copilot.tools.prediction.lightgbm_backend.pickle.dump", lambda *_args, **_kwargs: None
    )
    train_csv = tmp_path / "training.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO" for _ in range(12)],
            "feature_a": list(range(12)),
            "pEC50": [float(index) for index in range(12)],
        }
    ).to_csv(train_csv, index=False)

    result = backend.train_model(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "filtered"),
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        extra_args={
            "feature_columns": ["feature_a"],
            "final_refit": True,
            "refit_on_train_validation": True,
            "split_payload": [{"train": list(range(10)), "test": [10, 11]}],
        },
    )

    assert result["metrics_status"] == "evaluated"
    assert result["evaluation_required"] is False
    assert result["test_predictions_path"]
    assert Path(result["test_predictions_path"]).exists()
    assert result["effective_split_payload"][0]["test"] == [10, 11]


def test_lightgbm_final_refit_without_external_test_is_allowed(monkeypatch, tmp_path):
    """A CV final refit may train on all development rows without a test set."""
    backend = LightGBMBackend()
    fake_lgb, _ = _fake_lightgbm_module()
    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_import_lightgbm", lambda: fake_lgb)
    monkeypatch.setattr(
        "cs_copilot.tools.prediction.lightgbm_backend.pickle.dump", lambda *_args, **_kwargs: None
    )
    train_csv = tmp_path / "development.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO" for _ in range(10)],
            "feature_a": list(range(10)),
            "pEC50": [float(index) for index in range(10)],
        }
    ).to_csv(train_csv, index=False)

    result = backend.train_model(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "cv_final_refit"),
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        extra_args={
            "feature_columns": ["feature_a"],
            "final_refit": True,
            "refit_on_train_validation": True,
            "split_payload": [{"train": list(range(10))}],
        },
    )

    assert result["metrics_status"] == "not_evaluated"
    assert result["evaluation_required"] is True
    assert result["test_predictions_path"] is None


def test_lightgbm_predict_featurizes_smiles_for_morgan_model(tmp_path):
    backend = LightGBMBackend()
    model_path = tmp_path / "model.pkl"
    feature_columns = [f"fp_{index:04d}" for index in range(2048)]
    with model_path.open("wb") as fh:
        pickle.dump(
            {
                "model": _FakeLightGBMPredictor(),
                "feature_columns": feature_columns,
                "categorical_feature_columns": [],
                "task_type": "regression",
            },
            fh,
        )
    input_csv = tmp_path / "external.csv"
    pd.DataFrame({"SMILES": ["CCO", "CCC"], "pEC50": [5.0, 6.0]}).to_csv(input_csv, index=False)
    preds_path = tmp_path / "predictions.csv"

    result = backend.predict_from_csv(
        input_csv=str(input_csv),
        model_record=PredictionModelRecord(
            model_id="morgan_model",
            backend_name="lightgbm",
            model_path=str(model_path),
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=["pEC50"],
            ),
            inference_profile={"representation_name": "morgan_only"},
        ),
        preds_path=str(preds_path),
    )

    predictions = pd.read_csv(preds_path)
    assert result["rows"] == 2
    assert predictions["prediction"].tolist() == [0.0, 1.0]


def test_training_orchestration_materializes_summary_and_bundle_inputs(tmp_path):
    run_dir = tmp_path / "run"
    model = run_dir / "model_0" / "best.pkl"
    preds = run_dir / "model_0" / "test_predictions.csv"
    config = run_dir / "config.toml"
    splits = run_dir / "splits.json"
    model.parent.mkdir(parents=True)
    for path in (model, preds, config, splits):
        path.write_text(path.name)

    root = tmp_path / "root"
    artifacts = materialize_primary_protocol_artifacts(
        root_output_dir=root,
        primary_run={
            "model_path": str(model),
            "test_predictions_path": str(preds),
            "config_path": str(config),
            "splits_path": str(splits),
        },
        model_filename="best.pkl",
    )
    summary = write_training_summary(root / "cs_copilot_training_summary.json", {"ok": True})
    files = collect_training_bundle_files(
        train_csv=str(config),
        summary_path=summary,
        result={"model_path": artifacts["best_model_path"]},
        split_results=[],
        ad_summary={},
        plot_artifacts={},
    )

    assert Path(artifacts["best_model_path"]).exists()
    assert json.loads(summary.read_text())["ok"] is True
    assert Path(artifacts["best_model_path"]) in files


def test_model_registry_describe_backends_includes_official_capabilities():
    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)
    toolkit = ModelRegistryToolkit(
        backends=build_default_prediction_backends(),
        catalog=catalog,
        register_tools=False,
    )

    descriptions = toolkit.describe_backends()

    assert descriptions["chemprop"]["capabilities"]["backend_name"] == "chemprop"
    assert descriptions["lightgbm"]["capabilities"]["requires_feature_preparation"] is True
    assert descriptions["ensemble"]["capabilities"]["supports_component_orchestration"] is True


def test_chemprop_toolkit_is_backend_only():
    toolkit = ChempropToolkit()

    assert toolkit.backend.backend_name == "chemprop"
    assert hasattr(toolkit, "validate_chemprop_model_path")
    assert not hasattr(toolkit, "registry_toolkit")
    assert not hasattr(toolkit, "inference_toolkit")
    assert not hasattr(toolkit, "register_model")
    assert not hasattr(toolkit, "persist_registered_model")
    assert not hasattr(toolkit, "predict_from_csv")


def test_chemprop_standard_qsar_forces_single_replicate():
    toolkit = ChempropToolkit(register_tools=False)
    training_policy = {"extra_args": {"num_replicates": 3}}

    note = toolkit._apply_protocol_training_overrides(
        training_policy=training_policy,
        protocol_policy={"protocol": "standard_qsar"},
    )

    assert training_policy["extra_args"]["num_replicates"] == 1
    assert "standard_qsar" in note


def test_chemprop_robust_qsar_forces_single_replicate():
    toolkit = ChempropToolkit(register_tools=False)
    training_policy = {"extra_args": {"num_replicates": 3}}

    note = toolkit._apply_protocol_training_overrides(
        training_policy=training_policy,
        protocol_policy={"protocol": "robust_qsar"},
    )

    assert training_policy["extra_args"]["num_replicates"] == 1
    assert "split runs" in note


def test_chemprop_repeated_holdout_forces_single_replicate():
    toolkit = ChempropToolkit(register_tools=False)
    training_policy = {"extra_args": {"num_replicates": 3}}

    note = toolkit._apply_protocol_training_overrides(
        training_policy=training_policy,
        protocol_policy={"protocol": "repeated_scaffold_holdout"},
    )

    assert training_policy["extra_args"]["num_replicates"] == 1
    assert "repeated_scaffold_holdout" in note


def test_chemprop_normalizes_repeated_session_prefixed_paths(monkeypatch):
    monkeypatch.setattr(
        "cs_copilot.tools.prediction.chemprop_toolkit.S3.current_prefix",
        lambda: "sessions/abc123",
    )

    assert (
        _agent_storage_path(
            ".files/sessions/abc123/.files/sessions/abc123/curated_pxr_challenge_train.csv"
        )
        == "curated_pxr_challenge_train.csv"
    )


def test_chemprop_toolkit_writes_normalized_replicate_predictions(tmp_path):
    toolkit = ChempropToolkit(register_tools=False)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "pEC50": [5.0, 6.0, 4.0],
        }
    ).to_csv(train_csv, index=False)
    output_dir = tmp_path / "chemprop_run"
    output_dir.mkdir()
    (output_dir / "splits.json").write_text(json.dumps([{"train": [0], "val": [], "test": [1, 2]}]))
    for replicate_index, values in enumerate(([5.5, 4.5], [6.5, 3.5])):
        replicate_dir = output_dir / f"replicate_{replicate_index}" / "model_0"
        replicate_dir.mkdir(parents=True)
        (replicate_dir / "best.pt").write_text("model")
        pd.DataFrame({"smiles": ["CCC", "CCN"], "pEC50": values}).to_csv(
            replicate_dir / "test_predictions.csv",
            index=False,
        )

    result = toolkit._write_normalized_test_predictions(
        train_csv=str(train_csv),
        output_dir=output_dir,
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
    )

    normalized = pd.read_csv(result["test_predictions_path"])
    assert result["replicate_count"] == 2
    assert result["detected_replicate_count"] == 2
    assert result["excluded_replicate_count"] == 0
    assert result["prediction_aggregation"] == "mean_aligned_replicates"
    assert normalized["pEC50_true"].tolist() == [6.0, 4.0]
    assert normalized["prediction"].tolist() == [6.0, 4.0]
    assert normalized["pEC50_prediction"].tolist() == [6.0, 4.0]
    assert normalized["pEC50"].tolist() == [6.0, 4.0]
    assert normalized["prediction_std"].tolist() == [0.5, 0.5]


def test_chemprop_toolkit_excludes_unaligned_replicate_predictions(tmp_path):
    toolkit = ChempropToolkit(register_tools=False)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN"],
            "pEC50": [5.0, 6.0, 4.0],
        }
    ).to_csv(train_csv, index=False)
    output_dir = tmp_path / "chemprop_run"
    output_dir.mkdir()
    (output_dir / "splits.json").write_text(json.dumps([{"train": [0], "val": [], "test": [1, 2]}]))
    replicate_payloads = [
        (0, ["CCC", "CCN"], [5.5, 4.5]),
        (1, ["CCN", "CCC"], [6.5, 3.5]),
    ]
    for replicate_index, smiles_values, values in replicate_payloads:
        replicate_dir = output_dir / f"replicate_{replicate_index}" / "model_0"
        replicate_dir.mkdir(parents=True)
        (replicate_dir / "best.pt").write_text("model")
        pd.DataFrame({"smiles": smiles_values, "pEC50": values}).to_csv(
            replicate_dir / "test_predictions.csv",
            index=False,
        )

    result = toolkit._write_normalized_test_predictions(
        train_csv=str(train_csv),
        output_dir=output_dir,
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
    )

    normalized = pd.read_csv(result["test_predictions_path"])
    excluded = [
        item
        for item in result["replicate_artifacts"]
        if item.get("aligned_for_validation") is False
    ]
    assert result["replicate_count"] == 1
    assert result["detected_replicate_count"] == 2
    assert result["excluded_replicate_count"] == 1
    assert result["prediction_aggregation"] == "single_aligned_replicate"
    assert result["raw_test_prediction_paths"] == [
        str(output_dir / "replicate_0" / "model_0" / "test_predictions.csv")
    ]
    assert excluded[0]["replicate_index"] == 1
    assert excluded[0]["exclusion_reason"] == "smiles_not_aligned_to_split_test_rows"
    assert normalized["prediction"].tolist() == [5.5, 4.5]
    assert normalized["prediction_replicate_0"].tolist() == [5.5, 4.5]
    assert "prediction_replicate_1" not in normalized.columns


def test_chemprop_toolkit_writes_validation_predictions_from_checkpoint(tmp_path):
    class FakeChempropBackend:
        backend_name = "chemprop"

        def predict_from_csv(
            self, input_csv, model_record, preds_path, *, return_uncertainty=False
        ):
            frame = pd.read_csv(input_csv)
            pd.DataFrame({"smiles": frame["smiles"], "pEC50": [5.5, 4.5]}).to_csv(
                preds_path,
                index=False,
            )
            return {"preds_path": preds_path}

    toolkit = ChempropToolkit(backend=FakeChempropBackend(), register_tools=False)
    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN", "CCCl"],
            "pEC50": [5.0, 6.0, 4.0, 7.0],
        }
    ).to_csv(train_csv, index=False)
    output_dir = tmp_path / "chemprop_run"
    (output_dir / "model_0").mkdir(parents=True)
    model_path = output_dir / "model_0" / "best.pt"
    model_path.write_text("model")
    splits_file = output_dir / "splits.json"
    splits_file.write_text(json.dumps([{"train": [0], "val": [1, 2], "test": [3]}]))

    result = toolkit._write_validation_predictions(
        train_csv=str(train_csv),
        output_dir=output_dir,
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        splits_file=str(splits_file),
        model_path=str(model_path),
    )

    normalized = pd.read_csv(result["validation_predictions_path"])
    assert result["validation_metrics"]["n"] == 2
    assert normalized["source_row_index"].tolist() == [1, 2]
    assert normalized["pEC50_true"].tolist() == [6.0, 4.0]
    assert normalized["prediction"].tolist() == [5.5, 4.5]


def test_qsar_training_toolkit_routes_lightgbm_through_facade(monkeypatch, tmp_path):
    toolkit = QSARTrainingToolkit()
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("smiles,Y,feature_a\nCCO,1.0,0.1\nCCC,2.0,0.2\n")
    captured = {}

    def fake_train_lightgbm_model(**kwargs):
        captured.update(kwargs)
        return {
            "model_path": str(tmp_path / "best.pkl"),
            "validation_protocol": kwargs.get("validation_protocol"),
            "metrics": {"test": {"r2": 0.5}},
        }

    monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_train_lightgbm_model)

    result = toolkit.train_qsar_model(
        train_csv=str(train_csv),
        backend_name="lightgbm",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        feature_columns=["feature_a"],
        validation_protocol="standard_qsar",
    )

    assert result["backend_name"] == "lightgbm"
    assert captured["train_csv"] == str(train_csv)
    assert captured["feature_columns"] == ["feature_a"]
    assert result["recommended_registry_payload"]["backend_name"] == "lightgbm"


def test_qsar_training_toolkit_normalizes_tabular_smiles_column(tmp_path):
    class FakeMolecularFeatureToolkit:
        def smiles_to_morgan_fingerprints(
            self,
            *,
            input_csv,
            smiles_column,
            output_csv,
            radius,
            n_bits,
            include_input_columns,
            input_columns_to_keep,
            feature_prefix="fp_",
            fingerprint_kind="binary",
            **kwargs,
        ):
            assert smiles_column == "smiles"
            assert fingerprint_kind == "binary"
            source = pd.read_csv(input_csv)
            feature_df = source[input_columns_to_keep].copy()
            feature_df["fp_0000"] = [1, 0]
            feature_df.to_csv(output_csv, index=False)
            return {"output_csv": output_csv, "duration_seconds": 1.25, "num_features": 1}

        def build_tabular_qsar_dataset(
            self,
            *,
            base_csv,
            output_csv,
            feature_csvs,
            join_on,
            base_columns_to_keep,
            drop_duplicate_feature_columns,
            canonicalize_smiles_join=True,
        ):
            assert canonicalize_smiles_join is False
            assert join_on == ["__qsar_row_id"]
            assembled = pd.read_csv(base_csv)[base_columns_to_keep].copy()
            for feature_csv in feature_csvs:
                feature_df = pd.read_csv(feature_csv)
                feature_df = feature_df[
                    [
                        column
                        for column in feature_df.columns
                        if column in join_on or column not in assembled.columns
                    ]
                ]
                assembled = assembled.merge(
                    feature_df,
                    on=join_on,
                    how="left",
                    validate="one_to_one",
                )
            assembled.to_csv(output_csv, index=False)
            return {
                "output_csv": output_csv,
                "duration_seconds": 0.5,
                "num_added_feature_columns": 1,
                "final_column_count": len(assembled.columns),
                "canonicalize_smiles_join": canonicalize_smiles_join,
            }

    train_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "standardized_smiles": ["CCO", "CCC"],
            "Y": [1.0, 2.0],
        }
    ).to_csv(train_csv, index=False)
    toolkit = QSARTrainingToolkit(molecular_feature_toolkit=FakeMolecularFeatureToolkit())

    result = toolkit._prepare_tabular_training_dataset(
        train_csv=str(train_csv),
        output_dir=str(tmp_path / "out"),
        smiles_column="standardized_smiles",
        target_columns=["Y"],
        representation_name="morgan_only",
    )

    output_columns = list(pd.read_csv(result["train_csv"], nrows=0).columns)
    assert "smiles" in output_columns
    assert "standardized_smiles" not in output_columns
    assert result["feature_columns"] == ["fp_0000"]
    assert result["feature_preparation"]["feature_count"] == 1
    morgan_step = next(
        step
        for step in result["feature_preparation_durations"]["steps"]
        if step["step"] == "morgan_binary_fingerprints"
    )
    assert morgan_step["duration_seconds"] == 1.25


def test_prediction_registry_rejects_archive_model_paths_without_backend_validation():
    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            raise AssertionError("archive paths should be rejected before backend validation")

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=catalog,
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})

    result = toolkit.register_model(
        model_id="bad_bundle",
        model_path="/tmp/training_bundle.zip",
        backend_name="lightgbm",
        task_type="regression",
        target_columns=["pEC50"],
        agent=agent,
    )

    assert result["registered"] is False
    assert result["expected_model_extensions"] == [".pkl"]
    assert "bundle/archive" in result["usage_hint"]
    assert get_prediction_state(agent)["registered"] == {}


def test_prediction_registry_register_model_is_session_only(tmp_path):
    model_path = tmp_path / "best.pkl"
    model_path.write_text("model")
    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=catalog,
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})

    result = toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        target_columns=["pEC50"],
        status="workflow_demo",
        agent=agent,
    )

    assert result["registered"] is True
    assert result["persisted"] is False
    assert result["catalog_persisted"] is False
    assert result["persistence_state"] == "session_registered_only"
    assert result["next_required_tool"] == "persist_registered_model"
    assert "session" in result["usage_hint"]


def test_model_registry_persistence_uses_governance_recommended_status(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "pxr_challenge_train.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\n")
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-05-21T12:57:29+02:00",
                "validation_protocol": "standard_qsar",
                "representation_name": "morgan_count_only",
                "feature_columns": [f"feature_{index:04d}" for index in range(64)],
                "validation_assessment": {
                    "governance": {
                        "recommended_status": "workflow_demo",
                        "gates": {
                            "hardest_split_gate": {"pass": False},
                        },
                    },
                },
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    tuning_summary_path = str(run_dir / "hyperparameter_tuning_summary.json")
    tuning_provenance = {
        "engine": "optuna_tpe_multivariate",
        "sampler": {"name": "TPESampler", "multivariate": True, "group": False},
        "best_trial": {"number": 12, "params": {"max_depth": 8, "num_leaves": 64}},
        "parameterization": {"num_leaves": {"mode": "relative_to_depth_capacity"}},
        "summary_path": tuning_summary_path,
    }
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="experimental",
        training_data_summary={
            "hyperparameter_tuning": tuning_provenance,
            "hyperparameter_tuning_summary_path": tuning_summary_path,
        },
        agent=agent,
    )

    result = toolkit.persist_registered_model(
        model_id="session_model",
        status="experimental",
        agent=agent,
    )

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert result["status"] == "workflow_demo"
    assert result["record"]["status"] == "workflow_demo"
    assert "feature_columns" not in result["record"]["inference_profile"]
    assert result["record"]["inference_profile"]["feature_columns_count"] == 64
    assert persisted_metadata["status"] == "workflow_demo"
    assert persisted_metadata["inference_profile"]["feature_columns"] == [
        f"feature_{index:04d}" for index in range(64)
    ]
    assert persisted_metadata["inference_profile"]["representation_name"] == "morgan_count_only"
    assert persisted_metadata["hyperparameter_tuning"] == tuning_provenance
    assert persisted_metadata["hyperparameter_tuning_summary_path"] == tuning_summary_path
    assert persisted_metadata["training_data_summary"]["hyperparameter_tuning"] == tuning_provenance
    assert result["status_reason"]
    assert "workflow_demo" in result["status_reason"]


def test_model_registry_persists_variant_specific_outlier_artifacts(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    campaign_dir = tmp_path / "campaign"
    filtered_dir = campaign_dir / "outlier_filtered"
    filtered_model_dir = filtered_dir / "model_0"
    filtered_model_dir.mkdir(parents=True)
    filtered_model = filtered_model_dir / "best.pkl"
    filtered_model.write_text("filtered-model")
    filtered_predictions = filtered_model_dir / "test_predictions.csv"
    filtered_predictions.write_text("prediction\n2.0\n")
    baseline_predictions = campaign_dir / "model_0" / "test_predictions.csv"
    baseline_predictions.parent.mkdir(parents=True)
    baseline_predictions.write_text("prediction\n1.0\n")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\n")
    campaign_summary = campaign_dir / "cs_copilot_training_summary.json"
    campaign_summary.write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-07-15T12:00:00+02:00",
                "validation_protocol": "random_holdout",
                "representation_name": "rdkit_all",
                "test_predictions_path": str(baseline_predictions),
            }
        )
        + "\n"
    )
    tuning_summary = campaign_dir / "hyperparameter_tuning_summary.json"
    tuning_summary.write_text(json.dumps({"engine": "optuna_tpe", "best_trial": {}}) + "\n")
    analysis_dir = campaign_dir / "outlier_analysis"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True)
    analysis_summary = analysis_dir / "outlier_analysis_summary.json"
    selection_csv = analysis_dir / "outlier_selection_predictions.csv"
    filtered_csv = analysis_dir / "outlier_filtered_development.csv"
    comparison_csv = analysis_dir / "outlier_variant_comparison.csv"
    observed_plot = plots_dir / "outlier_selection_observed_vs_predicted.png"
    residual_plot = plots_dir / "outlier_selection_residuals_vs_observed.png"
    for path in (
        analysis_summary,
        selection_csv,
        filtered_csv,
        comparison_csv,
        observed_plot,
        residual_plot,
    ):
        path.write_text(path.name)
    outlier_analysis = {
        "enabled": True,
        "selected_count": 1,
        "summary_path": str(analysis_summary),
        "selection_predictions_path": str(selection_csv),
        "filtered_development_path": str(filtered_csv),
        "comparison_path": str(comparison_csv),
        "plot_artifacts": {
            "outlier_selection_observed_vs_predicted": str(observed_plot),
            "outlier_selection_residuals_vs_observed": str(residual_plot),
        },
    }

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="filtered_session_model",
        model_path=str(filtered_model),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        known_metrics={"test": {"rmse": 0.3}},
        training_data_summary={
            "validation_protocol": "random_holdout_outlier_filtered",
            "metrics_status": "evaluated",
            "outlier_variant": "outlier_filtered",
            "outlier_analysis": outlier_analysis,
            "hyperparameter_tuning": {"engine": "optuna_tpe"},
            "hyperparameter_tuning_summary_path": str(tuning_summary),
            "artifact_sources": {
                "training_summary_path": str(campaign_summary),
                "test_predictions_path": str(filtered_predictions),
                "hyperparameter_tuning_summary_path": str(tuning_summary),
                "outlier_analysis": outlier_analysis,
            },
        },
        agent=agent,
    )

    persisted = toolkit.persist_registered_model(
        model_id="filtered_session_model",
        agent=agent,
    )

    metadata = json.loads(Path(persisted["metadata_path"]).read_text())
    root = Path(persisted["model_root"])
    assert (
        root / metadata["artifacts"]["test_predictions_path"]
    ).read_text() == "prediction\n2.0\n"
    assert metadata["known_metrics"] == {"test": {"rmse": 0.3}}
    assert metadata["outlier_analysis"]["selected_count"] == 1
    assert (root / metadata["outlier_analysis"]["summary_path"]).exists()
    assert (root / metadata["outlier_analysis"]["comparison_path"]).exists()
    assert (root / metadata["artifacts"]["hyperparameter_tuning_summary_path"]).exists()
    assert (
        metadata["hyperparameter_tuning_summary_path"]
        == metadata["artifacts"]["hyperparameter_tuning_summary_path"]
    )


def test_model_registry_batch_persistence_uses_each_exact_candidate_payload(monkeypatch, tmp_path):
    class FakeBackend:
        backend_name = "lightgbm"

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        default_backend_name="lightgbm",
        register_tools=False,
    )
    calls = []

    def fake_register_model(*, agent, **payload):
        calls.append(("register", payload))
        return {"registered": True, "model_id": payload["model_id"]}

    def fake_persist_registered_model(*, model_id, agent):
        calls.append(("persist", {"model_id": model_id}))
        return {
            "persisted": True,
            "model_id": f"catalog_{model_id}",
            "model_root": str(tmp_path / model_id),
            "model_path": str(tmp_path / model_id / "best.pkl"),
            "metadata_path": str(tmp_path / model_id / "metadata.json"),
        }

    monkeypatch.setattr(toolkit, "register_model", fake_register_model)
    monkeypatch.setattr(toolkit, "persist_registered_model", fake_persist_registered_model)
    agent = SimpleNamespace(session_state={})
    candidates = [
        {
            "rank": 1,
            "candidate_id": "repeat_1_baseline",
            "split_label": "repeat_1_baseline",
            "registry_payload": {
                "model_id": "session_repeat_1_baseline",
                "model_path": str(tmp_path / "repeat_1_baseline.pkl"),
                "task_type": "regression",
                "training_data_summary": {"outlier_variant": "repeat_1_baseline"},
            },
        },
        {
            "rank": 2,
            "candidate_id": "repeat_2_outlier_filtered",
            "split_label": "repeat_2_outlier_filtered",
            "registry_payload": {
                "model_id": "session_repeat_2_outlier_filtered",
                "model_path": str(tmp_path / "repeat_2_outlier_filtered.pkl"),
                "task_type": "regression",
                "training_data_summary": {"outlier_variant": "repeat_2_outlier_filtered"},
            },
        },
    ]

    result = toolkit.register_and_persist_candidates(
        candidate_registry_payloads=candidates,
        agent=agent,
    )

    assert result["candidate_count"] == 2
    assert [name for name, _ in calls] == ["register", "persist", "register", "persist"]
    assert calls[0][1]["training_data_summary"]["outlier_variant"] == "repeat_1_baseline"
    assert calls[2][1]["training_data_summary"]["outlier_variant"] == "repeat_2_outlier_filtered"
    assert [item["model_id"] for item in result["candidates"]] == [
        "catalog_session_repeat_1_baseline",
        "catalog_session_repeat_2_outlier_filtered",
    ]


def test_model_registry_batch_persistence_loads_manifest_and_normalizes_ad_scores(
    monkeypatch, tmp_path
):
    class FakeBackend:
        backend_name = "lightgbm"

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        default_backend_name="lightgbm",
        register_tools=False,
    )
    calls = []

    def fake_register_model(*, agent, **payload):
        calls.append(("register", payload))
        return {"registered": True, "model_id": payload["model_id"]}

    def fake_persist_registered_model(*, model_id, agent):
        calls.append(("persist", {"model_id": model_id}))
        return {
            "persisted": True,
            "model_id": f"catalog_{model_id}",
            "model_root": str(tmp_path / model_id),
            "model_path": str(tmp_path / model_id / "best.pkl"),
            "metadata_path": str(tmp_path / model_id / "metadata.json"),
        }

    monkeypatch.setattr(toolkit, "register_model", fake_register_model)
    monkeypatch.setattr(toolkit, "persist_registered_model", fake_persist_registered_model)
    manifest_path = tmp_path / "catalog_candidates_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "candidate_registry_payloads": [
                    {
                        "rank": 1,
                        "candidate_id": "cv_baseline",
                        "split_label": "cv_baseline",
                        "registry_payload": {
                            "model_id": "session_cv_baseline",
                            "model_path": str(tmp_path / "cv_baseline.pkl"),
                            "task_type": "regression",
                            "split_score_summaries": {"test": {"n_in_domain": 12}},
                        },
                    }
                ],
            }
        )
        + "\n"
    )

    result = toolkit.register_and_persist_candidates(
        candidate_manifest_path=str(manifest_path),
        agent=SimpleNamespace(session_state={}),
    )

    assert result["candidate_count"] == 1
    assert result["candidate_manifest_path"] == str(manifest_path)
    assert [name for name, _ in calls] == ["register", "persist"]
    assert calls[0][1]["applicability_domain"]["split_score_summaries"] == {
        "test": {"n_in_domain": 12}
    }


def test_model_registry_persistence_copies_modern_applicability_domain(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("smiles,pEC50,desc_a\nCCO,5.0,1.0\nCCC,6.0,2.0\n")
    ad_summary = fit_bounding_box_domain(
        feature_frame=pd.DataFrame({"desc_a": [1.0, 2.0]}),
        feature_columns=["desc_a"],
        output_dir=run_dir / "applicability_domain",
        model_id="session_model",
        feature_space="rdkit_all",
        representation_name="rdkit_all",
    )
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-07-07T12:00:00+02:00",
                "validation_protocol": "standard_qsar",
                "representation_name": "rdkit_all",
                "feature_columns": ["desc_a"],
                "applicability_domain": ad_summary,
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="experimental",
        agent=agent,
    )

    result = toolkit.persist_registered_model(
        model_id="session_model",
        applicability_domain={"methods": ["bounding_box"]},
        agent=agent,
    )

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    persisted_ad = persisted_metadata["applicability_domain"]
    assert persisted_ad["primary_method"] == "bounding_box"
    assert persisted_ad["manifest_path"] == "artifacts/applicability_domain/manifest.json"
    assert persisted_ad["bounds_path"] == "artifacts/applicability_domain/bounding_box/bounds.npz"
    manifest_path = Path(result["metadata_path"]).parent / persisted_ad["manifest_path"]
    bounds_path = Path(result["metadata_path"]).parent / persisted_ad["bounds_path"]
    assert manifest_path.exists()
    assert bounds_path.exists()
    persisted_manifest = json.loads(manifest_path.read_text())
    assert persisted_manifest["bounds_path"] == persisted_ad["bounds_path"]
    assert (
        persisted_manifest["methods"]["bounding_box"]["bounds_path"] == persisted_ad["bounds_path"]
    )


def test_model_registry_persistence_does_not_add_bounding_box_to_iforest_only_ad(
    monkeypatch, tmp_path
):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    if_dir = run_dir / "applicability_domain" / "isolation_forest"
    if_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path.write_text("model")
    if_model_path = if_dir / "model.joblib"
    if_model_path.write_text("iforest")
    manifest_path = run_dir / "applicability_domain" / "manifest.json"
    ad_summary = {
        "available": True,
        "primary_method": "isolation_forest",
        "method": "isolation_forest",
        "feature_space": "rdkit_all",
        "manifest_path": str(manifest_path),
        "isolation_forest_model_path": str(if_model_path),
        "methods": {
            "isolation_forest": {
                "model_path": str(if_model_path),
                "feature_names": ["desc_a"],
                "feature_kinds": ["rdkit_descriptor"],
            }
        },
    }
    manifest_path.write_text(json.dumps(ad_summary) + "\n")
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(tmp_path / "train.csv"),
                "trained_at": "2026-07-07T12:00:00+02:00",
                "validation_protocol": "standard_qsar",
                "representation_name": "rdkit_all",
                "feature_columns": ["desc_a"],
                "applicability_domain": ad_summary,
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="experimental",
        agent=agent,
    )

    result = toolkit.persist_registered_model(model_id="session_model", agent=agent)

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    persisted_ad = persisted_metadata["applicability_domain"]
    persisted_manifest = json.loads(
        (Path(result["metadata_path"]).parent / persisted_ad["manifest_path"]).read_text()
    )
    assert persisted_ad["primary_method"] == "isolation_forest"
    assert set(persisted_ad["methods"]) == {"isolation_forest"}
    assert set(persisted_manifest["methods"]) == {"isolation_forest"}
    assert "bounds_path" not in persisted_ad or persisted_ad["bounds_path"] is None


def test_model_registry_persistence_copies_similarity_matrix_ad(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("smiles,pEC50,fp_0000,fp_0001\nCCO,5.0,1,1\nCCC,6.0,1,0\n")
    features = pd.DataFrame({"fp_0000": [1, 1, 0], "fp_0001": [1, 0, 1]})
    ad_summary = fit_modern_applicability_domain(
        feature_frame=features.iloc[[0, 1]].copy(),
        feature_columns=["fp_0000", "fp_0001"],
        output_dir=run_dir / "applicability_domain",
        model_id="session_model",
        feature_space="morgan_only",
        methods=["similarity_matrix"],
        all_feature_frame=features,
        train_indices=[0, 1],
        similarity_top_k_neighbors=1,
    )
    ad_plots_dir = run_dir / "applicability_domain" / "plots" / "test"
    ad_plots_dir.mkdir(parents=True)
    (ad_plots_dir / "ad_method_concordance.png").write_bytes(b"plot")
    ad_summary["plots_dir"] = str(ad_plots_dir.parent)
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-07-07T12:00:00+02:00",
                "validation_protocol": "standard_qsar",
                "representation_name": "morgan_only",
                "feature_columns": ["fp_0000", "fp_0001"],
                "applicability_domain": ad_summary,
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="experimental",
        agent=agent,
    )

    result = toolkit.persist_registered_model(model_id="session_model", agent=agent)

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert persisted_metadata["model_id"] == result["model_id"]
    assert persisted_metadata["backend_name"] == "lightgbm"
    assert persisted_metadata["task"]["target_columns"] == ["pEC50"]
    persisted_ad = persisted_metadata["applicability_domain"]
    similarity = persisted_ad["methods"]["similarity_matrix"]
    assert persisted_ad["primary_method"] == "similarity_matrix"
    assert similarity["manifest_path"] == (
        "artifacts/applicability_domain/similarity_matrix/manifest.json"
    )
    subspace = similarity["subspaces"]["morgan_binary"]
    model_root = Path(result["metadata_path"]).parent
    assert (model_root / subspace["matrix_all_path"]).exists()
    assert (model_root / subspace["reference_features_path"]).exists()
    assert persisted_ad["plots_dir"] == "artifacts/applicability_domain/plots"
    assert (model_root / persisted_ad["plots_dir"] / "test" / "ad_method_concordance.png").exists()


def test_export_prediction_summary_skips_latest_external_evaluation_without_history():
    toolkit = PredictionInferenceToolkit(
        backends={},
        registry_toolkit=SimpleNamespace(),
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    prediction_state = get_prediction_state(agent)
    prediction_state["last_external_evaluation"] = {
        "model_id": "pxr_model",
        "evaluation_id": "test_phase_1",
        "artifacts": {"predictions": "/tmp/predictions.csv"},
    }

    result = toolkit.export_prediction_summary(agent=agent)

    assert result["status"] == "skipped_no_prediction_history"
    assert result["summary_exported"] is False
    assert result["latest_external_evaluation"]["evaluation_id"] == "test_phase_1"


def test_model_registry_persistence_keeps_full_train_as_workflow_demo(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "full_train.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\nCCC,6.0\n")
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-07-07T12:00:00+02:00",
                "validation_protocol": "full_train",
                "validation_strategy_type": "full_train",
                "validation_strategy": {"type": "full_train", "split_sizes": [1.0]},
                "metrics_status": "not_evaluated",
                "evaluation_required": True,
                "metrics": {},
                "known_metrics": {},
                "validation_assessment": {
                    "governance": {
                        "recommended_status": "experimental",
                    },
                },
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="experimental",
        known_metrics={"stale": {"r2": 0.1}},
        agent=agent,
    )

    result = toolkit.persist_registered_model(
        model_id="session_model",
        status="experimental",
        agent=agent,
    )

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert result["status"] == "workflow_demo"
    assert result["record"]["known_metrics"] == {}
    assert persisted_metadata["known_metrics"] == {}
    assert persisted_metadata["external_evaluations"] == []
    assert persisted_metadata["metrics_status"] == "not_evaluated"
    assert persisted_metadata["evaluation_required"] is True
    assert persisted_metadata["training_data_summary"]["metrics_status"] == "not_evaluated"
    assert persisted_metadata["training_data_summary"]["evaluation_required"] is True
    assert persisted_metadata["training_data_summary"]["external_evaluations"] == []
    assert result["status_reason"] is None


def test_model_registry_resolves_persisted_catalog_metadata(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    model_root = internal_root / "persisted_model"
    model_root.mkdir(parents=True)
    (model_root / "best.pkl").write_text("model")
    metadata_path = model_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "model_id": "persisted_model",
                "backend_name": "lightgbm",
                "status": "workflow_demo",
                "task": {
                    "task_type": "regression",
                    "smiles_columns": ["smiles"],
                    "target_columns": ["pEC50"],
                },
                "artifacts": {"model_path": "best.pkl"},
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})

    resolved = toolkit.resolve_record("persisted_model", agent)
    assert resolved.metadata_path == str(metadata_path.resolve())

    result = toolkit.register_catalog_model("persisted_model", agent=agent)
    assert result["metadata_path"] == str(metadata_path.resolve())
    assert get_prediction_state(agent)["registered"]["persisted_model"]["metadata_path"] == str(
        metadata_path.resolve()
    )

    get_prediction_state(agent)["registered"]["persisted_model"]["metadata_path"] = None
    resolved_again = toolkit.resolve_record("persisted_model", agent)
    assert resolved_again.metadata_path == str(metadata_path.resolve())


def test_model_registry_persistence_keeps_split_specific_protocol(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "scaffold_repeat_2_split" / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "pxr_challenge_train.csv"
    train_csv.write_text("smiles,pEC50\nCCO,5.0\n")
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-05-21T12:57:29+02:00",
                "validation_protocol": "repeated_scaffold_holdout",
                "representation_name": "rdkit_all",
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="regression",
        smiles_columns=["smiles"],
        target_columns=["pEC50"],
        status="workflow_demo",
        training_data_summary={
            "validation_protocol": "repeated_scaffold_holdout_scaffold_repeat_2",
            "representation_name": "rdkit_all",
        },
        agent=agent,
    )

    result = toolkit.persist_registered_model(
        model_id="session_model",
        status="workflow_demo",
        agent=agent,
    )

    assert "scaffold_repeat_2" in result["model_id"]
    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert (
        persisted_metadata["training_data_summary"]["validation_protocol"]
        == "repeated_scaffold_holdout_scaffold_repeat_2"
    )


def test_model_registry_persistence_exposes_classification_metadata(monkeypatch, tmp_path):
    internal_root = tmp_path / "internal_models"
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps({"schema_version": 1, "models": []}) + "\n")
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    run_dir = tmp_path / "training_run"
    model_dir = run_dir / "model_0"
    model_dir.mkdir(parents=True)
    model_path = model_dir / "best.pkl"
    model_path.write_text("model")
    train_csv = tmp_path / "ames.csv"
    train_csv.write_text("smiles,Y\nCCO,1\nCCC,0\n")
    (run_dir / "cs_copilot_training_summary.json").write_text(
        json.dumps(
            {
                "train_csv": str(train_csv),
                "trained_at": "2026-07-03T14:11:43+02:00",
                "validation_protocol": "random_holdout",
                "representation_name": "morgan_count_only",
                "task_kind": "binary_classification",
                "class_labels": [0, 1],
                "class_count": 2,
                "label_mapping": {"0": 0, "1": 1},
                "positive_class_label": 1,
                "classification_targets": {
                    "Y": {
                        "class_labels": [0, 1],
                        "class_count": 2,
                        "label_mapping": {"0": 0, "1": 1},
                    }
                },
            }
        )
        + "\n"
    )

    class FakeBackend:
        backend_name = "lightgbm"
        MODEL_EXTENSIONS = (".pkl",)

        def validate_model_path(self, model_path):
            return Path(model_path)

    toolkit = ModelRegistryToolkit(
        backends={"lightgbm": FakeBackend()},
        catalog=PredictionModelCatalog.load(str(catalog_path)),
        default_backend_name="lightgbm",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})
    toolkit.register_model(
        model_id="session_model",
        model_path=str(model_path),
        backend_name="lightgbm",
        task_type="classification",
        smiles_columns=["smiles"],
        target_columns=["Y"],
        status="workflow_demo",
        agent=agent,
    )

    result = toolkit.persist_registered_model(
        model_id="session_model",
        status="workflow_demo",
        agent=agent,
    )

    persisted_metadata = json.loads(Path(result["metadata_path"]).read_text())
    assert persisted_metadata["task_type"] == "classification"
    assert persisted_metadata["task_kind"] == "binary_classification"
    assert persisted_metadata["class_labels"] == [0, 1]
    assert persisted_metadata["class_count"] == 2
    assert persisted_metadata["label_mapping"] == {"0": 0, "1": 1}
    assert persisted_metadata["positive_class_label"] == 1
    assert persisted_metadata["inference_profile"]["class_labels"] == [0, 1]
    assert persisted_metadata["inference_profile"]["classification_targets"]["Y"][
        "class_labels"
    ] == [0, 1]


def test_training_plots_build_classification_artifacts(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text(
        "Y_true,prediction,positive_probability\n" "0,0,0.1\n" "1,1,0.9\n" "1,0,0.4\n" "0,0,0.2\n"
    )
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps([{"train": [0, 1], "test": [0, 1, 2, 3]}]))
    primary_run = {
        "strategy_label": "random_holdout",
        "splits_path": str(splits_path),
        "test_predictions_path": str(predictions_path),
    }

    artifacts = build_training_plots_if_possible(
        train_csv=str(tmp_path / "missing.csv"),
        split_results=[primary_run],
        primary_run=primary_run,
        root_artifacts={
            "splits_path": str(splits_path),
            "test_predictions_path": str(predictions_path),
        },
        root_output_dir=tmp_path,
        target_column="Y",
        task_type="classification",
    )
    assert "confusion_matrix_random" in artifacts
    assert "confusion_matrix_random_normalized" in artifacts
    assert "roc_curve_random" in artifacts
    assert "precision_recall_curve_random" in artifacts
    assert "calibration_curve_random" in artifacts
    assert "positive_probability_distribution_random" in artifacts
    assert not any(key.startswith(("parity_plot", "residuals_plot")) for key in artifacts)
    assert all(Path(path).exists() for path in artifacts.values())


def test_training_plots_build_regression_artifacts_for_tuned_refit(tmp_path):
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("smiles,Y\nCCO,1.0\nCCN,2.0\nCCC,3.0\nCCCl,4.0\n")
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text("Y,prediction\n1.1,1.1\n1.9,1.9\n3.2,3.2\n3.8,3.8\n")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps([{"train": [], "test": [0, 1, 2, 3]}]))
    primary_run = {
        "strategy_label": "tuned_refit",
        "strategy": "tuned_refit",
        "backend_split_type": "random",
        "splits_path": str(splits_path),
        "test_predictions_path": str(predictions_path),
    }

    artifacts = build_training_plots_if_possible(
        train_csv=str(train_csv),
        split_results=[primary_run],
        primary_run=primary_run,
        root_artifacts={
            "splits_path": str(splits_path),
            "test_predictions_path": str(predictions_path),
        },
        root_output_dir=tmp_path,
        target_column="Y",
        task_type="regression",
    )

    assert {
        "target_distribution",
        "target_distribution_by_split",
        "parity_plot_random",
        "parity_plot_random_rmse",
        "residuals_plot_random",
    } <= set(artifacts)
    assert all(Path(path).exists() for path in artifacts.values())


def test_prediction_registry_summarize_model_unknown_id_returns_guidance():
    catalog = SimpleNamespace(refresh_from_internal_store=lambda persist=True: None)
    toolkit = ModelRegistryToolkit(
        backends={},
        catalog=catalog,
        default_backend_name="chemprop",
        register_tools=False,
    )
    agent = SimpleNamespace(session_state={})

    result = toolkit.summarize_model("missing_model", agent=agent)

    assert result["found"] is False
    assert result["model_id"] == "missing_model"
    assert result["registered_model_ids"] == []
    assert "summarize_catalog_model" in result["usage_hint"]


def test_chemprop_backend_describe_environment_shape():
    backend = ChempropBackend()

    env = backend.describe_environment()

    assert env["backend_name"] == "chemprop"
    assert "available" in env
    assert "cli_path" in env
    assert "package_version" in env
    assert env["capabilities"]["backend_name"] == "chemprop"


def test_chemprop_backend_validate_model_path_rejects_missing(tmp_path):
    backend = ChempropBackend()

    missing_path = tmp_path / "missing.ckpt"

    with pytest.raises(InvalidPredictionInputError):
        backend.validate_model_path(str(missing_path))


def test_chemprop_backend_validate_model_path_accepts_ckpt(tmp_path):
    backend = ChempropBackend()
    model_path = tmp_path / "model.ckpt"
    model_path.write_text("placeholder")

    resolved = backend.validate_model_path(str(model_path))

    assert resolved == Path(model_path)


def test_chemprop_backend_fingerprint_from_csv_normalizes_cli_output(monkeypatch, tmp_path):
    backend = ChempropBackend()
    input_csv = tmp_path / "input.csv"
    model_path = tmp_path / "model.pt"
    output_csv = tmp_path / "fingerprints.csv"
    input_csv.write_text("smiles\nCCO\n")
    model_path.write_text("model")

    def fake_run_cli(args, **kwargs):
        pd.DataFrame({"fp_0": [0.1], "fp_1": [0.2]}).to_csv(
            output_csv.with_stem(f"{output_csv.stem}_0"),
            index=False,
        )
        return SimpleNamespace(stdout="ok", stderr="")

    monkeypatch.setattr(backend, "_run_cli", fake_run_cli)

    result = backend.fingerprint_from_csv(
        input_csv=str(input_csv),
        model_path=str(model_path),
        output_csv=str(output_csv),
        smiles_columns=["smiles"],
        ffn_block_index=1,
    )

    frame = pd.read_csv(result["fingerprints_path"])
    assert result["feature_columns"] == ["chemprop_fp_0", "chemprop_fp_1"]
    assert list(frame.columns) == ["chemprop_fp_0", "chemprop_fp_1"]


def test_chemprop_embedding_ad_scores_via_backend_fingerprints(tmp_path):
    ad = fit_bounding_box_domain(
        feature_frame=pd.DataFrame({"chemprop_fp_0": [0.0, 1.0], "chemprop_fp_1": [2.0, 3.0]}),
        feature_columns=["chemprop_fp_0", "chemprop_fp_1"],
        output_dir=tmp_path / "ad",
        model_id="chemprop_model",
        feature_space="chemprop_embedding",
        representation_name="chemprop_embedding",
        feature_metadata={"chemprop_fingerprint": {"ffn_block_index": 1}},
    )
    input_csv = tmp_path / "input.csv"
    model_path = tmp_path / "model.pt"
    input_csv.write_text("smiles\nCCO\nCCC\n")
    model_path.write_text("model")

    class FakeChempropBackend:
        def fingerprint_from_csv(self, **kwargs):
            output_path = Path(kwargs["output_csv"])
            output_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"chemprop_fp_0": [0.5, 2.0], "chemprop_fp_1": [2.5, 2.5]}).to_csv(
                output_path, index=False
            )
            return {"fingerprints_path": str(output_path)}

    record = PredictionModelRecord(
        model_id="chemprop_model",
        backend_name="chemprop",
        model_path=str(model_path),
        task=PredictionTaskSpec(task_type="regression", smiles_columns=["smiles"]),
        applicability_domain=ad,
    )

    result = score_record_applicability_domain(
        record=record,
        input_csv=str(input_csv),
        output_dir=tmp_path / "scores",
        score_label="external",
        backend=FakeChempropBackend(),
    )

    assert list(result["scores"]["ad_status"]) == ["in_domain", "out_of_domain"]


def _tabicl_dataset(tmp_path, values, *, target_column="label") -> Path:
    path = tmp_path / "tabicl_train.csv"
    pd.DataFrame(
        {
            "smiles": [f"C{'C' * (idx % 3)}O" for idx in range(len(values))],
            "feature_a": list(range(len(values))),
            "feature_b": [idx % 5 for idx in range(len(values))],
            target_column: values,
        }
    ).to_csv(path, index=False)
    return path


def _patch_fake_tabicl_estimator(monkeypatch, backend: TabICLBackend, *, classification: bool):
    init_calls = []

    class FakeEstimator:
        def __init__(self, **kwargs):
            init_calls.append(kwargs)
            self.class_count = 0

        def fit(self, X, y):
            if classification:
                self.class_count = len({int(value) for value in y.tolist()})
            return self

        def predict(self, X):
            if not classification:
                return [0.5 for _ in range(len(X))]
            return [idx % self.class_count for idx in range(len(X))]

        def predict_proba(self, X):
            rows = []
            for idx in range(len(X)):
                row = [0.0] * self.class_count
                row[idx % self.class_count] = 1.0
                rows.append(row)
            return rows

        def save(self, path, **_kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake-tabicl-model")

    monkeypatch.setattr(backend, "_ensure_available", lambda: None)
    monkeypatch.setattr(backend, "_import_tabicl_classifier", lambda: FakeEstimator)
    monkeypatch.setattr(backend, "_import_tabicl_regressor", lambda: FakeEstimator)
    return init_calls


def _tabicl_extra_args(tmp_path, checkpoint_name: str) -> dict:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / checkpoint_name).write_text("checkpoint")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "split_payload": [{"train": list(range(9)), "test": [9, 10, 11]}],
        "allow_auto_download": False,
    }


def test_tabicl_binary_classification_uses_classifier_checkpoint(monkeypatch, tmp_path):
    backend = TabICLBackend()
    init_calls = _patch_fake_tabicl_estimator(monkeypatch, backend, classification=True)
    train_csv = _tabicl_dataset(tmp_path, ["inactive", "active"] * 6)
    extra_args = {
        **_tabicl_extra_args(tmp_path, DEFAULT_TABICL_CLASSIFIER_CHECKPOINT),
        "class_shuffle_method": "latin",
        "support_many_classes": False,
    }

    result = backend.train_model(
        str(train_csv),
        str(tmp_path / "tabicl_binary"),
        PredictionTaskSpec(
            task_type="classification",
            smiles_columns=["smiles"],
            target_columns=["label"],
        ),
        extra_args=extra_args,
    )

    assert result["class_count"] == 2
    assert result["task_kind"] == "binary_classification"
    assert init_calls[0]["checkpoint_version"] == DEFAULT_TABICL_CLASSIFIER_CHECKPOINT
    assert init_calls[0]["class_shuffle_method"] == "latin"
    assert init_calls[0]["support_many_classes"] is False


def test_tabicl_multiclass_classification_accepts_three_classes(monkeypatch, tmp_path):
    backend = TabICLBackend()
    init_calls = _patch_fake_tabicl_estimator(monkeypatch, backend, classification=True)
    train_csv = _tabicl_dataset(tmp_path, ["low", "medium", "high"] * 4)

    result = backend.train_model(
        str(train_csv),
        str(tmp_path / "tabicl_multiclass"),
        PredictionTaskSpec(
            task_type="multiclass_classification",
            smiles_columns=["smiles"],
            target_columns=["label"],
        ),
        extra_args=_tabicl_extra_args(tmp_path, DEFAULT_TABICL_CLASSIFIER_CHECKPOINT),
    )

    assert result["class_count"] == 3
    assert result["task_kind"] == "multiclass_classification"
    assert init_calls[0]["checkpoint_version"] == DEFAULT_TABICL_CLASSIFIER_CHECKPOINT


def test_tabicl_classification_rejects_single_class(monkeypatch, tmp_path):
    backend = TabICLBackend()
    _patch_fake_tabicl_estimator(monkeypatch, backend, classification=True)
    train_csv = _tabicl_dataset(tmp_path, ["active"] * 12)

    with pytest.raises(InvalidPredictionInputError, match="at least two classes"):
        backend.train_model(
            str(train_csv),
            str(tmp_path / "tabicl_one_class"),
            PredictionTaskSpec(
                task_type="classification",
                smiles_columns=["smiles"],
                target_columns=["label"],
            ),
            extra_args=_tabicl_extra_args(tmp_path, DEFAULT_TABICL_CLASSIFIER_CHECKPOINT),
        )


def test_tabicl_rejects_multi_target_classification(monkeypatch, tmp_path):
    backend = TabICLBackend()
    _patch_fake_tabicl_estimator(monkeypatch, backend, classification=True)
    train_csv = _tabicl_dataset(tmp_path, ["inactive", "active"] * 6)

    with pytest.raises(InvalidPredictionInputError, match="exactly one target"):
        backend.train_model(
            str(train_csv),
            str(tmp_path / "tabicl_multi_target"),
            PredictionTaskSpec(
                task_type="classification",
                smiles_columns=["smiles"],
                target_columns=["label", "other_label"],
            ),
        )


def test_tabicl_regression_uses_regressor_checkpoint(monkeypatch, tmp_path):
    backend = TabICLBackend()
    init_calls = _patch_fake_tabicl_estimator(monkeypatch, backend, classification=False)
    train_csv = _tabicl_dataset(tmp_path, [float(idx) for idx in range(12)], target_column="Y")

    result = backend.train_model(
        str(train_csv),
        str(tmp_path / "tabicl_regression"),
        PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["Y"],
        ),
        extra_args=_tabicl_extra_args(tmp_path, DEFAULT_TABICL_REGRESSOR_CHECKPOINT),
    )

    assert result["task_kind"] == "regression"
    assert init_calls[0]["checkpoint_version"] == DEFAULT_TABICL_REGRESSOR_CHECKPOINT


def test_prediction_model_record_as_dict():
    record = PredictionModelRecord(
        model_id="solubility_v1",
        backend_name="chemprop",
        model_path="/tmp/model.ckpt",
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["solubility"],
        ),
    )

    payload = record.as_dict()

    assert payload["model_id"] == "solubility_v1"
    assert payload["task"]["task_type"] == "regression"
    assert payload["task"]["target_columns"] == ["solubility"]
