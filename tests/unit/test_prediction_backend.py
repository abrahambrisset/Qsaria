import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

import cs_copilot.tools.prediction.catalog as catalog_module
import cs_copilot.tools.prediction.model_registry_toolkit as registry_module
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
from cs_copilot.tools.prediction.chemprop_backend import ChempropBackend
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.model_registry_toolkit import ModelRegistryToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit
from cs_copilot.tools.prediction.session_state import (
    bundle_artifacts,
    discover_curation_artifacts_near_dataset,
    get_prediction_state,
    latest_curation_artifacts,
    write_active_training_marker,
)
from cs_copilot.tools.prediction.training_orchestration import (
    apply_training_profile,
    collect_training_bundle_files,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    write_training_summary,
)


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
    assert "classification" in lightgbm.supported_task_types
    assert chemprop.supports_activity_cliff_feedback_loops is False
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
                assembled = assembled.merge(
                    pd.read_csv(feature_csv),
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
    assert result["status_reason"]
    assert "workflow_demo" in result["status_reason"]


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
