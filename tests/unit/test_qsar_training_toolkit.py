from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.prediction.backend import InvalidPredictionInputError
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.qsar_contracts import (
    ChempropConfig,
    ComputeConfig,
    CrossValidation,
    GeneratedRepresentation,
    HoldoutValidation,
    LightGBMConfig,
    LightGBMTrainingRequest,
    MolecularGraphRepresentation,
    QsariaTrainingRequest,
    RepeatedHoldoutValidation,
    TabICLConfig,
    TuningConfig,
)
from cs_copilot.tools.prediction.qsar_training_toolkit import (
    QSARTrainingToolkit,
    _compact_feature_preparation,
    _compact_registry_payload,
    _compact_training_tool_result,
    _resolve_existing_training_csv,
)
from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit


@pytest.mark.parametrize(
    ("validation", "protocol", "families"),
    [
        (None, "standard_qsar", ["random", "scaffold", "cluster_kmeans"]),
        ({"kind": "single_random_holdout", "seed": 123}, "single_random_holdout", ["random"]),
    ],
)
def test_train_qsar_model_executes_the_requested_holdout_protocol(
    tmp_path, validation, protocol, families
):
    fixture = Path(__file__).parents[2] / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
    source_csv = tmp_path / "train.csv"
    dataset = pd.read_csv(fixture).head(80).rename(columns={"smiles": "structure"})
    dataset["signal"] = dataset["clogp"] + 0.02
    dataset.to_csv(source_csv, index=False)
    request = QsariaTrainingRequest.model_validate(
        {
            "target_columns": ["clogp"],
            "smiles_column": "structure",
            "task_type": "regression",
            "representation": {"kind": "precomputed", "feature_columns": ["signal"]},
            "outlier_analysis": {"enabled": False},
            "applicability_domain": {"methods": ["bounding_box"]},
            "compute": {"profile": "local_light"},
            "backend": {"name": "lightgbm", "n_estimators": 4, "min_child_samples": 2},
            **({"validation": validation} if validation else {}),
        }
    )

    result = QSARTrainingToolkit().train_qsar_model(str(source_csv), request, str(tmp_path / "out"))

    assert result["validation_protocol"] == protocol
    assert result["resolved_plan"]["validation_protocol"] == protocol
    assert [run["strategy_family"] for run in result["split_results"]] == families
    assert len(result["resolved_plan"]["split_runs"]) == len(families)
    assert set(result["validation_assessment"]["delta_vs_random"]) == set(families) - {"random"}
    assert not result.get("hyperparameter_tuning")


def test_outlier_refit_keeps_the_holdout_provenance_for_agent_training(tmp_path):
    fixture = Path(__file__).parents[2] / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
    source = tmp_path / "fixture.csv"
    data = pd.read_csv(fixture)
    data["signal"] = range(len(data))
    data.to_csv(source, index=False)
    request = {
        "target_columns": ["clogp"],
        "task_type": "regression",
        "representation": {"kind": "precomputed", "feature_columns": ["signal"]},
        "validation": {"kind": "single_random_holdout", "seed": 42},
        "compute": {"profile": "local_light"},
        "backend": {"name": "lightgbm", "n_estimators": 4, "min_child_samples": 2},
        "applicability_domain": {"methods": ["bounding_box"]},
    }
    agent = SimpleNamespace(session_state={})

    result = QSARTrainingToolkit().train_qsar_model(
        str(source), request, str(tmp_path / "output"), agent=agent
    )

    assert result["outlier_analysis"]["status"] == "completed"
    (baseline,) = result["split_results"]
    assert baseline["strategy_label"] == "random_seed_42"
    assert Path(baseline["output_dir"]).name == "baseline"
    assert baseline["leakage"]["validation"]["status"] == "complete"
    assert (
        agent.session_state["prediction_models"]["training_runs"][-1]["split_runs"][0]["seed"] == 42
    )


def test_standard_qsar_requires_enough_scaffolds_for_grouped_holdout(tmp_path):
    source_csv = tmp_path / "train.csv"
    pd.DataFrame(
        {"smiles": ["C" * (i + 1) for i in range(20)], "activity": range(20), "signal": range(20)}
    ).to_csv(source_csv, index=False)
    request = QsariaTrainingRequest.model_validate(
        {
            "target_columns": ["activity"],
            "task_type": "regression",
            "representation": {"kind": "precomputed", "feature_columns": ["signal"]},
            "outlier_analysis": {"enabled": False},
            "applicability_domain": {"methods": ["bounding_box"]},
            "compute": {"profile": "local_light"},
            "backend": {"name": "lightgbm", "n_estimators": 4, "min_child_samples": 2},
        }
    )
    output_dir = tmp_path / "out"

    with pytest.raises(InvalidPredictionInputError, match="scaffold.*single_random_holdout"):
        QSARTrainingToolkit().train_qsar_model(str(source_csv), request, str(output_dir))

    assert not list(output_dir.rglob("best.pkl"))


def test_standard_qsar_checks_grouped_training_class_coverage_before_fitting(tmp_path):
    source_csv = tmp_path / "train.csv"
    smiles = ["C" * (i + 1) + "C1CCCCC1" for i in range(24)]
    smiles += ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1"]
    smiles += ["Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1"]
    pd.DataFrame(
        {"smiles": smiles, "activity": [0] * 27 + [1] * 3, "signal": range(len(smiles))}
    ).to_csv(source_csv, index=False)
    request = QsariaTrainingRequest.model_validate(
        {
            "target_columns": ["activity"],
            "task_type": "classification",
            "representation": {"kind": "precomputed", "feature_columns": ["signal"]},
            "validation": {"kind": "standard_qsar", "seed": 123},
            "outlier_analysis": {"enabled": False},
            "applicability_domain": {"methods": ["bounding_box"]},
            "compute": {"profile": "local_light"},
            "backend": {"name": "lightgbm", "n_estimators": 4, "min_child_samples": 2},
        }
    )
    output_dir = tmp_path / "out"

    with pytest.raises(
        InvalidPredictionInputError, match="scaffold_balanced.*training partition.*activity"
    ):
        QSARTrainingToolkit().train_qsar_model(str(source_csv), request, str(output_dir))

    assert not list(output_dir.rglob("best.pkl"))


@pytest.mark.parametrize(
    "validation",
    [
        {"kind": "single_random_holdout", "seed": 123},
        {"kind": "holdout", "split_family": "random", "seed": 123},
        {"kind": "repeated_holdout", "split_family": "random", "n_repeats": 2, "seed": 123},
        {"kind": "cross_validation", "n_folds": 3, "seed": 123, "final_refit": False},
    ],
)
def test_lightgbm_classification_training_stratifies_explicit_random_validation(
    tmp_path, validation
):
    fixture = Path(__file__).parents[2] / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
    source_csv = tmp_path / "train.csv"
    dataset = pd.read_csv(fixture).head(40)
    dataset["activity"] = [0] * 37 + [1] * 3
    dataset.to_csv(source_csv, index=False)
    request = QsariaTrainingRequest.model_validate(
        {
            "target_columns": ["activity"],
            "task_type": "classification",
            "representation": {"kind": "precomputed", "feature_columns": ["clogp"]},
            "validation": validation,
            "outlier_analysis": {"enabled": False},
            "applicability_domain": {"methods": ["bounding_box"]},
            "compute": {"profile": "local_light"},
            "backend": {"name": "lightgbm", "n_estimators": 4, "min_child_samples": 2},
        }
    )

    result = QSARTrainingToolkit().train_qsar_model(str(source_csv), request, str(tmp_path / "out"))

    for run in result["split_results"]:
        split = json.loads(Path(run["splits_path"]).read_text())[0]
        assert split["metadata"]["stratified"] is True
        for partition in ("train", "val", "test"):
            if partition in split:
                assert set(dataset.iloc[split[partition]]["activity"]) == {0, 1}


def test_explicit_standard_qsar_tuning_compares_all_holdout_families(tmp_path, monkeypatch):
    fixture = Path(__file__).parents[2] / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
    toolkit = LightGBMToolkit()

    def return_fitted_metrics(**_kwargs):
        return {
            "validation_protocol": "standard_qsar",
            "metrics": {"test": {"r2": 0.80, "rmse": 0.30, "n": 20}},
        }

    monkeypatch.setattr(toolkit, "_train_tuned_holdout", return_fitted_metrics)
    result = toolkit.train_lightgbm_model(
        train_csv=str(fixture),
        output_dir=str(tmp_path / "out"),
        task_type="regression",
        target_columns=["clogp"],
        feature_columns=["clogp"],
        random_state=123,
        hyperparameter_tuning={"enabled": True, "n_trials": 1},
        outlier_analysis={"enabled": False},
        resolved_parameters={"training_profile": "local_light"},
    )

    assert [run["strategy_family"] for run in result["split_results"]] == [
        "random",
        "scaffold",
        "cluster_kmeans",
    ]
    assert result["validation_assessment"]["governance"]["recommended_status"] == "validated"


def test_chemprop_standard_qsar_tuning_keeps_each_holdout_summary(tmp_path, monkeypatch):
    fixture = Path(__file__).parents[2] / "benchmarks/fixtures/qsaria-v1-synthetic.csv"
    toolkit = ChempropToolkit()

    def return_native_selection(*, output_dir, train_args, **_kwargs):
        summary_path = output_dir / "hyperparameter_tuning_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "engine": "chemprop_hpopt_hyperopt",
            "summary_path": str(summary_path),
            "best_trial": {"params": {"depth": 2}},
            "split_type": train_args["split_type"],
        }
        summary_path.write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(toolkit, "_run_chemprop_hpopt", return_native_selection)
    monkeypatch.setattr(
        toolkit,
        "_train_single_run",
        lambda **_: {"metrics": {"test": {"r2": 0.80, "rmse": 0.30, "n": 20}}},
    )
    monkeypatch.setattr(toolkit, "_build_applicability_domain", lambda **_: {})
    result = toolkit.train_model(
        train_csv=str(fixture),
        output_dir=str(tmp_path / "out"),
        bundle_path=str(tmp_path / "training.zip"),
        task_type="regression",
        target_columns=["clogp"],
        hyperparameter_tuning={"enabled": True, "n_trials": 1},
        outlier_analysis={"enabled": False},
        resolved_parameters={"training_profile": "local_light", "random_state": 123},
    )

    assert [run["strategy_family"] for run in result["split_results"]] == [
        "random",
        "scaffold",
        "cluster_kmeans",
    ]
    assert result["hyperparameter_tuning"]["split_type"] == "random"
    study_paths = [
        Path(run["hyperparameter_tuning_summary_path"]) for run in result["split_results"]
    ]
    assert len(set(study_paths)) == 3
    assert [json.loads(path.read_text())["split_type"] for path in study_paths] == [
        "random",
        "scaffold_balanced",
        "kmeans",
    ]
    compact = _compact_training_tool_result(result)
    assert [
        Path(run["hyperparameter_tuning_summary_path"]) for run in compact["split_results"]
    ] == (study_paths)


def _training_request(
    backend_name: str,
    *,
    representation_name: str = "rdkit_all",
    validation=None,
) -> QsariaTrainingRequest:
    backend = {
        "chemprop": ChempropConfig(),
        "lightgbm": LightGBMConfig(),
        "tabicl": TabICLConfig(),
    }[backend_name]
    representation = (
        MolecularGraphRepresentation()
        if backend_name == "chemprop"
        else GeneratedRepresentation(name=representation_name)
    )
    return QsariaTrainingRequest(
        target_columns=["Y"],
        task_type="regression",
        representation=representation,
        validation=validation or {"kind": "standard_qsar"},
        tuning=TuningConfig(enabled=backend_name != "tabicl"),
        backend=backend,
    )


def _fake_train_result(
    tmp_path: Path, *, backend_name: str, representation_name: str, validation_protocol: str
):
    model_path = tmp_path / backend_name / representation_name / "model_0" / "best.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("fake-model")
    return {
        "model_path": str(model_path),
        "best_model_path": str(model_path),
        "backend_name": backend_name,
        "representation_name": representation_name,
        "validation_protocol": validation_protocol,
        "candidate_train_csv": str(tmp_path / f"{representation_name}.csv"),
        "feature_columns": [f"feature_{index:04d}" for index in range(64)],
        "feature_preparation": {
            "representation_name": representation_name,
            "feature_cache_key": f"cache-{representation_name}",
            "feature_cache_status": "generated",
            "cache_hits": 0,
            "cache_misses": 2,
            "durations": {"total_duration_seconds": 0.1, "steps": []},
        },
        "feature_preparation_durations": {"total_duration_seconds": 0.1, "steps": []},
        "validation_assessment": {
            "hardest_split": "scaffold",
            "aggregated_split_metrics": {
                "random": {"r2": 0.5},
                "scaffold": {"r2": 0.4},
            },
        },
        "recommended_registry_payload": {
            "backend_name": backend_name,
            "model_path": str(model_path),
            "inference_profile": {"representation_name": representation_name},
        },
    }


def test_qsar_training_toolkit_exposes_shared_tuning_engine_registry():
    toolkit = QSARTrainingToolkit()

    engine = toolkit.describe_tuning_engines("optuna_tpe_multivariate")
    environment = toolkit.describe_qsar_training_environment()

    assert engine["name"] == "optuna_tpe_multivariate"
    assert engine["backend_availability"]["lightgbm"]["status"] == "supported"
    assert "optuna_tpe_multivariate" in environment["tuning_engines"]


def test_feature_preparation_compaction_drops_removed_legacy_marker():
    compact = _compact_feature_preparation(
        {
            "representation_name": "rdkit_all",
            "representation_legacy": False,
            "feature_count": 12,
        }
    )

    assert compact["representation_name"] == "rdkit_all"
    assert compact["feature_count"] == 12
    assert "representation_legacy" not in compact


def test_registry_payload_response_preserves_tuning_provenance(tmp_path):
    toolkit = QSARTrainingToolkit()
    tuning_summary_path = tmp_path / "hyperparameter_tuning_summary.json"
    tuning = {
        "engine": "optuna_tpe_multivariate",
        "sampler": {"name": "TPESampler", "multivariate": True, "group": False},
        "best_trial": {"number": 12, "params": {"max_depth": 8, "num_leaves": 64}},
        "parameterization": {"num_leaves": {"mode": "relative_to_depth_capacity"}},
        "summary_path": str(tuning_summary_path),
    }
    payload = toolkit._recommended_registry_payload(
        backend_name="lightgbm",
        task_type="regression",
        smiles_column="smiles",
        target_columns=["pEC50"],
        result={
            "model_path": str(tmp_path / "model_0" / "best.pkl"),
            "validation_protocol": "standard_qsar",
            "catalog_hyperparameter_tuning": tuning,
            "hyperparameter_tuning_summary_path": str(tuning_summary_path),
        },
    )

    compact = _compact_registry_payload(payload)
    training_summary = compact["training_data_summary"]

    assert training_summary["hyperparameter_tuning"] == tuning
    assert training_summary["hyperparameter_tuning_summary_path"] == str(tuning_summary_path)


def test_outlier_variants_are_exposed_as_independent_catalog_candidates(tmp_path):
    toolkit = QSARTrainingToolkit()
    baseline = tmp_path / "baseline.pkl"
    filtered = tmp_path / "filtered.pkl"
    baseline_test = tmp_path / "baseline_test_predictions.csv"
    filtered_test = tmp_path / "filtered_test_predictions.csv"
    baseline.write_text("baseline")
    filtered.write_text("filtered")
    baseline_test.write_text("prediction\n1.0\n")
    filtered_test.write_text("prediction\n2.0\n")
    result = {
        "representation_name": "rdkit_all",
        "validation_protocol": "standard_qsar",
        "outlier_analysis": {
            "selected_count": 2,
            "test_comparison_policy": "descriptive_only_no_automatic_winner",
        },
        "outlier_model_variants": [
            {
                "variant_id": "baseline",
                "run": {
                    "model_path": str(baseline),
                    "metrics": {"test": {"rmse": 0.4}},
                    "test_predictions_path": str(baseline_test),
                    "applicability_domain": {"available": True},
                },
            },
            {
                "variant_id": "outlier_filtered",
                "run": {
                    "model_path": str(filtered),
                    "metrics": {"test": {"rmse": 0.3}},
                    "test_predictions_path": str(filtered_test),
                    "applicability_domain": {"available": True},
                },
            },
        ],
    }

    payloads = toolkit._split_registry_payloads(
        backend_name="lightgbm",
        task_type="regression",
        smiles_column="smiles",
        target_columns=["Y"],
        result=result,
    )

    assert [item["split_label"] for item in payloads] == ["baseline", "outlier_filtered"]
    assert all(
        item["registry_payload"]["training_data_summary"]["outlier_analysis"]["selected_count"] == 2
        for item in payloads
    )
    assert payloads[0]["registry_payload"]["training_data_summary"]["artifact_sources"][
        "test_predictions_path"
    ] == str(baseline_test)
    assert payloads[1]["registry_payload"]["training_data_summary"]["artifact_sources"][
        "test_predictions_path"
    ] == str(filtered_test)


def test_compacted_candidate_payload_keeps_outlier_variant_provenance(tmp_path):
    model_path = tmp_path / "filtered.pkl"
    model_path.write_text("filtered")
    result = {
        "candidate_registry_payloads": [
            {
                "candidate_id": "repeat_2_outlier_filtered",
                "split_label": "repeat_2_outlier_filtered",
                "registry_payload": {
                    "model_id": "filtered_candidate",
                    "model_path": str(model_path),
                    "task_type": "regression",
                    "training_data_summary": {
                        "validation_strategy": {"type": "repeated_holdout"},
                        "outlier_variant": "repeat_2_outlier_filtered",
                        "outlier_analysis": {"selected_count": 3},
                        "catalog_model_policy": "outlier_variants_no_test_winner",
                        "artifact_sources": {
                            "test_predictions_path": str(tmp_path / "test_predictions.csv")
                        },
                    },
                },
            }
        ]
    }

    compact = _compact_training_tool_result(result)
    summary = compact["candidate_registry_payloads"][0]["registry_payload"]["training_data_summary"]

    assert summary["outlier_variant"] == "repeat_2_outlier_filtered"
    assert summary["outlier_analysis"] == {"selected_count": 3}
    assert summary["validation_strategy"] == {"type": "repeated_holdout"}
    assert summary["artifact_sources"]["test_predictions_path"].endswith("test_predictions.csv")


def test_manifest_backed_training_response_omits_lossless_payloads_and_split_indices(tmp_path):
    registry_payload = {
        "model_id": "outlier_filtered_candidate",
        "model_path": str(tmp_path / "filtered.pkl"),
        "task_type": "regression",
        "training_data_summary": {
            "outlier_variant": "outlier_filtered",
            "artifact_sources": {"applicability_domain": {"very": "large"}},
        },
    }
    result = {
        "candidate_persistence_manifest": {
            "path": str(tmp_path / "catalog_candidates_manifest.json"),
            "schema_version": "1.0",
            "candidate_count": 1,
            "candidates": [
                {
                    "rank": 1,
                    "candidate_id": "outlier_filtered",
                    "model_id": "outlier_filtered_candidate",
                }
            ],
        },
        "candidate_registry_payloads": [{"registry_payload": registry_payload}],
        "recommended_registry_payloads": [registry_payload],
        "outlier_model_variants": [
            {
                "variant_id": "outlier_filtered",
                "run": {
                    "model_path": str(tmp_path / "filtered.pkl"),
                    "metrics": {"test": {"rmse": 0.3}},
                    "effective_split_payload": {"train": list(range(5000))},
                    "split_payload": {"validation": list(range(5000))},
                    "applicability_domain": {
                        "available": True,
                        "manifest_path": str(tmp_path / "ad_manifest.json"),
                        "methods": {"large": {"payload": list(range(5000))}},
                    },
                },
            }
        ],
    }

    compact = _compact_training_tool_result(result)

    assert "candidate_registry_payloads" not in compact
    assert "recommended_registry_payloads" not in compact
    assert compact["candidate_persistence_manifest"]["candidate_count"] == 1
    run = compact["outlier_model_variants"][0]["run"]
    assert run["metrics"]["test"]["rmse"] == 0.3
    assert "effective_split_payload" not in run
    assert "split_payload" not in run
    assert "methods" not in run["applicability_domain"]
    assert len(json.dumps(compact)) < 4_000


def test_prepare_training_dataset_accepts_session_prefixed_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/path-normalization")
    try:
        with S3.open("pxr_curated.csv", "w") as handle:
            handle.write("smiles,pEC50,Emax\nCCO,4.2,1.1\nCCN,5.1,1.4\n")

        toolkit = QSARTrainingToolkit()
        result = toolkit.prepare_training_dataset(
            input_csv=".files/sessions/path-normalization/.files/sessions/path-normalization/pxr_curated.csv",
            smiles_column="smiles",
            target_columns=["pEC50", "Emax"],
            output_csv=".files/sessions/path-normalization/.files/sessions/path-normalization/pxr_training_ready.csv",
            confirm_explicit_export_request=True,
        )

        assert (
            result["output_csv"]
            == ".files/sessions/path-normalization/.files/sessions/path-normalization/pxr_training_ready.csv"
        )
        with S3.open("pxr_training_ready.csv", "r") as handle:
            assert handle.readline().strip() == "smiles,pEC50,Emax"
    finally:
        S3.set_session_prefix(original_prefix)


def test_prepare_training_dataset_is_export_only_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with S3.open("pxr_curated.csv", "w") as handle:
        handle.write("smiles,pEC50\nCCO,4.2\n")

    toolkit = QSARTrainingToolkit()

    try:
        toolkit.prepare_training_dataset(
            input_csv="pxr_curated.csv",
            smiles_column="smiles",
            target_columns=["pEC50"],
        )
    except ValueError as exc:
        assert "train_lightgbm_model" in str(exc)
        assert "confirm_explicit_export_request=True" in str(exc)
    else:
        raise AssertionError(
            "prepare_training_dataset should be blocked unless explicit export is confirmed"
        )


def test_prepare_training_dataset_is_python_only_not_an_agent_tool():
    toolkit = QSARTrainingToolkit()

    assert callable(toolkit.prepare_training_dataset)
    assert "prepare_training_dataset" not in toolkit.functions


def test_backend_toolkits_are_internal_and_facade_is_the_only_agent_surface():
    assert ChempropToolkit().functions == {}
    assert LightGBMToolkit().functions == {}
    assert TabICLToolkit().functions == {}

    assert set(QSARTrainingToolkit().functions) == {
        "describe_qsar_training_environment",
        "describe_backend_hyperparameters",
        "describe_tuning_engines",
        "describe_outlier_analysis",
        "train_standard_qsar_model",
        "train_qsar_model",
        "train_chemprop_model",
        "train_lightgbm_model",
        "train_tabicl_model",
    }


def test_agno_training_surface_uses_closed_typed_requests_and_hides_runtime_paths():
    toolkit = QSARTrainingToolkit()

    standard_schema = toolkit.functions["train_standard_qsar_model"].parameters
    assert standard_schema["additionalProperties"] is False
    assert set(standard_schema["properties"]) == {
        "train_csv",
        "backend_name",
        "target_columns",
        "task_type",
        "smiles_column",
    }
    assert "request" not in standard_schema["properties"]
    assert standard_schema["properties"]["backend_name"]["type"] == "string"
    assert standard_schema["properties"]["task_type"]["type"] == "string"

    for name in (
        "train_qsar_model",
        "train_chemprop_model",
        "train_lightgbm_model",
        "train_tabicl_model",
    ):
        schema = toolkit.functions[name].parameters
        assert schema["additionalProperties"] is False
        assert set(schema["properties"]) == {"train_csv", "request"}
        serialized = json.dumps(schema)
        for forbidden in (
            "extra_args",
            "output_dir",
            "bundle_dir",
            "feature_cache_dir",
            "heartbeat_path",
            "split_payload",
        ):
            assert forbidden not in serialized


def test_agno_standard_lightgbm_surface_builds_the_canonical_request(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    captured = {}

    def fake_train_qsar_model(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(toolkit, "train_qsar_model", fake_train_qsar_model)
    monkeypatch.setattr(
        toolkit,
        "_managed_agno_paths",
        lambda backend_name: (str(tmp_path / backend_name), str(tmp_path / "bundles")),
    )

    result = toolkit._agno_train_standard_qsar_model(
        train_csv=str(tmp_path / "curated.csv"),
        backend_name="lightgbm",
        target_columns=["pEC50"],
        task_type="regression",
    )

    request = captured["request"]
    assert result == {"status": "ok"}
    assert request.validation.kind == "standard_qsar"
    assert request.representation.kind == "generated"
    assert request.representation.name == "rdkit_all"
    assert request.tuning.enabled is False
    assert request.outlier_analysis.enabled is True


def test_agno_training_guard_allows_one_same_intent_retry_then_stops(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    agent = SimpleNamespace(session_state={"current_run_id": "run-1"})
    calls = []

    def failing_train(**kwargs):
        calls.append(kwargs["request"])
        raise RuntimeError("backend failed")

    monkeypatch.setattr(toolkit, "train_qsar_model", failing_train)
    request = _training_request("lightgbm")

    for _ in range(2):
        with pytest.raises(RuntimeError, match="backend failed"):
            toolkit._run_guarded_agno_training(
                train_csv=str(tmp_path / "curated.csv"),
                request=request,
                output_dir=str(tmp_path / "out"),
                bundle_dir=str(tmp_path / "bundles"),
                agent=agent,
            )

    with pytest.raises(RuntimeError, match="single allowed same-intent retry"):
        toolkit._run_guarded_agno_training(
            train_csv=str(tmp_path / "curated.csv"),
            request=request,
            output_dir=str(tmp_path / "out-3"),
            bundle_dir=str(tmp_path / "bundles"),
            agent=agent,
        )
    assert len(calls) == 2


def test_agno_training_guard_rejects_changed_science_after_failure(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    agent = SimpleNamespace(session_state={"current_run_id": "run-1"})

    def failing_train(**kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr(toolkit, "train_qsar_model", failing_train)
    with pytest.raises(RuntimeError, match="backend failed"):
        toolkit._run_guarded_agno_training(
            train_csv=str(tmp_path / "curated.csv"),
            request=_training_request("lightgbm"),
            output_dir=str(tmp_path / "out"),
            bundle_dir=str(tmp_path / "bundles"),
            agent=agent,
        )

    changed_request = _training_request(
        "lightgbm",
        validation=CrossValidation(n_folds=5),
    )
    with pytest.raises(RuntimeError, match="blocked a changed training request"):
        toolkit._run_guarded_agno_training(
            train_csv=str(tmp_path / "curated.csv"),
            request=changed_request,
            output_dir=str(tmp_path / "out-2"),
            bundle_dir=str(tmp_path / "bundles"),
            agent=agent,
        )


def test_training_csv_resolution_falls_back_to_latest_curation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    original_prefix = S3.current_prefix()
    S3.set_session_prefix("sessions/path-resolution")
    try:
        with S3.open("curated/pxr_curated.csv", "w") as handle:
            handle.write("smiles,pEC50\nCCO,4.2\n")
        agent = SimpleNamespace(
            session_state={
                "qsar_curation": {
                    "last_result": {
                        "curated_dataset_path": "curated/pxr_curated.csv",
                    }
                }
            }
        )

        resolved = _resolve_existing_training_csv("uploads/pxr_curated.csv", agent)

        assert resolved == "curated/pxr_curated.csv"
    finally:
        S3.set_session_prefix(original_prefix)


def _fake_repeated_train_result(tmp_path: Path, *, backend_name: str, representation_name: str):
    split_results = []
    for index, seed in enumerate([101, 202, 303], start=1):
        model_path = (
            tmp_path
            / backend_name
            / representation_name
            / f"random_repeat_{index}_split"
            / "model_0"
            / "best.pkl"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake-model")
        split_results.append(
            {
                "model_path": str(model_path),
                "best_model_path": str(model_path),
                "metrics": {"test": {"r2": 0.5 + index / 100}},
                "strategy_label": f"random_repeat_{index}",
                "strategy": "random",
                "strategy_family": "random",
                "seed": seed,
            }
        )
    result = dict(split_results[0])
    result.update(
        {
            "backend_name": backend_name,
            "representation_name": representation_name,
            "validation_protocol": "repeated_random_holdout",
            "split_results": split_results,
            "baseline_split_results": split_results,
        }
    )
    return result


def _fake_cv_train_result(tmp_path: Path, *, backend_name: str, representation_name: str):
    final_model_path = (
        tmp_path / backend_name / representation_name / "final_refit" / "model_0" / "best.pkl"
    )
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    final_model_path.write_text("fake-final-model")
    split_results = []
    for fold in range(1, 4):
        model_path = (
            tmp_path
            / backend_name
            / representation_name
            / f"cv_repeat_1_fold_{fold}"
            / "model_0"
            / "best.pkl"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("fake-fold-model")
        split_results.append(
            {
                "model_path": str(model_path),
                "best_model_path": str(model_path),
                "metrics": {"test": {"r2": 0.5 + fold / 100}},
                "strategy_label": f"cv_repeat_1_fold_{fold}",
                "strategy": f"cv_repeat_1_fold_{fold}",
                "strategy_family": "cross_validation",
                "seed": 123,
            }
        )
    return {
        "model_path": str(final_model_path),
        "best_model_path": str(final_model_path),
        "backend_name": backend_name,
        "representation_name": representation_name,
        "validation_protocol": "cross_validation",
        "validation_strategy_type": "cross_validation",
        "split_results": split_results,
        "cross_validation": {
            "summary": {
                "n_repeats": 1,
                "n_folds": 3,
                "metrics": {"rmse": {"mean": 0.7, "std": 0.01}},
            }
        },
        "final_refit_result": {
            "model_path": str(final_model_path),
            "best_model_path": str(final_model_path),
            "strategy_label": "final_refit",
        },
    }


def test_standard_qsar_tabular_training_uses_one_rdkit_representation(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    called_representations: list[str] = []
    captured_categorical_columns = None

    def fake_prepare(**kwargs):
        called_representations.append(kwargs["representation_name"])
        return {
            "train_csv": str(tmp_path / f"{kwargs['representation_name']}.csv"),
            "feature_columns": [f"feature_{index:04d}" for index in range(64)],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "feature_cache_key": f"cache-{kwargs['representation_name']}",
                "feature_cache_status": "generated",
                "cache_hits": 0,
                "cache_misses": 2,
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        }

    def fake_lightgbm_train(**kwargs):
        nonlocal captured_categorical_columns
        captured_categorical_columns = kwargs["categorical_feature_columns"]
        return _fake_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name=called_representations[-1],
            validation_protocol=kwargs["validation_protocol"],
        )

    monkeypatch.setattr(toolkit, "_prepare_tabular_training_dataset", fake_prepare)
    monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_lightgbm_train)

    result = toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request("lightgbm"),
        output_dir=str(tmp_path / "out"),
    )

    assert called_representations == ["rdkit_all"]
    assert captured_categorical_columns == []
    assert result["representation_name"] == "rdkit_all"
    assert "campaign_started" not in result
    assert "candidate_registry_payloads" not in result
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["feature_columns_omitted_count"] == 44
    inference_profile = result["recommended_registry_payload"]["inference_profile"]
    assert "feature_columns" not in inference_profile
    assert inference_profile["feature_columns_count"] == 64


def test_local_light_uses_one_worker_for_generated_features(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    workers = []

    def prepare(**kwargs):
        workers.append(kwargs["feature_n_jobs"])
        return {
            "train_csv": str(tmp_path / "features.csv"),
            "feature_columns": ["x"],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        }

    monkeypatch.setattr(toolkit, "_prepare_tabular_training_dataset", prepare)
    monkeypatch.setattr(
        "cs_copilot.tools.prediction.qsar_training_toolkit.describe_compute_environment",
        lambda: {"cpu_count": 8},
    )
    monkeypatch.setattr(
        toolkit.lightgbm_toolkit,
        "train_lightgbm_model",
        lambda **kwargs: _fake_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name="rdkit_all",
            validation_protocol=kwargs["validation_protocol"],
        ),
    )
    request = _training_request("lightgbm").model_copy(
        update={"compute": ComputeConfig(profile="local_light")}
    )

    toolkit.train_qsar_model(str(tmp_path / "train.csv"), request, str(tmp_path / "out"))

    assert workers == [1]


@pytest.mark.parametrize(
    "validation",
    [
        HoldoutValidation(split_family="random"),
        HoldoutValidation(split_family="scaffold"),
        RepeatedHoldoutValidation(split_family="random", n_repeats=2),
        CrossValidation(n_folds=3),
    ],
)
def test_generated_lightgbm_never_passes_none_categorical_columns(
    tmp_path, monkeypatch, validation
):
    toolkit = QSARTrainingToolkit()
    captured = {}
    monkeypatch.setattr(
        toolkit,
        "_prepare_tabular_training_dataset",
        lambda **kwargs: {
            "train_csv": str(tmp_path / "generated.csv"),
            "feature_columns": ["feature_a"],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        },
    )

    def fake_train(**kwargs):
        captured.update(kwargs)
        return _fake_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name="rdkit_all",
            validation_protocol=kwargs["validation_protocol"],
        )

    monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_train)

    toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request("lightgbm", validation=validation),
        output_dir=str(tmp_path / "out"),
    )

    assert captured["categorical_feature_columns"] == []


@pytest.mark.parametrize(
    ("backend_kwargs", "expected_explicit_tuning_parameters"),
    [
        ({}, set()),
        ({"n_estimators": 750, "learning_rate": 0.03}, {"n_estimators", "learning_rate"}),
    ],
)
def test_agno_lightgbm_preserves_only_explicit_backend_parameters(
    tmp_path,
    monkeypatch,
    backend_kwargs,
    expected_explicit_tuning_parameters,
):
    toolkit = QSARTrainingToolkit()
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        toolkit,
        "_managed_agno_paths",
        lambda _backend_name: (str(tmp_path / "out"), str(tmp_path / "bundles")),
    )
    monkeypatch.setattr(
        toolkit,
        "_prepare_tabular_training_dataset",
        lambda **kwargs: {
            "train_csv": str(tmp_path / "generated.csv"),
            "feature_columns": ["feature_a"],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        },
    )

    def fake_train(**kwargs):
        captured.update(kwargs)
        return _fake_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name="rdkit_all",
            validation_protocol=kwargs["validation_protocol"],
        )

    monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_train)
    request_kwargs = {
        "target_columns": ["pEC50"],
        "task_type": "regression",
    }
    if backend_kwargs:
        request_kwargs["backend"] = LightGBMConfig(**backend_kwargs)

    toolkit._agno_train_lightgbm_model(
        train_csv=str(tmp_path / "train.csv"),
        request=LightGBMTrainingRequest(**request_kwargs),
    )

    tuning_parameters = {"n_estimators", "learning_rate", "max_depth", "num_leaves"}
    resolved_parameters = captured["resolved_parameters"]
    assert isinstance(resolved_parameters, dict)
    assert tuning_parameters.intersection(resolved_parameters) == (
        expected_explicit_tuning_parameters
    )


@pytest.mark.parametrize("backend_name", ["chemprop", "lightgbm"])
def test_training_facade_forwards_explicit_bundle_destination(tmp_path, monkeypatch, backend_name):
    toolkit = QSARTrainingToolkit()
    captured: dict[str, object] = {}

    if backend_name == "lightgbm":
        monkeypatch.setattr(
            toolkit,
            "_prepare_tabular_training_dataset",
            lambda **kwargs: {
                "train_csv": str(tmp_path / "rdkit.csv"),
                "feature_columns": ["feature_a"],
                "feature_preparation": {
                    "representation_name": kwargs["representation_name"],
                    "durations": {"total_duration_seconds": 0.1, "steps": []},
                },
            },
        )

        def fake_train(**kwargs):
            captured.update(kwargs)
            return _fake_train_result(
                tmp_path,
                backend_name="lightgbm",
                representation_name="rdkit_all",
                validation_protocol=kwargs["validation_protocol"],
            )

        monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_train)
    else:

        def fake_train(**kwargs):
            captured.update(kwargs)
            return _fake_train_result(
                tmp_path,
                backend_name="chemprop",
                representation_name="molecular_graph",
                validation_protocol="standard_qsar",
            )

        monkeypatch.setattr(toolkit.chemprop_toolkit, "train_model", fake_train)

    output_dir = tmp_path / "out"
    bundle_dir = tmp_path / "bundles"
    toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request(backend_name),
        output_dir=str(output_dir),
        bundle_dir=str(bundle_dir),
    )

    assert captured["bundle_path"] == str((bundle_dir / "out_training_bundle.zip").resolve())


def test_explicit_combined_representation_does_not_start_campaign(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {
            "train_csv": str(tmp_path / "combined.csv"),
            "feature_columns": [f"feature_{index:04d}" for index in range(64)],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        }

    def fake_lightgbm_train(**kwargs):
        return _fake_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name=captured["representation_name"],
            validation_protocol=kwargs["validation_protocol"],
        )

    monkeypatch.setattr(toolkit, "_prepare_tabular_training_dataset", fake_prepare)
    monkeypatch.setattr(toolkit.lightgbm_toolkit, "train_lightgbm_model", fake_lightgbm_train)

    result = toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request(
            "lightgbm",
            representation_name="morgan_binary_count_rdkit_all",
        ),
        output_dir=str(tmp_path / "out"),
    )

    assert "campaign_started" not in result
    assert captured["representation_name"] == "morgan_binary_count_rdkit_all"
    assert result["representation_name"] == "morgan_binary_count_rdkit_all"
    assert result["recommended_registry_payload"]["model_id"]


def test_repeated_holdout_single_representation_returns_registry_payload_for_each_split(
    tmp_path, monkeypatch
):
    toolkit = QSARTrainingToolkit()

    monkeypatch.setattr(
        toolkit,
        "_prepare_tabular_training_dataset",
        lambda **kwargs: {
            "train_csv": str(tmp_path / "morgan.csv"),
            "feature_columns": ["feature_a", "feature_b"],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "input_csv": str(tmp_path / "train.csv"),
                "feature_csvs": [str(tmp_path / "morgan_features.csv")],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        },
    )
    monkeypatch.setattr(
        toolkit.lightgbm_toolkit,
        "train_lightgbm_model",
        lambda **kwargs: _fake_repeated_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name="morgan_only",
        ),
    )

    result = toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request(
            "lightgbm",
            representation_name="morgan_only",
            validation=RepeatedHoldoutValidation(
                split_family="random",
                n_repeats=3,
                split_sizes=[0.7, 0.15, 0.15],
            ),
        ),
        output_dir=str(tmp_path / "out"),
    )

    assert result["persistence_plan"]["persist_all_candidates"] is True
    assert result["persistence_plan"]["candidate_count"] == 3
    assert "candidate_registry_payloads" not in result
    manifest = json.loads(Path(result["candidate_persistence_manifest"]["path"]).read_text())
    candidates = manifest["candidate_registry_payloads"]
    assert result["candidate_manifest_path"] == result["candidate_persistence_manifest"]["path"]
    assert (
        result["persistence_plan"]["candidate_manifest_path"]
        == result["candidate_persistence_manifest"]["path"]
    )
    assert len(candidates) == 3
    assert "baseline_split_results" not in result
    assert "feature_csvs" not in result["feature_preparation"]
    assert "input_csv" not in result["feature_preparation"]
    assert [item["split_label"] for item in candidates] == [
        "random_repeat_1",
        "random_repeat_2",
        "random_repeat_3",
    ]
    assert len({item["registry_payload"]["model_id"] for item in candidates}) == 3


def test_cross_validation_single_representation_catalogs_only_final_refit(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()

    monkeypatch.setattr(
        toolkit,
        "_prepare_tabular_training_dataset",
        lambda **kwargs: {
            "train_csv": str(tmp_path / "morgan_count.csv"),
            "feature_columns": ["feature_a", "feature_b"],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        },
    )
    monkeypatch.setattr(
        toolkit.lightgbm_toolkit,
        "train_lightgbm_model",
        lambda **kwargs: _fake_cv_train_result(
            tmp_path,
            backend_name="lightgbm",
            representation_name="morgan_count_only",
        ),
    )

    result = toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request(
            "lightgbm",
            representation_name="morgan_count_only",
            validation=CrossValidation(n_folds=3, n_repeats=1),
        ),
        output_dir=str(tmp_path / "out"),
    )

    assert "candidate_registry_payloads" not in result
    assert "persistence_plan" not in result
    assert result["recommended_registry_payload"]["model_path"].endswith(
        "final_refit/model_0/best.pkl"
    )
    assert "cross_validation" in result["recommended_registry_payload"]["known_metrics"]


def test_standard_qsar_tabular_training_uses_rdkit_all_single_candidate(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {
            "train_csv": str(tmp_path / "rdkit_all.csv"),
            "feature_columns": [f"feature_{index:04d}" for index in range(64)],
            "feature_preparation": {
                "representation_name": kwargs["representation_name"],
                "durations": {"total_duration_seconds": 0.1, "steps": []},
            },
        }

    def fake_tabicl_train(**kwargs):
        return _fake_train_result(
            tmp_path,
            backend_name="tabicl",
            representation_name=captured["representation_name"],
            validation_protocol=kwargs["validation_protocol"],
        )

    monkeypatch.setattr(toolkit, "_prepare_tabular_training_dataset", fake_prepare)
    monkeypatch.setattr(toolkit.tabicl_toolkit, "train_tabicl_model", fake_tabicl_train)

    result = toolkit.train_qsar_model(
        train_csv=str(tmp_path / "train.csv"),
        request=_training_request("tabicl"),
        output_dir=str(tmp_path / "out"),
    )

    assert "campaign_started" not in result
    assert captured["representation_name"] == "rdkit_all"
    assert result["representation_name"] == "rdkit_all"
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["recommended_registry_payload"]["model_id"]
    assert "feature_columns" not in result["recommended_registry_payload"]["inference_profile"]
