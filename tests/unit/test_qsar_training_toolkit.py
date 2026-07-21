from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cs_copilot.storage import S3
from cs_copilot.tools.prediction.chemprop_toolkit import ChempropToolkit
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import (
    QSARTrainingToolkit,
    _compact_feature_preparation,
    _compact_registry_payload,
    _compact_training_tool_result,
    _resolve_existing_training_csv,
)
from cs_copilot.tools.prediction.tabicl_toolkit import TabICLToolkit


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
        "train_qsar_model",
        "train_chemprop_model",
        "train_lightgbm_model",
        "train_tabicl_model",
    }


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
        backend_name="lightgbm",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        validation_protocol="standard_qsar",
    )

    assert called_representations == ["rdkit_all"]
    assert result["representation_name"] == "rdkit_all"
    assert "campaign_started" not in result
    assert "candidate_registry_payloads" not in result
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["feature_columns_omitted_count"] == 44
    inference_profile = result["recommended_registry_payload"]["inference_profile"]
    assert "feature_columns" not in inference_profile
    assert inference_profile["feature_columns_count"] == 64


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
        backend_name=backend_name,
        task_type="regression",
        output_dir=str(output_dir),
        target_columns=["Y"],
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
        backend_name="lightgbm",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        validation_protocol="standard_qsar",
        representation_name="morgan_binary_count_rdkit_all",
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
        backend_name="lightgbm",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        validation_strategy={
            "type": "repeated_holdout",
            "split_family": "random",
            "n_repeats": 3,
            "split_sizes": [0.7, 0.15, 0.15],
        },
        representation_name="morgan_only",
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
        backend_name="lightgbm",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        validation_strategy={
            "type": "cross_validation",
            "split_family": "random",
            "n_folds": 3,
            "n_repeats": 1,
        },
        representation_name="morgan_count_only",
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
        backend_name="tabicl",
        task_type="regression",
        output_dir=str(tmp_path / "out"),
        target_columns=["Y"],
        validation_protocol="standard_qsar",
    )

    assert "campaign_started" not in result
    assert captured["representation_name"] == "rdkit_all"
    assert result["representation_name"] == "rdkit_all"
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["recommended_registry_payload"]["model_id"]
    assert "feature_columns" not in result["recommended_registry_payload"]["inference_profile"]
