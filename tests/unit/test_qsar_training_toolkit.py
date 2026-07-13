from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from cs_copilot.storage import S3
from cs_copilot.tools.prediction.qsar_training_toolkit import (
    QSARTrainingToolkit,
    _resolve_existing_training_csv,
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


def test_prepare_training_dataset_can_be_disabled_for_training_agent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with S3.open("pxr_curated.csv", "w") as handle:
        handle.write("smiles,pEC50\nCCO,4.2\n")

    toolkit = QSARTrainingToolkit(block_prepare_training_dataset=True)

    try:
        toolkit.prepare_training_dataset(
            input_csv="pxr_curated.csv",
            smiles_column="smiles",
            target_columns=["pEC50"],
            confirm_explicit_export_request=True,
        )
    except ValueError as exc:
        assert "disabled in the QSAR training workflow" in str(exc)
        assert "train_lightgbm_model" in str(exc)
    else:
        raise AssertionError("agent-scoped prepare_training_dataset should always be blocked")


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
    assert len(result["candidate_registry_payloads"]) == 3
    assert "baseline_split_results" not in result
    assert "feature_csvs" not in result["feature_preparation"]
    assert "input_csv" not in result["feature_preparation"]
    assert [item["split_label"] for item in result["candidate_registry_payloads"]] == [
        "random_repeat_1",
        "random_repeat_2",
        "random_repeat_3",
    ]
    assert (
        len(
            {item["registry_payload"]["model_id"] for item in result["candidate_registry_payloads"]}
        )
        == 3
    )


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


def test_fast_local_tabular_training_uses_rdkit_all_single_candidate(tmp_path, monkeypatch):
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
        validation_protocol="fast_local",
    )

    assert "campaign_started" not in result
    assert captured["representation_name"] == "rdkit_all"
    assert result["representation_name"] == "rdkit_all"
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["recommended_registry_payload"]["model_id"]
    assert "feature_columns" not in result["recommended_registry_payload"]["inference_profile"]
