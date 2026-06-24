from __future__ import annotations

from pathlib import Path

from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit


def _fake_train_result(tmp_path: Path, *, backend_name: str, representation_name: str, validation_protocol: str):
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


def test_standard_qsar_tabular_training_runs_modern_representation_campaign(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    called_representations: list[str] = []

    def fake_prepare(**kwargs):
        called_representations.append(kwargs["representation_name"])
        return {
            "train_csv": str(tmp_path / f"{kwargs['representation_name']}.csv"),
            "feature_columns": ["feature_a"],
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

    assert result["campaign_started"] is True
    assert result["representations"] == [
        "rdkit_all",
        "morgan_only",
        "morgan_count_only",
    ]
    assert called_representations == result["representations"]
    assert len(result["recommended_registry_payloads"]) == 3
    assert result["persistence_plan"]["persist_all_candidates"] is True
    assert result["persistence_plan"]["candidate_count"] == 3
    assert len(result["candidate_registry_payloads"]) == 3
    assert all(item["registry_payload"].get("model_id") for item in result["candidate_registry_payloads"])
    assert result["campaign_duration_seconds"] >= 0
    assert "feature_columns" not in result
    assert result["feature_columns_count"] == 64
    assert result["feature_columns_omitted_count"] == 44
    assert "feature_columns" not in result["candidate_results"][0]
    assert result["candidate_results"][0]["feature_columns_count"] == 64
    assert "feature_preparation_duration_seconds" in result["candidate_results"][0]
    assert "training_duration_seconds" in result["candidate_results"][0]
    first_payload_profile = result["recommended_registry_payloads"][0]["inference_profile"]
    assert "feature_columns" not in first_payload_profile
    assert first_payload_profile["feature_columns_count"] == 64


def test_explicit_combined_representation_does_not_start_campaign(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {
            "train_csv": str(tmp_path / "combined.csv"),
            "feature_columns": ["feature_a"],
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


def test_fast_local_tabular_training_uses_rdkit_all_single_candidate(tmp_path, monkeypatch):
    toolkit = QSARTrainingToolkit()
    captured = {}

    def fake_prepare(**kwargs):
        captured.update(kwargs)
        return {
            "train_csv": str(tmp_path / "rdkit_all.csv"),
            "feature_columns": ["feature_a"],
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
    assert "feature_columns" not in result["recommended_registry_payload"]["inference_profile"]
