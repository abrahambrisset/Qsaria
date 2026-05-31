from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.benchmark_toolkit import BenchmarkToolkit, _coerce_list
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.model_registry_toolkit import ModelRegistryToolkit
from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit


def _fake_agent() -> SimpleNamespace:
    return SimpleNamespace(
        session_state={
            "prediction_models": {
                "registered": {},
                "last_prediction": {},
                "prediction_history": [],
                "catalog_recommendations": {},
                "training_runs": [],
                "active_training_run": None,
            }
        }
    )


def _write_fake_training_outputs(
    root: Path,
    *,
    model_name: str,
    model_suffix: str,
) -> tuple[Path, Path, Path, Path]:
    model_dir = root / "model_0"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / model_name
    model_path.write_text("fake-model")

    test_predictions_path = model_dir / "test_predictions.csv"
    pd.DataFrame({"y_true": [1.0, 2.0], "y_pred": [1.1, 1.9]}).to_csv(
        test_predictions_path, index=False
    )

    config_path = root / "config.toml"
    config_path.write_text('backend_name = "fake"\n')

    splits_path = root / "splits.json"
    splits_path.write_text(json.dumps([{"train": [0], "val": [1], "test": [0, 1]}]))

    return model_path, test_predictions_path, config_path, splits_path


def test_tool_list_argument_coercion_accepts_scalar_strings():
    assert _coerce_list("pEC50") == ["pEC50"]
    assert _coerce_list("lightgbm") == ["lightgbm"]
    assert _coerce_list('["pEC50"]') == ["pEC50"]
    assert _coerce_list("pEC50,Emax") == ["pEC50", "Emax"]

    toolkit = LightGBMToolkit()
    assert toolkit._normalize_json_list_argument("pEC50", argument_name="target_columns") == [
        "pEC50"
    ]
    assert toolkit._normalize_json_list_argument(
        '["pEC50"]',
        argument_name="target_columns",
    ) == ["pEC50"]
    assert toolkit._normalize_json_list_argument(
        "0.8,0.1,0.1",
        argument_name="split_sizes",
    ) == [0.8, 0.1, 0.1]


def _fake_validation_assessment(
    protocol: str, random_r2: float, hardest_name: str, hardest_r2: float
) -> dict:
    aggregated = {
        "random": {"r2_mean": random_r2, "r2_std": 0.01 if protocol == "robust_qsar" else 0.0},
        hardest_name: {"r2_mean": hardest_r2},
    }
    if protocol == "robust_qsar":
        aggregated["random"]["num_runs"] = 3
    return {
        "hardest_split": hardest_name,
        "delta_vs_random": {hardest_name: {"r2": hardest_r2 - random_r2}},
        "aggregated_split_metrics": aggregated,
        "governance": {"recommended_status": "workflow_demo"},
    }


def test_resolve_benchmark_protocol_contract():
    toolkit = BenchmarkToolkit()
    assert toolkit._resolve_benchmark_protocol("benchmark_fast_local") == "fast_local"
    assert toolkit._resolve_benchmark_protocol("benchmark_standard_qsar") == "standard_qsar"
    assert toolkit._resolve_benchmark_protocol("benchmark_robust_qsar") == "robust_qsar"
    assert toolkit._resolve_benchmark_protocol("benchmark_challenging_qsar") == "challenging_qsar"
    assert toolkit._resolve_benchmark_protocol("fast_local") == "fast_local"
    assert toolkit._resolve_benchmark_protocol("standard_qsar") == "standard_qsar"
    assert toolkit._resolve_benchmark_protocol("robust_qsar") == "robust_qsar"
    assert toolkit._resolve_benchmark_protocol("challenging_qsar") == "challenging_qsar"


def test_expand_candidates_includes_heavy_tabicl_all():
    toolkit = BenchmarkToolkit()
    candidates = toolkit._expand_candidates(
        task_type="regression",
        target_columns=["Y"],
        requested_backends=["tabicl"],
        include_candidate_variants=True,
        requested_tabicl_variants=None,
        training_profile="heavy_validation",
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    assert "tabicl_rdkit_all" in candidate_ids
    assert "tabicl_morgan_only" in candidate_ids
    assert "tabicl_morgan_count_only" in candidate_ids
    assert "tabicl_morgan_binary_count_rdkit_all" in candidate_ids
    assert "tabicl_rdkit_basic_only" not in candidate_ids
    assert "tabicl_morgan_rdkit_basic" not in candidate_ids


def test_expand_candidates_includes_heavy_lightgbm_all():
    toolkit = BenchmarkToolkit()
    candidates = toolkit._expand_candidates(
        task_type="regression",
        target_columns=["Y"],
        requested_backends=["lightgbm"],
        include_candidate_variants=True,
        requested_tabicl_variants=None,
        training_profile="heavy_validation",
    )
    candidate_ids = [item["candidate_id"] for item in candidates]
    assert "lightgbm_rdkit_all" in candidate_ids
    assert "lightgbm_morgan_only" in candidate_ids
    assert "lightgbm_morgan_count_only" in candidate_ids
    assert "lightgbm_morgan_binary_count_rdkit_all" in candidate_ids
    assert "lightgbm_rdkit_basic_only" not in candidate_ids
    assert "lightgbm_morgan_rdkit_basic" not in candidate_ids


def test_rank_summary_rows_prefers_hardest_split_then_gap():
    toolkit = BenchmarkToolkit()
    ranked = toolkit._rank_summary_rows(
        [
            {
                "candidate_id": "a",
                "hardest_split_r2": 0.50,
                "delta_r2": -0.10,
                "random_r2": 0.70,
                "train_time_s": 100.0,
            },
            {
                "candidate_id": "b",
                "hardest_split_r2": 0.55,
                "delta_r2": -0.20,
                "random_r2": 0.90,
                "train_time_s": 50.0,
            },
        ]
    )
    assert ranked[0]["candidate_id"] == "b"


def test_rank_summary_rows_uses_classification_primary_score():
    toolkit = BenchmarkToolkit()
    ranked = toolkit._rank_summary_rows(
        [
            {
                "candidate_id": "a",
                "primary_metric": "balanced_accuracy",
                "primary_score": 0.62,
                "delta_primary_score": -0.08,
                "random_primary_score": 0.70,
                "train_time_s": 10.0,
            },
            {
                "candidate_id": "b",
                "primary_metric": "balanced_accuracy",
                "primary_score": 0.71,
                "delta_primary_score": -0.05,
                "random_primary_score": 0.76,
                "train_time_s": 20.0,
            },
        ]
    )
    assert ranked[0]["candidate_id"] == "b"


def test_benchmark_qsar_models_requires_explicit_benchmark_request(tmp_path):
    train_csv = tmp_path / "dataset_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "Y": [1.0, 2.0]}).to_csv(train_csv, index=False)

    toolkit = BenchmarkToolkit()
    agent = _fake_agent()

    result = toolkit.benchmark_qsar_models(
        train_csv=str(train_csv),
        task_type="regression",
        target_columns=["Y"],
        smiles_column="smiles",
        output_dir=str(tmp_path / "benchmark_output"),
        agent=agent,
    )

    assert result["benchmark_started"] is False
    assert result["blocked"] is True
    assert "benchmark_requested=True" in result["reason"]
    assert not (tmp_path / "benchmark_output").exists()


def test_benchmark_qsar_models_rejects_plain_validation_protocol_mode(tmp_path):
    train_csv = tmp_path / "dataset_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "Y": [1.0, 2.0]}).to_csv(train_csv, index=False)

    toolkit = BenchmarkToolkit()
    agent = _fake_agent()

    result = toolkit.benchmark_qsar_models(
        train_csv=str(train_csv),
        task_type="regression",
        target_columns=["Y"],
        smiles_column="smiles",
        benchmark_mode="standard_qsar",
        benchmark_requested=True,
        output_dir=str(tmp_path / "benchmark_output"),
        agent=agent,
    )

    assert result["benchmark_started"] is False
    assert result["blocked"] is True
    assert "plain validation protocol" in result["reason"]
    assert not (tmp_path / "benchmark_output").exists()


def test_benchmark_standard_qsar_persists_all_candidates(tmp_path, monkeypatch):
    catalog_path = tmp_path / "model_catalog.json"
    internal_root = tmp_path / "internal"
    train_csv = tmp_path / "dataset_curated.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "Y": [1.0, 2.0]}).to_csv(train_csv, index=False)

    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    import cs_copilot.tools.prediction.model_registry_toolkit as registry_module

    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)

    training_toolkit = QSARTrainingToolkit()
    monkeypatch.setattr(training_toolkit.chemprop_toolkit.backend, "is_available", lambda: True)
    monkeypatch.setattr(training_toolkit.lightgbm_toolkit.backend, "is_available", lambda: True)
    monkeypatch.setattr(training_toolkit.tabicl_toolkit.backend, "is_available", lambda: True)
    backends = training_toolkit.backend_mapping()

    registry_toolkit = ModelRegistryToolkit(
        backends=backends,
        catalog=catalog_module.PredictionModelCatalog(records=[], source_path=catalog_path),
        default_backend_name="chemprop",
        register_tools=False,
    )

    toolkit = BenchmarkToolkit(
        training_toolkit=training_toolkit,
        registry_toolkit=registry_toolkit,
    )

    monkeypatch.setattr(
        toolkit,
        "_resolve_compute_profile",
        lambda: {
            "compute_environment": {
                "execution_env": "apptainer_local",
                "cpu_count": 48,
                "gpu_available": True,
                "suggested_profile": "heavy_validation",
            },
            "training_profile": "heavy_validation",
            "profile_reason": "test",
        },
    )

    def fake_train_qsar_model(
        *,
        train_csv,
        backend_name,
        task_type,
        output_dir,
        smiles_column="smiles",
        target_columns=None,
        validation_protocol="standard_qsar",
        representation_name=None,
        extra_args=None,
        agent=None,
        **kwargs,
    ):
        root = Path(output_dir)
        model_name = "best.pt" if backend_name == "chemprop" else "best.pkl"
        model_suffix = ".pt" if backend_name == "chemprop" else ".pkl"
        model_path, test_predictions_path, config_path, splits_path = _write_fake_training_outputs(
            root,
            model_name=model_name,
            model_suffix=model_suffix,
        )
        deployment_predictions = root / "test_predictions.csv"
        deployment_predictions.write_text(test_predictions_path.read_text())

        random_r2 = {"chemprop": 0.60, "lightgbm": 0.57, "tabicl": 0.58}[backend_name]
        scaffold_r2 = {"chemprop": 0.52, "lightgbm": 0.49, "tabicl": 0.50}[backend_name]
        train_time = {"chemprop": 12.0, "lightgbm": 9.0, "tabicl": 10.0}[backend_name]
        feature_columns = [] if backend_name == "chemprop" else ["feature_a", "feature_b"]
        resolved_representation = representation_name or (
            "molecular_graph" if backend_name == "chemprop" else "morgan_rdkit_basic"
        )

        result = {
            "best_model_path": str(model_path),
            "model_path": str(model_path),
            "summary_path": str(root / "cs_copilot_training_summary.json"),
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "test_predictions_path": str(deployment_predictions),
            "validation_assessment": _fake_validation_assessment(
                validation_protocol,
                random_r2,
                "scaffold",
                scaffold_r2,
            ),
            "split_results": [
                {
                    "strategy_label": "random",
                    "metrics": {
                        "test": {
                            "r2": random_r2,
                            "rmse": 0.70,
                            "mae": 0.50,
                            "mse": 0.49,
                        }
                    },
                },
                {
                    "strategy_label": "scaffold",
                    "metrics": {
                        "test": {
                            "r2": scaffold_r2,
                            "rmse": 0.75,
                            "mae": 0.55,
                            "mse": 0.56,
                        }
                    },
                },
            ],
            "training_durations": {"total_duration_seconds": train_time},
            "metrics": {"test": {"r2": random_r2}},
            "feature_columns": feature_columns,
            "categorical_feature_columns": [],
            "applicability_domain": {},
            "train_csv": train_csv,
            "candidate_train_csv": train_csv,
            "backend_name": backend_name,
            "representation_name": resolved_representation,
            "trained_at": "2026-04-27T12:00:00+02:00",
        }
        (root / "cs_copilot_training_summary.json").write_text(json.dumps(result))
        agent.session_state["prediction_models"]["training_runs"].append(
            {
                "train_csv": train_csv,
                "output_dir": str(root.resolve()),
                "task_type": task_type,
                "smiles_columns": [smiles_column],
                "target_columns": target_columns or [],
                "validation_protocol": validation_protocol,
                "training_profile": "heavy_validation",
            }
        )
        return result

    monkeypatch.setattr(training_toolkit, "train_qsar_model", fake_train_qsar_model)

    agent = _fake_agent()
    output_dir = tmp_path / "benchmark_output"
    result = toolkit.benchmark_qsar_models(
        train_csv=str(train_csv),
        task_type="regression",
        target_columns=["Y"],
        smiles_column="smiles",
        benchmark_mode="benchmark_standard_qsar",
        output_dir=str(output_dir),
        benchmark_requested=True,
        agent=agent,
    )

    assert Path(result["summary_path"]).exists()
    assert Path(result["leaderboard_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["campaign_seed_policy"]["mode"] == "generated_per_benchmark_campaign"
    assert result["campaign_seed_policy"]["shared_across_candidates"] is True
    assert (
        result["seed_policy_report"] == "Politique de seeds : partagée au niveau campagne benchmark"
    )
    assert result["feature_cache"]["feature_cache_dir"].endswith("feature_cache")

    candidate_ids = {item["candidate_id"] for item in result["persisted_model_mapping"]}
    assert "chemprop_default" in candidate_ids
    assert "lightgbm_rdkit_all" in candidate_ids
    assert "lightgbm_morgan_only" in candidate_ids
    assert "lightgbm_morgan_count_only" in candidate_ids
    assert "lightgbm_morgan_binary_count_rdkit_all" in candidate_ids
    assert "tabicl_rdkit_all" in candidate_ids
    assert "tabicl_morgan_only" in candidate_ids
    assert "tabicl_morgan_count_only" in candidate_ids
    assert "tabicl_morgan_binary_count_rdkit_all" in candidate_ids

    for item in result["persisted_model_mapping"]:
        model_root = Path(item["internal_model_root"])
        assert model_root.exists()
        assert (model_root / "metadata.json").exists()
        assert (model_root / "model").exists()
        assert (model_root / "artifacts").exists()
        metadata = json.loads((model_root / "metadata.json").read_text())
        persisted_seed_policy = metadata["training_data_summary"]["seed_policy"]
        assert persisted_seed_policy["split_runs"] == result["campaign_seed_policy"]["split_runs"]
        assert (
            persisted_seed_policy["campaign_seed"]
            == result["campaign_seed_policy"]["campaign_seed"]
        )
        assert metadata["reproducibility"]["seed_policy_mode"] == "generated_per_benchmark_campaign"
        assert (
            metadata["training_data_summary"]["seed_policy_report"] == result["seed_policy_report"]
        )

    benchmark_summary = json.loads(Path(result["summary_path"]).read_text())
    assert (
        benchmark_summary["campaign_seed_policy"]["split_runs"]
        == result["campaign_seed_policy"]["split_runs"]
    )
    assert benchmark_summary["seed_policy_report"] == result["seed_policy_report"]
    assert benchmark_summary["feature_cache"]["feature_cache_dir"].endswith("feature_cache")

    leaderboard = pd.read_csv(result["leaderboard_path"])
    assert {"candidate_id", "model_id", "backend", "representation", "hardest_split_r2"}.issubset(
        set(leaderboard.columns)
    )
    chemprop_row = leaderboard.loc[leaderboard["candidate_id"] == "chemprop_default"].iloc[0]
    assert chemprop_row["representation"] == "molecular_graph"
