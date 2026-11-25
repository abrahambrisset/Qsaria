from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cs_copilot.tools.prediction import catalog as catalog_module
from cs_copilot.tools.prediction.benchmark_toolkit import BenchmarkToolkit, _coerce_list
from cs_copilot.tools.prediction.lightgbm_toolkit import LightGBMToolkit
from cs_copilot.tools.prediction.model_registry_toolkit import ModelRegistryToolkit
from cs_copilot.tools.prediction.qsar_contracts import (
    ComputeConfig,
    QsariaBenchmarkRequest,
    RepeatedHoldoutValidation,
    StandardQsarValidation,
    canonical_validation_strategy,
)
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
        "random": {"r2_mean": random_r2, "r2_std": 0.0},
        hardest_name: {"r2_mean": hardest_r2},
    }
    return {
        "hardest_split": hardest_name,
        "delta_vs_random": {hardest_name: {"r2": hardest_r2 - random_r2}},
        "aggregated_split_metrics": aggregated,
        "governance": {"recommended_status": "workflow_demo"},
    }


@pytest.fixture
def resumable_benchmark(tmp_path, monkeypatch):
    import cs_copilot.tools.prediction.model_registry_toolkit as registry_module

    internal_root = tmp_path / "internal"
    catalog_path = tmp_path / "catalog.json"
    monkeypatch.setattr(catalog_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    monkeypatch.setattr(registry_module, "DEFAULT_INTERNAL_MODEL_ROOT", internal_root)
    train_csv = tmp_path / "dataset.csv"
    train_csv.write_text("smiles,Y\nCCO,1\nCCC,2\n")
    calls = []
    failing = set()

    def fake_train(*, request, output_dir, seed_policy, **_kwargs):
        name = request.representation.name
        calls.append((name, seed_policy))
        if name in failing:
            raise RuntimeError(f"{name} fixture fit failed")
        path = Path(output_dir) / "model_0" / "best.pkl"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test model")
        return {
            "model_path": str(path),
            "best_model_path": str(path),
            "trained_at": "2026-04-27T12:00:00+02:00",
            "metrics": {"test": {"r2": 0.5}},
            "validation_assessment": _fake_validation_assessment(
                "standard_qsar", 0.5, "scaffold", 0.4
            ),
        }

    def toolkit():
        training = QSARTrainingToolkit()
        monkeypatch.setattr(training.lightgbm_toolkit.backend, "is_available", lambda: True)
        monkeypatch.setattr(training, "train_qsar_model", fake_train)
        registry = ModelRegistryToolkit(
            backends=training.backend_mapping(),
            catalog=catalog_module.PredictionModelCatalog.load(str(catalog_path)),
            register_tools=False,
        )
        return BenchmarkToolkit(training_toolkit=training, registry_toolkit=registry)

    def run(instance=None, **changes):
        return (instance or toolkit()).benchmark_qsar_models(
            train_csv=str(train_csv),
            request={
                "campaign_id": "fixture-benchmark",
                "task_type": "regression",
                "target_columns": ["Y"],
                "backends": ["lightgbm"],
                "compute": {"profile": "local_light"},
                **changes,
            },
            output_dir=str(tmp_path / "benchmark"),
            agent=_fake_agent(),
        )

    return SimpleNamespace(
        run=run,
        toolkit=toolkit,
        calls=calls,
        failing=failing,
        train_csv=train_csv,
        catalog_path=catalog_path,
    )


def test_benchmark_retries_failed_candidates_with_original_seeds(resumable_benchmark):
    fixture = resumable_benchmark
    fixture.failing.add("rdkit_all")
    partial = fixture.run()

    assert partial["campaign_status"] == "partial"
    assert partial["campaign_id"] == "fixture-benchmark"
    assert len(partial["persisted_model_mapping"]) == 2
    failed = partial["leaderboard"][-1]
    assert failed["candidate_id"] == "lightgbm_rdkit_all"
    assert failed["candidate_status"] == "failed"
    assert failed["error_stage"] == "training"
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "rdkit_all fixture fit failed"
    assert "rdkit_all fixture fit failed" in Path(partial["report_path"]).read_text()
    assert partial["best_overall_candidate"] != failed["candidate_id"]

    fixture.calls.clear()
    fixture.failing.clear()
    resumed = fixture.run()

    assert resumed["campaign_status"] == "completed"
    assert fixture.calls == [("rdkit_all", partial["campaign_seed_policy"])]
    assert resumed["campaign_seed_policy"] == partial["campaign_seed_policy"]
    assert len(resumed["persisted_model_mapping"]) == 3
    assert {row["model_id"] for row in partial["persisted_model_mapping"]} <= {
        row["model_id"] for row in resumed["persisted_model_mapping"]
    }
    assert (
        len(catalog_module.PredictionModelCatalog.load(str(fixture.catalog_path)).list_models())
        == 3
    )


def test_benchmark_reports_all_failed_and_rejects_changed_inputs(resumable_benchmark):
    fixture = resumable_benchmark
    fixture.failing.add("rdkit_all")
    all_failed = fixture.run(include_candidate_variants=False)

    assert all_failed["campaign_status"] == "all_candidates_failed"
    assert all_failed["status"] == "retryable_error"
    assert all_failed["best_overall_candidate"] is None
    assert all_failed["persisted_model_mapping"] == []
    leaderboard = pd.read_csv(all_failed["leaderboard_path"])
    assert leaderboard["candidate_status"].tolist() == ["failed"]
    checkpoint = Path(all_failed["campaign_checkpoint_path"])
    original = checkpoint.read_bytes()

    with pytest.raises(ValueError, match="benchmark campaign inputs changed"):
        fixture.run()
    fixture.train_csv.write_text("smiles,Y\nCCO,3\nCCC,2\n")
    with pytest.raises(ValueError, match="benchmark campaign inputs changed"):
        fixture.run(include_candidate_variants=False)
    assert checkpoint.read_bytes() == original
    assert len(fixture.calls) == 1


@pytest.mark.parametrize("catalog_committed", [False, True])
def test_benchmark_recovers_durable_training_after_interruption(
    resumable_benchmark, monkeypatch, catalog_committed
):
    fixture = resumable_benchmark
    instance = fixture.toolkit()
    registry = instance.registry_toolkit
    persist = registry.persist_registered_model

    def interrupt_persistence(**kwargs):
        if catalog_committed:
            persist(**kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(registry, "persist_registered_model", interrupt_persistence)
    with pytest.raises(KeyboardInterrupt):
        fixture.run(instance, include_candidate_variants=False)

    fixture.calls.clear()
    resumed_instance = fixture.toolkit()
    if catalog_committed:

        def duplicate_persistence(**_kwargs):
            pytest.fail("resuming a committed candidate should never persist it a second time")

        monkeypatch.setattr(
            resumed_instance.registry_toolkit, "persist_registered_model", duplicate_persistence
        )
    resumed = fixture.run(resumed_instance, include_candidate_variants=False)

    assert resumed["campaign_status"] == "completed"
    assert fixture.calls == []
    assert len(resumed["persisted_model_mapping"]) == 1
    assert (
        len(catalog_module.PredictionModelCatalog.load(str(fixture.catalog_path)).list_models())
        == 1
    )


def test_benchmark_flags_a_completed_candidate_with_a_missing_model(resumable_benchmark):
    fixture = resumable_benchmark
    completed = fixture.run(include_candidate_variants=False)
    model_path = Path(completed["persisted_model_mapping"][0]["best_model_path"])
    model_bytes = model_path.read_bytes()
    model_path.unlink()
    fixture.calls.clear()
    catalog_before = fixture.catalog_path.read_bytes()

    stale = fixture.run(include_candidate_variants=False)

    assert stale["campaign_status"] == "all_candidates_failed"
    assert stale["leaderboard"][0]["error_stage"] == "registry"
    assert "Registry evidence" in stale["leaderboard"][0]["error_message"]
    assert fixture.catalog_path.read_bytes() == catalog_before
    assert fixture.calls == []

    model_path.write_bytes(b"x" * len(model_bytes))
    corrupted = fixture.run(include_candidate_variants=False)
    assert corrupted["campaign_status"] == "all_candidates_failed"
    assert "checksum changed" in corrupted["leaderboard"][0]["error_message"]
    assert fixture.catalog_path.read_bytes() == catalog_before
    assert fixture.calls == []

    model_path.write_bytes(model_bytes)
    restored = fixture.run(include_candidate_variants=False)
    assert restored["persisted_model_mapping"] == completed["persisted_model_mapping"]
    assert fixture.catalog_path.read_bytes() == catalog_before
    assert fixture.calls == []


@pytest.mark.parametrize("surface", ["chainlit", "mcp"])
def test_named_campaign_resumes_from_public_surface(
    resumable_benchmark, tmp_path, monkeypatch, surface
):
    fixture = resumable_benchmark
    request = QsariaBenchmarkRequest(
        campaign_id="public-replay",
        task_type="regression",
        target_columns=["Y"],
        backends=["lightgbm"],
        include_candidate_variants=False,
        compute=ComputeConfig(profile="local_light"),
    )
    if surface == "chainlit":
        from cs_copilot.storage import S3

        monkeypatch.setattr(S3, "local_session_root", lambda: tmp_path / "chat-session")

        def run():
            return fixture.toolkit()._agno_benchmark_qsar_models(
                train_csv=str(fixture.train_csv), request=request, agent=_fake_agent()
            )

    else:
        from cs_copilot.mcp.qsaria.adapter import build_qsaria_tool
        from cs_copilot.mcp.qsaria.experiments import ExperimentManager
        from cs_copilot.mcp.tool_specs.qsaria import SPECS

        manager = ExperimentManager(local_root=tmp_path)
        experiment_id = manager.create_experiment("resume benchmark")["experiment_id"]
        spec = next(item for item in SPECS if item.method == "benchmark_qsar_models")

        def run():
            tool = build_qsaria_tool(spec, fixture.toolkit(), manager)
            return asyncio.run(
                tool(experiment_id=experiment_id, train_csv=str(fixture.train_csv), request=request)
            )

    first = run()
    second = run()

    assert first["campaign_id"] == second["campaign_id"] == "public-replay"
    assert first["campaign_checkpoint_path"] == second["campaign_checkpoint_path"]
    assert first["persisted_model_mapping"] == second["persisted_model_mapping"]
    assert len(fixture.calls) == 1


def test_benchmark_public_contract_has_no_benchmark_mode():
    import inspect

    parameters = inspect.signature(BenchmarkToolkit.benchmark_qsar_models).parameters

    assert "benchmark_mode" not in parameters
    assert "validation_strategy" not in parameters
    assert parameters["request"].annotation == "QsariaBenchmarkRequest"

    schema = BenchmarkToolkit().functions["benchmark_qsar_models"].parameters
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == {"train_csv", "request"}
    serialized = json.dumps(schema)
    assert "output_dir" not in serialized
    assert "bundle_dir" not in serialized
    assert "extra_args" not in serialized


def test_benchmark_candidate_reuses_campaign_seed_policy(tmp_path):
    calls = []

    class TrainingFacade:
        def train_qsar_model(self, **kwargs):
            calls.append(kwargs)
            return {"train_csv": kwargs["train_csv"]}

    campaign_seed_policy = {
        "mode": "generated_per_benchmark_campaign",
        "model_seed": 42,
        "split_runs": [{"label": "random_seed_71", "seed": 71, "backend_split_type": "random"}],
    }
    toolkit = BenchmarkToolkit(training_toolkit=TrainingFacade())
    toolkit._train_candidate(
        candidate={
            "candidate_id": "lightgbm_rdkit_all",
            "backend_name": "lightgbm",
            "representation_name": "rdkit_all",
        },
        train_csv=str(tmp_path / "curated.csv"),
        task_type="regression",
        smiles_column="smiles",
        target_columns=["Y"],
        candidate_dir=tmp_path / "candidate",
        campaign_seed_policy=campaign_seed_policy,
        validation=StandardQsarValidation(seed=71),
        compute=ComputeConfig(profile="local_light"),
        agent=_fake_agent(),
    )

    assert calls[0]["seed_policy"] == campaign_seed_policy
    assert calls[0]["request"].validation.seed == 71


@pytest.mark.parametrize("tuning_enabled", [False, True])
def test_benchmark_propagates_one_explicit_strategy_to_every_candidate(tmp_path, tuning_enabled):
    calls = []

    class _TrainingFacade:
        def train_qsar_model(self, **kwargs):
            calls.append(kwargs)
            return {"train_csv": kwargs["train_csv"]}

    toolkit = BenchmarkToolkit(training_toolkit=_TrainingFacade())
    validation = RepeatedHoldoutValidation(split_family="scaffold", n_repeats=3)
    seed_policy = {"split_runs": [{"seed": 42}], "model_seed": 42}
    for backend_name, representation_name in (
        ("chemprop", "molecular_graph"),
        ("lightgbm", "rdkit_all"),
        ("tabicl", "rdkit_all"),
    ):
        toolkit._train_candidate(
            candidate={
                "candidate_id": f"{backend_name}_{representation_name}",
                "backend_name": backend_name,
                "representation_name": representation_name,
            },
            train_csv=str(tmp_path / "curated.csv"),
            task_type="regression",
            smiles_column="smiles",
            target_columns=["Y"],
            candidate_dir=tmp_path / backend_name,
            campaign_seed_policy=seed_policy,
            validation=validation,
            compute=ComputeConfig(profile="heavy_validation", allow_heavy_compute=True),
            agent=_fake_agent(),
            **({"tuning_enabled": True} if tuning_enabled else {}),
        )

    assert [canonical_validation_strategy(call["request"].validation) for call in calls] == [
        canonical_validation_strategy(validation),
        canonical_validation_strategy(validation),
        canonical_validation_strategy(validation),
    ]
    assert [call["request"].tuning.enabled for call in calls] == [
        tuning_enabled,
        tuning_enabled,
        False,
    ]


def test_expand_candidates_includes_heavy_tabicl_all():
    toolkit = BenchmarkToolkit()
    toolkit.training_toolkit.backend_mapping = lambda: {
        "chemprop": SimpleNamespace(is_available=lambda: False),
        "lightgbm": SimpleNamespace(is_available=lambda: False),
        "tabicl": SimpleNamespace(is_available=lambda: True),
    }
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
    assert "tabicl_morgan_binary_count_rdkit_all" not in candidate_ids
    assert "tabicl_rdkit_basic_only" not in candidate_ids
    assert "tabicl_morgan_rdkit_basic" not in candidate_ids


def test_expand_candidates_includes_heavy_lightgbm_all():
    toolkit = BenchmarkToolkit()
    toolkit.training_toolkit.backend_mapping = lambda: {
        "chemprop": SimpleNamespace(is_available=lambda: False),
        "lightgbm": SimpleNamespace(is_available=lambda: True),
        "tabicl": SimpleNamespace(is_available=lambda: False),
    }
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
    assert "lightgbm_morgan_binary_count_rdkit_all" not in candidate_ids
    assert "lightgbm_rdkit_basic_only" not in candidate_ids
    assert "lightgbm_morgan_rdkit_basic" not in candidate_ids


@pytest.mark.parametrize("task_type", ["classification", "multiclass_classification"])
@pytest.mark.parametrize("tabicl_classifier_available", [True, False])
def test_classification_benchmark_includes_available_tabular_backends(
    task_type, tabicl_classifier_available
):
    toolkit = BenchmarkToolkit()
    toolkit.training_toolkit.backend_mapping = lambda: {
        "chemprop": SimpleNamespace(is_available=lambda: True),
        "lightgbm": SimpleNamespace(is_available=lambda: True),
        "tabicl": SimpleNamespace(
            is_available=lambda: True,
            describe_environment=lambda: {"classification_available": tabicl_classifier_available},
        ),
    }

    candidates = toolkit._expand_candidates(
        task_type=task_type,
        target_columns=["Y"],
        requested_backends=None,
        include_candidate_variants=False,
        requested_tabicl_variants=None,
        training_profile="local_light",
    )

    expected = {"chemprop", "lightgbm"}
    if tabicl_classifier_available:
        expected.add("tabicl")
    assert {candidate["backend_name"] for candidate in candidates} == expected


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

    forwarded_bundle_dirs = []

    def fake_train_qsar_model(
        *,
        train_csv,
        request,
        output_dir,
        agent=None,
        **kwargs,
    ):
        backend_name = request.backend.name
        task_type = request.task_type
        smiles_column = request.smiles_column
        target_columns = request.target_columns
        validation_protocol = (
            "standard_qsar" if isinstance(request.validation, StandardQsarValidation) else "custom"
        )
        representation_name = getattr(request.representation, "name", None)
        forwarded_bundle_dirs.append(kwargs.get("bundle_dir"))
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
            "molecular_graph" if backend_name == "chemprop" else "rdkit_all"
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
    bundle_dir = tmp_path / "benchmark_bundles"
    result = toolkit.benchmark_qsar_models(
        train_csv=str(train_csv),
        request=QsariaBenchmarkRequest(
            task_type="regression",
            target_columns=["Y"],
        ),
        output_dir=str(output_dir),
        bundle_dir=str(bundle_dir),
        agent=agent,
    )

    assert Path(result["summary_path"]).exists()
    assert Path(result["leaderboard_path"]).exists()
    assert Path(result["report_path"]).exists()
    assert result["workflow_kind"] == "benchmark"
    assert result["validation_protocol"] == "standard_qsar"
    assert "benchmark_mode" not in result
    assert result["campaign_seed_policy"]["mode"] == "generated_per_benchmark_campaign"
    assert result["campaign_seed_policy"]["shared_across_candidates"] is True
    assert (
        result["seed_policy_report"] == "Politique de seeds : partagée au niveau campagne benchmark"
    )
    assert result["feature_cache"]["feature_cache_dir"].endswith("feature_cache")
    assert len(forwarded_bundle_dirs) == len(result["persisted_model_mapping"])
    assert set(forwarded_bundle_dirs) == {str(bundle_dir)}

    candidate_ids = {item["candidate_id"] for item in result["persisted_model_mapping"]}
    assert "chemprop_default" in candidate_ids
    assert "lightgbm_rdkit_all" in candidate_ids
    assert "lightgbm_morgan_only" in candidate_ids
    assert "lightgbm_morgan_count_only" in candidate_ids
    assert "tabicl_rdkit_all" in candidate_ids
    assert "tabicl_morgan_only" in candidate_ids
    assert "tabicl_morgan_count_only" in candidate_ids
    assert "lightgbm_morgan_binary_count_rdkit_all" not in candidate_ids
    assert "tabicl_morgan_binary_count_rdkit_all" not in candidate_ids

    for item in result["persisted_model_mapping"]:
        model_root = Path(item["internal_model_root"])
        assert model_root.exists()
        assert (model_root / "metadata.json").exists()
        assert (model_root / "model").exists()
        assert (model_root / "artifacts").exists()
        metadata = json.loads((model_root / "metadata.json").read_text())
        assert metadata["training_data_summary"]["workflow_kind"] == "benchmark"
        assert "benchmark_mode" not in metadata["training_data_summary"]
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
