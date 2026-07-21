"""Contract tests for the experiment-aware Qsaria scientific surface."""

from __future__ import annotations

import asyncio
import inspect
import json
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.qsaria.adapter import QsariaToolSpec, build_qsaria_tool, is_qsaria_spec
from cs_copilot.mcp.qsaria.contracts import HANDOFF_SCHEMA_VERSION
from cs_copilot.mcp.qsaria.experiments import ExperimentManager
from cs_copilot.mcp.qsaria.toolkit_hub import ToolkitHub
from cs_copilot.mcp.tool_specs.qsaria import SPECS
from cs_copilot.tools.curation.dataset_curation_toolkit import DatasetCurationToolkit
from cs_copilot.tools.prediction.benchmark_toolkit import BenchmarkToolkit
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog
from cs_copilot.tools.prediction.qsar_contracts import (
    GeneratedRepresentation,
    LightGBMConfig,
    LightGBMTrainingRequest,
)


def _handoff(experiment_id, *, agent, status, summary, **overrides):
    payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "agent": agent,
        "status": status,
        "summary": summary,
        "facts": {},
        "artifact_ids": [],
        "model_ids": [],
        "warnings": [],
        "blockers": (
            [summary]
            if status in {"retryable_error", "terminal_failure", "needs_user_input"}
            else []
        ),
        "recommended_next_action": "",
    }
    payload.update(overrides)
    return payload


def test_qsaria_scientific_surface_matches_audited_tool_counts():
    expected_counts = {
        "qsaria_curation_": 5,
        "qsaria_training_": 8,
        "qsaria_registry_": 11,
        "qsaria_inference_": 4,
        "qsaria_ensemble_": 4,
        "qsaria_benchmark_": 1,
        "qsaria_activity_cliffs_": 2,
    }
    assert len(SPECS) == 35
    for prefix, count in expected_counts.items():
        assert sum(spec.mcp_name.startswith(prefix) for spec in SPECS) == count
    assert not any(spec.method == "prepare_training_dataset" for spec in SPECS)
    assert all(is_qsaria_spec(spec) for spec in SPECS)


def test_removed_public_inputs_are_absent_from_runtime_facades():
    assert (
        "curation_backend"
        not in inspect.signature(DatasetCurationToolkit.curate_qsar_dataset).parameters
    )
    assert (
        "benchmark_mode" not in inspect.signature(BenchmarkToolkit.benchmark_qsar_models).parameters
    )


def test_catalog_mutation_specs_are_explicit_and_locked():
    writes = {spec.mcp_name for spec in SPECS if spec.catalog_write}
    assert writes == {
        "qsaria_registry_persist_registered_model",
        "qsaria_registry_register_and_persist_candidates",
        "qsaria_ensemble_create_ensemble_from_catalog",
        "qsaria_ensemble_evaluate_ensemble_on_dataset",
        "qsaria_inference_evaluate_model_on_dataset",
        "qsaria_benchmark_benchmark_qsar_models",
    }
    assert all(spec.catalog_access for spec in SPECS if spec.catalog_write)


def test_output_parameters_are_forced_beneath_experiment():
    forced = {spec.mcp_name: set(spec.output_paths) for spec in SPECS if spec.output_paths}
    assert forced["qsaria_curation_curate_qsar_dataset"] == {
        "output_csv",
        "report_path",
    }
    assert forced["qsaria_training_train_qsar_model"] == {
        "bundle_dir",
        "output_dir",
    }
    assert forced["qsaria_curation_write_curation_report"] == {
        "report_path",
        "bundle_path",
    }
    assert forced["qsaria_inference_predict_from_csv"] == {
        "preds_path",
        "materialized_input_path",
    }
    assert forced["qsaria_inference_predict_from_smiles"] == {
        "preds_path",
        "input_csv_path",
    }
    assert forced["qsaria_inference_export_prediction_summary"] == {"summary_csv"}
    assert forced["qsaria_benchmark_benchmark_qsar_models"] == {
        "bundle_dir",
        "output_dir",
    }
    training_specs = [spec for spec in SPECS if spec.mcp_name.startswith("qsaria_training_train_")]
    assert len(training_specs) == 4
    assert all(spec.nested_output_paths == {} for spec in training_specs)
    assert all(spec.nested_blocked_inputs == () for spec in training_specs)
    report_spec = next(
        spec for spec in SPECS if spec.mcp_name == "qsaria_curation_write_curation_report"
    )
    assert report_spec.session_forces == {"curation_result": ("qsar_curation", "last_result")}
    candidate_spec = next(
        spec for spec in SPECS if spec.mcp_name == "qsaria_registry_register_and_persist_candidates"
    )
    assert candidate_spec.forces == {"candidate_registry_payloads": None}
    assert candidate_spec.registered_artifact_inputs == {
        "candidate_manifest_path": (
            "catalog_candidates_manifest.json",
            ("qsaria_training_",),
        )
    }
    assert candidate_spec.json_payload_inputs == ("candidate_manifest_path",)


class _DummyToolkit:
    def execute(
        self,
        value: str,
        output_dir: str | None = None,
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        assert session_state is agent.session_state
        agent.session_state["value"] = value
        return {"status": "completed", "output_dir": output_dir}

    def inspect(self, value: str = "ok", agent: Any | None = None) -> dict[str, Any]:
        assert agent is not None
        assert agent.model is None
        assert agent.llm_policy == "disabled"
        return {"value": value}

    def write_feature_cache(
        self,
        output_dir: str,
        resolved_parameters: dict[str, Any] | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        assert output_dir
        cache = Path((resolved_parameters or {})["feature_cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache / "cache.marker"
        marker.write_text("managed")
        return {
            "status": "completed",
            "resolved_parameters": dict(resolved_parameters or {}),
            "feature_cache_dir": str(cache),
            "marker": str(marker),
        }

    def write_training_bundle(
        self,
        output_dir: str,
        bundle_dir: str,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        bundle = Path(bundle_dir) / "training_output_training_bundle.zip"
        bundle.parent.mkdir(parents=True, exist_ok=True)
        bundle.write_text(output_dir, encoding="utf-8")
        return {"status": "completed", "bundle_file_ref": str(bundle)}

    def train_lightgbm_model(
        self,
        train_csv: str,
        request: LightGBMTrainingRequest,
        output_dir: str,
        bundle_dir: str | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        if isinstance(request, dict):
            request = LightGBMTrainingRequest.model_validate(request)
        return {
            "train_csv": train_csv,
            "task_type": request.task_type,
            "target_columns": request.target_columns,
            "output_dir": output_dir,
            "bundle_dir": bundle_dir,
            "request": request.model_dump(mode="json"),
        }

    def boom(self, agent: Any | None = None) -> None:
        assert agent is not None
        agent.session_state["before_error"] = True
        raise ValueError("scientific failure")

    def transient_timeout(self, agent: Any | None = None) -> None:
        assert agent is not None
        raise TimeoutError("temporary backend timeout")

    def accept_dataset_path(
        self,
        dataset_path: str,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        return {"status": "completed", "dataset": dataset_path}

    def write_curated_dataset(
        self,
        output_csv: str,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("smiles,pEC50\nCCO,4.2\n", encoding="utf-8")
        return {
            "status": "completed",
            "ready_for_qsar": True,
            "curated_dataset_path": str(output),
        }

    def accept_model_id(
        self,
        model_id: str,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        return {"status": "completed", "model_id": model_id}

    def accept_training_summary(
        self,
        training_data_summary: dict[str, Any],
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        return {"status": "completed", "training_data_summary": training_data_summary}

    def accept_inference_profile(
        self,
        inference_profile: dict[str, Any],
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        return {"status": "completed", "inference_profile": inference_profile}

    def accept_candidate_manifest(
        self,
        candidate_registry_payloads: list[dict[str, Any]] | None = None,
        candidate_manifest_path: str | None = None,
        agent: Any | None = None,
    ) -> dict[str, Any]:
        assert agent is not None
        return {
            "status": "completed",
            "candidate_manifest_path": candidate_manifest_path,
            "candidate_registry_payloads": candidate_registry_payloads,
        }

    def missing_external_targets(self, agent: Any | None = None) -> None:
        assert agent is not None
        raise ValueError("External evaluation is missing required target columns: ['logS']")

    def blocked_external_evaluation(self, agent: Any | None = None) -> dict[str, Any]:
        assert agent is not None
        return {
            "status": "blocked_failed_external_evaluation",
            "prediction_generated": False,
        }

    def blocked_curation(self, agent: Any | None = None) -> dict[str, Any]:
        assert agent is not None
        return {"status": "blocked", "ready_for_qsar": False, "reason": "invalid target"}

    def registration_rejected(self, agent: Any | None = None) -> dict[str, Any]:
        assert agent is not None
        return {
            "registered": False,
            "error": "Unknown catalog model_id: typo",
            "available_model_ids": ["real-model"],
        }

    def unknown_model(self, model_id: str, agent: Any | None = None) -> None:
        assert agent is not None
        raise ValueError(f"Unknown model_id: {model_id}")


def test_lightgbm_mcp_spec_accepts_only_the_typed_public_request(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("lightgbm cache contract")["experiment_id"]
    train_csv = tmp_path / "storage" / "curated.csv"
    train_csv.write_text("smiles,pEC50\nCCO,4.2\n", encoding="utf-8")
    spec = next(item for item in SPECS if item.method == "train_lightgbm_model")
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(
        tool(
            experiment_id=experiment_id,
            train_csv=str(train_csv),
            request=LightGBMTrainingRequest(
                task_type="regression",
                target_columns=["pEC50"],
                representation=GeneratedRepresentation(name="rdkit_all"),
                backend=LightGBMConfig(num_leaves=31),
            ),
        )
    )

    assert result["request"]["backend"]["num_leaves"] == 31
    assert "feature_cache_dir" not in json.dumps(result["request"])
    assert Path(result["output_dir"]).is_relative_to(
        tmp_path / "storage" / "sessions" / experiment_id
    )


class _DummyRun:
    def __init__(self) -> None:
        self.experiment_id = "exp_test"
        self.runtime_state: dict[str, Any] = {"artifacts": {}}
        self.context = MCPAgentContext(llm_policy="disabled")
        self.captured: Any = None
        self.inputs: Any = None

    def artifact_path(self, filename: str, *, category: str = "artifacts") -> str:
        return f"/tmp/sessions/exp_test/{category}/{filename}"

    def capture_result(self, result: Any, *, status: str = "success") -> None:
        self.captured = result

    def capture_inputs(self, payload: Any) -> None:
        self.inputs = payload

    def capture_error(self, error: Any, *, status: str) -> None:
        self.captured = {"error": str(error), "status": status}


class _DummyManager:
    def __init__(self) -> None:
        self.run_args: tuple[Any, ...] | None = None
        self.runtime = _DummyRun()

    @contextmanager
    def run(
        self,
        experiment_id: str,
        *,
        agent_name: str,
        tool_name: str,
        catalog_write: bool,
    ):
        self.run_args = (experiment_id, agent_name, tool_name, catalog_write)
        self.runtime.experiment_id = experiment_id
        yield self.runtime


def test_qsaria_adapter_adds_experiment_and_hides_injected_outputs():
    spec = QsariaToolSpec(
        mcp_name="qsaria_test_execute",
        toolkit_factory=_DummyToolkit,
        method="execute",
        summary="test",
        agent_name="qsaria_training",
        output_paths={"output_dir": "training_output"},
    )
    manager = _DummyManager()
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)  # type: ignore[arg-type]
    signature = inspect.signature(tool)

    assert list(signature.parameters) == ["experiment_id", "value"]
    result = asyncio.run(tool(experiment_id="exp_test", value="ok"))

    assert manager.run_args == (
        "exp_test",
        "qsaria_training",
        "qsaria_test_execute",
        False,
    )
    assert result == {
        "status": "completed",
        "output_dir": "/tmp/sessions/exp_test/artifacts/training_output",
    }
    assert manager.runtime.captured == result
    assert manager.runtime.context.model is None
    assert manager.runtime.context.llm_policy == "disabled"


def test_global_read_signatures_do_not_require_experiment_id():
    expected = {
        "qsaria_curation_inspect_dataset_schema",
        "qsaria_curation_identify_qsar_columns",
        "qsaria_curation_summarize_curated_dataset",
        "qsaria_training_describe_qsar_training_environment",
        "qsaria_training_describe_backend_hyperparameters",
        "qsaria_training_describe_tuning_engines",
        "qsaria_training_describe_outlier_analysis",
        "qsaria_registry_describe_backends",
        "qsaria_registry_describe_catalog",
        "qsaria_registry_list_catalog_models",
        "qsaria_registry_summarize_catalog_model",
        "qsaria_registry_recommend_catalog_model",
        "qsaria_ensemble_inspect_ensemble_candidates",
        "qsaria_ensemble_summarize_ensemble",
        "qsaria_activity_cliffs_list_activity_cliff_indexes",
    }
    global_specs = {spec.mcp_name: spec for spec in SPECS if not spec.requires_experiment}
    assert set(global_specs) == expected

    manager = _DummyManager()
    for original in global_specs.values():
        test_spec = QsariaToolSpec(
            mcp_name=original.mcp_name,
            toolkit_factory=_DummyToolkit,
            method="inspect",
            summary=original.summary,
            read_only=True,
            agent_name=original.agent_name,
            requires_experiment=False,
        )
        tool = build_qsaria_tool(test_spec, _DummyToolkit(), manager)  # type: ignore[arg-type]
        assert "experiment_id" not in inspect.signature(tool).parameters


def test_global_read_uses_ephemeral_model_free_context_without_manager_run():
    spec = QsariaToolSpec(
        mcp_name="qsaria_test_inspect",
        toolkit_factory=_DummyToolkit,
        method="inspect",
        summary="test",
        read_only=True,
        agent_name="qsaria_registry",
        requires_experiment=False,
    )
    manager = _DummyManager()
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)  # type: ignore[arg-type]

    assert asyncio.run(tool(value="global")) == {"value": "global"}
    assert manager.run_args is None


def test_adapter_persists_runtime_state_and_manifest_when_tool_raises(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("exercise adapter failure")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_test_boom",
        toolkit_factory=_DummyToolkit,
        method="boom",
        summary="test failure",
        agent_name="qsaria_training",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="scientific failure"):
        asyncio.run(tool(experiment_id=experiment_id))

    public_state = manager.get_experiment_state(experiment_id)
    runtime_path = tmp_path / "sessions" / experiment_id / "qsaria" / "runtime_state.json"
    runtime_state = json.loads(runtime_path.read_text(encoding="utf-8"))
    manifests = list(
        (tmp_path / "sessions" / experiment_id / "workflows" / "manifests").rglob("*.json")
    )

    assert public_state["status"] == "terminal_failure"
    assert runtime_state["session_state"]["before_error"] is True
    assert runtime_state["tool_events"][-1]["tool_name"] == "qsaria_test_boom"
    assert runtime_state["tool_events"][-1]["status"] == "terminal_failure"
    assert manifests


def test_missing_external_targets_are_persisted_as_terminal(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("external evaluation")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_inference_evaluate_model_on_dataset",
        toolkit_factory=_DummyToolkit,
        method="missing_external_targets",
        summary="terminal external evaluation",
        agent_name="qsaria_inference",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="missing required target columns"):
        asyncio.run(tool(experiment_id=experiment_id))

    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"


def test_blocked_external_evaluation_result_is_terminal(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("blocked evaluation")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_inference_predict_from_csv",
        toolkit_factory=_DummyToolkit,
        method="blocked_external_evaluation",
        summary="blocked external evaluation",
        agent_name="qsaria_inference",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(tool(experiment_id=experiment_id))

    assert result["status"] == "blocked_failed_external_evaluation"
    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"


def test_blocked_curation_result_is_terminal_without_llm_interpretation(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("blocked curation")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_curation_curate_qsar_dataset",
        toolkit_factory=_DummyToolkit,
        method="blocked_curation",
        summary="blocked curation",
        agent_name="qsaria_curation",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(tool(experiment_id=experiment_id))

    assert result["status"] == "blocked"
    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"


def test_registry_rejected_input_requests_a_user_correction(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("correct registry input")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_register_catalog_model",
        toolkit_factory=_DummyToolkit,
        method="registration_rejected",
        summary="correctable catalog input",
        agent_name="qsaria_registry",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(tool(experiment_id=experiment_id))

    assert result["registered"] is False
    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"


def test_inference_unknown_model_id_requests_a_user_correction(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("correct inference model id")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_inference_predict_from_csv",
        toolkit_factory=_DummyToolkit,
        method="unknown_model",
        summary="correctable inference input",
        agent_name="qsaria_inference",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="Unknown model_id"):
        asyncio.run(tool(experiment_id=experiment_id, model_id="typo"))

    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"


def test_transient_error_gets_exactly_one_automatic_retry(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("retry timeout")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_training_transient_timeout",
        toolkit_factory=_DummyToolkit,
        method="transient_timeout",
        summary="transient timeout",
        agent_name="qsaria_training",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="temporary backend timeout"):
        asyncio.run(tool(experiment_id=experiment_id))
    assert manager.get_experiment_state(experiment_id)["status"] == "retryable_error"

    with pytest.raises(MCPToolError, match="requires its structured agent handoff"):
        asyncio.run(tool(experiment_id=experiment_id))
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="retryable_error",
            summary="Temporary backend timeout",
            blockers=["temporary backend timeout"],
        ),
    )
    with pytest.raises(MCPToolError, match="temporary backend timeout"):
        asyncio.run(tool(experiment_id=experiment_id))
    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"

    with pytest.raises(MCPToolError, match="does not allow another scientific call"):
        asyncio.run(tool(experiment_id=experiment_id))
    runtime_path = tmp_path / "sessions" / experiment_id / "qsaria" / "runtime_state.json"
    events = json.loads(runtime_path.read_text())["tool_events"]
    assert [event["status"] for event in events] == ["retryable_error", "terminal_failure"]


def test_retryable_handoff_preserves_the_exact_tool_retry_boundary(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("retry exact tool")["experiment_id"]
    retry_spec = QsariaToolSpec(
        mcp_name="qsaria_training_transient_timeout",
        toolkit_factory=_DummyToolkit,
        method="transient_timeout",
        summary="transient timeout",
        agent_name="qsaria_training",
    )
    other_spec = QsariaToolSpec(
        mcp_name="qsaria_training_other_operation",
        toolkit_factory=_DummyToolkit,
        method="inspect",
        summary="different operation",
        agent_name="qsaria_training",
    )
    retry_tool = build_qsaria_tool(retry_spec, _DummyToolkit(), manager)
    other_tool = build_qsaria_tool(other_spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="temporary backend timeout"):
        asyncio.run(retry_tool(experiment_id=experiment_id))
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="retryable_error",
            summary="Transient backend timeout",
            blockers=["temporary backend timeout"],
        ),
    )

    with pytest.raises(MCPToolError, match="can only be resolved"):
        asyncio.run(other_tool(experiment_id=experiment_id))
    with pytest.raises(MCPToolError, match="temporary backend timeout"):
        asyncio.run(retry_tool(experiment_id=experiment_id))

    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"
    terminal_handoff = manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="terminal_failure",
            summary="The single retry was exhausted",
            blockers=["automatic retry exhausted"],
        ),
    )
    assert terminal_handoff["status"] == "terminal_failure"


def test_input_paths_are_bounded_and_user_correction_resolves_blocker(tmp_path):
    storage_root = tmp_path / "storage"
    manager = ExperimentManager(local_root=storage_root)
    experiment_id = manager.create_experiment("bounded input")["experiment_id"]
    allowed = storage_root / "inputs" / "allowed.csv"
    allowed.parent.mkdir(parents=True)
    allowed.write_text("smiles,target\nCCO,1\n")
    outside = tmp_path / "outside.csv"
    outside.write_text("secret\n")
    spec = QsariaToolSpec(
        mcp_name="qsaria_curation_accept_dataset_path",
        toolkit_factory=_DummyToolkit,
        method="accept_dataset_path",
        summary="bounded path",
        agent_name="qsaria_curation",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    accepted = asyncio.run(tool(experiment_id=experiment_id, dataset_path=str(allowed)))
    assert accepted["dataset"] == str(allowed.resolve())

    with pytest.raises(MCPToolError, match="outside the trusted Qsaria input roots"):
        asyncio.run(tool(experiment_id=experiment_id, dataset_path=str(outside)))
    blocked = manager.get_experiment_state(experiment_id)
    assert blocked["status"] == "needs_user_input"
    assert blocked["blockers"]

    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="needs_user_input",
            summary="Dataset path is outside trusted roots",
            blockers=list(blocked["blockers"]),
        ),
    )

    corrected = asyncio.run(tool(experiment_id=experiment_id, dataset_path=str(allowed)))
    assert corrected["status"] == "completed"
    resolved = manager.get_experiment_state(experiment_id)
    assert resolved["status"] == "active"
    assert resolved["blockers"] == []


def _registered_curated_artifact(manager: ExperimentManager, experiment_id: str) -> dict[str, Any]:
    producer_spec = QsariaToolSpec(
        mcp_name="qsaria_curation_write_curated_dataset",
        toolkit_factory=_DummyToolkit,
        method="write_curated_dataset",
        summary="write curated dataset",
        agent_name="qsaria_curation",
        output_paths={"output_csv": "curated_dataset.csv"},
        artifact_category="curation",
    )
    producer = build_qsaria_tool(producer_spec, _DummyToolkit(), manager)
    asyncio.run(producer(experiment_id=experiment_id))
    artifact = next(
        artifact
        for artifact in manager.list_artifacts(experiment_id, kind="dataset")
        if artifact["path"].endswith("/curated_dataset.csv")
    )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_curation",
            status="completed",
            summary="Curated dataset artifact recorded",
            artifact_ids=[artifact["artifact_id"]],
        ),
    )
    return artifact


def _dataset_consumer(manager: ExperimentManager):
    consumer_spec = QsariaToolSpec(
        mcp_name="qsaria_training_accept_curated_dataset",
        toolkit_factory=_DummyToolkit,
        method="accept_dataset_path",
        summary="consume curated dataset",
        agent_name="qsaria_training",
    )
    return build_qsaria_tool(consumer_spec, _DummyToolkit(), manager)


def test_registered_relative_artifact_chains_from_curation_to_training(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("curate then train")["experiment_id"]
    artifact = _registered_curated_artifact(manager, experiment_id)

    result = asyncio.run(
        _dataset_consumer(manager)(
            experiment_id=experiment_id,
            dataset_path=artifact["path"],
        )
    )

    expected = Path(manager.resolve_artifact_path(experiment_id, artifact["path"])).resolve()
    assert result["dataset"] == str(expected)
    assert expected.is_file()


def test_relative_artifact_cannot_cross_experiment_boundary(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    source_id = manager.create_experiment("source artifact")["experiment_id"]
    artifact = _registered_curated_artifact(manager, source_id)
    destination_id = manager.create_experiment("other experiment")["experiment_id"]

    with pytest.raises(MCPToolError, match="does not exist"):
        asyncio.run(
            _dataset_consumer(manager)(
                experiment_id=destination_id,
                dataset_path=artifact["path"],
            )
        )


def test_unregistered_session_relative_path_is_not_resolved(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("unregistered artifact")["experiment_id"]
    relative = "workflows/curation/unregistered/curated_dataset.csv"
    unregistered = Path(manager.resolve_artifact_path(experiment_id, relative))
    unregistered.parent.mkdir(parents=True, exist_ok=True)
    unregistered.write_text("smiles,pEC50\nCCO,4.2\n", encoding="utf-8")

    with pytest.raises(MCPToolError, match="does not exist"):
        asyncio.run(
            _dataset_consumer(manager)(
                experiment_id=experiment_id,
                dataset_path=relative,
            )
        )


def test_unreadable_input_is_classified_as_needs_user_input(tmp_path, monkeypatch):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("unreadable input")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_curation_accept_dataset_path",
        toolkit_factory=_DummyToolkit,
        method="accept_dataset_path",
        summary="unreadable path",
        agent_name="qsaria_curation",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    def reject_unreadable(value, *, parameter):
        raise PermissionError(f"{parameter} is not readable: {value}")

    monkeypatch.setattr(manager, "normalize_input_path", reject_unreadable)

    with pytest.raises(MCPToolError, match="not readable"):
        asyncio.run(tool(experiment_id=experiment_id, dataset_path="locked.csv"))
    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"


def test_model_identifiers_cannot_traverse_storage_paths(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("unsafe model id")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_accept_model_id",
        toolkit_factory=_DummyToolkit,
        method="accept_model_id",
        summary="validate model id",
        agent_name="qsaria_registry",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="path-safe model identifier"):
        asyncio.run(tool(experiment_id=experiment_id, model_id="../../outside"))
    assert manager.get_experiment_state(experiment_id)["status"] == "terminal_failure"


def test_nested_feature_cache_is_forced_inside_the_experiment(tmp_path):
    manager = ExperimentManager(local_root=tmp_path)
    experiment_id = manager.create_experiment("force feature cache")["experiment_id"]
    external_cache = tmp_path / "user-selected-cache"
    external_cache.mkdir()
    spec = QsariaToolSpec(
        mcp_name="qsaria_training_write_feature_cache",
        toolkit_factory=_DummyToolkit,
        method="write_feature_cache",
        summary="force nested output",
        agent_name="qsaria_training",
        output_paths={"output_dir": "training_output"},
        nested_output_paths={"resolved_parameters.feature_cache_dir": "feature_cache"},
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(
        tool(
            experiment_id=experiment_id,
            resolved_parameters={"feature_cache_dir": str(external_cache)},
        )
    )

    assert list(external_cache.iterdir()) == []
    managed_cache = Path(result["feature_cache_dir"])
    assert managed_cache.is_relative_to(tmp_path / "sessions" / experiment_id)
    assert (managed_cache / "cache.marker").read_text() == "managed"


def test_nested_training_write_locations_and_auto_download_are_not_client_controlled(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("bound nested training outputs")["experiment_id"]
    external = tmp_path / "external"
    external.mkdir()
    spec = QsariaToolSpec(
        mcp_name="qsaria_training_inspect_nested_args",
        toolkit_factory=_DummyToolkit,
        method="write_feature_cache",
        summary="bound nested training arguments",
        agent_name="qsaria_training",
        output_paths={"output_dir": "training_output"},
        nested_output_paths={
            "resolved_parameters.disk_offload_dir": "disk_offload",
            "resolved_parameters.feature_cache_dir": "feature_cache",
        },
        nested_blocked_inputs=(
            "resolved_parameters.allow_auto_download",
            "resolved_parameters.checkpoint_dir",
            "resolved_parameters.disk_offload_dir",
            "resolved_parameters.hpopt_save_dir",
        ),
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(
        tool(
            experiment_id=experiment_id,
            resolved_parameters={
                "allow_auto_download": True,
                "checkpoint_dir": str(external / "checkpoint"),
                "disk_offload_dir": str(external / "offload"),
                "feature_cache_dir": str(external / "cache"),
                "hpopt_save_dir": str(external / "hpopt"),
            },
        )
    )

    managed_cache = Path(result["feature_cache_dir"])
    assert managed_cache.is_relative_to(tmp_path / "storage" / "sessions" / experiment_id)
    managed_offload = Path(result["resolved_parameters"]["disk_offload_dir"])
    assert managed_offload.is_relative_to(tmp_path / "storage" / "sessions" / experiment_id)
    assert "allow_auto_download" not in result["resolved_parameters"]
    assert "checkpoint_dir" not in result["resolved_parameters"]
    assert "hpopt_save_dir" not in result["resolved_parameters"]
    assert not any(external.iterdir())


def test_training_bundles_are_isolated_across_concurrent_experiments(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    first_id = manager.create_experiment("first training")["experiment_id"]
    second_id = manager.create_experiment("second training")["experiment_id"]
    external_bundle_dir = tmp_path / "external-bundles"
    external_bundle_dir.mkdir()
    spec = QsariaToolSpec(
        mcp_name="qsaria_training_write_training_bundle",
        toolkit_factory=_DummyToolkit,
        method="write_training_bundle",
        summary="write isolated bundle",
        agent_name="qsaria_training",
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "training_output",
        },
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    async def run_both():
        return await asyncio.gather(
            tool(experiment_id=first_id, bundle_dir=str(external_bundle_dir)),
            tool(experiment_id=second_id, bundle_dir=str(external_bundle_dir)),
        )

    first, second = asyncio.run(run_both())
    first_bundle = Path(first["bundle_file_ref"])
    second_bundle = Path(second["bundle_file_ref"])

    assert first_bundle != second_bundle
    assert first_bundle.is_relative_to(tmp_path / "storage" / "sessions" / first_id)
    assert second_bundle.is_relative_to(tmp_path / "storage" / "sessions" / second_id)
    assert first_bundle.read_text(encoding="utf-8") != second_bundle.read_text(encoding="utf-8")
    assert not any(external_bundle_dir.iterdir())


def test_registry_artifact_maps_validate_values_even_under_arbitrary_keys(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("validate registry artifact values")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_accept_training_summary",
        toolkit_factory=_DummyToolkit,
        method="accept_training_summary",
        summary="validate nested registry artifact paths",
        agent_name="qsaria_registry",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="outside the trusted Qsaria input roots"):
        asyncio.run(
            tool(
                experiment_id=experiment_id,
                training_data_summary={
                    "artifact_sources": {"plot_artifacts": {"arbitrary_key": "/etc/passwd"}}
                },
            )
        )

    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"


def test_registry_artifact_maps_preserve_nested_scope_metadata(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("preserve registry artifact scope")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_accept_training_summary",
        toolkit_factory=_DummyToolkit,
        method="accept_training_summary",
        summary="preserve nested registry artifact metadata",
        agent_name="qsaria_registry",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    result = asyncio.run(
        tool(
            experiment_id=experiment_id,
            training_data_summary={
                "artifact_sources": {
                    "outlier_analysis": {
                        "artifacts": {
                            "selection_annotations": {
                                "activity_cliffs": {"scope": "development_only"},
                                "applicability_domain": {"scope": "train_fit_validation_score"},
                            }
                        }
                    }
                }
            },
        )
    )

    annotations = result["training_data_summary"]["artifact_sources"]["outlier_analysis"][
        "artifacts"
    ]["selection_annotations"]
    assert annotations["activity_cliffs"]["scope"] == "development_only"
    assert annotations["applicability_domain"]["scope"] == "train_fit_validation_score"

    with pytest.raises(MCPToolError, match="non-path artifact metadata value"):
        asyncio.run(
            tool(
                experiment_id=experiment_id,
                training_data_summary={
                    "artifact_sources": {
                        "outlier_analysis": {
                            "artifacts": {
                                "selection_annotations": {
                                    "activity_cliffs": {"scope": "/etc/passwd"}
                                }
                            }
                        }
                    }
                },
            )
        )


def test_registry_feature_columns_source_is_always_validated_as_a_path(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("validate feature source")["experiment_id"]
    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_accept_inference_profile",
        toolkit_factory=_DummyToolkit,
        method="accept_inference_profile",
        summary="validate feature source path",
        agent_name="qsaria_registry",
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="outside the trusted Qsaria input roots"):
        asyncio.run(
            tool(
                experiment_id=experiment_id,
                inference_profile={"feature_columns_source": "/etc/passwd"},
            )
        )

    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"


def test_candidate_manifest_must_be_training_artifact_and_its_paths_are_revalidated(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("validate candidate manifest")["experiment_id"]
    with manager.run(
        experiment_id,
        agent_name="qsaria_training",
        tool_name="qsaria_training_train_qsar_model",
    ) as runtime:
        manifest = Path(
            runtime.artifact_path(
                "catalog_candidates_manifest.json",
                category="training",
            )
        )
        manifest.parent.mkdir(parents=True, exist_ok=True)
        (manifest.parent / "model.pkl").write_bytes(b"model")
        manifest.write_text(
            json.dumps(
                {
                    "candidate_registry_payloads": [
                        {
                            "registry_payload": {
                                "model_id": "hostile",
                                "model_path": str(manifest.parent / "model.pkl"),
                                "task_type": "regression",
                                "training_data_summary": {
                                    "artifact_sources": {
                                        "plot_artifacts": {"arbitrary_key": "/etc/passwd"}
                                    }
                                },
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        runtime.capture_result({"candidate_manifest_path": str(manifest)})

    manifest_artifact = next(
        artifact
        for artifact in manager.list_artifacts(experiment_id)
        if artifact["path"].endswith("/catalog_candidates_manifest.json")
    )
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_training",
            status="completed",
            summary="Training candidate manifest recorded",
            artifact_ids=[manifest_artifact["artifact_id"]],
        ),
    )

    spec = QsariaToolSpec(
        mcp_name="qsaria_registry_accept_candidate_manifest",
        toolkit_factory=_DummyToolkit,
        method="accept_candidate_manifest",
        summary="validate candidate manifest",
        agent_name="qsaria_registry",
        forces={"candidate_registry_payloads": None},
        registered_artifact_inputs={
            "candidate_manifest_path": (
                "catalog_candidates_manifest.json",
                ("qsaria_training_",),
            )
        },
        json_payload_inputs=("candidate_manifest_path",),
    )
    tool = build_qsaria_tool(spec, _DummyToolkit(), manager)

    with pytest.raises(MCPToolError, match="outside the trusted Qsaria input roots"):
        asyncio.run(tool(experiment_id=experiment_id, candidate_manifest_path=str(manifest)))

    assert manager.get_experiment_state(experiment_id)["status"] == "needs_user_input"
    manager.record_handoff(
        experiment_id,
        _handoff(
            experiment_id,
            agent="qsaria_registry",
            status="needs_user_input",
            summary="Candidate manifest contains an untrusted artifact path",
            blockers=["candidate manifest path validation failed"],
        ),
    )

    unregistered = tmp_path / "storage" / "catalog_candidates_manifest.json"
    unregistered.write_text(
        json.dumps({"candidate_registry_payloads": [{"registry_payload": {}}]}),
        encoding="utf-8",
    )
    with pytest.raises(MCPToolError, match="not a registered artifact"):
        asyncio.run(
            tool(
                experiment_id=experiment_id,
                candidate_manifest_path=str(unregistered),
            )
        )


def test_curation_report_ignores_hostile_free_form_payload_and_uses_session_state(tmp_path):
    manager = ExperimentManager(local_root=tmp_path / "storage")
    experiment_id = manager.create_experiment("safe curation report")["experiment_id"]
    with manager.run(
        experiment_id,
        agent_name="qsaria_curation",
        tool_name="qsaria_curation_curate_qsar_dataset",
    ) as runtime:
        curated = Path(runtime.artifact_path("curated.csv", category="curation"))
        manifest = Path(runtime.artifact_path("manifest.json", category="curation"))
        diagnostics = Path(runtime.artifact_path("diagnostics.json", category="curation"))
        for path, content in (
            (curated, "smiles,target\nCCO,1\n"),
            (manifest, '{"files": {}}\n'),
            (diagnostics, '{"rows": 1}\n'),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        safe_result = {
            "status": "ready",
            "ready_for_qsar": True,
            "dataset_id": "safe_dataset",
            "curated_dataset_path": str(curated),
            "curation_artifacts": {
                "manifest_json": str(manifest),
                "identity_diagnostics_json": str(diagnostics),
            },
        }
        runtime.session_state["qsar_curation"] = {"last_result": safe_result}
        runtime.capture_result(safe_result)

    hostile_manifest = tmp_path / "hostile_manifest.json"
    hostile_manifest.write_text('{"must_remain": true}\n')
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("must not enter bundle")
    hostile_payload = {
        "dataset_id": "hostile",
        "curated_dataset_path": str(secret),
        "curation_artifacts": {
            "manifest_json": str(hostile_manifest),
            "arbitrary_blob": str(secret),
        },
    }
    spec = next(spec for spec in SPECS if spec.mcp_name == "qsaria_curation_write_curation_report")
    tool = build_qsaria_tool(spec, spec.toolkit_factory(), manager)

    assert "curation_result" not in inspect.signature(tool).parameters
    result = asyncio.run(
        tool(
            experiment_id=experiment_id,
            curation_result=hostile_payload,
        )
    )

    assert json.loads(hostile_manifest.read_text()) == {"must_remain": True}
    with zipfile.ZipFile(result["bundle_file_ref"]) as archive:
        assert secret.name not in archive.namelist()
        assert curated.name in archive.namelist()
    assert (
        json.loads(manifest.read_text())["files"]["curation_report_json"]["path"]
        == result["report_path"]
    )


def test_qsaria_toolkit_hub_does_not_import_agent_factories_or_teams():
    repository_root = Path(__file__).resolve().parents[3]
    source = (repository_root / "src/cs_copilot/mcp/qsaria/toolkit_hub.py").read_text(
        encoding="utf-8"
    )
    assert "cs_copilot.agents.factories" not in source
    assert "cs_copilot.agents.teams" not in source


def test_catalog_write_scope_rebinds_every_consumer_after_registry_replacement(tmp_path):
    source = tmp_path / "catalog.json"
    source.write_text('{"schema_version": 2, "models": []}\n', encoding="utf-8")
    canonical = PredictionModelCatalog.load(str(source))
    replacement = PredictionModelCatalog.load(str(source))
    assert type(replacement) is PredictionModelCatalog

    registry = SimpleNamespace(catalog=canonical)
    inference = SimpleNamespace(registry_toolkit=registry)
    ensemble = SimpleNamespace(catalog=canonical)
    benchmark = SimpleNamespace(registry_toolkit=registry)
    hub = ToolkitHub()
    hub._catalog = canonical
    hub._instances.update(
        {
            "registry": registry,
            "inference": inference,
            "ensemble": ensemble,
            "benchmark": benchmark,
        }
    )

    with hub.catalog_scope(access=True, write=True):
        # Reproduce ModelRegistryToolkit.persist_registered_model's final
        # assignment to a plain PredictionModelCatalog.
        registry.catalog = replacement

    assert registry.catalog is canonical
    assert inference.registry_toolkit.catalog is canonical
    assert ensemble.catalog is canonical
    assert benchmark.registry_toolkit.catalog is canonical
    assert type(canonical) is PredictionModelCatalog
