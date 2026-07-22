"""Durability, role isolation, and result tests for Qsaria operations."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from cs_copilot.mcp.qsaria.experiments import ExperimentManager
from cs_copilot.mcp.qsaria.operations import (
    OperationManager,
    OperationStore,
    allowed_operation_specs,
)
from cs_copilot.mcp.qsaria.toolkit_hub import ToolkitHub, configured_catalog_path


@pytest.fixture
def manager(tmp_path):
    value = ExperimentManager(local_root=tmp_path / "storage")
    value.create_experiment("Detached operation test", experiment_id="exp_operations_123")
    return value


def test_allowed_operations_are_exactly_role_bound() -> None:
    curation = set(allowed_operation_specs("curation"))
    training = set(allowed_operation_specs("training"))
    registry = set(allowed_operation_specs("registry"))
    inference = set(allowed_operation_specs("inference"))

    assert curation == {
        "qsaria_curation_curate_qsar_dataset",
        "qsaria_curation_write_curation_report",
    }
    assert training
    assert registry
    assert inference
    assert not (curation & training & registry & inference)
    assert all(name.startswith("qsaria_inference_") for name in inference)
    with pytest.raises(ValueError, match="unknown Qsaria operation role"):
        allowed_operation_specs("report")


def test_qsaria_catalog_path_is_configurable_without_changing_generic_default(
    tmp_path, monkeypatch
) -> None:
    shared = tmp_path / "data" / "model_assets" / "catalog" / "qsaria_model_catalog.json"
    monkeypatch.setenv("QSARIA_MODEL_CATALOG_PATH", str(shared))

    assert configured_catalog_path() == shared.resolve()
    assert ToolkitHub()._shared_catalog().source_path == shared.resolve()

    monkeypatch.delenv("QSARIA_MODEL_CATALOG_PATH")
    assert configured_catalog_path() is None


def test_store_persists_atomic_request_state_and_journal(manager) -> None:
    store = OperationStore(manager)
    state = store.create(
        experiment_id="exp_operations_123",
        operation_id="op_20260722T120000Z_0123456789ab",
        role="curation",
        tool_name="qsaria_curation_write_curation_report",
        arguments={},
        worker_token="token",
    )

    root = store.root("exp_operations_123", state["operation_id"])
    assert json.loads((root / "request.json").read_text())["role"] == "curation"
    assert json.loads((root / "state.json").read_text())["status"] == "queued"
    journal = json.loads((root / "journal.json").read_text())
    assert journal["events"][-1]["event"] == "queued"
    assert not list(root.glob("*.tmp"))


def test_stale_missing_worker_becomes_retryable_error(manager) -> None:
    store = OperationStore(manager)
    operation_id = "op_20260722T120001Z_0123456789ab"
    store.create(
        experiment_id="exp_operations_123",
        operation_id=operation_id,
        role="curation",
        tool_name="qsaria_curation_write_curation_report",
        arguments={},
        worker_token="token",
    )
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store.update_state(
        "exp_operations_123",
        operation_id,
        worker_token="token",
        updates={"status": "running", "heartbeat_at": stale, "worker_pid": 2_000_000_000},
    )

    state = store.read_state("exp_operations_123", operation_id)

    assert state["status"] == "retryable_error"
    assert state["error_type"] == "WorkerInterruptedError"
    assert "worker_token" not in state
    result = store.result("exp_operations_123", operation_id)
    assert result["ready"] is True
    assert result["result"]["error_type"] == "WorkerInterruptedError"


def test_stale_state_recovers_an_already_persisted_terminal_result(manager) -> None:
    store = OperationStore(manager)
    operation_id = "op_20260722T120002Z_0123456789ab"
    store.create(
        experiment_id="exp_operations_123",
        operation_id=operation_id,
        role="curation",
        tool_name="qsaria_curation_write_curation_report",
        arguments={},
        worker_token="token",
    )
    store.write_result(
        "exp_operations_123",
        operation_id,
        {
            "schema_version": "1.0",
            "status": "completed",
            "payload": {"status": "ok"},
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    stale = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
    store.update_state(
        "exp_operations_123",
        operation_id,
        worker_token="token",
        updates={"status": "running", "heartbeat_at": stale, "worker_pid": 2_000_000_000},
    )

    state = store.read_state("exp_operations_123", operation_id)

    assert state["status"] == "completed"
    assert store.result("exp_operations_123", operation_id)["result"]["payload"] == {"status": "ok"}


def test_start_rejects_cross_role_and_invalid_arguments(manager) -> None:
    operations = OperationManager(manager)
    with pytest.raises(ValueError, match="not an allowed curation operation"):
        operations.start(
            role="curation",
            experiment_id="exp_operations_123",
            operation="qsaria_training_train_lightgbm_model",
            arguments={},
        )
    with pytest.raises(ValueError, match="arguments must not contain experiment_id"):
        operations.start(
            role="curation",
            experiment_id="exp_operations_123",
            operation="qsaria_curation_curate_qsar_dataset",
            arguments={"experiment_id": "exp_other_123"},
        )


def test_start_validates_typed_training_request_before_launch(manager) -> None:
    operations = OperationManager(manager)
    with pytest.raises(ValueError, match="invalid arguments"):
        operations.start(
            role="training",
            experiment_id="exp_operations_123",
            operation="qsaria_training_train_lightgbm_model",
            arguments={
                "train_csv": "missing.csv",
                "request": {
                    "schema_version": "2.0",
                    "task_type": "regression",
                    "smiles_column": "smiles",
                    "target_columns": [],
                },
            },
        )
    assert operations.list("exp_operations_123") == []


def test_detached_curation_survives_start_call_and_returns_canonical_result(manager) -> None:
    dataset = manager.local_storage_root / "science_input.csv"
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_text(
        "smiles,pIC50\n" "CCO,5.1\n" "CCN,5.4\n" "CCC,5.2\n" "c1ccccc1,6.1\n" "CC(=O)O,4.9\n",
        encoding="utf-8",
    )
    operations = OperationManager(manager)
    started_at = time.monotonic()
    queued = operations.start(
        role="curation",
        experiment_id="exp_operations_123",
        operation="qsaria_curation_curate_qsar_dataset",
        arguments={
            "dataset_path": str(dataset),
            "task_type": "regression",
            "smiles_column": "smiles",
            "target_columns": ["pIC50"],
        },
    )
    assert time.monotonic() - started_at < 10
    assert queued["status"] == "queued"
    operation_id = queued["operation_id"]
    deadline = time.monotonic() + 60
    state = queued
    while state["status"] in {"queued", "running"} and time.monotonic() < deadline:
        time.sleep(0.2)
        state = operations.get_state("exp_operations_123", operation_id)

    assert state["status"] == "completed", state
    result = operations.get_result("exp_operations_123", operation_id)
    assert result["ready"] is True
    assert result["result"]["status"] == "completed"
    assert result["result"]["payload"]["curated_dataset_path"].endswith("curated_dataset.csv")
    restarted = OperationManager(ExperimentManager(local_root=manager.local_storage_root))
    assert restarted.get_result("exp_operations_123", operation_id) == result


def test_two_detached_experiments_remain_isolated(manager) -> None:
    second_id = "exp_operations_456"
    manager.create_experiment("Second detached operation", experiment_id=second_id)
    operations = OperationManager(manager)
    operation_ids: dict[str, str] = {}
    for experiment_id, target in (("exp_operations_123", 5.1), (second_id, 7.2)):
        dataset = manager.local_storage_root / f"{experiment_id}.csv"
        dataset.write_text(
            "smiles,pIC50\n"
            f"CCO,{target}\n"
            f"CCN,{target + 0.1}\n"
            f"CCC,{target + 0.2}\n"
            f"c1ccccc1,{target + 0.3}\n",
            encoding="utf-8",
        )
        receipt = operations.start(
            role="curation",
            experiment_id=experiment_id,
            operation="qsaria_curation_curate_qsar_dataset",
            arguments={
                "dataset_path": str(dataset),
                "task_type": "regression",
                "smiles_column": "smiles",
                "target_columns": ["pIC50"],
            },
        )
        operation_ids[experiment_id] = receipt["operation_id"]

    deadline = time.monotonic() + 60
    states: dict[str, dict] = {}
    while time.monotonic() < deadline:
        states = {
            experiment_id: operations.get_state(experiment_id, operation_id)
            for experiment_id, operation_id in operation_ids.items()
        }
        if all(state["status"] not in {"queued", "running"} for state in states.values()):
            break
        time.sleep(0.2)

    assert {state["status"] for state in states.values()} == {"completed"}
    assert len(set(operation_ids.values())) == 2
    for experiment_id, operation_id in operation_ids.items():
        result = operations.get_result(experiment_id, operation_id)
        path = result["result"]["payload"]["curated_dataset_path"]
        assert f"sessions/{experiment_id}/" in path
