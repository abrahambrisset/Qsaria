"""Detached worker entry point for durable Qsaria operations."""

from __future__ import annotations

import argparse
import os
import threading
import traceback
from pathlib import Path
from typing import Any

from .adapter import classify_qsaria_error, classify_qsaria_result, invoke_qsaria_spec
from .contracts import json_safe, utc_now
from .experiments import ExperimentManager
from .operations import HEARTBEAT_INTERVAL_SECONDS, OperationStore, allowed_operation_specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one detached Qsaria operation.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--operation-id", required=True)
    parser.add_argument("--storage-root", required=True)
    return parser.parse_args()


def _heartbeat(
    store: OperationStore,
    experiment_id: str,
    operation_id: str,
    worker_token: str,
    stop: threading.Event,
) -> None:
    while not stop.wait(HEARTBEAT_INTERVAL_SECONDS):
        try:
            store.update_state(
                experiment_id,
                operation_id,
                worker_token=worker_token,
                updates={"heartbeat_at": utc_now()},
            )
        except Exception:
            traceback.print_exc()
            return


def run_worker(
    *,
    experiment_id: str,
    operation_id: str,
    storage_root: str | os.PathLike[str],
) -> int:
    manager = ExperimentManager(local_root=Path(storage_root))
    store = OperationStore(manager)
    request = store.read_request(experiment_id, operation_id)
    worker_token = str(request["worker_token"])
    role = str(request["role"])
    tool_name = str(request["tool_name"])
    arguments = request.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise ValueError("operation arguments must be a JSON object")
    specs = allowed_operation_specs(role)
    try:
        spec = specs[tool_name]
    except KeyError as exc:
        raise ValueError(f"operation tool is no longer allowed for {role}: {tool_name}") from exc

    store.update_state(
        experiment_id,
        operation_id,
        worker_token=worker_token,
        updates={
            "status": "running",
            "worker_pid": os.getpid(),
            "started_at": utc_now(),
            "heartbeat_at": utc_now(),
        },
    )
    store.append_journal(
        experiment_id,
        operation_id,
        {
            "event": "worker_started",
            "status": "running",
            "worker_pid": os.getpid(),
            "recorded_at": utc_now(),
        },
    )
    stop = threading.Event()
    heartbeat = threading.Thread(
        target=_heartbeat,
        args=(store, experiment_id, operation_id, worker_token, stop),
        daemon=True,
        name=f"qsaria-heartbeat-{operation_id}",
    )
    heartbeat.start()
    try:
        instance = spec.toolkit_factory()
        result: Any = invoke_qsaria_spec(
            spec,
            instance,
            manager,
            {"experiment_id": experiment_id, **arguments},
        )
        status = classify_qsaria_result(result)
        store.write_result(
            experiment_id,
            operation_id,
            {
                "schema_version": "1.0",
                "experiment_id": experiment_id,
                "operation_id": operation_id,
                "tool_name": tool_name,
                "status": status,
                "payload": json_safe(result),
                "recorded_at": utc_now(),
            },
        )
        store.update_state(
            experiment_id,
            operation_id,
            worker_token=worker_token,
            updates={
                "status": status,
                "completed_at": utc_now(),
                "heartbeat_at": utc_now(),
            },
        )
        store.append_journal(
            experiment_id,
            operation_id,
            {"event": "worker_completed", "status": status, "recorded_at": utc_now()},
        )
        return 0 if status == "completed" else 2
    except BaseException as exc:  # noqa: BLE001 - durable terminal evidence is required
        status = classify_qsaria_error(spec, exc)
        error_payload = {
            "schema_version": "1.0",
            "experiment_id": experiment_id,
            "operation_id": operation_id,
            "tool_name": tool_name,
            "status": status,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "recorded_at": utc_now(),
        }
        store.write_error(experiment_id, operation_id, error_payload)
        store.update_state(
            experiment_id,
            operation_id,
            worker_token=worker_token,
            updates={
                "status": status,
                "completed_at": utc_now(),
                "heartbeat_at": utc_now(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        )
        store.append_journal(
            experiment_id,
            operation_id,
            {
                "event": "worker_failed",
                "status": status,
                "error_type": type(exc).__name__,
                "recorded_at": utc_now(),
            },
        )
        traceback.print_exc()
        return 2
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)


def main() -> int:
    args = _parse_args()
    return run_worker(
        experiment_id=args.experiment_id,
        operation_id=args.operation_id,
        storage_root=args.storage_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
