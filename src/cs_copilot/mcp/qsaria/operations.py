"""Durable detached operations for short-timeout MCP clients.

Claude Science 0.1.21 limits one MCP tool call to 60 seconds.  This module
adds a local start/poll lifecycle without changing the synchronous scientific
tools used by Codex, Claude Code, or the Qsaria UI.  Workers execute the same
audited :class:`QsariaToolSpec` through :func:`invoke_qsaria_spec`.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal

from pydantic import BaseModel, TypeAdapter

from .adapter import QsariaToolSpec, qsaria_public_signature
from .contracts import json_safe, utc_now, validate_experiment_id

if TYPE_CHECKING:
    from .experiments import ExperimentManager

OPERATION_SCHEMA_VERSION = "1.0"
OPERATIONS_REL_PATH = "qsaria/operations"
HEARTBEAT_INTERVAL_SECONDS = 12.0
STALE_HEARTBEAT_SECONDS = 90.0
PUBLIC_OPERATION_STATUSES = frozenset(
    {
        "queued",
        "running",
        "completed",
        "retryable_error",
        "terminal_failure",
        "needs_user_input",
    }
)
TERMINAL_OPERATION_STATUSES = frozenset(
    {"completed", "retryable_error", "terminal_failure", "needs_user_input"}
)
_ROLE_AGENT_NAMES = {
    "curation": "qsaria_curation",
    "training": "qsaria_training",
    "registry": "qsaria_registry",
    "inference": "qsaria_inference",
}


def _operation_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"op_{stamp}_{uuid.uuid4().hex[:12]}"


def _validate_operation_id(operation_id: str) -> str:
    value = str(operation_id or "").strip()
    parts = value.split("_")
    if (
        len(parts) != 3
        or parts[0] != "op"
        or len(parts[1]) != 16
        or not parts[1].endswith("Z")
        or len(parts[2]) != 12
        or not all(character in "0123456789abcdef" for character in parts[2])
    ):
        raise ValueError(f"invalid Qsaria operation_id: {operation_id!r}")
    return value


@contextmanager
def _operation_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    process_lock = _process_lock(str(path.resolve(strict=False)))
    with process_lock:
        with path.open("a+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows process-only fallback
                fcntl = None
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


_PROCESS_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


def _process_lock(key: str) -> threading.RLock:
    with _PROCESS_LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return payload


def _pid_alive(pid: Any) -> bool:
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _age_seconds(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds())


def _validated_json_arguments(
    signature: Any,
    *,
    experiment_id: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate JSON arguments against the underlying synchronous signature."""

    bound = signature.bind(experiment_id=experiment_id, **dict(arguments))
    validated: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        if name == "experiment_id":
            continue
        annotation = signature.parameters[name].annotation
        if annotation is not signature.empty:
            value = TypeAdapter(annotation).validate_python(value)
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        validated[name] = json_safe(value)
    return validated


class OperationStore:
    """Atomic local operation manifests scoped to one existing experiment."""

    def __init__(self, manager: "ExperimentManager") -> None:
        self.manager = manager

    def root(self, experiment_id: str, operation_id: str | None = None) -> Path:
        resolved_id = validate_experiment_id(experiment_id)
        relative = OPERATIONS_REL_PATH
        if operation_id is not None:
            relative = f"{relative}/{_validate_operation_id(operation_id)}"
        return Path(self.manager.resolve_artifact_path(resolved_id, relative))

    def create(
        self,
        *,
        experiment_id: str,
        operation_id: str,
        role: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        worker_token: str,
    ) -> dict[str, Any]:
        operation_root = self.root(experiment_id, operation_id)
        operation_root.mkdir(parents=True, exist_ok=False)
        now = utc_now()
        request = {
            "schema_version": OPERATION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "operation_id": operation_id,
            "role": role,
            "tool_name": tool_name,
            "arguments": json_safe(arguments),
            "worker_token": worker_token,
            "created_at": now,
        }
        state = {
            "schema_version": OPERATION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "operation_id": operation_id,
            "role": role,
            "tool_name": tool_name,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "heartbeat_at": None,
            "worker_pid": None,
            "worker_token": worker_token,
        }
        _atomic_json(operation_root / "request.json", request)
        _atomic_json(operation_root / "state.json", state)
        self.append_journal(
            experiment_id,
            operation_id,
            {"event": "queued", "status": "queued", "recorded_at": now},
        )
        return state

    def read_request(self, experiment_id: str, operation_id: str) -> dict[str, Any]:
        return _read_json(self.root(experiment_id, operation_id) / "request.json")

    def read_state(self, experiment_id: str, operation_id: str) -> dict[str, Any]:
        operation_root = self.root(experiment_id, operation_id)
        state_path = operation_root / "state.json"
        worker_lost = False
        recovered_status: str | None = None
        with _operation_lock(operation_root / ".operation.lock"):
            state = _read_json(state_path)
            status = str(state.get("status") or "")
            age = _age_seconds(state.get("heartbeat_at") or state.get("updated_at"))
            if (
                status in {"queued", "running"}
                and age is not None
                and age > STALE_HEARTBEAT_SECONDS
                and not _pid_alive(state.get("worker_pid"))
            ):
                now = utc_now()
                persisted_terminal: dict[str, Any] | None = None
                for filename in ("result.json", "error.json"):
                    candidate = operation_root / filename
                    if not candidate.is_file():
                        continue
                    payload = _read_json(candidate)
                    if payload.get("status") in TERMINAL_OPERATION_STATUSES:
                        persisted_terminal = payload
                        break
                if persisted_terminal is not None:
                    recovered_status = str(persisted_terminal["status"])
                    state.update(
                        {
                            "status": recovered_status,
                            "updated_at": now,
                            "completed_at": persisted_terminal.get("recorded_at") or now,
                        }
                    )
                    for key in ("error", "error_type"):
                        if persisted_terminal.get(key) is not None:
                            state[key] = persisted_terminal[key]
                else:
                    state.update(
                        {
                            "status": "retryable_error",
                            "updated_at": now,
                            "completed_at": now,
                            "error": (
                                "Detached Qsaria worker disappeared after its heartbeat became "
                                "stale"
                            ),
                            "error_type": "WorkerInterruptedError",
                        }
                    )
                    _atomic_json(
                        operation_root / "error.json",
                        {
                            "schema_version": OPERATION_SCHEMA_VERSION,
                            "experiment_id": experiment_id,
                            "operation_id": operation_id,
                            "tool_name": state["tool_name"],
                            "status": "retryable_error",
                            "error": state["error"],
                            "error_type": "WorkerInterruptedError",
                            "recorded_at": state["completed_at"],
                        },
                    )
                    worker_lost = True
                _atomic_json(state_path, state)
            normalized = json_safe(state)
            normalized.pop("worker_token", None)
        if recovered_status is not None:
            self.append_journal(
                experiment_id,
                operation_id,
                {
                    "event": "state_recovered_from_terminal_payload",
                    "status": recovered_status,
                    "recorded_at": state["completed_at"],
                },
            )
        if worker_lost:
            self.append_journal(
                experiment_id,
                operation_id,
                {
                    "event": "worker_lost",
                    "status": "retryable_error",
                    "recorded_at": state["completed_at"],
                },
            )
        return normalized

    def update_state(
        self,
        experiment_id: str,
        operation_id: str,
        *,
        worker_token: str,
        updates: Mapping[str, Any],
    ) -> dict[str, Any]:
        operation_root = self.root(experiment_id, operation_id)
        state_path = operation_root / "state.json"
        with _operation_lock(operation_root / ".operation.lock"):
            state = _read_json(state_path)
            if state.get("worker_token") != worker_token:
                raise PermissionError("operation worker token does not match")
            current_status = str(state.get("status") or "")
            requested_status = str(updates.get("status") or current_status)
            if current_status in TERMINAL_OPERATION_STATUSES and requested_status != current_status:
                return json_safe(state)
            if requested_status not in PUBLIC_OPERATION_STATUSES:
                raise ValueError(f"invalid Qsaria operation status: {requested_status!r}")
            state.update(json_safe(updates))
            state["updated_at"] = utc_now()
            _atomic_json(state_path, state)
            return json_safe(state)

    def write_result(
        self,
        experiment_id: str,
        operation_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        _atomic_json(self.root(experiment_id, operation_id) / "result.json", payload)

    def write_error(
        self,
        experiment_id: str,
        operation_id: str,
        payload: Mapping[str, Any],
    ) -> None:
        _atomic_json(self.root(experiment_id, operation_id) / "error.json", payload)

    def append_journal(
        self,
        experiment_id: str,
        operation_id: str,
        event: Mapping[str, Any],
    ) -> None:
        """Append one lifecycle event through an atomic journal replacement."""

        operation_root = self.root(experiment_id, operation_id)
        journal_path = operation_root / "journal.json"
        with _operation_lock(operation_root / ".journal.lock"):
            if journal_path.is_file():
                document = _read_json(journal_path)
                payload = document.get("events")
                if not isinstance(payload, list):
                    raise ValueError(f"expected events array at {journal_path}")
            else:
                payload = []
            payload.append(json_safe(event))
            _atomic_json(journal_path, {"events": payload})

    def list(self, experiment_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        if not 1 <= int(limit) <= 200:
            raise ValueError("limit must be between 1 and 200")
        operations_root = self.root(experiment_id)
        if not operations_root.exists():
            return []
        states: list[dict[str, Any]] = []
        for operation_root in operations_root.iterdir():
            if not operation_root.is_dir():
                continue
            try:
                states.append(self.read_state(experiment_id, operation_root.name))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        states.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
        return states[: int(limit)]

    def result(self, experiment_id: str, operation_id: str) -> dict[str, Any]:
        state = self.read_state(experiment_id, operation_id)
        if state.get("status") not in TERMINAL_OPERATION_STATUSES:
            return {"ready": False, "state": state}
        operation_root = self.root(experiment_id, operation_id)
        result_path = operation_root / "result.json"
        error_path = operation_root / "error.json"
        payload: dict[str, Any] | None = None
        if result_path.is_file():
            payload = _read_json(result_path)
        elif error_path.is_file():
            payload = _read_json(error_path)
        return {"ready": True, "state": state, "result": payload}


def allowed_operation_specs(role: str) -> dict[str, QsariaToolSpec]:
    """Return experiment-bound scientific tools owned by one Qsaria role."""

    try:
        agent_name = _ROLE_AGENT_NAMES[role]
    except KeyError as exc:
        raise ValueError(f"unknown Qsaria operation role: {role!r}") from exc
    from cs_copilot.mcp.tool_specs.qsaria import scientific_specs

    return {
        spec.mcp_name: spec
        for spec in scientific_specs()
        if spec.agent_name == agent_name and spec.requires_experiment
    }


class OperationManager:
    """Validate, launch and inspect detached Qsaria scientific operations."""

    def __init__(self, manager: "ExperimentManager") -> None:
        self.manager = manager
        self.store = OperationStore(manager)

    def start(
        self,
        *,
        role: Literal["curation", "training", "registry", "inference"],
        experiment_id: str,
        operation: str,
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        resolved_id = validate_experiment_id(experiment_id)
        self.manager.get_experiment_state(resolved_id)
        specs = allowed_operation_specs(role)
        try:
            spec = specs[str(operation)]
        except KeyError as exc:
            allowed = ", ".join(sorted(specs))
            raise ValueError(
                f"{operation!r} is not an allowed {role} operation; allowed: {allowed}"
            ) from exc
        call_arguments = dict(arguments or {})
        if "experiment_id" in call_arguments:
            raise ValueError("arguments must not contain experiment_id")
        instance = spec.toolkit_factory()
        signature = qsaria_public_signature(spec, instance)
        try:
            call_arguments = _validated_json_arguments(
                signature,
                experiment_id=resolved_id,
                arguments=call_arguments,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid arguments for {operation}: {exc}") from exc

        operation_id = _operation_id()
        worker_token = uuid.uuid4().hex
        state = self.store.create(
            experiment_id=resolved_id,
            operation_id=operation_id,
            role=role,
            tool_name=spec.mcp_name,
            arguments=call_arguments,
            worker_token=worker_token,
        )
        operation_root = self.store.root(resolved_id, operation_id)
        log_path = operation_root / "worker.log"
        command = [
            sys.executable,
            "-m",
            "cs_copilot.mcp.qsaria.operation_worker",
            "--experiment-id",
            resolved_id,
            "--operation-id",
            operation_id,
            "--storage-root",
            str(self.manager.local_storage_root),
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            with log_path.open("ab", buffering=0) as log_handle:
                process = subprocess.Popen(  # noqa: S603 - fixed interpreter/module command
                    command,
                    cwd=Path.cwd(),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    start_new_session=True,
                )
        except Exception as exc:
            now = utc_now()
            self.store.write_error(
                resolved_id,
                operation_id,
                {"error": str(exc), "error_type": type(exc).__name__},
            )
            state = self.store.update_state(
                resolved_id,
                operation_id,
                worker_token=worker_token,
                updates={
                    "status": "retryable_error",
                    "completed_at": now,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )
            self.store.append_journal(
                resolved_id,
                operation_id,
                {
                    "event": "worker_launch_failed",
                    "status": "retryable_error",
                    "error_type": type(exc).__name__,
                    "recorded_at": now,
                },
            )
            return state

        state = self.store.update_state(
            resolved_id,
            operation_id,
            worker_token=worker_token,
            updates={"worker_pid": process.pid, "worker_started_at": utc_now()},
        )
        self.store.append_journal(
            resolved_id,
            operation_id,
            {
                "event": "worker_launched",
                "status": state["status"],
                "worker_pid": process.pid,
                "recorded_at": utc_now(),
            },
        )
        return {
            "schema_version": OPERATION_SCHEMA_VERSION,
            "experiment_id": resolved_id,
            "operation_id": operation_id,
            "role": role,
            "tool_name": spec.mcp_name,
            "status": "queued",
            "created_at": state["created_at"],
            "worker_pid": process.pid,
        }

    def list(self, experiment_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        resolved_id = validate_experiment_id(experiment_id)
        self.manager.get_experiment_state(resolved_id)
        return self.store.list(resolved_id, limit=limit)

    def get_state(self, experiment_id: str, operation_id: str) -> dict[str, Any]:
        resolved_id = validate_experiment_id(experiment_id)
        self.manager.get_experiment_state(resolved_id)
        return self.store.read_state(resolved_id, operation_id)

    def get_result(self, experiment_id: str, operation_id: str) -> dict[str, Any]:
        resolved_id = validate_experiment_id(experiment_id)
        self.manager.get_experiment_state(resolved_id)
        return self.store.result(resolved_id, operation_id)


__all__ = [
    "HEARTBEAT_INTERVAL_SECONDS",
    "OPERATION_SCHEMA_VERSION",
    "OPERATIONS_REL_PATH",
    "OperationManager",
    "OperationStore",
    "PUBLIC_OPERATION_STATUSES",
    "STALE_HEARTBEAT_SECONDS",
    "TERMINAL_OPERATION_STATUSES",
    "allowed_operation_specs",
]
