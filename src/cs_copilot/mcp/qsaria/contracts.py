"""Stable, JSON-only contracts for Qsaria experiments exposed through MCP.

The scientific toolkits remain the source of truth for QSAR behaviour.  This
module only defines the small persistence and handoff envelope used by the
external Codex coordinator.
"""

from __future__ import annotations

import hashlib
import math
import re
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

EXPERIMENT_SCHEMA_VERSION: Final = "1.0"
RUNTIME_SCHEMA_VERSION: Final = "1.0"
HANDOFF_SCHEMA_VERSION: Final = "1.0"
REPORT_CONTEXT_SCHEMA_VERSION: Final = "1.0"
PLUGIN_CONTRACT_VERSION: Final = "0.2.0"

EXPERIMENT_ID_RE: Final = re.compile(r"^exp_[A-Za-z0-9][A-Za-z0-9._-]{2,95}$")
HANDOFF_STATUSES: Final = frozenset(
    {
        "completed",
        "partial",
        "retryable_error",
        "terminal_failure",
        "needs_user_input",
    }
)
HANDOFF_EXECUTION_MODES: Final = frozenset(
    {
        "project_agent",
        "coordinator_manual_tools",
    }
)
COMPLETION_STATUSES: Final = frozenset({"completed", "partial", "terminal_failure"})
QSARIA_AGENT_NAMES: Final = frozenset(
    {
        "qsaria_curation",
        "qsaria_training",
        "qsaria_registry",
        "qsaria_inference",
        "qsaria_report",
    }
)

_ARTIFACT_KEY_PARTS = (
    "artifact",
    "path",
    "file_ref",
    "file_path",
    "csv",
    "json",
    "parquet",
    "pickle",
    "pkl",
    "zip",
    "html",
    "markdown",
    "report",
    "plot",
    "image",
)
_MODEL_ID_RE = re.compile(r"(?:^|_)model_ids?$|^registered_model_ids$|^recommended_model_id$")
_PATH_SUFFIX_RE = re.compile(
    r"\.(?:csv|tsv|json|jsonl|parquet|pkl|pickle|joblib|pt|ckpt|zip|md|txt|html|pdf|png|jpg|jpeg|svg|yaml|yml|toml)$",
    re.I,
)


class QsariaExperimentError(RuntimeError):
    """Base exception for the deterministic experiment persistence layer."""


class InvalidExperimentIdError(ValueError, QsariaExperimentError):
    """Raised when an experiment identifier cannot be used as a safe path part."""


class ExperimentNotFoundError(FileNotFoundError, QsariaExperimentError):
    """Raised when an explicitly requested experiment does not exist."""


class ExperimentAlreadyExistsError(FileExistsError, QsariaExperimentError):
    """Raised when creation would replace an existing experiment."""


class InvalidHandoffError(ValueError, QsariaExperimentError):
    """Raised when a sub-agent handoff violates the public contract."""


class InvalidInputPathError(ValueError, QsariaExperimentError):
    """Raised when an MCP input path escapes the configured trusted roots."""


class IncompatibleExperimentSchemaError(ValueError, QsariaExperimentError):
    """Raised before an incompatible persisted state can be rewritten."""


class ExperimentStateTransitionError(QsariaExperimentError):
    """Raised when a tool would silently bypass a durable experiment outcome."""


def utc_now() -> str:
    """Return a stable UTC timestamp suitable for persisted JSON."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def generate_experiment_id() -> str:
    """Generate a readable, collision-resistant and path-safe experiment id."""

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"exp_{stamp}_{uuid.uuid4().hex[:10]}"


def validate_experiment_id(value: str) -> str:
    """Validate and return an experiment id without normalising user input."""

    raw_value = str(value or "")
    experiment_id = raw_value.strip()
    if raw_value != experiment_id:
        raise InvalidExperimentIdError(
            "experiment_id cannot contain leading or trailing whitespace"
        )
    if not EXPERIMENT_ID_RE.fullmatch(experiment_id):
        raise InvalidExperimentIdError(
            "experiment_id must start with 'exp_' and contain only letters, digits, '.', '_' "
            "or '-' (3 to 96 characters after the prefix)"
        )
    return experiment_id


def json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert runtime values to deterministic JSON-compatible structures.

    Toolkits currently place dictionaries, small dataframe previews and Path
    objects in session state.  The conversion is deliberately conservative:
    large/opaque objects become short type markers rather than unsafe pickles.
    """

    if depth > 20:
        return {"_type": type(value).__name__, "_truncated": True}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item, depth=depth + 1) for item in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item(), depth=depth + 1)
        except Exception:  # noqa: BLE001 - optional numpy-like protocol
            pass
    if hasattr(value, "to_dict"):
        try:
            payload = value.to_dict("records")
        except TypeError:
            try:
                payload = value.to_dict()
            except Exception:  # noqa: BLE001 - opaque third-party object
                payload = None
        except Exception:  # noqa: BLE001 - opaque third-party object
            payload = None
        if payload is not None:
            return json_safe(payload, depth=depth + 1)
    return {"_type": type(value).__name__, "_repr": str(value)[:500]}


def new_experiment_state(
    *,
    experiment_id: str,
    user_request: str,
    report_language: str,
    metadata: Mapping[str, Any] | None = None,
    versions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public, user-readable state for a new experiment."""

    now = utc_now()
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": validate_experiment_id(experiment_id),
        "state_revision": 0,
        "created_at": now,
        "updated_at": now,
        "status": "created",
        "phase": "bootstrap",
        "request_summary": str(user_request).strip(),
        "report_language": str(report_language or "fr").strip() or "fr",
        "handoffs": [],
        "artifact_ids": [],
        "model_ids": [],
        "warnings": [],
        "blockers": [],
        "report": None,
        "metadata": json_safe(dict(metadata or {})),
        "versions": json_safe(dict(versions or {})),
    }


def new_runtime_state(experiment_id: str) -> dict[str, Any]:
    """Build the private toolkit/runtime state for a new experiment."""

    now = utc_now()
    return {
        "schema_version": RUNTIME_SCHEMA_VERSION,
        "experiment_id": validate_experiment_id(experiment_id),
        "state_revision": 0,
        "created_at": now,
        "updated_at": now,
        "session_state": {},
        "tool_events": [],
        "reporting_packets": [],
        "persistence_events": [],
        "artifacts": {},
    }


def normalize_handoff(experiment_id: str, handoff: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalise one sub-agent handoff envelope."""

    if not isinstance(handoff, Mapping):
        raise InvalidHandoffError("handoff must be a mapping")
    required_fields = (
        "schema_version",
        "experiment_id",
        "agent",
        "status",
        "summary",
        "facts",
        "artifact_ids",
        "model_ids",
        "warnings",
        "blockers",
        "recommended_next_action",
    )
    missing = [field for field in required_fields if field not in handoff]
    if missing:
        raise InvalidHandoffError("handoff is missing required field(s): " + ", ".join(missing))
    if not isinstance(handoff.get("schema_version"), str):
        raise InvalidHandoffError("handoff schema_version must be a string")
    schema_version = handoff["schema_version"].strip()
    if schema_version != HANDOFF_SCHEMA_VERSION:
        raise InvalidHandoffError(f"handoff schema_version must be {HANDOFF_SCHEMA_VERSION!r}")
    expected_id = validate_experiment_id(experiment_id)
    if not isinstance(handoff.get("experiment_id"), str):
        raise InvalidHandoffError("handoff experiment_id must be a string")
    received_id = handoff["experiment_id"].strip()
    if received_id != expected_id:
        raise InvalidHandoffError(
            f"handoff experiment_id {received_id!r} does not match {expected_id!r}"
        )
    if not isinstance(handoff.get("agent"), str):
        raise InvalidHandoffError("handoff agent must be a string")
    agent = handoff["agent"].strip()
    if agent not in QSARIA_AGENT_NAMES:
        raise InvalidHandoffError(f"unknown Qsaria agent: {agent!r}")
    raw_execution_mode = handoff.get("execution_mode", "project_agent")
    if not isinstance(raw_execution_mode, str):
        raise InvalidHandoffError("handoff execution_mode must be a string")
    execution_mode = raw_execution_mode.strip()
    if execution_mode not in HANDOFF_EXECUTION_MODES:
        raise InvalidHandoffError(
            "execution_mode must be one of " + ", ".join(sorted(HANDOFF_EXECUTION_MODES))
        )
    if not isinstance(handoff.get("status"), str):
        raise InvalidHandoffError("handoff status must be a string")
    status = handoff["status"].strip()
    if status not in HANDOFF_STATUSES:
        raise InvalidHandoffError(f"status must be one of {', '.join(sorted(HANDOFF_STATUSES))}")
    if not isinstance(handoff.get("summary"), str):
        raise InvalidHandoffError("handoff summary must be a string")
    summary = handoff["summary"].strip()
    if not summary:
        raise InvalidHandoffError("handoff summary cannot be empty")
    facts = handoff.get("facts")
    if not isinstance(facts, Mapping):
        raise InvalidHandoffError("handoff facts must be a mapping")
    list_fields = ("artifact_ids", "model_ids", "warnings", "blockers")
    for field in list_fields:
        value = handoff.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise InvalidHandoffError(f"handoff {field} must be an array of strings")
    if not isinstance(handoff.get("recommended_next_action"), str):
        raise InvalidHandoffError("handoff recommended_next_action must be a string")
    blockers = _unique_strings(handoff["blockers"])
    if status in {"retryable_error", "terminal_failure", "needs_user_input"} and not blockers:
        raise InvalidHandoffError(f"handoff status {status!r} requires at least one blocker")

    return {
        "schema_version": schema_version,
        "experiment_id": expected_id,
        "agent": agent,
        "execution_mode": execution_mode,
        "status": status,
        "summary": summary,
        "facts": json_safe(facts),
        "artifact_ids": _unique_strings(handoff["artifact_ids"]),
        "model_ids": _unique_strings(handoff["model_ids"]),
        "warnings": _unique_strings(handoff["warnings"]),
        "blockers": blockers,
        "recommended_next_action": handoff["recommended_next_action"].strip(),
        "recorded_at": utc_now(),
    }


def extract_model_ids(payload: Any) -> list[str]:
    """Recursively extract model identifiers from ordinary toolkit results."""

    found: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                name = str(child_key).lower()
                if _MODEL_ID_RE.search(name):
                    found.extend(_as_string_values(child))
                visit(child, name)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key)

    visit(payload)
    return _unique_strings(found)


def extract_artifact_paths(payload: Any) -> list[str]:
    """Recursively extract path-like artifact references from toolkit results."""

    found: list[str] = []

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                name = str(child_key).lower()
                if _artifact_key(name):
                    found.extend(
                        item for item in _as_string_values(child) if _looks_like_path(item)
                    )
                visit(child, name)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for child in value:
                visit(child, key)
        elif key and _artifact_key(key) and isinstance(value, (str, Path)):
            text = str(value).strip()
            if _looks_like_path(text):
                found.append(text)

    visit(payload)
    return _unique_strings(found)


def artifact_id_for(experiment_id: str, path: str) -> str:
    """Return a stable public artifact id for one experiment/path pair."""

    digest = hashlib.sha256(f"{validate_experiment_id(experiment_id)}\0{path}".encode()).hexdigest()
    return f"art_{digest[:16]}"


def _artifact_key(key: str) -> bool:
    return any(part in key for part in _ARTIFACT_KEY_PARTS)


def _looks_like_path(value: str) -> bool:
    text = str(value).strip()
    if not text or "\n" in text or len(text) > 4096:
        return False
    if text.startswith(("s3://", "file://", "/", "./", ".files/", "data/", "workflows/")):
        return True
    return bool(_PATH_SUFFIX_RE.search(text))


def _as_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        return [str(value).strip()] if str(value).strip() else []
    if isinstance(value, Mapping):
        values: list[str] = []
        for child in value.values():
            values.extend(_as_string_values(child))
        return values
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        values = []
        for child in value:
            values.extend(_as_string_values(child))
        return values
    return []


def _unique_strings(values: Any) -> list[str]:
    if isinstance(values, (str, Path)):
        values = [values]
    result: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


__all__ = [
    "COMPLETION_STATUSES",
    "EXPERIMENT_SCHEMA_VERSION",
    "HANDOFF_SCHEMA_VERSION",
    "HANDOFF_EXECUTION_MODES",
    "HANDOFF_STATUSES",
    "QSARIA_AGENT_NAMES",
    "REPORT_CONTEXT_SCHEMA_VERSION",
    "RUNTIME_SCHEMA_VERSION",
    "ExperimentAlreadyExistsError",
    "ExperimentNotFoundError",
    "InvalidExperimentIdError",
    "InvalidHandoffError",
    "QsariaExperimentError",
    "artifact_id_for",
    "extract_artifact_paths",
    "extract_model_ids",
    "generate_experiment_id",
    "json_safe",
    "new_experiment_state",
    "new_runtime_state",
    "normalize_handoff",
    "utc_now",
    "validate_experiment_id",
]
