"""Persistent, model-free experiment lifecycle for the Qsaria MCP profile.

The manager deliberately binds storage and MCP contexts with reversible
``ContextVar`` tokens.  It never calls ``S3.set_session_prefix`` because that
legacy helper also mutates a process-wide fallback and would allow concurrent
Codex tasks to leak into one another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import os
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.tools.prediction.qsar_contracts import TRAINING_CONTRACT_VERSION

from .contracts import (
    COMPLETION_STATUSES,
    EXPERIMENT_SCHEMA_VERSION,
    EXTERNAL_COORDINATOR_CONTRACT,
    HANDOFF_SCHEMA_VERSION,
    PLUGIN_CONTRACT_VERSION,
    QSARIA_AGENT_NAMES,
    REPORT_CONTEXT_SCHEMA_VERSION,
    RUNTIME_SCHEMA_VERSION,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentStateTransitionError,
    IncompatibleExperimentSchemaError,
    InvalidHandoffError,
    InvalidInputPathError,
    artifact_id_for,
    extract_artifact_paths,
    extract_model_ids,
    generate_experiment_id,
    json_safe,
    new_experiment_state,
    new_runtime_state,
    normalize_handoff,
    utc_now,
    validate_experiment_id,
)

logger = logging.getLogger(__name__)

PUBLIC_STATE_REL_PATH = "qsaria/experiment_state.json"
RUNTIME_STATE_REL_PATH = "qsaria/runtime_state.json"
EXPERIMENTS_PREFIX = "sessions"
_MAX_TOOL_EVENTS = 1000
_MAX_PREVIEW_BYTES = 1024 * 1024
_GLOBAL_MODEL_ARTIFACT_PREFIX = "model_assets_internal"
_DEFAULT_PERSISTENCE_POLICY = "catalog"
_PERSISTENCE_POLICIES = frozenset({_DEFAULT_PERSISTENCE_POLICY, "session_only"})
_BATCH_PERSISTENCE_TOOL = "qsaria_registry_register_and_persist_candidates"
_REGISTER_MODEL_TOOL = "qsaria_registry_register_model"
_PERSIST_MODEL_TOOL = "qsaria_registry_persist_registered_model"

_LOCKS_GUARD = threading.Lock()
_PROCESS_LOCKS: dict[str, threading.RLock] = {}


def _process_lock(key: str) -> threading.RLock:
    with _LOCKS_GUARD:
        return _PROCESS_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Take an advisory cross-process lock on local Unix filesystems."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows fallback is process-only
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@dataclass
class ExperimentRuntime:
    """One bound toolkit invocation inside a persistent experiment."""

    manager: "ExperimentManager"
    experiment_id: str
    agent_name: str
    tool_name: str
    context: MCPAgentContext
    public_state: dict[str, Any]
    runtime_state: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started_at: str = field(default_factory=utc_now)
    _started_perf: float = field(default_factory=perf_counter, repr=False)
    _result: Any = field(default=None, repr=False)
    _error: str | None = field(default=None, repr=False)
    _outcome_status: str | None = field(default=None, repr=False)
    _input_model_ids: list[str] = field(default_factory=list, repr=False)
    _entry_status: str = field(default="created", repr=False)
    _automatic_retry: bool = field(default=False, repr=False)

    @property
    def session_state(self) -> dict[str, Any]:
        """Mutable toolkit session state (the same object as on ``context``)."""

        return self.context.session_state

    def capture_result(
        self,
        result: Any,
        *,
        status: Literal[
            "success",
            "retryable_error",
            "terminal_failure",
            "needs_user_input",
        ] = "success",
    ) -> Any:
        """Capture a result for artifact/model extraction and manifesting."""

        self._result = result
        self._outcome_status = status
        return result

    def capture_error(
        self,
        error: BaseException | str,
        *,
        status: Literal["retryable_error", "terminal_failure", "needs_user_input"] = (
            "terminal_failure"
        ),
    ) -> None:
        """Classify an invocation failure before it is re-raised to MCP."""

        self._error = str(error)
        self._outcome_status = status

    def capture_inputs(self, payload: Any) -> None:
        """Capture validated model identifiers before toolkit execution."""

        self._input_model_ids = extract_model_ids(payload)

    def artifact_rel_path(self, filename: str, *, category: str = "artifacts") -> str:
        """Return a deterministic session-relative output path for this call."""

        from cs_copilot.storage.layout import sanitize_path_part

        safe_category = sanitize_path_part(category, default="artifacts")
        safe_tool = sanitize_path_part(self.tool_name, default="tool")
        safe_filename = sanitize_path_part(filename, default="artifact")
        return PurePosixPath(
            "workflows",
            safe_category,
            safe_tool,
            self.call_id,
            safe_filename,
        ).as_posix()

    def artifact_path(self, filename: str, *, category: str = "artifacts") -> str:
        """Resolve :meth:`artifact_rel_path` through the configured storage backend."""

        return self.manager.resolve_artifact_path(
            self.experiment_id,
            self.artifact_rel_path(filename, category=category),
        )

    @property
    def duration_ms(self) -> float:
        return round((perf_counter() - self._started_perf) * 1000, 3)


class ExperimentManager:
    """Create, bind, inspect and persist Qsaria experiments.

    Parameters
    ----------
    local_root:
        Optional explicit local storage root, primarily useful for isolated
        tests.  When omitted the configured ``CS_COPILOT_STORAGE_ROOT`` / S3
        backend is used.
    """

    def __init__(self, local_root: str | os.PathLike[str] | None = None) -> None:
        self._explicit_local_root = Path(local_root) if local_root is not None else None
        if self._uses_s3() and not self._s3_single_writer_acknowledged():
            raise RuntimeError(
                "Qsaria V1 does not provide a distributed S3 lock. Set "
                "QSARIA_S3_SINGLE_WRITER=true only when this deployment guarantees "
                "a single Qsaria state writer."
            )

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    def bootstrap(self) -> dict[str, Any]:
        """Describe the profile without listing or resuming any experiment."""

        return {
            "status": "ok",
            "profile": "qsaria",
            "llm_policy": "disabled",
            "model": None,
            "auto_resume": False,
            "active_experiment_id": None,
            "experiments_listed": False,
            "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
            "handoff_schema_version": HANDOFF_SCHEMA_VERSION,
            "training_contract_version": TRAINING_CONTRACT_VERSION,
            "contracts": {
                "experiment": EXPERIMENT_SCHEMA_VERSION,
                "runtime_state": RUNTIME_SCHEMA_VERSION,
                "handoff": HANDOFF_SCHEMA_VERSION,
                "report_context": REPORT_CONTEXT_SCHEMA_VERSION,
                "report_facts": "2.0",
                "training": TRAINING_CONTRACT_VERSION,
            },
            "capabilities": {
                "lifecycle": [
                    "create_experiment",
                    "open_experiment",
                    "list_experiments",
                    "get_experiment_state",
                    "list_artifacts",
                    "get_artifact",
                ],
                "coordination": ["record_handoff", "complete_experiment"],
                "reporting": ["build_report_context", "save_report"],
                "scientific_surfaces": [
                    "curation",
                    "training",
                    "registry",
                    "inference",
                    "ensemble",
                    "benchmark",
                    "activity_cliffs",
                ],
                "sub_agents": sorted(
                    [
                        "qsaria_curation",
                        "qsaria_training",
                        "qsaria_registry",
                        "qsaria_inference",
                        "qsaria_report",
                    ]
                ),
            },
            "compatibility": {
                "target": EXTERNAL_COORDINATOR_CONTRACT,
                "coordinator_contract": EXTERNAL_COORDINATOR_CONTRACT,
                "supported_clients": ["codex_v1", "claude_code_v1"],
                "scientific_toolkits": "existing_qsaria_toolkits",
                "typed_training_requests": True,
                "free_training_arguments": False,
                "agno_chainlit_runtime_changed": False,
                "direct_s3_toolkit_outputs": False,
                "experiment_state_backend": "configured_local_or_s3",
                "scientific_artifact_layout": (".files/sessions/<experiment_id>/workflows/..."),
                "model_persistence_default": _DEFAULT_PERSISTENCE_POLICY,
                "model_persistence_opt_out": "session_only",
                "catalog_model_root": "data/model_assets/internal",
                "storage_concurrency": (
                    {
                        "mode": "s3_single_writer",
                        "distributed_lock": False,
                        "single_writer_acknowledged": True,
                    }
                    if self._uses_s3()
                    else {
                        "mode": "local_cross_process_lock",
                        "distributed_lock": True,
                        "single_writer_acknowledged": False,
                    }
                ),
            },
            "error_policy": {
                "retryable_error_max_automatic_retries": 1,
                "terminal_failure_retry": False,
                "needs_user_input_pauses_workflow": True,
                "blocked_failed_external_evaluation": "terminal_failure",
            },
            "evidence_precedence": [
                "structured_artifacts",
                "structured_handoffs",
                "report_narrative",
                "coordinator_interpretation",
            ],
            "versions": self._versions(),
        }

    def normalize_input_path(self, value: str | os.PathLike[str], *, parameter: str) -> str:
        """Resolve one MCP input path inside an explicitly trusted storage root.

        Local inputs are restricted to the server workspace, the configured
        Qsaria storage root, or roots explicitly listed in
        ``QSARIA_MCP_ALLOWED_INPUT_ROOTS``.  This keeps an auto-approved MCP
        tool from becoming a general-purpose filesystem reader.  Explicit S3
        URLs are limited to the configured bucket.
        """

        raw = os.fspath(value).strip()
        if not raw:
            raise InvalidInputPathError(f"{parameter} cannot be empty")

        parsed = urlsplit(raw)
        if parsed.scheme == "s3":
            if not self._uses_s3():
                raise InvalidInputPathError(
                    f"{parameter} uses S3 but the configured storage backend is local"
                )
            from cs_copilot.storage import get_s3_config

            configured_bucket = str(get_s3_config().bucket_name or "")
            if parsed.netloc != configured_bucket:
                raise InvalidInputPathError(
                    f"{parameter} must use the configured S3 bucket {configured_bucket!r}"
                )
            if not parsed.path.strip("/") or ".." in PurePosixPath(parsed.path).parts:
                raise InvalidInputPathError(f"{parameter} contains an unsafe S3 object key")
            return raw

        if parsed.scheme == "file":
            if parsed.netloc not in {"", "localhost"}:
                raise InvalidInputPathError(f"{parameter} cannot use a remote file host")
            local = Path(unquote(parsed.path)).expanduser()
        elif parsed.scheme:
            raise InvalidInputPathError(
                f"{parameter} uses unsupported path scheme {parsed.scheme!r}"
            )
        else:
            local = Path(raw).expanduser()

        candidate = local if local.is_absolute() else Path.cwd() / local
        resolved = candidate.resolve(strict=False)
        if not any(resolved.is_relative_to(root) for root in self._allowed_input_roots()):
            allowed = ", ".join(str(root) for root in self._allowed_input_roots())
            raise InvalidInputPathError(
                f"{parameter} is outside the trusted Qsaria input roots: {allowed}"
            )
        if not resolved.exists():
            raise FileNotFoundError(f"{parameter} does not exist: {resolved}")
        if not os.access(resolved, os.R_OK):
            raise PermissionError(f"{parameter} is not readable: {resolved}")
        return str(resolved)

    def normalize_experiment_input_path(
        self,
        experiment_id: str,
        value: str | os.PathLike[str],
        *,
        runtime_state: Mapping[str, Any],
        parameter: str,
    ) -> str:
        """Resolve a registered relative artifact before generic path validation.

        Public artifact metadata intentionally stores paths relative to the
        experiment root.  Scientific toolkits, however, consume local
        filesystem paths.  Resolve only an exact artifact entry belonging to
        the active experiment; ordinary workspace inputs continue through the
        generic trusted-root validator unchanged.
        """

        resolved_id = validate_experiment_id(experiment_id)
        raw = os.fspath(value).strip()
        if not raw:
            raise InvalidInputPathError(f"{parameter} cannot be empty")

        parsed = urlsplit(raw)
        local = Path(raw).expanduser()
        if not parsed.scheme and not local.is_absolute():
            artifact_path = self._normalise_artifact_path(resolved_id, raw)
            matching = next(
                (
                    entry
                    for entry in (runtime_state.get("artifacts") or {}).values()
                    if isinstance(entry, Mapping) and entry.get("path") == artifact_path
                ),
                None,
            )
            if matching is not None:
                if matching.get("exists") is not True:
                    raise FileNotFoundError(
                        f"{parameter} references unavailable artifact "
                        f"{matching.get('artifact_id') or artifact_path}"
                    )
                candidate = self._local_artifact_path(resolved_id, artifact_path)
                if candidate is None or not candidate.is_file():
                    raise FileNotFoundError(
                        f"{parameter} registered artifact is not available as a local "
                        f"toolkit input: {artifact_path}"
                    )
                return self.normalize_input_path(candidate, parameter=parameter)

        return self.normalize_input_path(value, parameter=parameter)

    def validate_registered_artifact_input(
        self,
        experiment_id: str,
        value: str | os.PathLike[str],
        *,
        runtime_state: Mapping[str, Any],
        parameter: str,
        expected_filename: str,
        source_prefixes: tuple[str, ...],
    ) -> str:
        """Require an input to be an artifact emitted by an approved experiment tool."""

        normalized = self.normalize_input_path(value, parameter=parameter)
        artifact_path = self._normalise_artifact_path(experiment_id, normalized)
        if PurePosixPath(artifact_path).name != expected_filename:
            raise InvalidInputPathError(
                f"{parameter} must reference {expected_filename!r} from this experiment"
            )
        matching = next(
            (
                entry
                for entry in (runtime_state.get("artifacts") or {}).values()
                if isinstance(entry, Mapping) and entry.get("path") == artifact_path
            ),
            None,
        )
        if matching is None:
            raise InvalidInputPathError(
                f"{parameter} is not a registered artifact of experiment {experiment_id}"
            )
        source = str(matching.get("source") or "")
        if not any(source.startswith(prefix) for prefix in source_prefixes):
            allowed = ", ".join(source_prefixes)
            raise InvalidInputPathError(
                f"{parameter} was produced by {source!r}; expected source prefix: {allowed}"
            )
        return normalized

    def resolve_artifact_path(self, experiment_id: str, rel_path: str) -> str:
        """Resolve a safe writable path using the toolkits' local artifact layout.

        Existing Qsaria toolkits write through :class:`pathlib.Path`, including
        when experiment metadata uses S3.  Keep those scientific artifacts in
        the established local ``.files/sessions`` layout; the configured
        storage backend still owns experiment state and report persistence.
        """

        resolved_id = validate_experiment_id(experiment_id)
        if not self._safe_relative(rel_path):
            raise ValueError(f"artifact path must be session-relative: {rel_path!r}")
        return str(self._safe_session_path(resolved_id, rel_path))

    def create_experiment(
        self,
        user_request: str,
        report_language: str = "fr",
        metadata: Mapping[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Create an experiment and both public/private state documents."""

        request = str(user_request or "").strip()
        if not request:
            raise ValueError("user_request cannot be empty")
        metadata_payload = dict(metadata or {})
        persistence_policy = str(
            metadata_payload.get("persistence_policy") or _DEFAULT_PERSISTENCE_POLICY
        ).strip()
        if persistence_policy not in _PERSISTENCE_POLICIES:
            allowed = ", ".join(sorted(_PERSISTENCE_POLICIES))
            raise ValueError(f"metadata.persistence_policy must be one of: {allowed}")
        metadata_payload["persistence_policy"] = persistence_policy
        resolved_id = validate_experiment_id(experiment_id or generate_experiment_id())
        public_state = new_experiment_state(
            experiment_id=resolved_id,
            user_request=request,
            report_language=report_language,
            metadata=metadata_payload,
            versions=self._versions(),
        )
        runtime_state = new_runtime_state(resolved_id)
        with self._locked(resolved_id, create=True):
            if self._state_exists(resolved_id):
                raise ExperimentAlreadyExistsError(f"experiment already exists: {resolved_id}")
            self._save_states(resolved_id, public_state, runtime_state)
        return json_safe(public_state)

    def open_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Explicitly open an experiment; never falls back to a latest session."""

        return self.get_experiment_state(experiment_id)

    def list_experiments(
        self,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List compact experiment summaries only when explicitly called."""

        if not 1 <= int(limit) <= 200:
            raise ValueError("limit must be between 1 and 200")
        summaries: list[dict[str, Any]] = []
        for experiment_id in self._discover_experiment_ids():
            try:
                with self._locked(experiment_id):
                    state, _runtime_state = self._load_states(experiment_id)
            except (ExperimentNotFoundError, json.JSONDecodeError, OSError):
                logger.warning("Skipping unreadable Qsaria experiment %s", experiment_id)
                continue
            if status and state.get("status") != status:
                continue
            summaries.append(
                {
                    "experiment_id": state.get("experiment_id"),
                    "created_at": state.get("created_at"),
                    "updated_at": state.get("updated_at"),
                    "status": state.get("status"),
                    "phase": state.get("phase"),
                    "request_summary": state.get("request_summary"),
                    "report_language": state.get("report_language"),
                    "artifact_count": len(state.get("artifact_ids") or []),
                    "model_ids": list(state.get("model_ids") or []),
                }
            )
        summaries.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
        return summaries[: int(limit)]

    def get_experiment_state(self, experiment_id: str) -> dict[str, Any]:
        """Return only the public experiment state."""

        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            public_state, _runtime_state = self._load_states(resolved_id)
            return json_safe(public_state)

    def record_handoff(
        self,
        experiment_id: str,
        handoff: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append a validated sub-agent handoff and update public evidence indexes."""

        resolved_id = self._require_existing(experiment_id)
        normalized = normalize_handoff(resolved_id, handoff)
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            current_status = str(public_state.get("status") or "created")
            if self._is_finalized_state(public_state):
                raise ExperimentStateTransitionError(
                    f"cannot record another handoff after experiment status {current_status!r}"
                )
            pending_handoff_agent = self._pending_handoff_agent(runtime_state)
            if pending_handoff_agent is not None and pending_handoff_agent != normalized["agent"]:
                raise ExperimentStateTransitionError(
                    f"role {pending_handoff_agent!r} must record its structured handoff before "
                    f"a handoff from {normalized['agent']!r} can be accepted"
                )
            existing_handoffs = list(public_state.get("handoffs") or [])
            if normalized["agent"] != "qsaria_report" and any(
                item.get("agent") == "qsaria_report" for item in existing_handoffs
            ):
                raise ExperimentStateTransitionError(
                    "the Report handoff is final; no later scientific handoff is allowed"
                )
            if normalized["agent"] == "qsaria_report":
                if any(item.get("agent") == "qsaria_report" for item in existing_handoffs):
                    raise ExperimentStateTransitionError(
                        "the Qsaria Report mission can record exactly one handoff"
                    )
                if normalized["status"] in {"retryable_error", "needs_user_input"}:
                    raise InvalidHandoffError(
                        "the single final Report mission cannot pause or retry; use "
                        "terminal_failure when it cannot produce a report"
                    )
                if normalized["status"] in {"completed", "partial"} or (
                    normalized["status"] == "terminal_failure" and public_state.get("report")
                ):
                    report_artifact_id = self._validate_saved_report_artifact(
                        resolved_id,
                        public_state,
                        runtime_state,
                    )
                    if normalized.get("artifact_ids") != [report_artifact_id]:
                        raise InvalidHandoffError(
                            "a successful Report handoff must claim exactly the saved report "
                            f"artifact_id {report_artifact_id!r}"
                        )
            pending_persistence = self._pending_persistence(runtime_state)
            if (
                normalized["agent"] == "qsaria_registry"
                and pending_persistence is not None
                and normalized["status"] in {"completed", "partial"}
            ):
                raise InvalidHandoffError(
                    "Registry cannot record a successful handoff while catalog persistence "
                    f"is still pending via {pending_persistence['required_tools']!r}"
                )
            if current_status == "terminal_failure":
                pending_failure = runtime_state.get("pending_failure")
                already_recorded = any(
                    item.get("agent") == normalized["agent"]
                    and item.get("status") == "terminal_failure"
                    for item in public_state.get("handoffs") or []
                )
                terminal_evidence = (
                    normalized["status"] == "terminal_failure"
                    and isinstance(pending_failure, Mapping)
                    and pending_failure.get("agent") == normalized["agent"]
                )
                terminal_evidence_recorded = any(
                    item.get("status") == "terminal_failure"
                    for item in public_state.get("handoffs") or []
                )
                final_report = (
                    normalized["agent"] == "qsaria_report"
                    and normalized["status"] in {"completed", "partial", "terminal_failure"}
                    and terminal_evidence_recorded
                )
                if not (terminal_evidence or final_report) or already_recorded:
                    raise ExperimentStateTransitionError(
                        "terminal experiments accept only one matching terminal or report handoff"
                    )
            pending_failure = runtime_state.get("pending_failure")
            if current_status in {"retryable_error", "needs_user_input"}:
                if not isinstance(pending_failure, Mapping):
                    raise ExperimentStateTransitionError(
                        f"experiment status {current_status!r} has no persisted failure evidence"
                    )
                if pending_failure.get("agent") != normalized["agent"]:
                    raise ExperimentStateTransitionError(
                        f"experiment status {current_status!r} requires a handoff from "
                        f"{pending_failure.get('agent')!r}"
                    )
                if normalized["status"] not in {current_status, "terminal_failure"}:
                    raise ExperimentStateTransitionError(
                        f"experiment status {current_status!r} requires either a matching "
                        "handoff or same-role escalation to terminal_failure, "
                        f"not {normalized['status']!r}"
                    )
            if normalized["status"] == "retryable_error":
                if (
                    isinstance(pending_failure, Mapping)
                    and pending_failure.get("agent") == normalized["agent"]
                    and int(pending_failure.get("automatic_retries") or 0) >= 1
                ):
                    raise ExperimentStateTransitionError(
                        "the agent's single automatic retry is exhausted; use terminal_failure"
                    )
                last_event = (runtime_state.get("tool_events") or [None])[-1]
                if (
                    isinstance(last_event, Mapping)
                    and last_event.get("agent") == normalized["agent"]
                    and last_event.get("automatic_retry") is True
                ):
                    raise ExperimentStateTransitionError(
                        "the agent's single automatic retry is exhausted; use terminal_failure"
                    )
            explicitly_claimed_artifacts = list(normalized.get("artifact_ids") or [])
            explicitly_claimed_models = extract_model_ids(normalized)
            known_model_ids = set(public_state.get("model_ids") or []) | set(
                (runtime_state.get("model_provenance") or {}).keys()
            )
            unknown_model_ids = [
                model_id
                for model_id in explicitly_claimed_models
                if model_id not in known_model_ids
            ]
            if unknown_model_ids:
                raise InvalidHandoffError(
                    "handoff references model_ids without prior tool/catalog provenance: "
                    + ", ".join(unknown_model_ids)
                )
            artifact_ids_before = set(public_state.get("artifact_ids") or [])
            shim = self._runtime_shim(
                resolved_id,
                normalized["agent"],
                "qsaria_record_handoff",
                public_state,
                runtime_state,
            )
            self._register_payload(shim, normalized, source=normalized["agent"])
            current_artifact_ids = public_state.get("artifact_ids") or []
            new_artifact_ids = [
                artifact_id
                for artifact_id in current_artifact_ids
                if artifact_id not in artifact_ids_before
            ]
            known_artifact_ids = set((runtime_state.get("artifacts") or {}).keys())
            unknown_artifact_ids = [
                artifact_id
                for artifact_id in explicitly_claimed_artifacts
                if artifact_id not in known_artifact_ids
            ]
            if unknown_artifact_ids:
                raise InvalidHandoffError(
                    "handoff references unknown artifact_ids: " + ", ".join(unknown_artifact_ids)
                )
            self._refresh_known_artifacts(resolved_id, runtime_state)
            unavailable_artifact_ids = [
                artifact_id
                for artifact_id in explicitly_claimed_artifacts
                if not (runtime_state.get("artifacts") or {}).get(artifact_id, {}).get("exists")
            ]
            if unavailable_artifact_ids:
                raise InvalidHandoffError(
                    "handoff references unavailable artifact_ids: "
                    + ", ".join(unavailable_artifact_ids)
                )
            normalized["artifact_ids"] = self._merge_strings(
                normalized.get("artifact_ids"),
                new_artifact_ids,
            )
            normalized["model_ids"] = self._merge_strings(
                normalized.get("model_ids"),
                extract_model_ids(normalized),
            )
            public_state.setdefault("handoffs", []).append(normalized)
            if normalized["agent"] == "qsaria_report" and isinstance(
                public_state.get("report"), dict
            ):
                public_state["report"]["validation_status"] = normalized["status"]
            public_state["artifact_ids"] = self._merge_strings(
                public_state.get("artifact_ids"), normalized.get("artifact_ids")
            )
            public_state["model_ids"] = self._merge_strings(
                public_state.get("model_ids"), normalized.get("model_ids")
            )
            public_state["warnings"] = self._merge_strings(
                public_state.get("warnings"), normalized.get("warnings")
            )
            public_state["blockers"] = self._merge_strings(
                public_state.get("blockers"), normalized.get("blockers")
            )
            if normalized["status"] in {"retryable_error", "needs_user_input"}:
                failure_text = next(
                    iter(normalized.get("blockers") or []),
                    normalized["summary"],
                )
                previous_pending = (
                    pending_failure
                    if isinstance(pending_failure, Mapping)
                    and pending_failure.get("agent") == normalized["agent"]
                    else {}
                )
                previous_errors = previous_pending.get("errors") or [previous_pending.get("error")]
                pending_errors = self._merge_strings(previous_errors, [failure_text])
                runtime_state["pending_failure"] = {
                    "agent": normalized["agent"],
                    "tool_name": previous_pending.get("tool_name"),
                    "status": normalized["status"],
                    "error": pending_errors[0],
                    "errors": pending_errors,
                    "automatic_retries": int(previous_pending.get("automatic_retries") or 0),
                    "source": ("tool_and_handoff" if previous_pending else "handoff"),
                    "handoff_recorded": True,
                    "updated_at": utc_now(),
                }
            elif (
                normalized["status"] == "terminal_failure"
                and isinstance(pending_failure, Mapping)
                and pending_failure.get("agent") == normalized["agent"]
            ):
                previous_errors = pending_failure.get("errors") or [pending_failure.get("error")]
                pending_errors = self._merge_strings(
                    previous_errors,
                    normalized.get("blockers"),
                )
                runtime_state["pending_failure"] = {
                    **dict(pending_failure),
                    "status": "terminal_failure",
                    "error": next(iter(pending_errors), normalized["summary"]),
                    "errors": pending_errors,
                    "handoff_recorded": True,
                    "updated_at": utc_now(),
                }
            elif (
                isinstance(pending_failure, Mapping)
                and pending_failure.get("agent") == normalized["agent"]
            ):
                runtime_state.pop("pending_failure", None)
                pending_errors = pending_failure.get("errors") or [pending_failure.get("error")]
                if pending_errors:
                    public_state["blockers"] = [
                        blocker
                        for blocker in public_state.get("blockers") or []
                        if blocker not in pending_errors
                    ]
            public_state["phase"] = normalized["agent"]
            public_state["status"] = (
                "terminal_failure"
                if current_status == "terminal_failure"
                else self._state_status_after_handoff(
                    normalized["status"],
                    public_state.get("handoffs") or [],
                )
            )
            if pending_handoff_agent == normalized["agent"]:
                runtime_state.pop("pending_handoff", None)
            public_state["updated_at"] = utc_now()
            runtime_state["updated_at"] = public_state["updated_at"]
            self._save_states(resolved_id, public_state, runtime_state)
            return json_safe(normalized)

    def complete_experiment(
        self,
        experiment_id: str,
        status: Literal["completed", "partial", "terminal_failure"] = "completed",
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Mark coordinator-level completion after report verification."""

        if status not in COMPLETION_STATUSES:
            raise ValueError(f"status must be one of {', '.join(sorted(COMPLETION_STATUSES))}")
        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            current_status = str(public_state.get("status") or "created")
            pending_handoff_agent = self._pending_handoff_agent(runtime_state)
            if pending_handoff_agent is not None:
                raise ExperimentStateTransitionError(
                    "cannot finalize an experiment before the structured handoff from "
                    f"{pending_handoff_agent!r} is recorded"
                )
            self._require_persistence_resolved(
                runtime_state,
                action="finalize the experiment",
            )
            if current_status == "terminal_failure" and status != "terminal_failure":
                raise ExperimentStateTransitionError(
                    f"cannot change completed experiment status {current_status!r} to {status!r}"
                )
            if self._is_finalized_state(public_state):
                if current_status == status:
                    return json_safe(public_state)
                raise ExperimentStateTransitionError(
                    f"cannot change completed experiment status {current_status!r} to {status!r}"
                )
            if current_status in {"retryable_error", "needs_user_input", "running"}:
                raise ExperimentStateTransitionError(
                    f"cannot finalize unresolved status {current_status!r}"
                )
            handoffs = list(public_state.get("handoffs") or [])
            if not handoffs:
                raise ExperimentStateTransitionError(
                    "cannot finalize an experiment before structured agent evidence is recorded"
                )
            pending_failure = runtime_state.get("pending_failure")
            if runtime_state.get("active_call") is not None:
                raise ExperimentStateTransitionError(
                    "cannot finalize an experiment while a scientific call is active"
                )
            if isinstance(pending_failure, Mapping):
                pending_status = str(pending_failure.get("status") or "")
                pending_resolved = pending_failure.get("handoff_recorded") is True
                if (
                    pending_status in {"retryable_error", "needs_user_input"}
                    or not pending_resolved
                ):
                    raise ExperimentStateTransitionError(
                        "cannot finalize while retryable or user-input failure evidence is "
                        "unresolved"
                    )
                if pending_status == "terminal_failure" and status != "terminal_failure":
                    raise ExperimentStateTransitionError(
                        "terminal failure evidence can only be finalized as terminal_failure"
                    )
            unresolved_partial_agents = self._unresolved_partial_agents(handoffs)
            if status == "completed" and unresolved_partial_agents:
                raise ExperimentStateTransitionError(
                    "cannot mark unresolved partial scientific results as completed; "
                    "use status='partial' or resolve the same role first: "
                    + ", ".join(unresolved_partial_agents)
                )
            if status == "terminal_failure":
                terminal_handoff_recorded = any(
                    item.get("status") == "terminal_failure" for item in handoffs
                )
                if not terminal_handoff_recorded:
                    raise ExperimentStateTransitionError(
                        "terminal completion requires a matching terminal agent handoff"
                    )
            if public_state.get("report"):
                report_artifact_id = self._validate_saved_report_artifact(
                    resolved_id,
                    public_state,
                    runtime_state,
                )
                report_handoffs = [
                    item for item in handoffs if item.get("agent") == "qsaria_report"
                ]
                report_status_allowed = bool(
                    len(report_handoffs) == 1
                    and (
                        report_handoffs[0].get("status") in {"completed", "partial"}
                        or (
                            status == "terminal_failure"
                            and report_handoffs[0].get("status") == "terminal_failure"
                        )
                    )
                )
                if not report_status_allowed or report_handoffs[0].get("artifact_ids") != [
                    report_artifact_id
                ]:
                    raise ExperimentStateTransitionError(
                        "a saved report requires exactly one compatible Qsaria Report handoff "
                        "claiming its verified artifact"
                    )
            now = utc_now()
            public_state.update(
                {
                    "status": status,
                    "phase": "completed",
                    "updated_at": now,
                    "completed_at": now,
                }
            )
            if summary is not None:
                public_state["completion_summary"] = str(summary).strip()
            runtime_state["updated_at"] = now
            self._save_states(resolved_id, public_state, runtime_state)
            return json_safe(public_state)

    # ------------------------------------------------------------------
    # Artifacts and reporting
    # ------------------------------------------------------------------
    def list_artifacts(
        self,
        experiment_id: str,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return registered and newly discovered artifacts for an experiment."""

        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            changed = self._discover_artifacts(resolved_id, public_state, runtime_state)
            if changed:
                self._save_states(resolved_id, public_state, runtime_state)
            entries = list((runtime_state.get("artifacts") or {}).values())
            if kind:
                entries = [entry for entry in entries if entry.get("kind") == kind]
            return sorted((json_safe(entry) for entry in entries), key=lambda item: item["path"])

    def get_artifact(
        self,
        experiment_id: str,
        artifact_id: str,
        max_preview_bytes: int = 65536,
    ) -> dict[str, Any]:
        """Return artifact metadata and a bounded text preview when appropriate."""

        if not 0 <= int(max_preview_bytes) <= _MAX_PREVIEW_BYTES:
            raise ValueError(f"max_preview_bytes must be between 0 and {_MAX_PREVIEW_BYTES}")
        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            changed = self._discover_artifacts(resolved_id, public_state, runtime_state)
            entry = (runtime_state.get("artifacts") or {}).get(str(artifact_id))
            if entry is None:
                raise FileNotFoundError(
                    f"unknown artifact_id {artifact_id!r} for experiment {resolved_id}"
                )
            result = json_safe(entry)
            if (
                int(max_preview_bytes)
                and entry.get("exists") is True
                and self._is_text_mime(str(entry.get("media_type") or ""))
            ):
                blob = self._read_artifact_bytes(
                    resolved_id,
                    str(entry["path"]),
                    int(max_preview_bytes) + 1,
                )
                result["preview"] = blob[: int(max_preview_bytes)].decode("utf-8", errors="replace")
                result["preview_truncated"] = len(blob) > int(max_preview_bytes)
            if changed:
                self._save_states(resolved_id, public_state, runtime_state)
            return result

    def build_report_context(self, experiment_id: str) -> dict[str, Any]:
        """Build fact-first read-only context for the Report sub-agent."""

        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            pending_handoff_agent = self._pending_handoff_agent(runtime_state)
            if pending_handoff_agent is not None:
                raise ExperimentStateTransitionError(
                    "cannot build the final report context before the structured handoff "
                    f"from {pending_handoff_agent!r} is recorded"
                )
            self._require_persistence_resolved(
                runtime_state,
                action="build the final report context",
            )
            changed = self._discover_artifacts(resolved_id, public_state, runtime_state)
            if changed:
                self._save_states(resolved_id, public_state, runtime_state)
            reporting_packets = self._reporting_packets(public_state, runtime_state)
            report_facts = [
                json_safe(packet["report_facts"])
                for packet in reporting_packets
                if isinstance(packet.get("report_facts"), Mapping)
            ]
            report_tables = [
                json_safe(packet["report_tables"])
                for packet in reporting_packets
                if isinstance(packet.get("report_tables"), Mapping) and packet.get("report_tables")
            ]
            return {
                "schema_version": REPORT_CONTEXT_SCHEMA_VERSION,
                "experiment_id": resolved_id,
                "experiment": json_safe(public_state),
                "handoffs": json_safe(public_state.get("handoffs") or []),
                "report_facts": report_facts,
                "report_tables": report_tables,
                "reporting_packets": reporting_packets,
                "artifacts": sorted(
                    (json_safe(item) for item in (runtime_state.get("artifacts") or {}).values()),
                    key=lambda item: item["path"],
                ),
                "evidence_precedence": [
                    "structured_artifacts",
                    "structured_tool_results",
                    "structured_handoffs",
                    "report_narrative",
                    "coordinator_interpretation",
                ],
            }

    def save_report(
        self,
        experiment_id: str,
        report_content: str | Mapping[str, Any],
        report_format: Literal["markdown", "json"] = "markdown",
    ) -> dict[str, Any]:
        """Persist the final Report Agent output as a normal experiment artifact."""

        if report_format not in {"markdown", "json"}:
            raise ValueError("report_format must be 'markdown' or 'json'")
        if report_format == "json":
            if isinstance(report_content, str):
                parsed = json.loads(report_content)
            elif isinstance(report_content, Mapping):
                parsed = dict(report_content)
            else:
                raise TypeError("JSON report_content must be a mapping or JSON string")
            content = json.dumps(json_safe(parsed), indent=2, sort_keys=True) + "\n"
            suffix = "json"
        else:
            if not isinstance(report_content, str) or not report_content.strip():
                raise ValueError("Markdown report_content must be a non-empty string")
            content = report_content.rstrip() + "\n"
            suffix = "md"

        resolved_id = self._require_existing(experiment_id)
        rel_path = PurePosixPath("workflows", "reports", f"qsaria_report.{suffix}").as_posix()
        with self._locked(resolved_id):
            public_state, runtime_state = self._load_states(resolved_id)
            current_status = str(public_state.get("status") or "created")
            pending_handoff_agent = self._pending_handoff_agent(runtime_state)
            if pending_handoff_agent is not None:
                raise ExperimentStateTransitionError(
                    "cannot save the final report before the structured handoff from "
                    f"{pending_handoff_agent!r} is recorded"
                )
            self._require_persistence_resolved(
                runtime_state,
                action="save the final report",
            )
            if self._is_finalized_state(public_state):
                raise ExperimentStateTransitionError(
                    f"cannot save a report after experiment status {current_status!r} is finalized"
                )
            if public_state.get("report"):
                raise FileExistsError(
                    f"a final report is already saved for experiment {resolved_id}"
                )
            handoffs = list(public_state.get("handoffs") or [])
            if not any(item.get("agent") != "qsaria_report" for item in handoffs):
                raise ExperimentStateTransitionError(
                    "cannot save a report before scientific handoff evidence exists"
                )
            if current_status in {"retryable_error", "needs_user_input"}:
                raise ExperimentStateTransitionError(
                    f"cannot save a report while status {current_status!r} is unresolved"
                )
            if current_status == "terminal_failure" and not any(
                item.get("status") == "terminal_failure" for item in handoffs
            ):
                raise ExperimentStateTransitionError(
                    "terminal report saving requires the terminal agent handoff first"
                )
            prepared_at = utc_now()
            encoded = content.encode("utf-8")
            runtime_state["pending_report"] = {
                "path": rel_path,
                "format": report_format,
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
                "content_bytes": len(encoded),
                "prepared_at": prepared_at,
            }
            runtime_state["updated_at"] = prepared_at
            self._save_states(resolved_id, public_state, runtime_state)
            self._write_text(resolved_id, rel_path, content)
            entry = self._register_artifact(
                resolved_id,
                rel_path,
                runtime_state,
                source="qsaria_report",
            )
            now = utc_now()
            public_state["artifact_ids"] = self._merge_strings(
                public_state.get("artifact_ids"), [entry["artifact_id"]]
            )
            public_state["report"] = {
                "artifact_id": entry["artifact_id"],
                "path": rel_path,
                "format": report_format,
                "content_sha256": hashlib.sha256(encoded).hexdigest(),
                "content_bytes": len(encoded),
                "saved_at": now,
            }
            runtime_state.pop("pending_report", None)
            public_state["phase"] = "qsaria_report"
            public_state["updated_at"] = now
            runtime_state["updated_at"] = now
            self._save_states(resolved_id, public_state, runtime_state)
            return json_safe(public_state["report"])

    # ------------------------------------------------------------------
    # Toolkit execution binding
    # ------------------------------------------------------------------
    @contextmanager
    def run(
        self,
        experiment_id: str,
        *,
        agent_name: str,
        tool_name: str,
        catalog_write: bool = False,
    ) -> Iterator[ExperimentRuntime]:
        """Bind one toolkit call and durably save its state in ``finally``.

        The yielded ``MCPAgentContext`` is deliberately model-free and uses
        ``llm_policy='disabled'``.  Both the storage prefix and current MCP
        context are restored to their exact previous values at exit.
        """

        resolved_id = self._require_existing(experiment_id)
        with self._locked(resolved_id):
            with ExitStack() as stack:
                if catalog_write:
                    stack.enter_context(self.catalog_lock())
                public_state, runtime_state = self._load_states(resolved_id)
                entry_status = str(public_state.get("status") or "created")
                automatic_retry = self._validate_run_transition(
                    public_state,
                    runtime_state,
                    agent_name=str(agent_name),
                    tool_name=str(tool_name),
                )
                session_state = runtime_state.get("session_state")
                if not isinstance(session_state, dict):
                    session_state = {}
                context = MCPAgentContext(
                    name=str(agent_name),
                    session_state=session_state,
                    model=None,
                    llm_policy="disabled",
                    llm=None,
                )
                runtime = ExperimentRuntime(
                    manager=self,
                    experiment_id=resolved_id,
                    agent_name=str(agent_name),
                    tool_name=str(tool_name),
                    context=context,
                    public_state=public_state,
                    runtime_state=runtime_state,
                    _entry_status=entry_status,
                    _automatic_retry=automatic_retry,
                )

                storage_token = self._bind_storage_prefix(resolved_id)
                context_token = self._bind_mcp_context(context)
                caught: BaseException | None = None
                try:
                    from cs_copilot.storage import ensure_output_context

                    ensure_output_context(context.session_state)
                    runtime_state["active_call"] = {
                        "call_id": runtime.call_id,
                        "agent": runtime.agent_name,
                        "tool_name": runtime.tool_name,
                        "started_at": runtime.started_at,
                        "automatic_retry": runtime._automatic_retry,
                    }
                    public_state["status"] = "running"
                    public_state["phase"] = str(tool_name)
                    public_state["updated_at"] = utc_now()
                    runtime_state["updated_at"] = public_state["updated_at"]
                    self._save_states(resolved_id, public_state, runtime_state)
                    try:
                        yield runtime
                    except BaseException as exc:
                        caught = exc
                        if runtime._outcome_status is None:
                            runtime.capture_error(exc)
                        raise
                finally:
                    try:
                        self._finalize_runtime(runtime)
                    except Exception:  # noqa: BLE001 - preserve the scientific exception
                        if caught is None:
                            raise
                        logger.exception(
                            "Could not persist Qsaria runtime after %s failed", runtime.tool_name
                        )
                    finally:
                        self._reset_mcp_context(context_token)
                        self._reset_storage_prefix(storage_token)

    @contextmanager
    def catalog_lock(self) -> Iterator[None]:
        """Use the same reentrant catalog lock as Agno/Chainlit toolkits."""

        from cs_copilot.tools.prediction.catalog import model_catalog_lock

        with model_catalog_lock():
            yield

    # ------------------------------------------------------------------
    # Internal persistence helpers
    # ------------------------------------------------------------------
    def _versions(self) -> dict[str, Any]:
        try:
            package_version = importlib_metadata.version("cs_copilot")
        except importlib_metadata.PackageNotFoundError:
            package_version = "0.3.0"
        return {
            "cs_copilot": package_version,
            "qsaria": package_version,
            "qsaria_contract": EXPERIMENT_SCHEMA_VERSION,
            "runtime_state": RUNTIME_SCHEMA_VERSION,
            "handoff": HANDOFF_SCHEMA_VERSION,
            "training": TRAINING_CONTRACT_VERSION,
            "plugin": PLUGIN_CONTRACT_VERSION,
        }

    def _uses_s3(self) -> bool:
        if self._explicit_local_root is not None:
            return False
        from cs_copilot.storage import is_s3_enabled

        return is_s3_enabled()

    @staticmethod
    def _s3_single_writer_acknowledged() -> bool:
        return os.getenv("QSARIA_S3_SINGLE_WRITER", "").strip().lower() == "true"

    def _allowed_input_roots(self) -> tuple[Path, ...]:
        configured = os.getenv("QSARIA_MCP_ALLOWED_INPUT_ROOTS", "")
        candidates = [Path.cwd(), self._local_root()]
        candidates.extend(Path(item) for item in configured.split(os.pathsep) if item.strip())
        roots: list[Path] = []
        for candidate in candidates:
            resolved = candidate.expanduser().resolve(strict=False)
            if resolved not in roots:
                roots.append(resolved)
        return tuple(roots)

    def _local_root(self) -> Path:
        if self._explicit_local_root is not None:
            return self._explicit_local_root
        from cs_copilot.storage import client as storage_client

        configured = os.getenv("CS_COPILOT_STORAGE_ROOT")
        return Path(configured) if configured else Path(storage_client.LOCAL_STORAGE_ROOT)

    def _session_root(self, experiment_id: str) -> Path:
        return self._local_root() / EXPERIMENTS_PREFIX / validate_experiment_id(experiment_id)

    def _safe_session_root(self, experiment_id: str) -> Path:
        storage_root = self._local_root().expanduser().resolve(strict=False)
        sessions_root = storage_root / EXPERIMENTS_PREFIX
        raw_session_root = sessions_root / validate_experiment_id(experiment_id)
        if sessions_root.is_symlink() or raw_session_root.is_symlink():
            raise InvalidInputPathError(
                f"experiment {experiment_id} cannot use a symbolic-link session root"
            )
        resolved = raw_session_root.resolve(strict=False)
        if not resolved.is_relative_to(storage_root):
            raise InvalidInputPathError(
                f"experiment {experiment_id} resolves outside the configured storage root"
            )
        return resolved

    def _safe_session_path(self, experiment_id: str, rel_path: str) -> Path:
        if not self._safe_relative(rel_path):
            raise InvalidInputPathError(f"unsafe experiment-relative path: {rel_path!r}")
        root = self._safe_session_root(experiment_id)
        current = root
        for part in PurePosixPath(rel_path).parts:
            current /= part
            if current.is_symlink():
                raise InvalidInputPathError(
                    f"experiment path {rel_path!r} contains a symbolic-link component"
                )
        resolved = (root / rel_path).resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise InvalidInputPathError(
                f"experiment path {rel_path!r} escapes through a symbolic-link parent"
            )
        return resolved

    def _backend_identity(self) -> str:
        if not self._uses_s3():
            return f"local:{self._local_root().resolve()}"
        from cs_copilot.storage import get_s3_config

        config = get_s3_config()
        return f"s3:{config.endpoint_url or 'aws'}:{config.bucket_name}"

    @contextmanager
    def _locked(self, experiment_id: str, *, create: bool = False) -> Iterator[None]:
        resolved_id = validate_experiment_id(experiment_id)
        key = f"{self._backend_identity()}:{resolved_id}"
        with _process_lock(key):
            if self._uses_s3():
                yield
            else:
                session_root = self._safe_session_root(resolved_id)
                if create:
                    session_root.mkdir(parents=True, exist_ok=True)
                lock_path = self._safe_session_path(
                    resolved_id,
                    "qsaria/.experiment.lock",
                )
                with _file_lock(lock_path):
                    yield

    def _require_existing(self, experiment_id: str) -> str:
        resolved_id = validate_experiment_id(experiment_id)
        if not self._state_exists(resolved_id):
            raise ExperimentNotFoundError(f"unknown Qsaria experiment: {resolved_id}")
        return resolved_id

    def _state_exists(self, experiment_id: str) -> bool:
        if not self._uses_s3():
            return self._safe_session_path(experiment_id, PUBLIC_STATE_REL_PATH).is_file()
        try:
            self._read_json(experiment_id, PUBLIC_STATE_REL_PATH)
        except (FileNotFoundError, OSError):
            return False
        return True

    def _load_public_state(self, experiment_id: str) -> dict[str, Any]:
        try:
            payload = self._read_json(experiment_id, PUBLIC_STATE_REL_PATH)
        except FileNotFoundError as exc:
            raise ExperimentNotFoundError(f"unknown Qsaria experiment: {experiment_id}") from exc
        if payload.get("experiment_id") != experiment_id:
            raise ValueError(f"corrupt experiment state for {experiment_id}: id mismatch")
        if payload.get("schema_version") != EXPERIMENT_SCHEMA_VERSION:
            raise IncompatibleExperimentSchemaError(
                f"experiment {experiment_id} uses unsupported public schema "
                f"{payload.get('schema_version')!r}; expected {EXPERIMENT_SCHEMA_VERSION!r}"
            )
        return payload

    def _load_states(self, experiment_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        public_state = self._load_public_state(experiment_id)
        try:
            runtime_state = self._read_json(experiment_id, RUNTIME_STATE_REL_PATH)
        except FileNotFoundError:
            runtime_state = new_runtime_state(experiment_id)
        if runtime_state.get("experiment_id") != experiment_id:
            raise ValueError(f"corrupt runtime state for {experiment_id}: id mismatch")
        if runtime_state.get("schema_version") != RUNTIME_SCHEMA_VERSION:
            raise IncompatibleExperimentSchemaError(
                f"experiment {experiment_id} uses unsupported runtime schema "
                f"{runtime_state.get('schema_version')!r}; expected {RUNTIME_SCHEMA_VERSION!r}"
            )
        public_revision = self._state_revision(public_state, label="public")
        runtime_revision = self._state_revision(runtime_state, label="runtime")
        snapshot = runtime_state.get("public_state_snapshot")
        if runtime_revision > public_revision:
            if not isinstance(snapshot, Mapping):
                raise ExperimentStateTransitionError(
                    "runtime state is newer than public state but has no recovery snapshot"
                )
            recovered = dict(snapshot)
            if (
                recovered.get("experiment_id") != experiment_id
                or recovered.get("schema_version") != EXPERIMENT_SCHEMA_VERSION
                or self._state_revision(recovered, label="recovery") != runtime_revision
            ):
                raise ExperimentStateTransitionError(
                    "runtime recovery snapshot does not match the experiment transaction"
                )
            self._write_json(experiment_id, PUBLIC_STATE_REL_PATH, recovered)
            public_state = recovered
        elif public_revision > runtime_revision:
            raise ExperimentStateTransitionError(
                "public experiment state is newer than runtime state; refusing unsafe recovery"
            )
        elif isinstance(snapshot, Mapping):
            recovered = dict(snapshot)
            if self._state_revision(recovered, label="recovery") == runtime_revision and json_safe(
                recovered
            ) != json_safe(public_state):
                self._write_json(experiment_id, PUBLIC_STATE_REL_PATH, recovered)
                public_state = recovered
        recovered_interruption = self._recover_interrupted_call(
            experiment_id,
            public_state,
            runtime_state,
        )
        recovered_report = self._reconcile_pending_report(
            experiment_id,
            public_state,
            runtime_state,
        )
        if recovered_interruption or recovered_report:
            self._save_states(experiment_id, public_state, runtime_state)
        return public_state, runtime_state

    def _recover_interrupted_call(
        self,
        experiment_id: str,
        public_state: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> bool:
        """Turn one persisted in-flight call into a reviewable retry state."""

        status = str(public_state.get("status") or "created")
        active_call = runtime_state.get("active_call")
        if status != "running":
            if active_call is not None:
                raise ExperimentStateTransitionError(
                    "runtime active_call exists while the public state is not running"
                )
            return False
        if not isinstance(active_call, Mapping):
            raise ExperimentStateTransitionError(
                f"experiment {experiment_id} is running without recoverable active_call evidence"
            )
        required = ("call_id", "agent", "tool_name", "started_at")
        if any(
            not isinstance(active_call.get(key), str) or not str(active_call.get(key)).strip()
            for key in required
        ):
            raise ExperimentStateTransitionError(
                f"experiment {experiment_id} has malformed active_call recovery evidence"
            )
        automatic_retry = active_call.get("automatic_retry", False)
        if not isinstance(automatic_retry, bool):
            raise ExperimentStateTransitionError(
                f"experiment {experiment_id} has malformed active_call retry evidence"
            )

        now = utc_now()
        agent = str(active_call["agent"])
        tool_name = str(active_call["tool_name"])
        recovered_status = "terminal_failure" if automatic_retry else "retryable_error"
        next_instruction = (
            "the single automatic retry is exhausted; inspect its artifacts and record a "
            "terminal_failure handoff"
            if automatic_retry
            else "inspect its artifacts and record a retryable_error handoff before retrying"
        )
        message = (
            f"interrupted Qsaria call {active_call['call_id']} for {tool_name}; {next_instruction}"
        )
        public_state["status"] = recovered_status
        public_state["phase"] = tool_name
        public_state["updated_at"] = now
        public_state["blockers"] = self._merge_strings(
            public_state.get("blockers"),
            [message],
        )
        runtime_state["pending_failure"] = {
            "agent": agent,
            "tool_name": tool_name,
            "status": recovered_status,
            "error": message,
            "errors": [message],
            "automatic_retries": 1 if automatic_retry else 0,
            "source": "interrupted_call",
            "handoff_recorded": False,
            "updated_at": now,
        }
        interrupted = runtime_state.setdefault("interrupted_calls", [])
        interrupted.append(
            {
                **json_safe(active_call),
                "recovered_at": now,
                "status": recovered_status,
            }
        )
        if len(interrupted) > _MAX_TOOL_EVENTS:
            del interrupted[: len(interrupted) - _MAX_TOOL_EVENTS]
        runtime_state.pop("active_call", None)
        runtime_state["updated_at"] = now
        return True

    def _reconcile_pending_report(
        self,
        experiment_id: str,
        public_state: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> bool:
        """Complete or abandon an interrupted atomic report transaction."""

        pending = runtime_state.get("pending_report")
        if pending is None:
            return False
        if not isinstance(pending, Mapping):
            raise ExperimentStateTransitionError("pending_report recovery evidence is malformed")
        rel_path = pending.get("path")
        report_format = pending.get("format")
        digest = pending.get("content_sha256")
        content_bytes = pending.get("content_bytes")
        prepared_at = pending.get("prepared_at")
        expected_suffix = {"markdown": "md", "json": "json"}.get(report_format)
        expected_path = (
            PurePosixPath("workflows", "reports", f"qsaria_report.{expected_suffix}").as_posix()
            if expected_suffix
            else None
        )
        if (
            rel_path != expected_path
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(content_bytes, int)
            or isinstance(content_bytes, bool)
            or content_bytes < 0
            or not isinstance(prepared_at, str)
            or not prepared_at.strip()
        ):
            raise ExperimentStateTransitionError("pending_report recovery evidence is malformed")
        existing_report = public_state.get("report")
        if existing_report:
            if not isinstance(existing_report, Mapping) or existing_report.get("path") != rel_path:
                raise ExperimentStateTransitionError(
                    "pending report conflicts with the persisted public report"
                )
            runtime_state.pop("pending_report", None)
            return True

        stat = self._artifact_stat(experiment_id, rel_path)
        if stat is None:
            runtime_state.pop("pending_report", None)
            return True
        if stat.get("size") != content_bytes:
            raise ExperimentStateTransitionError(
                "interrupted report bytes do not match the persisted report intent"
            )
        blob = self._read_artifact_bytes(experiment_id, rel_path, content_bytes + 1)
        if len(blob) != content_bytes or hashlib.sha256(blob).hexdigest() != digest:
            raise ExperimentStateTransitionError(
                "interrupted report digest does not match the persisted report intent"
            )
        entry = self._register_artifact(
            experiment_id,
            rel_path,
            runtime_state,
            source="qsaria_report",
        )
        public_state["artifact_ids"] = self._merge_strings(
            public_state.get("artifact_ids"),
            [entry["artifact_id"]],
        )
        public_state["report"] = {
            "artifact_id": entry["artifact_id"],
            "path": rel_path,
            "format": report_format,
            "content_sha256": digest,
            "content_bytes": content_bytes,
            "saved_at": prepared_at,
        }
        public_state["phase"] = "qsaria_report"
        public_state["updated_at"] = utc_now()
        runtime_state.pop("pending_report", None)
        runtime_state["updated_at"] = public_state["updated_at"]
        return True

    def _validate_saved_report_artifact(
        self,
        experiment_id: str,
        public_state: Mapping[str, Any],
        runtime_state: dict[str, Any],
    ) -> str:
        """Verify the final report against its persisted path, size and digest."""

        report = public_state.get("report")
        if not isinstance(report, Mapping):
            raise ExperimentStateTransitionError(
                "a successful Report handoff requires a saved final report artifact"
            )
        artifact_id = report.get("artifact_id")
        rel_path = report.get("path")
        report_format = report.get("format")
        digest = report.get("content_sha256")
        content_bytes = report.get("content_bytes")
        expected_suffix = {"markdown": "md", "json": "json"}.get(report_format)
        expected_path = (
            PurePosixPath("workflows", "reports", f"qsaria_report.{expected_suffix}").as_posix()
            if expected_suffix
            else None
        )
        if (
            not isinstance(artifact_id, str)
            or not artifact_id
            or rel_path != expected_path
            or artifact_id != artifact_id_for(experiment_id, str(rel_path))
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(content_bytes, int)
            or isinstance(content_bytes, bool)
            or content_bytes < 0
        ):
            raise ExperimentStateTransitionError("saved report metadata is malformed")
        entry = (runtime_state.get("artifacts") or {}).get(artifact_id)
        if (
            not isinstance(entry, Mapping)
            or entry.get("path") != rel_path
            or entry.get("source") != "qsaria_report"
        ):
            raise ExperimentStateTransitionError(
                "saved report artifact is missing from the structured artifact registry"
            )
        stat = self._artifact_stat(experiment_id, str(rel_path))
        if stat is None or stat.get("size") != content_bytes:
            raise ExperimentStateTransitionError(
                "saved report artifact is missing or has an unexpected size"
            )
        blob = self._read_artifact_bytes(experiment_id, str(rel_path), content_bytes + 1)
        if len(blob) != content_bytes or hashlib.sha256(blob).hexdigest() != digest:
            raise ExperimentStateTransitionError(
                "saved report artifact no longer matches its recorded digest"
            )
        refreshed = dict(entry)
        refreshed["exists"] = True
        refreshed["size"] = content_bytes
        runtime_state.setdefault("artifacts", {})[artifact_id] = refreshed
        return artifact_id

    def _save_states(
        self,
        experiment_id: str,
        public_state: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
    ) -> None:
        public_payload = dict(json_safe(public_state))
        runtime_payload = dict(json_safe(runtime_state))
        revision = (
            max(
                self._state_revision(public_payload, label="public"),
                self._state_revision(runtime_payload, label="runtime"),
            )
            + 1
        )
        public_payload["state_revision"] = revision
        runtime_payload["state_revision"] = revision
        runtime_payload["public_state_snapshot"] = public_payload
        self._write_json(experiment_id, RUNTIME_STATE_REL_PATH, runtime_payload)
        self._write_json(experiment_id, PUBLIC_STATE_REL_PATH, public_payload)
        if isinstance(public_state, dict):
            public_state.clear()
            public_state.update(public_payload)
        if isinstance(runtime_state, dict):
            runtime_state.clear()
            runtime_state.update(runtime_payload)

    @staticmethod
    def _state_revision(payload: Mapping[str, Any], *, label: str) -> int:
        value = payload.get("state_revision", 0)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ExperimentStateTransitionError(
                f"{label} state_revision must be a non-negative integer"
            )
        return value

    def _read_json(self, experiment_id: str, rel_path: str) -> dict[str, Any]:
        if not self._uses_s3():
            path = self._safe_session_path(experiment_id, rel_path)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            from cs_copilot.storage import S3

            token = self._bind_storage_prefix(experiment_id)
            try:
                with S3.open(rel_path, "r") as handle:
                    payload = json.load(handle)
            finally:
                self._reset_storage_prefix(token)
        if not isinstance(payload, dict):
            raise ValueError(f"expected JSON object at {rel_path}")
        return payload

    def _write_json(
        self,
        experiment_id: str,
        rel_path: str,
        payload: Mapping[str, Any],
    ) -> None:
        text = json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"
        self._write_text(experiment_id, rel_path, text)

    def _write_text(self, experiment_id: str, rel_path: str, text: str) -> None:
        if not self._uses_s3():
            path = self._safe_session_path(experiment_id, rel_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(
                f".{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temp.open("w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp, path)
            finally:
                if temp.exists():
                    temp.unlink()
            return

        from cs_copilot.storage import S3

        token = self._bind_storage_prefix(experiment_id)
        try:
            with S3.open(rel_path, "w") as handle:
                handle.write(text)
        finally:
            self._reset_storage_prefix(token)

    def _bind_storage_prefix(self, experiment_id: str):
        from cs_copilot.storage import client as storage_client

        return storage_client._SESSION_PREFIX.set(  # noqa: SLF001 - reversible by design
            f"{EXPERIMENTS_PREFIX}/{validate_experiment_id(experiment_id)}"
        )

    @staticmethod
    def _reset_storage_prefix(token: Any) -> None:
        from cs_copilot.storage import client as storage_client

        storage_client._SESSION_PREFIX.reset(token)  # noqa: SLF001 - reversible by design

    @staticmethod
    def _bind_mcp_context(context: MCPAgentContext):
        from cs_copilot.mcp import context as mcp_context

        return mcp_context._CTX.set(context)  # noqa: SLF001 - reversible by design

    @staticmethod
    def _reset_mcp_context(token: Any) -> None:
        from cs_copilot.mcp import context as mcp_context

        mcp_context._CTX.reset(token)  # noqa: SLF001 - reversible by design

    def _runtime_shim(
        self,
        experiment_id: str,
        agent_name: str,
        tool_name: str,
        public_state: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> ExperimentRuntime:
        context = MCPAgentContext(
            name=agent_name,
            session_state=runtime_state.setdefault("session_state", {}),
            model=None,
            llm_policy="disabled",
            llm=None,
        )
        return ExperimentRuntime(
            manager=self,
            experiment_id=experiment_id,
            agent_name=agent_name,
            tool_name=tool_name,
            context=context,
            public_state=public_state,
            runtime_state=runtime_state,
        )

    @staticmethod
    def _pending_handoff_agent(runtime_state: Mapping[str, Any]) -> str | None:
        pending = runtime_state.get("pending_handoff")
        if pending is None:
            return None
        if not isinstance(pending, Mapping):
            raise ExperimentStateTransitionError("runtime pending_handoff evidence is malformed")
        agent = pending.get("agent")
        if not isinstance(agent, str) or agent not in QSARIA_AGENT_NAMES:
            raise ExperimentStateTransitionError("runtime pending_handoff agent is malformed")
        return agent

    @staticmethod
    def _pending_persistence(runtime_state: Mapping[str, Any]) -> Mapping[str, Any] | None:
        pending = runtime_state.get("pending_persistence")
        if pending is None:
            return None
        if not isinstance(pending, Mapping):
            raise ExperimentStateTransitionError(
                "runtime pending_persistence evidence is malformed"
            )
        required_tools = pending.get("required_tools")
        if (
            not isinstance(required_tools, list)
            or not required_tools
            or not all(isinstance(item, str) and item for item in required_tools)
        ):
            raise ExperimentStateTransitionError(
                "runtime pending_persistence required_tools are malformed"
            )
        return pending

    @staticmethod
    def _require_persistence_resolved(
        runtime_state: Mapping[str, Any],
        *,
        action: str,
    ) -> None:
        pending = ExperimentManager._pending_persistence(runtime_state)
        if pending is None:
            return
        raise ExperimentStateTransitionError(
            f"cannot {action} while catalog persistence is pending; qsaria_registry must "
            f"complete one of {pending['required_tools']!r}"
        )

    @staticmethod
    def _reporting_packets(
        public_state: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Return tool-result reporting packets with handoffs as a legacy fallback."""

        packets: list[dict[str, Any]] = []
        runtime_agents: set[str] = set()
        for raw_packet in runtime_state.get("reporting_packets") or []:
            if not isinstance(raw_packet, Mapping):
                continue
            facts = raw_packet.get("report_facts")
            tables = raw_packet.get("report_tables")
            if not isinstance(facts, Mapping) and not isinstance(tables, Mapping):
                continue
            agent = str(raw_packet.get("agent") or "")
            if agent:
                runtime_agents.add(agent)
            packets.append(
                {
                    "source": "structured_tool_result",
                    "agent": agent or None,
                    "tool_name": raw_packet.get("tool_name"),
                    "call_id": raw_packet.get("call_id"),
                    "report_facts": json_safe(facts) if isinstance(facts, Mapping) else None,
                    "report_tables": json_safe(tables) if isinstance(tables, Mapping) else {},
                }
            )

        for handoff in public_state.get("handoffs") or []:
            if not isinstance(handoff, Mapping):
                continue
            agent = str(handoff.get("agent") or "")
            if agent in runtime_agents:
                continue
            facts = handoff.get("facts") or {}
            if not isinstance(facts, Mapping):
                continue
            report_facts = facts.get("report_facts")
            if not isinstance(report_facts, Mapping) and str(facts.get("schema_version")) == "2.0":
                report_facts = facts
            report_tables = facts.get("report_tables")
            if not isinstance(report_facts, Mapping) and not isinstance(report_tables, Mapping):
                continue
            packets.append(
                {
                    "source": "structured_handoff",
                    "agent": agent or None,
                    "tool_name": None,
                    "call_id": None,
                    "report_facts": (
                        json_safe(report_facts) if isinstance(report_facts, Mapping) else None
                    ),
                    "report_tables": (
                        json_safe(report_tables) if isinstance(report_tables, Mapping) else {}
                    ),
                }
            )
        return packets

    @staticmethod
    def _capture_reporting_packet(runtime: ExperimentRuntime) -> None:
        if not isinstance(runtime._result, Mapping):
            return
        reporting = runtime._result.get("reporting_handoff")
        if not isinstance(reporting, Mapping):
            return
        facts = reporting.get("report_facts")
        tables = reporting.get("report_tables")
        if not isinstance(facts, Mapping) and not isinstance(tables, Mapping):
            return
        packets = runtime.runtime_state.setdefault("reporting_packets", [])
        packets.append(
            {
                "agent": runtime.agent_name,
                "tool_name": runtime.tool_name,
                "call_id": runtime.call_id,
                "recorded_at": utc_now(),
                "report_facts": json_safe(facts) if isinstance(facts, Mapping) else None,
                "report_tables": json_safe(tables) if isinstance(tables, Mapping) else {},
            }
        )
        if len(packets) > _MAX_TOOL_EVENTS:
            del packets[: len(packets) - _MAX_TOOL_EVENTS]

    @staticmethod
    def _persistence_requirement(result: Mapping[str, Any]) -> dict[str, Any] | None:
        if result.get("persisted") is True:
            return None
        plan = result.get("persistence_plan")
        plan = plan if isinstance(plan, Mapping) else {}
        manifest_info = result.get("candidate_persistence_manifest")
        manifest_info = manifest_info if isinstance(manifest_info, Mapping) else {}
        manifest_path = (
            plan.get("candidate_manifest_path")
            or result.get("candidate_manifest_path")
            or manifest_info.get("path")
        )
        candidate_count = plan.get("candidate_count", manifest_info.get("candidate_count"))
        try:
            has_candidates = candidate_count is None or int(candidate_count) > 0
        except (TypeError, ValueError):
            has_candidates = False
        if plan.get("persist_all_candidates") is True and manifest_path and has_candidates:
            return {
                "mode": "candidate_manifest",
                "required_tools": [_BATCH_PERSISTENCE_TOOL],
                "candidate_manifest_path": str(manifest_path),
                "candidate_count": candidate_count,
            }

        payload = result.get("recommended_registry_payload")
        if isinstance(payload, Mapping) and payload.get("model_id") and payload.get("model_path"):
            return {
                "mode": "single_model",
                "required_tools": [_REGISTER_MODEL_TOOL],
                "recommended_model_id": str(payload["model_id"]),
                "recommended_model_path": str(payload["model_path"]),
            }
        return None

    def _capture_persistence_requirement(self, runtime: ExperimentRuntime) -> None:
        if runtime.agent_name != "qsaria_training" or not isinstance(runtime._result, Mapping):
            return
        requirement = self._persistence_requirement(runtime._result)
        if requirement is None:
            return
        metadata = runtime.public_state.get("metadata") or {}
        policy = (
            str(metadata.get("persistence_policy") or _DEFAULT_PERSISTENCE_POLICY)
            if isinstance(metadata, Mapping)
            else _DEFAULT_PERSISTENCE_POLICY
        )
        now = utc_now()
        if policy == "session_only":
            runtime.public_state["persistence"] = {
                "policy": "session_only",
                "status": "skipped_by_user",
                "source_agent": runtime.agent_name,
                "source_tool": runtime.tool_name,
                "source_call_id": runtime.call_id,
                "recorded_at": now,
            }
            return
        pending = {
            "schema_version": "1.0",
            "policy": _DEFAULT_PERSISTENCE_POLICY,
            "status": "pending",
            "source_agent": runtime.agent_name,
            "source_tool": runtime.tool_name,
            "source_call_id": runtime.call_id,
            "created_at": now,
            "updated_at": now,
            **requirement,
        }
        runtime.runtime_state["pending_persistence"] = pending
        runtime.public_state["persistence"] = json_safe(pending)

    @staticmethod
    def _persistence_result_evidence(result: Mapping[str, Any]) -> dict[str, Any]:
        candidates = result.get("candidates")
        return {
            "persisted": result.get("persisted") is True,
            "model_ids": extract_model_ids(result),
            "model_id": result.get("model_id"),
            "model_root": result.get("model_root"),
            "model_path": result.get("model_path"),
            "metadata_path": result.get("metadata_path"),
            "candidates": json_safe(candidates) if isinstance(candidates, list) else [],
        }

    def _update_persistence_after_registry(
        self,
        runtime: ExperimentRuntime,
        *,
        outcome: str,
    ) -> None:
        pending = self._pending_persistence(runtime.runtime_state)
        if pending is None or runtime.agent_name != "qsaria_registry":
            return
        now = utc_now()
        if outcome == "terminal_failure":
            failed = {
                **dict(pending),
                "status": "terminal_failure",
                "failed_tool": runtime.tool_name,
                "error": runtime._error,
                "updated_at": now,
            }
            runtime.runtime_state.pop("pending_persistence", None)
            runtime.runtime_state.setdefault("persistence_events", []).append(json_safe(failed))
            runtime.public_state["persistence"] = json_safe(failed)
            return
        if outcome != "success" or not isinstance(runtime._result, Mapping):
            return

        result = runtime._result
        if (
            pending.get("mode") == "single_model"
            and runtime.tool_name == _REGISTER_MODEL_TOOL
            and result.get("registered") is True
        ):
            updated = {
                **dict(pending),
                "status": "pending",
                "required_tools": [_PERSIST_MODEL_TOOL],
                "registered_model_id": result.get("model_id"),
                "updated_at": now,
            }
            runtime.runtime_state["pending_persistence"] = updated
            runtime.public_state["persistence"] = json_safe(updated)
            return

        completed = False
        if pending.get("mode") == "candidate_manifest":
            completed = (
                runtime.tool_name == _BATCH_PERSISTENCE_TOOL
                and result.get("persisted") is True
                and int(result.get("candidate_count") or 0) > 0
            )
        elif pending.get("mode") == "single_model":
            completed = runtime.tool_name == _PERSIST_MODEL_TOOL and result.get("persisted") is True
        if not completed:
            return

        evidence = self._persistence_result_evidence(result)
        completed_event = {
            **dict(pending),
            "status": "completed",
            "completed_tool": runtime.tool_name,
            "completed_call_id": runtime.call_id,
            "completed_at": now,
            "updated_at": now,
            "evidence": evidence,
        }
        runtime.runtime_state.pop("pending_persistence", None)
        runtime.runtime_state.setdefault("persistence_events", []).append(
            json_safe(completed_event)
        )
        runtime.public_state["persistence"] = json_safe(completed_event)

    @staticmethod
    def _validate_run_transition(
        public_state: Mapping[str, Any],
        runtime_state: Mapping[str, Any],
        *,
        agent_name: str,
        tool_name: str,
    ) -> bool:
        """Reject silent continuation past final states and bound retries."""

        status = str(public_state.get("status") or "created")
        if status == "terminal_failure" or ExperimentManager._is_finalized_state(public_state):
            raise ExperimentStateTransitionError(
                f"experiment status {status!r} does not allow another scientific call"
            )
        if public_state.get("report") or any(
            isinstance(item, Mapping) and item.get("agent") == "qsaria_report"
            for item in public_state.get("handoffs") or []
        ):
            raise ExperimentStateTransitionError(
                "the final Report phase has started; no later scientific call is allowed"
            )
        pending_handoff_agent = ExperimentManager._pending_handoff_agent(runtime_state)
        if pending_handoff_agent is not None and pending_handoff_agent != agent_name:
            raise ExperimentStateTransitionError(
                f"role {pending_handoff_agent!r} must record its structured handoff before "
                f"role {agent_name!r} can call {tool_name!r}"
            )
        pending_persistence = ExperimentManager._pending_persistence(runtime_state)
        if pending_persistence is not None and pending_handoff_agent != "qsaria_training":
            required_tools = pending_persistence["required_tools"]
            if agent_name != "qsaria_registry" or tool_name not in required_tools:
                raise ExperimentStateTransitionError(
                    "catalog persistence must be completed before another scientific role; "
                    f"qsaria_registry must call one of {required_tools!r}, not {tool_name!r}"
                )
        if status not in {"retryable_error", "needs_user_input", "running"}:
            return False

        pending = runtime_state.get("pending_failure")
        expected_tool = (
            str(pending.get("tool_name") or "")
            if isinstance(pending, Mapping)
            else str(public_state.get("phase") or "")
        )
        expected_agent = str(pending.get("agent") or "") if isinstance(pending, Mapping) else ""
        if expected_agent and expected_agent != agent_name:
            raise ExperimentStateTransitionError(
                f"experiment status {status!r} can only be resolved by role "
                f"{expected_agent!r}, not {agent_name!r}"
            )
        if expected_tool and expected_tool != tool_name:
            raise ExperimentStateTransitionError(
                f"experiment status {status!r} can only be resolved by {expected_tool!r}, "
                f"not {tool_name!r}"
            )
        if status in {"retryable_error", "needs_user_input"} and isinstance(pending, Mapping):
            if pending.get("handoff_recorded") is not True:
                raise ExperimentStateTransitionError(
                    f"experiment status {status!r} requires its structured agent handoff "
                    "before another scientific call"
                )
        if status == "retryable_error" and isinstance(pending, Mapping):
            if int(pending.get("automatic_retries") or 0) >= 1:
                raise ExperimentStateTransitionError(
                    f"the single automatic retry for {tool_name!r} has already been used"
                )
        return status in {"retryable_error", "running"}

    @staticmethod
    def _is_finalized_state(public_state: Mapping[str, Any]) -> bool:
        return bool(public_state.get("completed_at")) or public_state.get("phase") == "completed"

    def _finalize_runtime(self, runtime: ExperimentRuntime) -> None:
        now = utc_now()
        runtime.runtime_state.pop("active_call", None)
        runtime.runtime_state["session_state"] = json_safe(runtime.context.session_state)
        runtime.runtime_state["updated_at"] = now
        outcome = runtime._outcome_status or "success"
        if outcome == "retryable_error" and runtime._automatic_retry:
            outcome = "terminal_failure"
            original = runtime._error or "transient failure"
            runtime._error = f"automatic retry exhausted after one retry: {original}"
        artifact_ids: list[str] = []
        if runtime._result is not None:
            artifact_ids = self._register_payload(
                runtime,
                runtime._result,
                source=runtime.tool_name,
            )
        self._capture_reporting_packet(runtime)
        if outcome == "success":
            self._capture_persistence_requirement(runtime)
        self._update_persistence_after_registry(runtime, outcome=outcome)
        model_ids = self._merge_strings(
            runtime._input_model_ids,
            extract_model_ids(runtime._result),
        )
        if model_ids:
            runtime.public_state["model_ids"] = self._merge_strings(
                runtime.public_state.get("model_ids"),
                model_ids,
            )
            provenance = runtime.runtime_state.setdefault("model_provenance", {})
            for model_id in model_ids:
                provenance.setdefault(
                    model_id,
                    {
                        "source": runtime.tool_name,
                        "agent": runtime.agent_name,
                        "kind": (
                            "validated_input"
                            if model_id in runtime._input_model_ids
                            else "tool_result"
                        ),
                        "recorded_at": now,
                    },
                )

        event = {
            "call_id": runtime.call_id,
            "started_at": runtime.started_at,
            "completed_at": now,
            "duration_ms": runtime.duration_ms,
            "agent": runtime.agent_name,
            "tool_name": runtime.tool_name,
            "status": outcome,
            "automatic_retry": runtime._automatic_retry,
            "error": runtime._error,
            "result_summary": self._result_summary(runtime._result),
            "artifact_ids": artifact_ids,
            "model_ids": model_ids,
        }
        events = runtime.runtime_state.setdefault("tool_events", [])
        events.append(json_safe(event))
        if len(events) > _MAX_TOOL_EVENTS:
            del events[: len(events) - _MAX_TOOL_EVENTS]

        if outcome != "success":
            runtime.public_state["status"] = outcome
            if runtime._error:
                runtime.public_state["blockers"] = self._merge_strings(
                    runtime.public_state.get("blockers"), [runtime._error]
                )
            runtime.runtime_state["pending_failure"] = {
                "agent": runtime.agent_name,
                "tool_name": runtime.tool_name,
                "status": outcome,
                "error": runtime._error,
                "errors": [runtime._error] if runtime._error else [],
                "automatic_retries": 1 if runtime._automatic_retry else 0,
                "handoff_recorded": False,
                "updated_at": now,
            }
        elif runtime.public_state.get("status") == "running":
            runtime.public_state["status"] = "active"
            pending = runtime.runtime_state.pop("pending_failure", None)
            if isinstance(pending, Mapping):
                pending_errors = pending.get("errors") or [pending.get("error")]
                runtime.public_state["blockers"] = [
                    blocker
                    for blocker in runtime.public_state.get("blockers") or []
                    if blocker not in pending_errors
                ]
        if outcome == "success":
            pending_handoff = runtime.runtime_state.get("pending_handoff")
            if pending_handoff is None:
                call_ids: list[str] = []
                tool_names: list[str] = []
                started_at = now
            elif isinstance(pending_handoff, Mapping):
                pending_agent = self._pending_handoff_agent(runtime.runtime_state)
                if pending_agent != runtime.agent_name:
                    raise ExperimentStateTransitionError(
                        f"role {pending_agent!r} still owes a structured handoff"
                    )
                call_ids = list(pending_handoff.get("call_ids") or [])
                tool_names = list(pending_handoff.get("tool_names") or [])
                started_at = str(pending_handoff.get("started_at") or now)
            else:  # pragma: no cover - validated by _pending_handoff_agent
                raise ExperimentStateTransitionError(
                    "runtime pending_handoff evidence is malformed"
                )
            runtime.runtime_state["pending_handoff"] = {
                "agent": runtime.agent_name,
                "call_ids": self._merge_strings(call_ids, [runtime.call_id]),
                "tool_names": self._merge_strings(tool_names, [runtime.tool_name]),
                "started_at": started_at,
                "updated_at": now,
            }
        runtime.public_state["phase"] = runtime.tool_name
        runtime.public_state["updated_at"] = now

        manifest_rel = PurePosixPath(
            "workflows",
            "manifests",
            "qsaria",
            f"{runtime.started_at.replace(':', '').replace('-', '')}_{runtime.tool_name}_{runtime.call_id}.json",
        ).as_posix()
        self._write_json(runtime.experiment_id, manifest_rel, event)
        self._save_states(runtime.experiment_id, runtime.public_state, runtime.runtime_state)

    def _register_payload(
        self,
        runtime: ExperimentRuntime,
        payload: Any,
        *,
        source: str,
    ) -> list[str]:
        artifact_ids: list[str] = []
        for raw_path in extract_artifact_paths(payload):
            path = self._normalise_artifact_path(runtime.experiment_id, raw_path)
            if not self._is_managed_artifact_path(runtime.experiment_id, path):
                self._register_provenance(runtime, raw_path, source=source)
                continue
            entry = self._register_artifact(
                runtime.experiment_id,
                path,
                runtime.runtime_state,
                source=source,
            )
            artifact_ids = self._merge_strings(artifact_ids, [entry["artifact_id"]])
            runtime.public_state["artifact_ids"] = self._merge_strings(
                runtime.public_state.get("artifact_ids"), [entry["artifact_id"]]
            )
        runtime.public_state["model_ids"] = self._merge_strings(
            runtime.public_state.get("model_ids"), extract_model_ids(payload)
        )
        return artifact_ids

    def _register_provenance(
        self,
        runtime: ExperimentRuntime,
        raw_path: str,
        *,
        source: str,
    ) -> None:
        """Record a non-retrievable fingerprint for an external input reference."""

        text = str(raw_path or "").strip()
        parsed = urlsplit(text)
        name = PurePosixPath(parsed.path).name if parsed.scheme else Path(text).name
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:20]
        provenance_id = f"prov_{digest}"
        runtime.runtime_state.setdefault("provenance_inputs", {})[provenance_id] = {
            "provenance_id": provenance_id,
            "name": name or None,
            "scheme": parsed.scheme or "local",
            "source": source,
            "retrievable": False,
            "recorded_at": utc_now(),
        }

    def _register_artifact(
        self,
        experiment_id: str,
        raw_path: str,
        runtime_state: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        path = self._normalise_artifact_path(experiment_id, raw_path)
        if not self._is_managed_artifact_path(experiment_id, path):
            raise InvalidInputPathError("only managed Qsaria outputs can become artifacts")
        artifact_id = artifact_id_for(experiment_id, path)
        stat = self._artifact_stat(experiment_id, path)
        existing = (runtime_state.get("artifacts") or {}).get(artifact_id) or {}
        entry = {
            "artifact_id": artifact_id,
            "path": path,
            "kind": self._artifact_kind(path),
            "media_type": mimetypes.guess_type(path)[0] or "application/octet-stream",
            "source": existing.get("source") or source,
            "exists": stat is not None,
            "size": stat.get("size") if stat else None,
            "updated_at": existing.get("updated_at") or utc_now(),
        }
        comparable_existing = {key: value for key, value in existing.items() if key != "updated_at"}
        comparable_entry = {key: value for key, value in entry.items() if key != "updated_at"}
        if existing and comparable_existing != comparable_entry:
            entry["updated_at"] = utc_now()
        runtime_state.setdefault("artifacts", {})[artifact_id] = entry
        return entry

    def _discover_artifacts(
        self,
        experiment_id: str,
        public_state: dict[str, Any],
        runtime_state: dict[str, Any],
    ) -> bool:
        before = json.dumps(runtime_state.get("artifacts") or {}, sort_keys=True)
        self._refresh_known_artifacts(experiment_id, runtime_state)
        for rel_path in self._list_session_files(experiment_id):
            if self._is_internal_state_path(rel_path):
                continue
            entry = self._register_artifact(
                experiment_id,
                rel_path,
                runtime_state,
                source="discovered",
            )
            public_state["artifact_ids"] = self._merge_strings(
                public_state.get("artifact_ids"), [entry["artifact_id"]]
            )
        after = json.dumps(runtime_state.get("artifacts") or {}, sort_keys=True)
        if before != after:
            now = utc_now()
            public_state["updated_at"] = now
            runtime_state["updated_at"] = now
            return True
        return False

    def _refresh_known_artifacts(
        self,
        experiment_id: str,
        runtime_state: dict[str, Any],
    ) -> None:
        """Refresh existence and size for every previously registered artifact."""

        artifacts = runtime_state.setdefault("artifacts", {})
        if not isinstance(artifacts, dict):
            raise ExperimentStateTransitionError("runtime artifact registry is malformed")
        for artifact_id, raw_entry in list(artifacts.items()):
            if not isinstance(raw_entry, Mapping):
                raise ExperimentStateTransitionError(
                    f"artifact registry entry {artifact_id!r} is malformed"
                )
            path = raw_entry.get("path")
            if not isinstance(path, str) or artifact_id != artifact_id_for(experiment_id, path):
                raise ExperimentStateTransitionError(
                    f"artifact registry entry {artifact_id!r} has inconsistent identity"
                )
            stat = self._artifact_stat(experiment_id, path)
            refreshed = dict(raw_entry)
            refreshed["exists"] = stat is not None
            refreshed["size"] = stat.get("size") if stat else None
            comparable_before = {
                key: value for key, value in raw_entry.items() if key != "updated_at"
            }
            comparable_after = {
                key: value for key, value in refreshed.items() if key != "updated_at"
            }
            if comparable_before != comparable_after:
                refreshed["updated_at"] = utc_now()
            artifacts[artifact_id] = refreshed

    def _list_session_files(self, experiment_id: str) -> list[str]:
        root = self._safe_session_root(experiment_id)
        resolved_root = root.resolve(strict=False)
        local_files = (
            [
                path.relative_to(root).as_posix()
                for path in sorted(root.rglob("*"))
                if path.is_file() and path.resolve(strict=False).is_relative_to(resolved_root)
            ]
            if root.exists()
            else []
        )
        if not self._uses_s3():
            return local_files

        from cs_copilot.storage import get_s3_config

        config = get_s3_config()
        try:
            import s3fs

            fs = s3fs.S3FileSystem(**config.to_storage_options())
            root = f"{config.bucket_name}/{EXPERIMENTS_PREFIX}/{experiment_id}"
            keys = fs.find(root)
        except FileNotFoundError:
            return []
        prefix = f"{config.bucket_name}/{EXPERIMENTS_PREFIX}/{experiment_id}/"
        remote_files = [str(key)[len(prefix) :] for key in keys if str(key).startswith(prefix)]
        return sorted(set(local_files) | set(remote_files))

    def _discover_experiment_ids(self) -> list[str]:
        if not self._uses_s3():
            sessions_root = self._local_root() / EXPERIMENTS_PREFIX
            if not sessions_root.exists():
                return []
            ids = []
            for path in sessions_root.glob(f"*/{PUBLIC_STATE_REL_PATH}"):
                try:
                    ids.append(validate_experiment_id(path.parents[1].name))
                except ValueError:
                    continue
            return sorted(set(ids))

        from cs_copilot.storage import get_s3_config

        config = get_s3_config()
        try:
            import s3fs

            fs = s3fs.S3FileSystem(**config.to_storage_options())
            root = f"{config.bucket_name}/{EXPERIMENTS_PREFIX}"
            keys = fs.find(root)
        except FileNotFoundError:
            return []
        ids: list[str] = []
        suffix = f"/{PUBLIC_STATE_REL_PATH}"
        prefix = f"{config.bucket_name}/{EXPERIMENTS_PREFIX}/"
        for key in keys:
            text = str(key)
            if not text.startswith(prefix) or not text.endswith(suffix):
                continue
            candidate = text[len(prefix) : -len(suffix)]
            try:
                ids.append(validate_experiment_id(candidate))
            except ValueError:
                continue
        return sorted(set(ids))

    def _normalise_artifact_path(self, experiment_id: str, raw_path: str) -> str:
        text = str(raw_path or "").strip()
        if text.startswith("<file>") and text.endswith("</file>"):
            text = text[6:-7].strip()
        session_prefix = f"{EXPERIMENTS_PREFIX}/{experiment_id}/"
        if text.startswith("s3://"):
            if self._uses_s3():
                from cs_copilot.storage import get_s3_config

                parsed = urlsplit(text)
                marker = f"/{session_prefix}"
                if parsed.netloc == get_s3_config().bucket_name and parsed.path.startswith(marker):
                    return parsed.path[len(marker) :]
            return text
        if text.startswith("file://"):
            text = text[7:]
        path = Path(text).expanduser()
        session_root = self._session_root(experiment_id)
        if path.is_absolute():
            resolved = path.resolve(strict=False)
            try:
                return resolved.relative_to(session_root.resolve()).as_posix()
            except ValueError:
                pass
            model_root = Path("data/model_assets/internal").resolve()
            try:
                model_relative = resolved.relative_to(model_root).as_posix()
            except ValueError:
                return str(resolved)
            return PurePosixPath(_GLOBAL_MODEL_ARTIFACT_PREFIX, model_relative).as_posix()
        normalized = path.as_posix().lstrip("./")
        for marker in (
            f"{self._local_root().as_posix().strip('/')}/{session_prefix}",
            f".files/{session_prefix}",
            session_prefix,
        ):
            marker = marker.lstrip("./")
            if normalized.startswith(marker):
                return normalized[len(marker) :]
        return normalized

    def _artifact_stat(self, experiment_id: str, path: str) -> dict[str, Any] | None:
        candidate = self._local_artifact_path(experiment_id, path)
        if candidate is not None and candidate.is_file():
            return {"size": candidate.stat().st_size}
        if not self._uses_s3():
            return None
        try:
            import s3fs

            from cs_copilot.storage import S3, get_s3_config

            config = get_s3_config()
            fs = s3fs.S3FileSystem(**config.to_storage_options())
            if self._safe_relative(path):
                token = self._bind_storage_prefix(experiment_id)
                try:
                    resolved = S3.path(path)
                finally:
                    self._reset_storage_prefix(token)
            else:
                return None
            info = fs.info(resolved)
            return {"size": info.get("size")}
        except (FileNotFoundError, OSError):
            return None

    def _read_artifact_bytes(self, experiment_id: str, path: str, limit: int) -> bytes:
        candidate = self._local_artifact_path(experiment_id, path)
        if candidate is not None and candidate.is_file():
            with candidate.open("rb") as handle:
                return handle.read(limit)
        if not self._uses_s3() or Path(path).is_absolute():
            raise FileNotFoundError(f"unsafe or unavailable artifact path: {path}")
        from cs_copilot.storage import S3

        token = self._bind_storage_prefix(experiment_id)
        try:
            with S3.open(path, "rb") as handle:
                blob = handle.read(limit)
        finally:
            self._reset_storage_prefix(token)
        return blob.encode() if isinstance(blob, str) else blob

    def _local_artifact_path(self, experiment_id: str, path: str) -> Path | None:
        if path.startswith("s3://"):
            return None
        candidate = Path(path).expanduser()
        if candidate.parts and candidate.parts[0] == _GLOBAL_MODEL_ARTIFACT_PREFIX:
            relative = Path(*candidate.parts[1:])
            if not self._safe_relative(relative.as_posix()):
                return None
            root = Path("data/model_assets/internal").resolve()
            resolved = (root / relative).resolve(strict=False)
            return resolved if resolved.is_relative_to(root) else None
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            allowed_roots = (
                self._session_root(experiment_id).resolve(),
                Path("data/model_assets/internal").resolve(),
            )
            for root in allowed_roots:
                try:
                    resolved.relative_to(root)
                except ValueError:
                    continue
                return resolved
            return None
        if not self._safe_relative(path):
            return None
        root = self._safe_session_root(experiment_id)
        resolved = (root / candidate).resolve(strict=False)
        return resolved if resolved.is_relative_to(root) else None

    def _is_managed_artifact_path(self, experiment_id: str, path: str) -> bool:
        if path.startswith("s3://"):
            return False
        return self._artifact_stat(experiment_id, path) is not None

    @staticmethod
    def _safe_relative(path: str) -> bool:
        candidate = PurePosixPath(path)
        return not candidate.is_absolute() and ".." not in candidate.parts

    @staticmethod
    def _artifact_kind(path: str) -> str:
        suffix = Path(path).suffix.lower()
        if suffix in {".md", ".pdf", ".html"} or "report" in path.lower():
            return "report"
        if suffix in {".csv", ".tsv", ".parquet"}:
            return "dataset"
        if suffix in {".png", ".jpg", ".jpeg", ".svg"}:
            return "visualization"
        if suffix in {".pkl", ".pickle", ".joblib", ".pt", ".ckpt"}:
            return "model"
        if suffix in {".json", ".jsonl", ".yaml", ".yml", ".toml"}:
            return "metadata"
        if suffix == ".zip":
            return "bundle"
        return "artifact"

    @staticmethod
    def _is_text_mime(media_type: str) -> bool:
        return media_type.startswith("text/") or media_type in {
            "application/json",
            "application/ld+json",
            "application/xml",
            "application/x-yaml",
            "application/toml",
        }

    @staticmethod
    def _is_internal_state_path(path: str) -> bool:
        normalized = path.strip("/")
        return (
            normalized
            in {
                PUBLIC_STATE_REL_PATH,
                RUNTIME_STATE_REL_PATH,
                "qsaria/.experiment.lock",
            }
            or "/manifests/qsaria/" in f"/{normalized}/"
        )

    @staticmethod
    def _merge_strings(*groups: Any) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()
        for group in groups:
            values = [group] if isinstance(group, str) else (group or [])
            for value in values:
                text = str(value).strip()
                if text and text not in seen:
                    seen.add(text)
                    result.append(text)
        return result

    @staticmethod
    def _unresolved_partial_agents(handoffs: list[Mapping[str, Any]]) -> list[str]:
        latest_scientific_status: dict[str, str] = {}
        for handoff in handoffs:
            if not isinstance(handoff, Mapping):
                continue
            agent = str(handoff.get("agent") or "")
            if not agent:
                continue
            latest_scientific_status[agent] = str(handoff.get("status") or "")
        return sorted(
            agent for agent, status in latest_scientific_status.items() if status == "partial"
        )

    @staticmethod
    def _state_status_after_handoff(
        status: str,
        handoffs: list[Mapping[str, Any]],
    ) -> str:
        if status in {"retryable_error", "terminal_failure", "needs_user_input"}:
            return status
        if ExperimentManager._unresolved_partial_agents(handoffs):
            return "partial"
        return "active"

    @staticmethod
    def _result_summary(result: Any) -> dict[str, Any]:
        if result is None:
            return {"type": "none"}
        if isinstance(result, Mapping):
            summary: dict[str, Any] = {
                "type": "mapping",
                "keys": [str(key) for key in list(result)[:40]],
            }
            for key in ("status", "row_count", "num_rows", "model_id", "report_kind"):
                if key in result:
                    summary[key] = json_safe(result[key])
            return summary
        if isinstance(result, (list, tuple)):
            return {"type": type(result).__name__, "length": len(result)}
        return {"type": type(result).__name__, "preview": str(result)[:500]}


_DEFAULT_MANAGER: ExperimentManager | None = None
_DEFAULT_MANAGER_LOCK = threading.Lock()


def get_experiment_manager() -> ExperimentManager:
    """Return the process-wide manager used by MCP lifecycle/tool facades."""

    global _DEFAULT_MANAGER
    with _DEFAULT_MANAGER_LOCK:
        if _DEFAULT_MANAGER is None:
            _DEFAULT_MANAGER = ExperimentManager()
        return _DEFAULT_MANAGER


__all__ = [
    "PUBLIC_STATE_REL_PATH",
    "RUNTIME_STATE_REL_PATH",
    "ExperimentManager",
    "ExperimentRuntime",
    "get_experiment_manager",
]
