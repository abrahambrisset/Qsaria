"""Experiment-aware adapter for deterministic Qsaria toolkit methods."""

from __future__ import annotations

import asyncio
import errno
import inspect
import json
import logging
import os
import traceback
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping

from cs_copilot.mcp.context import MCPAgentContext
from cs_copilot.mcp.errors import MCPToolError
from cs_copilot.mcp.tool_adapter import (
    INJECTED_PARAMS,
    ToolSpec,
    _coerce_return_value,
    _resolve_annotations,
)

from .contracts import InvalidInputPathError
from .toolkit_hub import catalog_scope

if TYPE_CHECKING:
    from .experiments import ExperimentManager

logger = logging.getLogger(__name__)

_TERMINAL_EXTERNAL_EVALUATION_STATUS = "blocked_failed_external_evaluation"
_MISSING_EXTERNAL_TARGETS_MARKER = "missing required target columns"
_UNKNOWN_MODEL_ERROR_MARKERS = (
    "unknown catalog model_id",
    "unknown model_id",
)
_CORRECTABLE_INFERENCE_VALUE_ERROR_TOOLS = frozenset(
    {
        "qsaria_inference_predict_from_csv",
        "qsaria_inference_predict_from_smiles",
        "qsaria_inference_export_prediction_summary",
    }
)
_PATH_KEY_SUFFIXES = (
    "_path",
    "_paths",
    "_csv",
    "_csvs",
    "_file",
    "_files",
    "_dir",
    "_dirs",
    "_json",
    "_parquet",
    "_pickle",
    "_pkl",
    "_zip",
    "_html",
    "_pdf",
    "_png",
    "_svg",
    "_file_ref",
)
_TRANSIENT_ERRNOS = frozenset(
    value
    for value in (
        errno.EAGAIN,
        errno.EBUSY,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.ECONNRESET,
        errno.EHOSTUNREACH,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ETIMEDOUT,
    )
    if value is not None
)
_MODEL_IDENTIFIER_KEYS = frozenset(
    {
        "model_id",
        "model_ids",
        "registered_model_ids",
        "component_model_ids",
    }
)
_EXPLICIT_PATH_KEYS = frozenset({"feature_columns_source"})
_ARTIFACT_PATH_MAP_KEYS = frozenset(
    {
        "artifacts",
        "loop_comparison_plot_artifacts",
        "plot_artifacts",
    }
)
_ARTIFACT_METADATA_KEYS = frozenset(
    {
        # Outlier-analysis annotations live below an ``artifacts`` mapping,
        # but this value describes the scientific data scope rather than a
        # filesystem location.  Keep arbitrary artifact labels path-checked
        # while allowing this explicit metadata field to remain a string.
        "scope",
    }
)


@dataclass(frozen=True)
class QsariaToolSpec(ToolSpec):
    """A ToolSpec whose calls execute inside a durable Qsaria experiment.

    ``output_paths`` maps hidden toolkit parameters (for example
    ``output_dir``) to filenames/directories allocated by
    :class:`ExperimentRuntime`.  This prevents an MCP client from writing
    scientific outputs outside the active experiment.
    """

    agent_name: str = "qsaria"
    output_paths: Mapping[str, str] = field(default_factory=dict)
    nested_output_paths: Mapping[str, str] = field(default_factory=dict)
    nested_blocked_inputs: tuple[str, ...] = ()
    registered_artifact_inputs: Mapping[
        str,
        tuple[str, tuple[str, ...]],
    ] = field(default_factory=dict)
    json_payload_inputs: tuple[str, ...] = ()
    session_forces: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    artifact_category: str = "artifacts"
    catalog_access: bool = False
    catalog_write: bool = False
    requires_experiment: bool = True


def is_qsaria_spec(spec: ToolSpec) -> bool:
    """Return whether ``spec`` requires the Qsaria experiment adapter."""

    return isinstance(spec, QsariaToolSpec)


def _public_signature(
    bound_method: Callable[..., Any],
    spec: QsariaToolSpec,
) -> tuple[inspect.Signature, dict[str, Any], inspect.Signature]:
    """Return the underlying and public signatures plus resolved annotations."""

    underlying = inspect.signature(bound_method)
    resolved = _resolve_annotations(bound_method)
    hidden = (
        set(INJECTED_PARAMS) | set(spec.forces) | set(spec.output_paths) | set(spec.session_forces)
    )
    public_params: list[inspect.Parameter] = []
    if spec.requires_experiment:
        public_params.append(
            inspect.Parameter(
                "experiment_id",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            )
        )
    for name, parameter in underlying.parameters.items():
        if name == "self" or name in hidden:
            continue
        annotation = resolved.get(name, parameter.annotation)
        public_params.append(parameter.replace(annotation=annotation))

    return_annotation = resolved.get("return", underlying.return_annotation)
    public = inspect.Signature(public_params, return_annotation=return_annotation)
    return underlying, resolved, public


def _call_annotations(signature: inspect.Signature) -> dict[str, Any]:
    annotations = {
        parameter.name: parameter.annotation
        for parameter in signature.parameters.values()
        if parameter.annotation is not inspect.Parameter.empty
    }
    if signature.return_annotation is not inspect.Signature.empty:
        annotations["return"] = signature.return_annotation
    return annotations


def _forced_output_paths(spec: QsariaToolSpec, run: Any) -> dict[str, str]:
    """Allocate every output parameter beneath the active experiment."""

    return {
        parameter: run.artifact_path(filename, category=spec.artifact_category)
        for parameter, filename in spec.output_paths.items()
    }


def _without_nested_output_paths(
    call_kwargs: dict[str, Any],
    spec: QsariaToolSpec,
) -> dict[str, Any]:
    """Remove user-controlled nested write locations before path validation."""

    prepared = dict(call_kwargs)
    for dotted_name in set(spec.nested_output_paths) | set(spec.nested_blocked_inputs):
        parent_name, separator, child_name = dotted_name.partition(".")
        if not separator or not isinstance(prepared.get(parent_name), Mapping):
            continue
        parent = dict(prepared[parent_name])
        parent.pop(child_name, None)
        prepared[parent_name] = parent
    return prepared


def _without_session_forces(
    call_kwargs: dict[str, Any],
    spec: QsariaToolSpec,
) -> dict[str, Any]:
    prepared = dict(call_kwargs)
    for parameter in spec.session_forces:
        prepared.pop(parameter, None)
    return prepared


def _without_forced_parameters(
    call_kwargs: dict[str, Any],
    spec: QsariaToolSpec,
) -> dict[str, Any]:
    prepared = dict(call_kwargs)
    for parameter in set(spec.forces) | set(spec.output_paths):
        prepared.pop(parameter, None)
    return prepared


def _session_forced_values(spec: QsariaToolSpec, run: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for parameter, state_path in spec.session_forces.items():
        current: Any = run.context.session_state
        for part in state_path:
            if not isinstance(current, Mapping) or part not in current:
                dotted = ".".join(state_path)
                raise KeyError(f"{spec.mcp_name} requires persisted experiment state {dotted!r}")
            current = current[part]
        values[parameter] = current
    return values


def _inject_nested_output_paths(
    call_kwargs: dict[str, Any],
    spec: QsariaToolSpec,
    run: Any,
) -> None:
    """Route supported nested write locations into the active experiment."""

    for dotted_name, filename in spec.nested_output_paths.items():
        parent_name, separator, child_name = dotted_name.partition(".")
        if not separator:
            raise RuntimeError(f"nested output path must use parent.child syntax: {dotted_name}")
        parent = dict(call_kwargs.get(parent_name) or {})
        parent[child_name] = run.artifact_path(filename, category=spec.artifact_category)
        call_kwargs[parent_name] = parent


def _classify_result(result: Any) -> str:
    """Map explicit scientific result statuses onto the experiment state machine."""

    if not isinstance(result, Mapping):
        return "success"
    status = str(result.get("status") or "").strip().lower()
    if status == "retryable_error":
        return "retryable_error"
    if status in {"needs_user_input", "input_required", "awaiting_user_input"}:
        return "needs_user_input"
    if status in {
        "blocked",
        "error",
        "failed",
        "failure",
        "invalid",
        "terminal_failure",
        _TERMINAL_EXTERNAL_EVALUATION_STATUS,
    } or status.startswith(("blocked_", "failed_")):
        return "terminal_failure"
    if result.get("blocked") is True:
        return (
            "needs_user_input" if result.get("benchmark_started") is False else "terminal_failure"
        )
    if result.get("registered") is False and result.get("error"):
        return "needs_user_input"
    if result.get("persisted") is False and (result.get("error") or result.get("reason")):
        return "terminal_failure"
    if result.get("found") is False and result.get("error"):
        return "needs_user_input"
    if result.get("success") is False and result.get("error"):
        return "terminal_failure"
    return "success"


def _result_failure_message(result: Mapping[str, Any], outcome: str) -> str:
    for key in ("error", "message", "reason", "blocker", "blockers"):
        value = result.get(key)
        if value:
            return str(value)[:2000]
    return f"scientific result returned {outcome}: {result.get('status') or 'unspecified'}"


def _is_terminal_error(spec: QsariaToolSpec, error: BaseException) -> bool:
    message = str(error).lower()
    return (
        spec.mcp_name == "qsaria_inference_evaluate_model_on_dataset"
        and _MISSING_EXTERNAL_TARGETS_MARKER in message
    ) or _TERMINAL_EXTERNAL_EVALUATION_STATUS in message


def _path_key(name: str) -> bool:
    normalized = str(name).strip().lower()
    return (
        normalized == "path"
        or normalized in _EXPLICIT_PATH_KEYS
        or normalized.endswith(_PATH_KEY_SUFFIXES)
    )


def _normalize_input_paths(
    value: Any,
    manager: "ExperimentManager",
    *,
    key: str = "",
    artifact_map: bool = False,
    experiment_id: str | None = None,
    runtime_state: Mapping[str, Any] | None = None,
) -> Any:
    """Normalize path-shaped MCP inputs while preserving ordinary payloads."""

    normalized_key = key.strip().lower()
    if artifact_map and normalized_key in _ARTIFACT_METADATA_KEYS:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or len(value) > 128
            or any(marker in value for marker in ("/", "\\", "\x00"))
        ):
            raise InvalidInputPathError(f"{key} must be a non-path artifact metadata value")
        return value
    if normalized_key in _MODEL_IDENTIFIER_KEYS and isinstance(value, str):
        if (
            value != value.strip()
            or not value
            or len(value) > 256
            or value in {".", ".."}
            or any(marker in value for marker in ("/", "\\", "\x00"))
        ):
            raise ValueError(f"{key} must be a path-safe model identifier")
        return value
    if (
        (artifact_map and normalized_key not in _ARTIFACT_METADATA_KEYS) or _path_key(key)
    ) and isinstance(value, (str, os.PathLike)):
        if experiment_id is not None and runtime_state is not None:
            return manager.normalize_experiment_input_path(
                experiment_id,
                value,
                runtime_state=runtime_state,
                parameter=key,
            )
        return manager.normalize_input_path(value, parameter=key)
    if isinstance(value, Mapping):
        child_artifact_map = artifact_map or normalized_key in _ARTIFACT_PATH_MAP_KEYS
        return {
            child_key: _normalize_input_paths(
                child,
                manager,
                key=str(child_key),
                artifact_map=child_artifact_map,
                experiment_id=experiment_id,
                runtime_state=runtime_state,
            )
            for child_key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [
            _normalize_input_paths(
                child,
                manager,
                key=key,
                artifact_map=artifact_map,
                experiment_id=experiment_id,
                runtime_state=runtime_state,
            )
            for child in value
        ]
        return tuple(normalized) if isinstance(value, tuple) else normalized
    return value


def _validate_registered_artifact_inputs(
    call_kwargs: dict[str, Any],
    spec: QsariaToolSpec,
    manager: "ExperimentManager",
    run: Any,
) -> None:
    """Bind privileged file-driven operations to artifacts from approved tools."""

    for parameter, policy in spec.registered_artifact_inputs.items():
        value = call_kwargs.get(parameter)
        if value in (None, ""):
            continue
        expected_filename, source_prefixes = policy
        call_kwargs[parameter] = manager.validate_registered_artifact_input(
            run.experiment_id,
            value,
            runtime_state=run.runtime_state,
            parameter=parameter,
            expected_filename=expected_filename,
            source_prefixes=source_prefixes,
        )


def _validate_json_payload_inputs(
    call_kwargs: Mapping[str, Any],
    spec: QsariaToolSpec,
    manager: "ExperimentManager",
    run: Any,
) -> None:
    """Validate every path-bearing value inside trusted JSON input artifacts."""

    for parameter in spec.json_payload_inputs:
        value = call_kwargs.get(parameter)
        if value in (None, ""):
            continue
        path = Path(str(value))
        try:
            if path.stat().st_size > 64 * 1024 * 1024:
                raise InvalidInputPathError(f"{parameter} exceeds the 64 MiB validation limit")
            payload = json.loads(path.read_text(encoding="utf-8"))
        except InvalidInputPathError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise InvalidInputPathError(
                f"{parameter} is not a valid readable JSON artifact"
            ) from exc
        _normalize_input_paths(
            payload,
            manager,
            experiment_id=run.experiment_id,
            runtime_state=run.runtime_state,
        )


def _iter_exception_chain(error: BaseException):
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _classify_error(spec: QsariaToolSpec, error: BaseException) -> str:
    """Classify only explicit transient failures as automatically retryable."""

    if _is_terminal_error(spec, error):
        return "terminal_failure"
    chain = tuple(_iter_exception_chain(error))
    if any(
        isinstance(item, (InvalidInputPathError, FileNotFoundError, PermissionError))
        for item in chain
    ):
        return "needs_user_input"
    value_errors = [item for item in chain if isinstance(item, ValueError)]
    if value_errors:
        messages = "\n".join(str(item).lower() for item in value_errors)
        if any(marker in messages for marker in _UNKNOWN_MODEL_ERROR_MARKERS):
            return "needs_user_input"
        if spec.mcp_name in _CORRECTABLE_INFERENCE_VALUE_ERROR_TOOLS:
            return "needs_user_input"
    if any(isinstance(item, (TimeoutError, ConnectionError)) for item in chain):
        return "retryable_error"
    if any(
        isinstance(item, OSError) and getattr(item, "errno", None) in _TRANSIENT_ERRNOS
        for item in chain
    ):
        return "retryable_error"
    return "terminal_failure"


def build_qsaria_tool(
    spec: QsariaToolSpec,
    instance: Any,
    manager: "ExperimentManager",
) -> Callable[..., Any]:
    """Wrap a toolkit method with experiment restoration and persistence.

    The manager owns session-prefix isolation, the model-free
    ``MCPAgentContext``, durable state, manifests, and the cross-process catalog
    mutation lock.  Toolkit execution stays on a worker thread, matching the
    generic MCP adapter without using the ChEMBL job worker.
    """

    if not isinstance(spec, QsariaToolSpec):
        raise TypeError("build_qsaria_tool requires a QsariaToolSpec")

    bound_method = getattr(instance, spec.method)
    underlying, _resolved, public_signature = _public_signature(bound_method, spec)

    def _call_bound(call_kwargs: dict[str, Any]) -> Any:
        if inspect.iscoroutinefunction(bound_method):
            return asyncio.run(bound_method(**call_kwargs))
        return bound_method(**call_kwargs)

    def _invoke_sync(kwargs: dict[str, Any]) -> Any:
        call_kwargs: Dict[str, Any] = dict(kwargs)
        if not spec.requires_experiment:
            if (
                spec.output_paths
                or spec.nested_output_paths
                or spec.nested_blocked_inputs
                or spec.registered_artifact_inputs
                or spec.json_payload_inputs
                or spec.session_forces
                or spec.catalog_write
            ):
                raise RuntimeError(
                    "Experiment-free Qsaria tools must be read-only and cannot allocate outputs"
                )
            call_kwargs = _normalize_input_paths(call_kwargs, manager)
            ephemeral = MCPAgentContext(
                name=spec.agent_name,
                model=None,
                llm_policy="disabled",
            )
            if "agent" in underlying.parameters:
                call_kwargs["agent"] = ephemeral
            if "session_state" in underlying.parameters:
                call_kwargs["session_state"] = ephemeral.session_state
            call_kwargs.update(spec.forces)
            with catalog_scope(access=spec.catalog_access, write=False):
                return _coerce_return_value(_call_bound(call_kwargs))

        try:
            experiment_id = str(call_kwargs.pop("experiment_id"))
        except KeyError as exc:
            raise MCPToolError(f"{spec.mcp_name} requires experiment_id") from exc

        # Keep the synchronous manager/catalog context managers and toolkit
        # execution on one worker thread.  Besides keeping the MCP event loop
        # responsive, this is essential for RLock ownership: holding a
        # threading.RLock on the event-loop thread across ``await`` would let a
        # second task on that same thread re-enter the supposedly exclusive
        # experiment/catalog section.
        with manager.run(
            experiment_id,
            agent_name=spec.agent_name,
            tool_name=spec.mcp_name,
            catalog_write=spec.catalog_write,
        ) as run:
            try:
                call_kwargs = _without_nested_output_paths(call_kwargs, spec)
                call_kwargs = _without_session_forces(call_kwargs, spec)
                call_kwargs = _without_forced_parameters(call_kwargs, spec)
                call_kwargs = _normalize_input_paths(
                    call_kwargs,
                    manager,
                    experiment_id=run.experiment_id,
                    runtime_state=run.runtime_state,
                )
                _validate_registered_artifact_inputs(call_kwargs, spec, manager, run)
                _validate_json_payload_inputs(call_kwargs, spec, manager, run)
                if "agent" in underlying.parameters:
                    call_kwargs["agent"] = run.context
                if "session_state" in underlying.parameters:
                    call_kwargs["session_state"] = run.context.session_state
                call_kwargs.update(_session_forced_values(spec, run))
                call_kwargs.update(spec.forces)
                call_kwargs.update(_forced_output_paths(spec, run))
                _inject_nested_output_paths(call_kwargs, spec, run)
                run.capture_inputs(call_kwargs)
                with catalog_scope(access=spec.catalog_access, write=spec.catalog_write):
                    result = _call_bound(call_kwargs)
            except Exception as exc:
                run.capture_error(exc, status=_classify_error(spec, exc))
                raise
            coerced = _coerce_return_value(result)
            outcome = _classify_result(coerced)
            run.capture_result(coerced, status=outcome)
            if outcome != "success":
                run.capture_error(
                    _result_failure_message(coerced, outcome),
                    status=outcome,
                )
            return coerced

    async def _invoke(**kwargs: Any) -> Any:
        try:
            return await asyncio.to_thread(_invoke_sync, dict(kwargs))
        except MCPToolError:
            raise
        except Exception as exc:  # noqa: BLE001 - expose a stable protocol error
            logger.error(
                "Qsaria MCP tool %s failed: %s\n%s",
                spec.mcp_name,
                exc,
                traceback.format_exc(),
            )
            raise MCPToolError(f"{spec.mcp_name} failed: {exc}") from exc

    _invoke.__name__ = spec.mcp_name
    _invoke.__qualname__ = spec.mcp_name
    _invoke.__doc__ = spec.summary or (bound_method.__doc__ or "").strip()
    _invoke.__signature__ = public_signature  # type: ignore[attr-defined]
    _invoke.__annotations__ = _call_annotations(public_signature)
    _invoke.__wrapped__ = bound_method  # type: ignore[attr-defined]
    return _invoke


__all__ = ["QsariaToolSpec", "build_qsaria_tool", "is_qsaria_spec"]
