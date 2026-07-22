#!/usr/bin/env python
# coding: utf-8
"""Deterministic Agno handoffs for the isolated QSAR agent team.

The scientific toolkits already return compact ``report_facts`` and
``report_tables`` payloads.  This module prevents the Training LLM from
replacing those facts with a lossy narrative before Agno delegates to Report.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

_TRAINING_TOOL_NAMES = {
    "benchmark_qsar_models",
    "train_chemprop_model",
    "train_lightgbm_model",
    "train_qsar_model",
    "train_standard_qsar_model",
    "train_tabicl_model",
}
_PERSISTENCE_TOOL_NAMES = {
    "persist_registered_model",
    "register_and_persist_candidates",
}
_ARTIFACT_FIELDS = (
    "bundle_file_ref",
    "training_bundle",
    "summary_path",
    "canonical_summary_path",
    "config_path",
    "splits_path",
    "validation_predictions_path",
    "test_predictions_path",
    "candidate_manifest_path",
    "outlier_model_variants_manifest_path",
    "leaderboard_path",
    "report_path",
    "output_dir",
)


class TrainingArtifactEvidence(BaseModel):
    """One exact artifact reference returned by a successful tool call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    path: str
    kind: Literal["file", "directory", "returned_uri"]


class PersistedModelEvidence(BaseModel):
    """Compact durable-model evidence returned by Registry."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str
    persisted: bool = True
    status: Optional[str] = None
    model_root: Optional[str] = None
    model_path: Optional[str] = None
    metadata_path: Optional[str] = None


class TrainingPersistenceEvidence(BaseModel):
    """State of the persistence stage owned by the historical Agno Training role."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["not_required", "pending", "completed", "failed"]
    expected_model_count: Optional[int] = Field(default=None, ge=0)
    models: tuple[PersistedModelEvidence, ...] = ()


class QsarTrainingAgentHandoff(BaseModel):
    """Compact, validated output passed from Agno Training to Report."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    agent: Literal["qsar_training"] = "qsar_training"
    status: Literal["completed", "partial", "terminal_failure"]
    workflow_kind: Literal["training", "benchmark"] = "training"
    summary: str = Field(min_length=1)
    report_facts: Dict[str, Any] = Field(default_factory=dict)
    report_tables: Dict[str, Any] = Field(default_factory=dict)
    persistence: TrainingPersistenceEvidence
    artifacts: tuple[TrainingArtifactEvidence, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    recommended_next_action: str = ""


def _normalized_tool_name(value: Any) -> str:
    name = str(value or "").strip()
    for separator in (".", "__"):
        if separator in name:
            name = name.rsplit(separator, 1)[-1]
    return name


def _parse_tool_result(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
        if isinstance(parsed, Mapping):
            return dict(parsed)
    return None


def _short_error(value: Any, *, limit: int = 4000) -> str:
    text = str(value or "Unknown tool failure").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _persisted_model(payload: Mapping[str, Any]) -> Optional[PersistedModelEvidence]:
    model_id = payload.get("model_id")
    if not model_id:
        return None
    return PersistedModelEvidence(
        model_id=str(model_id),
        persisted=bool(payload.get("persisted", True)),
        status=str(payload["status"]) if payload.get("status") is not None else None,
        model_root=str(payload["model_root"]) if payload.get("model_root") else None,
        model_path=str(payload["model_path"]) if payload.get("model_path") else None,
        metadata_path=(str(payload["metadata_path"]) if payload.get("metadata_path") else None),
    )


def _collect_persisted_models(
    training_payload: Mapping[str, Any],
    executions: Sequence[Any],
) -> tuple[PersistedModelEvidence, ...]:
    models: list[PersistedModelEvidence] = []

    for item in training_payload.get("persisted_model_mapping") or []:
        if isinstance(item, Mapping):
            model = _persisted_model(item)
            if model is not None:
                models.append(model)

    for execution in executions:
        if _normalized_tool_name(getattr(execution, "tool_name", None)) not in (
            _PERSISTENCE_TOOL_NAMES
        ):
            continue
        if bool(getattr(execution, "tool_call_error", False)):
            continue
        payload = _parse_tool_result(getattr(execution, "result", None))
        if not payload:
            continue
        if isinstance(payload.get("candidates"), list):
            for item in payload["candidates"]:
                if isinstance(item, Mapping):
                    model = _persisted_model(item)
                    if model is not None:
                        models.append(model)
        else:
            model = _persisted_model(payload)
            if model is not None and model.persisted:
                models.append(model)

    by_id: dict[str, PersistedModelEvidence] = {}
    for model in models:
        by_id[model.model_id] = model
    return tuple(by_id.values())


def _artifact_kind(path: str) -> Optional[Literal["file", "directory", "returned_uri"]]:
    parsed = urlparse(path)
    if parsed.scheme and parsed.scheme not in {"file"}:
        return "returned_uri"
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return "file"
    if candidate.is_dir():
        return "directory"
    return None


def _collect_artifacts(
    training_payload: Mapping[str, Any],
    models: Sequence[PersistedModelEvidence],
) -> tuple[tuple[TrainingArtifactEvidence, ...], list[str]]:
    artifacts: list[TrainingArtifactEvidence] = []
    warnings: list[str] = []
    seen: set[str] = set()

    def add(name: str, raw_path: Any) -> None:
        if not isinstance(raw_path, str) or not raw_path.strip():
            return
        path = raw_path.strip()
        if path.startswith("<file>") and path.endswith("</file>"):
            path = path.removeprefix("<file>").removesuffix("</file>").strip()
        if path in seen:
            return
        kind = _artifact_kind(path)
        if kind is None:
            warnings.append(f"Omitted artifact reference because it does not exist: {name}={path}")
            return
        seen.add(path)
        artifacts.append(TrainingArtifactEvidence(name=name, path=path, kind=kind))

    for field in _ARTIFACT_FIELDS:
        add(field, training_payload.get(field))
    for model in models:
        add(f"model_root:{model.model_id}", model.model_root)
        add(f"model_path:{model.model_id}", model.model_path)
        add(f"metadata_path:{model.model_id}", model.metadata_path)
    return tuple(artifacts), warnings


def _expected_persistence_count(training_payload: Mapping[str, Any]) -> Optional[int]:
    plan = training_payload.get("persistence_plan")
    if isinstance(plan, Mapping):
        raw_count = plan.get("candidate_count")
        if raw_count is not None:
            try:
                return max(0, int(raw_count))
            except (TypeError, ValueError):
                pass
        if plan.get("persist_all_candidates"):
            candidates = training_payload.get("candidate_results") or []
            if isinstance(candidates, list) and candidates:
                return len(candidates)
            return 1
    mapping = training_payload.get("persisted_model_mapping")
    if isinstance(mapping, list) and mapping:
        return len(mapping)
    if training_payload.get("recommended_registry_payload"):
        return 1
    return None


def _fallback_report_facts(training_payload: Mapping[str, Any]) -> Dict[str, Any]:
    workflow_kind = str(training_payload.get("workflow_kind") or "training")
    if workflow_kind == "benchmark":
        return {
            "schema_version": "2.0",
            "report_kind": "benchmark",
            "workflow_kind": "benchmark",
            "validation_protocol": training_payload.get("validation_protocol"),
            "validation_strategy": training_payload.get("validation_strategy"),
            "candidate_results": training_payload.get("candidate_results") or [],
            "leaderboard": training_payload.get("leaderboard") or [],
            "recommendations": {
                key: training_payload.get(key)
                for key in (
                    "best_overall_candidate",
                    "best_hardest_split_candidate",
                    "best_stability_candidate",
                    "best_fast_candidate",
                    "recommended_candidate_for_followup",
                )
                if training_payload.get(key) is not None
            },
        }
    return {}


def build_qsar_training_agent_handoff(
    executions: Sequence[Any],
) -> Optional[QsarTrainingAgentHandoff]:
    """Build one deterministic handoff from the actual Agno tool executions."""

    relevant = [
        execution
        for execution in executions
        if _normalized_tool_name(getattr(execution, "tool_name", None)) in _TRAINING_TOOL_NAMES
    ]
    if not relevant:
        return None

    successful: list[tuple[Any, Dict[str, Any]]] = []
    failures: list[str] = []
    for execution in relevant:
        payload = _parse_tool_result(getattr(execution, "result", None))
        if bool(getattr(execution, "tool_call_error", False)) or payload is None:
            failures.append(_short_error(getattr(execution, "result", None)))
            continue
        successful.append((execution, payload))

    if not successful:
        blocker = failures[-1] if failures else "Training tool returned no structured result."
        return QsarTrainingAgentHandoff(
            status="terminal_failure",
            summary="QSAR training failed before structured training evidence was produced.",
            persistence=TrainingPersistenceEvidence(status="not_required"),
            blockers=(blocker,),
            recommended_next_action="Route the preserved training failure to QSAR Report.",
        )

    _, training_payload = successful[-1]
    workflow_kind: Literal["training", "benchmark"] = (
        "benchmark" if training_payload.get("workflow_kind") == "benchmark" else "training"
    )
    reporting = training_payload.get("reporting_handoff")
    reporting = dict(reporting) if isinstance(reporting, Mapping) else {}
    report_facts = reporting.get("report_facts")
    report_tables = reporting.get("report_tables")
    report_facts = dict(report_facts) if isinstance(report_facts, Mapping) else {}
    report_tables = dict(report_tables) if isinstance(report_tables, Mapping) else {}
    if not report_facts:
        report_facts = _fallback_report_facts(training_payload)

    models = _collect_persisted_models(training_payload, executions)
    expected_count = _expected_persistence_count(training_payload)
    persistence_failures = [
        _short_error(getattr(execution, "result", None))
        for execution in executions
        if _normalized_tool_name(getattr(execution, "tool_name", None)) in _PERSISTENCE_TOOL_NAMES
        and bool(getattr(execution, "tool_call_error", False))
    ]
    if expected_count is None and not models:
        persistence_status: Literal["not_required", "pending", "completed", "failed"] = (
            "not_required"
        )
    elif expected_count is not None and len(models) >= expected_count:
        persistence_status = "completed"
    elif persistence_failures:
        persistence_status = "failed"
    else:
        persistence_status = "pending"

    artifacts, artifact_warnings = _collect_artifacts(training_payload, models)
    warnings = _string_list(training_payload.get("warnings")) + artifact_warnings
    blockers = _string_list(training_payload.get("blockers")) + persistence_failures

    if not report_facts:
        blockers.append("Training result did not contain canonical report_facts.")
    if persistence_status == "pending":
        blockers.append("Training completed but durable model persistence is still pending.")

    status: Literal["completed", "partial", "terminal_failure"] = "completed"
    if blockers or persistence_status in {"pending", "failed"}:
        status = "partial"

    if status == "completed":
        next_action = "Route this structured handoff to QSAR Report exactly once."
    elif persistence_status in {"pending", "failed"}:
        next_action = "Route persistence to Model Registry before QSAR Report."
    else:
        next_action = "Route only the verified partial evidence to QSAR Report."

    return QsarTrainingAgentHandoff(
        status=status,
        workflow_kind=workflow_kind,
        summary=(
            "QSAR benchmark completed with structured evidence."
            if workflow_kind == "benchmark"
            else "QSAR training completed with canonical structured evidence."
        ),
        report_facts=report_facts,
        report_tables=report_tables,
        persistence=TrainingPersistenceEvidence(
            status=persistence_status,
            expected_model_count=expected_count,
            models=models,
        ),
        artifacts=artifacts,
        warnings=tuple(dict.fromkeys(warnings)),
        blockers=tuple(dict.fromkeys(blockers)),
        recommended_next_action=next_action,
    )


def finalize_qsar_training_agent_output(
    run_output: Any,
    session_state: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> None:
    """Agno post-hook replacing Training prose with the validated handoff JSON."""

    handoff = build_qsar_training_agent_handoff(getattr(run_output, "tools", None) or [])
    if handoff is None:
        return
    payload = handoff.model_dump(mode="json")
    if session_state is not None:
        prediction_state = session_state.setdefault("prediction_models", {})
        prediction_state["latest_training_handoff"] = payload
    run_output.content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    run_output.content_type = QsarTrainingAgentHandoff.__name__


def compact_qsar_report_session_context(
    session_state: Optional[Dict[str, Any]] = None,
    **_: Any,
) -> None:
    """Agno pre-hook exposing only report-relevant shared state to the Report LLM."""

    if not isinstance(session_state, dict):
        return
    compact: Dict[str, Any] = {}
    for key in (
        "current_run_id",
        "current_session_id",
        "current_user_id",
        "report_language",
        "REPORT_LANGUAGE",
    ):
        if key in session_state:
            compact[key] = session_state[key]

    curation = session_state.get("qsar_curation")
    if isinstance(curation, Mapping) and isinstance(curation.get("last_result"), Mapping):
        compact["qsar_curation"] = {"last_result": dict(curation["last_result"])}

    prediction_models = session_state.get("prediction_models")
    if isinstance(prediction_models, Mapping):
        compact_models: Dict[str, Any] = {}
        if isinstance(prediction_models.get("latest_training_handoff"), Mapping):
            compact_models["latest_training_handoff"] = dict(
                prediction_models["latest_training_handoff"]
            )
        if isinstance(prediction_models.get("last_prediction"), Mapping):
            compact_models["last_prediction"] = dict(prediction_models["last_prediction"])
        history = prediction_models.get("prediction_history")
        if isinstance(history, list) and history:
            compact_models["latest_prediction_history"] = history[-1]
        if compact_models:
            compact["prediction_models"] = compact_models

    prediction_outputs = session_state.get("prediction_outputs")
    if isinstance(prediction_outputs, Mapping) and prediction_outputs.get("latest_summary"):
        compact["prediction_outputs"] = {"latest_summary": prediction_outputs.get("latest_summary")}

    session_state.clear()
    session_state.update(compact)
