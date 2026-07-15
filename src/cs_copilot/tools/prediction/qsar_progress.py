"""Small, UI-agnostic helpers for truthful QSAR workflow progress.

The helpers deliberately describe only observed work.  They are shared by the
Chainlit relay and QSAR backends so status copy stays consistent without
inventing percentages, trial counts, or epochs that a backend cannot expose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


_TOOL_PHASES = {
    "inspect_dataset_schema": "Inspecting dataset",
    "identify_qsar_columns": "Identifying QSAR columns",
    "curate_qsar_dataset": "Curating dataset",
    "summarize_curated_dataset": "Summarizing curated dataset",
    "write_curation_report": "Writing curation report",
    "prepare_training_dataset": "Preparing training data",
    "train_qsar_model": "Preparing model training",
    "train_chemprop_model": "Preparing Chemprop training",
    "train_lightgbm_model": "Preparing LightGBM training",
    "train_tabicl_model": "Preparing TabICL training",
    "train_model": "Preparing Chemprop training",
    "predict_from_csv": "Generating predictions",
    "predict_from_smiles": "Generating predictions",
    "predict_with_lightgbm_from_csv": "Generating predictions",
    "predict_with_tabicl_from_csv": "Generating predictions",
    "evaluate_model_on_dataset": "Evaluating external dataset",
    "evaluate_ensemble_on_dataset": "Evaluating external dataset",
    "export_prediction_summary": "Writing prediction summary",
    "register_model": "Registering model",
    "register_catalog_model": "Registering model",
    "persist_registered_model": "Persisting model artifacts",
    "benchmark_qsar_models": "Benchmarking QSAR models",
    "create_ensemble_from_catalog": "Creating model ensemble",
}

_TOOL_COMPLETED_PHASES = {
    "inspect_dataset_schema": "Dataset inspection completed",
    "identify_qsar_columns": "QSAR columns identified",
    "curate_qsar_dataset": "Dataset curation completed",
    "summarize_curated_dataset": "Dataset summary completed",
    "write_curation_report": "Curation report written",
    "prepare_training_dataset": "Training data prepared",
    "train_qsar_model": "Model training completed",
    "train_chemprop_model": "Model training completed",
    "train_lightgbm_model": "Model training completed",
    "train_tabicl_model": "Model training completed",
    "train_model": "Model training completed",
    "predict_from_csv": "Predictions generated",
    "predict_from_smiles": "Predictions generated",
    "predict_with_lightgbm_from_csv": "Predictions generated",
    "predict_with_tabicl_from_csv": "Predictions generated",
    "evaluate_model_on_dataset": "External evaluation completed",
    "evaluate_ensemble_on_dataset": "External evaluation completed",
    "export_prediction_summary": "Prediction summary written",
    "register_model": "Model registered",
    "register_catalog_model": "Model registered",
    "persist_registered_model": "Model artifacts persisted",
    "benchmark_qsar_models": "QSAR benchmark completed",
    "create_ensemble_from_catalog": "Model ensemble created",
}


def phase_for_tool(tool_name: str | None) -> str | None:
    """Return the user-facing phase for a known Qsaria tool.

    ``None`` intentionally means that the caller should keep the current
    status.  It prevents unrelated or diagnostic tool names from leaking into
    the QSAR card.
    """

    return _TOOL_PHASES.get(str(tool_name or "").strip())


def completed_phase_for_tool(tool_name: str | None) -> str | None:
    """Return the compact completion phase for a known Qsaria tool."""

    return _TOOL_COMPLETED_PHASES.get(str(tool_name or "").strip())


def backend_display_name(backend_name: Any) -> str | None:
    """Convert known backend identifiers into compact display labels."""

    normalized = str(backend_name or "").strip().lower()
    return {
        "lightgbm": "LightGBM",
        "chemprop": "Chemprop",
        "tabicl": "TabICL",
    }.get(normalized)


def format_elapsed_seconds(elapsed_seconds: float | int) -> str:
    """Format elapsed time for the compact status card."""

    total_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def load_active_run_snapshot(active_run: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge an in-memory active-run record with its latest marker snapshot.

    Marker reads are best effort: the interface must continue rendering when a
    worker has just removed its marker or when a partial JSON write is observed.
    """

    snapshot = dict(active_run or {})
    marker_raw = snapshot.get("active_marker_path")
    if not marker_raw:
        return snapshot
    try:
        marker_payload = json.loads(Path(str(marker_raw)).read_text())
    except (OSError, TypeError, ValueError):
        return snapshot
    if isinstance(marker_payload, dict):
        snapshot.update(marker_payload)
    return snapshot


def apply_progress_update(
    record: dict[str, Any],
    phase: str,
    payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply one normalized, UI-safe progress update to an active-run record."""

    fields = dict(payload or {})
    record["phase"] = phase
    record["progress_message"] = fields.pop("detail", None)
    record.update({key: value for key, value in fields.items() if value is not None})
    return record


def status_label(status: str | None) -> str:
    """Return the fixed English label for a canonical status value."""

    normalized = str(status or STATUS_RUNNING).strip().lower()
    return {
        STATUS_COMPLETED: "Completed",
        STATUS_FAILED: "Failed",
    }.get(normalized, "Running")


def render_status_card(
    *,
    status: str,
    elapsed_seconds: float | int,
    phase: str | None = None,
    backend_name: Any = None,
    detail: str | None = None,
) -> str:
    """Render the compact, English-only Chainlit card content."""

    heading = status_label(status)
    backend = backend_display_name(backend_name)
    lines = ["QSAR workflow", f"{heading}{f' · {backend}' if backend else ''}"]
    if phase:
        lines.append(phase if not detail else f"{phase} — {detail}")
    elif detail:
        lines.append(detail)
    lines.append(f"Elapsed: {format_elapsed_seconds(elapsed_seconds)}")
    return "\n".join(lines)
