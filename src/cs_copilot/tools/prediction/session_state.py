#!/usr/bin/env python
# coding: utf-8
"""Shared QSAR prediction session-state and artifact helpers.

This module intentionally avoids backend-specific imports so every prediction
backend can reuse the same session and artifact conventions without depending
on Chemprop's toolkit implementation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List
from zipfile import ZIP_DEFLATED, ZipFile

if TYPE_CHECKING:
    from agno.agent import Agent

_BUNDLE_DIRECTORY_ROOTS_TO_SKIP = {"/", "/app", "/app/.files", "/app/.files/sessions"}
_BUNDLE_FILENAMES_TO_SKIP = {".training_in_progress"}


def get_prediction_state(agent: Agent) -> Dict[str, Any]:
    """Return the shared prediction state, initializing expected keys."""
    state = agent.session_state.setdefault("prediction_models", {})
    state.setdefault("registered", {})
    state.setdefault("last_prediction", {})
    state.setdefault("prediction_history", [])
    state.setdefault("catalog_recommendations", {})
    state.setdefault("training_runs", [])
    state.setdefault("active_training_run", None)
    return state


def write_active_training_marker(marker_path: Path, payload: Dict[str, Any]) -> None:
    """Persist the active training marker using the project's JSON convention."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps(payload, indent=2))


def latest_curation_artifacts(agent: Agent) -> Dict[str, Any]:
    """Return the latest QSAR curation artifacts recorded in the shared session."""
    curation_state = agent.session_state.get("qsar_curation") or {}
    latest = curation_state.get("last_result") or {}
    if not isinstance(latest, dict):
        return {}
    artifacts = dict(latest.get("curation_artifacts") or {})
    if latest.get("curated_dataset_path"):
        artifacts.setdefault("curated_dataset_csv", latest.get("curated_dataset_path"))
    if latest.get("report_path"):
        artifacts["curation_report_json"] = latest.get("report_path")
    if latest.get("bundle_file_ref"):
        artifacts["curation_bundle_zip"] = latest.get("bundle_file_ref")
    if not artifacts and not latest.get("curated_dataset_path"):
        return {}
    return {
        "curation_backend": latest.get("curation_backend_used") or latest.get("curation_backend"),
        "curated_dataset_path": latest.get("curated_dataset_path"),
        "rows_in": latest.get("rows_in"),
        "rows_out": latest.get("rows_out"),
        "artifacts": artifacts,
    }


def discover_curation_artifacts_near_dataset(dataset_path: str | None) -> Dict[str, Any]:
    """Best-effort discovery for curation files written next to session datasets."""
    if not dataset_path:
        return {}
    path = Path(str(dataset_path)).expanduser()
    parent = path.parent
    if not parent.exists():
        return {}
    artifacts: Dict[str, str] = {}
    curated_dataset_path: str | None = None
    if path.exists() and path.is_file() and path.name.endswith("_curated.csv"):
        curated_dataset_path = str(path)
    else:
        curated_candidates = sorted(
            parent.glob("*_curated.csv"),
            key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0,
            reverse=True,
        )
        if curated_candidates:
            curated_dataset_path = str(curated_candidates[0])
    if curated_dataset_path:
        artifacts["curated_dataset_csv"] = curated_dataset_path
    artifact_dirs = sorted(
        parent.glob("*_curation_artifacts"),
        key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0,
        reverse=True,
    )
    if artifact_dirs:
        artifact_dir = artifact_dirs[0]
        known_files = {
            "standardization_map_csv": "curation_standardization_map.csv",
            "duplicate_identity_groups_csv": "curation_duplicate_identity_groups.csv",
            "removed_rows_csv": "curation_removed_rows.csv",
            "identity_diagnostics_json": "curation_identity_diagnostics.json",
            "manifest_json": "curation_manifest.json",
        }
        for key, filename in known_files.items():
            candidate = artifact_dir / filename
            if candidate.exists():
                artifacts[key] = str(candidate)
    report_candidates = sorted(
        parent.glob("*curation_report*.json"),
        key=lambda candidate: candidate.stat().st_mtime if candidate.exists() else 0,
        reverse=True,
    )
    if report_candidates:
        artifacts["curation_report_json"] = str(report_candidates[0])
    if not artifacts and not curated_dataset_path:
        return {}
    return {
        "curated_dataset_path": curated_dataset_path,
        "artifacts": artifacts,
    }


def _iter_bundle_files(paths: Iterable[Path], bundle_path: Path) -> List[Path]:
    """Return existing files to include in a bundle, expanding directories."""
    resolved_bundle_path = bundle_path.resolve()
    discovered: List[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        if path.is_dir() and path.resolve().as_posix() in _BUNDLE_DIRECTORY_ROOTS_TO_SKIP:
            continue
        candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
        for candidate in candidates:
            if not candidate.is_file():
                continue
            if candidate.name in _BUNDLE_FILENAMES_TO_SKIP:
                continue
            resolved = candidate.resolve()
            if resolved == resolved_bundle_path or resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _bundle_common_root(files: List[Path]) -> Path | None:
    """Find a stable common parent for archive names."""
    if not files:
        return None
    try:
        return Path(os.path.commonpath([str(path.parent) for path in files]))
    except ValueError:
        return None


def _bundle_arcname(file_path: Path, common_root: Path | None) -> str:
    """Build a relative archive name without leaking absolute root entries."""
    if common_root is not None:
        try:
            return file_path.relative_to(common_root).as_posix()
        except ValueError:
            pass
    parts = [part for part in file_path.parts if part not in (file_path.anchor, os.sep)]
    return Path(*parts).as_posix() if parts else file_path.name


def bundle_artifacts(bundle_path: Path, files: List[Path]) -> Path:
    """Create a zip bundle from existing artifact files, de-duplicating names."""
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    existing_files = _iter_bundle_files(files, bundle_path)
    common_root = _bundle_common_root(existing_files)
    seen_names: Dict[str, int] = {}
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in existing_files:
            arcname = _bundle_arcname(file_path, common_root)
            duplicate_count = seen_names.get(arcname, 0)
            seen_names[arcname] = duplicate_count + 1
            if duplicate_count:
                path = Path(arcname)
                arcname = path.with_name(f"{path.stem}_{duplicate_count}{path.suffix}").as_posix()
            zf.write(file_path, arcname=arcname)
    return bundle_path
