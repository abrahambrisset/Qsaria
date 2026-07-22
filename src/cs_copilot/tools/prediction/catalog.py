#!/usr/bin/env python
# coding: utf-8
"""
Persistent catalog for predictive models.

The catalog is intentionally richer than session registration metadata so the
agent can select models using structured criteria instead of prompt-only
guesswork.  It is the first step toward a governed registry of validated
predictive assets.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .backend import PredictionModelRecord

DEFAULT_MODEL_CATALOG_PATH = Path(__file__).with_name("model_catalog.json")
DEFAULT_INTERNAL_MODEL_ROOT = Path("data/model_assets/internal").resolve()
PORTABLE_INTERNAL_MODEL_ROOT = Path("data/model_assets/internal")
DEFAULT_ALLOWED_STATUSES = ("production", "robust_validated", "validated")
_CATALOG_LOCKS_GUARD = threading.Lock()
_CATALOG_PROCESS_LOCKS: dict[str, threading.RLock] = {}
_CATALOG_LOCK_STATE = threading.local()
STATUS_WEIGHTS = {
    "production": 12,
    "robust_validated": 10,
    "validated": 8,
    "experimental": 4,
    "workflow_demo": -12,
    "deprecated": -20,
}


def _resolved_catalog_path(path: str | os.PathLike[str] | None = None) -> Path:
    source_path = Path(path).expanduser() if path is not None else DEFAULT_MODEL_CATALOG_PATH
    return source_path.resolve(strict=False)


def model_catalog_lock_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a writable runtime lock shared by writers of one catalog file."""

    source_path = _resolved_catalog_path(path)
    digest = hashlib.sha256(str(source_path).encode("utf-8")).hexdigest()
    try:
        user_scope = str(os.getuid())
    except AttributeError:  # pragma: no cover - Windows fallback
        user_scope = os.getenv("USERNAME", "default")
    return (
        Path(tempfile.gettempdir())
        / f"cs_copilot-{user_scope}"
        / "catalog_locks"
        / f"{digest}.lock"
    )


def _catalog_process_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _CATALOG_LOCKS_GUARD:
        return _CATALOG_PROCESS_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def model_catalog_lock(
    path: str | os.PathLike[str] | None = None,
) -> Iterator[None]:
    """Serialize catalog access across threads and local processes.

    The lock is reentrant on the owning thread.  This matters for Qsaria MCP:
    its experiment manager holds the catalog lock around the complete toolkit
    call, while the reused Agno/Chainlit toolkit acquires it again when it
    persists the resulting model.
    """

    source_path = _resolved_catalog_path(path)
    key = str(source_path)
    process_lock = _catalog_process_lock(source_path)
    with process_lock:
        depths = getattr(_CATALOG_LOCK_STATE, "depths", None)
        if depths is None:
            depths = {}
            _CATALOG_LOCK_STATE.depths = depths

        depth = depths.get(key, 0)
        if depth:
            depths[key] = depth + 1
            try:
                yield
            finally:
                depths[key] -= 1
            return

        lock_path = model_catalog_lock_path(source_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows fallback is process-only
                fcntl = None

            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            depths[key] = 1
            try:
                yield
            finally:
                depths.pop(key, None)
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_catalog_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 2, "models": []}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        raise ValueError(f"Model catalog is empty: {path}")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model catalog is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Model catalog root must be a JSON object.")
    if payload.get("schema_version") != 2:
        raise ValueError(
            "Unsupported model catalog schema_version. Qsaria 0.3.1 requires schema_version=2."
        )
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Model catalog `models` must be a list.")
    if not all(isinstance(item, dict) for item in models):
        raise ValueError("Every model catalog entry must be a JSON object.")
    return payload


def _atomic_write_catalog(path: Path, payload: dict[str, Any]) -> None:
    """Replace a catalog atomically with a temporary file in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _records_from_payload(payload: dict[str, Any]) -> list[PredictionModelRecord]:
    return [
        _record_with_runtime_paths(PredictionModelRecord.from_dict(record_payload))
        for record_payload in payload.get("models", [])
    ]


def _internal_path_tail(path: Path) -> Optional[Path]:
    """Return the portion below data/model_assets/internal for any runtime root."""

    marker = PORTABLE_INTERNAL_MODEL_ROOT.parts
    parts = path.parts
    for index in range(len(parts) - len(marker) + 1):
        if parts[index : index + len(marker)] == marker:
            return Path(*parts[index + len(marker) :])
    return None


def resolve_catalog_artifact_path(path: str | os.PathLike[str]) -> Path:
    """Resolve a catalog artifact against the active host/container model root.

    Catalogs created on macOS and consumed in the application container (or the
    reverse) may contain an absolute path from the other runtime. The stable
    portion is the path below ``data/model_assets/internal``.
    """

    expanded = Path(path).expanduser()
    if expanded.exists():
        return expanded.resolve()
    tail = _internal_path_tail(expanded)
    if tail is not None:
        candidate = DEFAULT_INTERNAL_MODEL_ROOT / tail
        if candidate.exists():
            return candidate.resolve()
    return expanded


def _portable_catalog_path(path: Optional[str]) -> Optional[str]:
    """Serialize internal artifacts independently of a host mount point."""

    if path is None:
        return None
    expanded = Path(path).expanduser()
    resolved = resolve_catalog_artifact_path(expanded)
    try:
        tail = resolved.resolve(strict=False).relative_to(
            DEFAULT_INTERNAL_MODEL_ROOT.resolve(strict=False)
        )
    except ValueError:
        tail = _internal_path_tail(expanded)
    if tail is None:
        return path
    return (PORTABLE_INTERNAL_MODEL_ROOT / tail).as_posix()


def _record_with_runtime_paths(record: PredictionModelRecord) -> PredictionModelRecord:
    model_path = str(resolve_catalog_artifact_path(record.model_path))
    metadata_path = (
        str(resolve_catalog_artifact_path(record.metadata_path))
        if record.metadata_path
        else None
    )
    if model_path == record.model_path and metadata_path == record.metadata_path:
        return record
    return replace(record, model_path=model_path, metadata_path=metadata_path)


def _record_payload_for_catalog(record: PredictionModelRecord) -> dict[str, Any]:
    payload = record.as_dict()
    payload["model_path"] = _portable_catalog_path(record.model_path)
    payload["metadata_path"] = _portable_catalog_path(record.metadata_path)
    return payload


def _normalize_text(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _flatten_text(values: Iterable[Any]) -> str:
    parts: List[str] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            parts.append(_flatten_text(value.values()))
        elif isinstance(value, (list, tuple, set)):
            parts.append(_flatten_text(value))
        else:
            parts.append(str(value))
    return " ".join(parts).lower()


def _record_from_internal_metadata(metadata_path: Path) -> Optional[PredictionModelRecord]:
    try:
        payload = json.loads(metadata_path.read_text())
    except Exception:
        return None

    model_id = payload.get("model_id")
    backend_name = payload.get("backend_name")
    task_payload = payload.get("task", {}) or {}
    model_path = (payload.get("artifacts", {}) or {}).get("model_path") or payload.get("model_path")
    if not (model_id and backend_name and model_path):
        return None

    model_root = metadata_path.parent
    resolved_model_path = str((model_root / model_path).resolve())
    training_data_summary = dict(payload.get("training_data_summary", {}) or {})
    for key in ("trained_at", "trained_date", "trained_time", "validation_protocol"):
        if payload.get(key) is not None:
            training_data_summary[key] = payload.get(key)

    applicability_domain = dict(payload.get("applicability_domain", {}) or {})

    return PredictionModelRecord(
        model_id=model_id,
        backend_name=backend_name,
        model_path=resolved_model_path,
        metadata_path=str(metadata_path.resolve()),
        display_name=payload.get("display_name"),
        description=payload.get("description"),
        tags=dict(payload.get("tags", {}) or {}),
        version=payload.get("version"),
        status=payload.get("status", "experimental"),
        owner=payload.get("owner"),
        source=payload.get("source"),
        domain_summary=payload.get("domain_summary"),
        strengths=list(payload.get("strengths", []) or []),
        limitations=list(payload.get("limitations", []) or []),
        recommended_for=list(payload.get("recommended_for", []) or []),
        not_recommended_for=list(payload.get("not_recommended_for", []) or []),
        known_metrics=dict(payload.get("known_metrics", {}) or {}),
        training_data_summary=training_data_summary,
        inference_profile=dict(payload.get("inference_profile", {}) or {}),
        selection_hints=dict(payload.get("selection_hints", {}) or {}),
        applicability_domain=applicability_domain,
        task=PredictionModelRecord.from_dict(
            {
                "model_id": model_id,
                "backend_name": backend_name,
                "model_path": resolved_model_path,
                "task": {
                    "task_type": task_payload.get("task_type", "regression"),
                    "smiles_columns": task_payload.get("smiles_columns", ["smiles"]),
                    "target_columns": task_payload.get("target_columns", []),
                    "reaction_columns": task_payload.get("reaction_columns", []),
                    "uncertainty_method": task_payload.get("uncertainty_method"),
                    "calibration_method": task_payload.get("calibration_method"),
                },
            }
        ).task,
    )


def _discover_internal_records(root: Path) -> List[PredictionModelRecord]:
    if not root.exists():
        return []
    records: List[PredictionModelRecord] = []
    for metadata_path in sorted(root.glob("*/metadata.json")):
        record = _record_from_internal_metadata(metadata_path)
        if record is not None:
            records.append(record)
    return records


@dataclass
class CatalogRecommendation:
    """Structured recommendation result for a model search."""

    record: PredictionModelRecord
    score: int
    reasons: List[str]
    warnings: List[str]
    backend_available: bool
    model_path_exists: bool
    runtime_compatible: bool

    def as_dict(self) -> Dict[str, Any]:
        payload = self.record.as_dict()
        payload.update(
            {
                "score": self.score,
                "reasons": list(self.reasons),
                "warnings": list(self.warnings),
                "backend_available": self.backend_available,
                "model_path_exists": self.model_path_exists,
                "runtime_compatible": self.runtime_compatible,
            }
        )
        return payload


class PredictionModelCatalog:
    """Access layer for the prediction model catalog."""

    def __init__(
        self,
        records: List[PredictionModelRecord],
        source_path: Path,
        schema_version: int = 2,
    ):
        self.records = [_record_with_runtime_paths(record) for record in records]
        self.source_path = source_path
        self.schema_version = schema_version

    @classmethod
    def load(cls, path: Optional[str] = None) -> "PredictionModelCatalog":
        source_path = Path(path).expanduser() if path else DEFAULT_MODEL_CATALOG_PATH
        # Loading is deliberately non-mutating. Explicit persistence creates
        # a missing catalog when needed.
        payload = _read_catalog_payload(source_path)
        records = _records_from_payload(payload)
        discovered_records = _discover_internal_records(DEFAULT_INTERNAL_MODEL_ROOT)
        if discovered_records:
            indexed = {record.model_id: record for record in records}
            for record in discovered_records:
                indexed[record.model_id] = record
            records = sorted(indexed.values(), key=lambda item: item.model_id)
        return cls(
            records=records,
            source_path=source_path,
            schema_version=2,
        )

    def refresh_from_internal_store(self, persist: bool = False) -> int:
        discovered_records = _discover_internal_records(DEFAULT_INTERNAL_MODEL_ROOT)
        indexed = {record.model_id: record for record in self.records}
        updated = 0
        for record in discovered_records:
            existing = indexed.get(record.model_id)
            if existing is None or existing.as_dict() != record.as_dict():
                indexed[record.model_id] = record
                updated += 1
        self.records = sorted(indexed.values(), key=lambda item: item.model_id)
        if persist:
            with model_catalog_lock(self.source_path):
                disk_payload = _read_catalog_payload(self.source_path)
                disk_records = _records_from_payload(disk_payload)

                # Preserve unsaved in-memory additions, prefer the latest
                # committed value for stale records, then let newly discovered
                # internal metadata supersede both.
                merged = {record.model_id: record for record in self.records}
                merged.update({record.model_id: record for record in disk_records})
                merged.update({record.model_id: record for record in discovered_records})
                self.records = sorted(merged.values(), key=lambda item: item.model_id)
                self.schema_version = 2
                payload = {
                    "schema_version": self.schema_version,
                    "models": [_record_payload_for_catalog(record) for record in self.records],
                }
                if payload != disk_payload:
                    _atomic_write_catalog(self.source_path, payload)
        return updated

    def save(self) -> None:
        payload = {
            "schema_version": 2,
            "models": [_record_payload_for_catalog(record) for record in self.records],
        }
        with model_catalog_lock(self.source_path):
            _atomic_write_catalog(self.source_path, payload)

    def list_models(self) -> List[PredictionModelRecord]:
        return list(self.records)

    def get_model(self, model_id: str) -> PredictionModelRecord:
        for record in self.records:
            if record.model_id == model_id:
                return record
        raise ValueError(f"Unknown catalog model_id: {model_id}")

    def upsert_model(self, record: PredictionModelRecord) -> PredictionModelRecord:
        """Insert or replace a model without losing concurrent catalog entries."""

        record = _record_with_runtime_paths(record)
        with model_catalog_lock(self.source_path):
            disk_payload = _read_catalog_payload(self.source_path)
            disk_records = _records_from_payload(disk_payload)

            # This instance may have been loaded before another writer
            # committed.  Merge its additions first, then prefer the current
            # disk values for conflicts, and finally apply this explicit
            # upsert as the authoritative value for ``record.model_id``.
            merged = {item.model_id: item for item in self.records}
            merged.update({item.model_id: item for item in disk_records})
            merged[record.model_id] = record
            self.records = sorted(merged.values(), key=lambda item: item.model_id)
            self.schema_version = 2
            _atomic_write_catalog(
                self.source_path,
                {
                    "schema_version": 2,
                    "models": [_record_payload_for_catalog(item) for item in self.records],
                },
            )
        return record

    def search(
        self,
        *,
        task_type: Optional[str] = None,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        backend_available: Optional[bool] = None,
        available_backend_names: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> List[CatalogRecommendation]:
        allowed = {status.lower() for status in (allowed_statuses or DEFAULT_ALLOWED_STATUSES)}
        normalized_task = _normalize_text(task_type)
        normalized_target = _normalize_text(target_hint)
        normalized_domain = _normalize_text(domain_hint)
        normalized_backend = _normalize_text(preferred_backend)
        normalized_available_backends = {
            _normalize_text(name) for name in (available_backend_names or []) if name
        }

        recommendations: List[CatalogRecommendation] = []
        for record in self.records:
            status = _normalize_text(record.status)
            if allowed and status not in allowed:
                continue

            path_exists = Path(record.model_path).expanduser().exists()
            if not include_unavailable_paths and not path_exists:
                continue

            score = 0
            reasons: List[str] = []
            warnings: List[str] = []

            if normalized_task:
                if _normalize_text(record.task.task_type) != normalized_task:
                    continue
                score += 50
                reasons.append(f"Task type matches `{record.task.task_type}`.")

            if normalized_backend:
                if _normalize_text(record.backend_name) != normalized_backend:
                    continue
                score += 8
                reasons.append(f"Backend preference matches `{record.backend_name}`.")

            if record.task.uncertainty_method:
                score += 4
            elif require_uncertainty:
                continue

            if require_uncertainty:
                reasons.append(f"Supports uncertainty via `{record.task.uncertainty_method}`.")

            status_weight = STATUS_WEIGHTS.get(status, 0)
            if status_weight:
                score += status_weight
            reasons.append(f"Catalog status is `{record.status}`.")

            target_text = _flatten_text(
                [
                    record.task.target_columns,
                    record.model_id,
                    record.display_name,
                    record.description,
                    record.recommended_for,
                    record.tags,
                ]
            )
            if normalized_target:
                if normalized_target in target_text:
                    score += 25
                    reasons.append(f"Target hint `{target_hint}` matches catalog metadata.")
                elif normalized_target in _flatten_text(record.not_recommended_for):
                    score -= 10
                    warnings.append(f"Catalog flags this model as a weak fit for `{target_hint}`.")

            domain_text = _flatten_text(
                [
                    record.domain_summary,
                    record.recommended_for,
                    record.training_data_summary,
                    record.selection_hints,
                    record.applicability_domain,
                ]
            )
            if normalized_domain:
                if normalized_domain in domain_text:
                    score += 18
                    reasons.append(
                        f"Domain hint `{domain_hint}` matches the model domain metadata."
                    )
                elif normalized_domain in _flatten_text(record.not_recommended_for):
                    score -= 8
                    warnings.append(f"Catalog indicates limited suitability for `{domain_hint}`.")

            if path_exists:
                score += 6
                reasons.append("Model artifact path is currently available.")
            else:
                score -= 18
                warnings.append("Model artifact path is not currently available.")

            if backend_available is True:
                score += 6
                reasons.append("Backend is available in the current environment.")
            elif backend_available is False:
                warnings.append("Backend is not available in the current environment.")

            runtime_validated = bool(
                (record.inference_profile or {}).get("runtime_validated", False)
            )
            runtime_compatible = True

            if normalized_available_backends:
                if _normalize_text(record.backend_name) in normalized_available_backends:
                    score += 10
                    reasons.append(
                        f"Backend `{record.backend_name}` is available in the active runtime."
                    )
                else:
                    runtime_compatible = False
                    score -= 24
                    warnings.append(
                        "Catalog metadata indicates this model uses a backend "
                        f"(`{record.backend_name}`) that is not currently available in the runtime."
                    )

            if runtime_validated:
                score += 8
                reasons.append("Runtime execution has been validated in this project.")
            else:
                runtime_compatible = False
                score -= 12
                warnings.append("Runtime execution has not yet been validated in this project.")

            recommendations.append(
                CatalogRecommendation(
                    record=record,
                    score=score,
                    reasons=reasons,
                    warnings=warnings,
                    backend_available=backend_available is not False,
                    model_path_exists=path_exists,
                    runtime_compatible=runtime_compatible and path_exists,
                )
            )

        recommendations.sort(key=lambda item: item.score, reverse=True)
        return recommendations

    def recommend(
        self,
        *,
        task_type: str,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        backend_available: Optional[bool] = None,
        available_backend_names: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> Dict[str, Any]:
        candidates = self.search(
            task_type=task_type,
            target_hint=target_hint,
            domain_hint=domain_hint,
            require_uncertainty=require_uncertainty,
            allowed_statuses=allowed_statuses,
            preferred_backend=preferred_backend,
            backend_available=backend_available,
            available_backend_names=available_backend_names,
            include_unavailable_paths=include_unavailable_paths,
        )
        if not candidates:
            return {
                "catalog_path": str(self.source_path),
                "selected_model": None,
                "runnable_candidate": None,
                "alternatives": [],
                "selection_summary": (
                    "No compatible model was found in the catalog for the requested task."
                ),
            }

        selected = candidates[0]
        runnable_candidate = next(
            (candidate for candidate in candidates if candidate.runtime_compatible),
            None,
        )
        summary = " | ".join(selected.reasons[:3])
        if runnable_candidate is None:
            summary += " | No runtime-compatible candidate is currently validated."
        return {
            "catalog_path": str(self.source_path),
            "selected_model": selected.as_dict(),
            "runnable_candidate": runnable_candidate.as_dict() if runnable_candidate else None,
            "alternatives": [candidate.as_dict() for candidate in candidates[1:4]],
            "selection_summary": summary,
        }
