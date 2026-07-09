#!/usr/bin/env python
# coding: utf-8
"""Backend-neutral model registry and catalog facade."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .backend import PredictionModelRecord, PredictionTaskSpec
from .backend_capabilities import get_backend_capabilities
from .catalog import DEFAULT_INTERNAL_MODEL_ROOT, PredictionModelCatalog
from .qsar_response_compaction import (
    compact_applicability_domain_for_response,
    compact_inference_profile_for_response as _shared_compact_inference_profile,
    compact_model_payload_for_response as _shared_compact_model_payload,
    compact_training_data_summary_for_response,
)
from .qsar_reporting import build_registry_reporting_handoff
from .qsar_training_policy import (
    coerce_project_timezone,
    project_now,
    safe_slug,
    seed_policy_reporting_text,
    seed_policy_reproducibility_metadata,
)
from .session_state import (
    discover_curation_artifacts_near_dataset,
    get_prediction_state,
    latest_curation_artifacts,
)

ARCHIVE_MODEL_PATH_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")
def _relative_posix(path: Path, start: Path) -> str:
    return path.relative_to(start).as_posix()


def _safe_display_token(value: str) -> str:
    token = value.replace("_", " ").strip()
    return " ".join(part.capitalize() for part in token.split())


def _sanitize_activity_cliff_description(
    description: Optional[str], activity_payload: Dict[str, Any]
) -> Optional[str]:
    if not description or not activity_payload.get("enabled"):
        return description
    loops_requested = activity_payload.get("feedback_loops_requested")
    if activity_payload.get("mode") != "with_feedback_loops" or not loops_requested:
        return description
    replacement = f"{loops_requested} feedback loops"
    sanitized = description.replace("1 feedback loop", replacement)
    sanitized = sanitized.replace("1 feedback loops", replacement)
    return sanitized


def _load_json_if_available(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _compact_inference_profile_for_response(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _shared_compact_inference_profile(profile)


def _compact_model_payload_for_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _shared_compact_model_payload(payload)


def _compact_recommendation_for_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(payload)
    for key in ("selected_model", "runnable_candidate"):
        if isinstance(compacted.get(key), dict):
            compacted[key] = _compact_model_payload_for_response(compacted[key])
    alternatives = compacted.get("alternatives")
    if isinstance(alternatives, list):
        compacted["alternatives"] = [
            _compact_model_payload_for_response(item) if isinstance(item, dict) else item
            for item in alternatives
        ]
    return compacted


def _compact_persisted_record_for_response(record: PredictionModelRecord) -> Dict[str, Any]:
    return {
        "model_id": record.model_id,
        "backend_name": record.backend_name,
        "status": record.status,
        "model_path": record.model_path,
        "metadata_path": record.metadata_path,
        "display_name": record.display_name,
        "version": record.version,
        "known_metrics": record.known_metrics,
        "training_data_summary": compact_training_data_summary_for_response(
            record.training_data_summary
        ),
        "inference_profile": _compact_inference_profile_for_response(record.inference_profile),
        "applicability_domain": compact_applicability_domain_for_response(
            record.applicability_domain
        ),
    }


def _hydrate_inference_profile_from_summary(
    *,
    current_profile: Dict[str, Any],
    override_profile: Optional[Dict[str, Any]],
    summary_payload: Dict[str, Any],
) -> Dict[str, Any]:
    profile = {
        **(current_profile or {}),
        **(override_profile or {}),
    }
    if not profile.get("feature_columns") and isinstance(
        summary_payload.get("feature_columns"), list
    ):
        profile["feature_columns"] = list(summary_payload["feature_columns"])
    if not profile.get("categorical_feature_columns") and isinstance(
        summary_payload.get("categorical_feature_columns"), list
    ):
        profile["categorical_feature_columns"] = list(
            summary_payload["categorical_feature_columns"]
        )
    if not profile.get("representation_name") and summary_payload.get("representation_name"):
        profile["representation_name"] = summary_payload.get("representation_name")
    if summary_payload.get("summary_path") and not profile.get("feature_columns_source"):
        profile["feature_columns_source"] = summary_payload.get("summary_path")
    for key in (
        "task_kind",
        "class_labels",
        "class_count",
        "label_mapping",
        "positive_class_label",
    ):
        if profile.get(key) is None and summary_payload.get(key) is not None:
            profile[key] = summary_payload.get(key)
    return profile


def _classification_metadata(record: PredictionModelRecord) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"task_type": record.task.task_type}
    sources = (record.inference_profile or {}, record.training_data_summary or {})
    for key in (
        "task_kind",
        "class_labels",
        "class_count",
        "label_mapping",
        "positive_class_label",
    ):
        for source in sources:
            if source.get(key) is not None:
                payload[key] = source.get(key)
                break
    return payload


def _resolved_path_key(path: str | Path) -> str:
    source_path = Path(str(path)).expanduser()
    try:
        return str(source_path.resolve())
    except Exception:
        return str(source_path)


def _hydrate_curation_metadata(
    *,
    curation_payload: Dict[str, Any],
    copied_curation_artifacts: Dict[str, str],
    model_root: Path,
) -> Dict[str, Any]:
    """Build catalog curation metadata from session payload plus persisted reports."""
    metadata: Dict[str, Any] = {
        "backend": curation_payload.get("curation_backend"),
        "curated_dataset_path": (
            copied_curation_artifacts.get("curated_dataset_csv")
            or curation_payload.get("curated_dataset_path")
        ),
        "rows_in": curation_payload.get("rows_in"),
        "rows_out": curation_payload.get("rows_out"),
        "artifacts": copied_curation_artifacts,
    }

    report_rel = copied_curation_artifacts.get("curation_report_json")
    report = _load_json_if_available(model_root / report_rel) if report_rel else {}
    manifest_rel = copied_curation_artifacts.get("manifest_json")
    manifest = _load_json_if_available(model_root / manifest_rel) if manifest_rel else {}

    if report:
        metadata["backend"] = (
            metadata.get("backend")
            or report.get("curation_backend_used")
            or report.get("curation_backend")
        )
        metadata["curated_dataset_path"] = metadata.get("curated_dataset_path") or report.get(
            "curated_dataset_path"
        )
        metadata["source_dataset_path"] = report.get("source_dataset_path")
        metadata["rows_in"] = (
            metadata.get("rows_in")
            if metadata.get("rows_in") is not None
            else report.get("rows_in")
        )
        metadata["rows_out"] = (
            metadata.get("rows_out")
            if metadata.get("rows_out") is not None
            else report.get("rows_out")
        )
        metadata["rows_removed"] = report.get("rows_removed")
        metadata["duplicate_groups_detected"] = report.get("duplicate_groups_detected")
        metadata["duplicate_groups_aggregated"] = report.get("duplicate_groups_aggregated")
        metadata["duplicate_conflicting_groups"] = report.get("duplicate_conflicting_groups")
        metadata["duplicate_conflicting_rows_removed"] = report.get(
            "duplicate_conflicting_rows_removed"
        )
        metadata["organometallic_rows_removed"] = report.get("organometallic_rows_removed")
        metadata["invalid_smiles_removed"] = report.get("invalid_smiles_removed")
    if manifest:
        metadata["backend"] = metadata.get("backend") or manifest.get("curation_backend")
        if manifest.get("curation_policy"):
            metadata["curation_policy"] = manifest.get("curation_policy")

    return metadata


def _extract_endpoint_and_dataset(
    train_csv: Optional[str], fallback_model_id: str
) -> tuple[str, str]:
    source = Path(train_csv or fallback_model_id).stem.lower()
    for suffix in ("_curated", "_cleaned", "_dataset", "_training"):
        if source.endswith(suffix):
            source = source[: -len(suffix)]
            break
    parts = [part for part in source.split("_") if part]
    if len(parts) >= 2:
        endpoint = parts[0]
        dataset = "_".join(parts[1:])
    elif len(parts) == 1:
        endpoint = parts[0]
        dataset = "dataset"
    else:
        endpoint = safe_slug(fallback_model_id) or "endpoint"
        dataset = "dataset"
    return safe_slug(endpoint) or "endpoint", safe_slug(dataset) or "dataset"


def _canonical_model_id(
    *,
    endpoint: str,
    dataset: str,
    protocol: str,
    backend: str,
    representation: Optional[str],
    version: str,
    trained_at: datetime,
) -> str:
    version_token = version if str(version).startswith("v") else f"v{version}"
    date_token = trained_at.strftime("%d%m%Y")
    time_token = trained_at.strftime("%H%M%S")
    parts = [
        safe_slug(endpoint),
        safe_slug(dataset),
        safe_slug(protocol),
        safe_slug(backend),
    ]
    if representation:
        parts.append(safe_slug(representation))
    parts.extend([safe_slug(version_token), date_token, time_token])
    return "_".join(parts)


def _canonical_display_name(
    *,
    endpoint: str,
    dataset: str,
    protocol: str,
    backend: str,
    representation: Optional[str],
    version: str,
) -> str:
    version_token = version if str(version).startswith("v") else f"v{version}"
    parts = [
        _safe_display_token(endpoint),
        _safe_display_token(dataset),
        _safe_display_token(protocol),
        _safe_display_token(backend),
    ]
    if representation:
        parts.append(_safe_display_token(representation))
    parts.append(version_token)
    return " ".join(parts).strip()


def _find_first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


def _first_existing_or_default(candidates: List[Path]) -> Path:
    return _find_first_existing_path(candidates) or candidates[0]


def _find_training_summary_from_model_run(run_dir: Optional[Path]) -> Optional[Path]:
    if run_dir is None:
        return None
    candidates: List[Path] = []
    current = run_dir.expanduser().resolve()
    for parent in [current, *current.parents[:3]]:
        candidates.extend(
            [
                parent / "cs_copilot_training_summary.json",
                parent / "tabicl_training_summary.json",
            ]
        )
    return _find_first_existing_path(candidates)


def _is_archive_model_path(path: str) -> bool:
    normalized = str(path or "").lower()
    return any(normalized.endswith(suffix) for suffix in ARCHIVE_MODEL_PATH_SUFFIXES)


class ModelRegistryToolkit(Toolkit):
    """Backend-neutral registry/catalog operations for prediction models."""

    def __init__(
        self,
        *,
        backends: Mapping[str, Any],
        catalog: Optional[PredictionModelCatalog] = None,
        default_backend_name: str = "chemprop",
        register_tools: bool = True,
    ):
        super().__init__("model_registry")
        self.backends = dict(backends)
        self.default_backend_name = default_backend_name
        self.catalog = catalog or PredictionModelCatalog.load()
        self.catalog.refresh_from_internal_store(persist=True)

        if register_tools:
            self.register(self.describe_backends)
            self.register(self.describe_catalog)
            self.register(self.list_catalog_models)
            self.register(self.summarize_catalog_model)
            self.register(self.recommend_catalog_model)
            self.register(self.register_catalog_model)
            self.register(self.register_model)
            self.register(self.persist_registered_model)
            self.register(self.list_registered_models)
            self.register(self.summarize_model)

    def describe_backends(self) -> Dict[str, Any]:
        """Describe all configured prediction backends."""
        return {name: backend.describe_environment() for name, backend in self.backends.items()}

    def get_backend(self, backend_name: str):
        backend = self.backends.get(backend_name)
        if backend is None:
            raise ValueError(f"Unsupported prediction backend: {backend_name}")
        return backend

    def describe_catalog(self) -> Dict[str, Any]:
        """Describe the persistent model catalog configured for prediction."""
        self.catalog.refresh_from_internal_store(persist=True)
        return {
            "catalog_path": str(self.catalog.source_path),
            "num_models": len(self.catalog.list_models()),
            "model_ids": [record.model_id for record in self.catalog.list_models()],
        }

    def annotate_record(self, record: PredictionModelRecord) -> Dict[str, Any]:
        payload = _compact_model_payload_for_response(record.as_dict())
        payload["backend_environment"] = self.get_backend(
            record.backend_name
        ).describe_environment()
        payload["model_path_exists"] = Path(record.model_path).expanduser().exists()
        return payload

    def list_catalog_models(
        self,
        allowed_statuses: Optional[List[str]] = None,
        include_unavailable_paths: bool = False,
    ) -> List[Dict[str, Any]]:
        """List models from the persistent catalog with runtime annotations."""
        self.catalog.refresh_from_internal_store(persist=True)
        available_backends = [
            name for name, backend in self.backends.items() if backend.is_available()
        ]
        candidates = self.catalog.search(
            allowed_statuses=allowed_statuses,
            backend_available=bool(available_backends),
            available_backend_names=available_backends,
            include_unavailable_paths=include_unavailable_paths,
        )
        return [
            _compact_model_payload_for_response(candidate.as_dict()) for candidate in candidates
        ]

    def summarize_catalog_model(self, model_id: str) -> Dict[str, Any]:
        """Return the catalog metadata for one model, enriched with runtime checks."""
        self.catalog.refresh_from_internal_store(persist=True)
        return self.annotate_record(self.catalog.get_model(model_id))

    def recommend_catalog_model(
        self,
        task_type: str,
        target_hint: Optional[str] = None,
        domain_hint: Optional[str] = None,
        require_uncertainty: bool = False,
        allowed_statuses: Optional[List[str]] = None,
        preferred_backend: Optional[str] = None,
        include_unavailable_paths: bool = True,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Recommend the best catalog model for a requested task."""
        self.catalog.refresh_from_internal_store(persist=True)
        recommendation = self.catalog.recommend(
            task_type=task_type,
            target_hint=target_hint,
            domain_hint=domain_hint,
            require_uncertainty=require_uncertainty,
            allowed_statuses=allowed_statuses,
            preferred_backend=preferred_backend,
            backend_available=any(backend.is_available() for backend in self.backends.values()),
            available_backend_names=[
                name for name, backend in self.backends.items() if backend.is_available()
            ],
            include_unavailable_paths=include_unavailable_paths,
        )

        compact_recommendation = _compact_recommendation_for_response(recommendation)
        if agent is not None:
            prediction_state = get_prediction_state(agent)
            prediction_state["catalog_recommendations"] = compact_recommendation

        return compact_recommendation

    def register_catalog_model(
        self, model_id: str, agent: Optional[Agent] = None
    ) -> Dict[str, Any]:
        """Register a model from the persistent catalog into the current session."""
        if agent is None:
            raise ValueError("Agent is required to register a catalog model")

        self.catalog.refresh_from_internal_store(persist=True)
        try:
            record = self.catalog.get_model(model_id)
        except ValueError as exc:
            available_ids = [record.model_id for record in self.catalog.list_models()]
            return {
                "registered": False,
                "error": str(exc),
                "model_id": model_id,
                "available_model_ids": available_ids,
                "usage_hint": (
                    "register_catalog_model only accepts an existing persistent catalog model_id. "
                    "For a newly trained session model, call persist_registered_model and use its returned "
                    "canonical model_id instead of inventing a display name."
                ),
            }
        result = self.register_model(
            model_id=record.model_id,
            model_path=record.model_path,
            backend_name=record.backend_name,
            task_type=record.task.task_type,
            smiles_columns=record.task.smiles_columns,
            target_columns=record.task.target_columns,
            reaction_columns=record.task.reaction_columns,
            uncertainty_method=record.task.uncertainty_method,
            calibration_method=record.task.calibration_method,
            description=record.description,
            tags=record.tags,
            version=record.version,
            status=record.status,
            owner=record.owner,
            source=record.source,
            domain_summary=record.domain_summary,
            strengths=record.strengths,
            limitations=record.limitations,
            recommended_for=record.recommended_for,
            not_recommended_for=record.not_recommended_for,
            known_metrics=record.known_metrics,
            training_data_summary=record.training_data_summary,
            inference_profile=record.inference_profile,
            selection_hints=record.selection_hints,
            applicability_domain=record.applicability_domain,
            agent=agent,
        )
        prediction_state = get_prediction_state(agent)
        prediction_state["registered"][record.model_id] = record.as_dict()
        result.update(
            {
                "metadata_path": record.metadata_path,
                "persisted": True,
                "catalog_persisted": True,
                "persistence_state": "catalog_registered",
                "next_required_tool": None,
            }
        )
        return result

    def register_model(
        self,
        model_id: str,
        model_path: str,
        task_type: str,
        backend_name: Optional[str] = None,
        smiles_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        reaction_columns: Optional[List[str]] = None,
        uncertainty_method: Optional[str] = None,
        calibration_method: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        version: Optional[str] = None,
        status: str = "experimental",
        owner: Optional[str] = None,
        source: Optional[str] = None,
        domain_summary: Optional[str] = None,
        strengths: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        recommended_for: Optional[List[str]] = None,
        not_recommended_for: Optional[List[str]] = None,
        known_metrics: Optional[Dict[str, Any]] = None,
        applicability_domain: Optional[Dict[str, Any]] = None,
        training_data_summary: Optional[Dict[str, Any]] = None,
        inference_profile: Optional[Dict[str, Any]] = None,
        selection_hints: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Register a prediction model in session state for later use."""
        if agent is None:
            raise ValueError("Agent is required to register a model")

        resolved_backend_name = backend_name or self.default_backend_name
        backend = self.get_backend(resolved_backend_name)
        if _is_archive_model_path(model_path):
            expected_extensions = getattr(backend, "MODEL_EXTENSIONS", None)
            return {
                "registered": False,
                "model_id": model_id,
                "model_path": model_path,
                "backend_name": backend.backend_name,
                "error": "Archive paths cannot be registered as model artifacts.",
                "expected_model_extensions": list(expected_extensions or []),
                "usage_hint": (
                    "Use the trained model artifact path returned by the training tool "
                    "(for example `best.pkl` for LightGBM/TabICL or `best.pt`/`.ckpt` "
                    "for Chemprop), not the downloadable training bundle/archive."
                ),
            }
        validated_path = backend.validate_model_path(model_path)
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=smiles_columns or ["smiles"],
            target_columns=target_columns or [],
            reaction_columns=reaction_columns or [],
            uncertainty_method=uncertainty_method,
            calibration_method=calibration_method,
        )
        record = PredictionModelRecord(
            model_id=model_id,
            backend_name=backend.backend_name,
            model_path=str(validated_path),
            metadata_path=None,
            task=task,
            description=description,
            tags=tags or {},
            version=version,
            status=status,
            owner=owner,
            source=source,
            domain_summary=domain_summary,
            strengths=strengths or [],
            limitations=limitations or [],
            recommended_for=recommended_for or [],
            not_recommended_for=not_recommended_for or [],
            known_metrics=known_metrics or {},
            training_data_summary=training_data_summary or {},
            inference_profile=inference_profile or {},
            selection_hints=selection_hints or {},
            applicability_domain=applicability_domain or {},
        )

        prediction_state = get_prediction_state(agent)
        prediction_state["registered"][model_id] = record.as_dict()
        payload = _compact_model_payload_for_response(record.as_dict())
        payload.update(
            {
                "registered": True,
                "persisted": False,
                "catalog_persisted": False,
                "persistence_state": "session_registered_only",
                "next_required_tool": "persist_registered_model",
                "usage_hint": (
                    "This model is registered only in the current session. "
                    "For persistent catalog storage, call `persist_registered_model` "
                    "and use the returned canonical `model_id`, `model_root`, "
                    "`model_path`, and `metadata_path`."
                ),
            }
        )
        return payload

    def _infer_training_run_dir(self, record: PredictionModelRecord) -> Optional[Path]:
        model_path = Path(record.model_path).expanduser()
        if not model_path.exists():
            return None
        if model_path.parent.name == "model_0":
            return model_path.parent.parent
        return model_path.parent

    def _materialize_internal_model(
        self,
        *,
        record: PredictionModelRecord,
        train_csv: Optional[str] = None,
        model_id: Optional[str] = None,
        source_artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run_dir = self._infer_training_run_dir(record)
        if run_dir is None or not run_dir.exists():
            return {"materialized": False, "reason": "training_run_not_found"}

        source_artifacts = source_artifacts or {}
        resolved_model_id = model_id or record.model_id
        model_root = DEFAULT_INTERNAL_MODEL_ROOT / resolved_model_id
        model_dir = model_root / "model"
        artifacts_dir = model_root / "artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        source_model_path = Path(record.model_path).expanduser()
        copied_files: Dict[str, Any] = {}
        backend_capabilities = get_backend_capabilities(record.backend_name)

        if source_model_path.exists():
            target_name = backend_capabilities.catalog_model_filename or source_model_path.name
            target_model_path = model_dir / target_name
            shutil.copy2(source_model_path, target_model_path)
            copied_files["model_path"] = _relative_posix(target_model_path, model_root)
        else:
            return {"materialized": False, "reason": "model_artifact_missing"}

        training_summary_candidates = [
            run_dir / filename for filename in backend_capabilities.training_summary_filenames
        ]
        test_prediction_candidates = [
            run_dir / relative_path
            for relative_path in backend_capabilities.test_prediction_relative_paths
        ]
        optional_artifacts = {
            "config_path": run_dir / "config.toml",
            "training_summary_path": _first_existing_or_default(training_summary_candidates),
            "splits_path": run_dir / "splits.json",
            "validation_predictions_path": run_dir / "model_0" / "validation_predictions.csv",
            "test_predictions_path": _first_existing_or_default(test_prediction_candidates),
            "chemprop_training_input_csv": run_dir
            / "chemprop_inputs"
            / "chemprop_training_input.csv",
            "chemprop_splits_file": run_dir / "chemprop_inputs" / "chemprop_splits.json",
            "chemprop_input_manifest_path": run_dir
            / "chemprop_inputs"
            / "chemprop_input_manifest.json",
            "reference_store_path": run_dir / "applicability_domain" / "reference_fingerprints.npz",
            "reference_manifest_path": run_dir / "applicability_domain" / "reference_manifest.json",
            "applicability_domain_path": run_dir
            / "applicability_domain"
            / "applicability_domain.json",
            "ad_manifest_path": run_dir / "applicability_domain" / "manifest.json",
            "ad_bounds_path": run_dir
            / "applicability_domain"
            / "bounding_box"
            / "bounds.npz",
            "ad_isolation_forest_model_path": run_dir
            / "applicability_domain"
            / "isolation_forest"
            / "model.joblib",
            "ad_similarity_matrix_dir": run_dir
            / "applicability_domain"
            / "similarity_matrix",
            "ad_scores_train_path": run_dir / "applicability_domain" / "scores_train.csv",
            "ad_scores_validation_path": run_dir
            / "applicability_domain"
            / "scores_validation.csv",
            "ad_scores_test_path": run_dir / "applicability_domain" / "scores_test.csv",
        }
        plot_sources: Dict[str, Path] = {}
        activity_cliff_sources: Dict[str, Path] = {}
        activity_cliff_variant_model_sources: List[Dict[str, Any]] = []
        split_prediction_sources: Dict[str, Path] = {}
        split_splits_sources: Dict[str, Path] = {}
        curation_sources: Dict[str, Path] = {}
        curation_payload: Dict[str, Any] = {}
        feature_preparation_payload: Dict[str, Any] = {}

        for key in (
            "config_path",
            "training_summary_path",
            "splits_path",
            "validation_predictions_path",
            "test_predictions_path",
            "chemprop_training_input_csv",
            "chemprop_splits_file",
            "chemprop_input_manifest_path",
            "reference_store_path",
            "reference_manifest_path",
            "applicability_domain_path",
            "ad_manifest_path",
            "ad_bounds_path",
            "ad_isolation_forest_model_path",
            "ad_scores_train_path",
            "ad_scores_validation_path",
            "ad_scores_test_path",
        ):
            raw_path = source_artifacts.get(key)
            if raw_path:
                optional_artifacts[key] = Path(str(raw_path)).expanduser()

        activity_plot_names = {
            "activity_cliff_score_histogram",
            "activity_cliff_tier_distribution",
            "activity_gap_vs_similarity",
        }
        for plot_name, raw_path in (source_artifacts.get("plot_artifacts") or {}).items():
            if plot_name in activity_plot_names:
                continue
            if raw_path:
                plot_sources[plot_name] = Path(str(raw_path)).expanduser()
        for split_result in source_artifacts.get("split_results") or []:
            raw_path = split_result.get("test_predictions_path")
            if not raw_path:
                continue
            split_label = (
                split_result.get("strategy_label")
                or split_result.get("strategy_family")
                or split_result.get("strategy")
                or split_result.get("split_type")
                or "split"
            )
            split_prediction_sources[safe_slug(str(split_label)) or "split"] = Path(
                str(raw_path)
            ).expanduser()
            raw_splits_path = split_result.get("splits_path")
            if raw_splits_path:
                split_splits_sources[safe_slug(str(split_label)) or "split"] = Path(
                    str(raw_splits_path)
                ).expanduser()
        curation_payload = source_artifacts.get("curation") or {}
        if curation_payload.get("curated_dataset_path"):
            curation_sources["curated_dataset_csv"] = Path(
                str(curation_payload["curated_dataset_path"])
            ).expanduser()
        for artifact_name, raw_path in (curation_payload.get("artifacts") or {}).items():
            if raw_path:
                curation_sources.setdefault(
                    str(artifact_name),
                    Path(str(raw_path)).expanduser(),
                )
        feature_preparation_payload = source_artifacts.get("feature_preparation") or {}
        activity_payload = source_artifacts.get("activity_cliffs") or {}
        for key in (
            "annotated_training_csv",
            "summary_path",
            "clean_training_csv",
        ):
            raw_path = activity_payload.get(key)
            if raw_path:
                activity_cliff_sources[key] = Path(str(raw_path)).expanduser()
        for variant in activity_payload.get("variants") or []:
            raw_path = variant.get("filtered_training_csv")
            if raw_path:
                activity_cliff_sources[f"variant_{variant.get('variant_id')}"] = Path(
                    str(raw_path)
                ).expanduser()
        for plot_name, raw_path in (activity_payload.get("plot_artifacts") or {}).items():
            if raw_path:
                activity_cliff_sources[f"plot_{plot_name}"] = Path(str(raw_path)).expanduser()
        for plot_name, raw_path in (
            activity_payload.get("loop_comparison_plot_artifacts") or {}
        ).items():
            if raw_path:
                activity_cliff_sources[f"loop_comparison_plot_{plot_name}"] = Path(
                    str(raw_path)
                ).expanduser()
        for variant in activity_payload.get("variant_training") or []:
            variant_id = variant.get("variant_id")
            if not variant_id:
                continue
            for split_result in variant.get("split_results") or []:
                split_label = (
                    split_result.get("strategy_label")
                    or split_result.get("strategy_family")
                    or "split"
                )
                for artifact_key in (
                    "model_path",
                    "test_predictions_path",
                    "splits_path",
                ):
                    raw_path = split_result.get(artifact_key)
                    if raw_path:
                        activity_cliff_variant_model_sources.append(
                            {
                                "variant_id": str(variant_id),
                                "split_label": str(split_label),
                                "artifact_key": artifact_key,
                                "source_path": Path(str(raw_path)).expanduser(),
                            }
                        )

        for key, source_path in optional_artifacts.items():
            if source_path.exists():
                if key == "config_path":
                    target_path = model_dir / "config.toml"
                elif key == "training_summary_path":
                    target_path = artifacts_dir / "cs_copilot_training_summary.json"
                elif key == "splits_path":
                    target_path = artifacts_dir / "splits.json"
                elif key == "validation_predictions_path":
                    target_path = artifacts_dir / "validation_predictions.csv"
                elif key == "test_predictions_path":
                    target_path = artifacts_dir / "test_predictions.csv"
                elif key.startswith("chemprop_"):
                    target_path = artifacts_dir / "chemprop_inputs" / source_path.name
                elif key == "ad_manifest_path":
                    target_path = artifacts_dir / "applicability_domain" / "manifest.json"
                elif key == "ad_bounds_path":
                    target_path = (
                        artifacts_dir / "applicability_domain" / "bounding_box" / "bounds.npz"
                    )
                elif key == "ad_isolation_forest_model_path":
                    target_path = (
                        artifacts_dir
                        / "applicability_domain"
                        / "isolation_forest"
                        / "model.joblib"
                    )
                elif key == "ad_similarity_matrix_dir":
                    target_path = artifacts_dir / "applicability_domain" / "similarity_matrix"
                elif key.startswith("ad_scores_"):
                    target_path = artifacts_dir / "applicability_domain" / source_path.name
                else:
                    target_path = artifacts_dir / source_path.name
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if source_path.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(source_path, target_path)
                else:
                    shutil.copy2(source_path, target_path)
                copied_files[key] = _relative_posix(target_path, model_root)

        copied_plot_artifacts: Dict[str, str] = {}
        if plot_sources:
            plots_dir = artifacts_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            for plot_name, source_path in plot_sources.items():
                if not source_path.exists():
                    continue
                target_path = plots_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_plot_artifacts[plot_name] = _relative_posix(target_path, model_root)
        if copied_plot_artifacts:
            copied_files["plot_artifacts"] = copied_plot_artifacts

        copied_split_predictions: Dict[str, str] = {}
        if split_prediction_sources:
            split_preds_dir = artifacts_dir / "test_predictions_by_split"
            split_preds_dir.mkdir(parents=True, exist_ok=True)
            for split_label, source_path in split_prediction_sources.items():
                if not source_path.exists():
                    continue
                target_path = split_preds_dir / f"test_predictions_{split_label}.csv"
                shutil.copy2(source_path, target_path)
                copied_split_predictions[split_label] = _relative_posix(target_path, model_root)
        if copied_split_predictions:
            copied_files["test_predictions_by_split"] = copied_split_predictions

        copied_split_splits: Dict[str, str] = {}
        if split_splits_sources:
            split_splits_dir = artifacts_dir / "splits_by_split"
            split_splits_dir.mkdir(parents=True, exist_ok=True)
            for split_label, source_path in split_splits_sources.items():
                if not source_path.exists():
                    continue
                target_path = split_splits_dir / f"splits_{split_label}.json"
                shutil.copy2(source_path, target_path)
                copied_split_splits[split_label] = _relative_posix(target_path, model_root)
        if copied_split_splits:
            copied_files["splits_by_split"] = copied_split_splits

        copied_curation_artifacts: Dict[str, str] = {}
        if curation_sources:
            curation_dir = artifacts_dir / "curation"
            curation_dir.mkdir(parents=True, exist_ok=True)
            for artifact_name, source_path in curation_sources.items():
                if not source_path.exists():
                    continue
                target_path = curation_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_curation_artifacts[artifact_name] = _relative_posix(target_path, model_root)
        if copied_curation_artifacts:
            copied_files["curation"] = copied_curation_artifacts

        copied_activity_cliff_artifacts: Dict[str, str] = {}
        if activity_cliff_sources:
            activity_dir = artifacts_dir / "activity_cliffs"
            activity_dir.mkdir(parents=True, exist_ok=True)
            for artifact_name, source_path in activity_cliff_sources.items():
                if not source_path.exists():
                    continue
                target_dir = (
                    activity_dir / "loop_comparison"
                    if str(artifact_name).startswith("loop_comparison_plot_")
                    else activity_dir
                )
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                copied_activity_cliff_artifacts[artifact_name] = _relative_posix(
                    target_path, model_root
                )
        if copied_activity_cliff_artifacts:
            copied_files["activity_cliffs"] = copied_activity_cliff_artifacts

        copied_activity_cliff_variant_models: Dict[str, str] = {}
        if activity_cliff_variant_model_sources:
            variants_dir = artifacts_dir / "activity_cliffs" / "variant_models"
            for item in activity_cliff_variant_model_sources:
                source_path = item["source_path"]
                if not source_path.exists():
                    continue
                target_dir = variants_dir / str(item["variant_id"]) / str(item["split_label"])
                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / source_path.name
                shutil.copy2(source_path, target_path)
                key = f"{item['variant_id']}_{item['split_label']}_{item['artifact_key']}"
                copied_activity_cliff_variant_models[key] = _relative_posix(target_path, model_root)
        if copied_activity_cliff_variant_models:
            copied_files["activity_cliff_variant_models"] = copied_activity_cliff_variant_models

        if train_csv:
            source_train_csv = Path(train_csv).expanduser()
            if source_train_csv.exists():
                target_train_csv = artifacts_dir / source_train_csv.name
                shutil.copy2(source_train_csv, target_train_csv)
                copied_files["training_dataset_path"] = _relative_posix(
                    target_train_csv, model_root
                )

        copied_feature_preparation: Dict[str, Any] = {}
        if feature_preparation_payload:
            features_dir = artifacts_dir / "features"
            copied_feature_tables: Dict[str, str] = {}
            path_rewrites: Dict[str, str] = {}
            if copied_files.get("training_dataset_path") and train_csv:
                path_rewrites[_resolved_path_key(train_csv)] = copied_files["training_dataset_path"]

            for index, raw_path in enumerate(
                feature_preparation_payload.get("feature_csvs") or [], start=1
            ):
                if not raw_path:
                    continue
                source_path = Path(str(raw_path)).expanduser()
                if not source_path.exists():
                    continue
                features_dir.mkdir(parents=True, exist_ok=True)
                target_path = features_dir / source_path.name
                if target_path.exists():
                    target_path = features_dir / f"{target_path.stem}_{index}{target_path.suffix}"
                shutil.copy2(source_path, target_path)
                relative_target = _relative_posix(target_path, model_root)
                copied_feature_tables[source_path.stem] = relative_target
                path_rewrites[_resolved_path_key(source_path)] = relative_target

            copied_feature_preparation = dict(feature_preparation_payload)
            if copied_feature_tables:
                copied_feature_preparation["feature_csvs"] = list(copied_feature_tables.values())
                copied_feature_preparation["feature_tables"] = copied_feature_tables
                copied_files["feature_tables"] = copied_feature_tables
            prepared_train_csv = copied_feature_preparation.get("prepared_train_csv")
            if prepared_train_csv and _resolved_path_key(prepared_train_csv) in path_rewrites:
                copied_feature_preparation["prepared_train_csv"] = path_rewrites[
                    _resolved_path_key(prepared_train_csv)
                ]
            durations = copied_feature_preparation.get("durations") or {}
            rewritten_steps = []
            for step in durations.get("steps") or []:
                rewritten_step = dict(step)
                output_csv = rewritten_step.get("output_csv")
                if output_csv and _resolved_path_key(output_csv) in path_rewrites:
                    rewritten_step["output_csv"] = path_rewrites[_resolved_path_key(output_csv)]
                rewritten_steps.append(rewritten_step)
            if durations:
                copied_feature_preparation["durations"] = {
                    **durations,
                    "steps": rewritten_steps,
                }

        if copied_files.get("ad_manifest_path"):
            manifest_path = model_root / copied_files["ad_manifest_path"]
            try:
                manifest_payload = (
                    json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                )
            except Exception:
                manifest_payload = {}
            if manifest_payload:
                manifest_payload["manifest_path"] = copied_files.get("ad_manifest_path")
                if copied_files.get("ad_bounds_path"):
                    manifest_payload["bounds_path"] = copied_files["ad_bounds_path"]
                    methods = dict(manifest_payload.get("methods") or {})
                    bounding_box = dict(methods.get("bounding_box") or {})
                    bounding_box["bounds_path"] = copied_files["ad_bounds_path"]
                    bounding_box["manifest_path"] = copied_files.get("ad_manifest_path")
                    methods["bounding_box"] = bounding_box
                    manifest_payload["methods"] = methods
                if copied_files.get("ad_isolation_forest_model_path"):
                    manifest_payload["isolation_forest_model_path"] = copied_files[
                        "ad_isolation_forest_model_path"
                    ]
                    methods = dict(manifest_payload.get("methods") or {})
                    isolation_forest = dict(methods.get("isolation_forest") or {})
                    isolation_forest["model_path"] = copied_files[
                        "ad_isolation_forest_model_path"
                    ]
                    methods["isolation_forest"] = isolation_forest
                    manifest_payload["methods"] = methods
                if copied_files.get("ad_similarity_matrix_dir"):
                    similarity_dir = copied_files["ad_similarity_matrix_dir"]
                    manifest_payload["similarity_matrix_manifest_path"] = (
                        f"{similarity_dir}/manifest.json"
                    )
                    manifest_payload["activity_cliffs_reusable"] = True
                    methods = dict(manifest_payload.get("methods") or {})
                    similarity = dict(methods.get("similarity_matrix") or {})
                    similarity["manifest_path"] = f"{similarity_dir}/manifest.json"
                    subspaces = dict(similarity.get("subspaces") or {})
                    for subspace, payload in list(subspaces.items()):
                        subspace_payload = dict(payload or {})
                        subspace_payload["matrix_all_path"] = (
                            f"{similarity_dir}/{subspace}/matrix_all.npy"
                        )
                        subspace_payload["reference_features_path"] = (
                            f"{similarity_dir}/{subspace}/reference_features.npz"
                        )
                        subspaces[subspace] = subspace_payload
                    similarity["subspaces"] = subspaces
                    methods["similarity_matrix"] = similarity
                    manifest_payload["methods"] = methods
                for source_key, manifest_key in (
                    ("ad_scores_train_path", "scores_train_path"),
                    ("ad_scores_validation_path", "scores_validation_path"),
                    ("ad_scores_test_path", "scores_test_path"),
                ):
                    if copied_files.get(source_key):
                        manifest_payload[manifest_key] = copied_files[source_key]
                manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n")

        metadata_description = _sanitize_activity_cliff_description(
            record.description or "",
            source_artifacts.get("activity_cliffs") or {},
        )
        metadata = {
            "model_id": resolved_model_id,
            "display_name": record.display_name or resolved_model_id,
            "version": record.version or "1.0",
            "status": record.status,
            "owner": record.owner or "chemspacecopilot",
            "source": record.source or "internal_training",
            "backend_name": record.backend_name,
            "task": {
                "task_type": record.task.task_type,
                "smiles_columns": list(record.task.smiles_columns),
                "target_columns": list(record.task.target_columns),
                "reaction_columns": list(record.task.reaction_columns),
                "uncertainty_method": record.task.uncertainty_method,
                "calibration_method": record.task.calibration_method,
            },
            "description": metadata_description or "",
            "domain_summary": record.domain_summary or "",
            "strengths": list(record.strengths),
            "limitations": list(record.limitations),
            "recommended_for": list(record.recommended_for),
            "not_recommended_for": list(record.not_recommended_for),
            "known_metrics": dict(record.known_metrics),
            "training_data_summary": dict(record.training_data_summary),
            "external_evaluations": list(
                record.training_data_summary.get("external_evaluations") or []
            ),
            "inference_profile": dict(record.inference_profile),
            "selection_hints": dict(record.selection_hints),
            "tags": dict(record.tags),
            "artifacts": copied_files,
        }
        metadata.update(_classification_metadata(record))
        metrics_status = record.training_data_summary.get("metrics_status")
        if metrics_status:
            metadata["metrics_status"] = metrics_status
            metadata["evaluation_required"] = bool(
                record.training_data_summary.get("evaluation_required")
            )
        if record.training_data_summary.get("seed_policy"):
            metadata["reproducibility"] = seed_policy_reproducibility_metadata(
                record.training_data_summary.get("seed_policy")
            )
        modern_ad_available = bool(
            copied_files.get("ad_manifest_path")
            or copied_files.get("ad_bounds_path")
            or copied_files.get("ad_isolation_forest_model_path")
            or copied_files.get("ad_similarity_matrix_dir")
        )
        if modern_ad_available:
            metadata_ad = dict(record.applicability_domain or {})
            metadata_ad.update(
                {
                    "available": True,
                    "primary_method": metadata_ad.get("primary_method") or "combined",
                    "method": metadata_ad.get("method") or "combined",
                    "manifest_path": copied_files.get("ad_manifest_path")
                    or metadata_ad.get("manifest_path"),
                    "bounds_path": copied_files.get("ad_bounds_path")
                    or metadata_ad.get("bounds_path"),
                    "isolation_forest_model_path": copied_files.get(
                        "ad_isolation_forest_model_path"
                    )
                    or metadata_ad.get("isolation_forest_model_path"),
                    "similarity_matrix_manifest_path": (
                        f"{copied_files.get('ad_similarity_matrix_dir')}/manifest.json"
                        if copied_files.get("ad_similarity_matrix_dir")
                        else metadata_ad.get("similarity_matrix_manifest_path")
                    ),
                }
            )
            methods = dict(metadata_ad.get("methods") or {})
            bounding_box = dict(methods.get("bounding_box") or {})
            if copied_files.get("ad_bounds_path"):
                bounding_box["bounds_path"] = copied_files["ad_bounds_path"]
            if copied_files.get("ad_manifest_path") and bounding_box:
                bounding_box["manifest_path"] = copied_files["ad_manifest_path"]
            if bounding_box:
                methods["bounding_box"] = bounding_box
            isolation_forest = dict(methods.get("isolation_forest") or {})
            if copied_files.get("ad_isolation_forest_model_path"):
                isolation_forest["model_path"] = copied_files["ad_isolation_forest_model_path"]
            if isolation_forest:
                methods["isolation_forest"] = isolation_forest
            similarity = dict(methods.get("similarity_matrix") or {})
            if copied_files.get("ad_similarity_matrix_dir"):
                similarity_dir = copied_files["ad_similarity_matrix_dir"]
                similarity["manifest_path"] = f"{similarity_dir}/manifest.json"
                subspaces = dict(similarity.get("subspaces") or {})
                for subspace, payload in list(subspaces.items()):
                    subspace_payload = dict(payload or {})
                    subspace_payload["matrix_all_path"] = (
                        f"{similarity_dir}/{subspace}/matrix_all.npy"
                    )
                    subspace_payload["reference_features_path"] = (
                        f"{similarity_dir}/{subspace}/reference_features.npz"
                    )
                    subspaces[subspace] = subspace_payload
                similarity["subspaces"] = subspaces
            if similarity:
                methods["similarity_matrix"] = similarity
            metadata_ad["methods"] = methods
            for source_key, metadata_key in (
                ("ad_scores_train_path", "scores_train_path"),
                ("ad_scores_validation_path", "scores_validation_path"),
                ("ad_scores_test_path", "scores_test_path"),
            ):
                if copied_files.get(source_key):
                    metadata_ad[metadata_key] = copied_files[source_key]
            if copied_files.get("applicability_domain_path"):
                metadata_ad["legacy_similarity_ad"] = {
                    "available": True,
                    "legacy": True,
                    "method": "legacy_similarity_ad",
                    "reference_store_path": copied_files.get("reference_store_path"),
                    "reference_manifest_path": copied_files.get("reference_manifest_path"),
                    "index_path": copied_files.get("applicability_domain_path"),
                }
            else:
                metadata_ad.pop("legacy_similarity_ad", None)
            metadata["applicability_domain"] = metadata_ad
        elif copied_files.get("applicability_domain_path"):
            metadata["applicability_domain"] = {
                "available": True,
                "legacy": True,
                "method": "legacy_similarity_ad",
                "reference_store_path": copied_files.get("reference_store_path"),
                "reference_manifest_path": copied_files.get("reference_manifest_path"),
                "index_path": copied_files.get("applicability_domain_path"),
            }
        if copied_plot_artifacts:
            metadata["plot_artifacts"] = copied_plot_artifacts
        if copied_split_predictions:
            metadata["test_predictions_by_split"] = copied_split_predictions
        if copied_split_splits:
            metadata["splits_by_split"] = copied_split_splits
        if copied_curation_artifacts:
            metadata["curation"] = _hydrate_curation_metadata(
                curation_payload=curation_payload,
                copied_curation_artifacts=copied_curation_artifacts,
                model_root=model_root,
            )
        if copied_feature_preparation:
            metadata["feature_preparation"] = copied_feature_preparation
        if copied_activity_cliff_artifacts:
            activity_payload = source_artifacts.get("activity_cliffs") or {}
            metadata["activity_cliffs"] = {
                "enabled": bool(activity_payload.get("enabled")),
                "mode": activity_payload.get("mode"),
                "index_name": activity_payload.get("index_name"),
                "flagged_count": activity_payload.get("flagged_count"),
                "priority_counts": activity_payload.get("priority_counts"),
                "recommended_variant": activity_payload.get("recommended_variant"),
                "recommendation_reason": activity_payload.get("recommendation_reason"),
                "artifacts": copied_activity_cliff_artifacts,
            }
            if activity_payload.get("variant_training"):
                metadata["activity_cliffs"]["variant_training"] = activity_payload.get(
                    "variant_training"
                )
            if activity_payload.get("variant_comparison_table"):
                metadata["activity_cliffs"]["variant_comparison_table"] = activity_payload.get(
                    "variant_comparison_table"
                )
            if activity_payload.get("reporting_handoff"):
                metadata["activity_cliffs"]["reporting_handoff"] = activity_payload.get(
                    "reporting_handoff"
                )
            if activity_payload.get("loop_comparison_plot_artifacts"):
                metadata["activity_cliffs"]["loop_comparison_plot_artifacts"] = {
                    key: copied_activity_cliff_artifacts.get(f"loop_comparison_plot_{key}")
                    for key in activity_payload.get("loop_comparison_plot_artifacts")
                    if copied_activity_cliff_artifacts.get(f"loop_comparison_plot_{key}")
                }
            if copied_activity_cliff_variant_models:
                metadata["activity_cliffs"][
                    "variant_model_artifacts"
                ] = copied_activity_cliff_variant_models
        metadata_path = model_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        return {
            "materialized": True,
            "model_root": str(model_root),
            "model_path": str(model_root / copied_files["model_path"]),
            "metadata_path": str(metadata_path),
            "artifacts": copied_files,
        }

    def _sync_internal_metadata(
        self,
        *,
        metadata_path: Optional[str],
        record: PredictionModelRecord,
        governance_assessment: Optional[Dict[str, Any]] = None,
        status_reason: Optional[str] = None,
    ) -> None:
        if not metadata_path:
            return
        target = Path(metadata_path).expanduser()
        if not target.exists():
            return
        try:
            payload = json.loads(target.read_text())
        except Exception:
            payload = {}
        payload.update(
            {
                "model_id": record.model_id,
                "display_name": record.display_name or record.model_id,
                "version": record.version or "1.0",
                "status": record.status,
                "owner": record.owner or "chemspacecopilot",
                "source": record.source or "internal_training",
                "backend_name": record.backend_name,
                "description": record.description or "",
                "domain_summary": record.domain_summary or "",
                "known_metrics": dict(record.known_metrics),
                "training_data_summary": dict(record.training_data_summary),
                "external_evaluations": payload.get("external_evaluations")
                or record.training_data_summary.get("external_evaluations")
                or [],
                "trained_at": (record.training_data_summary or {}).get("trained_at"),
                "trained_date": (record.training_data_summary or {}).get("trained_date"),
                "trained_time": (record.training_data_summary or {}).get("trained_time"),
                "inference_profile": dict(record.inference_profile),
                "selection_hints": dict(record.selection_hints),
                "strengths": list(record.strengths),
                "limitations": list(record.limitations),
                "recommended_for": list(record.recommended_for),
                "not_recommended_for": list(record.not_recommended_for),
                "tags": dict(record.tags),
                "reproducibility": seed_policy_reproducibility_metadata(
                    (record.training_data_summary or {}).get("seed_policy")
                ),
                "task": {
                    "task_type": record.task.task_type,
                    "smiles_columns": list(record.task.smiles_columns),
                    "target_columns": list(record.task.target_columns),
                    "reaction_columns": list(record.task.reaction_columns),
                    "uncertainty_method": record.task.uncertainty_method,
                    "calibration_method": record.task.calibration_method,
                },
            }
        )
        payload.update(_classification_metadata(record))
        metrics_status = (record.training_data_summary or {}).get("metrics_status")
        if metrics_status:
            payload["metrics_status"] = metrics_status
            payload["evaluation_required"] = bool(
                (record.training_data_summary or {}).get("evaluation_required")
            )
        artifacts = payload.get("artifacts") or {}
        if (
            artifacts.get("ad_manifest_path")
            or artifacts.get("ad_bounds_path")
            or artifacts.get("ad_isolation_forest_model_path")
            or artifacts.get("ad_similarity_matrix_dir")
        ):
            metadata_ad = dict(record.applicability_domain or {})
            metadata_ad.update(
                {
                    "available": True,
                    "primary_method": metadata_ad.get("primary_method") or "combined",
                    "method": metadata_ad.get("method") or "combined",
                    "manifest_path": artifacts.get("ad_manifest_path")
                    or metadata_ad.get("manifest_path"),
                    "bounds_path": artifacts.get("ad_bounds_path")
                    or metadata_ad.get("bounds_path"),
                    "isolation_forest_model_path": artifacts.get(
                        "ad_isolation_forest_model_path"
                    )
                    or metadata_ad.get("isolation_forest_model_path"),
                    "similarity_matrix_manifest_path": (
                        f"{artifacts.get('ad_similarity_matrix_dir')}/manifest.json"
                        if artifacts.get("ad_similarity_matrix_dir")
                        else metadata_ad.get("similarity_matrix_manifest_path")
                    ),
                }
            )
            methods = dict(metadata_ad.get("methods") or {})
            bounding_box = dict(methods.get("bounding_box") or {})
            if artifacts.get("ad_bounds_path"):
                bounding_box["bounds_path"] = artifacts.get("ad_bounds_path")
            if artifacts.get("ad_manifest_path") and bounding_box:
                bounding_box["manifest_path"] = artifacts.get("ad_manifest_path")
            if bounding_box:
                methods["bounding_box"] = bounding_box
            isolation_forest = dict(methods.get("isolation_forest") or {})
            if artifacts.get("ad_isolation_forest_model_path"):
                isolation_forest["model_path"] = artifacts.get(
                    "ad_isolation_forest_model_path"
                )
            if isolation_forest:
                methods["isolation_forest"] = isolation_forest
            similarity = dict(methods.get("similarity_matrix") or {})
            if artifacts.get("ad_similarity_matrix_dir"):
                similarity_dir = artifacts["ad_similarity_matrix_dir"]
                similarity["manifest_path"] = f"{similarity_dir}/manifest.json"
                subspaces = dict(similarity.get("subspaces") or {})
                for subspace, payload in list(subspaces.items()):
                    subspace_payload = dict(payload or {})
                    subspace_payload["matrix_all_path"] = (
                        f"{similarity_dir}/{subspace}/matrix_all.npy"
                    )
                    subspace_payload["reference_features_path"] = (
                        f"{similarity_dir}/{subspace}/reference_features.npz"
                    )
                    subspaces[subspace] = subspace_payload
                similarity["subspaces"] = subspaces
            if similarity:
                methods["similarity_matrix"] = similarity
            metadata_ad["methods"] = methods
            if artifacts.get("applicability_domain_path"):
                metadata_ad["legacy_similarity_ad"] = {
                    "available": True,
                    "legacy": True,
                    "method": "legacy_similarity_ad",
                    "reference_store_path": artifacts.get("reference_store_path"),
                    "reference_manifest_path": artifacts.get("reference_manifest_path"),
                    "index_path": artifacts.get("applicability_domain_path"),
                }
            else:
                metadata_ad.pop("legacy_similarity_ad", None)
            payload["applicability_domain"] = metadata_ad
        elif artifacts.get("applicability_domain_path"):
            payload["applicability_domain"] = {
                "available": True,
                "legacy": True,
                "method": "legacy_similarity_ad",
                "reference_store_path": artifacts.get("reference_store_path"),
                "reference_manifest_path": artifacts.get("reference_manifest_path"),
                "index_path": artifacts.get("applicability_domain_path"),
            }
        if artifacts.get("plot_artifacts"):
            payload["plot_artifacts"] = artifacts.get("plot_artifacts")
        if artifacts.get("activity_cliffs") and record.training_data_summary.get("activity_cliffs"):
            payload["activity_cliffs"] = record.training_data_summary.get("activity_cliffs")
            payload["activity_cliffs"]["artifacts"] = artifacts.get("activity_cliffs")
        if governance_assessment:
            payload["governance_assessment"] = governance_assessment
        if status_reason:
            payload["status_reason"] = status_reason
        target.write_text(json.dumps(payload, indent=2) + "\n")

    def persist_registered_model(
        self,
        model_id: str,
        status: Optional[str] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
        domain_summary: Optional[str] = None,
        owner: Optional[str] = None,
        version: Optional[str] = None,
        strengths: Optional[List[str]] = None,
        limitations: Optional[List[str]] = None,
        recommended_for: Optional[List[str]] = None,
        not_recommended_for: Optional[List[str]] = None,
        known_metrics: Optional[Dict[str, Any]] = None,
        applicability_domain: Optional[Dict[str, Any]] = None,
        training_data_summary: Optional[Dict[str, Any]] = None,
        inference_profile: Optional[Dict[str, Any]] = None,
        selection_hints: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Persist a session-registered model into the catalog JSON."""
        if agent is None:
            raise ValueError("Agent is required to persist a registered model")

        current = self.resolve_record(model_id, agent)
        prediction_state = get_prediction_state(agent)
        training_runs = prediction_state.get("training_runs") or []
        train_csv = None
        matching_training_run = None
        inferred_run_dir = self._infer_training_run_dir(current)
        if inferred_run_dir is not None:
            inferred_output = str(inferred_run_dir.resolve())
            for run in reversed(training_runs):
                if str(Path(run.get("output_dir", "")).expanduser().resolve()) == inferred_output:
                    matching_training_run = run
                    train_csv = run.get("train_csv")
                    break
                summary_candidate = _find_training_summary_from_model_run(inferred_run_dir)
                if summary_candidate is not None:
                    summary_root = str(summary_candidate.parent.resolve())
                    if str(Path(run.get("output_dir", "")).expanduser().resolve()) == summary_root:
                        matching_training_run = run
                        train_csv = run.get("train_csv")
                        break

        governance_assessment = {}
        recommended_status = None
        resolved_applicability_domain = dict(applicability_domain or {})
        summary_payload: Dict[str, Any] = {}
        summary_path: Optional[Path] = None
        if matching_training_run:
            output_dir = Path(matching_training_run.get("output_dir", "")).expanduser()
            summary_path = _find_first_existing_path(
                [
                    output_dir / "cs_copilot_training_summary.json",
                    output_dir / "tabicl_training_summary.json",
                ]
            )
        if summary_path is None:
            summary_path = _find_training_summary_from_model_run(inferred_run_dir)
        if summary_path is None:
            explicit_summary_path = (current.training_data_summary or {}).get(
                "training_summary_path"
            ) or (current.inference_profile or {}).get("feature_columns_source")
            if explicit_summary_path:
                candidate_summary_path = Path(str(explicit_summary_path)).expanduser()
                if candidate_summary_path.exists():
                    summary_path = candidate_summary_path
        if summary_path is not None and summary_path.exists():
            try:
                summary_payload = json.loads(summary_path.read_text())
                summary_payload.setdefault("summary_path", str(summary_path))
                train_csv = train_csv or summary_payload.get("train_csv")
                resolved_applicability_domain = {
                    **(summary_payload.get("applicability_domain") or {}),
                    **resolved_applicability_domain,
                }
                governance_assessment = (summary_payload.get("validation_assessment") or {}).get(
                    "governance"
                ) or {}
                recommended_status = governance_assessment.get("recommended_status")
                if summary_payload.get("metrics_status") == "not_evaluated":
                    recommended_status = None
            except Exception:
                governance_assessment = {}
                resolved_applicability_domain = dict(applicability_domain or {})
                summary_payload = {}

        resolved_version = version or current.version or "1"
        trained_at_raw = summary_payload.get("trained_at")
        try:
            trained_at = coerce_project_timezone(trained_at_raw)
        except Exception:
            trained_at = project_now()
        train_csv_for_name = train_csv or summary_payload.get("train_csv") or current.source
        benchmark_dataset_name = (current.training_data_summary or {}).get("benchmark_dataset_name")
        benchmark_target_name = (current.training_data_summary or {}).get("benchmark_target_name")
        if benchmark_dataset_name and benchmark_target_name:
            endpoint_name = safe_slug(str(benchmark_dataset_name)) or "endpoint"
            dataset_name = safe_slug(str(benchmark_target_name)) or "dataset"
        else:
            endpoint_name, dataset_name = _extract_endpoint_and_dataset(
                train_csv_for_name, current.model_id
            )
        protocol_name = (
            (current.training_data_summary or {}).get("validation_protocol")
            or (training_data_summary or {}).get("validation_protocol")
            or summary_payload.get("validation_protocol")
            or (matching_training_run.get("validation_protocol") if matching_training_run else None)
            or "protocol"
        )
        representation_name = (
            (current.training_data_summary or {}).get("representation_name")
            or (current.inference_profile or {}).get("representation_name")
            or (current.selection_hints or {}).get("representation_name")
        )
        canonical_model_id = _canonical_model_id(
            endpoint=endpoint_name,
            dataset=dataset_name,
            protocol=str(protocol_name or "protocol"),
            backend=current.backend_name,
            representation=representation_name,
            version=str(resolved_version),
            trained_at=trained_at,
        )
        canonical_display_name = _canonical_display_name(
            endpoint=endpoint_name,
            dataset=dataset_name,
            protocol=str(protocol_name or "protocol"),
            backend=current.backend_name,
            representation=representation_name,
            version=str(resolved_version),
        )

        summary_curation = summary_payload.get("curation") or {}
        fallback_curation = latest_curation_artifacts(
            agent
        ) or discover_curation_artifacts_near_dataset(train_csv)
        merged_curation = {
            **(fallback_curation or {}),
            **(summary_curation or {}),
        }
        merged_curation["artifacts"] = {
            **((fallback_curation or {}).get("artifacts") or {}),
            **((summary_curation or {}).get("artifacts") or {}),
        }
        if not merged_curation.get("curated_dataset_path"):
            merged_curation["curated_dataset_path"] = (fallback_curation or {}).get(
                "curated_dataset_path"
            )

        legacy_applicability_domain = resolved_applicability_domain.get("legacy_similarity_ad") or {}
        source_artifacts = {
            "training_summary_path": (
                str(summary_path) if summary_path and summary_path.exists() else None
            ),
            "config_path": summary_payload.get("config_path"),
            "splits_path": summary_payload.get("splits_path"),
            "validation_predictions_path": summary_payload.get("validation_predictions_path"),
            "test_predictions_path": summary_payload.get("test_predictions_path"),
            "chemprop_training_input_csv": summary_payload.get("chemprop_training_input_csv"),
            "chemprop_splits_file": summary_payload.get("chemprop_splits_file"),
            "chemprop_input_manifest_path": summary_payload.get("chemprop_input_manifest_path"),
            "split_results": summary_payload.get("split_results") or [],
            "reference_store_path": resolved_applicability_domain.get("reference_store_path")
            or legacy_applicability_domain.get("reference_store_path"),
            "reference_manifest_path": resolved_applicability_domain.get("reference_manifest_path")
            or legacy_applicability_domain.get("reference_manifest_path"),
            "applicability_domain_path": resolved_applicability_domain.get(
                "applicability_domain_path"
            )
            or legacy_applicability_domain.get("applicability_domain_path"),
            "ad_manifest_path": resolved_applicability_domain.get("manifest_path"),
            "ad_bounds_path": resolved_applicability_domain.get("bounds_path")
            or (
                (resolved_applicability_domain.get("methods") or {})
                .get("bounding_box", {})
                .get("bounds_path")
            ),
            "ad_isolation_forest_model_path": resolved_applicability_domain.get(
                "isolation_forest_model_path"
            )
            or (
                (resolved_applicability_domain.get("methods") or {})
                .get("isolation_forest", {})
                .get("model_path")
            ),
            "ad_scores_train_path": resolved_applicability_domain.get("scores_train_path"),
            "ad_scores_validation_path": resolved_applicability_domain.get(
                "scores_validation_path"
            ),
            "ad_scores_test_path": resolved_applicability_domain.get("scores_test_path"),
            "plot_artifacts": summary_payload.get("plot_artifacts") or {},
            "activity_cliffs": summary_payload.get("activity_cliffs") or {},
            "curation": merged_curation,
            "feature_preparation": summary_payload.get("feature_preparation") or {},
        }
        hydrated_inference_profile = _hydrate_inference_profile_from_summary(
            current_profile=current.inference_profile,
            override_profile=inference_profile,
            summary_payload=summary_payload,
        )
        materialization_record = replace(
            current,
            inference_profile=hydrated_inference_profile,
            applicability_domain=resolved_applicability_domain,
        )

        materialized = self._materialize_internal_model(
            record=materialization_record,
            train_csv=train_csv,
            model_id=canonical_model_id,
            source_artifacts=source_artifacts,
        )
        resolved_model_path = materialized.get("model_path", current.model_path)
        resolved_metadata_path = materialized.get("metadata_path", current.metadata_path)
        requested_status = status or current.status
        if summary_payload.get("metrics_status") == "not_evaluated":
            requested_status = "workflow_demo"
        resolved_status = requested_status
        status_reason = None
        if recommended_status and recommended_status != requested_status:
            resolved_status = recommended_status
            status_reason = (
                f"Requested `{requested_status}` was adjusted to governance-recommended "
                f"`{recommended_status}` based on the final validation gates."
            )
        activity_summary_payload = summary_payload.get("activity_cliffs") or {}
        resolved_description = _sanitize_activity_cliff_description(
            description or current.description,
            activity_summary_payload,
        )
        persisted_record = PredictionModelRecord(
            model_id=canonical_model_id,
            backend_name=current.backend_name,
            model_path=resolved_model_path,
            metadata_path=resolved_metadata_path,
            task=current.task,
            display_name=display_name or canonical_display_name,
            description=resolved_description,
            tags=tags or current.tags,
            version=resolved_version,
            status=resolved_status,
            owner=owner or current.owner,
            source=source or current.source,
            domain_summary=domain_summary or current.domain_summary,
            strengths=strengths or current.strengths,
            limitations=limitations or current.limitations,
            recommended_for=recommended_for or current.recommended_for,
            not_recommended_for=not_recommended_for or current.not_recommended_for,
            known_metrics=(
                {}
                if known_metrics is None
                and summary_payload.get("metrics_status") == "not_evaluated"
                else known_metrics or current.known_metrics
            ),
            training_data_summary={
                **current.training_data_summary,
                **(training_data_summary or {}),
                "trained_at": trained_at.isoformat(),
                "trained_date": trained_at.strftime("%d/%m/%Y"),
                "trained_time": trained_at.strftime("%H:%M:%S"),
                "endpoint_name": endpoint_name,
                "dataset_name": dataset_name,
                "validation_protocol": str(protocol_name or "protocol"),
                "metrics_status": (
                    summary_payload.get("metrics_status")
                    or (training_data_summary or {}).get("metrics_status")
                    or current.training_data_summary.get("metrics_status")
                    or ("not_evaluated" if summary_payload.get("evaluation_required") else None)
                ),
                "evaluation_required": bool(
                    summary_payload.get("evaluation_required")
                    or (training_data_summary or {}).get("evaluation_required")
                    or current.training_data_summary.get("evaluation_required")
                ),
                "external_evaluations": list(
                    current.training_data_summary.get("external_evaluations")
                    or (training_data_summary or {}).get("external_evaluations")
                    or []
                ),
                "seed_policy": summary_payload.get("seed_policy")
                or current.training_data_summary.get("seed_policy"),
                "seed_policy_report": seed_policy_reporting_text(
                    summary_payload.get("seed_policy")
                    or current.training_data_summary.get("seed_policy")
                ),
                "feature_preparation": (
                    summary_payload.get("feature_preparation")
                    or current.training_data_summary.get("feature_preparation")
                    or {}
                ),
                "feature_preparation_durations": (
                    summary_payload.get("feature_preparation_durations")
                    or (
                        summary_payload.get("feature_preparation")
                        or current.training_data_summary.get("feature_preparation")
                        or {}
                    ).get("durations")
                    or {}
                ),
                "replicate_policy": (
                    summary_payload.get("replicate_policy")
                    or current.training_data_summary.get("replicate_policy")
                    or {}
                ),
                "activity_cliffs": (
                    {
                        "enabled": bool(activity_summary_payload.get("enabled")),
                        "mode": activity_summary_payload.get("mode"),
                        "index_name": activity_summary_payload.get("index_name"),
                        "flagged_count": activity_summary_payload.get("flagged_count"),
                        "priority_counts": activity_summary_payload.get("priority_counts"),
                        "recommended_variant": activity_summary_payload.get("recommended_variant"),
                        "reporting_handoff": activity_summary_payload.get("reporting_handoff"),
                    }
                    if activity_summary_payload
                    else {}
                ),
            },
            inference_profile=hydrated_inference_profile,
            selection_hints={
                **current.selection_hints,
                **(selection_hints or {}),
                "governance_recommended_status": recommended_status,
            },
            applicability_domain={
                **current.applicability_domain,
                **resolved_applicability_domain,
            },
        )

        self.catalog.upsert_model(persisted_record)
        self.catalog = PredictionModelCatalog.load(str(self.catalog.source_path))
        self._sync_internal_metadata(
            metadata_path=resolved_metadata_path,
            record=persisted_record,
            governance_assessment=governance_assessment,
            status_reason=status_reason,
        )

        prediction_state["registered"].pop(model_id, None)
        prediction_state["registered"][canonical_model_id] = persisted_record.as_dict()

        response_payload = {
            "catalog_path": str(self.catalog.source_path),
            "model_id": persisted_record.model_id,
            "status": persisted_record.status,
            "persisted": True,
            "materialized": bool(materialized.get("materialized")),
            "model_root": materialized.get("model_root"),
            "model_path": persisted_record.model_path,
            "metadata_path": persisted_record.metadata_path,
            "governance_assessment": governance_assessment,
            "status_reason": status_reason,
            "record": _compact_persisted_record_for_response(persisted_record),
        }
        response_payload["reporting_handoff"] = build_registry_reporting_handoff(
            requested_status=requested_status,
            final_status=persisted_record.status,
            status_reason=status_reason,
            metrics_status=persisted_record.training_data_summary.get("metrics_status"),
            payload=response_payload,
        )
        return response_payload

    def list_registered_models(self, agent: Optional[Agent] = None) -> List[Dict[str, Any]]:
        """List models registered in the current session."""
        if agent is None:
            return []
        prediction_state = get_prediction_state(agent)
        return [
            _compact_model_payload_for_response(record)
            for record in prediction_state["registered"].values()
        ]

    def summarize_model(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Return the stored summary for a registered model."""
        if agent is None:
            raise ValueError("Agent is required to summarize a model")
        prediction_state = get_prediction_state(agent)
        if model_id not in prediction_state["registered"]:
            return {
                "found": False,
                "model_id": model_id,
                "error": f"Unknown model_id: {model_id}",
                "registered_model_ids": list(prediction_state["registered"].keys()),
                "usage_hint": (
                    "If registration just failed, retry `register_model` with the real backend "
                    "model artifact path rather than a bundle/archive path. If the model was "
                    "persisted to the catalog, use `summarize_catalog_model` with the returned "
                    "canonical catalog model_id."
                ),
            }
        return self.annotate_record(self.resolve_record(model_id, agent))

    def resolve_record(self, model_id: str, agent: Agent) -> PredictionModelRecord:
        prediction_state = get_prediction_state(agent)
        payload = prediction_state["registered"].get(model_id)
        if payload is not None:
            record = PredictionModelRecord.from_dict(payload)
            if record.metadata_path:
                return record
            try:
                self.catalog.refresh_from_internal_store(persist=True)
                catalog_record = self.catalog.get_model(model_id)
            except ValueError:
                return record
            prediction_state["registered"][model_id] = catalog_record.as_dict()
            return catalog_record

        self.catalog.refresh_from_internal_store(persist=True)
        try:
            record = self.catalog.get_model(model_id)
        except ValueError as exc:
            raise ValueError(f"Unknown model_id: {model_id}") from exc
        prediction_state["registered"][model_id] = record.as_dict()
        return record
