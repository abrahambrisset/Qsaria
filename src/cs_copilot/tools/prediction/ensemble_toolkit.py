#!/usr/bin/env python
# coding: utf-8
"""Toolkit for post-hoc, catalog-driven QSAR ensembles."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .backend import PredictionModelRecord, PredictionTaskSpec
from .backend_capabilities import backend_supports_component_orchestration
from .catalog import DEFAULT_INTERNAL_MODEL_ROOT, PredictionModelCatalog
from .chemprop_backend import ChempropBackend
from .ensemble_backend import EnsembleBackend
from .lightgbm_backend import LightGBMBackend
from .qsar_training_policy import project_now, safe_slug
from .tabicl_backend import TabICLBackend
from .training_orchestration import (
    compute_classification_metrics,
    is_classification_task,
    is_multiclass_task,
)

ABLATION_REPRESENTATIONS = {"morgan_only", "rdkit_only", "rdkit_basic_only", "rdkit_all_only"}
PROMOTED_STATUSES = {"production", "validated", "robust_validated"}
EVALUATION_KINDS = {
    "external_dataset",
    "internal_holdout_reuse",
    "catalog_common_holdout",
    "training_like_or_potentially_leaky",
}


def _is_ensemble_backend(backend_name: str) -> bool:
    try:
        return backend_supports_component_orchestration(backend_name)
    except KeyError as exc:
        raise ValueError(f"Backend `{backend_name}` has no registered capabilities.") from exc


def _state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("prediction_models", {})
    state.setdefault("registered", {})
    state.setdefault("last_prediction", {})
    state.setdefault("prediction_history", [])
    state.setdefault("catalog_recommendations", {})
    state.setdefault("training_runs", [])
    state.setdefault("ensembles", {})
    return state


def _load_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    candidate = Path(path).expanduser()
    if not candidate.exists():
        return {}
    try:
        payload = json.loads(candidate.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _resolve_catalog_model_id(catalog: PredictionModelCatalog, raw_model_id: str) -> str:
    requested = str(raw_model_id or "").strip()
    try:
        catalog.get_model(requested)
        return requested
    except ValueError:
        pass

    tokens = [
        match.group(0)
        for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", requested)
        if len(match.group(0)) > 8
    ]
    for token in sorted(tokens, key=len, reverse=True):
        try:
            catalog.get_model(token)
            return token
        except ValueError:
            continue
    raise ValueError(f"Unknown catalog model_id: {raw_model_id}")


def _flatten(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_flatten(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten(item) for item in value)
    return str(value)


def _target_matches(record: PredictionModelRecord, target: str) -> bool:
    normalized = target.strip().lower()
    explicit = {str(item).strip().lower() for item in record.task.target_columns}
    if normalized in explicit:
        return True
    haystack = _flatten(
        [
            record.model_id,
            record.display_name,
            record.description,
            record.tags,
            record.training_data_summary,
            record.selection_hints,
            record.known_metrics,
        ]
    ).lower()
    return normalized in haystack


def _representation(record: PredictionModelRecord) -> str:
    for source in (
        record.inference_profile,
        record.selection_hints,
        record.training_data_summary,
        record.tags,
    ):
        value = (source or {}).get("representation_name") or (source or {}).get("representation")
        if value:
            return str(value)
    return "unknown"


def _protocol(record: PredictionModelRecord) -> str:
    for source in (record.training_data_summary, record.selection_hints, record.tags):
        value = (
            (source or {}).get("validation_protocol")
            or (source or {}).get("benchmark_mode")
            or (source or {}).get("protocol")
        )
        if value:
            return str(value)
    return "unknown"


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _coerce_list(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [item.strip() for item in stripped.split(",") if item.strip()]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, str):
            return [parsed]
    return [str(value)]


def _nested_metric(payload: Dict[str, Any], *keys: str) -> Optional[float]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return _safe_float(current)


def _metric_summary(record: PredictionModelRecord) -> Dict[str, Any]:
    metrics = record.known_metrics or {}
    random_balanced_accuracy = (
        _nested_metric(metrics, "random", "balanced_accuracy_mean")
        or _nested_metric(metrics, "random", "balanced_accuracy")
        or _nested_metric(metrics, "random_split", "balanced_accuracy")
    )
    scaffold_balanced_accuracy = (
        _nested_metric(metrics, "scaffold", "balanced_accuracy_mean")
        or _nested_metric(metrics, "scaffold", "balanced_accuracy")
        or _nested_metric(metrics, "scaffold_split", "balanced_accuracy")
    )
    fallback_balanced_accuracy = (
        _safe_float(metrics.get("balanced_accuracy")) if isinstance(metrics, dict) else None
    )
    scaffold_r2 = (
        _nested_metric(metrics, "scaffold", "r2_mean")
        or _nested_metric(metrics, "scaffold", "r2")
        or _nested_metric(metrics, "scaffold_split", "r2")
    )
    random_r2 = (
        _nested_metric(metrics, "random", "r2_mean")
        or _nested_metric(metrics, "random", "r2")
        or _nested_metric(metrics, "random_split", "r2")
    )
    random_std = _nested_metric(metrics, "random", "r2_std")
    hardest_r2 = None
    hardest_name = None
    if isinstance(metrics, dict):
        for key in ("hardest_split", "hardest_split_name"):
            if metrics.get(key):
                hardest_name = str(metrics.get(key))
        if hardest_name:
            hardest_r2 = _nested_metric(metrics, hardest_name, "r2_mean") or _nested_metric(
                metrics, hardest_name, "r2"
            )
    fallback_r2 = _safe_float(metrics.get("r2")) if isinstance(metrics, dict) else None
    return {
        "scaffold_r2": scaffold_r2,
        "random_r2": random_r2,
        "random_r2_std": random_std,
        "hardest_split": hardest_name,
        "hardest_split_r2": hardest_r2 or scaffold_r2 or fallback_r2,
        "fallback_r2": fallback_r2,
        "random_balanced_accuracy": random_balanced_accuracy,
        "scaffold_balanced_accuracy": scaffold_balanced_accuracy,
        "hardest_split_balanced_accuracy": scaffold_balanced_accuracy or fallback_balanced_accuracy,
        "fallback_balanced_accuracy": fallback_balanced_accuracy,
        "roc_auc": _safe_float(metrics.get("roc_auc")) if isinstance(metrics, dict) else None,
        "f1_macro": _safe_float(metrics.get("f1_macro")) if isinstance(metrics, dict) else None,
    }


def _evidence_tier(record: PredictionModelRecord, metrics: Dict[str, Any]) -> str:
    text = f"{_protocol(record)} {_flatten(record.known_metrics)} {_flatten(record.training_data_summary)}".lower()
    if "external" in text or "challenging" in text:
        return "A"
    if "robust" in text and metrics.get("random_r2_std") is not None:
        return "A"
    if (
        metrics.get("scaffold_r2") is not None
        or metrics.get("hardest_split_r2") is not None
        or metrics.get("scaffold_balanced_accuracy") is not None
        or metrics.get("hardest_split_balanced_accuracy") is not None
    ):
        return "B"
    if (
        metrics.get("random_r2") is not None
        or metrics.get("fallback_r2") is not None
        or metrics.get("random_balanced_accuracy") is not None
        or metrics.get("fallback_balanced_accuracy") is not None
    ):
        return "C"
    return "D"


def _sort_score(evidence: Dict[str, Any]) -> tuple[float, float, float, float]:
    tier_weight = {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}.get(evidence.get("evidence_tier"), 0.0)
    metric = evidence.get("metrics", {}).get("hardest_split_r2")
    if metric is None:
        metric = evidence.get("metrics", {}).get("hardest_split_balanced_accuracy")
    if metric is None:
        metric = evidence.get("metrics", {}).get("scaffold_r2")
    if metric is None:
        metric = evidence.get("metrics", {}).get("scaffold_balanced_accuracy")
    if metric is None:
        metric = evidence.get("metrics", {}).get("random_r2")
    if metric is None:
        metric = evidence.get("metrics", {}).get("random_balanced_accuracy")
    if metric is None:
        metric = evidence.get("metrics", {}).get("fallback_r2")
    if metric is None:
        metric = evidence.get("metrics", {}).get("fallback_balanced_accuracy")
    status_weight = {
        "production": 4.0,
        "robust_validated": 3.5,
        "validated": 3.0,
        "experimental": 1.5,
        "workflow_demo": 1.0,
    }.get(str(evidence.get("status") or "").lower(), 0.0)
    ablation_penalty = -0.5 if evidence.get("is_ablation") else 0.0
    return (tier_weight, metric if metric is not None else -999.0, status_weight, ablation_penalty)


def _metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    truth = pd.to_numeric(y_true, errors="coerce")
    pred = pd.to_numeric(y_pred, errors="coerce")
    mask = truth.notna() & pred.notna()
    truth = truth[mask].astype(float)
    pred = pred[mask].astype(float)
    residual = pred - truth
    mse = float((residual**2).mean())
    mae = float(residual.abs().mean())
    rmse = float(math.sqrt(mse))
    denom = float(((truth - truth.mean()) ** 2).sum())
    r2 = float(1.0 - float((residual**2).sum()) / denom) if denom > 0 else float("nan")
    return {"n": int(mask.sum()), "r2": r2, "rmse": rmse, "mae": mae, "mse": mse}


def _task_compatible(record_task_type: str, requested_task_type: str) -> bool:
    if record_task_type == requested_task_type:
        return True
    return is_classification_task(record_task_type) and is_classification_task(requested_task_type)


class EnsembleToolkit(Toolkit):
    """Create and evaluate post-hoc catalog ensembles."""

    def __init__(self, catalog: Optional[PredictionModelCatalog] = None):
        super().__init__("ensemble_prediction")
        self.catalog = catalog or PredictionModelCatalog.load()
        self.backends = {
            "chemprop": ChempropBackend(),
            "lightgbm": LightGBMBackend(),
            "tabicl": TabICLBackend(),
        }
        self.ensemble_backend = EnsembleBackend(backends=self.backends)
        self.register(self.inspect_ensemble_candidates)
        self.register(self.create_ensemble_from_catalog)
        self.register(self.evaluate_ensemble_on_dataset)
        self.register(self.summarize_ensemble)

    def _backend_available(self, backend_name: str) -> bool:
        backend = self.backends.get(backend_name)
        return bool(backend and backend.is_available())

    def _candidate_evidence(
        self, record: PredictionModelRecord, target: str, task_type: str
    ) -> Dict[str, Any]:
        representation = _representation(record)
        representation_slug = safe_slug(representation) or "unknown"
        metrics = _metric_summary(record)
        metadata = _load_json(record.metadata_path)
        activity = (
            metadata.get("activity_cliffs")
            or record.training_data_summary.get("activity_cliffs")
            or {}
        )
        is_ablation = representation_slug in ABLATION_REPRESENTATIONS or any(
            token in safe_slug(record.model_id) for token in ABLATION_REPRESENTATIONS
        )
        compatible = True
        reasons: List[str] = []
        warnings: List[str] = []
        if not _task_compatible(record.task.task_type, task_type):
            compatible = False
            warnings.append(f"Task type mismatch: {record.task.task_type}.")
        if not _target_matches(record, target):
            compatible = False
            warnings.append(f"Target `{target}` not found in task or metadata.")
        if str(record.status).lower() == "deprecated":
            compatible = False
            warnings.append("Deprecated model.")
        path_exists = Path(record.model_path).expanduser().exists()
        if not path_exists:
            compatible = False
            warnings.append("Model artifact path is missing.")
        if not self._backend_available(record.backend_name):
            compatible = False
            warnings.append(f"Backend `{record.backend_name}` is not available.")
        if is_ablation:
            warnings.append(
                "Ablation representation; include only with explicit scientific justification."
            )
        if compatible:
            reasons.append("Compatible target, task, backend and model path.")
        if (
            metrics.get("hardest_split_r2") is not None
            or metrics.get("hardest_split_balanced_accuracy") is not None
        ):
            reasons.append("Has hardest/scaffold-style validation evidence.")
        elif (
            metrics.get("random_r2") is not None
            or metrics.get("random_balanced_accuracy") is not None
        ):
            reasons.append("Has random-split validation evidence only.")
        else:
            reasons.append("No direct validation metric was extracted; usable only with caution.")
        evidence = {
            "model_id": record.model_id,
            "display_name": record.display_name or record.model_id,
            "backend_name": record.backend_name,
            "representation_name": representation,
            "representation_key": f"{record.backend_name}:{representation_slug}",
            "task_type": record.task.task_type,
            "target_columns": list(record.task.target_columns),
            "status": record.status,
            "model_path": record.model_path,
            "metadata_path": record.metadata_path,
            "training_protocol": _protocol(record),
            "metrics": metrics,
            "evidence_tier": _evidence_tier(record, metrics),
            "activity_cliffs": {
                "enabled": bool(activity.get("enabled")),
                "recommended_variant": activity.get("recommended_variant"),
            },
            "is_ablation": is_ablation,
            "compatible": compatible,
            "reasons": reasons,
            "warnings": warnings,
            "selection_decision": "not_selected",
            "selection_reason": "",
        }
        evidence["selection_score"] = list(_sort_score(evidence))
        return evidence

    def _component_payload(
        self, record: PredictionModelRecord, evidence: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "model_id": record.model_id,
            "component_slug": safe_slug(record.model_id) or record.model_id,
            "display_name": record.display_name or record.model_id,
            "backend_name": record.backend_name,
            "model_path": record.model_path,
            "metadata_path": record.metadata_path,
            "status": record.status,
            "version": record.version,
            "description": record.description,
            "task": (
                record.task.as_dict()
                if hasattr(record.task, "as_dict")
                else {
                    "task_type": record.task.task_type,
                    "smiles_columns": list(record.task.smiles_columns),
                    "target_columns": list(record.task.target_columns),
                }
            ),
            "known_metrics": record.known_metrics,
            "training_data_summary": record.training_data_summary,
            "inference_profile": record.inference_profile,
            "selection_hints": record.selection_hints,
            "applicability_domain": record.applicability_domain,
            "selection_evidence": {
                "evidence_tier": evidence.get("evidence_tier"),
                "selection_reason": evidence.get("selection_reason"),
                "is_ablation": evidence.get("is_ablation"),
                "metrics": evidence.get("metrics"),
            },
        }

    def inspect_ensemble_candidates(
        self,
        target: str,
        task_type: str = "regression",
        include_incompatible: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Inspect catalog models and derive ensemble selection evidence."""
        self.catalog.refresh_from_internal_store(persist=True)
        evidence = [
            self._candidate_evidence(record, target, task_type)
            for record in self.catalog.list_models()
        ]
        if not include_incompatible:
            evidence = [item for item in evidence if item["compatible"]]
        evidence = sorted(evidence, key=_sort_score, reverse=True)
        result = {
            "target": target,
            "task_type": task_type,
            "models_inspected": len(self.catalog.list_models()),
            "compatible_count": sum(1 for item in evidence if item["compatible"]),
            "candidates": evidence,
        }
        if agent is not None:
            _state(agent)["ensemble_candidates"] = result
        return result

    def _select_components(
        self,
        *,
        target: str,
        task_type: str,
        model_ids: Optional[List[str]],
        max_components: int,
    ) -> tuple[List[PredictionModelRecord], List[Dict[str, Any]], List[Dict[str, Any]]]:
        records_by_id = {record.model_id: record for record in self.catalog.list_models()}
        all_evidence = [
            self._candidate_evidence(record, target, task_type)
            for record in self.catalog.list_models()
        ]
        evidence_by_id = {item["model_id"]: item for item in all_evidence}
        selected: List[PredictionModelRecord] = []
        if model_ids:
            missing = [model_id for model_id in model_ids if model_id not in records_by_id]
            if missing:
                raise ValueError(f"Unknown catalog model ids: {missing}")
            for model_id in model_ids:
                evidence = evidence_by_id[model_id]
                if not evidence["compatible"]:
                    raise ValueError(
                        f"Model `{model_id}` is not compatible: {evidence['warnings']}"
                    )
                selected.append(records_by_id[model_id])
        else:
            compatible = [item for item in all_evidence if item["compatible"]]
            compatible = sorted(compatible, key=_sort_score, reverse=True)
            seen_keys: set[str] = set()
            for evidence in compatible:
                if len(selected) >= max_components:
                    break
                key = str(evidence["representation_key"])
                if key in seen_keys:
                    evidence["selection_decision"] = "excluded"
                    evidence["selection_reason"] = (
                        "Redundant backend/representation combination for V1 default selection."
                    )
                    continue
                selected.append(records_by_id[evidence["model_id"]])
                seen_keys.add(key)
        selected_ids = {record.model_id for record in selected}
        for evidence in all_evidence:
            if evidence["model_id"] in selected_ids:
                evidence["selection_decision"] = "included"
                reason = (
                    "Explicitly requested by user."
                    if model_ids
                    else "Selected as a high-evidence, non-redundant component."
                )
                if evidence.get("is_ablation"):
                    reason += " Warning: ablation representation included."
                evidence["selection_reason"] = reason
            elif evidence["selection_decision"] == "not_selected":
                evidence["selection_decision"] = (
                    "excluded" if evidence["compatible"] else "incompatible"
                )
                evidence["selection_reason"] = (
                    "Lower-ranked or redundant compatible candidate."
                    if evidence["compatible"]
                    else "; ".join(evidence["warnings"])
                )
        selected_evidence = [evidence_by_id[record.model_id] for record in selected]
        return selected, selected_evidence, all_evidence

    def create_ensemble_from_catalog(
        self,
        target: str,
        task_type: str = "regression",
        model_ids: Optional[List[str]] = None,
        max_components: int = 5,
        aggregation_strategy: str = "median",
        display_name: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Create and persist a post-hoc ensemble from compatible catalog models."""
        task_is_classification = is_classification_task(task_type)
        if task_is_classification and is_multiclass_task(task_type):
            raise ValueError(
                "Ensemble multiclass classification is not enabled in this QSARIA version."
            )
        if task_is_classification and aggregation_strategy == "median":
            aggregation_strategy = "majority_vote"
        if task_is_classification:
            if aggregation_strategy != "majority_vote":
                raise ValueError("Classification ensembles support only majority_vote aggregation.")
        elif aggregation_strategy != "median":
            raise ValueError("Regression ensembles support only median aggregation.")
        self.catalog.refresh_from_internal_store(persist=True)
        model_ids = _coerce_list(model_ids)
        selected, selected_evidence, all_evidence = self._select_components(
            target=target,
            task_type=task_type,
            model_ids=model_ids,
            max_components=max_components,
        )
        if not selected:
            raise ValueError("No compatible components were selected for the ensemble.")

        now = project_now()
        target_slug = safe_slug(target) or "target"
        model_id = (
            f"{target_slug}_catalog_consensus_ensemble_v1_0_{now.strftime('%d%m%Y_%H%M%S_%f')}"
        )
        model_root = DEFAULT_INTERNAL_MODEL_ROOT / model_id
        model_dir = model_root / "model"
        artifacts_dir = model_root / "artifacts"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        components = [
            self._component_payload(record, evidence)
            for record, evidence in zip(selected, selected_evidence, strict=False)
        ]
        class_labels: List[Any] = []
        if task_is_classification:
            for record in selected:
                labels = list((record.inference_profile or {}).get("class_labels") or [])
                if labels and not class_labels:
                    class_labels = labels
                elif labels and class_labels and list(labels) != class_labels:
                    raise ValueError(
                        "Selected classification components do not share the same class labels."
                    )
            if len(class_labels) > 2:
                raise ValueError(
                    "Ensemble multiclass classification is not enabled in this QSARIA version."
                )
        ensemble_payload = {
            "schema_version": 1,
            "ensemble_kind": (
                "catalog_consensus_classification"
                if task_is_classification
                else "catalog_consensus_regression"
            ),
            "target": target,
            "task_type": task_type,
            "aggregation_strategy": aggregation_strategy,
            "uncertainty_strategy": (
                "vote_fraction" if task_is_classification else "component_disagreement_std"
            ),
            "class_labels": class_labels,
            "created_from": "catalog",
            "created_at": now.isoformat(),
            "selection_policy": {
                "max_components": max_components,
                "component_selection": (
                    "top_diverse_by_evidence" if not model_ids else "explicit_model_ids"
                ),
                "status_policy": "workflow_demo_until_ensemble_level_evaluation",
            },
            "components": components,
            "evaluations": [],
        }
        ensemble_path = model_dir / "ensemble.json"
        _write_json(ensemble_path, ensemble_payload)

        evidence_payload = {
            "schema_version": 1,
            "target": target,
            "task_type": task_type,
            "models_inspected": len(all_evidence),
            "compatible_count": sum(1 for item in all_evidence if item["compatible"]),
            "included_components": [item["model_id"] for item in selected_evidence],
            "candidates": all_evidence,
        }
        evidence_path = artifacts_dir / "selection_evidence.json"
        _write_json(evidence_path, evidence_payload)
        report_path = artifacts_dir / "selection_report.md"
        report_path.write_text(self._selection_report(evidence_payload, ensemble_payload))

        record = PredictionModelRecord(
            model_id=model_id,
            backend_name="ensemble",
            model_path=str(ensemble_path.resolve()),
            metadata_path=str((model_root / "metadata.json").resolve()),
            display_name=display_name or f"{target} catalog consensus ensemble",
            description="Post-hoc QSAR consensus ensemble created from persisted catalog models.",
            version="1.0",
            status="workflow_demo",
            owner="chemspacecopilot",
            source="catalog",
            known_metrics={},
            training_data_summary={
                "created_from": "catalog",
                "target": target,
                "component_count": len(components),
                "has_ensemble_level_evaluation": False,
            },
            inference_profile={
                "aggregation_strategy": aggregation_strategy,
                "uncertainty_strategy": (
                    "vote_fraction" if task_is_classification else "component_disagreement_std"
                ),
                "class_labels": class_labels,
                "class_count": len(class_labels) if class_labels else None,
                "positive_class_label": class_labels[1] if len(class_labels) == 2 else None,
            },
            selection_hints={
                "selection_evidence_path": str(evidence_path.resolve()),
                "selection_report_path": str(report_path.resolve()),
            },
            task=PredictionTaskSpec(
                task_type=task_type,
                smiles_columns=selected[0].task.smiles_columns or ["smiles"],
                target_columns=[target],
            ),
        )
        metadata = record.as_dict()
        metadata["artifacts"] = {
            "model_path": "model/ensemble.json",
            "selection_evidence": "artifacts/selection_evidence.json",
            "selection_report": "artifacts/selection_report.md",
            "evaluations_dir": "evaluations",
        }
        _write_json(model_root / "metadata.json", metadata)
        self.catalog.upsert_model(record)
        if agent is not None:
            _state(agent)["registered"][model_id] = record.as_dict()
            _state(agent)["ensembles"][model_id] = {
                "model_root": str(model_root.resolve()),
                "selection_evidence_path": str(evidence_path.resolve()),
            }
        return {
            "created": True,
            "catalog_persisted": True,
            "model_id": model_id,
            "backend_name": "ensemble",
            "status": "workflow_demo",
            "model_root": str(model_root.resolve()),
            "model_path": str(ensemble_path.resolve()),
            "metadata_path": str((model_root / "metadata.json").resolve()),
            "selection_evidence_path": str(evidence_path.resolve()),
            "selection_report_path": str(report_path.resolve()),
            "component_count": len(components),
            "components": [
                {"model_id": item["model_id"], "backend_name": item["backend_name"]}
                for item in components
            ],
            "known_metrics": {},
        }

    def _selection_report(
        self, evidence_payload: Dict[str, Any], ensemble_payload: Dict[str, Any]
    ) -> str:
        included = set(evidence_payload.get("included_components") or [])
        lines = [
            f"# Rapport Ensemble — {ensemble_payload.get('target')}",
            "",
            "## Objectif",
            "Creation post-hoc d'un ensemble QSAR depuis les modeles persistés du catalogue.",
            "",
            "## Criteres de selection",
            "- Compatibilite cible/tache/backend/chemin modele.",
            "- Interpretation des metriques selon leur niveau de preuve.",
            "- Preference pour des composants performants et non redondants.",
            "- Les metriques propres de l'ensemble restent absentes tant qu'il n'est pas evalue.",
            "",
            "## Composants retenus",
        ]
        for item in evidence_payload.get("candidates") or []:
            if item["model_id"] in included:
                lines.append(
                    f"- {item['model_id']} ({item['backend_name']}, {item['representation_name']}) "
                    f"tier={item['evidence_tier']} status={item['status']} - {item['selection_reason']}"
                )
        lines.extend(["", "## Candidats non retenus importants"])
        for item in (evidence_payload.get("candidates") or [])[:12]:
            if item["model_id"] not in included:
                lines.append(
                    f"- {item['model_id']} ({item['selection_decision']}): {item['selection_reason']}"
                )
        lines.extend(
            [
                "",
                "## Statut",
                "Statut initial: workflow_demo. Aucune metrique propre a l'ensemble n'est revendiquee avant evaluation explicite.",
            ]
        )
        return "\n".join(lines) + "\n"

    def summarize_ensemble(self, model_id: str, agent: Optional[Agent] = None) -> Dict[str, Any]:
        """Summarize a persisted ensemble model."""
        self.catalog.refresh_from_internal_store(persist=True)
        model_id = _resolve_catalog_model_id(self.catalog, model_id)
        record = self.catalog.get_model(model_id)
        if not _is_ensemble_backend(record.backend_name):
            raise ValueError(f"Model `{model_id}` is not an ensemble.")
        payload = _load_json(record.model_path)
        evidence_path = (record.selection_hints or {}).get("selection_evidence_path")
        evidence_payload = _load_json(evidence_path)
        included = set(evidence_payload.get("included_components") or [])
        candidates = evidence_payload.get("candidates") or []
        non_retained = [
            {
                "model_id": item.get("model_id"),
                "backend_name": item.get("backend_name"),
                "selection_decision": item.get("selection_decision"),
                "selection_reason": item.get("selection_reason"),
            }
            for item in candidates
            if item.get("model_id") not in included
        ]
        return {
            "model_id": model_id,
            "status": record.status,
            "model_path": record.model_path,
            "metadata_path": record.metadata_path,
            "aggregation_strategy": payload.get("aggregation_strategy"),
            "official_prediction": "ensemble_prediction_median",
            "uncertainty_strategy": payload.get("uncertainty_strategy"),
            "created_from": payload.get("created_from"),
            "created_at": payload.get("created_at"),
            "selection_policy": payload.get("selection_policy") or {},
            "selection_evidence_path": evidence_path,
            "selection_report_path": (record.selection_hints or {}).get("selection_report_path"),
            "models_inspected": evidence_payload.get("models_inspected"),
            "compatible_count": evidence_payload.get("compatible_count"),
            "non_retained_count": len(non_retained),
            "non_retained_models": non_retained[:12],
            "component_count": len(payload.get("components") or []),
            "components": [
                {
                    "model_id": item.get("model_id"),
                    "backend_name": item.get("backend_name"),
                    "status": item.get("status"),
                    "representation_name": item.get("representation_name"),
                    "evidence_tier": item.get("evidence_tier"),
                    "selection_reason": item.get("selection_reason"),
                }
                for item in payload.get("components") or []
            ],
            "evaluations": payload.get("evaluations") or [],
        }

    def evaluate_ensemble_on_dataset(
        self,
        model_id: str,
        test_csv: str,
        target_column: str,
        smiles_column: str = "smiles",
        evaluation_kind: str = "external_dataset",
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Evaluate an ensemble and its components on a labelled dataset."""
        if evaluation_kind not in EVALUATION_KINDS:
            raise ValueError(f"Unsupported evaluation_kind: {evaluation_kind}")
        self.catalog.refresh_from_internal_store(persist=True)
        model_id = _resolve_catalog_model_id(self.catalog, model_id)
        record = self.catalog.get_model(model_id)
        if not _is_ensemble_backend(record.backend_name):
            raise ValueError(f"Model `{model_id}` is not an ensemble.")
        ensemble_payload = _load_json(record.model_path)
        model_root = Path(record.model_path).expanduser().parents[1]
        now = project_now()
        evaluation_id = f"{evaluation_kind}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        eval_dir = model_root / "evaluations" / evaluation_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        preds_path = eval_dir / "ensemble_predictions.csv"

        result = self.ensemble_backend.predict_from_csv(
            input_csv=test_csv,
            model_record=record,
            preds_path=str(preds_path),
        )
        source = pd.read_csv(Path(test_csv).expanduser())
        predictions = pd.read_csv(preds_path)
        if target_column not in source.columns:
            raise ValueError(f"Target column `{target_column}` is missing from evaluation dataset.")
        comparison = pd.concat(
            [source.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1
        )
        comparison.to_csv(preds_path, index=False)

        prediction_columns = [
            column for column in predictions.columns if column.startswith("prediction_")
        ]
        metrics_rows = []
        task_is_classification = is_classification_task(record.task.task_type)
        if task_is_classification:
            for column in prediction_columns:
                metrics_rows.append(
                    {
                        "model": column.removeprefix("prediction_"),
                        **compute_classification_metrics(
                            source[target_column], predictions[column]
                        ),
                    }
                )
            metrics_rows.append(
                {
                    "model": "ensemble_majority_vote",
                    **compute_classification_metrics(
                        source[target_column],
                        predictions["ensemble_prediction_class"],
                        positive_scores=predictions.get("positive_probability"),
                    ),
                }
            )
        else:
            for column in prediction_columns:
                metrics_rows.append(
                    {
                        "model": column.removeprefix("prediction_"),
                        **_metrics(source[target_column], predictions[column]),
                    }
                )
            metrics_rows.append(
                {
                    "model": "ensemble_median",
                    **_metrics(source[target_column], predictions["ensemble_prediction_median"]),
                }
            )
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = eval_dir / "metrics_by_component.csv"
        metrics_df.to_csv(metrics_path, index=False)

        if task_is_classification:
            residuals = pd.DataFrame(
                {
                    "y_true": source[target_column],
                    "ensemble_prediction": predictions["ensemble_prediction_class"],
                    "ensemble_correct": source[target_column].astype(str)
                    == predictions["ensemble_prediction_class"].astype(str),
                    "ensemble_vote_fraction": pd.to_numeric(
                        predictions["ensemble_vote_fraction"], errors="coerce"
                    ),
                }
            )
        else:
            residuals = pd.DataFrame(
                {
                    "y_true": pd.to_numeric(source[target_column], errors="coerce"),
                    "ensemble_prediction": pd.to_numeric(
                        predictions["ensemble_prediction_median"], errors="coerce"
                    ),
                    "ensemble_residual": pd.to_numeric(
                        predictions["ensemble_prediction_median"], errors="coerce"
                    )
                    - pd.to_numeric(source[target_column], errors="coerce"),
                    "ensemble_disagreement_std": pd.to_numeric(
                        predictions["ensemble_prediction_std"], errors="coerce"
                    ),
                }
            )
        residuals_path = eval_dir / "residuals.csv"
        residuals.to_csv(residuals_path, index=False)
        disagreement_path = eval_dir / "disagreement_analysis.csv"
        sort_column = (
            "ensemble_vote_fraction" if task_is_classification else "ensemble_disagreement_std"
        )
        residuals.sort_values(sort_column, ascending=task_is_classification).to_csv(
            disagreement_path, index=False
        )

        component_predictions_path = eval_dir / "component_predictions.csv"
        predictions[prediction_columns].to_csv(component_predictions_path, index=False)
        ensemble_metrics = metrics_rows[-1]
        summary = {
            "evaluation_id": evaluation_id,
            "evaluation_kind": evaluation_kind,
            "dataset_path": str(test_csv),
            "target_column": target_column,
            "smiles_column": smiles_column,
            "created_at": now.isoformat(),
            "n": int(len(source)),
            "ensemble_metrics": ensemble_metrics,
            "component_metrics": metrics_rows[:-1],
            "artifacts": {
                "evaluation_summary": "evaluation_summary.json",
                "component_predictions": "component_predictions.csv",
                "ensemble_predictions": "ensemble_predictions.csv",
                "metrics_by_component": "metrics_by_component.csv",
                "residuals": "residuals.csv",
                "disagreement_analysis": "disagreement_analysis.csv",
                "evaluation_report": "evaluation_report.md",
            },
            "notes": [
                "Metrics are ensemble-level metrics computed on this evaluation dataset.",
                "Component disagreement std is not calibrated uncertainty.",
            ],
        }
        summary_path = eval_dir / "evaluation_summary.json"
        _write_json(summary_path, summary)
        report_path = eval_dir / "evaluation_report.md"
        report_path.write_text(self._evaluation_report(model_id, summary, metrics_rows))

        ensemble_payload.setdefault("evaluations", []).append(
            {
                "evaluation_id": evaluation_id,
                "evaluation_kind": evaluation_kind,
                "dataset_path": str(test_csv),
                "target_column": target_column,
                "created_at": now.isoformat(),
                "n": int(len(source)),
                "ensemble_metrics": ensemble_metrics,
                "summary_path": str(summary_path.resolve()),
            }
        )
        _write_json(Path(record.model_path).expanduser(), ensemble_payload)
        metadata = _load_json(record.metadata_path)
        metadata.setdefault("training_data_summary", {})["has_ensemble_level_evaluation"] = True
        metadata.setdefault("known_metrics", {})[evaluation_id] = ensemble_metrics
        _write_json(Path(record.metadata_path).expanduser(), metadata)
        return {
            "evaluated": True,
            "model_id": model_id,
            "evaluation_id": evaluation_id,
            "evaluation_kind": evaluation_kind,
            "ensemble_metrics": ensemble_metrics,
            "metrics_by_component_path": str(metrics_path.resolve()),
            "ensemble_predictions_path": str(preds_path.resolve()),
            "evaluation_summary_path": str(summary_path.resolve()),
            "evaluation_report_path": str(report_path.resolve()),
            "component_prediction_paths": result.get("component_prediction_paths") or {},
        }

    def _evaluation_report(
        self, model_id: str, summary: Dict[str, Any], metrics_rows: List[Dict[str, Any]]
    ) -> str:
        lines = [
            f"# Evaluation Ensemble — {model_id}",
            "",
            f"Evaluation: {summary['evaluation_id']} ({summary['evaluation_kind']})",
            "",
        ]
        if metrics_rows and "balanced_accuracy" in metrics_rows[0]:
            lines.extend(
                [
                    "| Modele | Balanced accuracy | F1 macro | ROC-AUC | n |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in metrics_rows:
                roc_auc = row.get("roc_auc")
                roc_auc_text = f"{roc_auc:.4f}" if roc_auc is not None else "NA"
                lines.append(
                    f"| {row['model']} | {row.get('balanced_accuracy', 0):.4f} | "
                    f"{row.get('f1_macro', 0):.4f} | {roc_auc_text} | {row.get('n')} |"
                )
        else:
            lines.extend(
                [
                    "| Modele | R2 | RMSE | MAE | MSE | n |",
                    "|---|---:|---:|---:|---:|---:|",
                ]
            )
            for row in metrics_rows:
                lines.append(
                    f"| {row['model']} | {row['r2']:.4f} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['mse']:.4f} | {row['n']} |"
                )
        lines.extend(
            [
                "",
                "Les metriques de l'ensemble sont propres a ce dataset d'evaluation. "
                "Les performances historiques des composants ne sont pas substituees a cette evaluation.",
            ]
        )
        return "\n".join(lines) + "\n"
