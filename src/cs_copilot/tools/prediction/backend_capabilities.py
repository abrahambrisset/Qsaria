#!/usr/bin/env python
# coding: utf-8
"""Machine-readable capability contracts for QSAR prediction backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Tuple

from .tabular_representations import SUPPORTED_TABULAR_REPRESENTATION_NAMES


@dataclass(frozen=True)
class BackendCapabilities:
    """Static capabilities expected from a prediction backend."""

    backend_name: str
    can_train: bool
    can_predict: bool
    prediction_input_kinds: Tuple[str, ...]
    requires_feature_preparation: bool
    supported_task_types: Tuple[str, ...]
    supported_representations: Tuple[str, ...]
    supports_applicability_domain: bool
    supports_uncertainty: str
    multi_target_task_types: Tuple[str, ...] = ()
    supports_component_orchestration: bool = False
    supports_activity_cliff_feedback_loops: bool = False
    gpu_support: str = "not_declared"
    catalog_model_filename: str | None = None
    training_summary_filenames: Tuple[str, ...] = ("cs_copilot_training_summary.json",)
    test_prediction_relative_paths: Tuple[str, ...] = (
        "model_0/test_predictions.csv",
        "test_predictions.csv",
    )

    def as_dict(self) -> Dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


BACKEND_CAPABILITIES: Dict[str, BackendCapabilities] = {
    "chemprop": BackendCapabilities(
        backend_name="chemprop",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression", "classification"),
        multi_target_task_types=("regression", "classification"),
        supported_representations=("molecular_graph",),
        supports_applicability_domain=True,
        supports_uncertainty="none",
        gpu_support="runtime_dependent",
    ),
    "lightgbm": BackendCapabilities(
        backend_name="lightgbm",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("tabular_features_csv",),
        requires_feature_preparation=True,
        supported_task_types=("regression", "classification", "multiclass_classification"),
        supported_representations=SUPPORTED_TABULAR_REPRESENTATION_NAMES,
        supports_applicability_domain=True,
        supports_uncertainty="none",
        supports_activity_cliff_feedback_loops=True,
        gpu_support="supported_when_available",
        test_prediction_relative_paths=("test_predictions.csv",),
    ),
    "tabicl": BackendCapabilities(
        backend_name="tabicl",
        can_train=True,
        can_predict=True,
        prediction_input_kinds=("tabular_features_csv",),
        requires_feature_preparation=True,
        supported_task_types=("regression", "classification", "multiclass_classification"),
        supported_representations=SUPPORTED_TABULAR_REPRESENTATION_NAMES,
        supports_applicability_domain=True,
        supports_uncertainty="none",
        gpu_support="runtime_dependent",
        catalog_model_filename="best.pkl",
        training_summary_filenames=(
            "cs_copilot_training_summary.json",
            "tabicl_training_summary.json",
        ),
        test_prediction_relative_paths=("test_predictions.csv",),
    ),
    "ensemble": BackendCapabilities(
        backend_name="ensemble",
        can_train=False,
        can_predict=True,
        prediction_input_kinds=("smiles_csv",),
        requires_feature_preparation=False,
        supported_task_types=("regression", "classification"),
        supported_representations=(
            "catalog_consensus_regression",
            "catalog_consensus_classification",
        ),
        supports_applicability_domain=False,
        supports_uncertainty="component_disagreement_std",
        supports_component_orchestration=True,
        gpu_support="not_applicable",
    ),
}


def get_backend_capabilities(
    backend_name: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> BackendCapabilities:
    """Return capabilities for a registered backend."""
    capabilities_registry = BACKEND_CAPABILITIES if registry is None else registry
    normalized = str(backend_name).strip().lower()
    try:
        return capabilities_registry[normalized]
    except KeyError as exc:
        raise KeyError(f"No backend capabilities registered for `{backend_name}`.") from exc


def describe_backend_capabilities() -> Dict[str, Dict[str, object]]:
    """Return all known backend capabilities as serializable dictionaries."""
    return {name: capabilities.as_dict() for name, capabilities in BACKEND_CAPABILITIES.items()}


def enrich_backend_environment(
    backend_name: str,
    environment: Mapping[str, Any],
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> Dict[str, Any]:
    """Attach the canonical capability block to a runtime environment snapshot."""
    enriched = dict(environment)
    enriched["capabilities"] = get_backend_capabilities(
        backend_name,
        registry=registry,
    ).as_dict()
    return enriched


def backend_requires_feature_preparation(
    backend_name: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> bool:
    """Return whether a backend needs tabular feature preparation for SMILES inputs."""
    return get_backend_capabilities(
        backend_name,
        registry=registry,
    ).requires_feature_preparation


def normalize_capability_task_type(task_type: str) -> str:
    normalized = str(task_type or "").strip().lower()
    if normalized in {"binary_classification", "classification"}:
        return "classification"
    if normalized in {"multiclass", "multiclass_classification"}:
        return "multiclass_classification"
    return normalized


def backend_supports_task_type(
    backend_name: str,
    task_type: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> bool:
    capabilities = get_backend_capabilities(backend_name, registry=registry)
    return normalize_capability_task_type(task_type) in capabilities.supported_task_types


def backend_supports_multi_target(
    backend_name: str,
    task_type: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> bool:
    capabilities = get_backend_capabilities(backend_name, registry=registry)
    return normalize_capability_task_type(task_type) in capabilities.multi_target_task_types


def backend_supports_component_orchestration(
    backend_name: str,
    *,
    registry: Mapping[str, BackendCapabilities] | None = None,
) -> bool:
    """Return whether a backend represents a component-orchestrating model."""
    return get_backend_capabilities(
        backend_name,
        registry=registry,
    ).supports_component_orchestration
