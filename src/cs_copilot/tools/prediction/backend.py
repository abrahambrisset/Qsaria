#!/usr/bin/env python
# coding: utf-8
"""
Prediction backend abstractions for pluggable molecular property modeling.

The goal of this module is to decouple agent/tooling logic from a specific
predictive engine such as Chemprop.  This gives the project a stable contract
for later growth toward multiple backends, multitask prediction, uncertainty,
reaction modeling, and active learning.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .qsar_contracts import BackendRunRequest


class PredictionBackendError(Exception):
    """Base exception for prediction backend failures."""


class BackendNotAvailableError(PredictionBackendError):
    """Raised when a backend is not installed or not usable in the environment."""


class InvalidPredictionInputError(PredictionBackendError):
    """Raised when model inputs or metadata are invalid."""


class PredictionExecutionError(PredictionBackendError):
    """Raised when a backend invocation fails during model execution."""


@dataclass
class PredictionTaskSpec:
    """Declarative description of a prediction task supported by a model."""

    task_type: str
    smiles_columns: List[str] = field(default_factory=lambda: ["smiles"])
    target_columns: List[str] = field(default_factory=list)
    reaction_columns: List[str] = field(default_factory=list)
    uncertainty_method: Optional[str] = None
    calibration_method: Optional[str] = None


@dataclass
class PredictionModelRecord:
    """Metadata tracked for a registered predictive model."""

    model_id: str
    backend_name: str
    model_path: str
    task: PredictionTaskSpec
    metadata_path: Optional[str] = None
    display_name: Optional[str] = None
    description: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    version: Optional[str] = None
    status: str = "experimental"
    owner: Optional[str] = None
    source: Optional[str] = None
    domain_summary: Optional[str] = None
    strengths: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    recommended_for: List[str] = field(default_factory=list)
    not_recommended_for: List[str] = field(default_factory=list)
    known_metrics: Dict[str, Any] = field(default_factory=dict)
    training_data_summary: Dict[str, Any] = field(default_factory=dict)
    inference_profile: Dict[str, Any] = field(default_factory=dict)
    selection_hints: Dict[str, Any] = field(default_factory=dict)
    applicability_domain: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "backend_name": self.backend_name,
            "model_path": self.model_path,
            "metadata_path": self.metadata_path,
            "display_name": self.display_name or self.model_id,
            "description": self.description or "",
            "version": self.version,
            "status": self.status,
            "owner": self.owner,
            "source": self.source,
            "domain_summary": self.domain_summary or "",
            "strengths": list(self.strengths),
            "limitations": list(self.limitations),
            "recommended_for": list(self.recommended_for),
            "not_recommended_for": list(self.not_recommended_for),
            "known_metrics": dict(self.known_metrics),
            "training_data_summary": dict(self.training_data_summary),
            "inference_profile": dict(self.inference_profile),
            "selection_hints": dict(self.selection_hints),
            "applicability_domain": dict(self.applicability_domain),
            "task": {
                "task_type": self.task.task_type,
                "smiles_columns": list(self.task.smiles_columns),
                "target_columns": list(self.task.target_columns),
                "reaction_columns": list(self.task.reaction_columns),
                "uncertainty_method": self.task.uncertainty_method,
                "calibration_method": self.task.calibration_method,
            },
            "tags": dict(self.tags),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PredictionModelRecord":
        """Reconstruct a record from a serialized payload."""
        task_payload = payload.get("task", {})
        return cls(
            model_id=payload["model_id"],
            backend_name=payload["backend_name"],
            model_path=payload["model_path"],
            metadata_path=payload.get("metadata_path"),
            display_name=payload.get("display_name"),
            description=payload.get("description"),
            tags=payload.get("tags", {}),
            version=payload.get("version"),
            status=payload.get("status", "experimental"),
            owner=payload.get("owner"),
            source=payload.get("source"),
            domain_summary=payload.get("domain_summary"),
            strengths=payload.get("strengths", []),
            limitations=payload.get("limitations", []),
            recommended_for=payload.get("recommended_for", []),
            not_recommended_for=payload.get("not_recommended_for", []),
            known_metrics=payload.get("known_metrics", {}),
            training_data_summary=payload.get("training_data_summary", {}),
            inference_profile=payload.get("inference_profile", {}),
            selection_hints=payload.get("selection_hints", {}),
            applicability_domain=payload.get("applicability_domain", {}),
            task=PredictionTaskSpec(
                task_type=task_payload.get("task_type", "regression"),
                smiles_columns=task_payload.get("smiles_columns", ["smiles"]),
                target_columns=task_payload.get("target_columns", []),
                reaction_columns=task_payload.get("reaction_columns", []),
                uncertainty_method=task_payload.get("uncertainty_method"),
                calibration_method=task_payload.get("calibration_method"),
            ),
        )


class PredictionBackend(ABC):
    """Common contract for predictive backends used by agent toolkits."""

    backend_name = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """Return True when the backend is usable in the current environment."""

    @abstractmethod
    def describe_environment(self) -> Dict[str, Any]:
        """Return a lightweight description of backend availability and versioning."""

    @abstractmethod
    def validate_model_path(self, model_path: str) -> Path:
        """Validate a model artifact path and return a normalized Path."""

    @abstractmethod
    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        """Run batch prediction from a CSV input file."""

    @abstractmethod
    def train_model(self, request: BackendRunRequest) -> Dict[str, Any]:
        """Train a model and return metadata about the training outputs."""
