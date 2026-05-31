#!/usr/bin/env python
# coding: utf-8
"""Deterministic QSAR workflow contracts used by the agent layer."""

from __future__ import annotations

import copy
import re
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Tuple


class QSARWorkflowKind(str, Enum):
    """Known coordinator-level QSAR workflows."""

    CURATION_ONLY = "curation_only"
    TRAINING = "training"
    PREDICTION = "prediction"
    REGISTRY = "registry"
    ENSEMBLE = "ensemble"
    EXPORT_ONLY = "export_only"


QSAR_AGENT_ROUTES: Dict[QSARWorkflowKind, Tuple[str, ...]] = {
    QSARWorkflowKind.CURATION_ONLY: ("dataset_curation", "qsar_report"),
    QSARWorkflowKind.TRAINING: (
        "dataset_curation",
        "qsar_training",
        "model_registry",
        "qsar_report",
    ),
    QSARWorkflowKind.PREDICTION: ("model_inference", "qsar_report"),
    QSARWorkflowKind.REGISTRY: ("model_registry", "qsar_report"),
    QSARWorkflowKind.ENSEMBLE: ("model_registry", "qsar_report"),
    QSARWorkflowKind.EXPORT_ONLY: ("qsar_report",),
}

QSAR_AGENT_NAMES: Dict[str, str] = {
    "dataset_curation": "Dataset Curation",
    "qsar_training": "QSAR Training",
    "model_registry": "Model Registry",
    "model_inference": "Model Inference",
    "qsar_report": "QSAR Report",
}

_LATEX_SHORTCUT_RE = re.compile(r"^\s*@?latex\s*$", re.I)
_FRENCH_SIGNAL_RE = re.compile(
    r"\b(cr[eé]e|créer|entraine|entra[iî]ne|pr[eé]dis|pr[eé]dire|mod[eè]le|"
    r"jeu de donn[eé]es|r[eé]sume|rapport|fichier|g[eé]n[eè]re|g[eé]n[eé]rer)\b",
    re.I,
)
_EXPORT_RE = re.compile(r"\b(latex|payload|tex|export|exporte|g[eé]n[eè]re|g[eé]n[eé]rer)\b", re.I)
_PREDICTION_CONTEXT_RE = re.compile(r"\b(latest|derni[eè]re|prediction|pr[eé]diction|payload)\b", re.I)
_ENSEMBLE_RE = re.compile(r"\b(ensemble|consensus)\b", re.I)
_ENSEMBLE_ACTION_RE = re.compile(
    r"\b(create|build|make|summari[sz]e|list|inspect|compare|cr[eé]e|créer|r[eé]sume)\b",
    re.I,
)
_TRAINING_RE = re.compile(
    r"\b(train|training|fit|validate|validation|benchmark|chemprop|lightgbm|tabicl|"
    r"entrain|entra[iî]n|apprendre)\b",
    re.I,
)
_CURATION_RE = re.compile(
    r"\b(curate|curation|clean|prepare|standardi[sz]e|deduplicate|dataset schema|"
    r"pr[eé]pare|nettoie|jeu de donn[eé]es)\b",
    re.I,
)
_PREDICTION_RE = re.compile(
    r"\b(predict|prediction|infer|inference|screen|applicability domain|ad |"
    r"lipophilicity|pr[eé]dis|pr[eé]diction|inf[eé]rence)\b",
    re.I,
)
_REGISTRY_RE = re.compile(
    r"\b(catalog|catalogue|registry|register|model summary|model comparison|recommend|"
    r"backend|capability|capabilities|available models|mod[eè]les disponibles|"
    r"backends?)\b",
    re.I,
)


@dataclass(frozen=True)
class QSARWorkflowPlan:
    """A deterministic routing decision for the QSAR coordinator."""

    workflow: QSARWorkflowKind
    route: Tuple[str, ...]
    report_language: str
    export_only: bool = False
    rerun_prediction: bool = True

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["workflow"] = self.workflow.value
        payload["route"] = list(self.route)
        return payload


def detect_report_language(message: str) -> str:
    """Return the report language required by the latest user message."""
    text = message or ""
    return "French" if _FRENCH_SIGNAL_RE.search(text) else "English"


def classify_qsar_workflow(message: str) -> QSARWorkflowPlan:
    """Classify a user request into the supported QSAR workflow routes.

    The classifier is intentionally conservative: it enforces the same high-level
    routes the prompts describe, while leaving domain details to the specialists.
    """
    text = message or ""
    language = detect_report_language(text)

    if _LATEX_SHORTCUT_RE.match(text) or (
        _EXPORT_RE.search(text) and _PREDICTION_CONTEXT_RE.search(text)
    ):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.EXPORT_ONLY,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.EXPORT_ONLY],
            report_language=language,
            export_only=True,
            rerun_prediction=False,
        )

    if _ENSEMBLE_RE.search(text) and _ENSEMBLE_ACTION_RE.search(text):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.ENSEMBLE,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.ENSEMBLE],
            report_language=language,
        )

    if _TRAINING_RE.search(text):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.TRAINING,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.TRAINING],
            report_language=language,
        )

    if _PREDICTION_RE.search(text):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.PREDICTION,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.PREDICTION],
            report_language=language,
        )

    if _REGISTRY_RE.search(text):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.REGISTRY,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.REGISTRY],
            report_language=language,
        )

    if _CURATION_RE.search(text):
        return QSARWorkflowPlan(
            workflow=QSARWorkflowKind.CURATION_ONLY,
            route=QSAR_AGENT_ROUTES[QSARWorkflowKind.CURATION_ONLY],
            report_language=language,
        )

    return QSARWorkflowPlan(
        workflow=QSARWorkflowKind.REGISTRY,
        route=QSAR_AGENT_ROUTES[QSARWorkflowKind.REGISTRY],
        report_language=language,
    )


def plan_qsar_workflow(message: str) -> Dict[str, Any]:
    """Return the deterministic QSAR route for a user message as plain data."""
    return classify_qsar_workflow(message).to_dict()


def describe_qsar_routes() -> str:
    """Return a compact, prompt-safe description of deterministic QSAR routes."""
    lines = []
    for workflow, route in QSAR_AGENT_ROUTES.items():
        readable_route = " -> ".join(QSAR_AGENT_NAMES[item] for item in route)
        lines.append(f"- {workflow.value}: {readable_route}")
    return "\n".join(lines)


def default_prediction_state() -> Dict[str, Any]:
    return {
        "registered": {},
        "last_prediction": {},
        "prediction_history": [],
        "catalog_recommendations": {},
        "training_runs": [],
        "active_training_run": None,
    }


def default_qsar_session_state() -> Dict[str, Any]:
    """Return the shared QSAR state skeleton used by all isolated QSAR agents."""
    return {
        "prediction_models": default_prediction_state(),
        "prediction_outputs": {
            "latest_predictions_csv": None,
            "latest_summary": None,
        },
        "qsar_curation": {
            "last_request": {},
            "last_result": {},
            "history": [],
        },
        "qsar_training": {
            "last_request": {},
            "last_result": {},
        },
        "qsar_registry": {
            "last_request": {},
            "last_result": {},
        },
        "qsar_inference": {
            "last_request": {},
            "last_result": {},
        },
        "qsar_report": {
            "last_request": {},
            "last_result": {},
        },
        "qsar_workflow": {
            "last_plan": {},
            "history": [],
        },
    }


def copy_qsar_session_state() -> Dict[str, Any]:
    """Return an independent copy of the default QSAR state."""
    return copy.deepcopy(default_qsar_session_state())

