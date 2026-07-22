"""Lifecycle and reporting specifications for the Qsaria MCP profile."""

from __future__ import annotations

from ..qsaria.facade import experiment_facade
from ..qsaria.lifecycle_adapter import QsariaLifecycleSpec


def _spec(
    *,
    name: str,
    method: str,
    summary: str,
    read_only: bool,
    group: str = "qsaria_lifecycle",
) -> QsariaLifecycleSpec:
    return QsariaLifecycleSpec(
        mcp_name=name,
        toolkit_factory=experiment_facade,
        method=method,
        summary=summary,
        group=group,
        read_only=read_only,
        destructive=False,
        open_world=False,
    )


SPECS: list[QsariaLifecycleSpec] = [
    _spec(
        name="qsaria_bootstrap",
        method="bootstrap",
        summary="Describe Qsaria capabilities and contracts without resuming an experiment.",
        read_only=True,
    ),
    _spec(
        name="qsaria_create_experiment",
        method="create",
        summary="Create a new durable Qsaria experiment for scientific work that writes.",
        read_only=False,
    ),
    _spec(
        name="qsaria_open_experiment",
        method="open",
        summary="Open one explicitly identified Qsaria experiment.",
        read_only=True,
    ),
    _spec(
        name="qsaria_list_experiments",
        method="list",
        summary="List Qsaria experiments only after an explicit request.",
        read_only=True,
    ),
    _spec(
        name="qsaria_get_experiment_state",
        method="get_state",
        summary="Read the public structured state of one Qsaria experiment.",
        read_only=True,
    ),
    _spec(
        name="qsaria_list_artifacts",
        method="list_artifacts",
        summary="List structured artifacts belonging to one Qsaria experiment.",
        read_only=False,
    ),
    _spec(
        name="qsaria_get_artifact",
        method="get_artifact",
        summary="Read metadata and a bounded preview for one experiment artifact.",
        read_only=False,
    ),
    _spec(
        name="qsaria_record_handoff",
        method="record_handoff",
        summary=(
            "Validate and record one fresh public Qsaria v1.0 sub-agent handoff; "
            "never pass a toolkit-returned handoff or reporting_handoff directly."
        ),
        read_only=False,
        group="qsaria_coordination",
    ),
    _spec(
        name="qsaria_complete_experiment",
        method="complete",
        summary="Finalize an experiment after coordinator verification.",
        read_only=False,
        group="qsaria_coordination",
    ),
    _spec(
        name="qsaria_report_build_context",
        method="report_build_context",
        summary="Build fact-first context for the final Qsaria Report sub-agent.",
        read_only=False,
        group="qsaria_report",
    ),
    _spec(
        name="qsaria_report_save",
        method="report_save",
        summary="Persist the single final Qsaria report as an experiment artifact.",
        read_only=False,
        group="qsaria_report",
    ),
    _spec(
        name="qsaria_curation_start_operation",
        method="curation_start_operation",
        summary="Start one durable detached Curation operation for a short-timeout client.",
        read_only=False,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_training_start_operation",
        method="training_start_operation",
        summary="Start one durable detached Training operation for a short-timeout client.",
        read_only=False,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_registry_start_operation",
        method="registry_start_operation",
        summary="Start one durable detached Registry or Ensemble operation.",
        read_only=False,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_inference_start_operation",
        method="inference_start_operation",
        summary="Start one durable detached Inference operation for a short-timeout client.",
        read_only=False,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_list_operations",
        method="list_operations",
        summary="List durable operations belonging to one Qsaria experiment.",
        read_only=True,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_get_operation_state",
        method="get_operation_state",
        summary="Read the non-blocking state of one durable Qsaria operation.",
        read_only=True,
        group="qsaria_operations",
    ),
    _spec(
        name="qsaria_get_operation_result",
        method="get_operation_result",
        summary="Read the final payload of one durable Qsaria operation when ready.",
        read_only=True,
        group="qsaria_operations",
    ),
]


__all__ = ["SPECS"]
