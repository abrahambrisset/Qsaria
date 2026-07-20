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
        summary="Validate and record one structured Qsaria sub-agent handoff.",
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
]


__all__ = ["SPECS"]
