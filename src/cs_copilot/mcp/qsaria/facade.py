"""Thin MCP-friendly facade over :mod:`.experiments`.

Tool registration lives outside this module so the generic MCP profile remains
unchanged.  The facade contains no scientific branching and never constructs
an LLM or an Agno team.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

from .experiments import ExperimentManager, get_experiment_manager


class ExperimentFacade:
    """Expose experiment lifecycle/reporting as ordinary typed callables."""

    def __init__(self, manager: ExperimentManager | None = None) -> None:
        self.manager = manager or get_experiment_manager()

    def bootstrap(self) -> dict[str, Any]:
        """Return Qsaria capabilities without listing or resuming experiments."""

        return self.manager.bootstrap()

    def create(
        self,
        user_request: str,
        report_language: str = "fr",
        metadata: Mapping[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new persistent Qsaria experiment."""

        return self.manager.create_experiment(
            user_request=user_request,
            report_language=report_language,
            metadata=metadata,
            experiment_id=experiment_id,
        )

    def open(self, experiment_id: str) -> dict[str, Any]:
        """Explicitly open a known experiment."""

        return self.manager.open_experiment(experiment_id)

    def list(
        self,
        limit: int = 20,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """List compact experiment summaries on explicit request."""

        return self.manager.list_experiments(limit=limit, status=status)

    def get_state(self, experiment_id: str) -> dict[str, Any]:
        """Return the public state of one experiment."""

        return self.manager.get_experiment_state(experiment_id)

    def list_artifacts(
        self,
        experiment_id: str,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        """List scientific artifacts belonging to an experiment."""

        return self.manager.list_artifacts(experiment_id, kind=kind)

    def get_artifact(
        self,
        experiment_id: str,
        artifact_id: str,
        max_preview_bytes: int = 65536,
    ) -> dict[str, Any]:
        """Get artifact metadata and a bounded text preview."""

        return self.manager.get_artifact(
            experiment_id,
            artifact_id,
            max_preview_bytes=max_preview_bytes,
        )

    def record_handoff(
        self,
        experiment_id: str,
        handoff: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Append one validated Qsaria sub-agent handoff."""

        return self.manager.record_handoff(experiment_id, handoff)

    def complete(
        self,
        experiment_id: str,
        status: Literal["completed", "partial", "terminal_failure"] = "completed",
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Finalize an experiment after coordinator verification."""

        return self.manager.complete_experiment(experiment_id, status=status, summary=summary)

    def report_build_context(self, experiment_id: str) -> dict[str, Any]:
        """Return fact-first context for the Qsaria Report sub-agent."""

        return self.manager.build_report_context(experiment_id)

    def report_save(
        self,
        experiment_id: str,
        report_content: str | Mapping[str, Any],
        report_format: Literal["markdown", "json"] = "markdown",
    ) -> dict[str, Any]:
        """Persist a final structured or Markdown Qsaria report."""

        return self.manager.save_report(
            experiment_id,
            report_content=report_content,
            report_format=report_format,
        )


def experiment_facade(manager: ExperimentManager | None = None) -> ExperimentFacade:
    """Build a facade sharing the server's :class:`ExperimentManager`."""

    return ExperimentFacade(manager)


__all__ = ["ExperimentFacade", "experiment_facade"]
