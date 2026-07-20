"""Scientific tool specifications for the isolated Qsaria MCP profile."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from ..qsaria.adapter import QsariaToolSpec
from ..qsaria.toolkit_hub import (
    activity_cliffs_toolkit,
    benchmark_toolkit,
    curation_toolkit,
    ensemble_toolkit,
    inference_toolkit,
    registry_toolkit,
    training_toolkit,
)


def _spec(
    *,
    surface: str,
    method: str,
    factory: Callable[[], object],
    agent_name: str,
    summary: str,
    read_only: bool,
    output_paths: Mapping[str, str] | None = None,
    nested_output_paths: Mapping[str, str] | None = None,
    nested_blocked_inputs: tuple[str, ...] = (),
    registered_artifact_inputs: (
        Mapping[
            str,
            tuple[str, tuple[str, ...]],
        ]
        | None
    ) = None,
    json_payload_inputs: tuple[str, ...] = (),
    session_forces: Mapping[str, tuple[str, ...]] | None = None,
    forces: Mapping[str, Any] | None = None,
    catalog_access: bool = False,
    catalog_write: bool = False,
    requires_experiment: bool = True,
) -> QsariaToolSpec:
    return QsariaToolSpec(
        mcp_name=f"qsaria_{surface}_{method}",
        toolkit_factory=factory,
        method=method,
        summary=summary,
        group=f"qsaria_{surface}",
        read_only=read_only,
        destructive=False,
        open_world=False,
        agent_name=agent_name,
        output_paths=dict(output_paths or {}),
        nested_output_paths=dict(nested_output_paths or {}),
        nested_blocked_inputs=tuple(nested_blocked_inputs),
        registered_artifact_inputs=dict(registered_artifact_inputs or {}),
        json_payload_inputs=tuple(json_payload_inputs),
        session_forces=dict(session_forces or {}),
        forces=dict(forces or {}),
        artifact_category=surface,
        catalog_access=catalog_access,
        catalog_write=catalog_write,
        requires_experiment=requires_experiment,
    )


_TRAINING_BLOCKED_NESTED_INPUTS = (
    "extra_args.allow_auto_download",
    "extra_args.checkpoint_dir",
    "extra_args.disk_offload_dir",
    "extra_args.feature_cache_dir",
    "extra_args.hpopt_save_dir",
)


SPECS: list[QsariaToolSpec] = [
    # Dataset Curation Agent — five production toolkit methods.
    _spec(
        surface="curation",
        method="inspect_dataset_schema",
        factory=curation_toolkit,
        agent_name="qsaria_curation",
        summary="Inspect a dataset schema for Qsaria routing without creating an experiment.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="curation",
        method="identify_qsar_columns",
        factory=curation_toolkit,
        agent_name="qsaria_curation",
        summary="Identify candidate structure and target columns for a Qsaria dataset.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="curation",
        method="curate_qsar_dataset",
        factory=curation_toolkit,
        agent_name="qsaria_curation",
        summary="Curate a QSAR-ready dataset and persist all outputs in the experiment.",
        read_only=False,
        output_paths={
            "output_csv": "curated_dataset.csv",
            "report_path": "curation_report.json",
        },
    ),
    _spec(
        surface="curation",
        method="summarize_curated_dataset",
        factory=curation_toolkit,
        agent_name="qsaria_curation",
        summary="Summarize a curated Qsaria dataset without changing it.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="curation",
        method="write_curation_report",
        factory=curation_toolkit,
        agent_name="qsaria_curation",
        summary="Persist the structured curation report in the active experiment.",
        read_only=False,
        output_paths={
            "report_path": "curation_report.json",
            "bundle_path": "curation_bundle.zip",
        },
        session_forces={"curation_result": ("qsar_curation", "last_result")},
    ),
    # QSAR Training Agent — preparation is intentionally not exposed.
    _spec(
        surface="training",
        method="describe_qsar_training_environment",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Describe Qsaria training backends and compute availability.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="training",
        method="describe_backend_hyperparameters",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Describe supported direct and tunable backend hyperparameters.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="training",
        method="describe_tuning_engines",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Describe Qsaria hyperparameter-tuning engines and availability.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="training",
        method="describe_outlier_analysis",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Describe the shared Qsaria post-selection outlier policy.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="training",
        method="train_qsar_model",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Train one selected Qsaria backend with experiment-scoped outputs.",
        read_only=False,
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "training_output",
        },
        nested_blocked_inputs=_TRAINING_BLOCKED_NESTED_INPUTS,
    ),
    _spec(
        surface="training",
        method="train_chemprop_model",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Train Chemprop through the shared Qsaria training facade.",
        read_only=False,
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "training_output",
        },
        nested_blocked_inputs=_TRAINING_BLOCKED_NESTED_INPUTS,
    ),
    _spec(
        surface="training",
        method="train_lightgbm_model",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Train LightGBM through the shared Qsaria training facade.",
        read_only=False,
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "training_output",
        },
        nested_blocked_inputs=_TRAINING_BLOCKED_NESTED_INPUTS,
    ),
    _spec(
        surface="training",
        method="train_tabicl_model",
        factory=training_toolkit,
        agent_name="qsaria_training",
        summary="Train TabICL through the shared Qsaria training facade.",
        read_only=False,
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "training_output",
        },
        nested_output_paths={
            "extra_args.disk_offload_dir": "disk_offload",
        },
        nested_blocked_inputs=_TRAINING_BLOCKED_NESTED_INPUTS,
    ),
    # Model Registry Agent — eleven production toolkit methods.
    _spec(
        surface="registry",
        method="describe_backends",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Describe prediction backends available to Qsaria.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="registry",
        method="describe_catalog",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Describe the persistent Qsaria model catalog without rewriting it.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    _spec(
        surface="registry",
        method="list_catalog_models",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="List persistent catalog models with current runtime annotations.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    _spec(
        surface="registry",
        method="summarize_catalog_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Summarize one persistent catalog model.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    _spec(
        surface="registry",
        method="recommend_catalog_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Rank catalog models for an explicit task and domain request.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    _spec(
        surface="registry",
        method="register_catalog_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Register an existing catalog model in the experiment runtime.",
        read_only=False,
        catalog_access=True,
    ),
    _spec(
        surface="registry",
        method="register_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary=(
            "Register one exact model payload in the current experiment runtime; use this "
            "single-candidate route when Training returned no candidate manifest."
        ),
        read_only=False,
    ),
    _spec(
        surface="registry",
        method="persist_registered_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Persist a registered model and atomically update the global catalog.",
        read_only=False,
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="registry",
        method="register_and_persist_candidates",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary=(
            "Register and persist completed training candidates from a verified non-empty "
            "candidate manifest; never use this batch route for one recommended payload."
        ),
        read_only=False,
        forces={"candidate_registry_payloads": None},
        registered_artifact_inputs={
            "candidate_manifest_path": (
                "catalog_candidates_manifest.json",
                ("qsaria_training_",),
            )
        },
        json_payload_inputs=("candidate_manifest_path",),
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="registry",
        method="list_registered_models",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="List models registered in the current experiment.",
        read_only=False,
    ),
    _spec(
        surface="registry",
        method="summarize_model",
        factory=registry_toolkit,
        agent_name="qsaria_registry",
        summary="Summarize a session-registered or persistent Qsaria model.",
        read_only=False,
        catalog_access=True,
    ),
    # Model Inference Agent — all outputs remain experiment-scoped.
    _spec(
        surface="inference",
        method="predict_from_csv",
        factory=inference_toolkit,
        agent_name="qsaria_inference",
        summary="Run model inference on a CSV and persist predictions in the experiment.",
        read_only=False,
        output_paths={
            "preds_path": "predictions.csv",
            "materialized_input_path": "normalized_input.csv",
        },
        catalog_access=True,
    ),
    _spec(
        surface="inference",
        method="predict_from_smiles",
        factory=inference_toolkit,
        agent_name="qsaria_inference",
        summary="Run model inference on an explicit SMILES list and persist predictions.",
        read_only=False,
        output_paths={
            "preds_path": "predictions.csv",
            "input_csv_path": "smiles_input.csv",
        },
        catalog_access=True,
    ),
    _spec(
        surface="inference",
        method="evaluate_model_on_dataset",
        factory=inference_toolkit,
        agent_name="qsaria_inference",
        summary="Evaluate a registered model on an external dataset.",
        read_only=False,
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="inference",
        method="export_prediction_summary",
        factory=inference_toolkit,
        agent_name="qsaria_inference",
        summary="Export the current experiment prediction history as CSV.",
        read_only=False,
        output_paths={"summary_csv": "prediction_summary.csv"},
    ),
    # Ensemble operations belong to Registry in the five-agent architecture.
    _spec(
        surface="ensemble",
        method="inspect_ensemble_candidates",
        factory=ensemble_toolkit,
        agent_name="qsaria_registry",
        summary="Inspect compatible catalog candidates for a Qsaria ensemble.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    _spec(
        surface="ensemble",
        method="create_ensemble_from_catalog",
        factory=ensemble_toolkit,
        agent_name="qsaria_registry",
        summary="Create and persist a post-hoc ensemble from catalog models.",
        read_only=False,
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="ensemble",
        method="evaluate_ensemble_on_dataset",
        factory=ensemble_toolkit,
        agent_name="qsaria_registry",
        summary="Evaluate a persistent ensemble against an external dataset.",
        read_only=False,
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="ensemble",
        method="summarize_ensemble",
        factory=ensemble_toolkit,
        agent_name="qsaria_registry",
        summary="Summarize a persistent Qsaria ensemble.",
        read_only=True,
        catalog_access=True,
        requires_experiment=False,
    ),
    # Training adjuncts retain their audited MCP names.
    _spec(
        surface="benchmark",
        method="benchmark_qsar_models",
        factory=benchmark_toolkit,
        agent_name="qsaria_training",
        summary="Run a multi-backend Qsaria benchmark and persist completed candidates.",
        read_only=False,
        output_paths={
            "bundle_dir": "training_bundles",
            "output_dir": "benchmark_output",
        },
        catalog_access=True,
        catalog_write=True,
    ),
    _spec(
        surface="activity_cliffs",
        method="list_activity_cliff_indexes",
        factory=activity_cliffs_toolkit,
        agent_name="qsaria_training",
        summary="List activity-cliff indexes available to Qsaria training.",
        read_only=True,
        requires_experiment=False,
    ),
    _spec(
        surface="activity_cliffs",
        method="prepare_activity_cliff_context",
        factory=activity_cliffs_toolkit,
        agent_name="qsaria_training",
        summary="Annotate training data with experiment-scoped activity-cliff artifacts.",
        read_only=False,
        output_paths={"output_dir": "activity_cliff_output"},
    ),
]


def scientific_specs() -> list[QsariaToolSpec]:
    """Return a copy of the audited 35-tool scientific surface."""

    return list(SPECS)


__all__ = ["SPECS", "scientific_specs"]
