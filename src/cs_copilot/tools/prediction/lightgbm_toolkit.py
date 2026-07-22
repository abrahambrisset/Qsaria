#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing LightGBM-backed tabular QSAR workflows.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import (
    build_activity_cliff_loop_comparison_plots,
    prepare_activity_cliff_context,
    split_activity_cliff_args,
)

from .applicability_domain import fit_modern_applicability_domain, score_modern_applicability_domain
from .backend import PredictionTaskSpec
from .hyperparameter_tuning import (
    HYPERPARAMETER_CONTRACT_VERSION,
    HyperparameterTuningError,
    LightGBMOptunaAdapter,
    build_tuning_progress_plot,
    normalize_tuning_config,
    tuning_metadata_for_catalog,
    tuning_sampler_metadata,
)
from .lightgbm_backend import LightGBMBackend
from .outlier_analysis import (
    OutlierAnalysisConfig,
    attach_activity_cliff_annotations,
    attach_ad_annotations,
    deduplicate_parameter_configurations,
    normalize_outlier_analysis_config,
    select_outliers,
    selection_predictions_from_frame,
    write_outlier_analysis_artifacts,
    write_outlier_variant_comparison,
)
from .qsar_contracts import build_backend_run_request
from .qsar_progress import apply_progress_update
from .qsar_splitters import (
    build_full_train_split_payload,
    build_qsar_split_payload,
    build_repeated_kfold_split_payloads,
)
from .qsar_training_policy import (
    assess_protocol_results,
    describe_compute_environment,
    project_now,
    resolve_backend_n_jobs,
    resolve_training_profile,
    safe_slug,
    seed_policy_reporting_text,
    seed_policy_reproducibility_metadata,
    summarize_training_durations,
)
from .qsar_validation_strategy import resolve_validation_strategy
from .session_state import (
    bundle_artifacts,
    discover_curation_artifacts_near_dataset,
    get_prediction_state,
    latest_curation_artifacts,
    write_active_training_marker,
)
from .tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
)
from .training_orchestration import (
    apply_training_profile,
    build_applicability_domain_for_training,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    collect_training_bundle_files,
    compute_classification_metrics,
    compute_regression_metrics,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)

logger = logging.getLogger(__name__)


def _ad_status_series(applicability_domain_result: Dict[str, Any]) -> Optional[pd.Series]:
    """Return AD statuses without evaluating a pandas DataFrame as a boolean."""
    scores = applicability_domain_result.get("scores")
    if not isinstance(scores, pd.DataFrame):
        return None
    return scores.get("ad_status")


class LightGBMToolkit(Toolkit):
    """Toolkit exposing LightGBM-backed QSAR orchestration for tabular datasets."""

    def __init__(self, backend: Optional[LightGBMBackend] = None):
        super().__init__("lightgbm_prediction")
        self.backend = backend or LightGBMBackend()

    def _train_backend(
        self,
        *,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        resolved_parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        request = build_backend_run_request(
            backend="lightgbm",
            train_csv=train_csv,
            output_dir=output_dir,
            task_type=task.task_type,
            smiles_columns=list(task.smiles_columns),
            target_columns=list(task.target_columns),
            reaction_columns=list(task.reaction_columns),
            resolved_parameters=resolved_parameters,
        )
        return self.backend.train_model(request)

    def is_lightgbm_available(self) -> bool:
        """Return whether the LightGBM backend is available in the current environment."""
        return self.backend.is_available()

    def describe_compute_environment(self) -> Dict[str, Any]:
        return describe_compute_environment()

    def _resolve_training_profile(self, compute_env: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_training_profile(compute_env)

    def _resolve_validation_protocol(
        self,
        *,
        requested_protocol: Optional[str],
        training_profile: str,
        seed_policy: Optional[Dict[str, Any]] = None,
        seed_policy_mode: str = "generated_per_run",
        base_seed: Optional[int] = None,
        validation_strategy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return resolve_validation_strategy(
            requested_protocol=requested_protocol,
            validation_strategy=validation_strategy,
            training_profile=training_profile,
            seed_policy=seed_policy,
            seed_policy_mode=seed_policy_mode,
            base_seed=base_seed,
        )

    def _run_outlier_refits(
        self,
        *,
        train_csv: str,
        split_source_df: pd.DataFrame,
        task: PredictionTaskSpec,
        split_payload: List[Dict[str, Any]],
        selection_prediction_frame: pd.DataFrame,
        output_dir: Path,
        base_args: Dict[str, Any],
        selected_parameters: Dict[str, Any],
        feature_columns: List[str],
        representation_name: Optional[str],
        activity_args: Dict[str, Any],
        applicability_domain_methods: Optional[List[str] | str],
        similarity_top_k_neighbors: int | str | None,
        similarity_threshold_percentile: float | str | None,
        selection_fraction: float,
        baseline_run: Optional[Dict[str, Any]] = None,
        fit_variants: bool = True,
        progress_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
        fold_label: str = "selection",
        repeat_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Select validation outliers and, when requested, train final variants.

        The selected model has already been fitted only on its train rows.  It
        is intentionally passed as a frame rather than persisted as another
        model artifact.
        """
        split = split_payload[0]
        train_indices = [int(index) for index in split.get("train") or []]
        validation_indices = [
            int(index) for index in (split.get("val") or split.get("validation") or [])
        ]
        test_indices = [int(index) for index in split.get("test") or []]
        if not train_indices or not validation_indices:
            raise ValueError(
                "Outlier analysis requires fixed non-empty train and validation indices."
            )
        target_column = task.target_columns[0]
        selection = selection_predictions_from_frame(
            selection_prediction_frame,
            source_row_indices=validation_indices,
            target_column=target_column,
            fold_label=fold_label,
            repeat_index=repeat_index,
        )

        outlier_root = output_dir / "outlier_analysis"
        development_indices = [*train_indices, *validation_indices]
        development_frame = split_source_df.iloc[development_indices].copy()
        development_frame.index = development_indices

        # Build the selection AD from train only.  It is an in-memory/temporary
        # diagnostic and never reuses final-model AD artifacts.
        with tempfile.TemporaryDirectory(prefix="qsaria_outlier_lightgbm_") as temporary_dir:
            temporary_ad = fit_modern_applicability_domain(
                feature_frame=split_source_df.iloc[train_indices][feature_columns].reset_index(
                    drop=True
                ),
                feature_columns=feature_columns,
                output_dir=Path(temporary_dir) / "applicability_domain",
                model_id="lightgbm_outlier_selection",
                feature_space=representation_name or "tabular",
                representation_name=representation_name,
                methods=applicability_domain_methods,
                random_state=int(base_args.get("random_state") or 0),
                all_feature_frame=split_source_df[feature_columns],
                train_indices=train_indices,
                split_indices=split,
                similarity_top_k_neighbors=similarity_top_k_neighbors,
                similarity_threshold_percentile=similarity_threshold_percentile,
            )
            validation_ad = score_modern_applicability_domain(
                feature_frame=split_source_df.iloc[validation_indices][feature_columns].reset_index(
                    drop=True
                ),
                applicability_domain=temporary_ad,
                row_indices=validation_indices,
            )
            statuses = _ad_status_series(validation_ad)
        selection = attach_ad_annotations(
            selection,
            ad_statuses=statuses.tolist() if statuses is not None else None,
        )

        # Activity cliffs used for removal are computed on development data
        # only.  The already existing full-dataset AC report is descriptive.
        ac_annotations: Optional[pd.DataFrame] = None
        if task.task_type == "regression":
            ac_source = outlier_root / "development_for_selection.csv"
            ac_source.parent.mkdir(parents=True, exist_ok=True)
            ac_input = development_frame.copy()
            ac_input = ac_input.drop(columns=["source_row_index"], errors="ignore")
            ac_input.insert(0, "source_row_index", ac_input.index)
            ac_input.to_csv(ac_source, index=False)
            try:
                ac_context = prepare_activity_cliff_context(
                    train_csv=str(ac_source),
                    output_dir=str(outlier_root / "activity_cliffs_development"),
                    smiles_column=task.smiles_columns[0] if task.smiles_columns else "smiles",
                    target_column=target_column,
                    **activity_args,
                )
                annotated_path = ac_context.get("annotated_training_csv")
                if annotated_path and Path(str(annotated_path)).exists():
                    ac_annotations = pd.read_csv(annotated_path)
            except Exception as exc:
                logger.warning(
                    "Development-only Activity Cliff selection annotations unavailable: %s", exc
                )
        selection = attach_activity_cliff_annotations(selection, annotations=ac_annotations)
        selected_rows, selection_summary = select_outliers(
            selection,
            task_type=task.task_type,
            selection_fraction=selection_fraction,
        )
        artifacts = write_outlier_analysis_artifacts(
            output_dir=outlier_root,
            selection_frame=selected_rows,
            selection_summary=selection_summary,
            development_frame=development_frame,
            extra_summary={
                "fold_label": fold_label,
                "repeat_index": repeat_index,
                "selection_model": "temporary_train_only_fit",
                "activity_cliff_scope": "development_only",
                "test_rows_used_for_selection": 0,
            },
        )
        selected_indices = [
            int(value)
            for value in selected_rows.loc[
                selected_rows["selected_for_removal"].astype(bool), "source_row_index"
            ].tolist()
        ]

        # CV first needs the OOF selection rows from every fold.  It must not
        # create a fold-local final model before those rows are pooled: that
        # would be both wasteful and scientifically different from the
        # protocol's one-list-per-repeat policy.
        if not fit_variants:
            return {
                "summary": {
                    **selection_summary,
                    "enabled": True,
                    "artifacts": artifacts,
                    "summary_path": artifacts.get("summary_path"),
                    "selection_predictions_path": artifacts.get("selection_predictions_path"),
                    "filtered_development_path": artifacts.get("filtered_development_path"),
                    "plot_artifacts": {
                        key: value
                        for key, value in artifacts.items()
                        if key.startswith("outlier_selection_")
                    },
                    "comparison_path": None,
                    "test_comparison_policy": "descriptive_only_no_automatic_winner",
                    "selected_source_row_indices": selected_indices,
                    "variants": [],
                },
                "variants": [],
            }

        final_payload = [
            {
                "train": development_indices,
                **({"test": test_indices} if test_indices else {}),
                "metadata": {
                    **dict(split.get("metadata") or {}),
                    "refit_on_train_validation": True,
                    "outlier_variant": "baseline",
                },
            }
        ]
        if baseline_run is None:
            if progress_callback is not None:
                progress_callback("Training baseline refit", {"detail": "train + validation"})
            baseline_run = self._train_backend(
                train_csv=train_csv,
                output_dir=str(output_dir / "baseline"),
                task=task,
                resolved_parameters={
                    **base_args,
                    **selected_parameters,
                    "split_payload": final_payload,
                    "final_refit": True,
                    "refit_on_train_validation": True,
                    "early_stopping_rounds": 0,
                },
            )
        baseline_run = dict(baseline_run)
        baseline_run["outlier_variant"] = "baseline"
        baseline_run["outlier_selected_count"] = len(selected_indices)

        variants = [{"variant_id": "baseline", "run": baseline_run}]
        if selected_indices:
            filtered_train = [
                index for index in development_indices if index not in set(selected_indices)
            ]
            filtered_payload = [
                {
                    "train": filtered_train,
                    **({"test": test_indices} if test_indices else {}),
                    "metadata": {
                        **dict(split.get("metadata") or {}),
                        "refit_on_train_validation": True,
                        "outlier_variant": "outlier_filtered",
                        "removed_source_row_indices": selected_indices,
                    },
                }
            ]
            if progress_callback is not None:
                progress_callback(
                    "Training filtered refit", {"detail": f"{len(selected_indices)} rows removed"}
                )
            try:
                filtered_run = self._train_backend(
                    train_csv=train_csv,
                    output_dir=str(output_dir / "outlier_filtered"),
                    task=task,
                    resolved_parameters={
                        **base_args,
                        **selected_parameters,
                        "split_payload": filtered_payload,
                        "final_refit": True,
                        "refit_on_train_validation": True,
                        "early_stopping_rounds": 0,
                    },
                )
                filtered_run["outlier_variant"] = "outlier_filtered"
                filtered_run["outlier_selected_count"] = len(selected_indices)
                variants.append({"variant_id": "outlier_filtered", "run": filtered_run})
            except Exception as exc:
                variants.append(
                    {
                        "variant_id": "outlier_filtered",
                        "status": "failed",
                        "reason": str(exc),
                    }
                )

        comparison_path = write_outlier_variant_comparison(
            output_dir=outlier_root,
            variants=variants,
            selected_count=len(selected_indices),
        )

        return {
            "summary": {
                **selection_summary,
                "enabled": True,
                "artifacts": artifacts,
                "summary_path": artifacts.get("summary_path"),
                "selection_predictions_path": artifacts.get("selection_predictions_path"),
                "filtered_development_path": artifacts.get("filtered_development_path"),
                "plot_artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key.startswith("outlier_selection_")
                },
                "comparison_path": comparison_path,
                "test_comparison_policy": "descriptive_only_no_automatic_winner",
                "selected_source_row_indices": selected_indices,
                "variants": [
                    {
                        "variant_id": item["variant_id"],
                        "status": item.get("status", "completed"),
                        "reason": item.get("reason"),
                    }
                    for item in variants
                ],
            },
            "variants": variants,
        }

    def _run_cross_validation_outlier_variants(
        self,
        *,
        train_csv: str,
        source_df: pd.DataFrame,
        task: PredictionTaskSpec,
        tuned_splits: List[Dict[str, Any]],
        output_dir: Path,
        training_policy: Dict[str, Any],
        feature_columns: List[str],
        categorical_feature_columns: List[str],
        representation_name: Optional[str],
        applicability_domain_methods: Optional[List[str] | str],
        similarity_top_k_neighbors: int | str | None,
        similarity_threshold_percentile: float | str | None,
        selection_fraction: float,
        progress_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Pool OOF selection rows per repeat and train CV final variants.

        Individual fold studies are needed to find their own HPO configuration,
        but the V1 removal policy is applied once to the pooled OOF rows of a
        repeat.  Each unique winning configuration is then refit on the full
        development subset, both with and without the same pooled removals.
        """
        per_repeat: Dict[int, List[Dict[str, Any]]] = {}
        for item in tuned_splits:
            per_repeat.setdefault(int(item.get("repeat_index") or 1), []).append(item)

        all_variants: List[Dict[str, Any]] = []
        repeat_summaries: List[Dict[str, Any]] = []
        for repeat_index, fold_results in sorted(per_repeat.items()):
            selection_frames: List[pd.DataFrame] = []
            development_indices: set[int] = set()
            outer_test_indices: set[int] = set()
            configuration_inputs: List[Dict[str, Any]] = []
            for fold_result in fold_results:
                analysis = fold_result.get("outlier_analysis") or {}
                selection_path = analysis.get("selection_predictions_path")
                if selection_path and Path(str(selection_path)).exists():
                    selection_frames.append(pd.read_csv(Path(str(selection_path))))
                split = (fold_result.get("selection_split_payload") or [{}])[0]
                development_indices.update(int(value) for value in split.get("train") or [])
                development_indices.update(
                    int(value) for value in (split.get("val") or split.get("validation") or [])
                )
                outer_test_indices.update(int(value) for value in split.get("test") or [])
                configuration_inputs.append(
                    {
                        "parameters": (
                            (fold_result.get("hyperparameter_tuning") or {})
                            .get("best_trial", {})
                            .get("params", {})
                        ),
                        "fold_label": fold_result.get("strategy_label"),
                        "repeat_index": repeat_index,
                        "trial_number": (
                            (fold_result.get("hyperparameter_tuning") or {})
                            .get("best_trial", {})
                            .get("number")
                        ),
                        "objective": (
                            (fold_result.get("hyperparameter_tuning") or {})
                            .get("best_trial", {})
                            .get("objective")
                        ),
                    }
                )
            if not selection_frames or not development_indices:
                repeat_summaries.append(
                    {
                        "repeat_index": repeat_index,
                        "status": "skipped",
                        "reason": "OOF selection predictions are unavailable.",
                    }
                )
                continue
            pooled_rows, selection_summary = select_outliers(
                pd.concat(selection_frames, ignore_index=True),
                task_type=task.task_type,
                selection_fraction=selection_fraction,
            )
            repeat_root = output_dir / "outlier_analysis" / f"repeat_{repeat_index}"
            development = source_df.iloc[sorted(development_indices)].copy()
            development.index = sorted(development_indices)
            artifacts = write_outlier_analysis_artifacts(
                output_dir=repeat_root,
                selection_frame=pooled_rows,
                selection_summary=selection_summary,
                development_frame=development,
                extra_summary={
                    "repeat_index": repeat_index,
                    "selection_model": "pooled_out_of_fold_best_models",
                    "activity_cliff_scope": "development_only_per_fold",
                    "test_rows_used_for_selection": 0,
                },
            )
            selected_indices = [
                int(value)
                for value in pooled_rows.loc[
                    pooled_rows["selected_for_removal"].astype(bool), "source_row_index"
                ].tolist()
            ]
            configurations = deduplicate_parameter_configurations(configuration_inputs)
            repeat_variants: List[Dict[str, Any]] = []
            for configuration in configurations:
                configuration_id = str(configuration["configuration_id"])
                parameters = dict(configuration["parameters"])
                source_folds = list(configuration["source_folds"])
                outer_test_list = sorted(outer_test_indices)
                selected_indices_for_configuration = list(selected_indices)
                base_args = {
                    **{
                        key: value
                        for key, value in training_policy["resolved_parameters"].items()
                        if key not in {"seed_policy", "split_payload", "validation_strategy"}
                    },
                    "feature_columns": feature_columns,
                    "categorical_feature_columns": categorical_feature_columns,
                    "split_type": "final_refit",
                    "random_state": int(
                        training_policy["resolved_parameters"].get("random_state") or 0
                    ),
                    "validation_protocol": "cross_validation",
                    "final_refit": True,
                    "refit_on_train_validation": True,
                    "early_stopping_rounds": 0,
                    "deterministic": True,
                }

                def _fit_variant(
                    variant_id: str,
                    indices: List[int],
                    *,
                    repeat_index: int = repeat_index,
                    configuration_id: str = configuration_id,
                    source_folds: List[str] = source_folds,
                    outer_test_indices: List[int] = outer_test_list,
                    selected_indices: List[int] = selected_indices_for_configuration,
                    base_args: Dict[str, Any] = base_args,
                    parameters: Dict[str, Any] = parameters,
                ) -> Dict[str, Any]:
                    payload = [
                        {
                            "train": indices,
                            **({"test": sorted(outer_test_indices)} if outer_test_indices else {}),
                            "metadata": {
                                "split_type": "cross_validation_final_refit",
                                "repeat_index": repeat_index,
                                "source_folds": source_folds,
                                "outlier_variant": variant_id,
                                "removed_source_row_indices": (
                                    selected_indices if variant_id == "outlier_filtered" else []
                                ),
                            },
                        }
                    ]
                    run = self._train_backend(
                        train_csv=train_csv,
                        output_dir=str(
                            output_dir
                            / "cv_final_variants"
                            / f"repeat_{repeat_index}"
                            / configuration_id
                            / variant_id
                        ),
                        task=task,
                        resolved_parameters={**base_args, **parameters, "split_payload": payload},
                    )
                    run.update(
                        {
                            "outlier_variant": variant_id,
                            "outlier_selected_count": len(selected_indices),
                            "repeat_index": repeat_index,
                            "source_folds": source_folds,
                            "selected_hyperparameters": parameters,
                            "split_payload": payload,
                            "strategy": "cross_validation_final_refit",
                            "strategy_family": "cross_validation",
                            "strategy_label": f"repeat_{repeat_index}_{configuration_id}_{variant_id}",
                        }
                    )
                    run["applicability_domain"] = self._build_applicability_domain(
                        train_csv=train_csv,
                        primary_run=run,
                        primary_output_dir=Path(str(run.get("output_dir") or output_dir)),
                        task=task,
                        feature_columns=feature_columns,
                        feature_space=representation_name,
                        prediction_artifact_paths={"test": run.get("test_predictions_path")},
                        applicability_domain_methods=applicability_domain_methods,
                        similarity_top_k_neighbors=similarity_top_k_neighbors,
                        similarity_threshold_percentile=similarity_threshold_percentile,
                    )
                    return run

                if progress_callback is not None:
                    progress_callback(
                        "Training baseline refit",
                        {"detail": f"repeat {repeat_index}, {configuration_id}"},
                    )
                baseline = _fit_variant("baseline", sorted(development_indices))
                repeat_variants.append(
                    {
                        "variant_id": f"repeat_{repeat_index}_{configuration_id}_baseline",
                        "run": baseline,
                        "repeat_index": repeat_index,
                        "configuration_id": configuration_id,
                        "source_folds": configuration["source_folds"],
                    }
                )
                if selected_indices:
                    if progress_callback is not None:
                        progress_callback(
                            "Training filtered refit",
                            {"detail": f"repeat {repeat_index}, {configuration_id}"},
                        )
                    try:
                        filtered = _fit_variant(
                            "outlier_filtered",
                            [
                                value
                                for value in sorted(development_indices)
                                if value not in set(selected_indices)
                            ],
                        )
                        repeat_variants.append(
                            {
                                "variant_id": f"repeat_{repeat_index}_{configuration_id}_outlier_filtered",
                                "run": filtered,
                                "repeat_index": repeat_index,
                                "configuration_id": configuration_id,
                                "source_folds": configuration["source_folds"],
                            }
                        )
                    except Exception as exc:
                        repeat_variants.append(
                            {
                                "variant_id": f"repeat_{repeat_index}_{configuration_id}_outlier_filtered",
                                "status": "failed",
                                "reason": str(exc),
                                "repeat_index": repeat_index,
                                "configuration_id": configuration_id,
                                "source_folds": configuration["source_folds"],
                            }
                        )
            comparison_path = write_outlier_variant_comparison(
                output_dir=repeat_root,
                variants=repeat_variants,
                selected_count=len(selected_indices),
            )
            repeat_summaries.append(
                {
                    **selection_summary,
                    "repeat_index": repeat_index,
                    "status": "completed",
                    "artifacts": artifacts,
                    "selection_predictions_path": artifacts.get("selection_predictions_path"),
                    "filtered_development_path": artifacts.get("filtered_development_path"),
                    "plot_artifacts": {
                        key: value
                        for key, value in artifacts.items()
                        if key.startswith("outlier_selection_")
                    },
                    "comparison_path": comparison_path,
                    "selected_source_row_indices": selected_indices,
                    "configuration_count": len(configurations),
                }
            )
            all_variants.extend(repeat_variants)
        return {"variants": all_variants, "repeat_summaries": repeat_summaries}

    def _training_defaults_for_profile(self, profile: str) -> Dict[str, Any]:
        base = {
            "split_sizes": [0.8, 0.1, 0.1],
            "split_type": "random",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_samples": 20,
        }
        if profile == "heavy_validation":
            return {
                **base,
                "n_estimators": 1000,
                "early_stopping_rounds": 0,
                "n_jobs": 8,
            }
        if profile == "local_standard":
            return {
                **base,
                "n_estimators": 500,
                "early_stopping_rounds": 0,
                "n_jobs": 4,
            }
        return {
            **base,
            "n_estimators": 300,
            "early_stopping_rounds": 0,
            "n_jobs": 1,
        }

    def _apply_training_profile(
        self,
        resolved_parameters: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        compute_env = self.describe_compute_environment()
        requested_n_jobs = (resolved_parameters or {}).get("n_jobs")

        def _limit(
            profile: str, merged: Dict[str, Any], allow_heavy_compute: bool
        ) -> Dict[str, Any]:
            if allow_heavy_compute:
                if profile == "heavy_validation":
                    merged["n_jobs"] = resolve_backend_n_jobs(
                        compute_env,
                        backend_name="lightgbm",
                        profile=profile,
                        requested_n_jobs=requested_n_jobs,
                    )
                return merged
            merged["n_jobs"] = resolve_backend_n_jobs(
                compute_env,
                backend_name="lightgbm",
                profile=profile,
                requested_n_jobs=requested_n_jobs,
            )
            return merged

        return apply_training_profile(
            resolved_parameters,
            defaults_for_profile=self._training_defaults_for_profile,
            limit_profile_args=_limit,
            compute_environment=compute_env,
            protected_profiles=("heavy_validation",),
        )

    def _materialize_primary_protocol_artifacts(
        self,
        *,
        root_output_dir: Path,
        primary_run: Dict[str, Any],
    ) -> Dict[str, Optional[str]]:
        return materialize_primary_protocol_artifacts(
            root_output_dir=root_output_dir,
            primary_run=primary_run,
            model_filename="best.pkl",
        )

    def _build_applicability_domain(
        self,
        *,
        train_csv: str,
        primary_run: Dict[str, Any],
        primary_output_dir: Path,
        task: PredictionTaskSpec,
        feature_columns: Optional[List[str]] = None,
        feature_space: Optional[str] = None,
        prediction_artifact_paths: Optional[Dict[str, Any]] = None,
        applicability_domain_methods: Optional[List[str] | str] = None,
        similarity_top_k_neighbors: int | str | None = None,
        similarity_threshold_percentile: float | str | None = None,
    ) -> Dict[str, Any]:
        return build_applicability_domain_for_training(
            train_csv=train_csv,
            primary_run=primary_run,
            primary_output_dir=primary_output_dir,
            task=task,
            feature_columns=feature_columns,
            feature_space=feature_space,
            prediction_artifact_paths=prediction_artifact_paths,
            applicability_domain_methods=applicability_domain_methods,
            similarity_top_k_neighbors=similarity_top_k_neighbors,
            similarity_threshold_percentile=similarity_threshold_percentile,
        )

    def _summarize_training_resources(
        self,
        *,
        compute_env: Dict[str, Any],
        effective_train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        n_jobs = effective_train_args.get("n_jobs")
        return {
            "execution_env": compute_env.get("execution_env"),
            "cpu_count": compute_env.get("cpu_count"),
            "gpu_available": compute_env.get("gpu_available"),
            "gpu_count": compute_env.get("gpu_count"),
            "gpu_name": compute_env.get("gpu_name"),
            "memory_gb_total": compute_env.get("memory_gb_total"),
            "batch_size": "N/A",
            "batch_size_reason": "LightGBM native training does not use mini-batches.",
            "workers": f"{n_jobs} (n_jobs)" if n_jobs is not None else "N/A",
            "workers_reason": "LightGBM parallelism is controlled by n_jobs.",
            "n_estimators": effective_train_args.get("n_estimators"),
            "n_jobs": n_jobs,
            "num_leaves": effective_train_args.get("num_leaves"),
            "learning_rate": effective_train_args.get("learning_rate"),
            "device_type": effective_train_args.get("device_type"),
        }

    def _compact_split_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "strategy_label": result.get("strategy_label"),
            "strategy_family": result.get("strategy_family"),
            "backend_split_type": result.get("backend_split_type"),
            "seed": result.get("seed"),
            "metrics": result.get("metrics", {}).get("test") or {},
            "model_path": result.get("model_path"),
            "test_predictions_path": result.get("test_predictions_path"),
            "splits_path": result.get("splits_path"),
            "output_dir": result.get("output_dir"),
            "source_train_count": result.get("source_train_count"),
            "effective_train_count": result.get("effective_train_count"),
            "validation_count": result.get("validation_count"),
            "test_count": result.get("test_count"),
            "has_validation_split": result.get("has_validation_split"),
            "split_metadata": result.get("split_metadata"),
            "removed_from_train_count": result.get("removed_from_train_count", 0),
            "requested_exclusion_count": result.get("requested_exclusion_count", 0),
            "duration_seconds": result.get("duration_seconds"),
        }

    def _variant_summary(
        self,
        *,
        variant: Dict[str, Any],
        split_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        assessment = assess_protocol_results(split_results)
        return {
            "variant_id": variant.get("variant_id"),
            "loop_index": variant.get("loop_index"),
            "removed_tiers": variant.get("removed_tiers") or [],
            "removed_count": variant.get("removed_count", 0),
            "remaining_rows": variant.get("remaining_rows"),
            "filtered_training_csv": variant.get("filtered_training_csv"),
            "training_completed": bool(split_results),
            "split_results": [self._compact_split_result(item) for item in split_results],
            "validation_assessment": assessment,
        }

    @staticmethod
    def _hardest_split_r2(assessment: Dict[str, Any]) -> Optional[float]:
        metrics = assessment.get("governance", {}).get("hardest_split_metrics") or {}
        value = metrics.get("r2")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _variant_comparison_rows(variant_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for variant in variant_summaries:
            for split_result in variant.get("split_results") or []:
                metrics = split_result.get("metrics") or {}
                rows.append(
                    {
                        "variant_id": variant.get("variant_id"),
                        "loop_index": variant.get("loop_index"),
                        "split": split_result.get("strategy_label"),
                        "removed_tiers": variant.get("removed_tiers") or [],
                        "removed_count_dataset": variant.get("removed_count", 0),
                        "requested_exclusion_count": split_result.get(
                            "requested_exclusion_count", 0
                        ),
                        "removed_from_train_count": split_result.get("removed_from_train_count", 0),
                        "source_train_count": split_result.get("source_train_count"),
                        "effective_train_count": split_result.get("effective_train_count"),
                        "validation_count": split_result.get("validation_count"),
                        "test_count": split_result.get("test_count"),
                        "r2": metrics.get("r2"),
                        "rmse": metrics.get("rmse"),
                        "mae": metrics.get("mae"),
                        "mse": metrics.get("mse"),
                        "n": metrics.get("n"),
                    }
                )
        return rows

    @staticmethod
    def _format_activity_cliff_value(value: Any) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4g}"
        if isinstance(value, list):
            return ", ".join(str(item) for item in value) if value else "-"
        return str(value)

    @classmethod
    def _activity_cliff_canonical_section_markdown(
        cls,
        *,
        activity_cliffs: Dict[str, Any],
        variant_comparison_rows: List[Dict[str, Any]],
        recommended_variant_text: Optional[str],
        neighborhood_policy_text: str,
        priority_counts_text: str,
    ) -> str:
        params = activity_cliffs.get("index_parameters") or {}
        mode = str(activity_cliffs.get("mode") or "standard")
        index_display = "SALI (Structure-Activity Landscape Index)"
        fingerprint = params.get("fingerprint")
        fingerprint_radius = cls._format_activity_cliff_value(params.get("fingerprint_radius"))
        fingerprint_dimensions = params.get("fingerprint_dimensions")
        fingerprint_bits = params.get("fingerprint_bits")
        similarity_metric = params.get("similarity_metric")
        if fingerprint == "morgan_count":
            fingerprint_text = (
                "Morgan count fingerprints, "
                f"rayon {fingerprint_radius}, "
                f"{cls._format_activity_cliff_value(fingerprint_dimensions)} dimensions"
            )
            similarity_text = "Tanimoto ponderee par les comptes"
        else:
            fingerprint_size_text = (
                f"{cls._format_activity_cliff_value(fingerprint_bits)} bits"
                if fingerprint_bits is not None
                else f"{cls._format_activity_cliff_value(fingerprint_dimensions)} dimensions"
            )
            fingerprint_text = (
                f"{cls._format_activity_cliff_value(fingerprint)}, "
                f"rayon {fingerprint_radius}, {fingerprint_size_text}"
            )
            similarity_text = cls._format_activity_cliff_value(similarity_metric)
        lines = [
            "Activity cliffs",
            "",
            f"L'analyse des falaises d'activite a ete realisee avec l'indice {index_display}.",
            "",
            "Parametres de voisinage :",
            f"- Empreinte : {fingerprint_text}",
            f"- Similarite : {similarity_text}",
            f"- Seuil de similarite : {cls._format_activity_cliff_value(params.get('similarity_threshold'))}",
            f"- Nombre maximal de voisins : {cls._format_activity_cliff_value(params.get('top_k_neighbors'))}",
            f"- Seuil de signalement : {cls._format_activity_cliff_value(params.get('flag_threshold'))}",
            f"- Normalisation : {cls._format_activity_cliff_value(params.get('normalization'))}",
            "",
            "Resultats d'annotation :",
            f"- Composes analyses : {cls._format_activity_cliff_value(activity_cliffs.get('ranked_molecule_count'))}",
            f"- Composes signales : {cls._format_activity_cliff_value(activity_cliffs.get('flagged_count'))}",
            f"- Repartition des priorites : {priority_counts_text}",
            f"- Mode : {mode}",
        ]
        if mode != "with_feedback_loops":
            lines.extend(
                [
                    "",
                    "Aucune boucle de retroaction n'a ete demandee. Aucun compose n'a ete retire ; "
                    "l'annotation Activity Cliffs enrichit uniquement le rapport et les artefacts.",
                ]
            )
            return "\n".join(lines)

        lines.extend(
            [
                "",
                "Les boucles de retroaction ont ete evaluees avec des holdouts de validation et de test fixes "
                "et non filtres. Seul l'ensemble d'entrainement est filtre selon les paliers SALI.",
                "",
                "Variantes comparees :",
                "| Variante | Split | Paliers retires | Retires dataset | Retires train | Train effectif | Validation | Test | RMSE | MAE | R2 | MSE |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in variant_comparison_rows:
            lines.append(
                "| "
                f"{cls._format_activity_cliff_value(row.get('variant_id'))} | "
                f"{cls._format_activity_cliff_value(row.get('split'))} | "
                f"{cls._format_activity_cliff_value(row.get('removed_tiers'))} | "
                f"{cls._format_activity_cliff_value(row.get('removed_count_dataset'))} | "
                f"{cls._format_activity_cliff_value(row.get('removed_from_train_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('effective_train_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('validation_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('test_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('rmse'))} | "
                f"{cls._format_activity_cliff_value(row.get('mae'))} | "
                f"{cls._format_activity_cliff_value(row.get('r2'))} | "
                f"{cls._format_activity_cliff_value(row.get('mse'))} |"
            )
        if recommended_variant_text:
            lines.extend(["", recommended_variant_text])
        lines.extend(["", f"Politique de voisinage canonique : {neighborhood_policy_text}"])
        return "\n".join(lines)

    @classmethod
    def _activity_cliff_validation_metrics_markdown(
        cls,
        *,
        variant_comparison_rows: List[Dict[str, Any]],
        recommended_variant: Optional[str],
    ) -> Optional[str]:
        if not variant_comparison_rows:
            return None
        lines = [
            "Metriques de validation par variante Activity Cliffs",
            "",
            "Toutes les variantes sont reportees avec les memes holdouts de validation et de test non filtres.",
            "",
            "| Variante | Split | Retires train | Train effectif | R2 | RMSE | MAE | MSE | n |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in variant_comparison_rows:
            marker = (
                " (recommandee)"
                if recommended_variant and row.get("variant_id") == recommended_variant
                else ""
            )
            lines.append(
                "| "
                f"{cls._format_activity_cliff_value(row.get('variant_id'))}{marker} | "
                f"{cls._format_activity_cliff_value(row.get('split'))} | "
                f"{cls._format_activity_cliff_value(row.get('removed_from_train_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('effective_train_count'))} | "
                f"{cls._format_activity_cliff_value(row.get('r2'))} | "
                f"{cls._format_activity_cliff_value(row.get('rmse'))} | "
                f"{cls._format_activity_cliff_value(row.get('mae'))} | "
                f"{cls._format_activity_cliff_value(row.get('mse'))} | "
                f"{cls._format_activity_cliff_value(row.get('n'))} |"
            )
        if recommended_variant:
            lines.extend(
                [
                    "",
                    f"Variante recommandee pour le statut final : {recommended_variant}.",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _activity_cliff_reporting_handoff(
        *,
        activity_cliffs: Dict[str, Any],
        variant_comparison_rows: List[Dict[str, Any]],
        recommended_variant: Optional[str],
        recommendation_reason: Optional[str],
    ) -> Dict[str, Any]:
        params = activity_cliffs.get("index_parameters") or {}
        priority_counts = activity_cliffs.get("priority_counts") or {}
        fingerprint = params.get("fingerprint")
        if fingerprint == "morgan_count":
            fingerprint_policy = (
                "Morgan count fingerprints, "
                f"radius={params.get('fingerprint_radius')}, "
                f"dimensions={params.get('fingerprint_dimensions')}, "
                "similarity_metric=count_tanimoto"
            )
        else:
            fingerprint_policy = (
                f"Fingerprint={fingerprint}, radius={params.get('fingerprint_radius')}, "
                f"bits={params.get('fingerprint_bits')}, "
                f"similarity_metric={params.get('similarity_metric')}"
            )
        neighborhood_policy_text = (
            f"{fingerprint_policy}, "
            f"similarity_threshold={params.get('similarity_threshold')}, "
            f"top-k neighbors={params.get('top_k_neighbors')}, "
            f"flag_threshold={params.get('flag_threshold')}, "
            f"normalization={params.get('normalization')}."
        )
        priority_counts_text = (
            f"none={priority_counts.get('none', 0)}, low={priority_counts.get('low', 0)}, "
            f"medium={priority_counts.get('medium', 0)}, high={priority_counts.get('high', 0)}"
        )
        recommended_variant_text = (
            f"Variante recommandee : {recommended_variant}. {recommendation_reason}"
            if recommended_variant and recommendation_reason
            else None
        )
        canonical_section_markdown = LightGBMToolkit._activity_cliff_canonical_section_markdown(
            activity_cliffs=activity_cliffs,
            variant_comparison_rows=variant_comparison_rows,
            recommended_variant_text=recommended_variant_text,
            neighborhood_policy_text=neighborhood_policy_text,
            priority_counts_text=priority_counts_text,
        )
        validation_metrics_markdown = LightGBMToolkit._activity_cliff_validation_metrics_markdown(
            variant_comparison_rows=variant_comparison_rows,
            recommended_variant=recommended_variant,
        )
        return {
            "mode": activity_cliffs.get("mode"),
            "index_name": activity_cliffs.get("index_name"),
            "index_display_name": "SALI (Structure-Activity Landscape Index)",
            "neighborhood_policy_text": neighborhood_policy_text,
            "flagged_count": activity_cliffs.get("flagged_count"),
            "priority_counts_text": priority_counts_text,
            "variant_comparison_rows": variant_comparison_rows,
            "recommended_variant": recommended_variant,
            "recommended_variant_text": recommended_variant_text,
            "holdout_policy_text": (
                "Validation and test holdouts are fixed and non-filtered; activity-cliff tiers are removed "
                "from the training split only."
            ),
            "canonical_section_markdown": canonical_section_markdown,
            "validation_metrics_markdown": validation_metrics_markdown,
        }

    def _attach_activity_cliff_variant_training(
        self,
        *,
        activity_cliffs: Dict[str, Any],
        baseline_split_results: List[Dict[str, Any]],
        variant_split_results: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        if not activity_cliffs.get("enabled"):
            return activity_cliffs
        if int(activity_cliffs.get("feedback_loops_requested") or 0) <= 0:
            activity_cliffs["reporting_handoff"] = self._activity_cliff_reporting_handoff(
                activity_cliffs=activity_cliffs,
                variant_comparison_rows=[],
                recommended_variant=None,
                recommendation_reason=None,
            )
            return activity_cliffs

        variants = list(activity_cliffs.get("variants") or [])
        by_variant = {
            "baseline_loop_0": baseline_split_results,
            **variant_split_results,
        }
        variant_summaries: List[Dict[str, Any]] = []
        for variant in variants:
            variant_id = str(variant.get("variant_id"))
            split_results = by_variant.get(variant_id, [])
            summary = self._variant_summary(variant=variant, split_results=split_results)
            variant["training_result"] = summary
            variant_summaries.append(summary)

        comparable = [
            item
            for item in variant_summaries
            if item.get("training_completed")
            and self._hardest_split_r2(item.get("validation_assessment") or {}) is not None
        ]
        recommended_variant = None
        recommendation_reason = None
        if comparable:
            comparable.sort(
                key=lambda item: (
                    self._hardest_split_r2(item.get("validation_assessment") or {})
                    or float("-inf"),
                    -int(item.get("loop_index") or 0),
                ),
                reverse=True,
            )
            recommended_variant = comparable[0].get("variant_id")
            best_r2 = self._hardest_split_r2(comparable[0].get("validation_assessment") or {})
            recommendation_reason = (
                "Selection par meilleur R2 sur le split le plus difficile, avec holdouts fixes "
                f"(R2={best_r2:.3f})."
                if best_r2 is not None
                else None
            )

        activity_cliffs["variant_training"] = variant_summaries
        variant_comparison_rows = self._variant_comparison_rows(variant_summaries)
        activity_cliffs["variant_comparison_table"] = variant_comparison_rows
        activity_cliffs["recommended_variant"] = recommended_variant
        activity_cliffs["recommendation_reason"] = recommendation_reason
        activity_cliffs["reporting_handoff"] = self._activity_cliff_reporting_handoff(
            activity_cliffs=activity_cliffs,
            variant_comparison_rows=variant_comparison_rows,
            recommended_variant=recommended_variant,
            recommendation_reason=recommendation_reason,
        )
        activity_cliffs["loop_training_policy"] = {
            "baseline_trained": True,
            "train_filtering": "remove selected activity-cliff tiers from train split only",
            "holdout_policy": "validation and test indices remain fixed and non-filtered",
            "comparison_metric": "hardest_split_r2",
        }
        return activity_cliffs

    def describe_lightgbm_backend(self) -> Dict[str, Any]:
        """Describe the LightGBM backend defaults and current runtime support."""
        description = self.backend.describe_environment()
        description.update(
            {
                "default_task_type": "regression",
                "default_target_scope": "single_target",
                "automatic_representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
                "default_representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
            }
        )
        return description

    def describe_lightgbm_environment(self) -> Dict[str, Any]:
        """Alias for backend/environment inspection."""
        return self.describe_lightgbm_backend()

    def validate_lightgbm_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a trained LightGBM model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "model_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
        }

    def _normalize_json_list_argument(
        self,
        value: Optional[List[Any] | str],
        *,
        argument_name: str,
    ) -> Optional[List[Any]]:
        return normalize_json_list_argument(
            value,
            argument_name=argument_name,
            coerce_numbers=argument_name == "split_sizes",
        )

    def _train_tuned_holdout(
        self,
        *,
        train_csv: str,
        resolved_output_dir: str,
        task: PredictionTaskSpec,
        protocol_policy: Dict[str, Any],
        training_policy: Dict[str, Any],
        split_source_df: pd.DataFrame,
        split_run: Dict[str, Any],
        feature_columns: List[str],
        categorical_feature_columns: List[str],
        representation_name: Optional[str],
        direct_model_args: Dict[str, Any],
        raw_tuning_config: Optional[Dict[str, Any]],
        applicability_domain_methods: Optional[List[str] | str],
        similarity_top_k_neighbors: int | str | None,
        similarity_threshold_percentile: float | str | None,
        outlier_config: OutlierAnalysisConfig,
        outlier_skip_reason: Optional[str],
        activity_args: Dict[str, Any],
        selection_only: bool = False,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Tune one fixed validation split and optionally create its final refit.

        Cross-validation passes ``selection_only=True``.  In that mode this
        method produces only the temporary best-model validation predictions
        and outlier-selection artifacts; the pooled OOF analysis owns every
        final refit.
        """
        total_started_at = project_now()
        if progress_callback is not None:
            progress_callback(
                "Hyperparameter optimization", {"detail": "Preparing validation split"}
            )
        root_output_path = Path(resolved_output_dir)
        effective_feature_columns = list(feature_columns) or [
            str(column)
            for column in split_source_df.columns
            if column not in set(task.target_columns + task.smiles_columns)
            and pd.api.types.is_numeric_dtype(split_source_df[column])
        ]
        if not effective_feature_columns:
            raise HyperparameterTuningError(
                "LightGBM tuning requires numeric feature columns or a prepared tabular representation."
            )
        split_sizes = split_run.get("split_sizes") or training_policy["resolved_parameters"].get(
            "split_sizes"
        )
        split_payload = split_run.get("split_payload") or build_qsar_split_payload(
            df=split_source_df,
            split_type=split_run["backend_split_type"],
            split_sizes=split_sizes,
            random_state=int(split_run["seed"]),
            smiles_column="smiles" if "smiles" in split_source_df.columns else None,
            feature_columns=effective_feature_columns,
        )
        split_indices = split_payload[0]
        train_indices = [int(item) for item in split_indices.get("train") or []]
        validation_indices = [int(item) for item in split_indices.get("val") or []]
        test_indices = [int(item) for item in split_indices.get("test") or []]
        if not train_indices or not validation_indices:
            raise HyperparameterTuningError(
                "Hyperparameter tuning requires non-empty train and validation splits."
            )

        model_parameter_names = {
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "max_depth",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "min_split_gain",
            "boosting_type",
        }
        fixed_model_args = {
            key: value for key, value in direct_model_args.items() if key in model_parameter_names
        }
        effective_tuning_request = dict(raw_tuning_config or {})
        effective_tuning_request.setdefault("seed", protocol_policy["seed_policy"]["model_seed"])
        tuning_config = normalize_tuning_config(
            effective_tuning_request,
            backend_name="lightgbm",
            task_type=task.task_type,
            eligible=True,
            fixed_parameters=fixed_model_args,
        )
        if tuning_config is None:
            raise HyperparameterTuningError(
                "Internal error: tuned holdout requires an enabled config."
            )
        if (
            int(direct_model_args.get("early_stopping_rounds") or 0) > 0
            and "n_estimators" in tuning_config.parameters
        ):
            raise HyperparameterTuningError(
                "early_stopping_rounds cannot be used while n_estimators is optimized."
            )
        requested_early_stopping_rounds = int(direct_model_args.get("early_stopping_rounds") or 0)

        feature_frame = split_source_df[effective_feature_columns].copy().reset_index(drop=True)
        base_args = {
            **{
                key: value
                for key, value in training_policy["resolved_parameters"].items()
                if key not in {"seed_policy", "split_payload", "validation_strategy"}
            },
            "feature_columns": effective_feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "split_sizes": split_sizes,
            "split_type": split_run["backend_split_type"],
            "random_state": int(protocol_policy["seed_policy"]["model_seed"]),
            "validation_protocol": protocol_policy["protocol"],
            "early_stopping_rounds": requested_early_stopping_rounds,
            "deterministic": True,
        }
        temporary_payload = [{"train": train_indices, "val": validation_indices}]
        with tempfile.TemporaryDirectory(prefix="qsaria_lightgbm_tuning_") as temporary_dir:
            temporary_ad = fit_modern_applicability_domain(
                feature_frame=feature_frame.iloc[train_indices].reset_index(drop=True),
                feature_columns=effective_feature_columns,
                output_dir=Path(temporary_dir) / "applicability_domain",
                model_id="lightgbm_tuning",
                feature_space=representation_name or "tabular",
                representation_name=representation_name,
                methods=applicability_domain_methods,
                random_state=int(protocol_policy["seed_policy"]["model_seed"]),
                all_feature_frame=feature_frame,
                train_indices=train_indices,
                split_indices=split_indices,
                similarity_top_k_neighbors=similarity_top_k_neighbors,
                similarity_threshold_percentile=similarity_threshold_percentile,
            )
            validation_ad = score_modern_applicability_domain(
                feature_frame=feature_frame.iloc[validation_indices].reset_index(drop=True),
                applicability_domain=temporary_ad,
                row_indices=validation_indices,
            )
            validation_status = _ad_status_series(validation_ad)
            if validation_status is None:
                raise HyperparameterTuningError(
                    "Could not score the validation applicability domain."
                )
            if tuning_config.objective.subset == "in_domain" and not bool(
                validation_status.eq("in_domain").any()
            ):
                raise HyperparameterTuningError(
                    "The validation split contains no in-domain molecule; choose objective.subset='all' "
                    "or revise the applicability-domain configuration."
                )

            def evaluate(candidate_parameters: Dict[str, Any]) -> Dict[str, Any]:
                candidate = self._train_backend(
                    train_csv=train_csv,
                    output_dir=temporary_dir,
                    task=task,
                    resolved_parameters={
                        **base_args,
                        **candidate_parameters,
                        "split_payload": temporary_payload,
                        "persist_artifacts": False,
                        "return_prediction_frames": True,
                    },
                )
                prediction_frame = candidate.get("validation_prediction_frame")
                if not isinstance(prediction_frame, pd.DataFrame) or prediction_frame.empty:
                    raise RuntimeError("LightGBM candidate did not return validation predictions.")
                status = validation_status.reset_index(drop=True)
                metrics_by_subset: Dict[str, Dict[str, Any]] = {
                    "all": dict((candidate.get("metrics") or {}).get("validation") or {})
                }
                for subset, mask in {
                    "in_domain": status.eq("in_domain"),
                    "out_of_domain": status.ne("in_domain"),
                }.items():
                    selected = prediction_frame.loc[mask.to_numpy()].reset_index(drop=True)
                    if selected.empty:
                        metrics_by_subset[subset] = {}
                    elif task.task_type == "regression":
                        metrics_by_subset[subset] = compute_regression_metrics(
                            selected["y_true"],
                            selected["y_pred"],
                            target_column=task.target_columns[0],
                        )
                    else:
                        metrics_by_subset[subset] = compute_classification_metrics(
                            selected["y_true"],
                            selected["y_pred"],
                            class_labels=candidate.get("class_labels") or None,
                            target_column=task.target_columns[0],
                        )
                selected_metrics = metrics_by_subset.get(tuning_config.objective.subset) or {}
                score = selected_metrics.get(tuning_config.objective.metric)
                if score is None:
                    raise RuntimeError(
                        f"Objective {tuning_config.objective.metric} is unavailable on "
                        f"{tuning_config.objective.subset}."
                    )
                return {
                    "objective": float(score),
                    "metrics": metrics_by_subset,
                    "diagnostics": {
                        "validation_count": len(validation_indices),
                        "in_domain_count": int(status.eq("in_domain").sum()),
                        "out_of_domain_count": int(status.ne("in_domain").sum()),
                        "in_domain_coverage": float(status.eq("in_domain").mean()),
                        "applicability_domain": validation_ad.get("summary") or {},
                    },
                }

            def report_trial_progress(payload: Dict[str, Any]) -> None:
                if progress_callback is None:
                    return
                trial_index = payload.get("trial_index")
                total_trials = payload.get("total_trials")
                detail = (
                    f"trial {trial_index} of {total_trials}"
                    if trial_index and total_trials
                    else "evaluating candidate"
                )
                progress_callback(
                    "Hyperparameter optimization",
                    {
                        "detail": detail,
                        "tuning_event": payload.get("event"),
                        "trial_index": trial_index,
                        "total_trials": total_trials,
                    },
                )

            summary = (
                LightGBMOptunaAdapter()
                .run(
                    config=tuning_config,
                    fixed_parameters=fixed_model_args,
                    evaluate=evaluate,
                    progress_callback=report_trial_progress,
                )
                .as_dict()
            )

        summary["seed"] = tuning_config.seed
        summary["contract_version"] = HYPERPARAMETER_CONTRACT_VERSION
        try:
            summary["engine_version"] = importlib.metadata.version("optuna")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - guarded by adapter
            summary["engine_version"] = None
        summary["sampler"] = tuning_sampler_metadata(
            str(tuning_config.engine),
            n_startup_trials=min(10, tuning_config.n_trials),
        )
        summary["pruner"] = {"name": "NopPruner", "enabled": False}

        summary_path = root_output_path / "hyperparameter_tuning_summary.json"
        summary["summary_path"] = str(summary_path)
        tuning_plot_path = build_tuning_progress_plot(
            summary,
            output_dir=root_output_path / "artifacts" / "plots",
        )
        tuning_plot_artifacts = (
            {"hyperparameter_tuning_progress": tuning_plot_path} if tuning_plot_path else {}
        )
        if tuning_plot_artifacts:
            summary["plot_artifacts"] = tuning_plot_artifacts
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if summary.get("status") == "failed":
            raise HyperparameterTuningError(summary.get("reason") or "LightGBM tuning failed.")
        best_trial = summary.get("best_trial") or {}
        selected_parameters = dict(best_trial.get("params") or fixed_model_args)
        selection_prediction_frame: Optional[pd.DataFrame] = None
        if outlier_config.enabled and outlier_skip_reason is None:
            if progress_callback is not None:
                progress_callback(
                    "Identifying validation outliers", {"detail": "fitting selected model"}
                )
            with tempfile.TemporaryDirectory(prefix="qsaria_lightgbm_selection_") as selection_dir:
                selection_run = self._train_backend(
                    train_csv=train_csv,
                    output_dir=selection_dir,
                    task=task,
                    resolved_parameters={
                        **base_args,
                        **selected_parameters,
                        "split_payload": [{"train": train_indices, "val": validation_indices}],
                        "persist_artifacts": False,
                        "return_prediction_frames": True,
                        "early_stopping_rounds": 0,
                    },
                )
                candidate_frame = selection_run.get("validation_prediction_frame")
                if not isinstance(candidate_frame, pd.DataFrame) or candidate_frame.empty:
                    raise HyperparameterTuningError(
                        "The selected LightGBM model did not expose validation predictions for outlier analysis."
                    )
                selection_prediction_frame = candidate_frame.copy()
        if selection_only:
            outlier_study: Dict[str, Any] = {
                "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
                "config": outlier_config.as_dict(),
                "status": "skipped" if outlier_skip_reason else "pending",
                "reason": outlier_skip_reason,
            }
            if selection_prediction_frame is not None:
                study = self._run_outlier_refits(
                    train_csv=train_csv,
                    split_source_df=split_source_df,
                    task=task,
                    split_payload=split_payload,
                    selection_prediction_frame=selection_prediction_frame,
                    output_dir=root_output_path,
                    base_args=base_args,
                    selected_parameters=selected_parameters,
                    feature_columns=effective_feature_columns,
                    representation_name=representation_name,
                    activity_args=activity_args,
                    applicability_domain_methods=applicability_domain_methods,
                    similarity_top_k_neighbors=similarity_top_k_neighbors,
                    similarity_threshold_percentile=similarity_threshold_percentile,
                    selection_fraction=outlier_config.selection_fraction,
                    fit_variants=False,
                    progress_callback=progress_callback,
                    fold_label=str(split_run.get("label") or "cross_validation_fold"),
                    repeat_index=split_run.get("repeat_index"),
                )
                outlier_study = study["summary"]
                outlier_study["status"] = "completed"
            selection_metrics = (best_trial.get("metrics") or {}).get("all") or {}
            return {
                "output_dir": resolved_output_dir,
                "strategy": "cross_validation_selection",
                "strategy_family": "cross_validation",
                "strategy_label": str(split_run.get("label") or "cross_validation_fold"),
                "backend_split_type": split_run["backend_split_type"],
                "seed": split_run["seed"],
                "metrics": {"validation": selection_metrics},
                "selection_validation": {
                    "metrics": best_trial.get("metrics") or {},
                    "diagnostics": best_trial.get("diagnostics") or {},
                },
                "selected_hyperparameters": selected_parameters,
                "hyperparameter_tuning": summary,
                "hyperparameter_tuning_summary_path": str(summary_path),
                "catalog_hyperparameter_tuning": tuning_metadata_for_catalog(summary),
                "plot_artifacts": {
                    **tuning_plot_artifacts,
                    **dict(outlier_study.get("plot_artifacts") or {}),
                },
                "outlier_analysis": outlier_study,
                "validation_protocol": protocol_policy["protocol"],
                "validation_strategy": protocol_policy.get("validation_strategy"),
                "validation_strategy_type": protocol_policy.get("validation_strategy_type"),
                "seed_policy": protocol_policy["seed_policy"],
                "training_profile": training_policy["training_profile"],
                "compute_environment": training_policy["compute_environment"],
            }
        final_split_payload = [
            {
                "train": [*train_indices, *validation_indices],
                "test": test_indices,
                "metadata": {
                    **dict(split_indices.get("metadata") or {}),
                    "refit_on_train_validation": True,
                },
            }
        ]
        if progress_callback is not None:
            progress_callback("Training baseline refit", {"detail": "train + validation"})
        final_run = self._train_backend(
            train_csv=train_csv,
            output_dir=resolved_output_dir,
            task=task,
            resolved_parameters={
                **base_args,
                **selected_parameters,
                "split_payload": final_split_payload,
                "final_refit": True,
                "early_stopping_rounds": 0,
                "refit_on_train_validation": True,
            },
        )
        final_run.update(
            {
                "strategy": "tuned_refit",
                "strategy_family": "tuned_refit",
                "strategy_label": "tuned_refit",
                "backend_split_type": split_run["backend_split_type"],
                "seed": split_run["seed"],
                "validation_protocol": protocol_policy["protocol"],
                "output_dir": resolved_output_dir,
                "split_payload": final_split_payload,
            }
        )
        outlier_study: Dict[str, Any] = {
            "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
            "config": outlier_config.as_dict(),
            "status": "skipped" if outlier_skip_reason else "pending",
            "reason": outlier_skip_reason,
        }
        outlier_variants: List[Dict[str, Any]] = []
        if selection_prediction_frame is not None:
            study = self._run_outlier_refits(
                train_csv=train_csv,
                split_source_df=split_source_df,
                task=task,
                split_payload=split_payload,
                selection_prediction_frame=selection_prediction_frame,
                output_dir=root_output_path,
                base_args=base_args,
                selected_parameters=selected_parameters,
                feature_columns=effective_feature_columns,
                representation_name=representation_name,
                activity_args=activity_args,
                applicability_domain_methods=applicability_domain_methods,
                similarity_top_k_neighbors=similarity_top_k_neighbors,
                similarity_threshold_percentile=similarity_threshold_percentile,
                selection_fraction=outlier_config.selection_fraction,
                baseline_run=final_run,
                progress_callback=progress_callback,
                fold_label=str(split_run.get("label") or "holdout"),
            )
            outlier_study = study["summary"]
            outlier_variants = study["variants"]
            outlier_study["status"] = "completed"
        if progress_callback is not None:
            progress_callback("Evaluating final test set", {})
        root_artifacts = self._materialize_primary_protocol_artifacts(
            root_output_dir=root_output_path,
            primary_run=final_run,
        )
        ad_summary = self._build_applicability_domain(
            train_csv=train_csv,
            primary_run=final_run,
            primary_output_dir=root_output_path,
            task=task,
            feature_columns=effective_feature_columns,
            feature_space=representation_name,
            prediction_artifact_paths={"test": root_artifacts.get("test_predictions_path")},
            applicability_domain_methods=applicability_domain_methods,
            similarity_top_k_neighbors=similarity_top_k_neighbors,
            similarity_threshold_percentile=similarity_threshold_percentile,
        )
        if outlier_variants:
            for variant in outlier_variants:
                run = variant.get("run")
                if not isinstance(run, dict):
                    continue
                if variant.get("variant_id") == "baseline":
                    run["applicability_domain"] = ad_summary
                    continue
                variant_output_dir = Path(str(run.get("output_dir") or root_output_path))
                run["applicability_domain"] = self._build_applicability_domain(
                    train_csv=train_csv,
                    primary_run=run,
                    primary_output_dir=variant_output_dir,
                    task=task,
                    feature_columns=effective_feature_columns,
                    feature_space=representation_name,
                    prediction_artifact_paths={"test": run.get("test_predictions_path")},
                    applicability_domain_methods=applicability_domain_methods,
                    similarity_top_k_neighbors=similarity_top_k_neighbors,
                    similarity_threshold_percentile=similarity_threshold_percentile,
                )
        standard_plot_artifacts = build_training_plots_if_possible(
            train_csv=train_csv,
            split_results=[final_run],
            primary_run=final_run,
            root_artifacts=root_artifacts,
            root_output_dir=root_output_path,
            target_column=task.target_columns[0] if task.target_columns else None,
            task_type=task.task_type,
        )
        outlier_plot_artifacts = dict(outlier_study.get("plot_artifacts") or {})
        plot_artifacts = {
            **standard_plot_artifacts,
            **tuning_plot_artifacts,
            **outlier_plot_artifacts,
        }
        selection_metrics = (best_trial.get("metrics") or {}).get("all") or {}
        final_metrics = dict(final_run.get("metrics") or {})
        final_metrics["validation"] = selection_metrics
        total_completed_at = project_now()
        training_durations = summarize_training_durations(
            split_results=[final_run],
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )
        trial_duration_seconds = round(
            sum(
                float(trial["duration_seconds"])
                for trial in summary.get("trials") or []
                if trial.get("duration_seconds") is not None
            ),
            3,
        )
        training_durations["hyperparameter_tuning"] = {
            "engine": summary.get("engine"),
            "requested_trials": summary.get("requested_trials"),
            "completed_trials": summary.get("completed_trials"),
            "failed_trials": summary.get("failed_trials"),
            "trial_duration_seconds": trial_duration_seconds,
        }
        result = dict(final_run)
        result.update(
            {
                "model_path": root_artifacts.get("best_model_path") or final_run.get("model_path"),
                "best_model_path": root_artifacts.get("best_model_path")
                or final_run.get("model_path"),
                "validation_predictions_path": None,
                "test_predictions_path": root_artifacts.get("test_predictions_path"),
                "config_path": root_artifacts.get("config_path") or final_run.get("config_path"),
                "splits_path": root_artifacts.get("splits_path") or final_run.get("splits_path"),
                "metrics": final_metrics,
                "selection_validation": {
                    "metrics": best_trial.get("metrics") or {},
                    "diagnostics": best_trial.get("diagnostics") or {},
                },
                "early_stopping_final_refit_note": (
                    "Disabled only for the final 90% refit because that fit has no validation split."
                    if requested_early_stopping_rounds > 0
                    else None
                ),
                "hyperparameter_tuning": summary,
                "hyperparameter_tuning_summary_path": str(summary_path),
                "catalog_hyperparameter_tuning": tuning_metadata_for_catalog(summary),
                "plot_artifacts": plot_artifacts,
                "applicability_domain": ad_summary,
                "outlier_analysis": outlier_study,
                "outlier_model_variants": outlier_variants,
                "validation_protocol": protocol_policy["protocol"],
                "validation_protocol_reason": protocol_policy["reason"],
                "validation_strategy": protocol_policy.get("validation_strategy"),
                "validation_strategy_type": protocol_policy.get("validation_strategy_type"),
                "seed_policy": protocol_policy["seed_policy"],
                "reproducibility": seed_policy_reproducibility_metadata(
                    protocol_policy["seed_policy"]
                ),
                "training_profile": training_policy["training_profile"],
                "compute_environment": training_policy["compute_environment"],
                "effective_train_args": final_run.get("effective_train_args") or {},
                "split_results": [final_run],
                "baseline_split_results": [final_run],
                "validation_assessment": assess_protocol_results([final_run]),
                "training_durations": training_durations,
                "catalog_model_policy": (
                    "outlier_variants_no_test_winner"
                    if outlier_variants
                    else "tuned_final_refit_only"
                ),
                "summary_path": str(root_output_path / "cs_copilot_training_summary.json"),
                "canonical_summary_path": str(
                    root_output_path / "cs_copilot_training_summary.json"
                ),
                "train_csv": train_csv,
                "target_columns": list(task.target_columns),
                "feature_columns": effective_feature_columns,
                "categorical_feature_columns": categorical_feature_columns,
            }
        )
        write_training_summary(Path(result["summary_path"]), result)
        return result

    def train_lightgbm_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
        representation_name: Optional[str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
        validation_protocol: Optional[str] = None,
        validation_strategy: Optional[Dict[str, Any]] = None,
        split_type: str = "random",
        split_sizes: Optional[List[float] | str] = None,
        random_state: Optional[int] = None,
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        applicability_domain_methods: Optional[List[str] | str] = None,
        similarity_top_k_neighbors: int | str | None = None,
        similarity_threshold_percentile: float | str | None = None,
        hyperparameter_tuning: Optional[Dict[str, Any]] = None,
        outlier_analysis: Optional[Dict[str, Any]] = None,
        resolved_parameters: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
        bundle_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train a LightGBM regressor with QSAR validation protocols."""
        normalized_target_columns = (
            self._normalize_json_list_argument(
                target_columns,
                argument_name="target_columns",
            )
            or []
        )
        normalized_feature_columns = self._normalize_json_list_argument(
            feature_columns,
            argument_name="feature_columns",
        )
        normalized_categorical_feature_columns = self._normalize_json_list_argument(
            categorical_feature_columns,
            argument_name="categorical_feature_columns",
        )
        normalized_split_sizes = self._normalize_json_list_argument(
            split_sizes,
            argument_name="split_sizes",
        )

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)

        requested_resolved_parameters, extra_activity_args = split_activity_cliff_args(
            resolved_parameters
        )
        requested_hyperparameter_tuning = (
            hyperparameter_tuning
            if hyperparameter_tuning is not None
            else requested_resolved_parameters.pop("hyperparameter_tuning", None)
        )
        requested_outlier_analysis = (
            outlier_analysis
            if outlier_analysis is not None
            else requested_resolved_parameters.pop("outlier_analysis", None)
        )
        requested_validation_strategy = (
            validation_strategy
            if validation_strategy is not None
            else requested_resolved_parameters.pop("validation_strategy", None)
        )
        requested_ad_methods = (
            applicability_domain_methods
            if applicability_domain_methods is not None
            else requested_resolved_parameters.pop("applicability_domain_methods", None)
        )
        requested_similarity_top_k = (
            similarity_top_k_neighbors
            if similarity_top_k_neighbors is not None
            else requested_resolved_parameters.pop("similarity_top_k_neighbors", None)
        )
        requested_similarity_percentile = (
            similarity_threshold_percentile
            if similarity_threshold_percentile is not None
            else requested_resolved_parameters.pop("similarity_threshold_percentile", None)
        )
        activity_args = {
            "activity_cliff_index": activity_cliff_index,
            "activity_cliff_feedback": activity_cliff_feedback,
            "activity_cliff_feedback_loops": activity_cliff_feedback_loops,
            "activity_cliff_similarity_threshold": activity_cliff_similarity_threshold,
            "activity_cliff_top_k_neighbors": activity_cliff_top_k_neighbors,
            "activity_cliff_flag_threshold": activity_cliff_flag_threshold,
            **extra_activity_args,
        }
        requested_resolved_parameters.setdefault("feature_columns", normalized_feature_columns)
        requested_resolved_parameters.setdefault(
            "categorical_feature_columns",
            normalized_categorical_feature_columns,
        )
        requested_resolved_parameters.setdefault("split_sizes", normalized_split_sizes)
        if random_state is not None:
            requested_resolved_parameters.setdefault("random_state", random_state)
        requested_resolved_parameters.setdefault("split_type", split_type)
        requested_resolved_parameters.setdefault("validation_protocol", validation_protocol)
        direct_model_args = dict(requested_resolved_parameters)

        target_column = normalized_target_columns[0] if normalized_target_columns else None
        activity_cliffs: Dict[str, Any] = {}
        if task_type == "regression" and len(normalized_target_columns) == 1 and target_column:
            try:
                activity_cliffs = prepare_activity_cliff_context(
                    train_csv=train_csv,
                    output_dir=resolved_output_dir,
                    smiles_column="smiles",
                    target_column=target_column,
                    **activity_args,
                )
            except Exception as exc:
                if activity_args.get("activity_cliff_index") != "sali":
                    raise
                activity_cliffs = {
                    "enabled": False,
                    "mode": "skipped",
                    "index_name": activity_args.get("activity_cliff_index", "sali"),
                    "warnings": [f"Activity-cliff annotation skipped: {exc}"],
                }

        training_policy = self._apply_training_profile(requested_resolved_parameters)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
            seed_policy=training_policy["resolved_parameters"].get("seed_policy"),
            base_seed=training_policy["resolved_parameters"].get("random_state"),
            validation_strategy=requested_validation_strategy,
        )
        has_validation = bool(protocol_policy.get("validation_strategy_type") == "cross_validation")
        if not has_validation:
            has_validation = any(
                len(run.get("split_sizes") or normalized_split_sizes or []) == 3
                for run in protocol_policy.get("split_runs") or []
            )
        outlier_config, outlier_skip_reason = normalize_outlier_analysis_config(
            requested_outlier_analysis,
            has_validation=has_validation,
            target_count=len(normalized_target_columns),
            activity_cliff_feedback=bool(activity_args.get("activity_cliff_feedback")),
        )
        training_policy["resolved_parameters"]["random_state"] = protocol_policy["seed_policy"][
            "model_seed"
        ]
        trained_at = project_now()
        active_marker_path = root_output_path / ".training_in_progress"
        prediction_state = get_prediction_state(agent) if agent is not None else None

        active_run_record = {
            "status": "running",
            "backend_name": "lightgbm",
            "train_csv": train_csv,
            "output_dir": resolved_output_dir,
            "validation_protocol": protocol_policy["protocol"],
            "training_profile": training_policy["training_profile"],
            "created_at": trained_at.isoformat(),
            "active_marker_path": str(active_marker_path),
            "current_split_label": None,
            "current_split_index": None,
            "total_splits": len(protocol_policy["split_runs"]),
            "progress_message": None,
            "phase": "Preparing LightGBM training",
        }

        def publish_active_progress(phase: str, payload: Optional[Dict[str, Any]] = None) -> None:
            """Persist compact telemetry for the UI without changing training behavior."""
            apply_progress_update(active_run_record, phase, payload)
            if prediction_state is not None:
                prediction_state["active_training_run"] = dict(active_run_record)
            write_active_training_marker(active_marker_path, active_run_record)

        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        write_active_training_marker(active_marker_path, active_run_record)

        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(normalized_target_columns),
        )
        with S3.open(train_csv, "r") as fh:
            split_source_df = strip_unnamed_columns(pd.read_csv(fh))
        is_cv_protocol = protocol_policy.get("validation_strategy_type") == "cross_validation"
        cv_split_payloads: Dict[str, List[Dict[str, Any]]] = {}
        if is_cv_protocol:
            cv_strategy = protocol_policy.get("validation_strategy") or {}
            cv_split_payloads = build_repeated_kfold_split_payloads(
                df=split_source_df,
                n_splits=int(cv_strategy.get("n_folds") or cv_strategy.get("n_splits") or 5),
                n_repeats=int(cv_strategy.get("n_repeats") or 1),
                random_state=int(
                    cv_strategy.get("seed") or protocol_policy["seed_policy"].get("model_seed") or 0
                ),
                outer_test_size=cv_strategy.get("outer_test_size"),
            )
        split_runs = list(protocol_policy.get("split_runs") or [])
        single_holdout = (
            len(split_runs) == 1
            and not is_cv_protocol
            and len(split_runs[0].get("split_sizes") or normalized_split_sizes or []) == 3
        )
        tuning_config = normalize_tuning_config(
            requested_hyperparameter_tuning,
            backend_name="lightgbm",
            task_type=task_type,
            eligible=has_validation,
            fixed_parameters={
                key
                for key in direct_model_args
                if key
                in {
                    "n_estimators",
                    "learning_rate",
                    "num_leaves",
                    "max_depth",
                    "subsample",
                    "colsample_bytree",
                    "min_child_samples",
                    "reg_alpha",
                    "reg_lambda",
                    "min_split_gain",
                    "boosting_type",
                }
            },
        )
        if tuning_config is not None:
            if activity_args.get("activity_cliff_feedback"):
                raise HyperparameterTuningError(
                    "Activity-cliff feedback loops cannot be combined with hyperparameter tuning in V1."
                )
            if not single_holdout:
                # Every repeated-holdout/CV fold owns a distinct validation
                # study.  The helper below therefore receives one fixed
                # train/validation payload at a time; no outer test is ever
                # handed to Optuna.  A later aggregation keeps every final
                # candidate visible rather than using test metrics to select.
                tuned_splits: List[Dict[str, Any]] = []
                try:
                    for run_index, base_split_run in enumerate(split_runs, start=1):
                        split_run = dict(base_split_run)
                        if split_run.get("label") in cv_split_payloads:
                            split_run["split_payload"] = cv_split_payloads[str(split_run["label"])]
                        elif split_run.get("split_payload") is None:
                            split_run["split_payload"] = build_qsar_split_payload(
                                df=split_source_df,
                                split_type=split_run["backend_split_type"],
                                split_sizes=split_run.get("split_sizes")
                                or normalized_split_sizes
                                or [],
                                random_state=int(split_run["seed"]),
                                smiles_column=(
                                    "smiles" if "smiles" in split_source_df.columns else None
                                ),
                                feature_columns=list(normalized_feature_columns or []),
                            )
                        split_label = str(split_run.get("label") or f"split_{run_index}")
                        publish_active_progress(
                            "Hyperparameter optimization",
                            {"detail": f"fold {run_index} of {len(split_runs)}"},
                        )
                        tuned_split_result = self._train_tuned_holdout(
                            train_csv=train_csv,
                            resolved_output_dir=str(root_output_path / safe_slug(split_label)),
                            task=task,
                            protocol_policy=protocol_policy,
                            training_policy=training_policy,
                            split_source_df=split_source_df,
                            split_run=split_run,
                            feature_columns=list(normalized_feature_columns or []),
                            categorical_feature_columns=list(
                                normalized_categorical_feature_columns or []
                            ),
                            representation_name=representation_name,
                            direct_model_args=direct_model_args,
                            raw_tuning_config=requested_hyperparameter_tuning,
                            applicability_domain_methods=requested_ad_methods,
                            similarity_top_k_neighbors=requested_similarity_top_k,
                            similarity_threshold_percentile=requested_similarity_percentile,
                            outlier_config=outlier_config,
                            outlier_skip_reason=outlier_skip_reason,
                            activity_args=activity_args,
                            selection_only=bool(
                                is_cv_protocol
                                and outlier_config.enabled
                                and outlier_skip_reason is None
                            ),
                            progress_callback=publish_active_progress,
                        )
                        tuned_split_result.update(
                            {
                                "strategy_label": split_label,
                                "strategy_family": (
                                    "cross_validation" if is_cv_protocol else "repeated_holdout"
                                ),
                                "repeat_index": split_run.get("repeat_index"),
                                "fold_index": split_run.get("fold_index"),
                                "selection_split_payload": split_run.get("split_payload"),
                            }
                        )
                        tuned_splits.append(tuned_split_result)
                except Exception:
                    active_run_record["status"] = "failed"
                    active_run_record["completed_at"] = project_now().isoformat()
                    if prediction_state is not None:
                        prediction_state["active_training_run"] = None
                    write_active_training_marker(active_marker_path, active_run_record)
                    raise
                primary_tuned = tuned_splits[0]

                # `baseline` and `outlier_filtered` are meaningful within a
                # split, not across a repeated protocol.  Give every catalog
                # candidate a stable, unique split-qualified identity before
                # it crosses the registry boundary.
                all_variants: List[Dict[str, Any]] = []
                for split_result in tuned_splits:
                    repeat_index = split_result.get("repeat_index")
                    fold_index = split_result.get("fold_index")
                    split_label = str(split_result.get("strategy_label") or "split")
                    for raw_variant in split_result.get("outlier_model_variants") or []:
                        variant = dict(raw_variant)
                        run = variant.get("run")
                        if isinstance(run, dict):
                            run = dict(run)
                            variant["run"] = run
                        base_variant_id = str(
                            variant.get("variant_id")
                            or (run or {}).get("outlier_variant")
                            or "variant"
                        )
                        if len(tuned_splits) > 1:
                            identity_parts = [
                                f"repeat_{repeat_index}" if repeat_index is not None else None,
                                f"fold_{fold_index}" if fold_index is not None else None,
                                base_variant_id,
                            ]
                            variant_id = "_".join(part for part in identity_parts if part)
                        else:
                            variant_id = base_variant_id
                        variant.update(
                            {
                                "variant_id": variant_id,
                                "repeat_index": repeat_index,
                                "fold_index": fold_index,
                                "source_split_label": split_label,
                            }
                        )
                        if isinstance(run, dict):
                            run.update(
                                {
                                    "outlier_variant": variant_id,
                                    "repeat_index": repeat_index,
                                    "fold_index": fold_index,
                                    "source_split_label": split_label,
                                }
                            )
                        all_variants.append(variant)
                cv_outlier_result: Dict[str, Any] = {}
                if is_cv_protocol and outlier_config.enabled and outlier_skip_reason is None:
                    cv_outlier_result = self._run_cross_validation_outlier_variants(
                        train_csv=train_csv,
                        source_df=split_source_df,
                        task=task,
                        tuned_splits=tuned_splits,
                        output_dir=root_output_path,
                        training_policy=training_policy,
                        feature_columns=list(normalized_feature_columns or []),
                        categorical_feature_columns=list(
                            normalized_categorical_feature_columns or []
                        ),
                        representation_name=representation_name,
                        applicability_domain_methods=requested_ad_methods,
                        similarity_top_k_neighbors=requested_similarity_top_k,
                        similarity_threshold_percentile=requested_similarity_percentile,
                        selection_fraction=outlier_config.selection_fraction,
                        progress_callback=publish_active_progress,
                    )
                    if cv_outlier_result.get("variants"):
                        all_variants = list(cv_outlier_result["variants"])
                composite = dict(primary_tuned)
                composite.update(
                    {
                        "output_dir": resolved_output_dir,
                        "split_results": tuned_splits,
                        "baseline_split_results": tuned_splits,
                        "outlier_model_variants": all_variants,
                        "validation_strategy_type": protocol_policy.get("validation_strategy_type"),
                        "validation_strategy": protocol_policy.get("validation_strategy"),
                        "validation_protocol": protocol_policy["protocol"],
                        "catalog_model_policy": "outlier_variants_no_test_winner",
                        "tuning_scope": "one_study_per_validation_fold",
                        "test_selection_policy": "no_test_metric_selects_a_candidate",
                        "summary_path": str(root_output_path / "cs_copilot_training_summary.json"),
                        "canonical_summary_path": str(
                            root_output_path / "cs_copilot_training_summary.json"
                        ),
                    }
                )
                composite["outlier_analysis"] = {
                    "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
                    "status": "completed" if all_variants else "skipped",
                    "reason": outlier_skip_reason,
                    "per_split": [item.get("outlier_analysis") or {} for item in tuned_splits],
                    "per_repeat": cv_outlier_result.get("repeat_summaries") or [],
                    "pooling_note": (
                        "OOF selection rows are pooled once per repeat before final candidate refits."
                        if is_cv_protocol
                        else "Each repeated holdout owns an independent validation outlier study."
                    ),
                }
                write_training_summary(Path(composite["summary_path"]), composite)
                active_run_record["status"] = "completed"
                active_run_record["completed_at"] = project_now().isoformat()
                if prediction_state is not None:
                    prediction_state["active_training_run"] = None
                write_active_training_marker(active_marker_path, active_run_record)
                return composite
            try:
                tuned_result = self._train_tuned_holdout(
                    train_csv=train_csv,
                    resolved_output_dir=resolved_output_dir,
                    task=task,
                    protocol_policy=protocol_policy,
                    training_policy=training_policy,
                    split_source_df=split_source_df,
                    split_run=split_runs[0],
                    feature_columns=list(normalized_feature_columns or []),
                    categorical_feature_columns=list(normalized_categorical_feature_columns or []),
                    representation_name=representation_name,
                    direct_model_args=direct_model_args,
                    raw_tuning_config=requested_hyperparameter_tuning,
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                    outlier_config=outlier_config,
                    outlier_skip_reason=outlier_skip_reason,
                    activity_args=activity_args,
                    progress_callback=publish_active_progress,
                )
            except Exception:
                active_run_record["status"] = "failed"
                active_run_record["completed_at"] = project_now().isoformat()
                if prediction_state is not None:
                    prediction_state["active_training_run"] = None
                write_active_training_marker(active_marker_path, active_run_record)
                raise
            active_run_record["status"] = "completed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            write_active_training_marker(active_marker_path, active_run_record)
            return tuned_result
        split_results: List[Dict[str, Any]] = []
        primary_run: Optional[Dict[str, Any]] = None
        total_started_at = project_now()
        multi_run_protocol = len(protocol_policy["split_runs"]) > 1

        try:
            for run_index, split_run in enumerate(protocol_policy["split_runs"], start=1):
                label = split_run["label"]
                run_output_dir = (
                    root_output_path / f"{safe_slug(label)}_split"
                    if multi_run_protocol
                    else root_output_path
                )
                run_output_dir.mkdir(parents=True, exist_ok=True)
                started_at = project_now()
                split_payload = split_run.get("split_payload")
                split_sizes_for_run = split_run.get("split_sizes") or normalized_split_sizes
                if split_payload is None:
                    if split_run["backend_split_type"] == "final_refit":
                        split_payload = build_full_train_split_payload(df=split_source_df)
                        split_sizes_for_run = [1.0]
                    elif label in cv_split_payloads:
                        split_payload = cv_split_payloads[label]
                    else:
                        split_payload = build_qsar_split_payload(
                            df=split_source_df,
                            split_type=split_run["backend_split_type"],
                            split_sizes=split_sizes_for_run,
                            random_state=int(split_run["seed"]),
                            smiles_column="smiles" if "smiles" in split_source_df.columns else None,
                            feature_columns=normalized_feature_columns,
                        )
                run_args = {
                    **{
                        key: value
                        for key, value in training_policy["resolved_parameters"].items()
                        if key != "seed_policy"
                    },
                    "feature_columns": normalized_feature_columns,
                    "categorical_feature_columns": normalized_categorical_feature_columns,
                    "split_sizes": split_sizes_for_run,
                    "split_type": split_run["backend_split_type"],
                    "split_payload": split_payload,
                    "random_state": split_run["seed"],
                    "validation_protocol": protocol_policy["protocol"],
                }
                if split_run["backend_split_type"] == "final_refit":
                    run_args["final_refit"] = True
                if is_cv_protocol:
                    run_args["early_stopping_rounds"] = 0

                active_run_record["current_split_label"] = label
                active_run_record["current_split_index"] = run_index
                publish_active_progress(
                    "Training model",
                    {"detail": f"run {run_index} of {len(protocol_policy['split_runs'])}"},
                )

                single_result = self._train_backend(
                    train_csv=train_csv,
                    output_dir=str(run_output_dir),
                    task=task,
                    resolved_parameters=run_args,
                )

                if label.startswith("cv_repeat_"):
                    strategy = label
                    strategy_family = "cross_validation"
                elif "scaffold" in label:
                    strategy = "scaffold"
                    strategy_family = "scaffold"
                elif "kmeans" in label or "cluster" in label:
                    strategy = "cluster_kmeans"
                    strategy_family = "cluster_kmeans"
                elif "random_seed_" in label:
                    strategy = label
                    strategy_family = "random"
                else:
                    strategy = "random"
                    strategy_family = "random"

                completed_at = project_now()
                single_result["strategy"] = strategy
                single_result["strategy_family"] = strategy_family
                single_result["strategy_label"] = label
                single_result["backend_split_type"] = split_run["backend_split_type"]
                single_result["seed"] = split_run["seed"]
                single_result["repeat_index"] = split_run.get("repeat_index")
                single_result["fold_index"] = split_run.get("fold_index")
                single_result["n_folds"] = split_run.get("n_folds")
                single_result["n_repeats"] = split_run.get("n_repeats")
                single_result["split_payload"] = split_payload or single_result.get("split_payload")
                single_result["validation_protocol"] = protocol_policy["protocol"]
                single_result["output_dir"] = str(run_output_dir)
                single_result["started_at"] = (
                    single_result.get("started_at") or started_at.isoformat()
                )
                single_result["completed_at"] = (
                    single_result.get("completed_at") or completed_at.isoformat()
                )
                single_result["duration_seconds"] = single_result.get("duration_seconds") or round(
                    (completed_at - started_at).total_seconds(), 3
                )
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
        finally:
            active_run_record["status"] = "completed" if primary_run is not None else "failed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            write_active_training_marker(active_marker_path, active_run_record)

        if primary_run is None:
            raise ValueError("LightGBM validation protocol did not produce a primary run.")

        activity_cliff_variant_split_results: Dict[str, List[Dict[str, Any]]] = {}
        if not is_cv_protocol and activity_cliffs.get("mode") == "with_feedback_loops":
            trainable_variants = [
                variant
                for variant in (activity_cliffs.get("variants") or [])
                if variant.get("loop_index", 0) > 0 and variant.get("removed_count", 0) > 0
            ]
            for variant in trainable_variants:
                variant_id = str(variant.get("variant_id"))
                removed_indices = [int(idx) for idx in (variant.get("removed_row_indices") or [])]
                variant_results: List[Dict[str, Any]] = []
                for baseline_result in split_results:
                    label = str(
                        baseline_result.get("strategy_label")
                        or baseline_result.get("strategy")
                        or "split"
                    )
                    run_output_dir = (
                        root_output_path
                        / "activity_cliff_variants"
                        / safe_slug(variant_id)
                        / f"{safe_slug(label)}_split"
                    )
                    run_output_dir.mkdir(parents=True, exist_ok=True)
                    started_at = project_now()
                    run_args = {
                        **{
                            key: value
                            for key, value in training_policy["resolved_parameters"].items()
                            if key != "seed_policy"
                        },
                        "feature_columns": normalized_feature_columns,
                        "categorical_feature_columns": normalized_categorical_feature_columns,
                        "split_sizes": baseline_result.get("split_sizes") or normalized_split_sizes,
                        "split_type": baseline_result.get("backend_split_type"),
                        "split_payload": baseline_result.get("split_payload"),
                        "excluded_train_indices": removed_indices,
                        "activity_cliff_variant_id": variant_id,
                        "random_state": baseline_result.get("seed", random_state),
                        "validation_protocol": protocol_policy["protocol"],
                    }

                    active_run_record["status"] = "running"
                    active_run_record["current_split_label"] = f"{variant_id}:{label}"
                    publish_active_progress(
                        "Training activity-cliff variant",
                        {"detail": f"{variant_id} on fixed {label} holdout"},
                    )

                    variant_result = self._train_backend(
                        train_csv=train_csv,
                        output_dir=str(run_output_dir),
                        task=task,
                        resolved_parameters=run_args,
                    )
                    completed_at = project_now()
                    variant_result["strategy"] = baseline_result.get("strategy")
                    variant_result["strategy_family"] = baseline_result.get("strategy_family")
                    variant_result["strategy_label"] = label
                    variant_result["backend_split_type"] = baseline_result.get("backend_split_type")
                    variant_result["seed"] = baseline_result.get("seed")
                    variant_result["validation_protocol"] = protocol_policy["protocol"]
                    variant_result["output_dir"] = str(run_output_dir)
                    variant_result["started_at"] = (
                        variant_result.get("started_at") or started_at.isoformat()
                    )
                    variant_result["completed_at"] = (
                        variant_result.get("completed_at") or completed_at.isoformat()
                    )
                    variant_result["duration_seconds"] = variant_result.get(
                        "duration_seconds"
                    ) or round((completed_at - started_at).total_seconds(), 3)
                    variant_result["activity_cliff_variant_id"] = variant_id
                    variant_result["removed_tiers"] = variant.get("removed_tiers") or []
                    variant_result["removed_count"] = variant.get("removed_count", 0)
                    variant_result["remaining_rows"] = variant.get("remaining_rows")
                    variant_results.append(variant_result)
                activity_cliff_variant_split_results[variant_id] = variant_results

        active_run_record["status"] = "completed"
        active_run_record["completed_at"] = project_now().isoformat()
        write_active_training_marker(active_marker_path, active_run_record)

        activity_cliffs = self._attach_activity_cliff_variant_training(
            activity_cliffs=activity_cliffs,
            baseline_split_results=split_results,
            variant_split_results=activity_cliff_variant_split_results,
        )
        if activity_cliffs.get("variant_comparison_table"):
            try:
                loop_plot_artifacts = build_activity_cliff_loop_comparison_plots(
                    activity_cliffs,
                    output_dir=str(Path(resolved_output_dir) / "activity_cliffs"),
                )
                if loop_plot_artifacts:
                    activity_cliffs["loop_comparison_plot_artifacts"] = loop_plot_artifacts
            except Exception:
                logger.warning("Could not generate activity-cliff loop comparison plots.")
        if activity_cliffs.get("summary_path"):
            try:
                Path(str(activity_cliffs["summary_path"])).write_text(
                    json.dumps(activity_cliffs, indent=2) + "\n"
                )
            except Exception:
                logger.warning(
                    "Could not update activity-cliff summary with loop training results."
                )

        cross_validation_artifacts: Dict[str, Any] = {}
        final_refit_run: Optional[Dict[str, Any]] = None
        if is_cv_protocol and target_column:
            cross_validation_artifacts = build_cross_validation_artifacts(
                split_results=split_results,
                output_dir=root_output_path / "cross_validation",
                target_column=target_column,
            )
        if is_cv_protocol and protocol_policy.get("final_refit", True):
            final_output_dir = root_output_path / "final_refit"
            final_output_dir.mkdir(parents=True, exist_ok=True)
            final_started_at = project_now()
            cv_outer_test_indices = list(
                ((next(iter(cv_split_payloads.values()), [{}]) or [{}])[0]).get("test") or []
            )
            final_split_payload = build_full_train_split_payload(
                df=split_source_df,
                test_indices=cv_outer_test_indices or None,
            )
            final_args = {
                **{
                    key: value
                    for key, value in training_policy["resolved_parameters"].items()
                    if key != "seed_policy"
                },
                "feature_columns": normalized_feature_columns,
                "categorical_feature_columns": normalized_categorical_feature_columns,
                "split_type": "final_refit",
                "split_payload": final_split_payload,
                "random_state": protocol_policy["seed_policy"]["model_seed"],
                "validation_protocol": protocol_policy["protocol"],
                "final_refit": True,
                "early_stopping_rounds": 0,
            }
            final_refit_run = self._train_backend(
                train_csv=train_csv,
                output_dir=str(final_output_dir),
                task=task,
                resolved_parameters=final_args,
            )
            final_completed_at = project_now()
            final_refit_run["strategy"] = "final_refit"
            final_refit_run["strategy_family"] = "final_refit"
            final_refit_run["strategy_label"] = "final_refit"
            final_refit_run["backend_split_type"] = "final_refit"
            final_refit_run["seed"] = protocol_policy["seed_policy"].get("model_seed")
            final_refit_run["split_payload"] = final_split_payload
            final_refit_run["validation_protocol"] = protocol_policy["protocol"]
            final_refit_run["output_dir"] = str(final_output_dir)
            final_refit_run["started_at"] = (
                final_refit_run.get("started_at") or final_started_at.isoformat()
            )
            final_refit_run["completed_at"] = (
                final_refit_run.get("completed_at") or final_completed_at.isoformat()
            )
            final_refit_run["duration_seconds"] = final_refit_run.get("duration_seconds") or round(
                (final_completed_at - final_started_at).total_seconds(),
                3,
            )

        final_split_results = split_results
        final_primary_run = final_refit_run or primary_run
        outlier_study: Dict[str, Any] = {
            "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
            "status": "skipped" if outlier_skip_reason else "pending",
            "reason": outlier_skip_reason,
            "config": outlier_config.as_dict(),
            "studies": [],
        }
        outlier_model_variants: List[Dict[str, Any]] = []
        # The default LightGBM path is usually tuned above.  This branch also
        # covers an explicitly disabled tuning study and repeated holdout: each
        # already-fitted train/validation model supplies the selection
        # prediction, then both final variants are retrained without retuning.
        if (
            outlier_config.enabled
            and outlier_skip_reason is None
            and not activity_cliffs.get("mode") == "with_feedback_loops"
        ):
            for split_result in split_results:
                split_payload = split_result.get("split_payload") or []
                split = split_payload[0] if split_payload else {}
                validation_indices = list(split.get("val") or split.get("validation") or [])
                prediction_path = split_result.get("validation_predictions_path")
                if (
                    not validation_indices
                    or not prediction_path
                    or not Path(str(prediction_path)).exists()
                ):
                    outlier_study["studies"].append(
                        {
                            "split_label": split_result.get("strategy_label"),
                            "status": "skipped",
                            "reason": "Selection validation predictions are unavailable.",
                        }
                    )
                    continue
                publish_active_progress(
                    "Identifying validation outliers",
                    {"detail": str(split_result.get("strategy_label") or "holdout")},
                )
                selection_frame = pd.read_csv(Path(str(prediction_path)))
                run_output = Path(str(split_result.get("output_dir") or root_output_path))
                base_args = {
                    **{
                        key: value
                        for key, value in training_policy["resolved_parameters"].items()
                        if key not in {"seed_policy", "split_payload", "validation_strategy"}
                    },
                    "feature_columns": list(
                        normalized_feature_columns or split_result.get("feature_columns") or []
                    ),
                    "categorical_feature_columns": list(
                        normalized_categorical_feature_columns or []
                    ),
                    "random_state": int(split_result.get("seed") or 0),
                    "validation_protocol": protocol_policy["protocol"],
                    "early_stopping_rounds": 0,
                    "deterministic": True,
                }
                model_parameters = {
                    key: value
                    for key, value in direct_model_args.items()
                    if key
                    in {
                        "n_estimators",
                        "learning_rate",
                        "num_leaves",
                        "max_depth",
                        "subsample",
                        "colsample_bytree",
                        "min_child_samples",
                        "reg_alpha",
                        "reg_lambda",
                        "min_split_gain",
                        "boosting_type",
                    }
                }
                study = self._run_outlier_refits(
                    train_csv=train_csv,
                    split_source_df=split_source_df,
                    task=task,
                    split_payload=split_payload,
                    selection_prediction_frame=selection_frame,
                    output_dir=run_output,
                    base_args=base_args,
                    selected_parameters=model_parameters,
                    feature_columns=list(
                        normalized_feature_columns or split_result.get("feature_columns") or []
                    ),
                    representation_name=representation_name,
                    activity_args=activity_args,
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                    selection_fraction=outlier_config.selection_fraction,
                    progress_callback=publish_active_progress,
                    fold_label=str(split_result.get("strategy_label") or "holdout"),
                    repeat_index=split_result.get("repeat_index"),
                )
                study_summary = dict(study["summary"])
                study_summary["split_label"] = split_result.get("strategy_label")
                outlier_study["studies"].append(study_summary)
                for variant in study["variants"]:
                    variant["split_label"] = split_result.get("strategy_label")
                    variant["repeat_index"] = split_result.get("repeat_index")
                    variant["variant_id"] = (
                        f"{safe_slug(str(split_result.get('strategy_label') or 'holdout'))}_"
                        f"{variant.get('variant_id')}"
                    )
                    outlier_model_variants.append(variant)
            if outlier_model_variants:
                outlier_study["status"] = "completed"
                primary_label = primary_run.get("strategy_label")
                baseline_variant = next(
                    (
                        item
                        for item in outlier_model_variants
                        if item.get("split_label") == primary_label
                        and str(item.get("variant_id", "")).endswith("_baseline")
                        and isinstance(item.get("run"), dict)
                    ),
                    None,
                )
                if baseline_variant is not None:
                    final_primary_run = baseline_variant["run"]
                    final_split_results = [
                        item["run"]
                        for item in outlier_model_variants
                        if isinstance(item.get("run"), dict)
                    ]
            elif not outlier_study["studies"]:
                outlier_study["status"] = "skipped"
                outlier_study["reason"] = "Selection validation predictions are unavailable."
        recommended_variant = activity_cliffs.get("recommended_variant")
        if (
            not is_cv_protocol
            and recommended_variant
            and recommended_variant != "baseline_loop_0"
            and recommended_variant in activity_cliff_variant_split_results
        ):
            candidate_split_results = activity_cliff_variant_split_results[recommended_variant]
            if candidate_split_results:
                final_split_results = candidate_split_results
                primary_label = primary_run.get("strategy_label")
                final_primary_run = next(
                    (
                        item
                        for item in candidate_split_results
                        if item.get("strategy_label") == primary_label
                    ),
                    candidate_split_results[0],
                )

        root_artifacts = self._materialize_primary_protocol_artifacts(
            root_output_dir=root_output_path,
            primary_run=final_primary_run,
        )
        ad_summary = self._build_applicability_domain(
            train_csv=train_csv,
            primary_run=final_primary_run,
            primary_output_dir=root_output_path,
            task=task,
            feature_columns=normalized_feature_columns
            or final_primary_run.get("feature_columns")
            or [],
            feature_space=representation_name
            or final_primary_run.get("representation_name")
            or final_primary_run.get("feature_space"),
            prediction_artifact_paths={
                "validation": root_artifacts.get("validation_predictions_path"),
                "test": root_artifacts.get("test_predictions_path"),
            },
            applicability_domain_methods=requested_ad_methods,
            similarity_top_k_neighbors=requested_similarity_top_k,
            similarity_threshold_percentile=requested_similarity_percentile,
        )
        if outlier_model_variants:
            for variant in outlier_model_variants:
                run = variant.get("run")
                if not isinstance(run, dict):
                    continue
                if run is final_primary_run:
                    run["applicability_domain"] = ad_summary
                    continue
                run_output_dir = Path(str(run.get("output_dir") or root_output_path))
                run["applicability_domain"] = self._build_applicability_domain(
                    train_csv=train_csv,
                    primary_run=run,
                    primary_output_dir=run_output_dir,
                    task=task,
                    feature_columns=normalized_feature_columns or run.get("feature_columns") or [],
                    feature_space=representation_name or run.get("representation_name"),
                    prediction_artifact_paths={"test": run.get("test_predictions_path")},
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                )
        plot_artifacts: Dict[str, str] = {}
        target_column = task.target_columns[0] if task.target_columns else None
        if protocol_policy.get("validation_strategy_type") != "full_train":
            plot_artifacts = build_training_plots_if_possible(
                train_csv=train_csv,
                split_results=final_split_results,
                primary_run=final_primary_run,
                root_artifacts=root_artifacts,
                root_output_dir=root_output_path,
                target_column=target_column,
                task_type=task.task_type,
            )

        validation_assessment = assess_protocol_results(final_split_results)
        total_completed_at = project_now()
        result = dict(final_primary_run)
        result["output_dir"] = resolved_output_dir
        result["model_path"] = root_artifacts.get("best_model_path") or final_primary_run.get(
            "model_path"
        )
        result["summary_path"] = str(root_output_path / "cs_copilot_training_summary.json")
        result["config_path"] = root_artifacts.get("config_path") or final_primary_run.get(
            "config_path"
        )
        result["splits_path"] = root_artifacts.get("splits_path") or final_primary_run.get(
            "splits_path"
        )
        result["validation_predictions_path"] = root_artifacts.get(
            "validation_predictions_path"
        ) or final_primary_run.get("validation_predictions_path")
        result["test_predictions_path"] = root_artifacts.get(
            "test_predictions_path"
        ) or final_primary_run.get("test_predictions_path")
        result["selected_activity_cliff_variant"] = recommended_variant or "baseline_loop_0"
        result["validation_protocol"] = protocol_policy["protocol"]
        if protocol_policy.get("validation_strategy_type") == "full_train":
            result["metrics"] = {}
            result["test_predictions_path"] = None
            result["metrics_status"] = "not_evaluated"
            result["evaluation_required"] = True
        result["validation_protocol_reason"] = protocol_policy["reason"]
        result["validation_strategy"] = protocol_policy.get("validation_strategy")
        result["validation_strategy_type"] = protocol_policy.get("validation_strategy_type")
        result["validation_aggregation"] = protocol_policy.get("aggregation")
        result["final_refit"] = protocol_policy.get("final_refit")
        result["seed_policy"] = protocol_policy["seed_policy"]
        result["seed_policy_report"] = seed_policy_reporting_text(protocol_policy["seed_policy"])
        result["reproducibility"] = seed_policy_reproducibility_metadata(
            protocol_policy["seed_policy"]
        )
        result["split_results"] = final_split_results
        result["baseline_split_results"] = split_results
        result["cross_validation"] = cross_validation_artifacts
        result["cv_artifacts"] = cross_validation_artifacts
        result["final_refit_result"] = final_refit_run
        result["catalog_model_policy"] = (
            "final_refit_only_fold_models_are_artifacts"
            if is_cv_protocol
            else result.get("catalog_model_policy")
        )
        result["validation_assessment"] = validation_assessment
        result["compute_environment"] = training_policy["compute_environment"]
        result["training_profile"] = training_policy["training_profile"]
        result["profile_reason"] = training_policy["profile_reason"]
        result["effective_train_args"] = {
            **{
                key: value
                for key, value in training_policy["resolved_parameters"].items()
                if key != "seed_policy"
            },
            "device_type": final_primary_run.get("effective_train_args", {}).get("device_type"),
        }
        result["training_resources"] = self._summarize_training_resources(
            compute_env=training_policy["compute_environment"],
            effective_train_args=result["effective_train_args"],
        )
        result["training_durations"] = summarize_training_durations(
            split_results=final_split_results,
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )
        curation_artifacts = latest_curation_artifacts(agent) if agent is not None else {}
        if not (curation_artifacts.get("artifacts") if curation_artifacts else None):
            curation_artifacts = discover_curation_artifacts_near_dataset(train_csv)
        result["curation"] = curation_artifacts
        result["applicability_domain"] = ad_summary
        result["activity_cliffs"] = activity_cliffs
        outlier_plots = {
            key: value
            for study in outlier_study.get("studies") or []
            for key, value in (study.get("plot_artifacts") or {}).items()
        }
        result["plot_artifacts"] = {**plot_artifacts, **outlier_plots}
        result["outlier_analysis"] = outlier_study
        result["outlier_model_variants"] = outlier_model_variants
        if outlier_model_variants:
            result["catalog_model_policy"] = "outlier_variants_no_test_winner"
        result["trained_at"] = trained_at.isoformat()
        result["trained_date"] = trained_at.strftime("%d/%m/%Y")
        result["trained_time"] = trained_at.strftime("%H:%M:%S")
        result["train_csv"] = train_csv
        result["target_columns"] = list(normalized_target_columns)
        result["feature_columns"] = list(
            normalized_feature_columns or (final_primary_run.get("feature_columns") or [])
        )
        result["categorical_feature_columns"] = list(
            normalized_categorical_feature_columns
            or (final_primary_run.get("categorical_feature_columns") or [])
        )
        result["canonical_summary_path"] = result["summary_path"]

        summary_path = Path(result["summary_path"])
        write_training_summary(summary_path, result)

        bundle_destination = (
            Path(bundle_path).expanduser()
            if bundle_path
            else Path(".files")
            / "prediction_outputs"
            / f"{Path(resolved_output_dir).name}_training_bundle.zip"
        ).resolve()
        bundle_files = collect_training_bundle_files(
            train_csv=train_csv,
            summary_path=summary_path,
            result=result,
            split_results=[*final_split_results, *split_results],
            ad_summary=ad_summary,
            plot_artifacts=plot_artifacts,
            curation_artifacts=curation_artifacts,
            activity_cliffs=activity_cliffs,
            extra_files=[Path(resolved_output_dir)],
        )

        bundle = bundle_artifacts(bundle_destination, bundle_files)
        result["bundle_file_ref"] = str(bundle)
        result["training_bundle"] = str(bundle)
        result["bundle_download_tag"] = f"<file>{bundle}</file>"
        write_training_summary(summary_path, result)

        if prediction_state is not None:
            prediction_state["training_runs"].append(
                {
                    "train_csv": train_csv,
                    "output_dir": resolved_output_dir,
                    "task_type": task_type,
                    "smiles_columns": task.smiles_columns,
                    "target_columns": task.target_columns,
                    "validation_protocol": protocol_policy["protocol"],
                    "training_profile": training_policy["training_profile"],
                    "seed_policy": protocol_policy["seed_policy"],
                    "split_runs": [
                        {
                            "label": item["strategy_label"],
                            "strategy": item["strategy"],
                            "strategy_family": item.get("strategy_family"),
                            "output_dir": item["output_dir"],
                            "seed": item["seed"],
                        }
                        for item in final_split_results
                    ],
                    "activity_cliffs": {
                        "enabled": bool(activity_cliffs.get("enabled")),
                        "mode": activity_cliffs.get("mode"),
                        "index_name": activity_cliffs.get("index_name"),
                        "summary_path": activity_cliffs.get("summary_path"),
                        "recommended_variant": activity_cliffs.get("recommended_variant"),
                    },
                }
            )
            prediction_state["active_training_run"] = None

        return result

    def predict_with_lightgbm_from_csv(
        self,
        input_csv: str,
        model_path: str,
        preds_path: str,
        target_columns: Optional[List[str] | str] = None,
        feature_columns: Optional[List[str] | str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
    ) -> Dict[str, Any]:
        """Run LightGBM batch prediction from a tabular CSV input file."""
        normalized_target_columns = (
            self._normalize_json_list_argument(
                target_columns,
                argument_name="target_columns",
            )
            or []
        )
        normalized_feature_columns = (
            self._normalize_json_list_argument(
                feature_columns,
                argument_name="feature_columns",
            )
            or []
        )
        normalized_categorical_feature_columns = (
            self._normalize_json_list_argument(
                categorical_feature_columns,
                argument_name="categorical_feature_columns",
            )
            or []
        )

        from .backend import PredictionModelRecord

        model_record = PredictionModelRecord(
            model_id=Path(model_path).stem,
            backend_name=self.backend.backend_name,
            model_path=model_path,
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=list(normalized_target_columns),
            ),
            inference_profile={
                "feature_columns": list(normalized_feature_columns),
                "categorical_feature_columns": list(normalized_categorical_feature_columns),
            },
        )
        return self.backend.predict_from_csv(
            input_csv=input_csv,
            model_record=model_record,
            preds_path=preds_path,
        )
