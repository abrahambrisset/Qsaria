#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing TabICLv2-backed tabular QSAR workflows.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import (
    prepare_activity_cliff_context,
    split_activity_cliff_args,
)

from .applicability_domain import fit_modern_applicability_domain, score_modern_applicability_domain
from .backend import PredictionExecutionError, PredictionTaskSpec
from .hyperparameter_tuning import normalize_tuning_config
from .outlier_analysis import (
    attach_activity_cliff_annotations,
    attach_ad_annotations,
    normalize_outlier_analysis_config,
    select_outliers,
    selection_predictions_from_frame,
    write_outlier_analysis_artifacts,
    write_outlier_variant_comparison,
)
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
    get_prediction_state,
    write_active_training_marker,
)
from .tabicl_backend import (
    DEFAULT_TABICL_CHECKPOINT_DIR,
    DEFAULT_TABICL_CLASSIFIER_CHECKPOINT,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    TabICLBackend,
)
from .tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
)
from .training_orchestration import (
    apply_training_profile,
    build_applicability_domain_for_training,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    is_classification_task,
    materialize_primary_protocol_artifacts,
    normalize_classification_label,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)

logger = logging.getLogger(__name__)


def _ad_status_series(applicability_domain_result: Dict[str, Any]) -> Optional[pd.Series]:
    scores = applicability_domain_result.get("scores")
    return scores.get("ad_status") if isinstance(scores, pd.DataFrame) else None


def _clean_split_source_target_for_task(
    df: pd.DataFrame,
    *,
    target_column: str,
    task_type: str,
) -> pd.DataFrame:
    if target_column not in df.columns:
        return df
    if is_classification_task(task_type):
        return df.loc[df[target_column].map(normalize_classification_label).notna()].reset_index(
            drop=True
        )
    cleaned = df.copy()
    cleaned[target_column] = pd.to_numeric(cleaned[target_column], errors="coerce")
    return cleaned.dropna(subset=[target_column]).reset_index(drop=True)


class TabICLToolkit(Toolkit):
    """Backend-specific TabICL training toolkit used behind QSARTrainingToolkit."""

    def __init__(self, backend: Optional[TabICLBackend] = None):
        super().__init__("tabicl_prediction")
        self.backend = backend or TabICLBackend()

    def is_tabicl_available(self) -> bool:
        """Return whether the TabICL backend is available in the current environment."""
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
        source_df: pd.DataFrame,
        task: PredictionTaskSpec,
        split_payload: List[Dict[str, Any]],
        selection_prediction_frame: pd.DataFrame,
        output_dir: Path,
        train_args: Dict[str, Any],
        feature_columns: List[str],
        representation_name: Optional[str],
        activity_args: Dict[str, Any],
        applicability_domain_methods: Optional[List[str] | str],
        similarity_top_k_neighbors: int | str | None,
        similarity_threshold_percentile: float | str | None,
        selection_fraction: float,
        fold_label: str,
        repeat_index: Optional[int] = None,
        progress_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Run the common selection policy and TabICL baseline/filtered refits."""
        split = split_payload[0]
        train_indices = [int(value) for value in split.get("train") or []]
        validation_indices = [
            int(value) for value in (split.get("val") or split.get("validation") or [])
        ]
        test_indices = [int(value) for value in split.get("test") or []]
        if not train_indices or not validation_indices:
            raise ValueError("Outlier analysis requires a non-empty train and validation split.")
        resolved_features = list(feature_columns)
        if not resolved_features:
            raise ValueError("Outlier analysis requires the resolved TabICL feature columns.")
        target_column = task.target_columns[0]
        selection = selection_predictions_from_frame(
            selection_prediction_frame,
            source_row_indices=validation_indices,
            target_column=target_column,
            fold_label=fold_label,
            repeat_index=repeat_index,
        )
        analysis_dir = output_dir / "outlier_analysis"
        development_indices = [*train_indices, *validation_indices]
        development_frame = source_df.iloc[development_indices].copy()
        development_frame.index = development_indices

        with tempfile.TemporaryDirectory(prefix="qsaria_outlier_tabicl_") as temporary_dir:
            temporary_ad = fit_modern_applicability_domain(
                feature_frame=source_df.iloc[train_indices][resolved_features].reset_index(
                    drop=True
                ),
                feature_columns=resolved_features,
                output_dir=Path(temporary_dir) / "applicability_domain",
                model_id="tabicl_outlier_selection",
                feature_space=representation_name or "tabular",
                representation_name=representation_name,
                methods=applicability_domain_methods,
                random_state=int(train_args.get("random_state") or 0),
                all_feature_frame=source_df[resolved_features],
                train_indices=train_indices,
                split_indices=split,
                similarity_top_k_neighbors=similarity_top_k_neighbors,
                similarity_threshold_percentile=similarity_threshold_percentile,
            )
            validation_ad = score_modern_applicability_domain(
                feature_frame=source_df.iloc[validation_indices][resolved_features].reset_index(
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

        ac_annotations: Optional[pd.DataFrame] = None
        if task.task_type == "regression":
            ac_input = development_frame.drop(columns=["source_row_index"], errors="ignore").copy()
            ac_input.insert(0, "source_row_index", ac_input.index)
            ac_source = analysis_dir / "development_for_selection.csv"
            ac_source.parent.mkdir(parents=True, exist_ok=True)
            ac_input.to_csv(ac_source, index=False)
            try:
                ac_context = prepare_activity_cliff_context(
                    train_csv=str(ac_source),
                    output_dir=str(analysis_dir / "activity_cliffs_development"),
                    smiles_column=task.smiles_columns[0] if task.smiles_columns else "smiles",
                    target_column=target_column,
                    **activity_args,
                )
                annotated_path = ac_context.get("annotated_training_csv")
                if annotated_path and Path(str(annotated_path)).exists():
                    ac_annotations = pd.read_csv(annotated_path)
            except Exception as exc:
                logger.warning("Development-only Activity Cliff annotations unavailable: %s", exc)
        selection = attach_activity_cliff_annotations(selection, annotations=ac_annotations)
        selected_rows, selection_summary = select_outliers(
            selection,
            task_type=task.task_type,
            selection_fraction=selection_fraction,
        )
        artifacts = write_outlier_analysis_artifacts(
            output_dir=analysis_dir,
            selection_frame=selected_rows,
            selection_summary=selection_summary,
            development_frame=development_frame,
            extra_summary={
                "fold_label": fold_label,
                "repeat_index": repeat_index,
                "selection_model": "train_only_validation_fit",
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

        def _fit_variant(variant_id: str, train_indices_for_variant: List[int]) -> Dict[str, Any]:
            payload = [
                {
                    "train": train_indices_for_variant,
                    **({"test": test_indices} if test_indices else {}),
                    "metadata": {
                        **dict(split.get("metadata") or {}),
                        "refit_on_train_validation": True,
                        "outlier_variant": variant_id,
                        "removed_source_row_indices": (
                            selected_indices if variant_id == "outlier_filtered" else []
                        ),
                    },
                }
            ]
            run = self.backend.train_model(
                train_csv=train_csv,
                output_dir=str(output_dir / "outlier_variants" / variant_id),
                task=task,
                extra_args={
                    **train_args,
                    "split_payload": payload,
                    "split_type": "final_refit",
                    "final_refit": True,
                },
            )
            run["outlier_variant"] = variant_id
            run["outlier_selected_count"] = len(selected_indices)
            run["split_payload"] = payload
            return run

        variants: List[Dict[str, Any]] = []
        try:
            if progress_callback is not None:
                progress_callback("Training baseline refit", {"detail": "train + validation"})
            variants.append(
                {"variant_id": "baseline", "run": _fit_variant("baseline", development_indices)}
            )
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL baseline outlier refit failed: {exc}") from exc
        if selected_indices:
            filtered_indices = [
                value for value in development_indices if value not in set(selected_indices)
            ]
            try:
                if progress_callback is not None:
                    progress_callback(
                        "Training filtered refit",
                        {"detail": f"{len(selected_indices)} rows removed"},
                    )
                variants.append(
                    {
                        "variant_id": "outlier_filtered",
                        "run": _fit_variant("outlier_filtered", filtered_indices),
                    }
                )
            except Exception as exc:
                variants.append(
                    {"variant_id": "outlier_filtered", "status": "failed", "reason": str(exc)}
                )

        comparison_path = write_outlier_variant_comparison(
            output_dir=analysis_dir,
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

    def _training_defaults_for_profile(self, profile: str) -> Dict[str, Any]:
        base = {
            "split_sizes": [0.8, 0.1, 0.1],
            "split_type": "random",
            "n_jobs": 1,
            "verbose": False,
        }
        if profile == "heavy_validation":
            return {
                **base,
                "batch_size": 64,
                "n_estimators": 8,
                "kv_cache": False,
                "n_jobs": 8,
            }
        if profile == "local_standard":
            return {
                **base,
                "batch_size": 64,
                "n_estimators": 4,
                "kv_cache": False,
            }
        return {
            **base,
            "batch_size": 32,
            "n_estimators": 4,
            "kv_cache": False,
        }

    def _apply_training_profile(
        self,
        extra_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        compute_env = self.describe_compute_environment()
        requested_n_jobs = (extra_args or {}).get("n_jobs")

        def _limit(
            profile: str, merged: Dict[str, Any], allow_heavy_compute: bool
        ) -> Dict[str, Any]:
            if allow_heavy_compute:
                if profile == "heavy_validation":
                    merged["batch_size"] = max(int(merged.get("batch_size", 64)), 64)
                    merged["n_estimators"] = max(int(merged.get("n_estimators", 8)), 8)
                    merged["n_jobs"] = resolve_backend_n_jobs(
                        compute_env,
                        backend_name="tabicl",
                        profile=profile,
                        requested_n_jobs=requested_n_jobs,
                    )
                    merged["kv_cache"] = bool(merged.get("kv_cache", False))
                return merged
            if profile == "local_light":
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["n_estimators"] = min(int(merged.get("n_estimators", 4)), 4)
                merged["kv_cache"] = False
            elif profile == "local_standard":
                merged["batch_size"] = min(int(merged.get("batch_size", 64)), 64)
                merged["n_estimators"] = min(int(merged.get("n_estimators", 4)), 4)
            merged["n_jobs"] = resolve_backend_n_jobs(
                compute_env,
                backend_name="tabicl",
                profile=profile,
                requested_n_jobs=requested_n_jobs,
            )
            return merged

        return apply_training_profile(
            extra_args,
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
        return {
            "execution_env": compute_env.get("execution_env"),
            "cpu_count": compute_env.get("cpu_count"),
            "gpu_available": compute_env.get("gpu_available"),
            "gpu_count": compute_env.get("gpu_count"),
            "gpu_name": compute_env.get("gpu_name"),
            "memory_gb_total": compute_env.get("memory_gb_total"),
            "batch_size": effective_train_args.get("batch_size"),
            "n_estimators": effective_train_args.get("n_estimators"),
            "n_jobs": effective_train_args.get("n_jobs"),
            "kv_cache": effective_train_args.get("kv_cache"),
        }

    def describe_tabicl_backend(self) -> Dict[str, Any]:
        """Describe the TabICL backend defaults and current runtime support."""
        description = self.backend.describe_environment()
        description.update(
            {
                "default_task_type": "regression",
                "default_checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
                "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "default_regressor_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "default_classifier_checkpoint_version": DEFAULT_TABICL_CLASSIFIER_CHECKPOINT,
                "supported_task_types": [
                    "regression",
                    "classification",
                    "multiclass_classification",
                ],
                "supported_validation_protocols": ["standard_qsar"],
                "supported_split_families": ["random", "scaffold", "cluster_kmeans"],
                "default_tabular_feature_policy": {
                    "automatic_representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
                    "default_single_representation": "morgan_rdkit_all",
                    "explicit_user_override": True,
                },
                "notes": [
                    "Supports the shared QSAR protocol names and split families.",
                    "Modern comparative campaigns use RDKit all, Morgan binary, Morgan count, and the complete combined pack.",
                    "If the user explicitly requests a representation, that override should win.",
                    "TabICL classification supports binary and multiclass single-target tasks.",
                    "TabICL uses separate classifier and regressor checkpoints.",
                    "TabICL does not expose native multi-target training.",
                    "The `.ckpt` checkpoint is a backend resource, not a trained model artifact.",
                    "`validate_tabicl_model_path` is intended for saved trained models such as `.pkl`.",
                ],
            }
        )
        return description

    def describe_tabicl_environment(self) -> Dict[str, Any]:
        """Return a lightweight runtime snapshot for TabICL availability."""
        snapshot = self.backend.describe_environment()
        snapshot["compute_environment"] = self.describe_compute_environment()
        return snapshot

    def validate_tabicl_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a saved TabICL model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "model_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
        }

    def validate_tabicl_checkpoint_path(
        self, checkpoint_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate the persisted TabICL base checkpoint path."""
        path = Path(
            checkpoint_path or (DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT)
        )
        resolved = path.expanduser().resolve()
        if resolved.suffix != ".ckpt":
            raise ValueError(f"TabICL checkpoint must end with '.ckpt'. Received: {resolved}")
        return {
            "checkpoint_path": str(resolved),
            "exists": resolved.exists(),
            "suffix": resolved.suffix,
            "is_backend_resource": True,
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

    def _build_active_run_record(
        self,
        *,
        train_csv: str,
        resolved_output_dir: str,
        protocol_policy: Dict[str, Any],
        training_policy: Dict[str, Any],
        trained_at,
        active_marker_path: Path,
        worker_pid: Optional[int] = None,
        worker_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "status": "running",
            "backend_name": "tabicl",
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
            "phase": "Preparing TabICL training",
            "worker_pid": worker_pid,
            "worker_status": worker_status,
        }

    def _sync_training_run_state_from_result(
        self,
        *,
        prediction_state: Optional[Dict[str, Any]],
        train_csv: str,
        resolved_output_dir: str,
        task_type: str,
        task: PredictionTaskSpec,
        protocol_policy: Dict[str, Any],
        training_policy: Dict[str, Any],
        split_results: List[Dict[str, Any]],
    ) -> None:
        if prediction_state is None:
            return
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
                    for item in split_results
                ],
            }
        )

    def _run_protocol_training(
        self,
        *,
        train_csv: str,
        task_type: str,
        resolved_output_dir: str,
        target_columns: List[str],
        feature_columns: Optional[List[str]],
        split_type: str,
        split_sizes: Optional[List[float]],
        random_state: int,
        extra_args: Optional[Dict[str, Any]],
        representation_name: Optional[str] = None,
        prediction_state: Optional[Dict[str, Any]] = None,
        active_marker_path: Optional[Path] = None,
        worker_pid: Optional[int] = None,
        worker_status: Optional[str] = None,
    ) -> Dict[str, Any]:
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)
        trained_at = project_now()

        requested_extra_args = dict(extra_args or {})
        requested_validation_strategy = requested_extra_args.pop("validation_strategy", None)
        requested_ad_methods = requested_extra_args.pop("applicability_domain_methods", None)
        requested_similarity_top_k = requested_extra_args.pop("similarity_top_k_neighbors", None)
        requested_similarity_percentile = requested_extra_args.pop(
            "similarity_threshold_percentile", None
        )
        requested_extra_args.setdefault("feature_columns", feature_columns)
        requested_extra_args.setdefault("split_sizes", split_sizes)
        requested_extra_args.setdefault("random_state", random_state)
        requested_extra_args.setdefault("split_type", split_type)
        requested_extra_args.setdefault(
            "validation_protocol", requested_extra_args.get("validation_protocol")
        )

        training_policy = self._apply_training_profile(requested_extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
            seed_policy=training_policy["extra_args"].get("seed_policy"),
            base_seed=training_policy["extra_args"].get("random_state"),
            validation_strategy=requested_validation_strategy,
        )
        training_policy["extra_args"]["random_state"] = protocol_policy["seed_policy"]["model_seed"]
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(target_columns),
        )
        target_column = target_columns[0]
        with S3.open(train_csv, "r") as fh:
            split_source_df = strip_unnamed_columns(pd.read_csv(fh))
        split_source_df = _clean_split_source_target_for_task(
            split_source_df,
            target_column=target_column,
            task_type=task_type,
        )
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

        marker_path = active_marker_path or (root_output_path / ".training_in_progress")
        active_run_record = self._build_active_run_record(
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            trained_at=trained_at,
            active_marker_path=marker_path,
            worker_pid=worker_pid,
            worker_status=worker_status,
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        write_active_training_marker(marker_path, active_run_record)

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
                split_sizes_for_run = split_run.get("split_sizes") or split_sizes
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
                            feature_columns=feature_columns,
                        )
                run_args = {
                    **{
                        key: value
                        for key, value in training_policy["extra_args"].items()
                        if key != "seed_policy"
                    },
                    "feature_columns": feature_columns,
                    "split_sizes": split_sizes_for_run,
                    "split_type": split_run["backend_split_type"],
                    "split_payload": split_payload,
                    "random_state": split_run["seed"],
                    "validation_protocol": protocol_policy["protocol"],
                    "heartbeat_path": str(marker_path),
                    "heartbeat_label": label,
                    "heartbeat_run_index": run_index,
                    "heartbeat_total_runs": len(protocol_policy["split_runs"]),
                }
                if split_run["backend_split_type"] == "final_refit":
                    run_args["final_refit"] = True
                run_args.setdefault("heartbeat_seconds", 120.0)
                run_args.setdefault(
                    "disk_offload_dir", str((run_output_dir / "disk_offload").resolve())
                )

                active_run_record["current_split_label"] = label
                active_run_record["current_split_index"] = run_index
                apply_progress_update(
                    active_run_record,
                    "Training model",
                    {"detail": f"run {run_index} of {len(protocol_policy['split_runs'])}"},
                )
                active_run_record["worker_status"] = "running"
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                write_active_training_marker(marker_path, active_run_record)

                single_result = self.backend.train_model(
                    train_csv=train_csv,
                    output_dir=str(run_output_dir),
                    task=task,
                    extra_args=run_args,
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
            active_run_record["worker_status"] = (
                "completed" if primary_run is not None else "failed"
            )
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            write_active_training_marker(marker_path, active_run_record)

        if primary_run is None:
            raise ValueError("TabICL validation protocol did not produce a primary run.")

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
                    for key, value in training_policy["extra_args"].items()
                    if key != "seed_policy"
                },
                "feature_columns": feature_columns,
                "split_type": "final_refit",
                "split_payload": final_split_payload,
                "random_state": protocol_policy["seed_policy"]["model_seed"],
                "validation_protocol": protocol_policy["protocol"],
                "final_refit": True,
            }
            final_refit_run = self.backend.train_model(
                train_csv=train_csv,
                output_dir=str(final_output_dir),
                task=task,
                extra_args=final_args,
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

        final_primary_run = final_refit_run or primary_run
        root_artifacts = self._materialize_primary_protocol_artifacts(
            root_output_dir=root_output_path,
            primary_run=final_primary_run,
        )
        ad_summary = self._build_applicability_domain(
            train_csv=train_csv,
            primary_run=final_primary_run,
            primary_output_dir=root_output_path,
            task=task,
            feature_columns=feature_columns or final_primary_run.get("feature_columns") or [],
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
        plot_artifacts: Dict[str, str] = {}
        target_column = task.target_columns[0] if task.target_columns else None
        if protocol_policy.get("validation_strategy_type") != "full_train":
            plot_artifacts = build_training_plots_if_possible(
                train_csv=train_csv,
                split_results=split_results,
                primary_run=final_primary_run,
                root_artifacts=root_artifacts,
                root_output_dir=root_output_path,
                target_column=target_column,
                task_type=task.task_type,
            )

        validation_assessment = assess_protocol_results(split_results)
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
        result["selection_metric"] = protocol_policy.get("selection_metric")
        result["final_refit"] = protocol_policy.get("final_refit")
        result["seed_policy"] = protocol_policy["seed_policy"]
        result["seed_policy_report"] = seed_policy_reporting_text(protocol_policy["seed_policy"])
        result["reproducibility"] = seed_policy_reproducibility_metadata(
            protocol_policy["seed_policy"]
        )
        result["split_results"] = split_results
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
            key: value
            for key, value in training_policy["extra_args"].items()
            if key != "seed_policy"
        }
        result["training_resources"] = self._summarize_training_resources(
            compute_env=training_policy["compute_environment"],
            effective_train_args=result["effective_train_args"],
        )
        result["training_durations"] = summarize_training_durations(
            split_results=split_results,
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )
        result["applicability_domain"] = ad_summary
        result["plot_artifacts"] = plot_artifacts
        result["trained_at"] = trained_at.isoformat()
        result["trained_date"] = trained_at.strftime("%d/%m/%Y")
        result["trained_time"] = trained_at.strftime("%H:%M:%S")
        result["train_csv"] = train_csv
        result["target_columns"] = list(target_columns)
        result["feature_columns"] = list(
            feature_columns or (primary_run.get("feature_columns") or [])
        )
        result["canonical_summary_path"] = result["summary_path"]

        summary_path = Path(result["summary_path"])
        write_training_summary(summary_path, result)

        self._sync_training_run_state_from_result(
            prediction_state=prediction_state,
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            task_type=task_type,
            task=task,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            split_results=split_results,
        )
        return result

    def _write_worker_job(
        self,
        *,
        job_dir: Path,
        payload: Dict[str, Any],
    ) -> Path:
        job_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in ("result.json", "error.json", "worker.log"):
            stale_path = job_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        job_path = job_dir / "job.json"
        job_path.write_text(json.dumps(payload, indent=2) + "\n")
        return job_path

    def _run_training_worker(
        self,
        *,
        job_path: Path,
        worker_log_path: Path,
        active_marker_path: Path,
    ) -> Dict[str, Any]:
        result_path = job_path.parent / "result.json"
        error_path = job_path.parent / "error.json"
        command = [
            sys.executable,
            "-m",
            "cs_copilot.tools.prediction.tabicl_train_worker",
            str(job_path),
        ]
        worker_log_path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        last_progress_message: Optional[str] = None
        with worker_log_path.open("w", encoding="utf-8") as log_fh:
            process = subprocess.Popen(
                command,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )

            result_seen_at: Optional[float] = None
            while True:
                return_code = process.poll()
                if result_path.exists() and result_seen_at is None:
                    result_seen_at = time.monotonic()
                if return_code is not None:
                    break
                if active_marker_path.exists():
                    try:
                        marker_payload = json.loads(active_marker_path.read_text())
                    except Exception:
                        marker_payload = None
                    if isinstance(marker_payload, dict):
                        progress_message = marker_payload.get("progress_message")
                        if progress_message and progress_message != last_progress_message:
                            logger.info("%s", progress_message)
                            last_progress_message = progress_message
                if result_seen_at is not None and (time.monotonic() - result_seen_at) >= 2.0:
                    logger.warning(
                        "TabICL worker produced result.json but remained alive; terminating worker pid=%s.",
                        process.pid,
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        logger.warning(
                            "TabICL worker did not terminate cleanly; killing pid=%s.", process.pid
                        )
                        process.kill()
                        process.wait(timeout=5.0)
                    break
                time.sleep(0.5)

        duration_seconds = round(time.monotonic() - start_time, 3)
        if result_path.exists():
            result = json.loads(result_path.read_text())
            result.setdefault("worker_duration_seconds", duration_seconds)
            return result

        if error_path.exists():
            payload = json.loads(error_path.read_text())
            message = payload.get("error_message") or "TabICL worker failed."
            traceback_text = payload.get("traceback")
            if traceback_text:
                raise RuntimeError(f"{message}\n{traceback_text}")
            raise RuntimeError(message)

        log_excerpt = (
            worker_log_path.read_text(encoding="utf-8") if worker_log_path.exists() else ""
        )
        if return_code == -9:
            exit_summary = "return_code=-9 (SIGKILL; often caused by memory pressure/OOM)"
        elif return_code is None:
            exit_summary = "return_code=unknown"
        else:
            exit_summary = f"return_code={return_code}"
        progress_hint = f" last_progress={last_progress_message}" if last_progress_message else ""
        raise RuntimeError(
            "TabICL worker exited without producing result.json or error.json. "
            f"{exit_summary} duration_seconds={duration_seconds}{progress_hint}. "
            "This usually means the worker process was killed before Python could write a structured error; "
            "for TabICL, the most likely cause on large datasets is excessive memory use. "
            f"worker_log={worker_log_path} details={log_excerpt[-4000:]}"
        )

    def train_tabicl_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
        representation_name: Optional[str] = None,
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
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a TabICLv2 model with shared QSAR validation protocols."""
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
        normalized_split_sizes = self._normalize_json_list_argument(
            split_sizes,
            argument_name="split_sizes",
        )

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        root_output_path.mkdir(parents=True, exist_ok=True)

        requested_extra_args, extra_activity_args = split_activity_cliff_args(extra_args)
        requested_hyperparameter_tuning = (
            hyperparameter_tuning
            if hyperparameter_tuning is not None
            else requested_extra_args.pop("hyperparameter_tuning", None)
        )
        requested_outlier_analysis = (
            outlier_analysis
            if outlier_analysis is not None
            else requested_extra_args.pop("outlier_analysis", None)
        )
        requested_validation_strategy = (
            validation_strategy
            if validation_strategy is not None
            else requested_extra_args.pop("validation_strategy", None)
        )
        requested_ad_methods = (
            applicability_domain_methods
            if applicability_domain_methods is not None
            else requested_extra_args.pop("applicability_domain_methods", None)
        )
        requested_similarity_top_k = (
            similarity_top_k_neighbors
            if similarity_top_k_neighbors is not None
            else requested_extra_args.pop("similarity_top_k_neighbors", None)
        )
        requested_similarity_percentile = (
            similarity_threshold_percentile
            if similarity_threshold_percentile is not None
            else requested_extra_args.pop("similarity_threshold_percentile", None)
        )
        if requested_ad_methods is not None:
            requested_extra_args["applicability_domain_methods"] = requested_ad_methods
        if requested_similarity_top_k is not None:
            requested_extra_args["similarity_top_k_neighbors"] = requested_similarity_top_k
        if requested_similarity_percentile is not None:
            requested_extra_args["similarity_threshold_percentile"] = (
                requested_similarity_percentile
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
        requested_extra_args.setdefault("feature_columns", normalized_feature_columns)
        requested_extra_args.setdefault("split_sizes", normalized_split_sizes)
        if random_state is not None:
            requested_extra_args.setdefault("random_state", random_state)
        requested_extra_args.setdefault("split_type", split_type)
        requested_extra_args.setdefault("validation_protocol", validation_protocol)

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

        training_policy = self._apply_training_profile(requested_extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
            seed_policy=training_policy["extra_args"].get("seed_policy"),
            base_seed=training_policy["extra_args"].get("random_state"),
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
        # TabICL deliberately exposes no HPO engine in V1.  Normalize here so an
        # explicit request is rejected before the worker process is created.
        normalize_tuning_config(
            requested_hyperparameter_tuning,
            backend_name="tabicl",
            task_type=task_type,
            eligible=False,
        )
        training_policy["extra_args"]["random_state"] = protocol_policy["seed_policy"]["model_seed"]
        trained_at = project_now()
        active_marker_path = root_output_path / ".training_in_progress"
        prediction_state = get_prediction_state(agent) if agent is not None else None

        active_run_record = self._build_active_run_record(
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            trained_at=trained_at,
            active_marker_path=active_marker_path,
            worker_status="starting",
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = dict(active_run_record)
        write_active_training_marker(active_marker_path, active_run_record)

        job_dir = root_output_path / "_worker_job"
        worker_log_path = job_dir / "worker.log"
        job_payload = {
            "train_csv": train_csv,
            "task_type": task_type,
            "output_dir": resolved_output_dir,
            "target_columns": normalized_target_columns,
            "feature_columns": normalized_feature_columns,
            "representation_name": representation_name,
            "split_type": split_type,
            "split_sizes": normalized_split_sizes,
            "random_state": protocol_policy["seed_policy"]["model_seed"],
            "extra_args": {
                **{
                    key: value
                    for key, value in training_policy["extra_args"].items()
                    if key != "seed_policy"
                },
                "validation_protocol": protocol_policy["protocol"],
                "seed_policy": protocol_policy["seed_policy"],
                "validation_strategy": requested_validation_strategy,
            },
        }
        job_path = self._write_worker_job(job_dir=job_dir, payload=job_payload)

        try:
            result = self._run_training_worker(
                job_path=job_path,
                worker_log_path=worker_log_path,
                active_marker_path=active_marker_path,
            )
        except Exception as exc:
            active_run_record["status"] = "failed"
            active_run_record["worker_status"] = "failed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            write_active_training_marker(active_marker_path, active_run_record)
            raise PredictionExecutionError(f"TabICL worker execution failed: {exc}") from exc

        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=["smiles"],
            target_columns=list(normalized_target_columns),
        )
        outlier_study: Dict[str, Any] = {
            "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
            "status": "skipped" if outlier_skip_reason else "pending",
            "reason": outlier_skip_reason,
            "config": outlier_config.as_dict(),
            "studies": [],
        }
        outlier_variants: List[Dict[str, Any]] = []
        root_summary_path = result.get("canonical_summary_path") or result.get("summary_path")
        # The worker already produced the train-only validation predictions.
        # Reuse them as selection evidence; only the two final refits run in
        # the parent process, with exactly the same direct TabICL parameters.
        if outlier_config.enabled and outlier_skip_reason is None:
            with S3.open(train_csv, "r") as fh:
                source_df = strip_unnamed_columns(pd.read_csv(fh))
            source_df = _clean_split_source_target_for_task(
                source_df,
                target_column=task.target_columns[0],
                task_type=task.task_type,
            )
            for split_result in list(result.get("split_results") or []):
                split_payload = split_result.get("split_payload") or []
                split = split_payload[0] if split_payload else {}
                validation_indices = split.get("val") or split.get("validation") or []
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
                apply_progress_update(
                    active_run_record,
                    "Identifying validation outliers",
                    {"detail": str(split_result.get("strategy_label") or "holdout")},
                )
                write_active_training_marker(active_marker_path, active_run_record)
                study = self._run_outlier_refits(
                    train_csv=train_csv,
                    source_df=source_df,
                    task=task,
                    split_payload=split_payload,
                    selection_prediction_frame=pd.read_csv(Path(str(prediction_path))),
                    output_dir=Path(str(split_result.get("output_dir") or resolved_output_dir)),
                    train_args={
                        **{
                            key: value
                            for key, value in training_policy["extra_args"].items()
                            if key not in {"seed_policy", "split_payload", "validation_strategy"}
                        },
                        "feature_columns": list(
                            normalized_feature_columns or split_result.get("feature_columns") or []
                        ),
                        "random_state": int(split_result.get("seed") or 0),
                        "validation_protocol": protocol_policy["protocol"],
                    },
                    feature_columns=list(
                        normalized_feature_columns or split_result.get("feature_columns") or []
                    ),
                    representation_name=representation_name,
                    activity_args=activity_args,
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                    selection_fraction=outlier_config.selection_fraction,
                    fold_label=str(split_result.get("strategy_label") or "holdout"),
                    repeat_index=split_result.get("repeat_index"),
                    progress_callback=lambda phase, payload: (
                        apply_progress_update(active_run_record, phase, payload),
                        write_active_training_marker(active_marker_path, active_run_record),
                    ),
                )
                study_summary = dict(study["summary"])
                study_summary["split_label"] = split_result.get("strategy_label")
                outlier_study["studies"].append(study_summary)
                for variant in study["variants"]:
                    run = variant.get("run")
                    if isinstance(run, dict):
                        run_output_dir = Path(str(run.get("output_dir") or resolved_output_dir))
                        run["applicability_domain"] = self._build_applicability_domain(
                            train_csv=train_csv,
                            primary_run=run,
                            primary_output_dir=run_output_dir,
                            task=task,
                            feature_columns=list(
                                normalized_feature_columns or run.get("feature_columns") or []
                            ),
                            feature_space=representation_name or run.get("feature_space"),
                            prediction_artifact_paths={"test": run.get("test_predictions_path")},
                            applicability_domain_methods=requested_ad_methods,
                            similarity_top_k_neighbors=requested_similarity_top_k,
                            similarity_threshold_percentile=requested_similarity_percentile,
                        )
                    variant["split_label"] = split_result.get("strategy_label")
                    variant["repeat_index"] = split_result.get("repeat_index")
                    variant["variant_id"] = (
                        f"{safe_slug(str(split_result.get('strategy_label') or 'holdout'))}_"
                        f"{variant.get('variant_id')}"
                    )
                    outlier_variants.append(variant)
            if outlier_variants:
                outlier_study["status"] = "completed"
                primary_label = (result.get("split_results") or [{}])[0].get("strategy_label")
                baseline = next(
                    (
                        item
                        for item in outlier_variants
                        if item.get("split_label") == primary_label
                        and str(item.get("variant_id", "")).endswith("_baseline")
                        and isinstance(item.get("run"), dict)
                    ),
                    None,
                )
                if baseline is not None:
                    result.update(baseline["run"])
                    result["output_dir"] = resolved_output_dir
                    if root_summary_path:
                        result["summary_path"] = root_summary_path
                        result["canonical_summary_path"] = root_summary_path
            elif not outlier_study["studies"]:
                outlier_study["status"] = "skipped"
                outlier_study["reason"] = "Selection validation predictions are unavailable."
        result["outlier_analysis"] = outlier_study
        result["outlier_model_variants"] = outlier_variants
        if outlier_variants:
            result["catalog_model_policy"] = "outlier_variants_no_test_winner"
            result["plot_artifacts"] = {
                **(result.get("plot_artifacts") or {}),
                **{
                    key: value
                    for study in outlier_study.get("studies") or []
                    for key, value in (study.get("plot_artifacts") or {}).items()
                },
            }
        self._sync_training_run_state_from_result(
            prediction_state=prediction_state,
            train_csv=train_csv,
            resolved_output_dir=resolved_output_dir,
            task_type=task_type,
            task=task,
            protocol_policy=protocol_policy,
            training_policy=training_policy,
            split_results=list(result.get("split_results") or []),
        )
        if prediction_state is not None:
            prediction_state["active_training_run"] = None
        result["activity_cliffs"] = activity_cliffs
        result["plot_artifacts"] = result.get("plot_artifacts") or {}
        summary_path = result.get("canonical_summary_path") or result.get("summary_path")
        if summary_path:
            summary_file = Path(str(summary_path)).expanduser()
            try:
                payload = json.loads(summary_file.read_text()) if summary_file.exists() else {}
            except Exception:
                payload = {}
            payload.update(result)
            summary_file.parent.mkdir(parents=True, exist_ok=True)
            summary_file.write_text(json.dumps(payload, indent=2) + "\n")
        return result

    def predict_with_tabicl_from_csv(
        self,
        input_csv: str,
        model_path: str,
        preds_path: str,
        target_columns: Optional[List[str] | str] = None,
        feature_columns: Optional[List[str] | str] = None,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run TabICL batch prediction from a tabular CSV input file."""
        if isinstance(target_columns, str):
            parsed = json.loads(target_columns)
            if not isinstance(parsed, list):
                raise ValueError("target_columns must be a list or a JSON-encoded list.")
            target_columns = parsed
        if isinstance(feature_columns, str):
            parsed = json.loads(feature_columns)
            if not isinstance(parsed, list):
                raise ValueError("feature_columns must be a list or a JSON-encoded list.")
            feature_columns = parsed

        model_record_extra = dict(extra_args or {})
        if feature_columns:
            model_record_extra.setdefault("feature_columns", feature_columns)

        from .backend import PredictionModelRecord

        model_record = PredictionModelRecord(
            model_id=Path(model_path).stem,
            backend_name=self.backend.backend_name,
            model_path=model_path,
            task=PredictionTaskSpec(
                task_type="regression",
                smiles_columns=["smiles"],
                target_columns=list(target_columns or []),
            ),
            inference_profile={"feature_columns": list(feature_columns or [])},
        )
        return self.backend.predict_from_csv(
            input_csv=input_csv,
            model_record=model_record,
            preds_path=preds_path,
            extra_args=model_record_extra,
        )
