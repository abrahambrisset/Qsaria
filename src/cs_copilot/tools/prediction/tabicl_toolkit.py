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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import prepare_activity_cliff_context, split_activity_cliff_args

from .backend import PredictionExecutionError, PredictionTaskSpec
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
from .qsar_splitters import (
    build_full_train_split_payload,
    build_qsar_split_payload,
    build_repeated_kfold_split_payloads,
)
from .tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    LEGACY_TABULAR_REPRESENTATION_NAMES,
)
from .session_state import (
    get_prediction_state,
    write_active_training_marker,
)
from .training_orchestration import (
    apply_training_profile,
    build_applicability_domain_for_training,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)
from .tabicl_backend import (
    DEFAULT_TABICL_CHECKPOINT_DIR,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
    TabICLBackend,
)

logger = logging.getLogger(__name__)


class TabICLToolkit(Toolkit):
    """Backend-specific TabICL training toolkit used behind QSARTrainingToolkit."""

    def __init__(self, backend: Optional[TabICLBackend] = None, *, register_tools: bool = True):
        super().__init__("tabicl_prediction")
        self.backend = backend or TabICLBackend()
        if register_tools:
            self.register(self.describe_tabicl_backend)
            self.register(self.describe_tabicl_environment)
            self.register(self.is_tabicl_available)
            self.register(self.validate_tabicl_model_path)
            self.register(self.validate_tabicl_checkpoint_path)
            self.register(self.train_tabicl_model)
            self.register(self.predict_with_tabicl_from_csv)

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

        def _limit(profile: str, merged: Dict[str, Any], allow_heavy_compute: bool) -> Dict[str, Any]:
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

    def _resolve_tabicl_run_artifacts(self, output_dir: Path) -> Dict[str, Path]:
        model_dir = output_dir / "model_0"
        return {
            "best_model_path": model_dir / "best.pkl",
            "test_predictions_path": model_dir / "test_predictions.csv",
            "config_path": output_dir / "config.toml",
            "splits_path": output_dir / "splits.json",
        }

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
    ) -> Dict[str, Any]:
        return build_applicability_domain_for_training(
            train_csv=train_csv,
            primary_run=primary_run,
            primary_output_dir=primary_output_dir,
            task=task,
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
                "supported_validation_protocols": [
                    "fast_local",
                    "standard_qsar",
                    "robust_qsar",
                    "challenging_qsar",
                ],
                "supported_split_families": ["random", "scaffold", "cluster_kmeans"],
                "default_tabular_feature_policy": {
                    "automatic_representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
                    "legacy_representations": list(LEGACY_TABULAR_REPRESENTATION_NAMES),
                    "default_single_representation": "morgan_rdkit_all",
                    "explicit_user_override": True,
                },
                "notes": [
                    "Supports the shared QSAR protocol names and split families.",
                    "Modern comparative campaigns use RDKit all, Morgan binary, Morgan count, and the complete combined pack.",
                    "RDKit basic representations are legacy-only and require explicit user override.",
                    "If the user explicitly requests a representation, that override should win.",
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

    def validate_tabicl_checkpoint_path(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Validate the persisted TabICL base checkpoint path."""
        path = Path(checkpoint_path or (DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT))
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
        requested_extra_args.setdefault("feature_columns", feature_columns)
        requested_extra_args.setdefault("split_sizes", split_sizes)
        requested_extra_args.setdefault("random_state", random_state)
        requested_extra_args.setdefault("split_type", split_type)
        requested_extra_args.setdefault("validation_protocol", requested_extra_args.get("validation_protocol"))

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
        if target_column in split_source_df.columns:
            split_source_df[target_column] = pd.to_numeric(split_source_df[target_column], errors="coerce")
            split_source_df = split_source_df.dropna(subset=[target_column]).reset_index(drop=True)
        is_cv_protocol = protocol_policy.get("validation_strategy_type") == "cross_validation"
        cv_split_payloads: Dict[str, List[Dict[str, Any]]] = {}
        if is_cv_protocol:
            cv_strategy = protocol_policy.get("validation_strategy") or {}
            cv_split_payloads = build_repeated_kfold_split_payloads(
                df=split_source_df,
                n_splits=int(cv_strategy.get("n_folds") or cv_strategy.get("n_splits") or 5),
                n_repeats=int(cv_strategy.get("n_repeats") or 1),
                random_state=int(cv_strategy.get("seed") or protocol_policy["seed_policy"].get("model_seed") or 0),
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
                    root_output_path / f"{safe_slug(label)}_split" if multi_run_protocol else root_output_path
                )
                run_output_dir.mkdir(parents=True, exist_ok=True)
                started_at = project_now()
                split_payload = split_run.get("split_payload")
                split_sizes_for_run = split_run.get("split_sizes") or split_sizes
                if split_payload is None:
                    if label in cv_split_payloads:
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
                    **{key: value for key, value in training_policy["extra_args"].items() if key != "seed_policy"},
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
                run_args.setdefault("heartbeat_seconds", 120.0)
                run_args.setdefault("disk_offload_dir", str((run_output_dir / "disk_offload").resolve()))

                active_run_record["current_split_label"] = label
                active_run_record["current_split_index"] = run_index
                active_run_record["progress_message"] = (
                    f"TabICL training progress: run {run_index}/{len(protocol_policy['split_runs'])} - {label}"
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
                single_result["started_at"] = single_result.get("started_at") or started_at.isoformat()
                single_result["completed_at"] = single_result.get("completed_at") or completed_at.isoformat()
                single_result["duration_seconds"] = single_result.get("duration_seconds") or round(
                    (completed_at - started_at).total_seconds(), 3
                )
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
        finally:
            active_run_record["status"] = "completed" if primary_run is not None else "failed"
            active_run_record["completed_at"] = project_now().isoformat()
            active_run_record["worker_status"] = "completed" if primary_run is not None else "failed"
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
            final_split_payload = build_full_train_split_payload(df=split_source_df)
            final_args = {
                **{key: value for key, value in training_policy["extra_args"].items() if key != "seed_policy"},
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
            final_refit_run["started_at"] = final_refit_run.get("started_at") or final_started_at.isoformat()
            final_refit_run["completed_at"] = final_refit_run.get("completed_at") or final_completed_at.isoformat()
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
        )
        plot_artifacts: Dict[str, str] = {}
        target_column = task.target_columns[0] if task.target_columns else None
        plot_artifacts = build_training_plots_if_possible(
            train_csv=train_csv,
            split_results=split_results,
            primary_run=final_primary_run,
            root_artifacts=root_artifacts,
            root_output_dir=root_output_path,
            target_column=target_column,
        )

        validation_assessment = assess_protocol_results(split_results)
        total_completed_at = project_now()
        result = dict(final_primary_run)
        result["output_dir"] = resolved_output_dir
        result["model_path"] = root_artifacts.get("best_model_path") or final_primary_run.get("model_path")
        result["summary_path"] = str(root_output_path / "cs_copilot_training_summary.json")
        result["config_path"] = root_artifacts.get("config_path") or final_primary_run.get("config_path")
        result["splits_path"] = root_artifacts.get("splits_path") or final_primary_run.get("splits_path")
        result["test_predictions_path"] = root_artifacts.get("test_predictions_path") or final_primary_run.get(
            "test_predictions_path"
        )
        result["validation_protocol"] = protocol_policy["protocol"]
        result["validation_protocol_reason"] = protocol_policy["reason"]
        result["validation_strategy"] = protocol_policy.get("validation_strategy")
        result["validation_strategy_type"] = protocol_policy.get("validation_strategy_type")
        result["validation_aggregation"] = protocol_policy.get("aggregation")
        result["selection_metric"] = protocol_policy.get("selection_metric")
        result["final_refit"] = protocol_policy.get("final_refit")
        result["seed_policy"] = protocol_policy["seed_policy"]
        result["seed_policy_report"] = seed_policy_reporting_text(protocol_policy["seed_policy"])
        result["reproducibility"] = seed_policy_reproducibility_metadata(protocol_policy["seed_policy"])
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
            key: value for key, value in training_policy["extra_args"].items() if key != "seed_policy"
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
        result["feature_columns"] = list(feature_columns or (primary_run.get("feature_columns") or []))
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
                        logger.warning("TabICL worker did not terminate cleanly; killing pid=%s.", process.pid)
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

        log_excerpt = worker_log_path.read_text(encoding="utf-8") if worker_log_path.exists() else ""
        raise RuntimeError(
            "TabICL worker exited without producing result.json or error.json. "
            f"worker_log={worker_log_path} details={log_excerpt[-4000:]}"
        )

    def train_tabicl_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
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
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a TabICLv2 regressor with shared QSAR validation protocols."""
        normalized_target_columns = self._normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []
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
        requested_validation_strategy = (
            validation_strategy if validation_strategy is not None else requested_extra_args.pop("validation_strategy", None)
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
            "split_type": split_type,
            "split_sizes": normalized_split_sizes,
            "random_state": protocol_policy["seed_policy"]["model_seed"],
            "extra_args": {
                **{
                    key: value for key, value in training_policy["extra_args"].items() if key != "seed_policy"
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
