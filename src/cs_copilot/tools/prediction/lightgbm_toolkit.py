#!/usr/bin/env python
# coding: utf-8
"""
Toolkit exposing LightGBM-backed tabular QSAR workflows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import (
    build_activity_cliff_loop_comparison_plots,
    prepare_activity_cliff_context,
    split_activity_cliff_args,
)

from .backend import PredictionTaskSpec
from .lightgbm_backend import LightGBMBackend
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
    LEGACY_TABULAR_REPRESENTATION_NAMES,
)
from .training_orchestration import (
    apply_training_profile,
    build_applicability_domain_for_training,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    collect_training_bundle_files,
    materialize_primary_protocol_artifacts,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)

logger = logging.getLogger(__name__)


class LightGBMToolkit(Toolkit):
    """Toolkit exposing LightGBM-backed QSAR orchestration for tabular datasets."""

    def __init__(self, backend: Optional[LightGBMBackend] = None, *, register_tools: bool = True):
        super().__init__("lightgbm_prediction")
        self.backend = backend or LightGBMBackend()
        if register_tools:
            self.register(self.describe_lightgbm_backend)
            self.register(self.describe_lightgbm_environment)
            self.register(self.is_lightgbm_available)
            self.register(self.validate_lightgbm_model_path)
            self.register(self.train_lightgbm_model)
            self.register(self.predict_with_lightgbm_from_csv)

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
                "early_stopping_rounds": 50,
                "n_jobs": 8,
            }
        if profile == "local_standard":
            return {
                **base,
                "n_estimators": 500,
                "early_stopping_rounds": 50,
                "n_jobs": 4,
            }
        return {
            **base,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "n_jobs": 1,
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
                    merged["n_estimators"] = max(int(merged.get("n_estimators", 1000)), 1000)
                    merged["early_stopping_rounds"] = max(
                        int(merged.get("early_stopping_rounds", 50)),
                        50,
                    )
                    merged["n_jobs"] = resolve_backend_n_jobs(
                        compute_env,
                        backend_name="lightgbm",
                        profile=profile,
                        requested_n_jobs=requested_n_jobs,
                    )
                return merged
            if profile == "local_light":
                merged["n_estimators"] = min(int(merged.get("n_estimators", 300)), 300)
                merged["early_stopping_rounds"] = min(
                    int(merged.get("early_stopping_rounds", 30)),
                    30,
                )
            elif profile == "local_standard":
                merged["n_estimators"] = min(int(merged.get("n_estimators", 500)), 500)
                merged["early_stopping_rounds"] = min(
                    int(merged.get("early_stopping_rounds", 50)),
                    50,
                )
            merged["n_jobs"] = resolve_backend_n_jobs(
                compute_env,
                backend_name="lightgbm",
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

    def _resolve_lightgbm_run_artifacts(self, output_dir: Path) -> Dict[str, Path]:
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
                "legacy_representations": list(LEGACY_TABULAR_REPRESENTATION_NAMES),
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

    def train_lightgbm_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        feature_columns: Optional[List[str] | str] = None,
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
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
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

        requested_extra_args, extra_activity_args = split_activity_cliff_args(extra_args)
        requested_validation_strategy = (
            validation_strategy
            if validation_strategy is not None
            else requested_extra_args.pop("validation_strategy", None)
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
        requested_extra_args.setdefault(
            "categorical_feature_columns",
            normalized_categorical_feature_columns,
        )
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
        }
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
            )
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
                        for key, value in training_policy["extra_args"].items()
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
                active_run_record["progress_message"] = (
                    "LightGBM training progress: "
                    f"run {run_index}/{len(protocol_policy['split_runs'])} - {label}"
                )
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                write_active_training_marker(active_marker_path, active_run_record)

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
                            for key, value in training_policy["extra_args"].items()
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
                    active_run_record["progress_message"] = (
                        "LightGBM activity-cliff loop training: "
                        f"{variant_id} on fixed {label} holdout"
                    )
                    if prediction_state is not None:
                        prediction_state["active_training_run"] = dict(active_run_record)
                    write_active_training_marker(active_marker_path, active_run_record)

                    variant_result = self.backend.train_model(
                        train_csv=train_csv,
                        output_dir=str(run_output_dir),
                        task=task,
                        extra_args=run_args,
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
            final_split_payload = build_full_train_split_payload(df=split_source_df)
            final_args = {
                **{
                    key: value
                    for key, value in training_policy["extra_args"].items()
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

        final_split_results = split_results
        final_primary_run = final_refit_run or primary_run
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
        result["selection_metric"] = protocol_policy.get("selection_metric")
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
                for key, value in training_policy["extra_args"].items()
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
        result["plot_artifacts"] = plot_artifacts
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

        bundle_path = (
            Path(".files")
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

        bundle = bundle_artifacts(bundle_path, bundle_files)
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
        extra_args: Optional[Dict[str, Any]] = None,
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
            extra_args=extra_args,
        )
