#!/usr/bin/env python
# coding: utf-8
"""
Benchmark orchestration for multi-backend QSAR campaigns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from .model_registry_toolkit import ModelRegistryToolkit
from .qsar_training_policy import (
    describe_compute_environment,
    resolve_training_profile,
    safe_slug,
    seed_policy_reporting_text,
    seed_policy_reproducibility_metadata,
)
from .qsar_training_toolkit import QSARTrainingToolkit
from .qsar_validation_strategy import resolve_validation_strategy
from .tabular_representations import (
    get_tabular_representation,
    tabular_candidates_for_backend,
)

logger = logging.getLogger(__name__)


def _coerce_list(value: Optional[List[str] | str]) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            if "," in stripped:
                return [item.strip() for item in stripped.split(",") if item.strip()]
            return [stripped]
        if not isinstance(parsed, list):
            if isinstance(parsed, str):
                return [parsed]
            raise ValueError("Expected a list, scalar string, or JSON-encoded list.")
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _benchmark_dataset_token(train_csv: str) -> str:
    stem = Path(train_csv).stem.lower()
    suffixes = (
        "_tabular_qsar",
        "_tabular",
        "_normalized",
        "_prepared",
        "_curated",
        "_cleaned",
        "_dataset",
        "_training",
    )
    for suffix in suffixes:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for trailing in ("_train", "_test"):
        if stem.endswith(trailing):
            stem = stem[: -len(trailing)]
            break
    return safe_slug(stem) or "dataset"


def _benchmark_target_token(target_columns: List[str]) -> str:
    if not target_columns:
        return "target"
    return safe_slug(str(target_columns[0])) or "target"


class BenchmarkToolkit(Toolkit):
    """Toolkit orchestrating multi-backend QSAR benchmark campaigns."""

    BENCHMARK_MODE_TO_PROTOCOL = {
        "benchmark_fast_local": "fast_local",
        "benchmark_standard_qsar": "standard_qsar",
        "benchmark_robust_qsar": "robust_qsar",
        "benchmark_challenging_qsar": "challenging_qsar",
    }
    BENCHMARK_MODE_ALIASES = {
        "fast_local": "benchmark_fast_local",
        "standard_qsar": "benchmark_standard_qsar",
        "robust_qsar": "benchmark_robust_qsar",
        "challenging_qsar": "benchmark_challenging_qsar",
    }

    def __init__(
        self,
        *,
        training_toolkit: Optional[QSARTrainingToolkit] = None,
        registry_toolkit: Optional[ModelRegistryToolkit] = None,
    ):
        super().__init__("benchmark_prediction")
        self.training_toolkit = training_toolkit or QSARTrainingToolkit()
        self.registry_toolkit = registry_toolkit
        self.register(self.benchmark_qsar_models)

    def _get_registry_toolkit(self) -> ModelRegistryToolkit:
        if self.registry_toolkit is None:
            self.registry_toolkit = ModelRegistryToolkit(
                backends=self.training_toolkit.backend_mapping(),
                default_backend_name="chemprop",
                register_tools=False,
            )
        return self.registry_toolkit

    def _resolve_benchmark_protocol(self, benchmark_mode: str) -> str:
        normalized = benchmark_mode.strip().lower()
        normalized = self.BENCHMARK_MODE_ALIASES.get(normalized, normalized)
        protocol = self.BENCHMARK_MODE_TO_PROTOCOL.get(normalized)
        if protocol is None:
            raise ValueError(
                "Unsupported benchmark_mode. Expected one of "
                f"{sorted(self.BENCHMARK_MODE_TO_PROTOCOL)} "
                f"or aliases {sorted(self.BENCHMARK_MODE_ALIASES)}."
            )
        return protocol

    def _resolve_compute_profile(self) -> Dict[str, Any]:
        compute_env = describe_compute_environment()
        resolved = resolve_training_profile(compute_env)
        return {
            "compute_environment": compute_env,
            "training_profile": resolved["profile"],
            "profile_reason": resolved["reason"],
        }

    def _resolve_backends(
        self,
        *,
        task_type: str,
        target_columns: List[str],
        requested_backends: Optional[List[str]],
    ) -> List[str]:
        backends = self.training_toolkit.backend_mapping()
        available: List[str] = []
        if backends["chemprop"].is_available():
            available.append("chemprop")
        if (
            backends["lightgbm"].is_available()
            and task_type == "regression"
            and len(target_columns) == 1
        ):
            available.append("lightgbm")
        if (
            backends["tabicl"].is_available()
            and task_type == "regression"
            and len(target_columns) == 1
        ):
            available.append("tabicl")

        if requested_backends:
            filtered = [name for name in requested_backends if name in available]
            if not filtered:
                raise ValueError(
                    f"No requested backends are available or compatible: {requested_backends}"
                )
            return filtered
        return available

    def _expand_lightgbm_candidates(
        self,
        *,
        include_candidate_variants: bool,
        training_profile: str,
        benchmark_protocol: str = "standard_qsar",
    ) -> List[Dict[str, Any]]:
        return tabular_candidates_for_backend(
            "lightgbm",
            single_default_protocol=None if include_candidate_variants else benchmark_protocol,
            training_profile=training_profile,
        )

    def _expand_tabicl_candidates(
        self,
        *,
        include_candidate_variants: bool,
        requested_variants: Optional[List[str]],
        training_profile: str,
        benchmark_protocol: str = "standard_qsar",
    ) -> List[Dict[str, Any]]:
        if requested_variants:
            representation_names: List[str] = []
            for name in requested_variants:
                normalized = str(name).strip().lower()
                if normalized.startswith("tabicl_"):
                    normalized = normalized.replace("tabicl_", "", 1)
                get_tabular_representation(normalized)
                representation_names.append(normalized)
            return tabular_candidates_for_backend(
                "tabicl",
                representation_names=representation_names,
                training_profile=training_profile,
            )

        return tabular_candidates_for_backend(
            "tabicl",
            single_default_protocol=None if include_candidate_variants else benchmark_protocol,
            training_profile=training_profile,
        )

    def _expand_candidates(
        self,
        *,
        task_type: str,
        target_columns: List[str],
        requested_backends: Optional[List[str]],
        include_candidate_variants: bool,
        requested_tabicl_variants: Optional[List[str]],
        training_profile: str,
        benchmark_protocol: str = "standard_qsar",
    ) -> List[Dict[str, Any]]:
        backends = self._resolve_backends(
            task_type=task_type,
            target_columns=target_columns,
            requested_backends=requested_backends,
        )
        candidates: List[Dict[str, Any]] = []
        if "chemprop" in backends:
            candidates.append(
                {
                    "candidate_id": "chemprop_default",
                    "backend_name": "chemprop",
                    "representation_name": "molecular_graph",
                }
            )
        if "lightgbm" in backends:
            candidates.extend(
                self._expand_lightgbm_candidates(
                    include_candidate_variants=include_candidate_variants,
                    training_profile=training_profile,
                    benchmark_protocol=benchmark_protocol,
                )
            )
        if "tabicl" in backends:
            candidates.extend(
                self._expand_tabicl_candidates(
                    include_candidate_variants=include_candidate_variants,
                    requested_variants=requested_tabicl_variants,
                    training_profile=training_profile,
                    benchmark_protocol=benchmark_protocol,
                )
            )
        return candidates

    def _train_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        train_csv: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        benchmark_protocol: str,
        candidate_dir: Path,
        allow_heavy_compute: bool,
        training_profile: Optional[str],
        campaign_seed_policy: Dict[str, Any],
        validation_strategy: Optional[Dict[str, Any]],
        agent: Agent,
    ) -> Dict[str, Any]:
        requested_extra_args: Dict[str, Any] = {
            "validation_protocol": benchmark_protocol,
            "allow_heavy_compute": allow_heavy_compute,
            "seed_policy": campaign_seed_policy,
            "feature_cache_dir": str(candidate_dir.parent / "feature_cache"),
        }
        if training_profile:
            requested_extra_args["training_profile"] = training_profile
        if validation_strategy is not None:
            requested_extra_args["validation_strategy"] = validation_strategy

        result = self.training_toolkit.train_qsar_model(
            train_csv=train_csv,
            backend_name=candidate["backend_name"],
            task_type=task_type,
            output_dir=str(candidate_dir),
            smiles_column=smiles_column,
            target_columns=list(target_columns),
            validation_protocol=benchmark_protocol,
            representation_name=(
                None
                if candidate["backend_name"] == "chemprop"
                else candidate["representation_name"]
            ),
            extra_args=requested_extra_args,
            validation_strategy=validation_strategy,
            agent=agent,
        )
        result["representation_name"] = candidate["representation_name"]
        result.setdefault("candidate_train_csv", result.get("train_csv") or train_csv)
        return result

    def _display_name_for_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        target_columns: List[str],
    ) -> str:
        target_label = ", ".join(target_columns)
        return f"{candidate['backend_name'].upper()} {candidate['representation_name']} benchmark model for {target_label}"

    def _register_and_persist_candidate(
        self,
        *,
        candidate: Dict[str, Any],
        result: Dict[str, Any],
        train_csv: str,
        benchmark_mode: str,
        benchmark_protocol: str,
        campaign_root: Path,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        training_profile: str,
        campaign_seed_policy: Dict[str, Any],
        agent: Agent,
    ) -> Dict[str, Any]:
        temporary_model_id = f"{candidate['candidate_id']}_session"
        model_path = result.get("best_model_path") or result.get("model_path")
        if not model_path:
            raise ValueError(
                f"Candidate {candidate['candidate_id']} did not produce a model artifact path."
            )

        metrics_payload = (
            ((result.get("validation_assessment") or {}).get("aggregated_split_metrics"))
            or result.get("metrics")
            or {}
        )
        registry_toolkit = self._get_registry_toolkit()
        registry_toolkit.register_model(
            model_id=temporary_model_id,
            model_path=str(model_path),
            backend_name=candidate["backend_name"],
            task_type=task_type,
            smiles_columns=[smiles_column],
            target_columns=list(target_columns),
            description=f"Benchmark candidate {candidate['candidate_id']} trained under {benchmark_mode}.",
            tags={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
            },
            version="1.0.0",
            status=((result.get("validation_assessment") or {}).get("governance") or {}).get(
                "recommended_status",
                "experimental",
            ),
            owner="qsar_training_agent",
            source=result.get("candidate_train_csv") or result.get("train_csv"),
            known_metrics=metrics_payload if isinstance(metrics_payload, dict) else {},
            training_data_summary={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
                "benchmark_dataset_name": _benchmark_dataset_token(train_csv),
                "benchmark_target_name": _benchmark_target_token(target_columns),
                "validation_protocol": benchmark_protocol,
                "training_profile": training_profile,
                "campaign_root": str(campaign_root),
                "seed_policy": result.get("seed_policy") or campaign_seed_policy,
                "seed_policy_report": seed_policy_reporting_text(
                    result.get("seed_policy") or campaign_seed_policy
                ),
                "campaign_seed_policy": campaign_seed_policy,
                "reproducibility": seed_policy_reproducibility_metadata(
                    result.get("seed_policy") or campaign_seed_policy
                ),
                "feature_preparation": result.get("feature_preparation") or {},
                "feature_preparation_durations": result.get("feature_preparation_durations") or {},
                "trained_at": result.get("trained_at"),
            },
            inference_profile={
                "feature_columns": list(result.get("feature_columns") or []),
                "categorical_feature_columns": list(
                    result.get("categorical_feature_columns") or []
                ),
                "representation_name": candidate["representation_name"],
            },
            selection_hints={
                "benchmark_mode": benchmark_mode,
                "candidate_id": candidate["candidate_id"],
                "representation_name": candidate["representation_name"],
            },
            applicability_domain=result.get("applicability_domain") or {},
            agent=agent,
        )
        persisted = registry_toolkit.persist_registered_model(
            model_id=temporary_model_id,
            display_name=self._display_name_for_candidate(
                candidate=candidate,
                target_columns=target_columns,
            ),
            version="1.0.0",
            agent=agent,
        )
        return persisted

    def _candidate_summary_row(
        self,
        *,
        candidate: Dict[str, Any],
        result: Dict[str, Any],
        persisted: Dict[str, Any],
        benchmark_protocol: str,
    ) -> Dict[str, Any]:
        validation_assessment = result.get("validation_assessment") or {}
        aggregated = validation_assessment.get("aggregated_split_metrics") or {}
        hardest_split = validation_assessment.get("hardest_split")
        hardest_family = aggregated.get(hardest_split) if hardest_split else None
        random_family = aggregated.get("random") or {}
        scaffold_family = aggregated.get("scaffold") or {}
        kmeans_family = aggregated.get("cluster_kmeans") or {}
        hardest_r2 = _safe_float(
            (hardest_family or {}).get("r2_mean") or (hardest_family or {}).get("r2")
        )
        hardest_balanced_accuracy = _safe_float(
            (hardest_family or {}).get("balanced_accuracy_mean")
            or (hardest_family or {}).get("balanced_accuracy")
        )
        delta_vs_random = validation_assessment.get("delta_vs_random") or {}
        hardest_delta_r2 = None
        hardest_delta_balanced_accuracy = None
        if hardest_split:
            hardest_delta_r2 = _safe_float((delta_vs_random.get(hardest_split) or {}).get("r2"))
            hardest_delta_balanced_accuracy = _safe_float(
                (delta_vs_random.get(hardest_split) or {}).get("balanced_accuracy")
            )

        row = {
            "candidate_id": candidate["candidate_id"],
            "model_id": persisted.get("model_id"),
            "backend": candidate["backend_name"],
            "representation": candidate["representation_name"],
            "protocol": benchmark_protocol,
            "random_r2": _safe_float(random_family.get("r2_mean") or random_family.get("r2")),
            "scaffold_r2": _safe_float(scaffold_family.get("r2_mean") or scaffold_family.get("r2")),
            "cluster_kmeans_r2": _safe_float(
                kmeans_family.get("r2_mean") or kmeans_family.get("r2")
            ),
            "random_balanced_accuracy": _safe_float(
                random_family.get("balanced_accuracy_mean")
                or random_family.get("balanced_accuracy")
            ),
            "scaffold_balanced_accuracy": _safe_float(
                scaffold_family.get("balanced_accuracy_mean")
                or scaffold_family.get("balanced_accuracy")
            ),
            "cluster_kmeans_balanced_accuracy": _safe_float(
                kmeans_family.get("balanced_accuracy_mean")
                or kmeans_family.get("balanced_accuracy")
            ),
            "hardest_split": hardest_split,
            "hardest_split_r2": hardest_r2,
            "hardest_split_balanced_accuracy": hardest_balanced_accuracy,
            "delta_r2": hardest_delta_r2,
            "delta_balanced_accuracy": hardest_delta_balanced_accuracy,
            "roc_auc": _safe_float(
                (hardest_family or random_family).get("roc_auc_mean")
                or (hardest_family or random_family).get("roc_auc")
            ),
            "f1_macro": _safe_float(
                (hardest_family or random_family).get("f1_macro_mean")
                or (hardest_family or random_family).get("f1_macro")
            ),
            "random_family_r2_mean": _safe_float(random_family.get("r2_mean")),
            "random_family_r2_std": _safe_float(random_family.get("r2_std")),
            "random_family_balanced_accuracy_mean": _safe_float(
                random_family.get("balanced_accuracy_mean")
            ),
            "random_family_balanced_accuracy_std": _safe_float(
                random_family.get("balanced_accuracy_std")
            ),
            "train_time_s": _safe_float(
                (result.get("training_durations") or {}).get("total_duration_seconds")
            ),
            "status": persisted.get("status"),
            "internal_model_root": persisted.get("model_root"),
            "best_model_path": persisted.get("model_path"),
            "feature_cache_key": (result.get("feature_preparation") or {}).get("feature_cache_key"),
            "feature_cache_status": (result.get("feature_preparation") or {}).get(
                "feature_cache_status"
            ),
            "feature_cache_hits": (result.get("feature_preparation") or {}).get("cache_hits"),
            "feature_cache_misses": (result.get("feature_preparation") or {}).get("cache_misses"),
        }
        return row

    def _compact_candidate_result(
        self,
        *,
        result: Dict[str, Any],
        row: Dict[str, Any],
    ) -> Dict[str, Any]:
        validation_assessment = result.get("validation_assessment") or {}
        split_summaries: List[Dict[str, Any]] = []
        for split_result in result.get("split_results") or []:
            metrics = (split_result.get("metrics") or {}).get("test") or {}
            split_summaries.append(
                {
                    "strategy_label": split_result.get("strategy_label"),
                    "test_count": split_result.get("test_count"),
                    "metrics": {
                        "r2": _safe_float(metrics.get("r2")),
                        "rmse": _safe_float(metrics.get("rmse")),
                        "mae": _safe_float(metrics.get("mae")),
                        "mse": _safe_float(metrics.get("mse")),
                        "balanced_accuracy": _safe_float(metrics.get("balanced_accuracy")),
                        "roc_auc": _safe_float(metrics.get("roc_auc")),
                        "f1_macro": _safe_float(metrics.get("f1_macro")),
                    },
                }
            )
        return {
            "candidate_id": row.get("candidate_id"),
            "model_id": row.get("model_id"),
            "backend_name": row.get("backend"),
            "representation_name": row.get("representation"),
            "status": row.get("status"),
            "hardest_split": row.get("hardest_split"),
            "hardest_split_r2": row.get("hardest_split_r2"),
            "hardest_split_balanced_accuracy": row.get("hardest_split_balanced_accuracy"),
            "random_r2": row.get("random_r2"),
            "random_balanced_accuracy": row.get("random_balanced_accuracy"),
            "scaffold_r2": row.get("scaffold_r2"),
            "scaffold_balanced_accuracy": row.get("scaffold_balanced_accuracy"),
            "cluster_kmeans_r2": row.get("cluster_kmeans_r2"),
            "cluster_kmeans_balanced_accuracy": row.get("cluster_kmeans_balanced_accuracy"),
            "delta_r2": row.get("delta_r2"),
            "delta_balanced_accuracy": row.get("delta_balanced_accuracy"),
            "roc_auc": row.get("roc_auc"),
            "f1_macro": row.get("f1_macro"),
            "train_time_s": row.get("train_time_s"),
            "governance": validation_assessment.get("governance"),
            "split_results": split_summaries,
            "internal_model_root": row.get("internal_model_root"),
            "best_model_path": row.get("best_model_path"),
            "feature_preparation": result.get("feature_preparation") or {},
        }

    def _rank_summary_rows(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(row: Dict[str, Any]) -> tuple[float, float, float, float, float]:
            classification_row = (
                row.get("hardest_split_balanced_accuracy") is not None
                or row.get("random_balanced_accuracy") is not None
            )
            hardest_metric = row.get(
                "hardest_split_balanced_accuracy" if classification_row else "hardest_split_r2"
            )
            delta_metric = row.get("delta_balanced_accuracy" if classification_row else "delta_r2")
            random_metric = row.get(
                "random_balanced_accuracy" if classification_row else "random_r2"
            )
            tie_breaker = row.get("roc_auc") if classification_row else None
            train_time = row.get("train_time_s")
            return (
                -(hardest_metric if hardest_metric is not None else float("-inf")),
                -(tie_breaker if tie_breaker is not None else float("-inf")),
                -(delta_metric if delta_metric is not None else float("-inf")),
                -(random_metric if random_metric is not None else float("-inf")),
                train_time if train_time is not None else float("inf"),
            )

        return sorted(rows, key=sort_key)

    def _resolve_recommendations(
        self,
        rows: List[Dict[str, Any]],
        benchmark_protocol: str,
    ) -> Dict[str, Optional[str]]:
        ranked = self._rank_summary_rows(rows)
        best_overall = ranked[0]["candidate_id"] if ranked else None
        classification_rows = any(
            row.get("hardest_split_balanced_accuracy") is not None
            or row.get("random_balanced_accuracy") is not None
            for row in rows
        )
        best_hardest = max(
            rows,
            key=lambda row: (
                row.get(
                    "hardest_split_balanced_accuracy" if classification_rows else "hardest_split_r2"
                )
                if row.get(
                    "hardest_split_balanced_accuracy" if classification_rows else "hardest_split_r2"
                )
                is not None
                else float("-inf")
            ),
            default=None,
        )
        best_fast = min(
            rows,
            key=lambda row: (
                row.get("train_time_s") if row.get("train_time_s") is not None else float("inf")
            ),
            default=None,
        )
        best_stability = None
        if benchmark_protocol == "robust_qsar":
            stability_key = (
                "random_family_balanced_accuracy_std"
                if classification_rows
                else "random_family_r2_std"
            )
            stability_rows = [row for row in rows if row.get(stability_key) is not None]
            if stability_rows:
                best_stability = min(
                    stability_rows,
                    key=lambda row: row.get(stability_key, float("inf")),
                )

        return {
            "best_overall_candidate": best_overall,
            "best_hardest_split_candidate": best_hardest["candidate_id"] if best_hardest else None,
            "best_stability_candidate": best_stability["candidate_id"] if best_stability else None,
            "best_fast_candidate": best_fast["candidate_id"] if best_fast else None,
            "recommended_candidate_for_followup": best_overall,
        }

    def _render_benchmark_report(
        self,
        *,
        benchmark_mode: str,
        benchmark_protocol: str,
        compute_environment: Dict[str, Any],
        candidate_rows: List[Dict[str, Any]],
        candidate_results: List[Dict[str, Any]],
        recommendations: Dict[str, Optional[str]],
        campaign_seed_policy: Dict[str, Any],
    ) -> str:
        lines: List[str] = []
        lines.append(f"# Benchmark QSAR Report — {benchmark_mode}")
        lines.append("")
        lines.append("## Executive summary")
        lines.append("")
        lines.append(f"- Benchmark mode: `{benchmark_mode}`")
        lines.append(f"- Base protocol: `{benchmark_protocol}`")
        lines.append(f"- Candidates compared: `{len(candidate_rows)}`")
        lines.append(f"- {seed_policy_reporting_text(campaign_seed_policy)}")
        for key, value in recommendations.items():
            if value:
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
        lines.append("")
        lines.append("## Benchmark context")
        lines.append("")
        lines.append(f"- Execution environment: `{compute_environment.get('execution_env')}`")
        lines.append(f"- CPU count: `{compute_environment.get('cpu_count')}`")
        lines.append(f"- GPU available: `{compute_environment.get('gpu_available')}`")
        lines.append(f"- Suggested profile: `{compute_environment.get('suggested_profile')}`")
        lines.append(f"- Campaign seed: `{campaign_seed_policy.get('campaign_seed')}`")
        split_seed_text = ", ".join(
            f"{item.get('label')}={item.get('seed')}"
            for item in campaign_seed_policy.get("split_runs", [])
        )
        if split_seed_text:
            lines.append(f"- Shared split seeds: `{split_seed_text}`")
        cache_hits = sum(
            int(((result.get("feature_preparation") or {}).get("cache_hits") or 0))
            for result in candidate_results
        )
        cache_misses = sum(
            int(((result.get("feature_preparation") or {}).get("cache_misses") or 0))
            for result in candidate_results
        )
        if cache_hits or cache_misses:
            lines.append(f"- Feature cache hits/misses: `{cache_hits}` / `{cache_misses}`")
        lines.append("")
        lines.append("## Candidate inventory")
        lines.append("")
        for row in candidate_rows:
            lines.append(
                f"- `{row['candidate_id']}` — backend `{row['backend']}`, representation `{row['representation']}`, model `{row['model_id']}`"
            )
        lines.append("")
        lines.append("## Leaderboard")
        lines.append("")
        classification_rows = any(
            row.get("random_balanced_accuracy") is not None for row in candidate_rows
        )
        if classification_rows:
            lines.append(
                "| candidate_id | backend | representation | hardest_split | hardest_balanced_accuracy | random_balanced_accuracy | roc_auc | f1_macro | train_time_s | status |"
            )
            lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---|")
            for row in self._rank_summary_rows(candidate_rows):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("candidate_id") or ""),
                            str(row.get("backend") or ""),
                            str(row.get("representation") or ""),
                            str(row.get("hardest_split") or ""),
                            (
                                f"{row['hardest_split_balanced_accuracy']:.3f}"
                                if row.get("hardest_split_balanced_accuracy") is not None
                                else ""
                            ),
                            (
                                f"{row['random_balanced_accuracy']:.3f}"
                                if row.get("random_balanced_accuracy") is not None
                                else ""
                            ),
                            f"{row['roc_auc']:.3f}" if row.get("roc_auc") is not None else "",
                            f"{row['f1_macro']:.3f}" if row.get("f1_macro") is not None else "",
                            (
                                f"{row['train_time_s']:.3f}"
                                if row.get("train_time_s") is not None
                                else ""
                            ),
                            str(row.get("status") or ""),
                        ]
                    )
                    + " |"
                )
        else:
            lines.append(
                "| candidate_id | backend | representation | hardest_split | hardest_split_r2 | random_r2 | scaffold_r2 | cluster_kmeans_r2 | delta_r2 | train_time_s | status |"
            )
            lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
            for row in self._rank_summary_rows(candidate_rows):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row.get("candidate_id") or ""),
                            str(row.get("backend") or ""),
                            str(row.get("representation") or ""),
                            str(row.get("hardest_split") or ""),
                            (
                                f"{row['hardest_split_r2']:.3f}"
                                if row.get("hardest_split_r2") is not None
                                else ""
                            ),
                            f"{row['random_r2']:.3f}" if row.get("random_r2") is not None else "",
                            (
                                f"{row['scaffold_r2']:.3f}"
                                if row.get("scaffold_r2") is not None
                                else ""
                            ),
                            (
                                f"{row['cluster_kmeans_r2']:.3f}"
                                if row.get("cluster_kmeans_r2") is not None
                                else ""
                            ),
                            f"{row['delta_r2']:.3f}" if row.get("delta_r2") is not None else "",
                            (
                                f"{row['train_time_s']:.3f}"
                                if row.get("train_time_s") is not None
                                else ""
                            ),
                            str(row.get("status") or ""),
                        ]
                    )
                    + " |"
                )
        lines.append("")
        lines.append("## Split-by-split comparative analysis")
        lines.append("")
        for item in candidate_results:
            lines.append(f"### {item['candidate_id']}")
            lines.append("")
            lines.append(f"- backend: `{item['backend_name']}`")
            lines.append(f"- representation: `{item['representation_name']}`")
            validation_assessment = item.get("validation_assessment") or {}
            lines.append(f"- hardest split: `{validation_assessment.get('hardest_split')}`")
            for split_result in item.get("split_results") or []:
                metrics = (split_result.get("metrics") or {}).get("test") or {}
                if metrics.get("balanced_accuracy") is not None:
                    lines.append(
                        f"- `{split_result.get('strategy_label')}`: "
                        f"balanced_accuracy={metrics.get('balanced_accuracy')}, "
                        f"roc_auc={metrics.get('roc_auc')}, f1_macro={metrics.get('f1_macro')}"
                    )
                else:
                    lines.append(
                        f"- `{split_result.get('strategy_label')}`: "
                        f"R²={metrics.get('r2')}, RMSE={metrics.get('rmse')}, MAE={metrics.get('mae')}, MSE={metrics.get('mse')}"
                    )
            lines.append("")
        lines.append("## Governance and recommendation")
        lines.append("")
        for key, value in recommendations.items():
            if value:
                lines.append(f"- {key.replace('_', ' ')}: `{value}`")
        lines.append("")
        lines.append("## Artifact and model-path appendix")
        lines.append("")
        for row in candidate_rows:
            lines.append(f"### {row['candidate_id']}")
            lines.append("")
            lines.append(f"- model_id: `{row['model_id']}`")
            lines.append(f"- internal_model_root: `{row['internal_model_root']}`")
            lines.append(f"- best_model_path: `{row['best_model_path']}`")
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def benchmark_qsar_models(
        self,
        train_csv: str,
        task_type: str,
        target_columns: List[str] | str,
        smiles_column: str = "smiles",
        benchmark_mode: str = "benchmark_standard_qsar",
        backends: Optional[List[str] | str] = None,
        include_candidate_variants: bool = True,
        tabicl_candidate_variants: Optional[List[str] | str] = None,
        output_dir: str = ".files/benchmark_output",
        allow_heavy_compute: bool = False,
        training_profile: Optional[str] = None,
        validation_strategy: Optional[Dict[str, Any]] = None,
        benchmark_requested: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run a multi-backend benchmark campaign for a QSAR-ready dataset."""
        if agent is None:
            raise ValueError("Agent is required when running a benchmark campaign")
        if not benchmark_requested:
            return {
                "benchmark_started": False,
                "blocked": True,
                "reason": (
                    "`benchmark_qsar_models` requires `benchmark_requested=True`. "
                    "Use single-backend training tools for ordinary standard_qsar, robust_qsar, "
                    "or Activity-Cliff-enriched training workflows."
                ),
                "next_step": (
                    "If the user explicitly asked for a benchmark or candidate leaderboard, "
                    "call this tool again with `benchmark_requested=True`; otherwise continue "
                    "with model registration/persistence for the completed single training run."
                ),
            }
        if not str(benchmark_mode or "").startswith("benchmark_"):
            return {
                "benchmark_started": False,
                "blocked": True,
                "reason": (
                    "`benchmark_qsar_models` only accepts explicit benchmark modes such as "
                    "`benchmark_standard_qsar`. A plain validation protocol like "
                    f"`{benchmark_mode}` indicates ordinary single-model training, not a benchmark."
                ),
                "next_step": (
                    "Continue with model registration/persistence for the completed single training "
                    "run, or call this tool with a `benchmark_*` mode only if the user explicitly "
                    "asked for a benchmark."
                ),
            }

        target_columns = _coerce_list(target_columns) or []
        requested_backends = _coerce_list(backends)
        requested_tabicl_variants = _coerce_list(tabicl_candidate_variants)

        benchmark_protocol = self._resolve_benchmark_protocol(benchmark_mode)
        compute_payload = self._resolve_compute_profile()
        effective_training_profile = training_profile or compute_payload["training_profile"]
        protocol_policy = resolve_validation_strategy(
            requested_protocol=benchmark_protocol,
            validation_strategy=validation_strategy,
            training_profile=effective_training_profile,
            seed_policy_mode="generated_per_benchmark_campaign",
        )
        benchmark_protocol = protocol_policy["protocol"]
        effective_include_candidate_variants = (
            include_candidate_variants if benchmark_protocol != "fast_local" else False
        )
        campaign_seed_policy = protocol_policy["seed_policy"]

        candidates = self._expand_candidates(
            task_type=task_type,
            target_columns=target_columns,
            requested_backends=requested_backends,
            include_candidate_variants=effective_include_candidate_variants,
            requested_tabicl_variants=requested_tabicl_variants,
            training_profile=effective_training_profile,
            benchmark_protocol=benchmark_protocol,
        )
        if not candidates:
            raise ValueError("No compatible benchmark candidates are available.")

        campaign_root = Path(output_dir).expanduser().resolve()
        campaign_root.mkdir(parents=True, exist_ok=True)

        candidate_rows: List[Dict[str, Any]] = []
        candidate_results: List[Dict[str, Any]] = []
        compact_candidate_results: List[Dict[str, Any]] = []
        persisted_model_mapping: List[Dict[str, Any]] = []

        total_candidates = len(candidates)
        for candidate_index, candidate in enumerate(candidates, start=1):
            candidate_dir = campaign_root / candidate["candidate_id"]
            candidate_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Benchmark progress: candidate %d/%d - %s (backend=%s, representation=%s) starting.",
                candidate_index,
                total_candidates,
                candidate["candidate_id"],
                candidate["backend_name"],
                candidate["representation_name"],
            )
            try:
                result = self._train_candidate(
                    candidate=candidate,
                    train_csv=train_csv,
                    task_type=task_type,
                    smiles_column=smiles_column,
                    target_columns=target_columns,
                    benchmark_protocol=benchmark_protocol,
                    candidate_dir=candidate_dir,
                    allow_heavy_compute=allow_heavy_compute,
                    training_profile=training_profile,
                    campaign_seed_policy=campaign_seed_policy,
                    validation_strategy=validation_strategy,
                    agent=agent,
                )
            except Exception:
                logger.exception(
                    "Benchmark progress: candidate %d/%d - %s failed.",
                    candidate_index,
                    total_candidates,
                    candidate["candidate_id"],
                )
                raise
            persisted = self._register_and_persist_candidate(
                candidate=candidate,
                result=result,
                train_csv=train_csv,
                benchmark_mode=benchmark_mode,
                benchmark_protocol=benchmark_protocol,
                campaign_root=campaign_root,
                task_type=task_type,
                smiles_column=smiles_column,
                target_columns=target_columns,
                training_profile=effective_training_profile,
                campaign_seed_policy=campaign_seed_policy,
                agent=agent,
            )
            result["candidate_id"] = candidate["candidate_id"]
            result["backend_name"] = candidate["backend_name"]
            result["representation_name"] = candidate["representation_name"]
            candidate_results.append(result)
            candidate_row = self._candidate_summary_row(
                candidate=candidate,
                result=result,
                persisted=persisted,
                benchmark_protocol=benchmark_protocol,
            )
            candidate_rows.append(candidate_row)
            compact_candidate_results.append(
                self._compact_candidate_result(
                    result=result,
                    row=candidate_row,
                )
            )
            logger.info(
                "Benchmark progress: candidate %d/%d - %s completed (status=%s, hardest_split_r2=%s).",
                candidate_index,
                total_candidates,
                candidate["candidate_id"],
                persisted.get("status"),
                candidate_row.get("hardest_split_r2"),
            )
            persisted_model_mapping.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "model_id": persisted.get("model_id"),
                    "backend_name": candidate["backend_name"],
                    "representation_name": candidate["representation_name"],
                    "internal_model_root": persisted.get("model_root"),
                    "catalog_status": persisted.get("status"),
                    "best_model_path": persisted.get("model_path"),
                }
            )

        ranked_rows = self._rank_summary_rows(candidate_rows)
        recommendations = self._resolve_recommendations(ranked_rows, benchmark_protocol)
        feature_cache = {
            "feature_cache_dir": str(campaign_root / "feature_cache"),
            "cache_hits": sum(int(row.get("feature_cache_hits") or 0) for row in candidate_rows),
            "cache_misses": sum(
                int(row.get("feature_cache_misses") or 0) for row in candidate_rows
            ),
        }

        leaderboard_path = campaign_root / "leaderboard.csv"
        pd.DataFrame(ranked_rows).to_csv(leaderboard_path, index=False)

        report_path = campaign_root / "benchmark_report.md"
        report_path.write_text(
            self._render_benchmark_report(
                benchmark_mode=benchmark_mode,
                benchmark_protocol=benchmark_protocol,
                compute_environment=compute_payload["compute_environment"],
                candidate_rows=ranked_rows,
                candidate_results=candidate_results,
                recommendations=recommendations,
                campaign_seed_policy=campaign_seed_policy,
            )
        )

        benchmark_summary = {
            "benchmark_mode": benchmark_mode,
            "benchmark_protocol": benchmark_protocol,
            "train_csv": train_csv,
            "task_type": task_type,
            "target_columns": target_columns,
            "smiles_column": smiles_column,
            "compute_environment": compute_payload["compute_environment"],
            "training_profile": effective_training_profile,
            "campaign_seed_policy": campaign_seed_policy,
            "validation_strategy": protocol_policy.get("validation_strategy"),
            "validation_strategy_type": protocol_policy.get("validation_strategy_type"),
            "validation_aggregation": protocol_policy.get("aggregation"),
            "selection_metric": protocol_policy.get("selection_metric"),
            "seed_policy_report": seed_policy_reporting_text(campaign_seed_policy),
            "reproducibility": seed_policy_reproducibility_metadata(campaign_seed_policy),
            "candidate_inventory": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "backend_name": candidate["backend_name"],
                    "representation_name": candidate["representation_name"],
                    "validation_protocol": benchmark_protocol,
                    "training_profile": effective_training_profile,
                }
                for candidate in candidates
            ],
            "candidate_results": compact_candidate_results,
            "persisted_model_mapping": persisted_model_mapping,
            "feature_cache": feature_cache,
            "leaderboard_path": str(leaderboard_path),
            "report_path": str(report_path),
            **recommendations,
        }
        summary_path = campaign_root / "benchmark_summary.json"
        summary_path.write_text(json.dumps(benchmark_summary, indent=2) + "\n")

        return {
            "benchmark_mode": benchmark_mode,
            "benchmark_protocol": benchmark_protocol,
            "output_dir": str(campaign_root),
            "campaign_seed_policy": campaign_seed_policy,
            "validation_strategy": protocol_policy.get("validation_strategy"),
            "validation_strategy_type": protocol_policy.get("validation_strategy_type"),
            "validation_aggregation": protocol_policy.get("aggregation"),
            "selection_metric": protocol_policy.get("selection_metric"),
            "seed_policy_report": seed_policy_reporting_text(campaign_seed_policy),
            "reproducibility": seed_policy_reproducibility_metadata(campaign_seed_policy),
            "candidate_results": compact_candidate_results,
            "leaderboard": ranked_rows,
            "persisted_model_mapping": persisted_model_mapping,
            "feature_cache": feature_cache,
            "leaderboard_path": str(leaderboard_path),
            "summary_path": str(summary_path),
            "report_path": str(report_path),
            **recommendations,
        }
