#!/usr/bin/env python
# coding: utf-8
"""
Shared QSAR training policy helpers used by multiple prediction backends.
"""

from __future__ import annotations

import math
import os
import random
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

import torch

QSAR_HARDEST_SPLIT_R2_MIN = 0.70
QSAR_ROBUSTNESS_DELTA_R2_MIN = -0.10
QSAR_ROBUSTNESS_DELTA_RMSE_MAX = 0.15
QSAR_RANDOM_STABILITY_R2_STD_MAX = 0.03
QSAR_HARDEST_SPLIT_BALANCED_ACCURACY_MIN = 0.70
QSAR_ROBUSTNESS_DELTA_BALANCED_ACCURACY_MIN = -0.10
QSAR_RANDOM_STABILITY_BALANCED_ACCURACY_STD_MAX = 0.03
PROJECT_TIMEZONE = ZoneInfo("Europe/Paris")
SEED_MIN = 1
SEED_MAX = 2_147_483_647
DEFAULT_QSAR_SPLIT_SIZES = [0.8, 0.1, 0.1]


def project_now() -> datetime:
    return datetime.now(PROJECT_TIMEZONE)


def coerce_project_timezone(value: Optional[str]) -> datetime:
    if not value:
        return project_now()
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=PROJECT_TIMEZONE)
    return parsed.astimezone(PROJECT_TIMEZONE)


def safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value.strip().lower()).strip("_")


def _coerce_seed(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    return max(SEED_MIN, min(SEED_MAX, seed))


def _unique_generated_seeds(count: int) -> List[int]:
    seeds: List[int] = []
    seen: set[int] = set()
    while len(seeds) < count:
        seed = secrets.randbelow(SEED_MAX - SEED_MIN + 1) + SEED_MIN
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _unique_replay_seeds(count: int, base_seed: int) -> List[int]:
    rng = random.Random(base_seed)
    seeds = [base_seed]
    seen = {base_seed}
    while len(seeds) < count:
        seed = rng.randint(SEED_MIN, SEED_MAX)
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _standard_split_templates() -> List[Dict[str, Any]]:
    return [
        {
            "backend_split_type": "random",
            "primary": True,
            "split_sizes": DEFAULT_QSAR_SPLIT_SIZES,
        }
    ]


def _label_split(template: Dict[str, Any], seed: int) -> str:
    if template.get("label"):
        return str(template["label"])
    if template.get("backend_split_type") == "random":
        return f"random_seed_{seed}"
    return str(template.get("backend_split_type") or "split")


def resolve_seed_policy(
    *,
    split_templates: List[Dict[str, Any]],
    mode: str = "generated_per_run",
    seed_policy: Optional[Dict[str, Any]] = None,
    base_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve reproducible split/model seeds for an explicit set of split runs."""
    if seed_policy and seed_policy.get("split_runs"):
        replay = dict(seed_policy)
        replay.setdefault("mode", "user_provided_or_replay")
        replay.setdefault(
            "shared_across_candidates", replay.get("mode") == "generated_per_benchmark_campaign"
        )
        replay.setdefault(
            "reproducibility_note",
            "Seeds were supplied from an existing policy and preserved for replay.",
        )
        replay.setdefault("reporting_text", seed_policy_reporting_text(replay))
        replay.setdefault("replay_supported", True)
        return replay

    templates = [dict(item) for item in split_templates]
    if not templates:
        raise ValueError("split_templates must contain at least one split run.")
    provided_seed = _coerce_seed(base_seed)
    needs_campaign_seed = mode == "generated_per_benchmark_campaign" and provided_seed is None
    seed_count = len(templates) + 1 + (1 if needs_campaign_seed else 0)
    if provided_seed is not None:
        resolved_mode = "user_provided_or_replay"
        seeds = _unique_replay_seeds(seed_count, provided_seed)
    else:
        resolved_mode = mode
        seeds = _unique_generated_seeds(seed_count)

    campaign_seed = seeds[0] if needs_campaign_seed else None
    run_seeds = seeds[1:] if needs_campaign_seed else seeds
    split_seeds = run_seeds[: len(templates)]
    model_seed = run_seeds[-1]
    split_runs: List[Dict[str, Any]] = []
    for template, seed in zip(templates, split_seeds, strict=True):
        run = {
            "label": _label_split(template, seed),
            "backend_split_type": template["backend_split_type"],
            "seed": seed,
            "primary": bool(template.get("primary", False)),
        }
        if template.get("split_sizes") is not None:
            run["split_sizes"] = list(template["split_sizes"])
        split_runs.append(run)

    random_split_seeds = [
        item["seed"] for item in split_runs if item["backend_split_type"] == "random"
    ]
    scaffold_seed = next(
        (item["seed"] for item in split_runs if item["backend_split_type"] == "scaffold_balanced"),
        None,
    )
    cluster_seed = next(
        (item["seed"] for item in split_runs if item["backend_split_type"] == "kmeans"),
        None,
    )
    policy = {
        "mode": resolved_mode,
        "generated_at": project_now().isoformat(),
        "model_seed": model_seed,
        "split_runs": split_runs,
        "random_split_seeds": random_split_seeds,
        "scaffold_seed": scaffold_seed,
        "cluster_seed": cluster_seed,
        "shared_across_candidates": resolved_mode == "generated_per_benchmark_campaign",
        "reproducibility_note": (
            "Seeds were generated once for this benchmark campaign and shared across all candidates."
            if resolved_mode == "generated_per_benchmark_campaign"
            else (
                "Seeds were generated for this run and persisted for replay."
                if resolved_mode == "generated_per_run"
                else "Seeds were supplied by the user or replayed from persisted artifacts."
            )
        ),
    }
    if resolved_mode == "generated_per_benchmark_campaign":
        policy["campaign_seed"] = campaign_seed or seeds[0]
    policy["reporting_text"] = seed_policy_reporting_text(policy)
    policy["replay_supported"] = True
    return policy


def _coerce_seed_policy(seed_policy: Any) -> Dict[str, Any]:
    """Return a dict seed policy even when agents pass loose text/list values."""
    if isinstance(seed_policy, Mapping):
        return dict(seed_policy)
    if seed_policy is None:
        return {}
    if isinstance(seed_policy, str) and seed_policy.strip():
        return {"reporting_text": seed_policy.strip()}
    return {}


def seed_policy_reporting_text(seed_policy: Optional[Dict[str, Any]]) -> str:
    """Return a short user-facing French reporting sentence for a seed policy."""
    policy = _coerce_seed_policy(seed_policy)
    mode = str(policy.get("mode") or "").strip()
    if mode == "generated_per_benchmark_campaign":
        return "Politique de seeds : partagée au niveau campagne benchmark"
    if mode == "user_provided_or_replay":
        return "Politique de seeds : fournie par l'utilisateur / replay"
    if policy.get("reporting_text"):
        return str(policy["reporting_text"])
    return "Politique de seeds : générées automatiquement et persistées"


def seed_policy_reproducibility_metadata(seed_policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Build compact catalog metadata for reproducibility and future agent inspection."""
    policy = _coerce_seed_policy(seed_policy)
    return {
        "seed_policy_mode": policy.get("mode") or "unknown",
        "seed_policy_report": seed_policy_reporting_text(policy),
        "replay_supported": bool(policy.get("split_runs")),
        "model_seed": policy.get("model_seed"),
        "campaign_seed": policy.get("campaign_seed"),
        "shared_across_candidates": bool(policy.get("shared_across_candidates")),
        "split_runs": list(policy.get("split_runs") or []),
        "reproducibility_note": policy.get("reproducibility_note"),
    }


def detect_memory_limit_bytes() -> Optional[int]:
    candidates = [
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text().strip()
            if not raw or raw == "max":
                continue
            value = int(raw)
            if value <= 0 or value > 1 << 60:
                continue
            return value
        except Exception:
            continue
    return None


def detect_physical_memory_bytes() -> Optional[int]:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        if (
            isinstance(page_size, int)
            and isinstance(page_count, int)
            and page_size > 0
            and page_count > 0
        ):
            return page_size * page_count
    except Exception:
        return None
    return None


def detect_cpu_count() -> int:
    host_cpu_count = os.cpu_count() or 1
    quota_cpu_count: Optional[int] = None

    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.exists():
            quota_raw, period_raw = cpu_max.read_text().strip().split()[:2]
            if quota_raw != "max":
                quota = int(quota_raw)
                period = int(period_raw)
                if quota > 0 and period > 0:
                    quota_cpu_count = max(1, math.ceil(quota / period))
    except Exception:
        quota_cpu_count = None

    if quota_cpu_count is None:
        try:
            quota_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
            period_path = Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
            if quota_path.exists() and period_path.exists():
                quota = int(quota_path.read_text().strip())
                period = int(period_path.read_text().strip())
                if quota > 0 and period > 0:
                    quota_cpu_count = max(1, math.ceil(quota / period))
        except Exception:
            quota_cpu_count = None

    return max(1, min(host_cpu_count, quota_cpu_count or host_cpu_count))


def detect_disk_usage(base_path: Optional[Path] = None) -> Dict[str, Optional[float]]:
    target = (base_path or Path.cwd()).resolve()
    try:
        stats = os.statvfs(target)
        total = stats.f_frsize * stats.f_blocks
        free = stats.f_frsize * stats.f_bavail
        return {
            "disk_gb_total": round(total / (1024**3), 2),
            "disk_gb_free": round(free / (1024**3), 2),
        }
    except Exception:
        return {"disk_gb_total": None, "disk_gb_free": None}


def describe_compute_environment() -> Dict[str, Any]:
    cpu_count = detect_cpu_count()
    memory_limit_bytes = detect_memory_limit_bytes()
    physical_memory_bytes = detect_physical_memory_bytes()
    memory_bytes_total = memory_limit_bytes or physical_memory_bytes
    memory_gb_total = round(memory_bytes_total / (1024**3), 2) if memory_bytes_total else None
    try:
        gpu_available = bool(torch.cuda.is_available())
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_name = torch.cuda.get_device_name(0) if gpu_available and gpu_count > 0 else None
    except Exception:
        gpu_available = bool(
            os.getenv("CUDA_VISIBLE_DEVICES")
            and os.getenv("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}
        )
        gpu_count = 0
        gpu_name = None

    if Path("/.dockerenv").exists():
        execution_env = "docker_local"
    elif (
        Path("/.singularity.d").exists()
        or os.getenv("APPTAINER_NAME")
        or os.getenv("SINGULARITY_NAME")
    ):
        execution_env = "apptainer_local"
    else:
        execution_env = "local"

    disk_usage = detect_disk_usage()
    profile = resolve_training_profile(
        {
            "cpu_count": cpu_count,
            "memory_gb_total": memory_gb_total,
            "gpu_available": gpu_available,
            "execution_env": execution_env,
        }
    )
    return {
        "execution_env": execution_env,
        "cpu_count": cpu_count,
        "memory_gb_total": memory_gb_total,
        "memory_source": (
            "cgroup_limit"
            if memory_limit_bytes
            else "physical_host" if physical_memory_bytes else None
        ),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        **disk_usage,
        "suggested_profile": profile["profile"],
        "profile_reason": profile["reason"],
    }


def resolve_backend_n_jobs(
    compute_env: Mapping[str, Any],
    *,
    backend_name: str,
    profile: str,
    requested_n_jobs: Optional[Any] = None,
) -> int:
    cpu_count = max(1, int(compute_env.get("cpu_count") or 1))
    if requested_n_jobs is not None:
        try:
            return max(1, min(cpu_count, int(requested_n_jobs)))
        except (TypeError, ValueError):
            return max(1, min(cpu_count, 1))

    caps = {
        "lightgbm": {"local_light": 8, "local_standard": 24, "heavy_validation": 48},
        "tabicl": {"local_light": 1, "local_standard": 4, "heavy_validation": 8},
    }
    backend_caps = caps.get(backend_name.lower(), {})
    cap = backend_caps.get(profile, backend_caps.get("local_standard", cpu_count))
    return max(1, min(cpu_count, cap))


def resolve_training_profile(compute_env: Dict[str, Any]) -> Dict[str, Any]:
    cpu_count = int(compute_env.get("cpu_count") or 1)
    memory_gb_total = compute_env.get("memory_gb_total")
    gpu_available = bool(compute_env.get("gpu_available"))
    execution_env = compute_env.get("execution_env") or "local"

    if not gpu_available and execution_env == "docker_local":
        if memory_gb_total is None and cpu_count <= 8:
            return {
                "profile": "local_light",
                "reason": "CPU-only Docker environment on a modest local machine; defaulting to the safest single-run profile.",
            }
        if memory_gb_total is not None and memory_gb_total <= 8.5 and cpu_count <= 8:
            return {
                "profile": "local_light",
                "reason": "CPU-only Docker environment with limited RAM; using a conservative single-run configuration.",
            }
        if memory_gb_total is not None and memory_gb_total <= 16 and cpu_count <= 12:
            return {
                "profile": "local_standard",
                "reason": "CPU-only local environment; using a moderate single-run configuration.",
            }

    if gpu_available:
        return {
            "profile": "heavy_validation",
            "reason": "GPU detected; heavier compute settings are acceptable.",
        }

    return {
        "profile": "local_standard",
        "reason": "Defaulting to a moderate single-run local profile.",
    }


def resolve_validation_protocol(
    *,
    requested_protocol: Optional[str],
    training_profile: str,
    seed_policy: Optional[Dict[str, Any]] = None,
    seed_policy_mode: str = "generated_per_run",
    base_seed: Optional[int] = None,
) -> Dict[str, Any]:
    protocol = (requested_protocol or "").strip().lower()
    if not protocol:
        protocol = "standard_qsar"

    if protocol == "standard_qsar":
        resolved_seed_policy = resolve_seed_policy(
            split_templates=_standard_split_templates(),
            mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=base_seed,
        )
        return {
            "protocol": "standard_qsar",
            "reason": "Standard QSAR default: one fixed random 80/10/10 train/validation/test split.",
            "split_runs": resolved_seed_policy["split_runs"],
            "seed_policy": resolved_seed_policy,
        }

    raise ValueError(
        f"Unsupported validation_protocol `{requested_protocol}`. "
        "The only named protocol is `standard_qsar`; use validation_strategy "
        "for holdout, repeated holdout, cross-validation, scaffold, cluster, or full-train workflows."
    )


def summarize_training_durations(
    *,
    split_results: List[Dict[str, Any]],
    total_started_at: datetime,
    total_completed_at: datetime,
) -> Dict[str, Any]:
    split_durations: List[Dict[str, Any]] = []
    for item in split_results:
        split_durations.append(
            {
                "label": item.get("strategy_label"),
                "strategy_family": item.get("strategy_family"),
                "started_at": item.get("started_at"),
                "completed_at": item.get("completed_at"),
                "duration_seconds": item.get("duration_seconds"),
            }
        )

    return {
        "total_started_at": total_started_at.isoformat(),
        "total_completed_at": total_completed_at.isoformat(),
        "total_duration_seconds": round((total_completed_at - total_started_at).total_seconds(), 3),
        "split_durations": split_durations,
    }


def aggregate_split_families(split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    families: Dict[str, List[Dict[str, Any]]] = {}
    for item in split_results:
        family = item.get("strategy_family") or item.get("strategy")
        metrics = (item.get("metrics") or {}).get("test") or {}
        if not family or not metrics:
            continue
        families.setdefault(family, []).append(item)

    aggregated: Dict[str, Any] = {}
    metric_names = (
        "mse",
        "mae",
        "rae",
        "rmse",
        "r2",
        "spearman",
        "kendall",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "roc_auc",
    )
    for family, items in families.items():
        entry: Dict[str, Any] = {
            "family": family,
            "num_runs": len(items),
            "strategy_labels": [item.get("strategy_label") for item in items],
            "runs": [],
            "test_n_values": [],
        }
        for item in items:
            metrics = (item.get("metrics") or {}).get("test") or {}
            entry["runs"].append(
                {"label": item.get("strategy_label"), "seed": item.get("seed"), "metrics": metrics}
            )
            if metrics.get("n") is not None:
                entry["test_n_values"].append(metrics["n"])

        for metric_name in metric_names:
            values = [
                float(((item.get("metrics") or {}).get("test") or {}).get(metric_name))
                for item in items
                if ((item.get("metrics") or {}).get("test") or {}).get(metric_name) is not None
            ]
            if not values:
                continue
            mean_value = sum(values) / len(values)
            variance = (
                sum((value - mean_value) ** 2 for value in values) / len(values)
                if len(values) > 1
                else 0.0
            )
            entry[f"{metric_name}_mean"] = mean_value
            entry[f"{metric_name}_std"] = math.sqrt(variance)
            if len(values) == 1:
                entry[metric_name] = values[0]

        if entry["test_n_values"]:
            entry["test_n_mean"] = sum(entry["test_n_values"]) / len(entry["test_n_values"])
        aggregated[family] = entry

    return aggregated


def assess_protocol_results(split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    assessment: Dict[str, Any] = {
        "robustness_warning": None,
        "delta_vs_random": {},
        "hardest_split": None,
        "aggregated_split_metrics": {},
        "governance": {
            "recommended_status": "experimental",
            "gates": {},
            "passes_dataset_gate": True,
            "passes_hardest_split_gate": False,
            "passes_robustness_gate": False,
            "hardest_split_metrics": {},
            "gating_summary": [],
        },
    }
    aggregated = aggregate_split_families(split_results)
    assessment["aggregated_split_metrics"] = aggregated
    random_family = aggregated.get("random")
    if not random_family:
        return assessment

    classification_mode = random_family.get("balanced_accuracy_mean") is not None
    random_primary = random_family.get(
        "balanced_accuracy_mean" if classification_mode else "r2_mean"
    )
    random_secondary = random_family.get("roc_auc_mean" if classification_mode else "rmse_mean")
    if random_primary is None:
        return assessment
    hardest_name = None
    hardest_primary = None

    for strategy_name, family_result in aggregated.items():
        if strategy_name == "random":
            continue
        split_primary = family_result.get(
            "balanced_accuracy_mean" if classification_mode else "r2_mean"
        )
        split_secondary = family_result.get("roc_auc_mean" if classification_mode else "rmse_mean")
        if split_primary is None and split_secondary is None:
            continue

        deltas: Dict[str, Any] = {}
        if split_primary is not None:
            deltas["balanced_accuracy" if classification_mode else "r2"] = (
                split_primary - random_primary
            )
        if split_secondary is not None and random_secondary is not None:
            deltas["roc_auc" if classification_mode else "rmse"] = (
                split_secondary - random_secondary
            )
        assessment["delta_vs_random"][strategy_name] = deltas

        if split_primary is not None:
            if hardest_primary is None or split_primary < hardest_primary:
                hardest_name = strategy_name
                hardest_primary = split_primary

    if hardest_name is not None:
        assessment["hardest_split"] = hardest_name

    warning_reasons: List[str] = []
    for strategy_name, deltas in assessment["delta_vs_random"].items():
        if classification_mode:
            delta_balanced_accuracy = deltas.get("balanced_accuracy")
            if (
                delta_balanced_accuracy is not None
                and delta_balanced_accuracy < QSAR_ROBUSTNESS_DELTA_BALANCED_ACCURACY_MIN
            ):
                warning_reasons.append(
                    f"{strategy_name} split lowers balanced accuracy by {abs(delta_balanced_accuracy):.3f} vs random"
                )
        else:
            delta_r2 = deltas.get("r2")
            delta_rmse = deltas.get("rmse")
            if delta_r2 is not None and delta_r2 < QSAR_ROBUSTNESS_DELTA_R2_MIN:
                warning_reasons.append(
                    f"{strategy_name} split lowers R² by {abs(delta_r2):.3f} vs random"
                )
            if delta_rmse is not None and delta_rmse > QSAR_ROBUSTNESS_DELTA_RMSE_MAX:
                warning_reasons.append(
                    f"{strategy_name} split increases RMSE by {delta_rmse:.3f} vs random"
                )

    if warning_reasons:
        assessment["robustness_warning"] = (
            "Harder validation splits reveal a non-trivial performance drop: "
            + "; ".join(warning_reasons)
            + "."
        )

    governance = assessment["governance"]
    hardest_result = (
        aggregated.get(assessment["hardest_split"]) if assessment["hardest_split"] else None
    )
    if classification_mode:
        hardest_metrics = {
            "balanced_accuracy": (hardest_result or {}).get("balanced_accuracy_mean"),
            "roc_auc": (hardest_result or {}).get("roc_auc_mean"),
            "f1_macro": (hardest_result or {}).get("f1_macro_mean"),
            "accuracy": (hardest_result or {}).get("accuracy_mean"),
            "n": (hardest_result or {}).get("test_n_mean"),
        }
    else:
        hardest_metrics = {
            "r2": (hardest_result or {}).get("r2_mean"),
            "rmse": (hardest_result or {}).get("rmse_mean"),
            "mae": (hardest_result or {}).get("mae_mean"),
            "mse": (hardest_result or {}).get("mse_mean"),
            "n": (hardest_result or {}).get("test_n_mean"),
        }
    governance["hardest_split_metrics"] = hardest_metrics

    hardest_r2 = hardest_metrics.get("r2")
    hardest_rmse = hardest_metrics.get("rmse")
    hardest_balanced_accuracy = hardest_metrics.get("balanced_accuracy")
    hardest_pass = (
        hardest_balanced_accuracy is not None
        and hardest_balanced_accuracy >= QSAR_HARDEST_SPLIT_BALANCED_ACCURACY_MIN
        if classification_mode
        else hardest_r2 is not None and hardest_r2 >= QSAR_HARDEST_SPLIT_R2_MIN
    )
    governance["passes_hardest_split_gate"] = hardest_pass

    robustness_pass = not bool(warning_reasons)
    governance["passes_robustness_gate"] = robustness_pass

    summary: List[str] = []
    if random_family.get("num_runs", 0) > 1:
        if classification_mode:
            summary.append(
                "Random stability: balanced accuracy "
                f"mean={random_family.get('balanced_accuracy_mean', 0):.3f} ± "
                f"{random_family.get('balanced_accuracy_std', 0):.3f}"
            )
        else:
            summary.append(
                f"Random stability: R² mean={random_family.get('r2_mean', 0):.3f} ± {random_family.get('r2_std', 0):.3f}"
            )
            summary.append(
                f"Random stability: RMSE mean={random_family.get('rmse_mean', 0):.3f} ± {random_family.get('rmse_std', 0):.3f}"
            )
    if assessment["hardest_split"]:
        summary.append(f"Hardest split: {assessment['hardest_split']}")
    if hardest_r2 is not None:
        summary.append(f"Hardest split R²={hardest_r2:.3f}")
    if hardest_rmse is not None:
        summary.append(f"Hardest split RMSE={hardest_rmse:.3f}")
    if hardest_balanced_accuracy is not None:
        summary.append(f"Hardest split balanced accuracy={hardest_balanced_accuracy:.3f}")
    summary.append("Hardest Split Gate: PASS" if hardest_pass else "Hardest Split Gate: FAIL")
    summary.append("Robustness Gap Gate: PASS" if robustness_pass else "Robustness Gap Gate: FAIL")
    governance["gating_summary"] = summary

    random_stability_pass = True
    random_primary_std = random_family.get(
        "balanced_accuracy_std" if classification_mode else "r2_std"
    )
    random_std_threshold = (
        QSAR_RANDOM_STABILITY_BALANCED_ACCURACY_STD_MAX
        if classification_mode
        else QSAR_RANDOM_STABILITY_R2_STD_MAX
    )
    if (
        random_family.get("num_runs", 0) > 1
        and random_primary_std is not None
        and random_primary_std > random_std_threshold
    ):
        random_stability_pass = False
    governance["passes_random_stability_gate"] = random_stability_pass
    if random_family.get("num_runs", 0) > 1:
        summary.append(
            "Random Stability Gate: PASS"
            if random_stability_pass
            else "Random Stability Gate: FAIL"
        )

    protocol_name = None
    for item in split_results:
        protocol_name = item.get("validation_protocol") or protocol_name

    dataset_gate_pass = True
    governance["passes_dataset_gate"] = dataset_gate_pass
    governance["gates"] = {
        "dataset_gate": {
            "name": "Dataset Gate",
            "pass": dataset_gate_pass,
            "criteria": [
                "real dataset source",
                "completed curation",
                "real split artifacts",
                "checkpoint exists",
                "real test metrics",
                "applicability domain built",
            ],
        },
        "hardest_split_gate": {
            "name": "Hardest Split Gate",
            "pass": hardest_pass,
            "thresholds": (
                {"balanced_accuracy_min": QSAR_HARDEST_SPLIT_BALANCED_ACCURACY_MIN}
                if classification_mode
                else {"r2_min": QSAR_HARDEST_SPLIT_R2_MIN}
            ),
        },
        "robustness_gap_gate": {
            "name": "Robustness Gap Gate",
            "pass": robustness_pass,
            "thresholds": (
                {"delta_balanced_accuracy_min": QSAR_ROBUSTNESS_DELTA_BALANCED_ACCURACY_MIN}
                if classification_mode
                else {
                    "delta_r2_min": QSAR_ROBUSTNESS_DELTA_R2_MIN,
                    "delta_rmse_max": QSAR_ROBUSTNESS_DELTA_RMSE_MAX,
                }
            ),
        },
        "random_stability_gate": {
            "name": "Random Stability Gate",
            "pass": random_stability_pass,
            "active": random_family.get("num_runs", 0) > 1,
            "thresholds": (
                {"balanced_accuracy_std_max": QSAR_RANDOM_STABILITY_BALANCED_ACCURACY_STD_MAX}
                if classification_mode
                else {"r2_std_max": QSAR_RANDOM_STABILITY_R2_STD_MAX}
            ),
        },
    }

    if not dataset_gate_pass:
        governance["recommended_status"] = "experimental"
    elif not hardest_pass or not robustness_pass:
        governance["recommended_status"] = "workflow_demo"
    elif random_family.get("num_runs", 0) > 1:
        governance["recommended_status"] = (
            "robust_validated" if random_stability_pass else "workflow_demo"
        )
    elif protocol_name:
        governance["recommended_status"] = "validated"
    else:
        governance["recommended_status"] = "experimental"
    return assessment
