#!/usr/bin/env python
# coding: utf-8
"""Configurable QSAR validation strategy resolver."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from .qsar_training_policy import resolve_seed_policy, resolve_validation_protocol


DEFAULT_SPLIT_SIZES = [0.8, 0.1, 0.1]
DEFAULT_SELECTION_METRIC = "rmse"

SPLIT_FAMILY_TO_BACKEND_TYPE = {
    "random": "random",
    "scaffold": "scaffold_balanced",
    "cluster": "kmeans",
    "cluster_kmeans": "kmeans",
    "kmeans": "kmeans",
}


@dataclass(frozen=True)
class SplitRun:
    label: str
    backend_split_type: str
    seed: int
    primary: bool = False
    split_family: str = "random"
    split_sizes: Optional[List[float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _coerce_strategy(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return {}
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError("validation_strategy string must be valid JSON.") from exc
        return _coerce_strategy(parsed)
    if not isinstance(raw, Mapping):
        raise ValueError("validation_strategy must be a dictionary/object.")
    strategy = dict(raw)
    if "type" not in strategy and strategy.get("method") is not None:
        strategy["type"] = strategy["method"]
    if "split_family" not in strategy and strategy.get("split_type") is not None:
        strategy["split_family"] = strategy["split_type"]
    return strategy


def _coerce_split_family(raw: Any) -> str:
    family = str(raw or "random").strip().lower()
    if family not in SPLIT_FAMILY_TO_BACKEND_TYPE:
        raise ValueError(
            "Unsupported split_family. Expected one of "
            f"{sorted(SPLIT_FAMILY_TO_BACKEND_TYPE)}."
        )
    if family in {"cluster_kmeans", "kmeans"}:
        return "cluster"
    return family


def _infer_split_family(strategy: Mapping[str, Any]) -> str:
    if strategy.get("split_family") is not None:
        return _coerce_split_family(strategy.get("split_family"))
    for family in ("random", "scaffold", "cluster", "kmeans", "cluster_kmeans"):
        if any(key in strategy for key in (f"{family}_split", f"{family}_holdout")):
            return _coerce_split_family(family)
    return "random"


def _coerce_split_sizes(raw: Any) -> List[float]:
    if raw is None:
        return list(DEFAULT_SPLIT_SIZES)
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError("validation_strategy.split_sizes must be [train, test] or [train, validation, test].")
    values = [float(item) for item in raw]
    total = sum(values)
    if values[0] <= 0 or values[-1] <= 0 or any(item < 0 for item in values) or abs(total - 1.0) > 1e-6:
        raise ValueError("validation_strategy.split_sizes must sum to 1.0 with positive train/test ratios.")
    if len(values) == 3 and values[1] == 0:
        return [values[0], values[2]]
    return values


def _split_sizes_from_ratios(config: Mapping[str, Any]) -> Optional[List[float]]:
    val_ratio = config.get("validation_ratio", config.get("val_ratio", config.get("val_fraction")))
    test_ratio = config.get("test_ratio", config.get("test_fraction"))
    train_ratio = config.get("train_ratio", config.get("train_fraction"))
    if val_ratio is None and test_ratio is None and train_ratio is None:
        return None
    if val_ratio is None:
        if test_ratio is None:
            raise ValueError("validation_strategy requires test_ratio when validation_ratio is omitted.")
        if train_ratio is None and test_ratio is not None:
            train_ratio = round(1.0 - float(test_ratio), 12)
        return _coerce_split_sizes([train_ratio, test_ratio])
    if test_ratio is None:
        raise ValueError("validation_strategy requires test_ratio when validation_ratio is provided.")
    if train_ratio is None and test_ratio is not None:
        train_ratio = round(1.0 - float(val_ratio) - float(test_ratio), 12)
    return _coerce_split_sizes([train_ratio, val_ratio, test_ratio])


def _coerce_strategy_split_sizes(strategy: Mapping[str, Any], family: str) -> List[float]:
    aliases = [f"{family}_split", f"{family}_holdout"]
    if family == "cluster":
        aliases.extend(["kmeans_split", "kmeans_holdout", "cluster_kmeans_split", "cluster_kmeans_holdout"])
    for alias in aliases:
        split_config = strategy.get(alias)
        if isinstance(split_config, Mapping):
            split_sizes = _split_sizes_from_ratios(split_config)
            if split_sizes is not None:
                return split_sizes
    split_sizes = _split_sizes_from_ratios(strategy)
    if split_sizes is not None:
        return split_sizes
    if strategy.get("split_sizes") is not None:
        return _coerce_split_sizes(strategy.get("split_sizes"))
    return _coerce_split_sizes(None)


def _coerce_positive_int(raw: Any, *, default: int, name: str, minimum: int = 1) -> int:
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"validation_strategy.{name} must be >= {minimum}.")
    return value


def _seed_policy_for_custom_strategy(
    *,
    strategy_name: str,
    run_count: int,
    seed_policy_mode: str,
    seed_policy: Optional[Dict[str, Any]],
    base_seed: Optional[int],
) -> Dict[str, Any]:
    # Reuse the robust policy when we need many generated seeds, then relabel runs below.
    protocol = "robust_qsar" if run_count > 2 else "standard_qsar" if run_count == 2 else "fast_local"
    policy = resolve_seed_policy(
        protocol=protocol,
        mode=seed_policy_mode,
        seed_policy=seed_policy,
        base_seed=base_seed,
    )
    split_runs = list(policy.get("split_runs") or [])
    if len(split_runs) >= run_count:
        seeds = [int(item["seed"]) for item in split_runs[:run_count]]
        model_seed = int(policy.get("model_seed") or split_runs[-1]["seed"])
    else:
        seeds = [int(item["seed"]) for item in split_runs]
        next_seed = int(policy.get("model_seed") or base_seed or 42)
        while len(seeds) < run_count:
            if next_seed not in seeds:
                seeds.append(next_seed)
            next_seed += 1
        model_seed = int(policy.get("model_seed") or seeds[-1])
    return {
        **policy,
        "strategy_name": strategy_name,
        "model_seed": model_seed,
        "generated_split_seeds": seeds,
        "split_runs": [],
    }


def _build_policy(
    *,
    strategy_name: str,
    strategy_type: str,
    reason: str,
    split_runs: List[SplitRun],
    seed_policy: Dict[str, Any],
    selection_metric: str,
    validation_strategy: Dict[str, Any],
    aggregation: str = "mean_std",
    final_refit: bool = False,
) -> Dict[str, Any]:
    run_dicts = [run.as_dict() for run in split_runs]
    seed_policy = dict(seed_policy)
    seed_policy["split_runs"] = run_dicts
    seed_policy["validation_strategy"] = validation_strategy
    return {
        "protocol": strategy_name,
        "reason": reason,
        "split_runs": run_dicts,
        "seed_policy": seed_policy,
        "validation_strategy": validation_strategy,
        "validation_strategy_type": strategy_type,
        "aggregation": aggregation,
        "selection_metric": selection_metric,
        "final_refit": final_refit,
    }


def resolve_validation_strategy(
    *,
    requested_protocol: Optional[str],
    validation_strategy: Optional[Mapping[str, Any]],
    training_profile: str,
    seed_policy: Optional[Dict[str, Any]] = None,
    seed_policy_mode: str = "generated_per_run",
    base_seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve a legacy protocol or explicit validation strategy into split runs."""
    strategy = _coerce_strategy(validation_strategy)
    if not strategy:
        return resolve_validation_protocol(
            requested_protocol=requested_protocol,
            training_profile=training_profile,
            seed_policy=seed_policy,
            seed_policy_mode=seed_policy_mode,
            base_seed=base_seed,
        )

    strategy_type = str(strategy.get("type") or strategy.get("strategy") or "holdout").strip().lower()
    selection_metric = str(strategy.get("selection_metric") or DEFAULT_SELECTION_METRIC).strip().lower()

    if strategy_type == "holdout":
        family = _infer_split_family(strategy)
        split_sizes = _coerce_strategy_split_sizes(strategy, family)
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name=f"{family}_holdout",
            run_count=1,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or strategy.get("split_seed") or base_seed,
        )
        seeds = [int(item) for item in seed_payload.get("generated_split_seeds", [])]
        seed = seeds[0] if seeds else int(seed_payload.get("model_seed") or 42)
        run = SplitRun(
            label=f"{family}_holdout",
            backend_split_type=SPLIT_FAMILY_TO_BACKEND_TYPE[family],
            seed=seed,
            primary=True,
            split_family=family,
            split_sizes=split_sizes,
        )
        return _build_policy(
            strategy_name=f"{family}_holdout",
            strategy_type=strategy_type,
            reason="Custom holdout validation strategy.",
            split_runs=[run],
            seed_policy=seed_payload,
            selection_metric=selection_metric,
            validation_strategy={**strategy, "split_sizes": split_sizes, "split_family": family},
        )

    if strategy_type == "repeated_holdout":
        family = _infer_split_family(strategy)
        n_repeats = _coerce_positive_int(strategy.get("n_repeats"), default=3, name="n_repeats", minimum=2)
        split_sizes = _coerce_strategy_split_sizes(strategy, family)
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name=f"repeated_{family}_holdout",
            run_count=n_repeats,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or strategy.get("split_seed") or base_seed,
        )
        seeds = [int(item) for item in seed_payload.get("generated_split_seeds", [])][:n_repeats]
        if len(seeds) < n_repeats:
            seeds = [int(seed_payload.get("model_seed") or 42) + idx for idx in range(n_repeats)]
        runs = [
            SplitRun(
                label=f"{family}_repeat_{index}",
                backend_split_type=SPLIT_FAMILY_TO_BACKEND_TYPE[family],
                seed=seeds[index - 1],
                primary=index == 1,
                split_family=family,
                split_sizes=split_sizes,
            )
            for index in range(1, n_repeats + 1)
        ]
        return _build_policy(
            strategy_name=f"repeated_{family}_holdout",
            strategy_type=strategy_type,
            reason="Custom repeated holdout validation strategy.",
            split_runs=runs,
            seed_policy=seed_payload,
            selection_metric=selection_metric,
            validation_strategy={
                **strategy,
                "split_family": family,
                "split_sizes": split_sizes,
                "n_repeats": n_repeats,
            },
        )

    raise ValueError(
        "Unsupported validation_strategy.type. Expected one of "
        "holdout, repeated_holdout."
    )
