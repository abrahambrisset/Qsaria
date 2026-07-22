#!/usr/bin/env python
# coding: utf-8
"""Configurable QSAR validation strategy resolver."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

from .qsar_training_policy import (
    DEFAULT_QSAR_SPLIT_SIZES,
    resolve_seed_policy,
    resolve_validation_protocol,
)

DEFAULT_SPLIT_SIZES = list(DEFAULT_QSAR_SPLIT_SIZES)

SPLIT_FAMILY_TO_BACKEND_TYPE = {
    "random": "random",
    "scaffold": "scaffold_balanced",
    "cluster": "kmeans",
}


@dataclass(frozen=True)
class SplitRun:
    label: str
    backend_split_type: str
    seed: int
    primary: bool = False
    split_family: str = "random"
    split_sizes: Optional[List[float]] = None
    repeat_index: Optional[int] = None
    fold_index: Optional[int] = None
    n_folds: Optional[int] = None
    n_repeats: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _coerce_strategy(raw: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("validation_strategy must be a dictionary/object.")
    return dict(raw)


def _coerce_split_family(raw: Any) -> str:
    family = str(raw or "random").strip().lower()
    if family not in SPLIT_FAMILY_TO_BACKEND_TYPE:
        raise ValueError(
            f"Unsupported split_family. Expected one of {sorted(SPLIT_FAMILY_TO_BACKEND_TYPE)}."
        )
    return family


def _infer_split_family(strategy: Mapping[str, Any]) -> str:
    return _coerce_split_family(strategy.get("split_family") or "random")


def _coerce_split_sizes(raw: Any) -> List[float]:
    if raw is None:
        return list(DEFAULT_SPLIT_SIZES)
    if not isinstance(raw, (list, tuple)) or len(raw) not in (2, 3):
        raise ValueError(
            "validation_strategy.split_sizes must be [train, test] or [train, validation, test]."
        )
    values = [float(item) for item in raw]
    if len(values) == 2 and values[0] == 1.0 and values[1] == 0.0:
        raise ValueError(
            "validation_strategy.split_sizes=[1.0, 0.0] is not a valid holdout. "
            "Use validation_strategy={'type': 'full_train'} to train on 100% of the dataset without test metrics."
        )
    total = sum(values)
    if (
        values[0] <= 0
        or values[-1] <= 0
        or any(item < 0 for item in values)
        or abs(total - 1.0) > 1e-6
    ):
        raise ValueError(
            "validation_strategy.split_sizes must sum to 1.0 with positive train/test ratios."
        )
    if len(values) == 3 and values[1] == 0:
        return [values[0], values[2]]
    return values


def _coerce_strategy_split_sizes(strategy: Mapping[str, Any], family: str) -> List[float]:
    del family
    return _coerce_split_sizes(strategy.get("split_sizes"))


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
    templates = [
        {
            "backend_split_type": "random",
            "primary": index == 0,
            "split_sizes": DEFAULT_SPLIT_SIZES,
        }
        for index in range(run_count)
    ]
    policy = resolve_seed_policy(
        split_templates=templates,
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
    """Resolve the standard protocol or an explicit validation strategy into split runs."""
    strategy = _coerce_strategy(validation_strategy)
    if not strategy:
        return resolve_validation_protocol(
            requested_protocol=requested_protocol,
            training_profile=training_profile,
            seed_policy=seed_policy,
            seed_policy_mode=seed_policy_mode,
            base_seed=base_seed,
        )

    strategy_type = str(strategy.get("type") or "").strip().lower()
    allowed_fields = {
        "holdout": {"type", "split_family", "split_sizes", "seed"},
        "repeated_holdout": {
            "type",
            "split_family",
            "split_sizes",
            "seed",
            "n_repeats",
        },
        "cross_validation": {
            "type",
            "split_family",
            "n_folds",
            "n_repeats",
            "outer_test_size",
            "seed",
            "final_refit",
        },
        "full_train": {"type", "seed"},
    }
    if strategy_type not in allowed_fields:
        raise ValueError(
            "Unsupported validation_strategy.type. Expected one of "
            "holdout, repeated_holdout, cross_validation, full_train."
        )
    unexpected = sorted(set(strategy) - allowed_fields[strategy_type])
    if unexpected:
        raise ValueError(
            f"Unsupported validation_strategy fields for {strategy_type}: {', '.join(unexpected)}."
        )
    if strategy_type == "full_train":
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name="full_train",
            run_count=1,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or base_seed,
        )
        seeds = [int(item) for item in seed_payload.get("generated_split_seeds", [])]
        seed = seeds[0] if seeds else int(seed_payload.get("model_seed") or 42)
        run = SplitRun(
            label="full_train",
            backend_split_type="final_refit",
            seed=seed,
            primary=True,
            split_family="full_train",
            split_sizes=[1.0],
        )
        return _build_policy(
            strategy_name="full_train",
            strategy_type="full_train",
            reason="Train on 100% of the dataset without internal validation or test metrics.",
            split_runs=[run],
            seed_policy=seed_payload,
            validation_strategy={**strategy, "type": "full_train", "split_sizes": [1.0]},
            aggregation="none",
            final_refit=True,
        )

    if strategy_type == "holdout":
        family = _infer_split_family(strategy)
        split_sizes = _coerce_strategy_split_sizes(strategy, family)
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name=f"{family}_holdout",
            run_count=1,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or base_seed,
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
            validation_strategy={**strategy, "split_sizes": split_sizes, "split_family": family},
        )

    if strategy_type == "repeated_holdout":
        family = _infer_split_family(strategy)
        n_repeats = _coerce_positive_int(
            strategy.get("n_repeats"), default=3, name="n_repeats", minimum=2
        )
        split_sizes = _coerce_strategy_split_sizes(strategy, family)
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name=f"repeated_{family}_holdout",
            run_count=n_repeats,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or base_seed,
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
            validation_strategy={
                **strategy,
                "split_family": family,
                "split_sizes": split_sizes,
                "n_repeats": n_repeats,
            },
        )

    if strategy_type == "cross_validation":
        family = _infer_split_family(strategy)
        if family != "random":
            raise ValueError("cross_validation currently supports split_family='random' only.")
        # A CV fold is always an inner validation set.  A separate external
        # test is intentionally named `outer_test_size`; accepting familiar
        # holdout aliases would silently produce a CV with no external test.
        n_folds = _coerce_positive_int(
            strategy.get("n_folds"),
            default=5,
            name="n_folds",
            minimum=2,
        )
        n_repeats = _coerce_positive_int(
            strategy.get("n_repeats"),
            default=1,
            name="n_repeats",
            minimum=1,
        )
        raw_outer_test_size = strategy.get("outer_test_size")
        if raw_outer_test_size in (None, 0, 0.0):
            outer_test_size = None
        else:
            try:
                outer_test_size = float(raw_outer_test_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "validation_strategy.outer_test_size must be a number strictly between 0 and 1."
                ) from exc
            if not 0.0 < outer_test_size < 1.0:
                raise ValueError(
                    "validation_strategy.outer_test_size must be strictly between 0 and 1."
                )
        seed_payload = _seed_policy_for_custom_strategy(
            strategy_name="cross_validation",
            run_count=1,
            seed_policy_mode=seed_policy_mode,
            seed_policy=seed_policy,
            base_seed=strategy.get("seed") or base_seed,
        )
        seed = int(
            (seed_payload.get("generated_split_seeds") or [seed_payload.get("model_seed") or 0])[0]
        )
        runs: List[SplitRun] = []
        for repeat_index in range(1, n_repeats + 1):
            for fold_index in range(1, n_folds + 1):
                runs.append(
                    SplitRun(
                        label=f"cv_repeat_{repeat_index}_fold_{fold_index}",
                        backend_split_type="cross_validation",
                        seed=seed,
                        primary=repeat_index == 1 and fold_index == 1,
                        split_family="random",
                        repeat_index=repeat_index,
                        fold_index=fold_index,
                        n_folds=n_folds,
                        n_repeats=n_repeats,
                    )
                )
        return _build_policy(
            strategy_name="cross_validation",
            strategy_type="cross_validation",
            reason="Repeated random K-fold cross-validation strategy.",
            split_runs=runs,
            seed_policy=seed_payload,
            validation_strategy={
                **strategy,
                "type": "cross_validation",
                "split_family": "random",
                "n_folds": n_folds,
                "n_repeats": n_repeats,
                "seed": seed,
                "outer_test_size": outer_test_size,
                "final_refit": bool(strategy.get("final_refit", True)),
            },
            aggregation="out_of_fold_mean_std",
            final_refit=bool(strategy.get("final_refit", True)),
        )

    raise AssertionError("Validated strategy type was not resolved.")
