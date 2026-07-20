#!/usr/bin/env python
# coding: utf-8
"""Shared hyperparameter-tuning contracts and lightweight study helpers.

The module deliberately owns *configuration* rather than model training.  Each
backend adapter receives a fixed development split and returns only compact
metrics; candidate models are never part of the persisted study contract.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol

HYPERPARAMETER_CONTRACT_VERSION = "1.2"


class HyperparameterTuningError(ValueError):
    """Raised when a tuning request is invalid for a backend or protocol."""


@dataclass(frozen=True)
class TuningEngineSpec:
    """Versioned description of one reusable hyperparameter-search engine."""

    name: str
    display_name: str
    description: str
    family: str
    sampler: Dict[str, Any] = field(default_factory=dict)
    backend_availability: Dict[str, Dict[str, str]] = field(default_factory=dict)
    stability: str = "stable"
    contract_version: str = HYPERPARAMETER_CONTRACT_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HyperparameterSpec:
    """A Qsaria-supported, user-facing model or training hyperparameter."""

    name: str
    description: str
    value_type: str
    default: Any = None
    direct_settable: bool = True
    tuning_supported: bool = False
    default_tuning: bool = False
    search_space: Optional[Dict[str, Any]] = None
    engines: tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TuningObjective:
    metric: str
    direction: str
    subset: str = "in_domain"

    def as_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TuningConfig:
    enabled: bool = True
    engine: Optional[str] = None
    n_trials: int = 50
    parameters: tuple[str, ...] = ()
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    objective: Optional[TuningObjective] = None
    seed: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["parameters"] = list(self.parameters)
        return payload


@dataclass(frozen=True)
class TuningStudySummary:
    """Compact, JSON-safe result kept with the final training run."""

    backend_name: str
    engine: str
    status: str
    objective: Dict[str, Any]
    requested_trials: int
    completed_trials: int
    failed_trials: int
    best_trial: Optional[Dict[str, Any]] = None
    trials: tuple[Dict[str, Any], ...] = ()
    parameterization: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    contract_version: str = HYPERPARAMETER_CONTRACT_VERSION
    engine_version: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["trials"] = list(self.trials)
        return payload


class TuningAdapter(Protocol):
    """Backend-specific bridge used by the common tuning registry."""

    engine_name: str

    def validate(self, config: TuningConfig) -> None:
        """Validate a normalized request before any training work starts."""

    def run(self, *args: Any, **kwargs: Any) -> TuningStudySummary | Dict[str, Any]:
        """Execute the engine and return a compact, serializable result."""


_REGRESSION_TUNING_METRICS = ("r2", "rmse", "mae")
_CLASSIFICATION_TUNING_METRICS = ("balanced_accuracy", "roc_auc", "f1_macro", "accuracy")
_MINIMIZE_TUNING_METRICS = {"mae", "mape", "mse", "rmse", "val_loss", "loss"}

# Optuna's multivariate TPE requires a stable distribution for every parameter
# it models jointly. ``num_leaves`` has a legal upper bound that depends on
# ``max_depth``; it is therefore represented internally as a fraction of the
# legal capacity and mapped back to the native LightGBM parameter before fit.
_MULTIVARIATE_LEAF_CAPACITY_COORDINATE = "__qsaria_leaf_capacity_fraction"
_MULTIVARIATE_LEAF_CAPACITY_SPACE = {"type": "float", "low": 0.0, "high": 1.0}


def _trial_metric_value(
    trial: Mapping[str, Any],
    *,
    metric: str,
    subset: str,
) -> Optional[float]:
    """Return one finite trial metric from its compact persisted record."""
    if metric == "val_loss" and isinstance(trial.get("objective"), (int, float)):
        return float(trial["objective"])
    metrics = trial.get("metrics") or {}
    subset_metrics = metrics.get(subset) if isinstance(metrics, Mapping) else None
    value = subset_metrics.get(metric) if isinstance(subset_metrics, Mapping) else None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _best_so_far(values: Iterable[Optional[float]], *, direction: str) -> list[Optional[float]]:
    """Build a monotonic incumbent curve while preserving missing trials."""
    best: Optional[float] = None
    progression: list[Optional[float]] = []
    for value in values:
        if value is not None and (
            best is None
            or (direction == "minimize" and value < best)
            or (direction != "minimize" and value > best)
        ):
            best = value
        progression.append(best)
    return progression


def _running_mean(values: Iterable[Optional[float]]) -> list[Optional[float]]:
    """Build the cumulative mean curve while preserving missing trials."""
    total = 0.0
    count = 0
    progression: list[Optional[float]] = []
    for value in values:
        if value is not None:
            total += value
            count += 1
        progression.append(total / count if count else None)
    return progression


def build_tuning_progress_plot(
    summary: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> Optional[str]:
    """Persist trial metrics and their incumbent progression as one compact plot.

    Candidate scores are deliberately shown as non-monotonic points: TPE explores
    configurations that can be worse than the incumbent. Each panel also shows
    the cumulative trial mean and the best score reached so far, which is the
    useful convergence signal.
    """
    trials = [
        trial
        for trial in (summary.get("trials") or [])
        if isinstance(trial, Mapping) and trial.get("state") == "complete"
    ]
    if not trials:
        return None

    objective = summary.get("objective") or {}
    objective_metric = str(objective.get("metric") or "objective")
    objective_subset = str(objective.get("subset") or "all")
    objective_direction = str(objective.get("direction") or "maximize")
    metric_names = {
        metric
        for trial in trials
        for metrics in [trial.get("metrics") or {}]
        if isinstance(metrics, Mapping)
        for all_metrics in [metrics.get("all") or {}]
        if isinstance(all_metrics, Mapping)
        for metric, value in all_metrics.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    canonical_metrics = (
        _REGRESSION_TUNING_METRICS
        if metric_names.intersection(_REGRESSION_TUNING_METRICS)
        else _CLASSIFICATION_TUNING_METRICS
    )
    panels: list[tuple[str, str, str]] = [(objective_metric, objective_subset, objective_direction)]
    for metric in canonical_metrics:
        if metric == objective_metric and objective_subset == "all":
            continue
        if metric in metric_names:
            direction = "minimize" if metric in _MINIMIZE_TUNING_METRICS else "maximize"
            panels.append((metric, "all", direction))
    if not panels:
        return None

    # Import lazily so the hyperparameter contract remains lightweight in callers
    # that only inspect schemas.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2 if len(panels) > 1 else 1
    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6.6 * columns, 4.2 * rows), squeeze=False)
    trial_numbers = [int(trial.get("number", index)) + 1 for index, trial in enumerate(trials)]

    # ``plt.subplots`` fills a rectangular grid.  For an odd number of panels
    # (for example the five classification metrics), that grid has one spare
    # axis.  Pair only the populated axes here; the spare axes are deliberately
    # hidden below.
    populated_axes = list(axes.flat)[: len(panels)]
    for axis, (metric, subset, direction) in zip(populated_axes, panels, strict=True):
        values = [_trial_metric_value(trial, metric=metric, subset=subset) for trial in trials]
        if not any(value is not None for value in values):
            axis.set_visible(False)
            continue
        incumbent = _best_so_far(values, direction=direction)
        running_mean = _running_mean(values)
        raw_x = [
            trial for trial, value in zip(trial_numbers, values, strict=True) if value is not None
        ]
        raw_y = [value for value in values if value is not None]
        axis.plot(raw_x, raw_y, "o", color="#4C78A8", alpha=0.7, label="trial")
        mean_x = [
            trial
            for trial, value in zip(trial_numbers, running_mean, strict=True)
            if value is not None
        ]
        mean_y = [value for value in running_mean if value is not None]
        axis.plot(
            mean_x,
            mean_y,
            color="#54A24B",
            linewidth=2.0,
            label="moyenne cumulative des trials",
        )
        incumbent_x = [
            trial
            for trial, value in zip(trial_numbers, incumbent, strict=True)
            if value is not None
        ]
        incumbent_y = [value for value in incumbent if value is not None]
        axis.step(
            incumbent_x,
            incumbent_y,
            where="post",
            color="#F58518",
            linewidth=2.2,
            label="meilleur score cumule",
        )
        suffix = " — objectif" if (metric, subset) == (objective_metric, objective_subset) else ""
        axis.set_title(f"{metric} ({subset}){suffix}")
        axis.set_xlabel("trial")
        axis.set_ylabel("valeur")
        axis.grid(alpha=0.2)
        axis.legend(loc="best")

    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Progression du tuning d'hyperparametres "
        f"({len(trials)} trials complets, {summary.get('engine') or 'moteur inconnu'})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    plot_path = Path(output_dir) / "hyperparameter_tuning_progress.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return str(plot_path)


TUNING_ENGINE_CATALOG: Dict[str, TuningEngineSpec] = {
    "optuna_tpe": TuningEngineSpec(
        name="optuna_tpe",
        display_name="Optuna TPE",
        description="Independent Tree-structured Parzen Estimator sampler.",
        family="optuna",
        sampler={"name": "TPESampler", "multivariate": False},
        backend_availability={
            "lightgbm": {"status": "supported", "detail": "LightGBM Optuna adapter."},
            "chemprop": {"status": "not_connected", "detail": "Chemprop uses native HyperOpt."},
            "tabicl": {"status": "unsupported", "detail": "TabICL has no tuning adapter."},
        },
    ),
    "optuna_tpe_multivariate": TuningEngineSpec(
        name="optuna_tpe_multivariate",
        display_name="Optuna TPE multivariate",
        description=(
            "Joint TPE sampler that models compatible hyperparameter combinations. "
            "LightGBM depth-constrained leaves use a stable internal coordinate."
        ),
        family="optuna",
        sampler={"name": "TPESampler", "multivariate": True, "group": False},
        backend_availability={
            "lightgbm": {"status": "supported", "detail": "LightGBM Optuna adapter."},
            "chemprop": {
                "status": "not_connected",
                "detail": "Reserved for a future Chemprop Optuna adapter; native HyperOpt remains active.",
            },
            "tabicl": {"status": "unsupported", "detail": "TabICL has no tuning adapter."},
        },
        stability="experimental",
    ),
    "chemprop_hpopt_hyperopt": TuningEngineSpec(
        name="chemprop_hpopt_hyperopt",
        display_name="Chemprop native HyperOpt",
        description="Chemprop native hpopt workflow using Ray Tune and HyperOpt.",
        family="chemprop",
        backend_availability={
            "lightgbm": {"status": "unsupported", "detail": "LightGBM uses Optuna adapters."},
            "chemprop": {"status": "native", "detail": "Chemprop native HPO adapter."},
            "tabicl": {"status": "unsupported", "detail": "TabICL has no tuning adapter."},
        },
    ),
}


LIGHTGBM_SPECS = (
    HyperparameterSpec(
        "n_estimators",
        "Number of boosting trees.",
        "int",
        500,
        tuning_supported=True,
        default_tuning=True,
        search_space={"type": "int", "low": 200, "high": 1200},
        engines=("optuna_tpe", "optuna_tpe_multivariate"),
    ),
    HyperparameterSpec(
        "learning_rate",
        "Shrinkage applied to each boosting step.",
        "float",
        0.05,
        tuning_supported=True,
        default_tuning=True,
        search_space={"type": "float", "low": 0.01, "high": 0.15, "log": True},
        engines=("optuna_tpe", "optuna_tpe_multivariate"),
    ),
    HyperparameterSpec(
        "max_depth",
        "Maximum tree depth; starts at 4 so the standard minimum of 15 leaves is legal.",
        "int",
        None,
        tuning_supported=True,
        default_tuning=True,
        search_space={"type": "int", "low": 4, "high": 12},
        engines=("optuna_tpe", "optuna_tpe_multivariate"),
    ),
    HyperparameterSpec(
        "num_leaves",
        "Maximum number of leaves per tree.",
        "int",
        63,
        tuning_supported=True,
        default_tuning=True,
        search_space={"type": "int", "low": 15, "high": 127},
        engines=("optuna_tpe", "optuna_tpe_multivariate"),
    ),
    HyperparameterSpec("subsample", "Row subsampling fraction.", "float", 0.8),
    HyperparameterSpec("colsample_bytree", "Feature subsampling fraction.", "float", 0.8),
    HyperparameterSpec("min_child_samples", "Minimum rows per leaf.", "int", 20),
    HyperparameterSpec("reg_alpha", "L1 regularization coefficient.", "float", 0.0),
    HyperparameterSpec("reg_lambda", "L2 regularization coefficient.", "float", 0.0),
    HyperparameterSpec("min_split_gain", "Minimum gain required to split.", "float", 0.0),
    HyperparameterSpec("boosting_type", "LightGBM boosting strategy.", "str", "gbdt"),
    HyperparameterSpec("early_stopping_rounds", "Advanced validation early stopping.", "int", 0),
)

CHEMPROP_SPECS = (
    HyperparameterSpec(
        "depth",
        "Number of message-passing steps.",
        "int",
        tuning_supported=True,
        default_tuning=True,
        engines=("chemprop_hpopt_hyperopt",),
    ),
    HyperparameterSpec(
        "message_hidden_dim",
        "Hidden dimension of the message-passing network.",
        "int",
        tuning_supported=True,
        default_tuning=True,
        engines=("chemprop_hpopt_hyperopt",),
    ),
    HyperparameterSpec(
        "ffn_hidden_dim",
        "Hidden dimension of the feed-forward prediction network.",
        "int",
        tuning_supported=True,
        default_tuning=True,
        engines=("chemprop_hpopt_hyperopt",),
    ),
    HyperparameterSpec(
        "ffn_num_layers",
        "Number of feed-forward prediction layers.",
        "int",
        tuning_supported=True,
        default_tuning=True,
        engines=("chemprop_hpopt_hyperopt",),
    ),
    HyperparameterSpec(
        "dropout",
        "Dropout probability in the message-passing model.",
        "float",
        tuning_supported=True,
        default_tuning=True,
        engines=("chemprop_hpopt_hyperopt",),
    ),
    HyperparameterSpec("epochs", "Maximum training epochs.", "int", 50),
    HyperparameterSpec("batch_size", "Batch size.", "int", 32),
    HyperparameterSpec("num_replicates", "Number of Chemprop training replicates.", "int", 1),
    HyperparameterSpec("ensemble_size", "Models trained in each replicate.", "int", 1),
    HyperparameterSpec("num_workers", "Data-loader worker count.", "int", 0),
    HyperparameterSpec("metric", "Native Chemprop evaluation metric.", "str", "rmse"),
    HyperparameterSpec(
        "multiclass_num_classes",
        "Shared number of classes for multiclass targets.",
        "int",
    ),
    HyperparameterSpec("accelerator", "Lightning accelerator selection.", "str"),
    HyperparameterSpec("devices", "Lightning device selection.", "int"),
    HyperparameterSpec("init_lr", "Initial learning rate.", "float", 0.0001),
    HyperparameterSpec("max_lr", "Maximum learning rate.", "float", 0.001),
    HyperparameterSpec("final_lr", "Final learning rate.", "float", 0.0001),
    HyperparameterSpec("warmup_epochs", "Learning-rate warmup epochs.", "int", 2),
    HyperparameterSpec("patience", "Validation patience for native Chemprop training.", "int"),
    HyperparameterSpec("tracking_metric", "Native Chemprop tracking metric.", "str", "val_loss"),
)

TABICL_SPECS = (
    HyperparameterSpec("n_estimators", "Number of TabICL ensemble estimators.", "int", 4),
    HyperparameterSpec("batch_size", "TabICL training/inference batch size.", "int", 32),
    HyperparameterSpec("norm_methods", "TabICL normalization methods.", "list"),
    HyperparameterSpec("feat_shuffle_method", "Feature shuffle strategy.", "str"),
    HyperparameterSpec("outlier_threshold", "Outlier handling threshold.", "float"),
    HyperparameterSpec("class_shuffle_method", "Classification-label shuffle strategy.", "str"),
    HyperparameterSpec("softmax_temperature", "Classification softmax temperature.", "float"),
    HyperparameterSpec("average_logits", "Average TabICL classifier logits.", "bool"),
    HyperparameterSpec("support_many_classes", "Enable wide multiclass support.", "bool"),
    HyperparameterSpec("device", "Execution device selection.", "str"),
    HyperparameterSpec("use_amp", "Use automatic mixed precision.", "bool"),
    HyperparameterSpec("use_fa3", "Use FlashAttention 3 when available.", "bool"),
    HyperparameterSpec("offload_mode", "TabICL offload strategy.", "str"),
)


BACKEND_TUNING_CATALOG: Dict[str, Dict[str, Any]] = {
    "lightgbm": {
        "contract_version": HYPERPARAMETER_CONTRACT_VERSION,
        "backend_name": "lightgbm",
        "supports_hyperparameter_tuning": True,
        "default_engine": "optuna_tpe",
        "supported_engines": ["optuna_tpe", "optuna_tpe_multivariate"],
        "default_trials": 50,
        "default_objectives": {
            "regression": {"metric": "rmse", "direction": "minimize", "subset": "in_domain"},
            "classification": {
                "metric": "balanced_accuracy",
                "direction": "maximize",
                "subset": "in_domain",
            },
            "multiclass_classification": {
                "metric": "balanced_accuracy",
                "direction": "maximize",
                "subset": "in_domain",
            },
        },
        "parameters": [item.as_dict() for item in LIGHTGBM_SPECS],
    },
    "chemprop": {
        "contract_version": HYPERPARAMETER_CONTRACT_VERSION,
        "backend_name": "chemprop",
        "supports_hyperparameter_tuning": True,
        "default_engine": "chemprop_hpopt_hyperopt",
        "supported_engines": ["chemprop_hpopt_hyperopt"],
        "default_trials": 50,
        "default_objectives": {
            "regression": {"metric": "val_loss", "direction": "minimize", "subset": "all"},
            "classification": {"metric": "val_loss", "direction": "minimize", "subset": "all"},
            "multiclass_classification": {
                "metric": "val_loss",
                "direction": "minimize",
                "subset": "all",
            },
        },
        "parameters": [item.as_dict() for item in CHEMPROP_SPECS],
    },
    "tabicl": {
        "contract_version": HYPERPARAMETER_CONTRACT_VERSION,
        "backend_name": "tabicl",
        "supports_hyperparameter_tuning": False,
        "default_engine": None,
        "supported_engines": [],
        "default_trials": None,
        "default_objectives": {},
        "parameters": [item.as_dict() for item in TABICL_SPECS],
    },
}


def describe_backend_hyperparameters(backend_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the Qsaria-owned tuning contract for one backend or all backends."""
    if backend_name is None:
        return {name: dict(payload) for name, payload in BACKEND_TUNING_CATALOG.items()}
    normalized = str(backend_name).strip().lower()
    if normalized not in BACKEND_TUNING_CATALOG:
        raise HyperparameterTuningError(f"Unknown tuning backend `{backend_name}`.")
    return dict(BACKEND_TUNING_CATALOG[normalized])


def describe_tuning_engines(engine_name: Optional[str] = None) -> Dict[str, Any]:
    """Return the reusable tuning-engine registry and per-backend availability."""
    if engine_name is None:
        return {name: spec.as_dict() for name, spec in TUNING_ENGINE_CATALOG.items()}
    normalized = str(engine_name).strip().lower()
    try:
        return TUNING_ENGINE_CATALOG[normalized].as_dict()
    except KeyError as exc:
        raise HyperparameterTuningError(f"Unknown tuning engine `{engine_name}`.") from exc


def tuning_sampler_metadata(engine_name: str, *, n_startup_trials: int) -> Dict[str, Any]:
    """Return the persisted sampler settings for an Optuna TPE engine."""
    engine = describe_tuning_engines(engine_name)
    sampler = dict(engine.get("sampler") or {})
    if sampler.get("name") != "TPESampler":
        return {}
    return {
        **sampler,
        "group": bool(sampler.get("group", False)),
        "n_startup_trials": int(n_startup_trials),
    }


def _specs(backend_name: str) -> Dict[str, Dict[str, Any]]:
    return {
        str(item["name"]): item
        for item in describe_backend_hyperparameters(backend_name).get("parameters", [])
    }


def default_tuning_parameter_names(
    backend_name: str,
    engine_name: Optional[str] = None,
) -> tuple[str, ...]:
    return tuple(
        item["name"]
        for item in _specs(backend_name).values()
        if item["default_tuning"] and (engine_name is None or engine_name in item["engines"])
    )


def default_tuning_objective(backend_name: str, task_type: str) -> TuningObjective:
    normalized_task = "classification" if task_type == "binary_classification" else task_type
    catalog = describe_backend_hyperparameters(backend_name)
    raw = (catalog.get("default_objectives") or {}).get(normalized_task)
    if not raw:
        raise HyperparameterTuningError(
            f"No default tuning objective is declared for {backend_name}/{task_type}."
        )
    return TuningObjective(
        metric=str(raw["metric"]),
        direction=str(raw["direction"]),
        subset=str(raw.get("subset") or "all"),
    )


def normalize_tuning_config(
    raw: Optional[Mapping[str, Any]],
    *,
    backend_name: str,
    task_type: str,
    eligible: bool,
    fixed_parameters: Iterable[str] = (),
) -> Optional[TuningConfig]:
    """Normalize a public request and enforce the shared V1 contract."""
    catalog = describe_backend_hyperparameters(backend_name)
    raw_config = dict(raw or {})
    explicitly_requested = raw is not None
    enabled = bool(raw_config.get("enabled", True))
    if not enabled:
        return None
    if not catalog["supports_hyperparameter_tuning"]:
        if explicitly_requested:
            raise HyperparameterTuningError(
                f"{backend_name} does not support automatic hyperparameter tuning in V1."
            )
        return None
    if not eligible:
        if explicitly_requested:
            raise HyperparameterTuningError(
                "Hyperparameter tuning requires a single holdout or cross-validation fold "
                "with a real validation split. Full-train and test-only holdouts do not provide one."
            )
        return None

    engine = str(raw_config.get("engine") or catalog["default_engine"])
    if engine not in catalog["supported_engines"]:
        raise HyperparameterTuningError(
            f"Unsupported tuning engine `{engine}` for {backend_name}. "
            f"Expected one of {catalog['supported_engines']}."
        )
    n_trials = int(raw_config.get("n_trials", catalog["default_trials"]))
    if n_trials < 1:
        raise HyperparameterTuningError("hyperparameter_tuning.n_trials must be >= 1.")

    specs = _specs(backend_name)
    requested_parameters = raw_config.get("parameters")
    if requested_parameters is None:
        parameters = list(default_tuning_parameter_names(backend_name, engine))
    elif not isinstance(requested_parameters, (list, tuple)):
        raise HyperparameterTuningError("hyperparameter_tuning.parameters must be a list.")
    else:
        parameters = [str(item) for item in requested_parameters]
    for name in parameters:
        spec = specs.get(name)
        if spec is None:
            raise HyperparameterTuningError(
                f"Unknown Qsaria hyperparameter `{name}` for {backend_name}."
            )
        if not spec["tuning_supported"]:
            raise HyperparameterTuningError(
                f"Hyperparameter `{name}` is direct-settable but not tunable by {engine} in V1."
            )
        if engine not in spec["engines"]:
            raise HyperparameterTuningError(
                f"Hyperparameter `{name}` is not tunable by {engine} for {backend_name}."
            )

    fixed = {str(item) for item in fixed_parameters}
    parameters = [name for name in parameters if name not in fixed]
    raw_space = raw_config.get("search_space") or {}
    if not isinstance(raw_space, Mapping):
        raise HyperparameterTuningError("hyperparameter_tuning.search_space must be an object.")
    search_space = {str(name): dict(value) for name, value in raw_space.items()}
    unknown_spaces = sorted(set(search_space).difference(parameters))
    if unknown_spaces:
        raise HyperparameterTuningError(
            "A custom search space may only target selected, non-fixed parameters: "
            + ", ".join(unknown_spaces)
        )

    raw_objective = raw_config.get("objective") or {}
    if not isinstance(raw_objective, Mapping):
        raise HyperparameterTuningError("hyperparameter_tuning.objective must be an object.")
    default_objective = default_tuning_objective(backend_name, task_type)
    objective = TuningObjective(
        metric=str(raw_objective.get("metric") or default_objective.metric),
        direction=str(raw_objective.get("direction") or default_objective.direction),
        subset=str(raw_objective.get("subset") or default_objective.subset),
    )
    if objective.subset not in {"all", "in_domain", "out_of_domain"}:
        raise HyperparameterTuningError(
            "objective.subset must be all, in_domain, or out_of_domain."
        )
    if objective.direction not in {"minimize", "maximize"}:
        raise HyperparameterTuningError("objective.direction must be minimize or maximize.")
    if backend_name == "lightgbm":
        regression_metrics = {
            "mse": "minimize",
            "mae": "minimize",
            "rmse": "minimize",
            "r2": "maximize",
        }
        classification_metrics = {
            "accuracy": "maximize",
            "balanced_accuracy": "maximize",
            "precision_macro": "maximize",
            "recall_macro": "maximize",
            "f1_macro": "maximize",
            "roc_auc": "maximize",
        }
        metric_directions = (
            regression_metrics if task_type == "regression" else classification_metrics
        )
        expected_direction = metric_directions.get(objective.metric)
        if expected_direction is None:
            raise HyperparameterTuningError(
                f"Metric `{objective.metric}` is not compatible with LightGBM {task_type} tuning."
            )
        if objective.direction != expected_direction:
            raise HyperparameterTuningError(
                f"Metric `{objective.metric}` must use direction `{expected_direction}`."
            )
    if backend_name == "chemprop" and objective.metric != "val_loss":
        raise HyperparameterTuningError(
            "Chemprop native HPO V1 always selects the global native val_loss."
        )
    if backend_name == "chemprop" and objective.subset != "all":
        raise HyperparameterTuningError(
            "Chemprop native HPO V1 always uses the global validation loss; objective.subset must be all."
        )
    return TuningConfig(
        enabled=True,
        engine=engine,
        n_trials=n_trials,
        parameters=tuple(parameters),
        search_space=search_space,
        objective=objective,
        seed=int(raw_config["seed"]) if raw_config.get("seed") is not None else None,
    )


def _suggest_lightgbm_parameter(trial: Any, name: str, space: Mapping[str, Any]) -> Any:
    kind = str(space.get("type") or "")
    low = space.get("low")
    high = space.get("high")
    if kind == "int":
        return trial.suggest_int(name, int(low), int(high), step=int(space.get("step") or 1))
    if kind == "float":
        return trial.suggest_float(name, float(low), float(high), log=bool(space.get("log", False)))
    raise HyperparameterTuningError(f"Unsupported search-space type `{kind}` for `{name}`.")


def _resolved_lightgbm_space(
    *,
    specs: Mapping[str, Mapping[str, Any]],
    config: TuningConfig,
    name: str,
) -> Dict[str, Any]:
    """Resolve one declared space with an optional user override."""
    space = dict(specs[name].get("search_space") or {})
    space.update(config.search_space.get(name) or {})
    return space


def _uses_multivariate_leaf_parameterization(config: TuningConfig) -> bool:
    """Whether depth and leaves are jointly tuned through a static coordinate."""
    return (
        config.engine == "optuna_tpe_multivariate"
        and "max_depth" in config.parameters
        and "num_leaves" in config.parameters
    )


def _leaf_bounds_for_depth(space: Mapping[str, Any], *, max_depth: int) -> tuple[int, int]:
    """Return the native LightGBM leaf bounds that are legal at one depth."""
    lower = int(space["low"])
    upper = min(int(space["high"]), 2 ** int(max_depth))
    if upper < lower:
        raise HyperparameterTuningError(
            "The requested num_leaves range is incompatible with max_depth="
            f"{max_depth}: expected a legal value <= {upper}, got a lower bound of {lower}."
        )
    return lower, upper


def _derive_num_leaves_from_capacity_fraction(
    *,
    fraction: float,
    max_depth: int,
    leaf_space: Mapping[str, Any],
) -> int:
    """Map a fixed [0, 1] coordinate to a legal, native LightGBM leaf count."""
    lower, upper = _leaf_bounds_for_depth(leaf_space, max_depth=max_depth)
    if not 0.0 <= float(fraction) <= 1.0:  # pragma: no cover - guaranteed by Optuna
        raise HyperparameterTuningError("leaf_capacity_fraction must be between 0 and 1.")
    return lower + math.floor(float(fraction) * (upper - lower) + 0.5)


def _multivariate_leaf_parameterization_metadata(
    leaf_space: Mapping[str, Any],
) -> Dict[str, Any]:
    """Describe the reversible internal coordinate persisted with a study."""
    return {
        "num_leaves": {
            "mode": "relative_to_depth_capacity",
            "internal_coordinate": "leaf_capacity_fraction",
            "internal_distribution": dict(_MULTIVARIATE_LEAF_CAPACITY_SPACE),
            "native_bounds": {"low": int(leaf_space["low"]), "high": int(leaf_space["high"])},
            "legal_capacity": "min(native_high, 2**max_depth)",
            "derivation": "round_half_up(native_low + fraction * (legal_capacity - native_low))",
            "reported_parameter": "num_leaves",
        }
    }


class LightGBMOptunaAdapter:
    """LightGBM bridge for the shared Optuna/TPE engine family."""

    engine_name = "optuna_tpe"
    supported_engine_names = ("optuna_tpe", "optuna_tpe_multivariate")

    def validate(self, config: TuningConfig) -> None:
        if config.engine not in self.supported_engine_names:
            raise HyperparameterTuningError(
                f"Expected one of {self.supported_engine_names}, got {config.engine}."
            )
        if config.objective is None:
            raise HyperparameterTuningError("LightGBM tuning requires an objective.")

    @staticmethod
    def _build_sampler(optuna: Any, config: TuningConfig) -> Any:
        """Build the configured Optuna sampler from the global engine registry."""
        engine = describe_tuning_engines(str(config.engine))
        sampler = dict(engine.get("sampler") or {})
        sampler_name = sampler.pop("name", None)
        if sampler_name != "TPESampler":
            raise HyperparameterTuningError(
                f"Engine `{config.engine}` is not backed by an Optuna TPESampler."
            )
        return optuna.samplers.TPESampler(
            seed=config.seed,
            n_startup_trials=min(10, config.n_trials),
            **sampler,
        )

    def run(
        self,
        *,
        config: TuningConfig,
        fixed_parameters: Mapping[str, Any],
        evaluate: Callable[[Dict[str, Any]], Dict[str, Any]],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> TuningStudySummary:
        """Run a sequential study; ``evaluate`` must never persist a candidate model."""
        self.validate(config)
        if {
            "max_depth",
            "num_leaves",
        }.issubset(fixed_parameters) and int(
            fixed_parameters["num_leaves"]
        ) > 2 ** int(fixed_parameters["max_depth"]):
            raise HyperparameterTuningError(
                "LightGBM requires num_leaves <= 2**max_depth for a fixed direct configuration."
            )
        if not config.parameters:
            return TuningStudySummary(
                backend_name="lightgbm",
                engine=str(config.engine),
                status="skipped",
                objective=config.objective.as_dict(),
                requested_trials=config.n_trials,
                completed_trials=0,
                failed_trials=0,
                reason="All default tuning parameters were fixed directly by the user.",
            )
        try:
            import optuna
        except ImportError as exc:  # pragma: no cover - dependency contract test covers this
            raise HyperparameterTuningError("Optuna is required for LightGBM tuning.") from exc

        specs = _specs("lightgbm")
        parameter_names = tuple(name for name in specs if name in config.parameters)
        uses_leaf_capacity_coordinate = _uses_multivariate_leaf_parameterization(config)
        leaf_space = (
            _resolved_lightgbm_space(specs=specs, config=config, name="num_leaves")
            if uses_leaf_capacity_coordinate
            else None
        )
        if leaf_space is not None:
            depth_space = _resolved_lightgbm_space(
                specs=specs,
                config=config,
                name="max_depth",
            )
            _leaf_bounds_for_depth(leaf_space, max_depth=int(depth_space["low"]))
        parameterization = (
            _multivariate_leaf_parameterization_metadata(leaf_space)
            if leaf_space is not None
            else {}
        )
        sampler = self._build_sampler(optuna, config)
        study = optuna.create_study(
            direction=config.objective.direction,
            sampler=sampler,
            pruner=optuna.pruners.NopPruner(),
        )

        def notify_progress(**payload: Any) -> None:
            """Publish best-effort telemetry without affecting the study result."""
            if progress_callback is None:
                return
            try:
                progress_callback(payload)
            except Exception:  # pragma: no cover - UI telemetry must never stop tuning
                pass

        def objective(trial: Any) -> float:
            notify_progress(
                event="trial_started",
                trial_number=trial.number,
                trial_index=trial.number + 1,
                total_trials=config.n_trials,
            )
            params = dict(fixed_parameters)
            tuning_coordinates: Dict[str, Any] = {}
            for name in parameter_names:
                space = _resolved_lightgbm_space(specs=specs, config=config, name=name)
                if name == "max_depth" and "num_leaves" in fixed_parameters:
                    required_depth = math.ceil(math.log2(int(fixed_parameters["num_leaves"])))
                    space["low"] = max(int(space["low"]), required_depth)
                    if int(space["low"]) > int(space["high"]):
                        raise HyperparameterTuningError(
                            "The requested max_depth range cannot represent the directly fixed "
                            f"num_leaves={fixed_parameters['num_leaves']}."
                        )
                if name == "num_leaves":
                    if uses_leaf_capacity_coordinate:
                        fraction = trial.suggest_float(
                            _MULTIVARIATE_LEAF_CAPACITY_COORDINATE,
                            float(_MULTIVARIATE_LEAF_CAPACITY_SPACE["low"]),
                            float(_MULTIVARIATE_LEAF_CAPACITY_SPACE["high"]),
                        )
                        params[name] = _derive_num_leaves_from_capacity_fraction(
                            fraction=fraction,
                            max_depth=int(params["max_depth"]),
                            leaf_space=space,
                        )
                        tuning_coordinates["leaf_capacity_fraction"] = float(fraction)
                        continue
                    max_depth = int(params.get("max_depth", 12))
                    lower, upper = _leaf_bounds_for_depth(space, max_depth=max_depth)
                    space["low"] = lower
                    space["high"] = upper
                params[name] = _suggest_lightgbm_parameter(trial, name, space)
            trial.set_user_attr(
                "resolved_params",
                {name: params[name] for name in parameter_names},
            )
            if tuning_coordinates:
                trial.set_user_attr("tuning_coordinates", tuning_coordinates)
            try:
                outcome = evaluate(params)
            except Exception:
                notify_progress(
                    event="trial_failed",
                    trial_number=trial.number,
                    trial_index=trial.number + 1,
                    total_trials=config.n_trials,
                )
                raise
            objective_value = outcome.get("objective")
            if objective_value is None:
                raise HyperparameterTuningError(
                    "A LightGBM trial did not produce an objective score."
                )
            trial.set_user_attr("metrics", outcome.get("metrics") or {})
            trial.set_user_attr("diagnostics", outcome.get("diagnostics") or {})
            notify_progress(
                event="trial_completed",
                trial_number=trial.number,
                trial_index=trial.number + 1,
                total_trials=config.n_trials,
                objective=float(objective_value),
            )
            return float(objective_value)

        study.optimize(objective, n_trials=config.n_trials, n_jobs=1, catch=(RuntimeError,))
        trial_rows = []
        for trial in study.trials:
            trial_rows.append(
                {
                    "number": trial.number,
                    "state": trial.state.name.lower(),
                    "params": dict(trial.user_attrs.get("resolved_params") or trial.params),
                    "tuning_coordinates": trial.user_attrs.get("tuning_coordinates") or {},
                    "objective": trial.value,
                    "metrics": trial.user_attrs.get("metrics") or {},
                    "diagnostics": trial.user_attrs.get("diagnostics") or {},
                    "duration_seconds": (
                        round((trial.datetime_complete - trial.datetime_start).total_seconds(), 3)
                        if trial.datetime_complete and trial.datetime_start
                        else None
                    ),
                }
            )
        completed = [item for item in trial_rows if item["state"] == "complete"]
        if not completed:
            return TuningStudySummary(
                backend_name="lightgbm",
                engine=str(config.engine),
                status="failed",
                objective=config.objective.as_dict(),
                requested_trials=config.n_trials,
                completed_trials=0,
                failed_trials=len(trial_rows),
                trials=tuple(trial_rows),
                parameterization=parameterization,
                reason="No LightGBM tuning trial completed successfully.",
            )
        best = study.best_trial
        best_row = next(item for item in trial_rows if item["number"] == best.number)
        best_row["params"] = {**fixed_parameters, **best_row["params"]}
        return TuningStudySummary(
            backend_name="lightgbm",
            engine=str(config.engine),
            status="completed",
            objective=config.objective.as_dict(),
            requested_trials=config.n_trials,
            completed_trials=len(completed),
            failed_trials=len(trial_rows) - len(completed),
            best_trial=best_row,
            trials=tuple(trial_rows),
            parameterization=parameterization,
        )


class ChempropHpoptAdapter:
    """Validation contract for the native Chemprop/Ray HyperOpt bridge."""

    engine_name = "chemprop_hpopt_hyperopt"

    def validate(self, config: TuningConfig) -> None:
        if config.engine != self.engine_name:
            raise HyperparameterTuningError(f"Expected {self.engine_name}, got {config.engine}.")
        if config.objective is None or config.objective.metric != "val_loss":
            raise HyperparameterTuningError("Chemprop HPO V1 must track val_loss.")

    def run(
        self,
        *,
        config: TuningConfig,
        execute: Callable[[], Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate and delegate to the native Chemprop/Ray runner."""
        self.validate(config)
        return execute()


def tuning_metadata_for_catalog(summary: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Keep only compact, useful provenance in model metadata."""
    if not summary:
        return {}
    return {
        "contract_version": summary.get("contract_version"),
        "status": summary.get("status"),
        "engine": summary.get("engine"),
        "engine_version": summary.get("engine_version"),
        "sampler": summary.get("sampler"),
        "seed": summary.get("seed"),
        "objective": summary.get("objective"),
        "requested_trials": summary.get("requested_trials"),
        "completed_trials": summary.get("completed_trials"),
        "best_trial": summary.get("best_trial"),
        "parameterization": summary.get("parameterization") or {},
        "summary_path": summary.get("summary_path"),
    }
