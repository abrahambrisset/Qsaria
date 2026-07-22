#!/usr/bin/env python
# coding: utf-8
"""Strict public and internal contracts for Qsaria model training.

Public models in this module are the only agent-facing source of truth.  Runtime
models are deliberately separate so paths, split payloads, persistence switches,
and worker telemetry can never be supplied by an agent.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRAINING_CONTRACT_VERSION = "2.0"

TaskType = Literal["regression", "classification", "multiclass_classification"]
RepresentationName = Literal[
    "rdkit_all",
    "morgan_only",
    "morgan_count_only",
    "morgan_binary_count_rdkit_all",
    "morgan_rdkit_all",
]
SplitFamily = Literal["random", "scaffold", "cluster"]
ComputeProfileName = Literal["auto", "local_light", "local_standard", "heavy_validation"]
TabICLNormMethod = Literal["none", "power", "quantile", "quantile_rtdl", "robust"]
TabICLShuffleMethod = Literal["none", "shift", "random", "latin"]
ChempropMetric = Literal[
    "mse",
    "mae",
    "rmse",
    "bounded-mse",
    "bounded-mae",
    "bounded-rmse",
    "r2",
    "binary-mcc",
    "multiclass-mcc",
    "roc",
    "prc",
    "accuracy",
    "f1",
]
TuningMetric = Literal[
    "mse",
    "mae",
    "rmse",
    "r2",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "roc_auc",
    "val_loss",
]

_TUNING_METRIC_ALIASES = {
    "mse": "mse",
    "mean squared error": "mse",
    "mae": "mae",
    "mean absolute error": "mae",
    "rmse": "rmse",
    "root mean squared error": "rmse",
    "r2": "r2",
    "r2 score": "r2",
    "coefficient of determination": "r2",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "balanced accuracy": "balanced_accuracy",
    "balanced acc": "balanced_accuracy",
    "precision macro": "precision_macro",
    "macro precision": "precision_macro",
    "recall macro": "recall_macro",
    "macro recall": "recall_macro",
    "f1 macro": "f1_macro",
    "macro f1": "f1_macro",
    "roc auc": "roc_auc",
    "auc roc": "roc_auc",
    "val loss": "val_loss",
    "validation loss": "val_loss",
}

_LIGHTGBM_REGRESSION_TUNING_DIRECTIONS = {
    "mse": "minimize",
    "mae": "minimize",
    "rmse": "minimize",
    "r2": "maximize",
}
_LIGHTGBM_CLASSIFICATION_TUNING_DIRECTIONS = {
    "accuracy": "maximize",
    "balanced_accuracy": "maximize",
    "precision_macro": "maximize",
    "recall_macro": "maximize",
    "f1_macro": "maximize",
    "roc_auc": "maximize",
}


class StrictContract(BaseModel):
    """Base for all Qsaria contracts exposed through tools or worker JSON."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class MolecularGraphRepresentation(StrictContract):
    kind: Literal["molecular_graph"] = "molecular_graph"


class GeneratedRepresentation(StrictContract):
    kind: Literal["generated"] = "generated"
    name: RepresentationName = "rdkit_all"


class PrecomputedRepresentation(StrictContract):
    kind: Literal["precomputed"] = "precomputed"
    feature_columns: List[str] = Field(min_length=1)
    categorical_feature_columns: List[str] = Field(default_factory=list)

    @field_validator("feature_columns", "categorical_feature_columns")
    @classmethod
    def _unique_nonempty_columns(cls, values: List[str]) -> List[str]:
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Feature column names must be non-empty strings.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Feature column names must be unique.")
        return normalized

    @model_validator(mode="after")
    def _categorical_columns_are_features(self) -> "PrecomputedRepresentation":
        unknown = sorted(set(self.categorical_feature_columns) - set(self.feature_columns))
        if unknown:
            raise ValueError(
                "categorical_feature_columns must be included in feature_columns: "
                + ", ".join(unknown)
            )
        return self


RepresentationConfig = Annotated[
    Union[MolecularGraphRepresentation, GeneratedRepresentation, PrecomputedRepresentation],
    Field(discriminator="kind"),
]


class StandardQsarValidation(StrictContract):
    kind: Literal["standard_qsar"] = "standard_qsar"
    seed: Optional[int] = Field(default=None, ge=0)


class HoldoutValidation(StrictContract):
    kind: Literal["holdout"] = "holdout"
    split_family: SplitFamily = "random"
    split_sizes: List[float] = Field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("split_sizes")
    @classmethod
    def _valid_split_sizes(cls, values: List[float]) -> List[float]:
        if len(values) not in {2, 3}:
            raise ValueError("split_sizes must contain two or three proportions.")
        normalized = [float(value) for value in values]
        if any(value <= 0.0 or value >= 1.0 for value in normalized):
            raise ValueError("Every split proportion must be strictly between 0 and 1.")
        if abs(sum(normalized) - 1.0) > 1e-8:
            raise ValueError("split_sizes must sum to 1.0.")
        return normalized


class RepeatedHoldoutValidation(HoldoutValidation):
    kind: Literal["repeated_holdout"] = "repeated_holdout"
    n_repeats: int = Field(default=3, ge=2)


class CrossValidation(StrictContract):
    kind: Literal["cross_validation"] = "cross_validation"
    split_family: Literal["random"] = "random"
    n_folds: int = Field(default=5, ge=2)
    n_repeats: int = Field(default=1, ge=1)
    outer_test_size: Optional[float] = Field(default=None, gt=0.0, lt=1.0)
    seed: Optional[int] = Field(default=None, ge=0)
    final_refit: bool = True


class FullTrainValidation(StrictContract):
    kind: Literal["full_train"] = "full_train"
    seed: Optional[int] = Field(default=None, ge=0)


ValidationConfig = Annotated[
    Union[
        StandardQsarValidation,
        HoldoutValidation,
        RepeatedHoldoutValidation,
        CrossValidation,
        FullTrainValidation,
    ],
    Field(discriminator="kind"),
]


class IntSearchRange(StrictContract):
    type: Literal["int"] = "int"
    low: int
    high: int
    step: int = Field(default=1, ge=1)
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "IntSearchRange":
        if self.high < self.low:
            raise ValueError("Search range high must be greater than or equal to low.")
        if self.log and self.step != 1:
            raise ValueError("A logarithmic integer range cannot define step != 1.")
        return self


class FloatSearchRange(StrictContract):
    type: Literal["float"] = "float"
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> "FloatSearchRange":
        if self.high <= self.low:
            raise ValueError("Search range high must be greater than low.")
        if self.log and self.low <= 0:
            raise ValueError("A logarithmic search range requires low > 0.")
        return self


class LightGBMTuningSpace(StrictContract):
    backend: Literal["lightgbm"] = "lightgbm"
    n_estimators: Optional[IntSearchRange] = None
    learning_rate: Optional[FloatSearchRange] = None
    max_depth: Optional[IntSearchRange] = None
    num_leaves: Optional[IntSearchRange] = None


class ChempropTuningSpace(StrictContract):
    backend: Literal["chemprop"] = "chemprop"
    depth: Optional[IntSearchRange] = None
    message_hidden_dim: Optional[IntSearchRange] = None
    ffn_hidden_dim: Optional[IntSearchRange] = None
    ffn_num_layers: Optional[IntSearchRange] = None
    dropout: Optional[FloatSearchRange] = None


TuningSpace = Annotated[
    Union[LightGBMTuningSpace, ChempropTuningSpace], Field(discriminator="backend")
]


class TuningObjective(StrictContract):
    metric: TuningMetric
    direction: Literal["minimize", "maximize"]
    subset: Literal["all", "in_domain", "out_of_domain"] = "in_domain"

    @field_validator("metric", mode="before")
    @classmethod
    def _canonical_metric(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        key = value.strip().casefold().replace("²", "2").replace("^2", "2")
        key = " ".join(key.replace("_", " ").replace("-", " ").split())
        return _TUNING_METRIC_ALIASES.get(key, key.replace(" ", "_"))


def validate_tuning_objective_compatibility(
    *,
    backend_name: str,
    task_type: str,
    metric: str,
    direction: str,
    subset: str,
) -> None:
    """Validate one canonical tuning objective before scientific execution."""

    if backend_name == "lightgbm":
        metric_directions = (
            _LIGHTGBM_REGRESSION_TUNING_DIRECTIONS
            if task_type == "regression"
            else _LIGHTGBM_CLASSIFICATION_TUNING_DIRECTIONS
        )
        expected_direction = metric_directions.get(metric)
        if expected_direction is None:
            raise ValueError(
                f"Metric `{metric}` is not compatible with LightGBM {task_type} tuning."
            )
        if direction != expected_direction:
            raise ValueError(f"Metric `{metric}` must use direction `{expected_direction}`.")
        return
    if backend_name == "chemprop":
        if metric != "val_loss":
            raise ValueError("Chemprop native HPO always selects the global native val_loss.")
        if direction != "minimize":
            raise ValueError("Metric `val_loss` must use direction `minimize`.")
        if subset != "all":
            raise ValueError(
                "Chemprop native HPO always uses the global validation loss; "
                "objective.subset must be all."
            )
        return
    raise ValueError(f"{backend_name} does not support a tuning objective.")


class TuningConfig(StrictContract):
    enabled: bool = True
    engine: Optional[
        Literal["optuna_tpe", "optuna_tpe_multivariate", "chemprop_hpopt_hyperopt"]
    ] = None
    n_trials: int = Field(default=50, ge=1)
    parameters: Optional[List[str]] = None
    search_space: Optional[TuningSpace] = None
    objective: Optional[TuningObjective] = None
    seed: Optional[int] = Field(default=None, ge=0)

    @field_validator("parameters")
    @classmethod
    def _unique_parameters(cls, values: Optional[List[str]]) -> Optional[List[str]]:
        if values is None:
            return None
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Tuning parameter names must be non-empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Tuning parameter names must be unique.")
        return normalized

    @model_validator(mode="after")
    def _disabled_has_no_search_request(self) -> "TuningConfig":
        configured = (
            self.engine is not None
            or self.n_trials != 50
            or self.parameters is not None
            or self.search_space is not None
            or self.objective is not None
            or self.seed is not None
        )
        if not self.enabled and configured:
            raise ValueError(
                "Disabled tuning cannot define engine, trials, parameters, search space, "
                "objective, or seed."
            )
        return self


class OutlierAnalysisConfig(StrictContract):
    enabled: bool = True
    selection_fraction: float = Field(default=0.10, gt=0.0, le=1.0)


class ActivityCliffConfig(StrictContract):
    index: Literal["sali"] = "sali"
    feedback: bool = False
    feedback_loops: int = Field(default=0, ge=0, le=3)
    similarity_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    top_k_neighbors: int = Field(default=10, ge=1)
    flag_threshold: float = Field(default=0.35, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _feedback_loop_consistency(self) -> "ActivityCliffConfig":
        if self.feedback and self.feedback_loops == 0:
            raise ValueError("feedback_loops must be between 1 and 3 when feedback is enabled.")
        if not self.feedback and self.feedback_loops != 0:
            raise ValueError("feedback_loops must be 0 when feedback is disabled.")
        return self


class ApplicabilityDomainConfig(StrictContract):
    methods: List[
        Literal["bounding_box", "isolation_forest", "similarity_matrix"]
    ] = Field(
        default_factory=lambda: ["bounding_box", "isolation_forest", "similarity_matrix"]
    )
    similarity_top_k_neighbors: Literal[1, 3, 5] = 5
    similarity_threshold_percentile: float = Field(default=5.0, ge=0.0, le=100.0)

    @field_validator("methods")
    @classmethod
    def _unique_methods(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one applicability-domain method is required.")
        if len(set(values)) != len(values):
            raise ValueError("Applicability-domain methods must be unique.")
        return values


class ComputeConfig(StrictContract):
    profile: ComputeProfileName = "auto"
    allow_heavy_compute: bool = False

    @model_validator(mode="after")
    def _heavy_requires_opt_in(self) -> "ComputeConfig":
        if self.profile == "heavy_validation" and not self.allow_heavy_compute:
            raise ValueError(
                "compute.allow_heavy_compute must be true for profile='heavy_validation'."
            )
        return self


class LightGBMConfig(StrictContract):
    name: Literal["lightgbm"] = "lightgbm"
    n_estimators: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    max_depth: Optional[int] = Field(default=None, ge=1)
    num_leaves: int = Field(default=63, ge=2)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    min_child_samples: int = Field(default=20, ge=1)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=0.0, ge=0.0)
    min_split_gain: float = Field(default=0.0, ge=0.0)
    boosting_type: Literal["gbdt", "dart", "rf"] = "gbdt"
    early_stopping_rounds: int = Field(default=0, ge=0)
    device: Literal["auto", "cpu", "gpu"] = "auto"
    gpu_fallback_to_cpu: bool = True


class ChempropConfig(StrictContract):
    name: Literal["chemprop"] = "chemprop"
    depth: Optional[int] = Field(default=None, ge=1)
    message_hidden_dim: Optional[int] = Field(default=None, ge=1)
    ffn_hidden_dim: Optional[int] = Field(default=None, ge=1)
    ffn_num_layers: Optional[int] = Field(default=None, ge=1)
    dropout: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    num_replicates: int = Field(default=1, ge=1)
    ensemble_size: int = Field(default=1, ge=1)
    num_workers: int = Field(default=0, ge=0)
    metric: ChempropMetric = "rmse"
    multiclass_num_classes: Optional[int] = Field(default=None, ge=2)
    accelerator: Optional[Literal["auto", "cpu", "gpu", "mps"]] = None
    devices: Optional[int] = Field(default=None, ge=1)
    init_lr: float = Field(default=0.0001, gt=0.0)
    max_lr: float = Field(default=0.001, gt=0.0)
    final_lr: float = Field(default=0.0001, gt=0.0)
    warmup_epochs: int = Field(default=2, ge=0)
    patience: Optional[int] = Field(default=None, ge=1)
    tracking_metric: str = Field(default="val_loss", pattern=r"^[A-Za-z0-9_-]+$")

    @model_validator(mode="after")
    def _ordered_learning_rates(self) -> "ChempropConfig":
        if self.max_lr < max(self.init_lr, self.final_lr):
            raise ValueError("max_lr must be greater than or equal to init_lr and final_lr.")
        if self.warmup_epochs >= self.epochs:
            raise ValueError("warmup_epochs must be strictly lower than epochs.")
        return self


class TabICLConfig(StrictContract):
    name: Literal["tabicl"] = "tabicl"
    n_estimators: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    norm_methods: Optional[List[TabICLNormMethod]] = None
    feat_shuffle_method: Optional[TabICLShuffleMethod] = None
    outlier_threshold: Optional[float] = Field(default=None, gt=0.0)
    class_shuffle_method: Optional[TabICLShuffleMethod] = None
    softmax_temperature: Optional[float] = Field(default=None, gt=0.0)
    average_logits: Optional[bool] = None
    support_many_classes: Optional[bool] = None


BackendConfig = Annotated[
    Union[LightGBMConfig, ChempropConfig, TabICLConfig], Field(discriminator="name")
]


class TrainingRequestBase(StrictContract):
    schema_version: Literal["2.0"] = "2.0"
    smiles_column: str = "smiles"
    target_columns: List[str] = Field(min_length=1)
    task_type: TaskType
    representation: RepresentationConfig
    validation: ValidationConfig = Field(default_factory=StandardQsarValidation)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    outlier_analysis: OutlierAnalysisConfig = Field(default_factory=OutlierAnalysisConfig)
    activity_cliffs: ActivityCliffConfig = Field(default_factory=ActivityCliffConfig)
    applicability_domain: ApplicabilityDomainConfig = Field(
        default_factory=ApplicabilityDomainConfig
    )
    compute: ComputeConfig = Field(default_factory=ComputeConfig)

    @field_validator("smiles_column")
    @classmethod
    def _nonempty_smiles_column(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("smiles_column must be non-empty.")
        return normalized

    @field_validator("target_columns")
    @classmethod
    def _valid_target_columns(cls, values: List[str]) -> List[str]:
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Target column names must be non-empty strings.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Target column names must be unique.")
        return normalized


class QsariaTrainingRequest(TrainingRequestBase):
    backend: BackendConfig

    @model_validator(mode="after")
    def _backend_compatibility(self) -> "QsariaTrainingRequest":
        backend_name = self.backend.name
        if backend_name == "chemprop" and self.representation.kind != "molecular_graph":
            raise ValueError("Chemprop requires representation.kind='molecular_graph'.")
        if backend_name in {"lightgbm", "tabicl"} and self.representation.kind == "molecular_graph":
            raise ValueError(f"{backend_name} requires a generated or precomputed representation.")
        if (
            backend_name != "lightgbm"
            and isinstance(self.representation, PrecomputedRepresentation)
            and self.representation.categorical_feature_columns
        ):
            raise ValueError("Categorical precomputed features are supported by LightGBM only.")

        allowed_tuning_parameters = {
            "lightgbm": {"n_estimators", "learning_rate", "max_depth", "num_leaves"},
            "chemprop": {
                "depth",
                "message_hidden_dim",
                "ffn_hidden_dim",
                "ffn_num_layers",
                "dropout",
            },
            "tabicl": set(),
        }[backend_name]
        unknown = sorted(set(self.tuning.parameters or []) - allowed_tuning_parameters)
        if unknown:
            raise ValueError(
                f"Unsupported {backend_name} tuning parameters: " + ", ".join(unknown)
            )
        if backend_name == "tabicl" and self.tuning.enabled:
            raise ValueError("TabICL does not support hyperparameter tuning.")
        if (
            backend_name == "lightgbm"
            and isinstance(self.validation, StandardQsarValidation)
            and not self.tuning.enabled
        ):
            raise ValueError(
                "LightGBM standard_qsar includes the canonical 50-trial Optuna/TPE "
                "tuning stage. Use an explicit holdout validation contract when "
                "requesting LightGBM without tuning."
            )
        if self.tuning.search_space is not None:
            if self.tuning.search_space.backend != backend_name:
                raise ValueError("The tuning search space must match backend.name.")
            supplied = {
                name
                for name, value in self.tuning.search_space.model_dump().items()
                if name != "backend" and value is not None
            }
            if self.tuning.parameters is not None and not supplied.issubset(
                set(self.tuning.parameters)
            ):
                raise ValueError(
                    "Every customized search-space field must also appear in tuning.parameters."
                )
        compatible_engines = {
            "lightgbm": {None, "optuna_tpe", "optuna_tpe_multivariate"},
            "chemprop": {None, "chemprop_hpopt_hyperopt"},
            "tabicl": {None},
        }[backend_name]
        if self.tuning.engine not in compatible_engines:
            raise ValueError(f"Tuning engine {self.tuning.engine!r} is not supported by {backend_name}.")
        if self.tuning.objective is not None:
            validate_tuning_objective_compatibility(
                backend_name=backend_name,
                task_type=self.task_type,
                metric=self.tuning.objective.metric,
                direction=self.tuning.objective.direction,
                subset=self.tuning.objective.subset,
            )
        has_validation = not isinstance(self.validation, FullTrainValidation)
        if isinstance(self.validation, (HoldoutValidation, RepeatedHoldoutValidation)):
            has_validation = len(self.validation.split_sizes) == 3
        if not has_validation and self.tuning.enabled:
            raise ValueError("Hyperparameter tuning requires a validation subset.")
        if not has_validation and self.outlier_analysis.enabled:
            raise ValueError("Outlier analysis requires validation or out-of-fold predictions.")
        if self.activity_cliffs.feedback and (
            self.task_type != "regression" or len(self.target_columns) != 1
        ):
            raise ValueError(
                "Activity Cliff feedback is supported only for single-target regression."
            )
        if isinstance(self.backend, ChempropConfig):
            if self.backend.multiclass_num_classes is not None and (
                self.task_type != "multiclass_classification"
            ):
                raise ValueError(
                    "chemprop.multiclass_num_classes is valid only for multiclass classification."
                )
        if isinstance(self.backend, TabICLConfig) and self.task_type == "regression":
            classification_fields = {
                "class_shuffle_method": self.backend.class_shuffle_method,
                "softmax_temperature": self.backend.softmax_temperature,
                "average_logits": self.backend.average_logits,
                "support_many_classes": self.backend.support_many_classes,
            }
            supplied = sorted(name for name, value in classification_fields.items() if value is not None)
            if supplied:
                raise ValueError(
                    "TabICL classification options are invalid for regression: "
                    + ", ".join(supplied)
                )
        return self


class LightGBMTrainingRequest(TrainingRequestBase):
    representation: Union[GeneratedRepresentation, PrecomputedRepresentation] = Field(
        default_factory=GeneratedRepresentation, discriminator="kind"
    )
    backend: LightGBMConfig = Field(default_factory=LightGBMConfig)

    @model_validator(mode="after")
    def _validate_complete(self) -> "LightGBMTrainingRequest":
        QsariaTrainingRequest.model_validate(self.model_dump())
        return self


class ChempropTrainingRequest(TrainingRequestBase):
    representation: MolecularGraphRepresentation = Field(
        default_factory=MolecularGraphRepresentation
    )
    backend: ChempropConfig = Field(default_factory=ChempropConfig)

    @model_validator(mode="after")
    def _validate_complete(self) -> "ChempropTrainingRequest":
        QsariaTrainingRequest.model_validate(self.model_dump())
        return self


class TabICLTrainingRequest(TrainingRequestBase):
    representation: Union[GeneratedRepresentation, PrecomputedRepresentation] = Field(
        default_factory=GeneratedRepresentation, discriminator="kind"
    )
    tuning: TuningConfig = Field(default_factory=lambda: TuningConfig(enabled=False))
    backend: TabICLConfig = Field(default_factory=TabICLConfig)

    @model_validator(mode="after")
    def _validate_complete(self) -> "TabICLTrainingRequest":
        QsariaTrainingRequest.model_validate(self.model_dump())
        return self


class QsariaBenchmarkRequest(StrictContract):
    schema_version: Literal["2.0"] = "2.0"
    smiles_column: str = "smiles"
    target_columns: List[str] = Field(min_length=1)
    task_type: TaskType
    backends: List[Literal["chemprop", "lightgbm", "tabicl"]] = Field(
        default_factory=lambda: ["chemprop", "lightgbm", "tabicl"]
    )
    include_candidate_variants: bool = True
    tabicl_candidate_variants: List[RepresentationName] = Field(default_factory=list)
    validation: ValidationConfig = Field(default_factory=StandardQsarValidation)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    benchmark_requested: Literal[True] = True

    @field_validator("backends")
    @classmethod
    def _unique_backends(cls, values: List[str]) -> List[str]:
        if not values:
            raise ValueError("At least one benchmark backend is required.")
        if len(set(values)) != len(values):
            raise ValueError("Benchmark backends must be unique.")
        return values


# Internal contracts.  These models are never exposed through Agno or MCP.


class RuntimePaths(StrictContract):
    output_dir: str
    bundle_path: Optional[str] = None
    feature_cache_dir: Optional[str] = None
    hpo_dir: Optional[str] = None
    offload_dir: Optional[str] = None
    heartbeat_path: Optional[str] = None


class BackendRunBase(StrictContract):
    training_contract_version: Literal["2.0"] = "2.0"
    train_csv: str
    output_dir: str
    task_type: TaskType
    smiles_columns: List[str] = Field(default_factory=lambda: ["smiles"])
    target_columns: List[str] = Field(min_length=1)
    reaction_columns: List[str] = Field(default_factory=list)

    def task_payload(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "smiles_columns": list(self.smiles_columns),
            "target_columns": list(self.target_columns),
            "reaction_columns": list(self.reaction_columns),
        }

    def parameter_payload(self) -> Dict[str, Any]:
        return self.model_dump(
            exclude={
                "training_contract_version",
                "backend",
                "train_csv",
                "output_dir",
                "task_type",
                "smiles_columns",
                "target_columns",
                "reaction_columns",
            },
            exclude_none=True,
        )


class LightGBMRunRequest(BackendRunBase):
    backend: Literal["lightgbm"] = "lightgbm"
    feature_columns: Optional[List[str]] = None
    categorical_feature_columns: List[str] = Field(default_factory=list)
    split_sizes: Optional[List[float]] = None
    split_type: str = "random"
    split_payload: Optional[List[Dict[str, Any]]] = None
    excluded_train_indices: List[int] = Field(default_factory=list)
    activity_cliff_variant_id: Optional[str] = None
    validation_protocol: str = "standard_qsar"
    random_state: int = 42
    n_estimators: int = Field(default=500, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    num_leaves: int = Field(default=63, ge=2)
    subsample: float = Field(default=0.8, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=0.8, gt=0.0, le=1.0)
    min_child_samples: int = Field(default=20, ge=1)
    reg_alpha: float = Field(default=0.0, ge=0.0)
    reg_lambda: float = Field(default=0.0, ge=0.0)
    max_depth: Optional[int] = Field(default=None, ge=1)
    min_split_gain: float = Field(default=0.0, ge=0.0)
    n_jobs: int = Field(default=1, ge=1)
    device_type: Optional[Literal["cpu", "gpu"]] = None
    use_gpu: Optional[bool] = None
    gpu_fallback_to_cpu: bool = True
    early_stopping_rounds: int = Field(default=0, ge=0)
    verbosity: int = -1
    boosting_type: Literal["gbdt", "dart", "rf"] = "gbdt"
    objective: Optional[str] = None
    metric: Optional[str] = None
    force_col_wise: Optional[bool] = None
    force_row_wise: Optional[bool] = None
    zero_as_missing: Optional[bool] = None
    use_missing: Optional[bool] = None
    deterministic: Optional[bool] = None
    final_refit: bool = False
    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    class_labels: List[Any] = Field(default_factory=list)
    persist_artifacts: bool = True
    return_prediction_frames: bool = False
    refit_on_train_validation: bool = False


class ChempropRunRequest(BackendRunBase):
    backend: Literal["chemprop"] = "chemprop"
    depth: Optional[int] = Field(default=None, ge=1)
    message_hidden_dim: Optional[int] = Field(default=None, ge=1)
    ffn_hidden_dim: Optional[int] = Field(default=None, ge=1)
    ffn_num_layers: Optional[int] = Field(default=None, ge=1)
    dropout: Optional[float] = Field(default=None, ge=0.0, lt=1.0)
    epochs: Optional[int] = Field(default=None, ge=1)
    batch_size: Optional[int] = Field(default=None, ge=1)
    num_replicates: Optional[int] = Field(default=None, ge=1)
    ensemble_size: Optional[int] = Field(default=None, ge=1)
    num_workers: Optional[int] = Field(default=None, ge=0)
    metric: Optional[str] = None
    multiclass_num_classes: Optional[int] = Field(default=None, ge=2)
    accelerator: Optional[str] = None
    devices: Optional[Union[int, str, List[int]]] = None
    init_lr: Optional[float] = Field(default=None, gt=0.0)
    max_lr: Optional[float] = Field(default=None, gt=0.0)
    final_lr: Optional[float] = Field(default=None, gt=0.0)
    warmup_epochs: Optional[int] = Field(default=None, ge=0)
    patience: Optional[int] = Field(default=None, ge=1)
    tracking_metric: Optional[str] = None
    splits_file: Optional[str] = None
    split_type: Optional[str] = None
    split: Optional[str] = None
    split_sizes: Optional[List[float]] = None
    data_seed: Optional[int] = Field(default=None, ge=0)
    raytune_num_samples: Optional[int] = Field(default=None, ge=1)
    raytune_num_workers: Optional[int] = Field(default=None, ge=1)
    raytune_max_concurrent_trials: Optional[int] = Field(default=None, ge=1)
    raytune_search_algorithm: Optional[str] = None
    raytune_trial_scheduler: Optional[str] = None
    raytune_use_gpu: Optional[bool] = None
    raytune_num_gpus: Optional[float] = Field(default=None, ge=0.0)
    raytune_temp_dir: Optional[str] = None
    hyperopt_seed: Optional[int] = Field(default=None, ge=0)
    hyperopt_random_state_seed: Optional[int] = Field(default=None, ge=0)
    search_parameter_keywords: Optional[List[str]] = None
    custom_tuning_space: Optional[ChempropTuningSpace] = None
    hpopt_save_dir: Optional[str] = None
    validation_protocol: Optional[str] = None
    random_state: Optional[int] = Field(default=None, ge=0)
    final_refit: bool = False
    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    class_labels: List[Any] = Field(default_factory=list)


class TabICLRunRequest(BackendRunBase):
    backend: Literal["tabicl"] = "tabicl"
    n_estimators: Optional[int] = Field(default=None, ge=1)
    norm_methods: Optional[List[TabICLNormMethod]] = None
    feat_shuffle_method: Optional[TabICLShuffleMethod] = None
    outlier_threshold: Optional[float] = Field(default=None, gt=0.0)
    batch_size: Optional[int] = Field(default=None, ge=1)
    kv_cache: Optional[bool] = None
    model_path: Optional[str] = None
    allow_auto_download: bool = False
    checkpoint_version: Optional[str] = None
    device: Optional[str] = None
    use_amp: Optional[bool] = None
    use_fa3: Optional[bool] = None
    offload_mode: Optional[str] = None
    disk_offload_dir: Optional[str] = None
    random_state: int = Field(default=0, ge=0)
    n_jobs: int = Field(default=1, ge=1)
    verbose: bool = False
    inference_config: Optional[Dict[str, Any]] = None
    class_shuffle_method: Optional[TabICLShuffleMethod] = None
    softmax_temperature: Optional[float] = Field(default=None, gt=0.0)
    average_logits: Optional[bool] = None
    support_many_classes: Optional[bool] = None
    checkpoint_dir: Optional[str] = None
    save_model_weights: bool = True
    save_training_data: bool = True
    save_kv_cache: bool = False
    split_sizes: Optional[List[float]] = None
    split_type: str = "random"
    validation_protocol: str = "standard_qsar"
    feature_columns: Optional[List[str]] = None
    split_payload: Optional[List[Dict[str, Any]]] = None
    applicability_domain_methods: Optional[
        List[Literal["bounding_box", "isolation_forest", "similarity_matrix"]]
    ] = None
    similarity_top_k_neighbors: Optional[Literal[1, 3, 5]] = None
    similarity_threshold_percentile: Optional[float] = Field(
        default=None, ge=0.0, le=100.0
    )
    heartbeat_seconds: float = Field(default=120.0, gt=0.0)
    heartbeat_path: Optional[str] = None
    heartbeat_label: Optional[str] = None
    heartbeat_run_index: Optional[int] = Field(default=None, ge=1)
    heartbeat_total_runs: Optional[int] = Field(default=None, ge=1)
    final_refit: bool = False
    classification_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    class_labels: List[Any] = Field(default_factory=list)


BackendRunRequest = Annotated[
    Union[LightGBMRunRequest, ChempropRunRequest, TabICLRunRequest],
    Field(discriminator="backend"),
]


def build_backend_run_request(
    *,
    backend: str,
    train_csv: str,
    output_dir: str,
    task_type: str,
    smiles_columns: List[str],
    target_columns: List[str],
    reaction_columns: Optional[List[str]] = None,
    resolved_parameters: Optional[Dict[str, Any]] = None,
) -> Union[LightGBMRunRequest, ChempropRunRequest, TabICLRunRequest]:
    """Validate code-generated runtime parameters at the backend boundary."""
    payload = {
        "train_csv": train_csv,
        "output_dir": output_dir,
        "task_type": task_type,
        "smiles_columns": list(smiles_columns),
        "target_columns": list(target_columns),
        "reaction_columns": list(reaction_columns or []),
        **dict(resolved_parameters or {}),
    }
    model = {
        "lightgbm": LightGBMRunRequest,
        "chemprop": ChempropRunRequest,
        "tabicl": TabICLRunRequest,
    }.get(str(backend).strip().lower())
    if model is None:
        raise ValueError(f"Unsupported backend run request: {backend}")
    return model.model_validate(payload)


class ResolvedTrainingPlan(StrictContract):
    training_contract_version: Literal["2.0"] = "2.0"
    requested_contract: QsariaTrainingRequest
    backend_name: Literal["chemprop", "lightgbm", "tabicl"]
    representation_name: str
    feature_columns: List[str] = Field(default_factory=list)
    categorical_feature_columns: List[str] = Field(default_factory=list)
    validation_protocol: Literal["standard_qsar", "custom"]
    validation_strategy: Dict[str, Any]
    split_runs: List[Dict[str, Any]] = Field(default_factory=list)
    seed_policy: Dict[str, Any] = Field(default_factory=dict)
    compute_profile: Literal["local_light", "local_standard", "heavy_validation"]
    tuning: TuningConfig
    outlier_analysis: OutlierAnalysisConfig
    activity_cliffs: ActivityCliffConfig
    applicability_domain: ApplicabilityDomainConfig
    effective_parameters: Dict[str, Any] = Field(default_factory=dict)
    runtime_paths: RuntimePaths


class TabICLWorkerJob(StrictContract):
    schema_version: Literal["1.0"] = "1.0"
    training_contract_version: Literal["2.0"] = "2.0"
    run_request: TabICLRunRequest
    representation_name: Optional[str] = None
    validation_strategy: Optional[Dict[str, Any]] = None
    seed_policy: Dict[str, Any]


class ChempropHpoptWorkerJob(StrictContract):
    """Versioned internal job for a Chemprop HPO subprocess."""

    schema_version: Literal["1.0"] = "1.0"
    training_contract_version: Literal["2.0"] = "2.0"
    argv: List[str] = Field(min_length=1)
    search_space: ChempropTuningSpace


def canonical_validation_strategy(validation: ValidationConfig) -> Optional[Dict[str, Any]]:
    """Return the canonical strategy payload expected by existing policy resolvers."""
    if isinstance(validation, StandardQsarValidation):
        return None
    payload = validation.model_dump(exclude_none=True)
    payload["type"] = payload.pop("kind")
    return payload
