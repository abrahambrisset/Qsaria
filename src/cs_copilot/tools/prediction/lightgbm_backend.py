#!/usr/bin/env python
# coding: utf-8
"""
LightGBM backend adapter for tabular QSAR workflows.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from cs_copilot.storage import S3
from cs_copilot.tools.activity_cliffs import ACTIVITY_CLIFF_ANNOTATION_PREFIX
from cs_copilot.tools.chemistry.standardize import resolve_smiles_column_name
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import enrich_backend_environment
from .qsar_splitters import build_full_train_split_payload, build_qsar_split_payload
from .qsar_training_policy import describe_compute_environment, project_now, safe_slug
from .tabular_representations import get_tabular_representation
from .training_orchestration import (
    classification_task_kind,
    compute_classification_metrics,
    compute_regression_metrics,
    decode_classification_labels,
    encode_classification_labels,
    is_classification_task,
    is_multiclass_task,
    json_safe_label,
    resolve_class_labels,
)

logger = logging.getLogger(__name__)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _coerce_split_sizes(split_sizes: Optional[List[float]]) -> List[float]:
    if not split_sizes:
        return [0.8, 0.1, 0.1]
    if len(split_sizes) == 1 and float(split_sizes[0]) == 1.0:
        return [1.0]
    if len(split_sizes) == 2 and float(split_sizes[0]) == 1.0 and float(split_sizes[1]) == 0.0:
        raise InvalidPredictionInputError(
            "split_sizes=[1.0, 0.0] is not a valid holdout. "
            "Use validation_strategy={'type': 'full_train'} for 100% training without test metrics."
        )
    if len(split_sizes) not in (2, 3):
        raise InvalidPredictionInputError("split_sizes must contain [train, test] or [train, val, test].")
    total = float(sum(split_sizes))
    if total <= 0:
        raise InvalidPredictionInputError("split_sizes must sum to a positive value.")
    normalized = [float(value) / total for value in split_sizes]
    if normalized[0] <= 0 or normalized[-1] <= 0 or any(value < 0 for value in normalized):
        raise InvalidPredictionInputError("split_sizes require positive train/test and non-negative validation.")
    if len(normalized) == 3 and normalized[1] == 0:
        return [normalized[0], normalized[2]]
    return normalized


def _normalize_category_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


class LightGBMBackend(PredictionBackend):
    """Prediction backend built around LightGBM regressors."""

    backend_name = "lightgbm"
    MODEL_EXTENSIONS = (".pkl",)

    def __init__(self) -> None:
        self._gpu_runtime_blocked_reason: Optional[str] = None

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("lightgbm")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("lightgbm") is not None

    def describe_environment(self) -> Dict[str, Any]:
        compute_env = describe_compute_environment()
        return enrich_backend_environment(
            self.backend_name,
            {
                "backend_name": self.backend_name,
                "available": self.is_available(),
                "package_version": self._package_version(),
                "cpu_available": True,
                "gpu_detected": bool(compute_env.get("gpu_available")),
                "gpu_count": compute_env.get("gpu_count"),
                "gpu_name": compute_env.get("gpu_name"),
                "supports_gpu_when_available": True,
                "gpu_runtime_blocked_reason": self._gpu_runtime_blocked_reason,
            },
        )

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"LightGBM model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        return path.resolve()

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendNotAvailableError(
                "LightGBM backend is not available. Install the optional `lightgbm` dependency first. "
                f"Environment snapshot: {self.describe_environment()}"
            )

    def _import_lightgbm(self):
        self._ensure_available()
        return importlib.import_module("lightgbm")

    def _sanitize_train_extra_args(self, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(extra_args or {})
        allowed = {
            "feature_columns",
            "categorical_feature_columns",
            "split_sizes",
            "split_type",
            "split_payload",
            "excluded_train_indices",
            "activity_cliff_variant_id",
            "validation_protocol",
            "random_state",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "subsample",
            "colsample_bytree",
            "min_child_samples",
            "reg_alpha",
            "reg_lambda",
            "max_depth",
            "min_split_gain",
            "n_jobs",
            "device_type",
            "use_gpu",
            "gpu_fallback_to_cpu",
            "early_stopping_rounds",
            "verbosity",
            "boosting_type",
            "objective",
            "metric",
            "force_col_wise",
            "force_row_wise",
            "zero_as_missing",
            "use_missing",
            "deterministic",
            "final_refit",
            "classification_threshold",
            "class_labels",
        }
        dropped = sorted(key for key in raw if key not in allowed)
        sanitized = {key: value for key, value in raw.items() if key in allowed}
        if dropped:
            logger.warning(
                "Dropping unsupported LightGBM train args for this V1 backend: %s",
                ", ".join(dropped),
            )
        return sanitized

    def _resolve_categorical_feature_columns(
        self,
        df: pd.DataFrame,
        *,
        feature_columns: List[str],
        extra_args: Dict[str, Any],
    ) -> List[str]:
        explicit = extra_args.get("categorical_feature_columns") or []
        if isinstance(explicit, str):
            explicit = [explicit]
        columns = [str(column) for column in explicit]
        missing = [column for column in columns if column not in df.columns]
        if missing:
            raise InvalidPredictionInputError(
                f"Requested categorical_feature_columns are missing: {missing}"
            )
        not_subset = [column for column in columns if column not in feature_columns]
        if not_subset:
            raise InvalidPredictionInputError(
                "categorical_feature_columns must be a subset of feature_columns. "
                f"Unexpected columns: {not_subset}"
            )
        return columns

    def _select_feature_columns(
        self,
        df: pd.DataFrame,
        task: PredictionTaskSpec,
        extra_args: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        excluded = set(task.target_columns) | set(task.smiles_columns)
        explicit = extra_args.get("feature_columns")
        if isinstance(explicit, str):
            explicit = [explicit]

        if explicit:
            feature_columns = [str(column) for column in explicit]
            leaked = [
                column
                for column in feature_columns
                if column.startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)
            ]
            if leaked:
                raise InvalidPredictionInputError(
                    "Activity-cliff annotation columns cannot be used as LightGBM features. "
                    f"Invalid columns: {leaked}"
                )
            categorical_feature_columns = self._resolve_categorical_feature_columns(
                df,
                feature_columns=feature_columns,
                extra_args=extra_args,
            )
            missing = [column for column in feature_columns if column not in df.columns]
            if missing:
                raise InvalidPredictionInputError(f"Requested feature columns are missing: {missing}")
            invalid = []
            for column in feature_columns:
                if column in categorical_feature_columns:
                    continue
                if not pd.api.types.is_numeric_dtype(df[column]):
                    invalid.append(column)
            if invalid:
                raise InvalidPredictionInputError(
                    "LightGBM feature_columns must be numeric unless they are explicitly listed in "
                    f"categorical_feature_columns. Invalid columns: {invalid}"
                )
            return feature_columns, categorical_feature_columns

        numeric_columns = [
            column
            for column in df.columns
            if column not in excluded
            and not str(column).startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)
            and pd.api.types.is_numeric_dtype(df[column])
        ]
        categorical_feature_columns = self._resolve_categorical_feature_columns(
            df,
            feature_columns=numeric_columns + list(extra_args.get("categorical_feature_columns") or []),
            extra_args=extra_args,
        )
        feature_columns = list(numeric_columns)
        for column in categorical_feature_columns:
            if column in excluded:
                raise InvalidPredictionInputError(
                    f"Categorical feature column '{column}' cannot also be a target or smiles column."
                )
            if column not in feature_columns:
                feature_columns.append(column)

        if not feature_columns:
            raise InvalidPredictionInputError(
                "No usable LightGBM feature columns were found. Provide numeric feature columns or "
                "explicit categorical_feature_columns."
            )
        return feature_columns, categorical_feature_columns

    def _encode_categorical_frame(
        self,
        frame: pd.DataFrame,
        categorical_feature_columns: List[str],
        *,
        category_mappings: Optional[Dict[str, Dict[Any, int]]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, int]]]:
        encoded = frame.copy()
        resolved_mappings: Dict[str, Dict[Any, int]] = {}
        for column in categorical_feature_columns:
            if column not in encoded.columns:
                raise InvalidPredictionInputError(
                    f"Categorical feature column '{column}' is missing from the feature frame."
                )
            mapping = dict((category_mappings or {}).get(column) or {})
            if not mapping:
                for raw_value in encoded[column].tolist():
                    normalized = _normalize_category_value(raw_value)
                    if normalized is None or normalized in mapping:
                        continue
                    mapping[normalized] = len(mapping)
            encoded[column] = (
                encoded[column]
                .map(lambda raw, mapping=mapping: mapping.get(_normalize_category_value(raw), -1))
                .astype("int32")
            )
            resolved_mappings[column] = mapping
        return encoded, resolved_mappings

    def _compute_regression_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        residuals = y_true - y_pred
        mse = float((residuals.pow(2)).mean())
        mae = float(residuals.abs().mean())
        rmse = float(math.sqrt(mse))
        centered = y_true - float(y_true.mean())
        ss_tot = float((centered.pow(2)).sum())
        ss_res = float((residuals.pow(2)).sum())
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None
        return {
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "n": int(len(y_true)),
        }

    def _default_model_params(self, extra_args: Dict[str, Any]) -> Dict[str, Any]:
        params = {
            "objective": str(extra_args.get("objective") or "regression"),
            "boosting_type": str(extra_args.get("boosting_type") or "gbdt"),
            "learning_rate": float(extra_args.get("learning_rate", 0.05)),
            "num_leaves": int(extra_args.get("num_leaves", 63)),
            "subsample": float(extra_args.get("subsample", 0.8)),
            "colsample_bytree": float(extra_args.get("colsample_bytree", 0.8)),
            "min_child_samples": int(extra_args.get("min_child_samples", 20)),
            "n_estimators": int(extra_args.get("n_estimators", 500)),
            "random_state": int(extra_args.get("random_state", 42)),
            "n_jobs": int(extra_args.get("n_jobs", 1)),
            "verbosity": int(extra_args.get("verbosity", -1)),
        }
        if "max_depth" in extra_args:
            params["max_depth"] = int(extra_args["max_depth"])
        if "reg_alpha" in extra_args:
            params["reg_alpha"] = float(extra_args["reg_alpha"])
        if "reg_lambda" in extra_args:
            params["reg_lambda"] = float(extra_args["reg_lambda"])
        if "min_split_gain" in extra_args:
            params["min_split_gain"] = float(extra_args["min_split_gain"])
        if "force_col_wise" in extra_args:
            params["force_col_wise"] = bool(extra_args["force_col_wise"])
        if "force_row_wise" in extra_args:
            params["force_row_wise"] = bool(extra_args["force_row_wise"])
        if "zero_as_missing" in extra_args:
            params["zero_as_missing"] = bool(extra_args["zero_as_missing"])
        if "use_missing" in extra_args:
            params["use_missing"] = bool(extra_args["use_missing"])
        if "deterministic" in extra_args:
            params["deterministic"] = bool(extra_args["deterministic"])
        return params

    def _resolve_device_type(self, extra_args: Dict[str, Any]) -> Tuple[str, bool, Dict[str, Any]]:
        compute_env = describe_compute_environment()
        if self._gpu_runtime_blocked_reason:
            return "cpu", False, compute_env
        explicit_device = extra_args.get("device_type")
        explicit_use_gpu = extra_args.get("use_gpu")
        if explicit_device:
            return str(explicit_device).lower(), False, compute_env
        if explicit_use_gpu is not None:
            return ("gpu" if bool(explicit_use_gpu) else "cpu"), False, compute_env
        return "cpu", False, compute_env

    def _fit_regressor(
        self,
        *,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        categorical_feature_columns: List[str],
        early_stopping_rounds: int,
    ):
        lgb = self._import_lightgbm()
        regressor = lgb.LGBMRegressor(**model_params)
        callbacks: List[Any] = [lgb.log_evaluation(period=0)]
        has_validation = X_val is not None and y_val is not None
        if has_validation and early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )
        fit_kwargs: Dict[str, Any] = {"callbacks": callbacks}
        if has_validation:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if categorical_feature_columns:
            fit_kwargs["categorical_feature"] = list(categorical_feature_columns)
        regressor.fit(X_train, y_train, **fit_kwargs)
        return regressor

    def _fit_classifier(
        self,
        *,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        categorical_feature_columns: List[str],
        early_stopping_rounds: int,
    ):
        lgb = self._import_lightgbm()
        classifier = lgb.LGBMClassifier(**model_params)
        callbacks: List[Any] = [lgb.log_evaluation(period=0)]
        has_validation = X_val is not None and y_val is not None
        if has_validation and early_stopping_rounds > 0:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))
        fit_kwargs: Dict[str, Any] = {"callbacks": callbacks}
        if has_validation:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
        if categorical_feature_columns:
            fit_kwargs["categorical_feature"] = list(categorical_feature_columns)
        classifier.fit(X_train, y_train, **fit_kwargs)
        return classifier

    def _classification_output_frame(
        self,
        *,
        predictions: Any,
        probabilities: Any,
        class_labels: List[Any],
        target_column: str,
        source: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        predicted_labels = decode_classification_labels(predictions, class_labels)
        output = pd.DataFrame(
            {
                "prediction": predicted_labels,
                target_column: predicted_labels,
                "prediction_class_code": pd.Series(predictions).astype(int),
            }
        )
        if probabilities is not None:
            proba = pd.DataFrame(probabilities)
            for index, class_label in enumerate(class_labels[: proba.shape[1]]):
                column = f"probability_{safe_slug(str(json_safe_label(class_label))) or f'class_{index}'}"
                output[column] = pd.to_numeric(proba.iloc[:, index], errors="coerce")
            if len(class_labels) == 2 and proba.shape[1] >= 2:
                output["positive_probability"] = pd.to_numeric(proba.iloc[:, 1], errors="coerce")
        if source is not None:
            for column in ("smiles", "Drug_ID"):
                if column in source.columns:
                    output.insert(0, column, source[column].reset_index(drop=True))
        return output

    def _featurize_prediction_input_if_possible(
        self,
        df: pd.DataFrame,
        *,
        input_csv: str,
        model_record: PredictionModelRecord,
        feature_columns: List[str],
    ) -> pd.DataFrame:
        representation_name = (
            (model_record.inference_profile or {}).get("representation_name")
            or (model_record.training_data_summary or {}).get("representation_name")
            or (model_record.selection_hints or {}).get("representation_name")
        )
        if not representation_name:
            return df
        try:
            spec = get_tabular_representation(str(representation_name))
        except ValueError:
            return df
        try:
            smiles_column = resolve_smiles_column_name(df, "smiles")
        except ValueError:
            return df

        input_path = Path(input_csv).expanduser()
        feature_dir = (
            input_path.parent / ".lightgbm_prediction_features"
            if input_path.is_absolute()
            else Path(".files") / "prediction_features" / safe_slug(model_record.model_id)
        )
        feature_dir.mkdir(parents=True, exist_ok=True)
        toolkit = MolecularFeatureToolkit()
        feature_frames: List[pd.DataFrame] = []

        if spec.use_morgan_binary:
            output_csv = str(feature_dir / "morgan_binary.csv")
            toolkit.smiles_to_morgan_fingerprints(
                input_csv=input_csv,
                smiles_column=smiles_column,
                output_csv=output_csv,
                include_input_columns=True,
                input_columns_to_keep=["smiles"],
                fingerprint_kind="binary",
                n_jobs=1,
            )
            feature_frames.append(_strip_unnamed_columns(pd.read_csv(output_csv)))
        if spec.use_morgan_count:
            output_csv = str(feature_dir / "morgan_count.csv")
            toolkit.smiles_to_morgan_fingerprints(
                input_csv=input_csv,
                smiles_column=smiles_column,
                output_csv=output_csv,
                include_input_columns=True,
                input_columns_to_keep=["smiles"],
                fingerprint_kind="count",
                n_jobs=1,
            )
            feature_frames.append(_strip_unnamed_columns(pd.read_csv(output_csv)))
        if spec.use_rdkit:
            output_csv = str(feature_dir / "rdkit_descriptors.csv")
            toolkit.smiles_to_rdkit_descriptors(
                input_csv=input_csv,
                smiles_column=smiles_column,
                output_csv=output_csv,
                descriptor_set=str(spec.descriptor_set or "basic"),
                include_input_columns=True,
                input_columns_to_keep=["smiles"],
                n_jobs=1,
            )
            feature_frames.append(_strip_unnamed_columns(pd.read_csv(output_csv)))

        assembled = df.copy()
        feature_additions: List[pd.DataFrame] = []
        for feature_df in feature_frames:
            columns_to_add = [
                column for column in feature_columns
                if column in feature_df.columns and column not in assembled.columns
            ]
            if columns_to_add:
                feature_additions.append(feature_df[columns_to_add].reset_index(drop=True))
        if feature_additions:
            assembled = pd.concat([assembled.reset_index(drop=True), *feature_additions], axis=1)
        return assembled

    def _is_gpu_runtime_unavailable(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "no opencl device found" in message or "opencl" in message

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if return_uncertainty:
            raise InvalidPredictionInputError("LightGBM V1 does not support predictive uncertainty export.")

        model_path = self.validate_model_path(model_record.model_path)
        try:
            with model_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            raise PredictionExecutionError(
                f"Could not load LightGBM model artifact {model_path}: {exc}"
            ) from exc

        with S3.open(input_csv, "r") as fh:
            df = _strip_unnamed_columns(pd.read_csv(fh))

        feature_columns = list((payload or {}).get("feature_columns") or [])
        categorical_feature_columns = list(
            (payload or {}).get("categorical_feature_columns") or []
        )
        category_mappings = dict((payload or {}).get("categorical_mappings") or {})
        target_columns = list(model_record.task.target_columns)

        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            df = self._featurize_prediction_input_if_possible(
                df,
                input_csv=input_csv,
                model_record=model_record,
                feature_columns=feature_columns,
            )
            missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise InvalidPredictionInputError(
                f"Prediction input is missing feature columns: {missing_features}"
            )

        features = df[feature_columns].copy()
        features, _ = self._encode_categorical_frame(
            features,
            categorical_feature_columns,
            category_mappings=category_mappings,
        )
        task_type = str((payload or {}).get("task_type") or model_record.task.task_type or "regression")
        try:
            if is_classification_task(task_type):
                class_labels = list((payload or {}).get("class_labels") or [])
                if not class_labels:
                    raise InvalidPredictionInputError("LightGBM classification artifact is missing class_labels metadata.")
                y_pred = payload["model"].predict(features)
                probabilities = payload["model"].predict_proba(features) if hasattr(payload["model"], "predict_proba") else None
                output = self._classification_output_frame(
                    predictions=y_pred,
                    probabilities=probabilities,
                    class_labels=class_labels,
                    target_column=target_columns[0] if len(target_columns) == 1 else "prediction",
                    source=df,
                )
            else:
                y_pred = payload["model"].predict(features)
                output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
        except Exception as exc:
            raise PredictionExecutionError(f"LightGBM prediction failed: {exc}") from exc
        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)

        return {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
        }

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_available()
        task_is_classification = is_classification_task(task.task_type)
        if task.task_type != "regression" and not task_is_classification:
            raise InvalidPredictionInputError("LightGBM V1 supports regression and classification tasks.")
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError("LightGBM V1 requires exactly one target column.")

        sanitized_args = self._sanitize_train_extra_args(extra_args)
        raw_split_sizes = sanitized_args.pop("split_sizes", None)
        split_payload = sanitized_args.pop("split_payload", None)
        excluded_train_indices = {
            int(idx) for idx in (sanitized_args.pop("excluded_train_indices", None) or [])
        }
        activity_cliff_variant_id = sanitized_args.pop("activity_cliff_variant_id", None)
        random_state = int(sanitized_args.get("random_state", 42))
        split_type = str(sanitized_args.get("split_type", "random"))
        validation_protocol = str(sanitized_args.get("validation_protocol", "standard_qsar"))
        final_refit = bool(sanitized_args.get("final_refit", False))
        split_sizes = [1.0] if final_refit and raw_split_sizes in (None, [1.0], (1.0,)) else _coerce_split_sizes(raw_split_sizes)
        target_column = task.target_columns[0]
        started_at = project_now()

        with S3.open(train_csv, "r") as fh:
            dataset = _strip_unnamed_columns(pd.read_csv(fh))

        if target_column not in dataset.columns:
            raise InvalidPredictionInputError(f"Missing target column: {target_column}")
        if dataset[target_column].isna().any():
            raise InvalidPredictionInputError(
                f"LightGBM training target '{target_column}' contains missing values."
            )

        feature_columns, categorical_feature_columns = self._select_feature_columns(
            dataset,
            task,
            sanitized_args,
        )
        working = dataset[
            feature_columns + [target_column] + [c for c in ("smiles", "Drug_ID") if c in dataset.columns]
        ].copy()
        if len(working) < 10:
            raise InvalidPredictionInputError(
                "LightGBM V1 requires at least 10 rows after target cleanup."
            )

        encoded_features, categorical_mappings = self._encode_categorical_frame(
            working[feature_columns].copy(),
            categorical_feature_columns,
        )
        encoded_working = working.copy()
        encoded_working[feature_columns] = encoded_features

        if final_refit:
            split_payload = build_full_train_split_payload(df=encoded_working)
        elif not split_payload:
            split_payload = build_qsar_split_payload(
                df=encoded_working,
                split_type=split_type,
                split_sizes=split_sizes,
                random_state=random_state,
                smiles_column=task.smiles_columns[0] if task.smiles_columns else None,
                feature_columns=feature_columns,
            )

        if not split_payload or "train" not in split_payload[0]:
            raise InvalidPredictionInputError(
                "LightGBM split payload must provide non-empty train/test indices."
            )

        split_indices = split_payload[0]
        source_train_idx = [int(idx) for idx in (split_indices.get("train") or [])]
        train_idx = [idx for idx in source_train_idx if idx not in excluded_train_indices]
        excluded_from_train = sorted(set(source_train_idx) & excluded_train_indices)
        val_idx = split_indices.get("val") or []
        test_idx = split_indices.get("test") or []
        if not train_idx or (not final_refit and not test_idx):
            raise InvalidPredictionInputError(
                "LightGBM split payload must provide non-empty train/test indices."
            )
        effective_split_payload = [
            {
                **split_indices,
                "train": train_idx,
                "excluded_from_train": excluded_from_train,
            }
        ]

        X_train = encoded_working.iloc[train_idx][feature_columns].copy()
        class_labels: List[Any] = []
        class_mapping: Dict[str, int] = {}
        if task_is_classification:
            class_labels = resolve_class_labels(encoded_working[target_column])
            if len(class_labels) < 2:
                raise InvalidPredictionInputError("LightGBM classification requires at least two target classes.")
            if is_multiclass_task(task.task_type) is False and len(class_labels) > 2:
                # Plain `classification` may still infer multiclass from data; keep it explicit in metadata.
                pass
            encoded_target, class_mapping = encode_classification_labels(encoded_working[target_column], class_labels)
            if encoded_target.isna().any():
                raise InvalidPredictionInputError("LightGBM classification target contains labels outside class_labels.")
            y_all = encoded_target.astype(int)
        else:
            y_all = pd.to_numeric(encoded_working[target_column], errors="coerce").astype(float)

        y_train = y_all.iloc[train_idx].copy()
        X_val = encoded_working.iloc[val_idx][feature_columns].copy() if val_idx else None
        y_val = y_all.iloc[val_idx].copy() if val_idx else None
        X_test = encoded_working.iloc[test_idx][feature_columns].copy() if test_idx else None
        y_test = y_all.iloc[test_idx].copy() if test_idx else None
        if task_is_classification:
            present_train_classes = {int(value) for value in y_train.tolist()}
            missing_train_classes = sorted(set(range(len(class_labels))) - present_train_classes)
            if missing_train_classes:
                missing = [json_safe_label(class_labels[index]) for index in missing_train_classes]
                raise InvalidPredictionInputError(
                    f"LightGBM classification training split is missing target classes {missing}."
                )

        requested_device_type, auto_device, compute_env = self._resolve_device_type(sanitized_args)
        gpu_fallback_to_cpu = bool(sanitized_args.get("gpu_fallback_to_cpu", True))
        early_stopping_rounds = int(sanitized_args.get("early_stopping_rounds", 50))
        model_params = self._default_model_params(sanitized_args)
        if task_is_classification:
            model_params["objective"] = "multiclass" if len(class_labels) > 2 else "binary"
            model_params["metric"] = sanitized_args.get("metric") or ("multi_logloss" if len(class_labels) > 2 else "binary_logloss")
            if len(class_labels) > 2:
                model_params["num_class"] = len(class_labels)
        model_params["device_type"] = requested_device_type
        fit_error: Optional[Exception] = None

        try:
            fit_fn = self._fit_classifier if task_is_classification else self._fit_regressor
            regressor = fit_fn(
                model_params=model_params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_feature_columns=categorical_feature_columns,
                early_stopping_rounds=early_stopping_rounds,
            )
            actual_device_type = requested_device_type
        except Exception as exc:
            fit_error = exc
            if requested_device_type != "cpu" and (auto_device or gpu_fallback_to_cpu):
                if self._is_gpu_runtime_unavailable(exc):
                    self._gpu_runtime_blocked_reason = str(exc)
                logger.warning(
                    "LightGBM GPU training failed with device_type=%s; retrying on CPU. Error: %s",
                    requested_device_type,
                    exc,
                )
                model_params["device_type"] = "cpu"
                fit_fn = self._fit_classifier if task_is_classification else self._fit_regressor
                regressor = fit_fn(
                    model_params=model_params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    categorical_feature_columns=categorical_feature_columns,
                    early_stopping_rounds=early_stopping_rounds,
                )
                actual_device_type = "cpu"
            else:
                raise PredictionExecutionError(f"LightGBM training failed: {exc}") from exc

        y_pred = None
        y_proba = None
        if X_test is not None and y_test is not None:
            try:
                if task_is_classification:
                    y_pred = pd.Series(regressor.predict(X_test), index=X_test.index)
                    y_proba = regressor.predict_proba(X_test) if hasattr(regressor, "predict_proba") else None
                else:
                    y_pred = pd.Series(regressor.predict(X_test), index=X_test.index, dtype=float)
            except Exception as exc:
                raise PredictionExecutionError(f"LightGBM training failed: {exc}") from exc

        if y_pred is not None and y_test is not None and task_is_classification:
            proba_frame = pd.DataFrame(y_proba) if y_proba is not None else None
            positive_scores = (
                pd.Series(proba_frame.iloc[:, 1]).reset_index(drop=True)
                if proba_frame is not None and len(class_labels) == 2 and proba_frame.shape[1] >= 2
                else None
            )
            metrics = {
                "test": compute_classification_metrics(
                    decode_classification_labels(y_test.reset_index(drop=True), class_labels),
                    decode_classification_labels(y_pred.reset_index(drop=True), class_labels),
                    class_labels=class_labels,
                    positive_scores=positive_scores,
                    target_column=target_column,
                )
            }
        else:
            metrics = {"test": compute_regression_metrics(y_test, y_pred, target_column=target_column)} if y_pred is not None else {}
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        model_dir = output_path / "model_0"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_payload = {
            "backend_name": self.backend_name,
            "task_type": task.task_type,
            "target_columns": [target_column],
            "feature_columns": feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "categorical_mappings": categorical_mappings,
            "categorical_unknown_policy": "map_to_missing",
            "model": regressor,
            "train_csv": train_csv,
            "trained_at": started_at.isoformat(),
            "actual_device_type": actual_device_type,
        }
        if task_is_classification:
            model_payload.update(
                {
                    "task_kind": classification_task_kind(task.task_type, len(class_labels)),
                    "class_labels": [json_safe_label(label) for label in class_labels],
                    "class_count": len(class_labels),
                    "label_mapping": class_mapping,
                    "positive_class_label": json_safe_label(class_labels[1]) if len(class_labels) == 2 else None,
                }
            )

        model_path = model_dir / "best.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model_payload, fh)

        test_predictions_path = model_dir / "test_predictions.csv"
        if y_pred is not None and y_test is not None:
            if task_is_classification:
                predictions_df = self._classification_output_frame(
                    predictions=y_pred.reset_index(drop=True),
                    probabilities=y_proba,
                    class_labels=class_labels,
                    target_column=target_column,
                    source=working.iloc[test_idx].reset_index(drop=True),
                )
                predictions_df[f"{target_column}_true"] = decode_classification_labels(
                    y_test.reset_index(drop=True),
                    class_labels,
                )
                predictions_df["y_true"] = predictions_df[f"{target_column}_true"]
                predictions_df["y_pred"] = predictions_df["prediction"]
            else:
                predictions_df = pd.DataFrame(
                    {
                        target_column: y_pred.reset_index(drop=True),
                        "prediction": y_pred.reset_index(drop=True),
                        "y_true": y_test.reset_index(drop=True),
                        "y_pred": y_pred.reset_index(drop=True),
                    }
                )
            with S3.open(str(test_predictions_path), "w") as fh:
                predictions_df.to_csv(fh, index=False)
        else:
            test_predictions_path = None

        splits_path = output_path / "splits.json"
        splits_path.write_text(json.dumps(effective_split_payload, indent=2) + "\n")

        config_path = output_path / "config.toml"
        config_path.write_text(
            "\n".join(
                [
                    'backend_name = "lightgbm"',
                    f'task_type = "{task.task_type}"',
                    f'target_column = "{target_column}"',
                    f'split_type = "{split_type}"',
                    f'validation_protocol = "{validation_protocol}"',
                    f'device_type = "{actual_device_type}"',
                ]
            )
            + "\n"
        )

        completed_at = project_now()
        return {
            "model_path": str(model_path),
            "best_model_path": str(model_path),
            "test_predictions_path": str(test_predictions_path) if test_predictions_path else None,
            "splits_path": str(splits_path),
            "config_path": str(config_path),
            "task_type": task.task_type,
            "metrics": metrics,
            "metrics_status": "not_evaluated" if final_refit else "evaluated",
            "evaluation_required": bool(final_refit),
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "categorical_feature_columns": categorical_feature_columns,
            "categorical_mappings": categorical_mappings,
            "target_column": target_column,
            "task_kind": classification_task_kind(task.task_type, len(class_labels)) if task_is_classification else "regression",
            "class_labels": [json_safe_label(label) for label in class_labels],
            "class_count": len(class_labels) if task_is_classification else None,
            "label_mapping": class_mapping,
            "positive_class_label": json_safe_label(class_labels[1]) if task_is_classification and len(class_labels) == 2 else None,
            "split_payload": split_payload,
            "effective_split_payload": effective_split_payload,
            "split_metadata": split_indices.get("metadata") or {},
            "has_validation_split": bool(val_idx),
            "excluded_train_indices": sorted(excluded_train_indices),
            "source_train_count": int(len(source_train_idx)),
            "effective_train_count": int(len(train_idx)),
            "validation_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
            "final_refit": final_refit,
            "removed_from_train_count": int(len(excluded_from_train)),
            "requested_exclusion_count": int(len(excluded_train_indices)),
            "activity_cliff_variant_id": activity_cliff_variant_id,
            "split_type": split_type,
            "validation_protocol": validation_protocol,
            "random_state": random_state,
            "compute_environment": compute_env,
            "effective_train_args": {
                **sanitized_args,
                "device_type": actual_device_type,
                "requested_device_type": requested_device_type,
                "early_stopping_rounds": early_stopping_rounds,
                "early_stopping_used": bool(val_idx and early_stopping_rounds > 0),
            },
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "fit_error": str(fit_error) if fit_error and actual_device_type == "cpu" else None,
        }
