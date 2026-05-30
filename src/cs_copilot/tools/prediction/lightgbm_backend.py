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

import numpy as np
import pandas as pd

from cs_copilot.storage import S3
from cs_copilot.tools.activity_cliffs import ACTIVITY_CLIFF_ANNOTATION_PREFIX

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import enrich_backend_environment
from .qsar_training_policy import describe_compute_environment, project_now
from .tabular_splitters import build_tabular_split_payload

logger = logging.getLogger(__name__)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _coerce_split_sizes(split_sizes: Optional[List[float]]) -> List[float]:
    if not split_sizes:
        return [0.8, 0.1, 0.1]
    if len(split_sizes) != 3:
        raise InvalidPredictionInputError(
            "split_sizes must contain exactly 3 values: train, val, test."
        )
    total = float(sum(split_sizes))
    if total <= 0:
        raise InvalidPredictionInputError("split_sizes must sum to a positive value.")
    normalized = [float(value) / total for value in split_sizes]
    if any(value <= 0 for value in normalized):
        raise InvalidPredictionInputError("split_sizes must all be positive.")
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


CLASSIFICATION_TASK_TYPES = {
    "classification",
    "binary_classification",
    "multiclass",
    "multiclass_classification",
}


def _is_classification_task(task_type: str) -> bool:
    return str(task_type or "").strip().lower() in CLASSIFICATION_TASK_TYPES


def _safe_class_token(value: Any, fallback: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")
    return token or fallback


def _json_safe_label(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


class LightGBMBackend(PredictionBackend):
    """Prediction backend built around LightGBM regressors and classifiers."""

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
            "class_weight",
            "is_unbalance",
            "scale_pos_weight",
            "positive_class_label",
            "classification_threshold",
            "force_col_wise",
            "force_row_wise",
            "zero_as_missing",
            "use_missing",
            "deterministic",
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
                raise InvalidPredictionInputError(
                    f"Requested feature columns are missing: {missing}"
                )
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
            feature_columns=numeric_columns
            + list(extra_args.get("categorical_feature_columns") or []),
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

    @staticmethod
    def _label_key(value: Any) -> str:
        return json.dumps(_json_safe_label(value), sort_keys=True, default=str)

    @staticmethod
    def _labels_match(left: Any, right: Any) -> bool:
        return left == right or str(left).strip().lower() == str(right).strip().lower()

    def _sort_class_labels(self, labels: List[Any]) -> List[Any]:
        try:
            return sorted(labels)
        except TypeError:
            return sorted(labels, key=lambda value: str(value).lower())

    def _resolve_class_labels(
        self,
        series: pd.Series,
        extra_args: Dict[str, Any],
    ) -> List[Any]:
        labels: List[Any] = []
        seen: set[str] = set()
        for raw_value in series.tolist():
            normalized = _normalize_category_value(raw_value)
            if normalized is None:
                continue
            normalized = _json_safe_label(normalized)
            key = self._label_key(normalized)
            if key in seen:
                continue
            seen.add(key)
            labels.append(normalized)

        if len(labels) < 2:
            raise InvalidPredictionInputError(
                "LightGBM classification requires at least two target classes after cleanup."
            )

        explicit_positive = extra_args.get("positive_class_label")
        if len(labels) == 2 and explicit_positive is not None:
            positive_matches = [
                label for label in labels if self._labels_match(label, explicit_positive)
            ]
            if not positive_matches:
                raise InvalidPredictionInputError(
                    f"positive_class_label={explicit_positive!r} was not found in target classes {labels}."
                )
            positive = positive_matches[0]
            negative = next(label for label in labels if label != positive)
            return [negative, positive]

        if len(labels) == 2:
            positive_tokens = {"1", "true", "active", "actif", "positive", "pos", "yes", "y"}
            negative_tokens = {"0", "false", "inactive", "inactif", "negative", "neg", "no", "n"}
            positives = [label for label in labels if str(label).strip().lower() in positive_tokens]
            negatives = [label for label in labels if str(label).strip().lower() in negative_tokens]
            if len(positives) == 1 and len(negatives) == 1:
                return [negatives[0], positives[0]]

        return self._sort_class_labels(labels)

    def _encode_classification_target(
        self,
        series: pd.Series,
        extra_args: Dict[str, Any],
    ) -> Tuple[pd.Series, List[Any], Dict[str, int]]:
        if series.isna().any():
            raise InvalidPredictionInputError(
                "LightGBM classification target contains missing values."
            )
        class_labels = self._resolve_class_labels(series, extra_args)
        class_mapping = {self._label_key(label): index for index, label in enumerate(class_labels)}
        encoded_values: List[int] = []
        for raw_value in series.tolist():
            normalized = _normalize_category_value(raw_value)
            key = self._label_key(normalized)
            if key not in class_mapping:
                raise InvalidPredictionInputError(
                    f"Could not encode classification target value {raw_value!r}."
                )
            encoded_values.append(class_mapping[key])
        encoded = pd.Series(encoded_values, index=series.index, dtype="int64")
        return (
            encoded,
            class_labels,
            {str(_json_safe_label(label)): index for index, label in enumerate(class_labels)},
        )

    @staticmethod
    def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
        numeric = [value for value in values if value is not None]
        if not numeric:
            return None
        return float(sum(numeric) / len(numeric))

    @staticmethod
    def _binary_roc_auc(y_true: pd.Series, scores: pd.Series) -> Optional[float]:
        positive = y_true.astype(int) == 1
        n_pos = int(positive.sum())
        n_neg = int((~positive).sum())
        if n_pos == 0 or n_neg == 0:
            return None
        ranks = scores.astype(float).rank(method="average")
        rank_sum_pos = float(ranks[positive].sum())
        auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / float(n_pos * n_neg)
        return float(auc)

    def _compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        y_proba: Optional[Any],
        class_labels: List[Any],
    ) -> Dict[str, Any]:
        true_codes = pd.Series(y_true).astype(int).reset_index(drop=True)
        pred_codes = pd.Series(y_pred).astype(int).reset_index(drop=True)
        classes = list(range(len(class_labels)))
        n = int(len(true_codes))
        accuracy = float((true_codes == pred_codes).mean()) if n else None

        per_class: Dict[str, Dict[str, Any]] = {}
        recalls: List[Optional[float]] = []
        precisions: List[Optional[float]] = []
        f1_values: List[Optional[float]] = []
        for class_index, class_label in enumerate(class_labels):
            true_positive = int(((true_codes == class_index) & (pred_codes == class_index)).sum())
            false_positive = int(((true_codes != class_index) & (pred_codes == class_index)).sum())
            false_negative = int(((true_codes == class_index) & (pred_codes != class_index)).sum())
            true_negative = int(((true_codes != class_index) & (pred_codes != class_index)).sum())
            precision = (
                float(true_positive / (true_positive + false_positive))
                if true_positive + false_positive > 0
                else 0.0
            )
            recall = (
                float(true_positive / (true_positive + false_negative))
                if true_positive + false_negative > 0
                else None
            )
            f1 = (
                float(2.0 * precision * recall / (precision + recall))
                if recall is not None and precision + recall > 0
                else 0.0
            )
            label_token = _safe_class_token(class_label, f"class_{class_index}")
            per_class[label_token] = {
                "class_label": _json_safe_label(class_label),
                "class_index": class_index,
                "support": int((true_codes == class_index).sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
            }
            if recall is not None:
                recalls.append(recall)
            precisions.append(precision)
            f1_values.append(f1)

        confusion_matrix = [
            [
                int(((true_codes == row_class) & (pred_codes == col_class)).sum())
                for col_class in classes
            ]
            for row_class in classes
        ]
        metrics: Dict[str, Any] = {
            "accuracy": accuracy,
            "balanced_accuracy": self._mean_or_none(recalls),
            "precision_macro": self._mean_or_none(precisions),
            "recall_macro": self._mean_or_none(recalls),
            "f1_macro": self._mean_or_none(f1_values),
            "n": n,
            "num_classes": len(class_labels),
            "class_labels": [_json_safe_label(label) for label in class_labels],
            "class_counts": {
                str(_json_safe_label(class_labels[class_index])): int(
                    (true_codes == class_index).sum()
                )
                for class_index in classes
            },
            "confusion_matrix": confusion_matrix,
            "per_class": per_class,
        }

        proba_frame: Optional[pd.DataFrame] = None
        if y_proba is not None:
            probabilities = np.asarray(y_proba, dtype=float)
            if probabilities.ndim == 1 and len(class_labels) == 2:
                probabilities = np.vstack([1.0 - probabilities, probabilities]).T
            if probabilities.ndim == 2 and probabilities.shape[1] >= len(class_labels):
                proba_frame = pd.DataFrame(probabilities[:, : len(class_labels)])
                eps = 1e-15
                true_probabilities = []
                for row_index, true_class in enumerate(true_codes.tolist()):
                    if true_class < proba_frame.shape[1]:
                        true_probabilities.append(float(proba_frame.iloc[row_index, true_class]))
                if true_probabilities:
                    clipped = [min(max(value, eps), 1.0 - eps) for value in true_probabilities]
                    metrics["log_loss"] = float(
                        -sum(math.log(value) for value in clipped) / len(clipped)
                    )

        if len(class_labels) == 2:
            positive_scores = proba_frame.iloc[:, 1] if proba_frame is not None else None
            positive_mask = true_codes == 1
            negative_mask = true_codes == 0
            true_positive = int(((true_codes == 1) & (pred_codes == 1)).sum())
            false_positive = int(((true_codes == 0) & (pred_codes == 1)).sum())
            false_negative = int(((true_codes == 1) & (pred_codes == 0)).sum())
            true_negative = int(((true_codes == 0) & (pred_codes == 0)).sum())
            precision = (
                float(true_positive / (true_positive + false_positive))
                if true_positive + false_positive > 0
                else 0.0
            )
            recall = (
                float(true_positive / (true_positive + false_negative))
                if true_positive + false_negative > 0
                else None
            )
            specificity = (
                float(true_negative / (true_negative + false_positive))
                if true_negative + false_positive > 0
                else None
            )
            metrics.update(
                {
                    "positive_class": _json_safe_label(class_labels[1]),
                    "negative_class": _json_safe_label(class_labels[0]),
                    "precision": precision,
                    "recall": recall,
                    "sensitivity": recall,
                    "specificity": specificity,
                    "f1": (
                        float(2.0 * precision * recall / (precision + recall))
                        if recall is not None and precision + recall > 0
                        else 0.0
                    ),
                    "true_positive": true_positive,
                    "false_positive": false_positive,
                    "false_negative": false_negative,
                    "true_negative": true_negative,
                    "positive_count": int(positive_mask.sum()),
                    "negative_count": int(negative_mask.sum()),
                }
            )
            if positive_scores is not None:
                metrics["roc_auc"] = self._binary_roc_auc(true_codes, positive_scores)
                metrics["brier_score"] = float(
                    ((positive_scores.astype(float) - positive_mask.astype(float)) ** 2).mean()
                )

        return metrics

    def _default_model_params(
        self,
        extra_args: Dict[str, Any],
        *,
        task_kind: str = "regression",
        num_classes: Optional[int] = None,
    ) -> Dict[str, Any]:
        if task_kind == "classification":
            default_objective = "multiclass" if (num_classes or 0) > 2 else "binary"
            default_metric = "multi_logloss" if (num_classes or 0) > 2 else "binary_logloss"
        else:
            default_objective = "regression"
            default_metric = None
        params = {
            "objective": str(extra_args.get("objective") or default_objective),
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
        if default_metric and not extra_args.get("metric"):
            params["metric"] = default_metric
        elif extra_args.get("metric"):
            params["metric"] = extra_args["metric"]
        if task_kind == "classification" and (num_classes or 0) > 2:
            params["num_class"] = int(num_classes or 0)
        if "class_weight" in extra_args:
            params["class_weight"] = extra_args["class_weight"]
        if "is_unbalance" in extra_args:
            params["is_unbalance"] = bool(extra_args["is_unbalance"])
        if "scale_pos_weight" in extra_args:
            params["scale_pos_weight"] = float(extra_args["scale_pos_weight"])
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
        if compute_env.get("gpu_available"):
            return "gpu", True, compute_env
        return "cpu", False, compute_env

    def _fit_regressor(
        self,
        *,
        model_params: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_feature_columns: List[str],
        early_stopping_rounds: int,
    ):
        lgb = self._import_lightgbm()
        regressor = lgb.LGBMRegressor(**model_params)
        callbacks: List[Any] = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )
        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
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
        X_val: pd.DataFrame,
        y_val: pd.Series,
        categorical_feature_columns: List[str],
        early_stopping_rounds: int,
    ):
        lgb = self._import_lightgbm()
        classifier = lgb.LGBMClassifier(**model_params)
        callbacks: List[Any] = [lgb.log_evaluation(period=0)]
        if early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
            )
        fit_kwargs: Dict[str, Any] = {
            "eval_set": [(X_val, y_val)],
            "callbacks": callbacks,
        }
        if categorical_feature_columns:
            fit_kwargs["categorical_feature"] = list(categorical_feature_columns)
        classifier.fit(X_train, y_train, **fit_kwargs)
        return classifier

    def _predict_class_codes(
        self,
        estimator: Any,
        features: pd.DataFrame,
        probabilities: Optional[Any],
        extra_args: Optional[Dict[str, Any]],
        *,
        num_classes: int,
    ) -> pd.Series:
        if probabilities is not None and num_classes == 2:
            threshold = float((extra_args or {}).get("classification_threshold", 0.5))
            probabilities_array = np.asarray(probabilities, dtype=float)
            if probabilities_array.ndim == 2 and probabilities_array.shape[1] >= 2:
                return pd.Series(
                    (probabilities_array[:, 1] >= threshold).astype(int), index=features.index
                )
        return pd.Series(estimator.predict(features), index=features.index).astype(int)

    def _classification_output_frame(
        self,
        *,
        predicted_codes: pd.Series,
        probabilities: Optional[Any],
        class_labels: List[Any],
        target_column: Optional[str] = None,
        true_codes: Optional[pd.Series] = None,
        true_labels: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        predicted_labels = pd.Series(
            [_json_safe_label(class_labels[int(code)]) for code in predicted_codes.tolist()],
            index=predicted_codes.index,
        ).reset_index(drop=True)
        output = pd.DataFrame(
            {
                "prediction": predicted_labels,
                "predicted_class": predicted_labels,
                "predicted_class_index": predicted_codes.reset_index(drop=True).astype(int),
            }
        )
        if target_column:
            output[target_column] = predicted_labels
        if true_codes is not None:
            output["y_true_encoded"] = true_codes.reset_index(drop=True).astype(int)
        if true_labels is not None:
            safe_true = true_labels.map(_json_safe_label).reset_index(drop=True)
            output["y_true"] = safe_true
            if target_column:
                output[f"{target_column}_true"] = safe_true
        output["y_pred"] = predicted_labels
        output["y_pred_encoded"] = output["predicted_class_index"]

        if probabilities is not None:
            probabilities_array = np.asarray(probabilities, dtype=float)
            if probabilities_array.ndim == 1 and len(class_labels) == 2:
                probabilities_array = np.vstack([1.0 - probabilities_array, probabilities_array]).T
            if probabilities_array.ndim == 2:
                used_tokens: set[str] = set()
                for class_index, class_label in enumerate(class_labels):
                    if class_index >= probabilities_array.shape[1]:
                        continue
                    token = _safe_class_token(class_label, f"class_{class_index}")
                    if token in used_tokens:
                        token = f"{token}_{class_index}"
                    used_tokens.add(token)
                    output[f"probability_{token}"] = probabilities_array[:, class_index]
                if len(class_labels) == 2 and probabilities_array.shape[1] >= 2:
                    output["positive_class_probability"] = probabilities_array[:, 1]
        return output

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
            raise InvalidPredictionInputError(
                "LightGBM V1 does not support predictive uncertainty export."
            )

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
        categorical_feature_columns = list((payload or {}).get("categorical_feature_columns") or [])
        category_mappings = dict((payload or {}).get("categorical_mappings") or {})
        target_columns = list(model_record.task.target_columns)
        task_type = str((payload or {}).get("task_type") or model_record.task.task_type)
        task_kind = "classification" if _is_classification_task(task_type) else "regression"

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
        resolved_class_labels: List[Any] = []
        try:
            estimator = payload["model"]
            if task_kind == "classification":
                class_labels = list((payload or {}).get("class_labels") or [])
                if not class_labels:
                    class_labels = [
                        _json_safe_label(label) for label in getattr(estimator, "classes_", [])
                    ]
                if not class_labels:
                    raise InvalidPredictionInputError(
                        "LightGBM classification artifact is missing class_labels metadata."
                    )
                resolved_class_labels = class_labels
                probabilities = (
                    estimator.predict_proba(features)
                    if hasattr(estimator, "predict_proba")
                    else None
                )
                predict_args = {
                    "classification_threshold": (payload or {}).get(
                        "classification_threshold", 0.5
                    ),
                    **(extra_args or {}),
                }
                y_pred = self._predict_class_codes(
                    estimator,
                    features,
                    probabilities,
                    predict_args,
                    num_classes=len(class_labels),
                )
                output = self._classification_output_frame(
                    predicted_codes=y_pred,
                    probabilities=probabilities,
                    class_labels=class_labels,
                    target_column=target_columns[0] if len(target_columns) == 1 else None,
                )
            else:
                y_pred = estimator.predict(features)
                output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
        except Exception as exc:
            raise PredictionExecutionError(f"LightGBM prediction failed: {exc}") from exc

        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)

        result = {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "task_type": task_type,
        }
        if task_kind == "classification":
            result["class_labels"] = [_json_safe_label(label) for label in resolved_class_labels]
            result["prediction_columns"] = list(output.columns)
        return result

    def train_model(
        self,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        *,
        extra_args: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_available()
        normalized_task_type = str(task.task_type or "").strip().lower()
        if normalized_task_type == "regression":
            task_kind = "regression"
        elif _is_classification_task(normalized_task_type):
            task_kind = "classification"
        else:
            raise InvalidPredictionInputError(
                "LightGBM V1 supports task_type='regression' or task_type='classification'."
            )
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError("LightGBM V1 requires exactly one target column.")

        sanitized_args = self._sanitize_train_extra_args(extra_args)
        split_sizes = _coerce_split_sizes(sanitized_args.pop("split_sizes", None))
        split_payload = sanitized_args.pop("split_payload", None)
        excluded_train_indices = {
            int(idx) for idx in (sanitized_args.pop("excluded_train_indices", None) or [])
        }
        activity_cliff_variant_id = sanitized_args.pop("activity_cliff_variant_id", None)
        random_state = int(sanitized_args.get("random_state", 42))
        split_type = str(sanitized_args.get("split_type", "random"))
        validation_protocol = str(sanitized_args.get("validation_protocol", "standard_qsar"))
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
            feature_columns
            + [target_column]
            + [c for c in ("smiles", "Drug_ID") if c in dataset.columns]
        ].copy()
        if len(working) < 10:
            raise InvalidPredictionInputError(
                "LightGBM V1 requires at least 10 rows after target cleanup."
            )

        encoded_target_column = "__lightgbm_target_encoded"
        class_labels: List[Any] = []
        class_mapping: Dict[str, int] = {}
        if task_kind == "classification":
            encoded_target, class_labels, class_mapping = self._encode_classification_target(
                working[target_column],
                sanitized_args,
            )
            working[encoded_target_column] = encoded_target
        else:
            target_numeric = pd.to_numeric(working[target_column], errors="coerce")
            if target_numeric.isna().any():
                raise InvalidPredictionInputError(
                    f"LightGBM regression target '{target_column}' contains non-numeric values."
                )
            working[target_column] = target_numeric.astype(float)

        encoded_features, categorical_mappings = self._encode_categorical_frame(
            working[feature_columns].copy(),
            categorical_feature_columns,
        )
        encoded_working = working.copy()
        encoded_working[feature_columns] = encoded_features

        if not split_payload:
            split_payload = build_tabular_split_payload(
                df=encoded_working,
                split_type=split_type,
                split_sizes=split_sizes,
                random_state=random_state,
                smiles_column=task.smiles_columns[0] if task.smiles_columns else None,
                feature_columns=feature_columns,
                stratify_column=target_column if task_kind == "classification" else None,
            )

        if not split_payload or "train" not in split_payload[0]:
            raise InvalidPredictionInputError(
                "LightGBM split payload must provide non-empty train/val/test indices."
            )

        split_indices = split_payload[0]
        source_train_idx = [int(idx) for idx in (split_indices.get("train") or [])]
        train_idx = [idx for idx in source_train_idx if idx not in excluded_train_indices]
        excluded_from_train = sorted(set(source_train_idx) & excluded_train_indices)
        val_idx = split_indices.get("val") or []
        test_idx = split_indices.get("test") or []
        if not train_idx or not val_idx or not test_idx:
            raise InvalidPredictionInputError(
                "LightGBM split payload must provide non-empty train/val/test indices."
            )
        effective_split_payload = [
            {
                **split_indices,
                "train": train_idx,
                "excluded_from_train": excluded_from_train,
            }
        ]

        X_train = encoded_working.iloc[train_idx][feature_columns].copy()
        X_val = encoded_working.iloc[val_idx][feature_columns].copy()
        X_test = encoded_working.iloc[test_idx][feature_columns].copy()
        if task_kind == "classification":
            y_train = encoded_working.iloc[train_idx][encoded_target_column].astype(int)
            y_val = encoded_working.iloc[val_idx][encoded_target_column].astype(int)
            y_test = encoded_working.iloc[test_idx][encoded_target_column].astype(int)
            present_train_classes = {int(value) for value in y_train.tolist()}
            if len(present_train_classes) < 2:
                raise InvalidPredictionInputError(
                    "LightGBM classification training split contains fewer than two classes."
                )
            missing_train_classes = sorted(set(range(len(class_labels))) - present_train_classes)
            if missing_train_classes:
                missing_labels = [
                    _json_safe_label(class_labels[index]) for index in missing_train_classes
                ]
                raise InvalidPredictionInputError(
                    "LightGBM classification training split is missing target classes "
                    f"{missing_labels}. Use a larger dataset or a stratified/random split."
                )
        else:
            y_train = pd.to_numeric(
                encoded_working.iloc[train_idx][target_column], errors="coerce"
            ).astype(float)
            y_val = pd.to_numeric(
                encoded_working.iloc[val_idx][target_column], errors="coerce"
            ).astype(float)
            y_test = pd.to_numeric(
                encoded_working.iloc[test_idx][target_column], errors="coerce"
            ).astype(float)

        requested_device_type, auto_device, compute_env = self._resolve_device_type(sanitized_args)
        gpu_fallback_to_cpu = bool(sanitized_args.get("gpu_fallback_to_cpu", True))
        early_stopping_rounds = int(sanitized_args.get("early_stopping_rounds", 50))
        model_params = self._default_model_params(
            sanitized_args,
            task_kind=task_kind,
            num_classes=len(class_labels) if task_kind == "classification" else None,
        )
        model_params["device_type"] = requested_device_type
        fit_error: Optional[Exception] = None

        def fit_once(params: Dict[str, Any]):
            if task_kind == "classification":
                return self._fit_classifier(
                    model_params=params,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    categorical_feature_columns=categorical_feature_columns,
                    early_stopping_rounds=early_stopping_rounds,
                )
            return self._fit_regressor(
                model_params=params,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                categorical_feature_columns=categorical_feature_columns,
                early_stopping_rounds=early_stopping_rounds,
            )

        try:
            estimator = fit_once(model_params)
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
                estimator = fit_once(model_params)
                actual_device_type = "cpu"
            else:
                raise PredictionExecutionError(f"LightGBM training failed: {exc}") from exc

        try:
            if task_kind == "classification":
                probabilities = (
                    estimator.predict_proba(X_test) if hasattr(estimator, "predict_proba") else None
                )
                y_pred = self._predict_class_codes(
                    estimator,
                    X_test,
                    probabilities,
                    sanitized_args,
                    num_classes=len(class_labels),
                )
                metrics = {
                    "test": self._compute_classification_metrics(
                        y_test,
                        y_pred,
                        probabilities,
                        class_labels,
                    )
                }
            else:
                probabilities = None
                y_pred = pd.Series(estimator.predict(X_test), index=X_test.index, dtype=float)
                metrics = {"test": self._compute_regression_metrics(y_test, y_pred)}
        except Exception as exc:
            raise PredictionExecutionError(f"LightGBM training failed: {exc}") from exc

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        model_dir = output_path / "model_0"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_payload = {
            "backend_name": self.backend_name,
            "task_type": task.task_type,
            "task_kind": task_kind,
            "target_columns": [target_column],
            "feature_columns": feature_columns,
            "categorical_feature_columns": categorical_feature_columns,
            "categorical_mappings": categorical_mappings,
            "categorical_unknown_policy": "map_to_missing",
            "model": estimator,
            "train_csv": train_csv,
            "trained_at": started_at.isoformat(),
            "actual_device_type": actual_device_type,
        }
        if task_kind == "classification":
            model_payload.update(
                {
                    "class_labels": [_json_safe_label(label) for label in class_labels],
                    "class_mapping": class_mapping,
                    "positive_class_label": (
                        _json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
                    ),
                    "classification_threshold": float(
                        sanitized_args.get("classification_threshold", 0.5)
                    ),
                }
            )

        model_path = model_dir / "best.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model_payload, fh)

        test_predictions_path = model_dir / "test_predictions.csv"
        if task_kind == "classification":
            true_labels = working.iloc[test_idx][target_column].reset_index(drop=True)
            predictions_df = self._classification_output_frame(
                predicted_codes=y_pred,
                probabilities=probabilities,
                class_labels=class_labels,
                target_column=target_column,
                true_codes=y_test.reset_index(drop=True),
                true_labels=true_labels,
            )
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
        result = {
            "model_path": str(model_path),
            "best_model_path": str(model_path),
            "test_predictions_path": str(test_predictions_path),
            "splits_path": str(splits_path),
            "config_path": str(config_path),
            "metrics": metrics,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "categorical_feature_columns": categorical_feature_columns,
            "categorical_mappings": categorical_mappings,
            "target_column": target_column,
            "task_type": task.task_type,
            "split_payload": split_payload,
            "effective_split_payload": effective_split_payload,
            "excluded_train_indices": sorted(excluded_train_indices),
            "source_train_count": int(len(source_train_idx)),
            "effective_train_count": int(len(train_idx)),
            "validation_count": int(len(val_idx)),
            "test_count": int(len(test_idx)),
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
            },
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "fit_error": str(fit_error) if fit_error and actual_device_type == "cpu" else None,
        }
        if task_kind == "classification":
            result.update(
                {
                    "class_labels": [_json_safe_label(label) for label in class_labels],
                    "class_count": len(class_labels),
                    "class_mapping": class_mapping,
                    "positive_class_label": (
                        _json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
                    ),
                    "classification_threshold": float(
                        sanitized_args.get("classification_threshold", 0.5)
                    ),
                }
            )
        return result
