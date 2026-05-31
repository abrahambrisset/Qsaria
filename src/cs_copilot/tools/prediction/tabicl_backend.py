#!/usr/bin/env python
# coding: utf-8
"""
TabICLv2 backend adapter for tabular QSAR workflows.
"""

from __future__ import annotations

import ctypes
import gc
import importlib.metadata
import importlib.util
import json
import logging
import math
import pickle
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from .qsar_training_policy import project_now
from .tabular_splitters import build_tabular_split_payload

logger = logging.getLogger(__name__)

DEFAULT_TABICL_CHECKPOINT_DIR = Path("data/model_assets/checkpoints/tabicl").resolve()
DEFAULT_TABICL_REGRESSOR_CHECKPOINT = "tabicl-regressor-v2-20260212.ckpt"
DEFAULT_TABICL_CLASSIFIER_CHECKPOINT = "tabicl-classifier-v2-20260212.ckpt"


def _release_process_memory() -> None:
    """Best-effort CPU/GPU memory cleanup after a heavy TabICL run."""
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass

    # On glibc-based Linux systems, this can return free heap pages to the OS.
    try:
        libc = ctypes.CDLL("libc.so.6")
        if hasattr(libc, "malloc_trim"):
            libc.malloc_trim(0)
    except Exception:
        pass


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


CLASSIFICATION_TASK_TYPES = {
    "classification",
    "binary_classification",
    "multiclass",
    "multiclass_classification",
}


def _is_classification_task(task_type: str) -> bool:
    return str(task_type or "").strip().lower() in CLASSIFICATION_TASK_TYPES


def _json_safe_label(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _normalize_classification_label(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return _json_safe_label(value)


def _label_key(value: Any) -> str:
    return json.dumps(_json_safe_label(value), sort_keys=True, default=str)


def _safe_class_token(value: Any, fallback: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")
    return token or fallback


def _sort_class_labels(labels: List[Any]) -> List[Any]:
    try:
        return sorted(labels)
    except TypeError:
        return sorted(labels, key=lambda value: str(value).lower())


def _resolve_class_labels(series: pd.Series) -> List[Any]:
    labels: List[Any] = []
    seen: set[str] = set()
    for raw_value in series.tolist():
        normalized = _normalize_classification_label(raw_value)
        if normalized is None:
            continue
        key = _label_key(normalized)
        if key in seen:
            continue
        seen.add(key)
        labels.append(normalized)
    if len(labels) < 2:
        raise InvalidPredictionInputError(
            "TabICL classification requires at least two target classes after cleanup."
        )
    if len(labels) == 2:
        positive_tokens = {"1", "true", "active", "actif", "positive", "pos", "yes", "y"}
        negative_tokens = {"0", "false", "inactive", "inactif", "negative", "neg", "no", "n"}
        positives = [label for label in labels if str(label).strip().lower() in positive_tokens]
        negatives = [label for label in labels if str(label).strip().lower() in negative_tokens]
        if len(positives) == 1 and len(negatives) == 1:
            return [negatives[0], positives[0]]
    return _sort_class_labels(labels)


def _encode_classification_labels(series: pd.Series, class_labels: List[Any]) -> pd.Series:
    mapping = {_label_key(label): index for index, label in enumerate(class_labels)}
    encoded: List[Optional[int]] = []
    for raw_value in series.tolist():
        normalized = _normalize_classification_label(raw_value)
        if normalized is None:
            encoded.append(None)
            continue
        encoded.append(mapping.get(_label_key(normalized)))
    return pd.Series(encoded, index=series.index, dtype="object")


def _labels_from_codes(codes: pd.Series, class_labels: List[Any]) -> pd.Series:
    labels: List[Any] = []
    for raw_code in codes.tolist():
        if raw_code is None or pd.isna(raw_code):
            labels.append(None)
            continue
        code = int(raw_code)
        labels.append(
            _json_safe_label(class_labels[code]) if 0 <= code < len(class_labels) else None
        )
    return pd.Series(labels, index=codes.index, dtype="object")


def _mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


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


class TabICLBackend(PredictionBackend):
    """Prediction backend built around TabICLv2 regressors and classifiers."""

    backend_name = "tabicl"
    MODEL_EXTENSIONS = (".pkl",)

    def _package_version(self) -> Optional[str]:
        try:
            return importlib.metadata.version("tabicl")
        except importlib.metadata.PackageNotFoundError:
            return None

    def is_available(self) -> bool:
        return importlib.util.find_spec("tabicl") is not None

    def describe_environment(self) -> Dict[str, Any]:
        checkpoint_path = DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        classifier_checkpoint_path = (
            DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_CLASSIFIER_CHECKPOINT
        )
        return enrich_backend_environment(
            self.backend_name,
            {
                "backend_name": self.backend_name,
                "available": self.is_available(),
                "package_version": self._package_version(),
                "checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
                "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "default_classifier_checkpoint_version": DEFAULT_TABICL_CLASSIFIER_CHECKPOINT,
                "default_checkpoint_path": str(checkpoint_path),
                "default_classifier_checkpoint_path": str(classifier_checkpoint_path),
                "default_checkpoint_present": checkpoint_path.exists(),
                "default_classifier_checkpoint_present": classifier_checkpoint_path.exists(),
            },
        )

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Model path does not exist: {model_path}")
        if path.suffix not in self.MODEL_EXTENSIONS:
            raise InvalidPredictionInputError(
                f"TabICL model artifact must end with one of {self.MODEL_EXTENSIONS}: {model_path}"
            )
        return path.resolve()

    def _ensure_available(self) -> None:
        if not self.is_available():
            raise BackendNotAvailableError(
                "TabICL backend is not available. Install the optional `tabicl` dependency first. "
                f"Environment snapshot: {self.describe_environment()}"
            )

    def _resolve_checkpoint_config(
        self,
        extra_args: Optional[Dict[str, Any]],
        *,
        task_kind: str = "regression",
    ) -> Dict[str, Any]:
        extra_args = dict(extra_args or {})
        default_checkpoint = (
            DEFAULT_TABICL_CLASSIFIER_CHECKPOINT
            if task_kind == "classification"
            else DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        )
        checkpoint_version = str(extra_args.get("checkpoint_version") or default_checkpoint)
        checkpoint_dir = Path(
            extra_args.get("checkpoint_dir") or DEFAULT_TABICL_CHECKPOINT_DIR
        ).expanduser()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_version
        allow_auto_download = bool(extra_args.get("allow_auto_download", False))
        return {
            "checkpoint_version": checkpoint_version,
            "checkpoint_dir": checkpoint_dir.resolve(),
            "checkpoint_path": checkpoint_path.resolve(),
            "allow_auto_download": allow_auto_download,
        }

    def _import_tabicl_regressor(self):
        self._ensure_available()
        from tabicl import TabICLRegressor

        return TabICLRegressor

    def _import_tabicl_classifier(self):
        self._ensure_available()
        from tabicl import TabICLClassifier

        return TabICLClassifier

    def _sanitize_train_extra_args(self, extra_args: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        raw = dict(extra_args or {})
        allowed = {
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "outlier_threshold",
            "batch_size",
            "kv_cache",
            "model_path",
            "allow_auto_download",
            "checkpoint_version",
            "device",
            "use_amp",
            "use_fa3",
            "offload_mode",
            "disk_offload_dir",
            "random_state",
            "n_jobs",
            "verbose",
            "inference_config",
            "checkpoint_dir",
            "save_model_weights",
            "save_training_data",
            "save_kv_cache",
            "split_sizes",
            "split_type",
            "validation_protocol",
            "feature_columns",
            "split_payload",
            "heartbeat_seconds",
            "heartbeat_path",
            "heartbeat_label",
            "heartbeat_run_index",
            "heartbeat_total_runs",
            "class_shuffle_method",
            "softmax_temperature",
            "average_logits",
            "support_many_classes",
            "classification_threshold",
        }
        dropped = sorted(key for key in raw if key not in allowed)
        sanitized = {key: value for key, value in raw.items() if key in allowed}
        if dropped:
            logger.warning(
                "Dropping unsupported TabICL train args for this V1 backend: %s",
                ", ".join(dropped),
            )
        return sanitized

    def _select_feature_columns(
        self,
        df: pd.DataFrame,
        target_columns: List[str],
        extra_args: Dict[str, Any],
    ) -> List[str]:
        excluded = set(target_columns)
        numeric_columns = [
            column
            for column in df.columns
            if column not in excluded
            and not str(column).startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)
            and pd.api.types.is_numeric_dtype(df[column])
        ]

        explicit = extra_args.get("feature_columns")
        if explicit:
            if isinstance(explicit, str):
                explicit = [explicit]
            missing = [column for column in explicit if column not in df.columns]
            if missing:
                raise InvalidPredictionInputError(
                    f"Requested feature columns are missing: {missing}"
                )
            leaked = [
                column
                for column in explicit
                if str(column).startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)
            ]
            if leaked:
                raise InvalidPredictionInputError(
                    "Activity-cliff annotation columns cannot be used as TabICL features. "
                    f"Invalid columns: {leaked}"
                )
            non_numeric = [
                column for column in explicit if not pd.api.types.is_numeric_dtype(df[column])
            ]
            if non_numeric:
                if set(non_numeric) == set(explicit) and numeric_columns:
                    logger.warning(
                        "Ignoring non-numeric explicit TabICL feature columns %s and falling back to numeric columns.",
                        non_numeric,
                    )
                else:
                    raise InvalidPredictionInputError(
                        "TabICL feature_columns must be numeric columns only. "
                        f"Non-numeric columns received: {non_numeric}"
                    )
            else:
                return list(explicit)

        if not numeric_columns:
            raise InvalidPredictionInputError(
                "No numeric feature columns were found for TabICL. "
                "Provide a tabular dataset with numeric feature columns or pass feature_columns explicitly."
            )
        return numeric_columns

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

    def _compute_classification_metrics(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        class_labels: List[Any],
        positive_scores: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        true_codes = pd.Series(y_true).reset_index(drop=True)
        pred_codes = pd.Series(y_pred).reset_index(drop=True)
        valid_mask = true_codes.notna() & pred_codes.notna()
        true_codes = true_codes[valid_mask].astype(int).reset_index(drop=True)
        pred_codes = pred_codes[valid_mask].astype(int).reset_index(drop=True)
        scores = None
        if positive_scores is not None:
            scores = pd.to_numeric(
                pd.Series(positive_scores).reset_index(drop=True)[valid_mask], errors="coerce"
            )
        classes = list(range(len(class_labels)))
        n = int(len(true_codes))
        accuracy = float((true_codes == pred_codes).mean()) if n else None
        recalls: List[Optional[float]] = []
        precisions: List[Optional[float]] = []
        f1_values: List[Optional[float]] = []
        per_class: Dict[str, Dict[str, Any]] = {}
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
            if recall is not None:
                recalls.append(recall)
            precisions.append(precision)
            f1_values.append(f1)
            token = _safe_class_token(class_label, f"class_{class_index}")
            per_class[token] = {
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
        confusion_matrix = [
            [
                int(((true_codes == row_class) & (pred_codes == col_class)).sum())
                for col_class in classes
            ]
            for row_class in classes
        ]
        metrics: Dict[str, Any] = {
            "accuracy": accuracy,
            "balanced_accuracy": _mean_or_none(recalls),
            "precision_macro": _mean_or_none(precisions),
            "recall_macro": _mean_or_none(recalls),
            "f1_macro": _mean_or_none(f1_values),
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
        if len(class_labels) == 2:
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
            if scores is not None and scores.notna().any():
                valid_scores = scores.notna()
                aligned_true = true_codes[valid_scores].reset_index(drop=True)
                aligned_scores = scores[valid_scores].astype(float).reset_index(drop=True)
                metrics["roc_auc"] = _binary_roc_auc(aligned_true, aligned_scores)
                metrics["brier_score"] = float(
                    ((aligned_scores - (aligned_true == 1).astype(float)) ** 2).mean()
                )
        return metrics

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
        predicted_labels = _labels_from_codes(predicted_codes.reset_index(drop=True), class_labels)
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

    def _persist_checkpoint_if_possible(self, estimator: Any, destination: Path) -> Optional[str]:
        if destination.exists():
            return str(destination)

        candidates = [
            getattr(estimator, "model_path", None),
            getattr(estimator, "checkpoint_path", None),
            getattr(estimator, "checkpoint_file", None),
            getattr(estimator, "checkpoint", None),
            getattr(estimator, "_model_path", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            candidate_path = Path(str(candidate)).expanduser()
            if candidate_path.exists() and candidate_path.is_file():
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(candidate_path, destination)
                    return str(destination)
                except Exception as exc:
                    logger.warning(
                        "Could not persist TabICL checkpoint to %s: %s", destination, exc
                    )
                    return None
        return None

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
                "TabICL V1 does not support predictive uncertainty export."
            )

        model_path = self.validate_model_path(model_record.model_path)
        try:
            with model_path.open("rb") as fh:
                payload = pickle.load(fh)
        except Exception as exc:
            raise PredictionExecutionError(
                f"Could not load TabICL model artifact {model_path}: {exc}"
            ) from exc

        estimator = payload.get("model") if isinstance(payload, dict) else payload
        artifact_metadata = dict(payload.get("metadata") or {}) if isinstance(payload, dict) else {}
        metadata_path = model_path.with_suffix(".metadata.json")
        if metadata_path.exists():
            try:
                artifact_metadata.update(json.loads(metadata_path.read_text()))
            except Exception:
                pass

        with S3.open(input_csv, "r") as fh:
            df = _strip_unnamed_columns(pd.read_csv(fh))

        target_columns = list(model_record.task.target_columns)
        task_type = str(artifact_metadata.get("task_type") or model_record.task.task_type)
        task_kind = "classification" if _is_classification_task(task_type) else "regression"
        feature_columns = list(
            (model_record.inference_profile or {}).get("feature_columns")
            or artifact_metadata.get("feature_columns")
            or []
        )
        if not feature_columns:
            feature_columns = self._select_feature_columns(df, target_columns, {})

        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise InvalidPredictionInputError(
                f"Prediction input is missing feature columns: {missing_features}"
            )

        X = df[feature_columns].copy()
        try:
            if task_kind == "classification":
                class_labels = list(artifact_metadata.get("class_labels") or [])
                if not class_labels:
                    class_labels = [
                        _json_safe_label(label) for label in getattr(estimator, "classes_", [])
                    ]
                if not class_labels:
                    raise InvalidPredictionInputError(
                        "TabICL classification artifact is missing class_labels metadata."
                    )
                probabilities = (
                    estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None
                )
                predicted_codes = self._predict_class_codes(
                    estimator,
                    X,
                    probabilities,
                    {
                        "classification_threshold": artifact_metadata.get(
                            "classification_threshold", 0.5
                        ),
                        **(extra_args or {}),
                    },
                    num_classes=len(class_labels),
                )
                output = self._classification_output_frame(
                    predicted_codes=predicted_codes,
                    probabilities=probabilities,
                    class_labels=class_labels,
                    target_column=target_columns[0] if len(target_columns) == 1 else None,
                )
            else:
                y_pred = estimator.predict(X)
                output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL prediction failed: {exc}") from exc

        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)

        result = {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
            "task_type": task_type,
        }
        if task_kind == "classification":
            result["class_labels"] = [_json_safe_label(label) for label in class_labels]
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
                "TabICL V1 supports task_type='regression' or task_type='classification'."
            )
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError("TabICL V1 requires exactly one target column.")

        sanitized_args = self._sanitize_train_extra_args(extra_args)
        checkpoint_cfg = self._resolve_checkpoint_config(sanitized_args, task_kind=task_kind)
        if not checkpoint_cfg["checkpoint_path"].exists():
            raise InvalidPredictionInputError(
                "TabICL checkpoint not found at the expected persistent path: "
                f"{checkpoint_cfg['checkpoint_path']}. Provision this checkpoint before training."
            )
        split_sizes = _coerce_split_sizes(sanitized_args.pop("split_sizes", None))
        split_payload = sanitized_args.pop("split_payload", None)
        random_state = int(sanitized_args.get("random_state", 42))
        split_type = str(sanitized_args.get("split_type", "random"))
        validation_protocol = str(sanitized_args.get("validation_protocol", "standard_qsar"))
        started_at = project_now()

        with S3.open(train_csv, "r") as fh:
            dataset = _strip_unnamed_columns(pd.read_csv(fh))

        target_column = task.target_columns[0]
        if target_column not in dataset.columns:
            raise InvalidPredictionInputError(f"Missing target column: {target_column}")

        feature_columns = self._select_feature_columns(dataset, [target_column], sanitized_args)
        working = dataset[
            feature_columns
            + [target_column]
            + [c for c in ("smiles", "Drug_ID") if c in dataset.columns]
        ].copy()
        class_labels: List[Any] = []
        encoded_target_column = "__tabicl_target_encoded"
        if task_kind == "classification":
            normalized_labels = working[target_column].map(_normalize_classification_label)
            missing_mask = normalized_labels.isna()
            working = working.loc[~missing_mask].copy()
            working[target_column] = normalized_labels.loc[working.index]
            class_labels = _resolve_class_labels(working[target_column])
            working[encoded_target_column] = _encode_classification_labels(
                working[target_column], class_labels
            ).astype(int)
        else:
            working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
            working = working.dropna(subset=[target_column]).reset_index(drop=True)
        working = working.reset_index(drop=True)
        if len(working) < 10:
            raise InvalidPredictionInputError(
                "TabICL V1 requires at least 10 rows after target cleanup."
            )

        if split_payload is None:
            split_payload = build_tabular_split_payload(
                df=working,
                split_type=split_type,
                split_sizes=split_sizes,
                random_state=random_state,
                smiles_column="smiles" if "smiles" in working.columns else None,
                feature_columns=feature_columns,
                stratify_column=target_column if task_kind == "classification" else None,
            )
        if (
            not isinstance(split_payload, list)
            or not split_payload
            or not isinstance(split_payload[0], dict)
        ):
            raise InvalidPredictionInputError(
                "TabICL split_payload must be a non-empty list with train/val/test index mappings."
            )

        split_map = split_payload[0]
        train_indices = [int(idx) for idx in (split_map.get("train") or [])]
        val_indices = [int(idx) for idx in (split_map.get("val") or [])]
        test_indices = [int(idx) for idx in (split_map.get("test") or [])]
        if not train_indices or not val_indices or not test_indices:
            raise InvalidPredictionInputError(
                "TabICL split payload must provide non-empty train/val/test indices."
            )
        train_df = working.iloc[train_indices].reset_index(drop=True)
        val_df = working.iloc[val_indices].reset_index(drop=True)
        test_df = working.iloc[test_indices].reset_index(drop=True)

        X_train = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
        if task_kind == "classification":
            y_train = train_df[encoded_target_column].astype(int).copy()
            y_test = test_df[encoded_target_column].astype(int).copy()
            present_train_classes = {int(value) for value in y_train.tolist()}
            if len(present_train_classes) < 2:
                raise InvalidPredictionInputError(
                    "TabICL classification training split contains fewer than two classes."
                )
            missing_train_classes = sorted(set(range(len(class_labels))) - present_train_classes)
            if missing_train_classes:
                missing_labels = [
                    _json_safe_label(class_labels[index]) for index in missing_train_classes
                ]
                raise InvalidPredictionInputError(
                    "TabICL classification training split is missing target classes "
                    f"{missing_labels}. Use a larger dataset or a stratified/random split."
                )
        else:
            y_train = train_df[target_column].astype(float).copy()
            y_test = test_df[target_column].astype(float).copy()

        EstimatorClass = (
            self._import_tabicl_classifier()
            if task_kind == "classification"
            else self._import_tabicl_regressor()
        )
        model_path_arg = str(checkpoint_cfg["checkpoint_path"])
        init_kwargs = {
            "model_path": model_path_arg,
            "allow_auto_download": checkpoint_cfg["allow_auto_download"],
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "random_state": random_state,
            "verbose": bool(sanitized_args.get("verbose", False)),
        }
        optional_keys = (
            "n_estimators",
            "norm_methods",
            "feat_shuffle_method",
            "outlier_threshold",
            "batch_size",
            "kv_cache",
            "device",
            "use_amp",
            "use_fa3",
            "offload_mode",
            "disk_offload_dir",
            "n_jobs",
            "inference_config",
        )
        if task_kind == "classification":
            optional_keys = optional_keys + (
                "class_shuffle_method",
                "softmax_temperature",
                "average_logits",
                "support_many_classes",
            )
        for key in optional_keys:
            if key in sanitized_args:
                init_kwargs[key] = sanitized_args[key]

        estimator = EstimatorClass(**init_kwargs)
        heartbeat_train_rows = int(len(train_df))
        heartbeat_val_rows = int(len(val_df))
        heartbeat_test_rows = int(len(test_df))
        heartbeat_seconds = float(sanitized_args.get("heartbeat_seconds", 120.0))
        heartbeat_path_raw = sanitized_args.get("heartbeat_path")
        heartbeat_label = str(sanitized_args.get("heartbeat_label") or split_type)
        heartbeat_run_index = sanitized_args.get("heartbeat_run_index")
        heartbeat_total_runs = sanitized_args.get("heartbeat_total_runs")
        heartbeat_path = (
            Path(str(heartbeat_path_raw)).expanduser().resolve() if heartbeat_path_raw else None
        )
        heartbeat_stop = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None

        def _emit_heartbeat() -> None:
            progress_message = None
            if heartbeat_run_index and heartbeat_total_runs:
                progress_message = f"TabICL training progress: run {heartbeat_run_index}/{heartbeat_total_runs} - {heartbeat_label}"
            payload = {
                "status": "running",
                "phase": "fit",
                "backend_name": self.backend_name,
                "label": heartbeat_label,
                "run_index": heartbeat_run_index,
                "total_runs": heartbeat_total_runs,
                "progress_message": progress_message,
                "split_type": split_type,
                "validation_protocol": validation_protocol,
                "train_csv": train_csv,
                "output_dir": output_dir,
                "started_at": started_at.isoformat(),
                "last_heartbeat_at": project_now().isoformat(),
                "elapsed_seconds": round((project_now() - started_at).total_seconds(), 3),
                "target_column": target_column,
                "feature_count": len(feature_columns),
                "train_rows": heartbeat_train_rows,
                "val_rows": heartbeat_val_rows,
                "test_rows": heartbeat_test_rows,
            }
            if heartbeat_path is not None:
                heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
                heartbeat_path.write_text(json.dumps(payload, indent=2) + "\n")
            if progress_message:
                logger.info("%s", progress_message)
            else:
                logger.info(
                    "TabICL training heartbeat: label=%s split=%s elapsed=%.1fs train=%d val=%d test=%d features=%d output_dir=%s",
                    heartbeat_label,
                    split_type,
                    payload["elapsed_seconds"],
                    payload["train_rows"],
                    payload["val_rows"],
                    payload["test_rows"],
                    payload["feature_count"],
                    output_dir,
                )

        def _heartbeat_loop() -> None:
            while not heartbeat_stop.wait(heartbeat_seconds):
                _emit_heartbeat()

        if heartbeat_seconds > 0:
            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                name=f"tabicl-heartbeat-{heartbeat_label}",
                daemon=True,
            )
            heartbeat_thread.start()
        try:
            estimator.fit(X_train, y_train)
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
            else:
                probabilities = None
                y_pred = pd.Series(estimator.predict(X_test), index=X_test.index, dtype=float)
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL training failed: {exc}") from exc
        finally:
            if heartbeat_thread is not None:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=max(1.0, heartbeat_seconds))

        checkpoint_path = checkpoint_cfg["checkpoint_path"]
        persisted_checkpoint = self._persist_checkpoint_if_possible(estimator, checkpoint_path)

        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        model_artifact_path = output_path / "tabicl_model.pkl"
        model_metadata_path = model_artifact_path.with_suffix(".metadata.json")
        test_predictions_path = output_path / "test_predictions.csv"
        summary_path = output_path / "tabicl_training_summary.json"
        canonical_summary_path = output_path / "cs_copilot_training_summary.json"
        config_path = output_path / "config.toml"
        splits_path = output_path / "splits.json"

        save_model_weights = bool(sanitized_args.get("save_model_weights", True))
        save_training_data = bool(sanitized_args.get("save_training_data", True))
        save_kv_cache = bool(sanitized_args.get("save_kv_cache", False))
        metadata = {
            "backend_name": self.backend_name,
            "task_type": task.task_type,
            "task_kind": task_kind,
            "target_columns": [target_column],
            "feature_columns": feature_columns,
            "class_labels": [_json_safe_label(label) for label in class_labels],
            "positive_class_label": (
                _json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
            ),
            "classification_threshold": float(sanitized_args.get("classification_threshold", 0.5)),
        }
        try:
            estimator.save(
                str(model_artifact_path),
                save_model_weights=save_model_weights,
                save_training_data=save_training_data,
                save_kv_cache=save_kv_cache,
            )
        except Exception:
            with model_artifact_path.open("wb") as fh:
                pickle.dump({"model": estimator, "metadata": metadata}, fh)
        model_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

        if task_kind == "classification":
            true_labels = test_df[target_column].reset_index(drop=True)
            predictions_df = pd.DataFrame(
                {
                    **{
                        column: test_df[column].reset_index(drop=True)
                        for column in ("Drug_ID", "smiles")
                        if column in test_df.columns
                    }
                }
            )
            predictions_df = pd.concat(
                [
                    predictions_df,
                    self._classification_output_frame(
                        predicted_codes=y_pred,
                        probabilities=probabilities,
                        class_labels=class_labels,
                        target_column=target_column,
                        true_codes=y_test.reset_index(drop=True),
                        true_labels=true_labels,
                    ),
                ],
                axis=1,
            )
            positive_scores = (
                pd.Series(np.asarray(probabilities, dtype=float)[:, 1])
                if probabilities is not None and len(class_labels) == 2
                else None
            )
            metrics = self._compute_classification_metrics(
                y_test.reset_index(drop=True),
                y_pred.reset_index(drop=True),
                class_labels,
                positive_scores=positive_scores,
            )
        else:
            predictions_df = pd.DataFrame(
                {
                    **{
                        column: test_df[column].reset_index(drop=True)
                        for column in ("Drug_ID", "smiles")
                        if column in test_df.columns
                    },
                    f"{target_column}_true": y_test.reset_index(drop=True),
                    target_column: y_pred.reset_index(drop=True),
                    "y_true": y_test.reset_index(drop=True),
                    "y_pred": y_pred.reset_index(drop=True),
                }
            )
            metrics = self._compute_regression_metrics(
                predictions_df["y_true"].astype(float),
                predictions_df["y_pred"].astype(float),
            )

        with S3.open(str(test_predictions_path), "w") as fh:
            predictions_df.to_csv(fh, index=False)

        train_rows = int(len(train_df))
        val_rows = int(len(val_df))
        test_rows = int(len(test_df))

        summary = {
            "backend_name": self.backend_name,
            "train_csv": train_csv,
            "task_type": task.task_type,
            "task_kind": task_kind,
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "split_type": split_type,
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "random_state": random_state,
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "checkpoint_present_after_run": checkpoint_path.exists(),
            "metrics": {"test": metrics},
            "model_artifact_path": str(model_artifact_path),
            "model_metadata_path": str(model_metadata_path),
            "test_predictions_path": str(test_predictions_path),
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "output_dir": str(output_path),
            "started_at": started_at.isoformat(),
        }
        if task_kind == "classification":
            summary.update(
                {
                    "class_labels": [_json_safe_label(label) for label in class_labels],
                    "class_count": len(class_labels),
                    "positive_class_label": (
                        _json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
                    ),
                    "classification_threshold": float(
                        sanitized_args.get("classification_threshold", 0.5)
                    ),
                }
            )
        config_payload = [
            'backend_name = "tabicl"',
            f'task_type = "{task.task_type}"',
            f'target_column = "{target_column}"',
            f"random_state = {random_state}",
            f'split_type = "{split_type}"',
            f'validation_protocol = "{validation_protocol}"',
            f'checkpoint_version = "{checkpoint_cfg["checkpoint_version"]}"',
        ]
        config_path.write_text("\n".join(config_payload) + "\n")
        splits_path.write_text(json.dumps(split_payload, indent=2) + "\n")
        completed_at = project_now()
        summary["completed_at"] = completed_at.isoformat()
        summary["duration_seconds"] = round((completed_at - started_at).total_seconds(), 3)
        with S3.open(str(summary_path), "w") as fh:
            json.dump(summary, fh, indent=2)
        canonical_summary_path.write_text(json.dumps(summary, indent=2) + "\n")

        result = {
            "backend_name": self.backend_name,
            "model_path": str(model_artifact_path),
            "model_metadata_path": str(model_metadata_path),
            "output_dir": str(output_path),
            "train_csv": train_csv,
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "metrics": {"test": metrics},
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "target_column": target_column,
            "task_type": task.task_type,
            "task_kind": task_kind,
            "split_type": split_type,
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "test_predictions_path": str(test_predictions_path),
            "summary_path": str(summary_path),
            "canonical_summary_path": str(canonical_summary_path),
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "save_model_weights": save_model_weights,
            "save_training_data": save_training_data,
            "save_kv_cache": save_kv_cache,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
        }
        if task_kind == "classification":
            result.update(
                {
                    "class_labels": [_json_safe_label(label) for label in class_labels],
                    "class_count": len(class_labels),
                    "positive_class_label": (
                        _json_safe_label(class_labels[1]) if len(class_labels) == 2 else None
                    ),
                    "classification_threshold": float(
                        sanitized_args.get("classification_threshold", 0.5)
                    ),
                }
            )

        # TabICL can leave large CPU/GPU buffers resident in the Python process
        # after a run. Clear the heaviest objects explicitly before returning.
        del estimator
        del dataset, working, train_df, val_df, test_df
        del X_train, X_test, y_train, y_test, y_pred, predictions_df
        _release_process_memory()

        return result
