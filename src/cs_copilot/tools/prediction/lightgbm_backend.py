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
from .qsar_splitters import build_qsar_split_payload

logger = logging.getLogger(__name__)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def _coerce_split_sizes(split_sizes: Optional[List[float]]) -> List[float]:
    if not split_sizes:
        return [0.8, 0.1, 0.1]
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
                .map(lambda raw: mapping.get(_normalize_category_value(raw), -1))
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
            raise InvalidPredictionInputError(
                f"Prediction input is missing feature columns: {missing_features}"
            )

        features = df[feature_columns].copy()
        features, _ = self._encode_categorical_frame(
            features,
            categorical_feature_columns,
            category_mappings=category_mappings,
        )
        try:
            y_pred = payload["model"].predict(features)
        except Exception as exc:
            raise PredictionExecutionError(f"LightGBM prediction failed: {exc}") from exc

        output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
        if len(target_columns) == 1:
            output[target_columns[0]] = output["prediction"]
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
        if task.task_type != "regression":
            raise InvalidPredictionInputError("LightGBM V1 only supports regression tasks.")
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

        if not split_payload:
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
        if not train_idx or not test_idx:
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
        y_train = pd.to_numeric(encoded_working.iloc[train_idx][target_column], errors="coerce").astype(float)
        X_val = encoded_working.iloc[val_idx][feature_columns].copy() if val_idx else None
        y_val = (
            pd.to_numeric(encoded_working.iloc[val_idx][target_column], errors="coerce").astype(float)
            if val_idx
            else None
        )
        X_test = encoded_working.iloc[test_idx][feature_columns].copy()
        y_test = pd.to_numeric(encoded_working.iloc[test_idx][target_column], errors="coerce").astype(float)

        requested_device_type, auto_device, compute_env = self._resolve_device_type(sanitized_args)
        gpu_fallback_to_cpu = bool(sanitized_args.get("gpu_fallback_to_cpu", True))
        early_stopping_rounds = int(sanitized_args.get("early_stopping_rounds", 50))
        model_params = self._default_model_params(sanitized_args)
        model_params["device_type"] = requested_device_type
        fit_error: Optional[Exception] = None

        try:
            regressor = self._fit_regressor(
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
                regressor = self._fit_regressor(
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

        try:
            y_pred = pd.Series(regressor.predict(X_test), index=X_test.index, dtype=float)
        except Exception as exc:
            raise PredictionExecutionError(f"LightGBM training failed: {exc}") from exc

        metrics = {"test": self._compute_regression_metrics(y_test, y_pred)}
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

        model_path = model_dir / "best.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(model_payload, fh)

        test_predictions_path = model_dir / "test_predictions.csv"
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
        return {
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
            "split_payload": split_payload,
            "effective_split_payload": effective_split_payload,
            "split_metadata": split_indices.get("metadata") or {},
            "has_validation_split": bool(val_idx),
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
                "early_stopping_used": bool(val_idx and early_stopping_rounds > 0),
            },
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "duration_seconds": round((completed_at - started_at).total_seconds(), 3),
            "fit_error": str(fit_error) if fit_error and actual_device_type == "cpu" else None,
        }
