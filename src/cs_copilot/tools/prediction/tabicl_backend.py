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
from .qsar_splitters import build_full_train_split_payload, build_qsar_split_payload
from .qsar_training_policy import project_now
from .training_orchestration import (
    classification_task_kind,
    compute_classification_metrics,
    compute_regression_metrics,
    decode_classification_labels,
    encode_classification_labels,
    is_classification_task,
    json_safe_label,
    resolve_class_labels,
)

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


class TabICLBackend(PredictionBackend):
    """Prediction backend built around TabICLv2 tabular estimators."""

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
        regressor_checkpoint_path = DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        classifier_checkpoint_path = (
            DEFAULT_TABICL_CHECKPOINT_DIR / DEFAULT_TABICL_CLASSIFIER_CHECKPOINT
        )
        classifier_available = False
        if self.is_available():
            try:
                from tabicl import TabICLClassifier  # noqa: F401

                classifier_available = True
            except Exception:
                classifier_available = False
        return enrich_backend_environment(
            self.backend_name,
            {
                "backend_name": self.backend_name,
                "available": self.is_available(),
                "classification_available": classifier_available,
                "package_version": self._package_version(),
                "checkpoint_dir": str(DEFAULT_TABICL_CHECKPOINT_DIR),
                "default_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "default_checkpoint_path": str(regressor_checkpoint_path),
                "default_checkpoint_present": regressor_checkpoint_path.exists(),
                "default_regressor_checkpoint_version": DEFAULT_TABICL_REGRESSOR_CHECKPOINT,
                "default_regressor_checkpoint_path": str(regressor_checkpoint_path),
                "default_regressor_checkpoint_present": regressor_checkpoint_path.exists(),
                "default_classifier_checkpoint_version": DEFAULT_TABICL_CLASSIFIER_CHECKPOINT,
                "default_classifier_checkpoint_path": str(classifier_checkpoint_path),
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

    def _default_checkpoint_for_task(self, task_type: str) -> str:
        return (
            DEFAULT_TABICL_CLASSIFIER_CHECKPOINT
            if is_classification_task(task_type)
            else DEFAULT_TABICL_REGRESSOR_CHECKPOINT
        )

    def _resolve_checkpoint_config(
        self,
        extra_args: Optional[Dict[str, Any]],
        *,
        task_type: str = "regression",
    ) -> Dict[str, Any]:
        extra_args = dict(extra_args or {})
        checkpoint_version = str(
            extra_args.get("checkpoint_version") or self._default_checkpoint_for_task(task_type)
        )
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
        try:
            from tabicl import TabICLClassifier
        except ImportError as exc:
            raise BackendNotAvailableError(
                "This TabICL runtime does not expose TabICLClassifier; classification is unavailable."
            ) from exc
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
            "class_shuffle_method",
            "softmax_temperature",
            "average_logits",
            "support_many_classes",
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
            "final_refit",
            "classification_threshold",
            "class_labels",
        }
        dropped = sorted(key for key in raw if key not in allowed)
        sanitized = {key: value for key, value in raw.items() if key in allowed}
        if dropped:
            logger.warning(
                "Dropping unsupported TabICL train args: %s",
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
                raise InvalidPredictionInputError(f"Requested feature columns are missing: {missing}")
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
                    logger.warning("Could not persist TabICL checkpoint to %s: %s", destination, exc)
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
            raise InvalidPredictionInputError("TabICL does not support predictive uncertainty export.")

        model_path = self.validate_model_path(model_record.model_path)
        try:
            with model_path.open("rb") as fh:
                estimator = pickle.load(fh)
        except Exception as exc:
            raise PredictionExecutionError(f"Could not load TabICL model artifact {model_path}: {exc}") from exc

        with S3.open(input_csv, "r") as fh:
            df = _strip_unnamed_columns(pd.read_csv(fh))

        target_columns = list(model_record.task.target_columns)
        feature_columns = list((model_record.inference_profile or {}).get("feature_columns", []))
        if not feature_columns:
            feature_columns = self._select_feature_columns(df, target_columns, {})

        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            raise InvalidPredictionInputError(f"Prediction input is missing feature columns: {missing_features}")

        X = df[feature_columns].copy()
        try:
            if is_classification_task(model_record.task.task_type):
                class_labels = list((model_record.inference_profile or {}).get("class_labels") or [])
                if not class_labels:
                    raise InvalidPredictionInputError("TabICL classification artifact is missing class_labels metadata.")
                raw_pred = estimator.predict(X)
                predicted = decode_classification_labels(raw_pred, class_labels)
                output = pd.DataFrame(
                    {
                        "prediction": predicted,
                        "prediction_class_code": pd.Series(raw_pred).astype(int),
                    }
                )
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
                if hasattr(estimator, "predict_proba"):
                    proba = pd.DataFrame(estimator.predict_proba(X))
                    for index, class_label in enumerate(class_labels[: proba.shape[1]]):
                        output[f"probability_{json_safe_label(class_label)}"] = pd.to_numeric(proba.iloc[:, index], errors="coerce")
                    if len(class_labels) == 2 and proba.shape[1] >= 2:
                        output["positive_probability"] = pd.to_numeric(proba.iloc[:, 1], errors="coerce")
            else:
                y_pred = estimator.predict(X)
                output = pd.DataFrame({"prediction": pd.Series(y_pred).astype(float)})
                if len(target_columns) == 1:
                    output[target_columns[0]] = output["prediction"]
        except Exception as exc:
            raise PredictionExecutionError(f"TabICL prediction failed: {exc}") from exc
        with S3.open(preds_path, "w") as fh:
            output.to_csv(fh, index=False)

        return {
            "preds_path": preds_path,
            "rows": int(len(output)),
            "feature_columns": feature_columns,
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
            raise InvalidPredictionInputError("TabICL supports regression and classification tasks.")
        if len(task.target_columns) != 1:
            raise InvalidPredictionInputError("TabICL requires exactly one target column.")

        sanitized_args = self._sanitize_train_extra_args(extra_args)
        checkpoint_cfg = self._resolve_checkpoint_config(
            sanitized_args,
            task_type=task.task_type,
        )
        if not checkpoint_cfg["checkpoint_path"].exists():
            raise InvalidPredictionInputError(
                "TabICL checkpoint not found at the expected persistent path: "
                f"{checkpoint_cfg['checkpoint_path']}. Provision this checkpoint before training."
            )
        raw_split_sizes = sanitized_args.pop("split_sizes", None)
        split_payload = sanitized_args.pop("split_payload", None)
        random_state = int(sanitized_args.get("random_state", 42))
        split_type = str(sanitized_args.get("split_type", "random"))
        validation_protocol = str(sanitized_args.get("validation_protocol", "standard_qsar"))
        final_refit = bool(sanitized_args.get("final_refit", False))
        split_sizes = [1.0] if final_refit and raw_split_sizes in (None, [1.0], (1.0,)) else _coerce_split_sizes(raw_split_sizes)
        started_at = project_now()

        with S3.open(train_csv, "r") as fh:
            dataset = _strip_unnamed_columns(pd.read_csv(fh))

        target_column = task.target_columns[0]
        if target_column not in dataset.columns:
            raise InvalidPredictionInputError(f"Missing target column: {target_column}")

        feature_columns = self._select_feature_columns(dataset, [target_column], sanitized_args)
        working = dataset[feature_columns + [target_column] + [c for c in ("smiles", "Drug_ID") if c in dataset.columns]].copy()
        class_labels: List[Any] = []
        class_mapping: Dict[str, int] = {}
        if task_is_classification:
            class_labels = resolve_class_labels(working[target_column])
            if len(class_labels) < 2:
                raise InvalidPredictionInputError("TabICL classification requires at least two classes.")
            encoded_target, class_mapping = encode_classification_labels(working[target_column], class_labels)
            working[target_column] = encoded_target
        else:
            working[target_column] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=[target_column]).reset_index(drop=True)
        if len(working) < 10:
            raise InvalidPredictionInputError("TabICL requires at least 10 rows after target cleanup.")

        if final_refit:
            split_payload = build_full_train_split_payload(df=working)
        elif split_payload is None:
            split_payload = build_qsar_split_payload(
                df=working,
                split_type=split_type,
                split_sizes=split_sizes,
                random_state=random_state,
                smiles_column="smiles" if "smiles" in working.columns else None,
                feature_columns=feature_columns,
            )
        if not isinstance(split_payload, list) or not split_payload or not isinstance(split_payload[0], dict):
            raise InvalidPredictionInputError(
                "TabICL split_payload must be a non-empty list with train/test index mappings."
            )

        split_map = split_payload[0]
        train_indices = [int(idx) for idx in (split_map.get("train") or [])]
        val_indices = [int(idx) for idx in (split_map.get("val") or [])]
        test_indices = [int(idx) for idx in (split_map.get("test") or [])]
        if not train_indices or (not final_refit and not test_indices):
            raise InvalidPredictionInputError("TabICL split payload must provide non-empty train/test indices.")
        train_df = working.iloc[train_indices].reset_index(drop=True)
        val_df = working.iloc[val_indices].reset_index(drop=True)
        test_df = working.iloc[test_indices].reset_index(drop=True)
        train_rows = int(len(train_df))
        val_rows = int(len(val_df))
        test_rows = int(len(test_df))

        X_train = train_df[feature_columns].copy()
        y_train = train_df[target_column].astype(int if task_is_classification else float).copy()
        X_test = test_df[feature_columns].copy() if not test_df.empty else None
        y_test = test_df[target_column].astype(int if task_is_classification else float).copy() if not test_df.empty else None
        if task_is_classification:
            missing_train_classes = sorted(
                set(range(len(class_labels))) - {int(value) for value in y_train.tolist()}
            )
            if missing_train_classes:
                missing = [json_safe_label(class_labels[index]) for index in missing_train_classes]
                raise InvalidPredictionInputError(
                    f"TabICL classification training split is missing target classes {missing}."
                )

        Estimator = self._import_tabicl_classifier() if task_is_classification else self._import_tabicl_regressor()
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
        classifier_optional_keys = (
            "class_shuffle_method",
            "softmax_temperature",
            "average_logits",
            "support_many_classes",
        )
        for key in optional_keys + (classifier_optional_keys if task_is_classification else ()):
            if key in sanitized_args:
                init_kwargs[key] = sanitized_args[key]

        estimator = Estimator(**init_kwargs)
        heartbeat_seconds = float(sanitized_args.get("heartbeat_seconds", 120.0))
        heartbeat_path_raw = sanitized_args.get("heartbeat_path")
        heartbeat_label = str(sanitized_args.get("heartbeat_label") or split_type)
        heartbeat_run_index = sanitized_args.get("heartbeat_run_index")
        heartbeat_total_runs = sanitized_args.get("heartbeat_total_runs")
        heartbeat_path = Path(str(heartbeat_path_raw)).expanduser().resolve() if heartbeat_path_raw else None
        heartbeat_stop = threading.Event()
        heartbeat_thread: Optional[threading.Thread] = None

        def _emit_heartbeat() -> None:
            progress_message = None
            if heartbeat_run_index and heartbeat_total_runs:
                progress_message = (
                    f"TabICL training progress: run {heartbeat_run_index}/{heartbeat_total_runs} - {heartbeat_label}"
                )
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
                "train_rows": train_rows,
                "val_rows": val_rows,
                "test_rows": test_rows,
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
            y_pred = (
                pd.Series(estimator.predict(X_test), index=X_test.index)
                if X_test is not None
                else None
            )
            y_proba = (
                estimator.predict_proba(X_test)
                if task_is_classification and X_test is not None and hasattr(estimator, "predict_proba")
                else None
            )
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
        test_predictions_path = output_path / "test_predictions.csv"
        summary_path = output_path / "tabicl_training_summary.json"
        canonical_summary_path = output_path / "cs_copilot_training_summary.json"
        config_path = output_path / "config.toml"
        splits_path = output_path / "splits.json"

        save_model_weights = bool(sanitized_args.get("save_model_weights", True))
        save_training_data = bool(sanitized_args.get("save_training_data", True))
        save_kv_cache = bool(sanitized_args.get("save_kv_cache", False))
        try:
            estimator.save(
                str(model_artifact_path),
                save_model_weights=save_model_weights,
                save_training_data=save_training_data,
                save_kv_cache=save_kv_cache,
            )
        except Exception:
            with model_artifact_path.open("wb") as fh:
                pickle.dump(estimator, fh)

        if y_pred is not None and y_test is not None:
            id_columns = {
                column: test_df[column].reset_index(drop=True)
                for column in ("Drug_ID", "smiles")
                if column in test_df.columns
            }
            if task_is_classification:
                y_true_labels = decode_classification_labels(y_test.reset_index(drop=True), class_labels)
                y_pred_labels = decode_classification_labels(y_pred.reset_index(drop=True), class_labels)
                predictions_df = pd.DataFrame(
                    {
                        **id_columns,
                        f"{target_column}_true": y_true_labels,
                        target_column: y_pred_labels,
                        "prediction": y_pred_labels,
                        "y_true": y_true_labels,
                        "y_pred": y_pred_labels,
                        "prediction_class_code": y_pred.reset_index(drop=True).astype(int),
                    }
                )
                positive_scores = None
                if y_proba is not None:
                    proba = pd.DataFrame(y_proba)
                    for index, class_label in enumerate(class_labels[: proba.shape[1]]):
                        predictions_df[f"probability_{json_safe_label(class_label)}"] = pd.to_numeric(
                            proba.iloc[:, index],
                            errors="coerce",
                        )
                    if proba.shape[1] >= 2:
                        positive_scores = pd.to_numeric(proba.iloc[:, 1], errors="coerce")
                        predictions_df["positive_probability"] = positive_scores
                metrics = compute_classification_metrics(
                    y_true_labels,
                    y_pred_labels,
                    class_labels=class_labels,
                    positive_scores=positive_scores,
                    target_column=target_column,
                )
            else:
                predictions_df = pd.DataFrame(
                    {
                        **id_columns,
                        f"{target_column}_true": y_test.reset_index(drop=True),
                        target_column: y_pred.reset_index(drop=True),
                        "y_true": y_test.reset_index(drop=True),
                        "y_pred": y_pred.reset_index(drop=True),
                    }
                )
                metrics = compute_regression_metrics(
                    predictions_df["y_true"].astype(float),
                    predictions_df["y_pred"].astype(float),
                    target_column=target_column,
                )
            with S3.open(str(test_predictions_path), "w") as fh:
                predictions_df.to_csv(fh, index=False)
        else:
            predictions_df = None
            test_predictions_path = None
            metrics = {}

        metrics_payload = {"test": metrics} if metrics else {}
        summary = {
            "backend_name": self.backend_name,
            "train_csv": train_csv,
            "task_type": task.task_type,
            "task_kind": classification_task_kind(task.task_type, len(class_labels)) if task_is_classification else "regression",
            "target_column": target_column,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "split_type": split_type,
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "split_metadata": split_map.get("metadata") or {},
            "has_validation_split": bool(val_indices),
            "random_state": random_state,
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "checkpoint_present_after_run": checkpoint_path.exists(),
            "metrics": metrics_payload,
            "metrics_status": "not_evaluated" if final_refit else "evaluated",
            "evaluation_required": bool(final_refit),
            "class_labels": [json_safe_label(label) for label in class_labels],
            "class_count": len(class_labels) if task_is_classification else None,
            "label_mapping": class_mapping,
            "positive_class_label": json_safe_label(class_labels[1]) if task_is_classification else None,
            "model_artifact_path": str(model_artifact_path),
            "test_predictions_path": str(test_predictions_path) if test_predictions_path else None,
            "config_path": str(config_path),
            "splits_path": str(splits_path),
            "output_dir": str(output_path),
            "started_at": started_at.isoformat(),
        }
        config_payload = [
            'backend_name = "tabicl"',
            f'task_type = "{task.task_type}"',
            f'target_column = "{target_column}"',
            f'random_state = {random_state}',
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
            "output_dir": str(output_path),
            "train_csv": train_csv,
            "checkpoint_version": checkpoint_cfg["checkpoint_version"],
            "checkpoint_path": persisted_checkpoint or str(checkpoint_path),
            "metrics": metrics_payload,
            "metrics_status": "not_evaluated" if final_refit else "evaluated",
            "evaluation_required": bool(final_refit),
            "class_labels": [json_safe_label(label) for label in class_labels],
            "class_count": len(class_labels) if task_is_classification else None,
            "label_mapping": class_mapping,
            "positive_class_label": json_safe_label(class_labels[1]) if task_is_classification else None,
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "target_column": target_column,
            "task_kind": classification_task_kind(task.task_type, len(class_labels)) if task_is_classification else "regression",
            "split_type": split_type,
            "validation_protocol": validation_protocol,
            "split_sizes": split_sizes,
            "split_metadata": split_map.get("metadata") or {},
            "has_validation_split": bool(val_indices),
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "test_predictions_path": str(test_predictions_path) if test_predictions_path else None,
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
            "final_refit": final_refit,
        }

        # TabICL can leave large CPU/GPU buffers resident in the Python process
        # after a run. Clear the heaviest objects explicitly before returning.
        del estimator
        del dataset, working, train_df, val_df, test_df
        del X_train, X_test, y_train, y_test, y_pred, predictions_df
        _release_process_memory()

        return result
