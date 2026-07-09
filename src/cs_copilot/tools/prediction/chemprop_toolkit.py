#!/usr/bin/env python
# coding: utf-8
"""
Internal toolkit for Chemprop-specific QSAR training flows.
"""

from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import (
    prepare_activity_cliff_context,
    split_activity_cliff_args,
)

from .backend import PredictionModelRecord, PredictionTaskSpec
from .chemprop_adapter import materialize_chemprop_inputs
from .chemprop_backend import DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX, ChempropBackend
from .qsar_splitters import (
    build_full_train_split_payload,
    build_qsar_split_payload,
    build_repeated_kfold_split_payloads,
)
from .qsar_training_policy import (
    assess_protocol_results,
    describe_compute_environment,
    project_now,
    resolve_training_profile,
    safe_slug,
    seed_policy_reporting_text,
    seed_policy_reproducibility_metadata,
    summarize_training_durations,
)
from .qsar_validation_strategy import resolve_validation_strategy
from .session_state import (
    bundle_artifacts,
    get_prediction_state,
    write_active_training_marker,
)
from .training_orchestration import (
    apply_training_profile,
    build_applicability_domain_for_training,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    collect_training_bundle_files,
    compute_classification_metrics,
    compute_regression_metrics,
    decode_classification_labels,
    is_classification_task,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)


def _strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    return strip_unnamed_columns(df)


def _agent_storage_path(path: str | Path) -> str:
    """Normalize agent-returned session paths for S3.open."""
    raw = str(path)
    if raw.startswith(("s3://", "/", "file://")):
        return raw

    prefix = S3.current_prefix().strip("/")
    for root in (".files", "data"):
        session_prefix = f"{root}/{prefix}/"
        while raw.startswith(session_prefix):
            raw = raw[len(session_prefix) :]

    while raw.startswith(f"{prefix}/"):
        raw = raw[len(prefix) + 1 :]

    return raw


def _agent_local_path(path: str | Path) -> str:
    """Normalize agent-returned session paths for filesystem-only helpers."""
    storage_path = _agent_storage_path(path)
    if storage_path.startswith(("s3://", "/", "file://")):
        return storage_path
    return S3.path(storage_path)


def _find_first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except Exception:
            continue
    return None


class ChempropToolkit(Toolkit):
    """Backend-specific Chemprop training toolkit used behind QSARTrainingToolkit."""

    def __init__(
        self,
        backend: Optional[ChempropBackend] = None,
        *,
        register_tools: bool = True,
    ):
        super().__init__("chemprop_prediction")
        self.backend = backend or ChempropBackend()
        if not register_tools:
            return

        self.register(self.describe_backend)
        self.register(self.describe_compute_environment)
        self.register(self.validate_chemprop_model_path)
        self.register(self.train_model)

    def _detect_memory_limit_bytes(self) -> Optional[int]:
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
                # Ignore absurdly large “no real limit” cgroup values.
                if value <= 0 or value > 1 << 60:
                    continue
                return value
            except Exception:
                continue
        return None

    def _resolve_chemprop_run_artifacts(self, output_dir: Path) -> Dict[str, Optional[Path]]:
        output_path = output_dir.expanduser().resolve()
        best_model_path = _find_first_existing_path(
            [
                output_path / "model_0" / "best.pt",
                output_path / "replicate_0" / "model_0" / "best.pt",
            ]
        )
        test_predictions_path = _find_first_existing_path(
            [
                output_path / "model_0" / "test_predictions.csv",
                output_path / "replicate_0" / "model_0" / "test_predictions.csv",
            ]
        )
        validation_predictions_path = _find_first_existing_path(
            [
                output_path / "model_0" / "validation_predictions.csv",
                output_path / "replicate_0" / "model_0" / "validation_predictions.csv",
            ]
        )
        return {
            "best_model_path": best_model_path,
            "validation_predictions_path": validation_predictions_path,
            "test_predictions_path": test_predictions_path,
            "config_path": output_path / "config.toml",
            "splits_path": output_path / "splits.json",
        }

    def _replicate_artifacts(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Return Chemprop replicate artifacts present in a split output directory."""
        output_path = output_dir.expanduser().resolve()
        replicate_dirs = sorted(
            output_path.glob("replicate_*"),
            key=lambda path: (
                int(path.name.split("_")[-1]) if path.name.split("_")[-1].isdigit() else 0
            ),
        )
        artifacts: List[Dict[str, Any]] = []
        for replicate_dir in replicate_dirs:
            raw_index = replicate_dir.name.split("_")[-1]
            replicate_index = int(raw_index) if raw_index.isdigit() else len(artifacts)
            model_path = replicate_dir / "model_0" / "best.pt"
            predictions_path = replicate_dir / "model_0" / "test_predictions.csv"
            artifacts.append(
                {
                    "replicate_index": replicate_index,
                    "model_path": str(model_path) if model_path.exists() else None,
                    "raw_test_predictions_path": (
                        str(predictions_path) if predictions_path.exists() else None
                    ),
                }
            )

        if not artifacts:
            model_path = output_path / "model_0" / "best.pt"
            predictions_path = output_path / "model_0" / "test_predictions.csv"
            if model_path.exists() or predictions_path.exists():
                artifacts.append(
                    {
                        "replicate_index": 0,
                        "model_path": str(model_path) if model_path.exists() else None,
                        "raw_test_predictions_path": (
                            str(predictions_path) if predictions_path.exists() else None
                        ),
                    }
                )
        return artifacts

    def _write_normalized_test_predictions(
        self,
        *,
        train_csv: str,
        output_dir: Path,
        task: PredictionTaskSpec,
        splits_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a self-contained Chemprop test-prediction CSV.

        Chemprop writes prediction CSVs with the target column name reused for
        predictions.  For validation artifacts we keep that compatibility
        column but add explicit truth/prediction/error columns and aggregate
        replicate predictions when multiple replicate outputs are present.
        """
        target_columns = list(task.target_columns or [])
        target_column = target_columns[0] if target_columns else None
        if not target_column:
            return {}

        output_path = output_dir.expanduser().resolve()
        splits_path = (
            Path(str(splits_file)).expanduser() if splits_file else output_path / "splits.json"
        )
        if not splits_path.exists():
            return {}

        replicate_artifacts = self._replicate_artifacts(output_path)
        if not any(item.get("raw_test_predictions_path") for item in replicate_artifacts):
            return {}

        dataset = _strip_unnamed_columns(
            pd.read_csv(Path(_agent_local_path(train_csv)).expanduser())
        )
        split_payload = json.loads(splits_path.read_text())
        if not split_payload or "test" not in split_payload[0]:
            return {}
        test_indices = split_payload[0].get("test") or []
        actual = dataset.iloc[test_indices].reset_index(drop=True)
        missing_actual_targets = [
            column for column in target_columns if column not in actual.columns
        ]
        if missing_actual_targets:
            return {}

        smiles_column = task.smiles_columns[0] if task.smiles_columns else "smiles"
        actual_smiles = (
            actual[smiles_column].astype(str).reset_index(drop=True)
            if smiles_column in actual.columns
            else None
        )
        prediction_series_by_target: Dict[str, List[pd.Series]] = {
            column: [] for column in target_columns
        }
        prediction_source_paths: List[str] = []
        included_replicate_indices: List[int] = []
        normalized_replicate_artifacts: List[Dict[str, Any]] = []
        for item in replicate_artifacts:
            artifact = dict(item)
            raw_predictions_path = item.get("raw_test_predictions_path")
            if not raw_predictions_path:
                artifact["aligned_for_validation"] = False
                artifact["exclusion_reason"] = "missing_test_predictions"
                normalized_replicate_artifacts.append(artifact)
                continue
            prediction_path = Path(str(raw_predictions_path)).expanduser()
            if not prediction_path.exists():
                artifact["aligned_for_validation"] = False
                artifact["exclusion_reason"] = "missing_test_predictions"
                normalized_replicate_artifacts.append(artifact)
                continue
            predictions = _strip_unnamed_columns(pd.read_csv(prediction_path))
            exclusion_reason = None
            missing_prediction_targets = [
                column for column in target_columns if column not in predictions.columns
            ]
            if missing_prediction_targets:
                exclusion_reason = "missing_target_prediction_column"
            elif len(predictions) != len(actual):
                exclusion_reason = "row_count_mismatch"
            elif actual_smiles is not None:
                if smiles_column not in predictions.columns:
                    exclusion_reason = "missing_smiles_alignment_column"
                else:
                    predicted_smiles = predictions[smiles_column].astype(str).reset_index(drop=True)
                    if not predicted_smiles.equals(actual_smiles):
                        exclusion_reason = "smiles_not_aligned_to_split_test_rows"
            if exclusion_reason:
                artifact["aligned_for_validation"] = False
                artifact["exclusion_reason"] = exclusion_reason
                normalized_replicate_artifacts.append(artifact)
                continue
            replicate_index = item.get("replicate_index")
            if not isinstance(replicate_index, int):
                replicate_index = len(included_replicate_indices)
            artifact["aligned_for_validation"] = True
            artifact["exclusion_reason"] = None
            normalized_replicate_artifacts.append(artifact)
            for column in target_columns:
                prediction_series_by_target[column].append(
                    pd.to_numeric(predictions[column], errors="coerce")
                )
            prediction_source_paths.append(str(prediction_path))
            included_replicate_indices.append(replicate_index)

        prediction_series = prediction_series_by_target.get(target_column) or []
        if not prediction_series:
            return {}

        prediction_frame = pd.concat(prediction_series, axis=1)
        prediction_frame.columns = [
            f"prediction_replicate_{idx}" for idx in included_replicate_indices
        ]
        is_classification = is_classification_task(task.task_type)
        y_true_numeric = pd.to_numeric(actual[target_column], errors="coerce")
        y_pred_numeric = prediction_frame.mean(axis=1, skipna=True)
        prediction_std = (
            prediction_frame.std(axis=1, ddof=0).fillna(0.0)
            if len(prediction_series) > 1
            else pd.Series([0.0] * len(prediction_frame))
        )
        smiles_values = (
            actual[smiles_column].reset_index(drop=True)
            if smiles_column in actual.columns
            else pd.Series([None] * len(actual))
        )
        if is_classification:
            manifest_path = output_path / "chemprop_inputs" / "chemprop_input_manifest.json"
            classification_targets: Dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    classification_targets = manifest.get("classification_targets") or {}
                except Exception:
                    classification_targets = {}
            class_info: Dict[str, Any] = classification_targets.get(target_column) or {}
            class_labels = list(class_info.get("class_labels") or [0, 1])
            positive_probability = y_pred_numeric.clip(lower=0.0, upper=1.0)
            predicted_codes = (positive_probability >= 0.5).astype(int)
            true_labels = decode_classification_labels(
                y_true_numeric.fillna(-1).astype(int).tolist(), class_labels
            )
            predicted_labels = decode_classification_labels(predicted_codes.tolist(), class_labels)
            normalized = pd.DataFrame(
                {
                    "source_row_index": test_indices,
                    "smiles": smiles_values,
                    f"{target_column}_true": true_labels,
                    f"{target_column}_prediction": predicted_labels,
                    "prediction": predicted_labels,
                    target_column: predicted_labels,
                    "positive_probability": positive_probability,
                    "prediction_class_code": predicted_codes,
                    "true_class_code": y_true_numeric,
                    "prediction_std": prediction_std,
                    "replicate_count": len(prediction_series),
                    "detected_replicate_count": len(replicate_artifacts),
                }
            )
            for extra_target in target_columns[1:]:
                extra_series = prediction_series_by_target.get(extra_target) or []
                if not extra_series:
                    continue
                extra_frame = pd.concat(extra_series, axis=1)
                extra_pred = extra_frame.mean(axis=1, skipna=True).clip(lower=0.0, upper=1.0)
                extra_codes = (extra_pred >= 0.5).astype(int)
                extra_labels = list(
                    (classification_targets.get(extra_target) or {}).get("class_labels") or [0, 1]
                )
                normalized[f"{extra_target}_true"] = decode_classification_labels(
                    pd.to_numeric(actual[extra_target], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    .tolist(),
                    extra_labels,
                )
                normalized[f"{extra_target}_prediction"] = decode_classification_labels(
                    extra_codes.tolist(), extra_labels
                )
                normalized[f"{extra_target}_positive_probability"] = extra_pred
        else:
            normalized = pd.DataFrame(
                {
                    "source_row_index": test_indices,
                    "smiles": smiles_values,
                    f"{target_column}_true": y_true_numeric,
                    f"{target_column}_prediction": y_pred_numeric,
                    "prediction": y_pred_numeric,
                    target_column: y_pred_numeric,
                    "prediction_std": prediction_std,
                    "residual": y_true_numeric - y_pred_numeric,
                    "absolute_error": (y_true_numeric - y_pred_numeric).abs(),
                    "replicate_count": len(prediction_series),
                    "detected_replicate_count": len(replicate_artifacts),
                }
            )
            for extra_target in target_columns[1:]:
                extra_series = prediction_series_by_target.get(extra_target) or []
                if not extra_series:
                    continue
                extra_frame = pd.concat(extra_series, axis=1)
                extra_true = pd.to_numeric(actual[extra_target], errors="coerce")
                extra_pred = extra_frame.mean(axis=1, skipna=True)
                normalized[f"{extra_target}_true"] = extra_true
                normalized[f"{extra_target}_prediction"] = extra_pred
                normalized[f"{extra_target}_residual"] = extra_true - extra_pred
                normalized[f"{extra_target}_absolute_error"] = (extra_true - extra_pred).abs()
        normalized = pd.concat([normalized, prediction_frame], axis=1)

        normalized_path = output_path / "model_0" / "test_predictions.csv"
        normalized_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(normalized_path, index=False)
        aggregation = (
            "mean_aligned_replicates" if len(prediction_series) > 1 else "single_aligned_replicate"
        )
        return {
            "test_predictions_path": str(normalized_path),
            "raw_test_prediction_paths": prediction_source_paths,
            "replicate_artifacts": normalized_replicate_artifacts,
            "replicate_count": len(prediction_series),
            "detected_replicate_count": len(replicate_artifacts),
            "excluded_replicate_count": len(replicate_artifacts) - len(prediction_series),
            "prediction_aggregation": aggregation,
            "replicate_alignment_policy": (
                "Only replicate prediction files whose SMILES exactly match the split test rows "
                "are aggregated for validation metrics."
            ),
            "prediction_column": "prediction",
            "target_true_column": f"{target_column}_true",
            "target_prediction_column": f"{target_column}_prediction",
        }

    def _write_validation_predictions(
        self,
        *,
        train_csv: str,
        output_dir: Path,
        task: PredictionTaskSpec,
        splits_file: Optional[str],
        model_path: Optional[str],
    ) -> Dict[str, Any]:
        target_columns = list(task.target_columns or [])
        target_column = target_columns[0] if target_columns else None
        if not target_column or not model_path:
            return {}

        output_path = output_dir.expanduser().resolve()
        splits_path = (
            Path(str(splits_file)).expanduser() if splits_file else output_path / "splits.json"
        )
        model_artifact = Path(str(model_path)).expanduser()
        if not splits_path.exists() or not model_artifact.exists():
            return {}

        split_payload = json.loads(splits_path.read_text())
        split_map = split_payload[0] if split_payload else {}
        validation_indices = split_map.get("val") or split_map.get("validation") or []
        if not validation_indices:
            return {}

        dataset = _strip_unnamed_columns(
            pd.read_csv(Path(_agent_local_path(train_csv)).expanduser())
        )
        actual = dataset.iloc[[int(index) for index in validation_indices]].reset_index(drop=True)
        missing_targets = [column for column in target_columns if column not in actual.columns]
        if missing_targets:
            return {}

        smiles_column = task.smiles_columns[0] if task.smiles_columns else "smiles"
        model_dir = output_path / "model_0"
        model_dir.mkdir(parents=True, exist_ok=True)
        validation_input_path = model_dir / "validation_input.csv"
        raw_predictions_path = model_dir / "validation_predictions_raw.csv"
        validation_predictions_path = model_dir / "validation_predictions.csv"
        actual.drop(columns=target_columns, errors="ignore").to_csv(
            validation_input_path, index=False
        )

        record = PredictionModelRecord(
            model_id=f"{output_path.name}_validation",
            backend_name=self.backend.backend_name,
            model_path=str(model_artifact),
            task=task,
        )
        self.backend.predict_from_csv(
            input_csv=str(validation_input_path),
            model_record=record,
            preds_path=str(raw_predictions_path),
            return_uncertainty=False,
        )
        predictions = _strip_unnamed_columns(pd.read_csv(raw_predictions_path))
        if len(predictions) != len(actual):
            return {}

        def prediction_column(target: str) -> Optional[str]:
            candidates = [target, f"{target}_prediction"]
            if len(target_columns) == 1:
                candidates.append("prediction")
            return next((column for column in candidates if column in predictions.columns), None)

        is_classification = is_classification_task(task.task_type)
        manifest_path = output_path / "chemprop_inputs" / "chemprop_input_manifest.json"
        classification_targets: Dict[str, Any] = {}
        if is_classification and manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                classification_targets = manifest.get("classification_targets") or {}
            except Exception:
                classification_targets = {}

        smiles_values = (
            actual[smiles_column].reset_index(drop=True)
            if smiles_column in actual.columns
            else pd.Series([None] * len(actual))
        )
        primary_prediction_column = prediction_column(target_column)
        if not primary_prediction_column:
            return {}

        if is_classification:
            class_labels = list(
                (classification_targets.get(target_column) or {}).get("class_labels") or [0, 1]
            )
            positive_probability = pd.to_numeric(
                predictions[primary_prediction_column], errors="coerce"
            ).clip(lower=0.0, upper=1.0)
            predicted_codes = (positive_probability >= 0.5).astype(int)
            true_codes = pd.to_numeric(actual[target_column], errors="coerce")
            y_true = decode_classification_labels(
                true_codes.fillna(-1).astype(int).tolist(), class_labels
            )
            y_pred = decode_classification_labels(predicted_codes.tolist(), class_labels)
            normalized = pd.DataFrame(
                {
                    "source_row_index": validation_indices,
                    "smiles": smiles_values,
                    f"{target_column}_true": y_true,
                    f"{target_column}_prediction": y_pred,
                    "prediction": y_pred,
                    target_column: y_pred,
                    "positive_probability": positive_probability,
                    "prediction_class_code": predicted_codes,
                    "true_class_code": true_codes,
                    "replicate_count": 1,
                    "detected_replicate_count": 1,
                }
            )
            metric_values = compute_classification_metrics(
                pd.Series(y_true),
                pd.Series(y_pred),
                class_labels=class_labels,
                positive_scores=positive_probability,
                target_column=target_column,
            )
        else:
            y_true = pd.to_numeric(actual[target_column], errors="coerce")
            y_pred = pd.to_numeric(predictions[primary_prediction_column], errors="coerce")
            normalized = pd.DataFrame(
                {
                    "source_row_index": validation_indices,
                    "smiles": smiles_values,
                    f"{target_column}_true": y_true,
                    f"{target_column}_prediction": y_pred,
                    "prediction": y_pred,
                    target_column: y_pred,
                    "residual": y_true - y_pred,
                    "absolute_error": (y_true - y_pred).abs(),
                    "replicate_count": 1,
                    "detected_replicate_count": 1,
                }
            )
            metric_values = compute_regression_metrics(
                y_true,
                y_pred,
                target_column=target_column,
            )

        target_metrics: Dict[str, Any] = {target_column: metric_values}
        for extra_target in target_columns[1:]:
            extra_prediction_column = prediction_column(extra_target)
            if not extra_prediction_column:
                continue
            if is_classification:
                extra_labels = list(
                    (classification_targets.get(extra_target) or {}).get("class_labels") or [0, 1]
                )
                extra_probability = pd.to_numeric(
                    predictions[extra_prediction_column], errors="coerce"
                ).clip(lower=0.0, upper=1.0)
                extra_codes = (extra_probability >= 0.5).astype(int)
                extra_true_codes = pd.to_numeric(actual[extra_target], errors="coerce")
                extra_true = decode_classification_labels(
                    extra_true_codes.fillna(-1).astype(int).tolist(), extra_labels
                )
                extra_pred = decode_classification_labels(extra_codes.tolist(), extra_labels)
                normalized[f"{extra_target}_true"] = extra_true
                normalized[f"{extra_target}_prediction"] = extra_pred
                normalized[f"{extra_target}_positive_probability"] = extra_probability
                target_metrics[extra_target] = compute_classification_metrics(
                    pd.Series(extra_true),
                    pd.Series(extra_pred),
                    class_labels=extra_labels,
                    positive_scores=extra_probability,
                    target_column=extra_target,
                )
            else:
                extra_true = pd.to_numeric(actual[extra_target], errors="coerce")
                extra_pred = pd.to_numeric(predictions[extra_prediction_column], errors="coerce")
                normalized[f"{extra_target}_true"] = extra_true
                normalized[f"{extra_target}_prediction"] = extra_pred
                normalized[f"{extra_target}_residual"] = extra_true - extra_pred
                normalized[f"{extra_target}_absolute_error"] = (extra_true - extra_pred).abs()
                target_metrics[extra_target] = compute_regression_metrics(
                    extra_true,
                    extra_pred,
                    target_column=extra_target,
                )

        normalized.to_csv(validation_predictions_path, index=False)
        return {
            "validation_predictions_path": str(validation_predictions_path),
            "raw_validation_predictions_path": str(raw_predictions_path),
            "validation_prediction_input_csv": str(validation_input_path),
            "validation_metrics": metric_values,
            "validation_target_metrics": target_metrics,
        }

    def _detect_physical_memory_bytes(self) -> Optional[int]:
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

    def _detect_disk_usage(self, base_path: Optional[Path] = None) -> Dict[str, Optional[float]]:
        target = (base_path or Path.cwd()).resolve()
        try:
            usage = shutil.disk_usage(target)
        except Exception:
            return {
                "disk_path": str(target),
                "disk_gb_total": None,
                "disk_gb_free": None,
                "disk_gb_used": None,
            }

        gib = 1024**3
        return {
            "disk_path": str(target),
            "disk_gb_total": round(usage.total / gib, 2),
            "disk_gb_free": round(usage.free / gib, 2),
            "disk_gb_used": round(usage.used / gib, 2),
        }

    def describe_compute_environment(self) -> Dict[str, Any]:
        """Describe the local compute budget used to choose safe training defaults."""
        return describe_compute_environment()

    def validate_chemprop_model_path(self, model_path: str) -> Dict[str, Any]:
        """Validate a Chemprop model artifact path."""
        resolved = self.backend.validate_model_path(model_path)
        return {
            "valid": True,
            "model_path": str(resolved),
            "backend_name": self.backend.backend_name,
        }

    def _resolve_training_profile(self, compute_env: Dict[str, Any]) -> Dict[str, Any]:
        return resolve_training_profile(compute_env)

    def _training_defaults_for_profile(self, profile: str) -> Dict[str, Any]:
        if profile == "local_light":
            return {
                "epochs": 30,
                "batch_size": 32,
                "num_replicates": 1,
                "ensemble_size": 1,
                "num_workers": 0,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        if profile == "local_standard":
            return {
                "epochs": 50,
                "batch_size": 32,
                "num_replicates": 1,
                "ensemble_size": 1,
                "num_workers": 0,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        if profile == "heavy_validation":
            return {
                "epochs": 100,
                "batch_size": 64,
                "num_replicates": 3,
                "ensemble_size": 1,
                "num_workers": 16,
                "patience": 15,
                "metric": "rmse",
                "split_type": "random",
                "split_sizes": [0.8, 0.1, 0.1],
            }
        return {
            "epochs": 50,
            "batch_size": 32,
            "num_replicates": 1,
            "ensemble_size": 1,
            "num_workers": 0,
            "metric": "rmse",
            "split_type": "random",
            "split_sizes": [0.8, 0.1, 0.1],
        }

    def _apply_training_profile(
        self,
        extra_args: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        def _limit(
            profile: str, merged: Dict[str, Any], allow_heavy_compute: bool
        ) -> Dict[str, Any]:
            if allow_heavy_compute:
                if profile == "heavy_validation":
                    # On high-compute GPU runs, treat the profile values as floor values:
                    # the agent may request a more aggressive configuration, but not a slower one.
                    merged["epochs"] = max(int(merged.get("epochs", 100)), 100)
                    merged["batch_size"] = max(int(merged.get("batch_size", 64)), 64)
                    merged["num_workers"] = max(int(merged.get("num_workers", 16)), 16)
                return merged
            if profile == "local_light":
                merged["epochs"] = min(int(merged.get("epochs", 30)), 30)
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["ensemble_size"] = 1
                merged["num_replicates"] = 1
                merged["num_workers"] = 0
            elif profile == "local_standard":
                merged["epochs"] = min(int(merged.get("epochs", 50)), 50)
                merged["batch_size"] = min(int(merged.get("batch_size", 32)), 32)
                merged["ensemble_size"] = min(int(merged.get("ensemble_size", 1)), 1)
                merged["num_replicates"] = min(int(merged.get("num_replicates", 1)), 1)
                merged["num_workers"] = 0
            return merged

        return apply_training_profile(
            extra_args,
            defaults_for_profile=self._training_defaults_for_profile,
            limit_profile_args=_limit,
            compute_environment=self.describe_compute_environment(),
            protected_profiles=("heavy_validation", "benchmark"),
        )

    def _apply_protocol_training_overrides(
        self,
        *,
        training_policy: Dict[str, Any],
        protocol_policy: Dict[str, Any],
    ) -> Optional[str]:
        """Apply Chemprop-specific training overrides once the QSAR protocol is known."""
        extra_args = training_policy.setdefault("extra_args", {})
        protocol = protocol_policy.get("protocol") or "qsar"
        requested_replicates = int(extra_args.get("num_replicates") or 1)
        extra_args["num_replicates"] = 1
        if requested_replicates != 1:
            return (
                f"Chemprop {protocol} protocols use one replicate per split. "
                "Robustness is measured through protocol split runs, not Chemprop replicate multiplication."
            )
        return None

    def _resolve_validation_protocol(
        self,
        *,
        requested_protocol: Optional[str],
        training_profile: str,
        seed_policy: Optional[Dict[str, Any]] = None,
        seed_policy_mode: str = "generated_per_run",
        base_seed: Optional[int] = None,
        validation_strategy: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return resolve_validation_strategy(
            requested_protocol=requested_protocol,
            validation_strategy=validation_strategy,
            training_profile=training_profile,
            seed_policy=seed_policy,
            seed_policy_mode=seed_policy_mode,
            base_seed=base_seed,
        )

    def _train_single_run(
        self,
        *,
        train_csv: str,
        task: PredictionTaskSpec,
        output_dir: str,
        train_args: Dict[str, Any],
        split_payload: List[Dict[str, List[int]]],
        split_label: str,
        seed: Optional[int],
    ) -> Dict[str, Any]:
        run_output_dir = Path(output_dir).expanduser().resolve()
        chemprop_input = materialize_chemprop_inputs(
            source_csv=_agent_local_path(train_csv),
            output_dir=run_output_dir / "chemprop_inputs",
            task=task,
            split_payload=split_payload,
            split_label=split_label,
            seed=seed,
        )
        backend_train_args = {
            key: value
            for key, value in train_args.items()
            if key
            not in {
                "split_type",
                "split",
                "split_sizes",
                "data_seed",
                "validation_protocol",
                "validation_strategy",
                "seed_policy",
                "final_refit",
            }
        }
        backend_train_args["splits_file"] = chemprop_input["chemprop_splits_file"]
        result = self.backend.train_model(
            train_csv=chemprop_input["chemprop_training_input_csv"],
            output_dir=output_dir,
            task=task,
            extra_args=backend_train_args,
        )
        result.update(
            self._compute_training_metrics(
                train_csv=chemprop_input["chemprop_training_input_csv"],
                output_dir=output_dir,
                task=task,
                splits_file=chemprop_input["chemprop_splits_file"],
            )
        )
        if train_args.get("final_refit") and not result.get("metrics"):
            result["metrics"] = {}
            result["target_metrics"] = {}
            result["metrics_status"] = "not_evaluated"
            result["evaluation_required"] = True
        artifacts = self._resolve_chemprop_run_artifacts(run_output_dir)
        if artifacts.get("best_model_path"):
            result.setdefault("best_model_path", str(artifacts["best_model_path"]))
            result.setdefault("model_path", str(artifacts["best_model_path"]))
        if artifacts.get("config_path") and artifacts["config_path"].exists():
            result.setdefault("config_path", str(artifacts["config_path"]))
        if artifacts.get("splits_path") and artifacts["splits_path"].exists():
            result.setdefault("splits_path", str(artifacts["splits_path"]))
        if artifacts.get("test_predictions_path"):
            result.setdefault("test_predictions_path", str(artifacts["test_predictions_path"]))
        validation_predictions = self._write_validation_predictions(
            train_csv=chemprop_input["chemprop_training_input_csv"],
            output_dir=run_output_dir,
            task=task,
            splits_file=chemprop_input["chemprop_splits_file"],
            model_path=result.get("best_model_path") or result.get("model_path"),
        )
        if validation_predictions:
            result["validation_predictions_path"] = validation_predictions[
                "validation_predictions_path"
            ]
            result["raw_validation_predictions_path"] = validation_predictions.get(
                "raw_validation_predictions_path"
            )
            result["validation_prediction_input_csv"] = validation_predictions.get(
                "validation_prediction_input_csv"
            )
            if validation_predictions.get("validation_metrics"):
                result.setdefault("metrics", {})["validation"] = validation_predictions[
                    "validation_metrics"
                ]
            if validation_predictions.get("validation_target_metrics"):
                result["validation_target_metrics"] = validation_predictions[
                    "validation_target_metrics"
                ]
        result["chemprop_input"] = chemprop_input
        result["chemprop_training_input_csv"] = chemprop_input["chemprop_training_input_csv"]
        result["chemprop_splits_file"] = chemprop_input["chemprop_splits_file"]
        result["chemprop_input_manifest_path"] = chemprop_input["manifest_path"]
        result["split_payload"] = split_payload
        return result

    def _build_split_payload(
        self,
        *,
        train_csv: str,
        task: PredictionTaskSpec,
        split_type: str,
        split_sizes: List[float],
        seed: int,
    ) -> List[Dict[str, List[int]]]:
        dataset = strip_unnamed_columns(
            pd.read_csv(Path(_agent_local_path(train_csv)).expanduser())
        )
        normalized_split_sizes = normalize_json_list_argument(
            split_sizes,
            argument_name="split_sizes",
            coerce_numbers=True,
        )
        if not normalized_split_sizes or len(normalized_split_sizes) not in (2, 3):
            raise ValueError(
                "Chemprop split_sizes must contain [train, test] or [train, validation, test]."
            )
        if split_type == "kmeans":
            raise ValueError(
                "Chemprop graph training does not support cluster holdout without an explicit "
                "graph-compatible cluster plan. Use random/scaffold holdout or a tabular backend."
            )
        feature_columns = [
            column
            for column in dataset.columns
            if column not in set((task.smiles_columns or []) + (task.target_columns or []))
        ]
        return build_qsar_split_payload(
            df=dataset,
            split_type=split_type,
            split_sizes=normalized_split_sizes,
            random_state=seed,
            smiles_column=(task.smiles_columns or ["smiles"])[0],
            feature_columns=feature_columns,
        )

    def _summarize_training_resources(
        self,
        *,
        compute_env: Dict[str, Any],
        effective_train_args: Dict[str, Any],
    ) -> Dict[str, Any]:
        cpu_count = int(compute_env.get("cpu_count") or 1)
        gpu_available = bool(compute_env.get("gpu_available"))
        gpu_count = int(compute_env.get("gpu_count") or 0)
        num_workers = int(effective_train_args.get("num_workers") or 0)
        batch_size = int(effective_train_args.get("batch_size") or 0)
        ensemble_size = int(effective_train_args.get("ensemble_size") or 1)
        num_replicates = int(effective_train_args.get("num_replicates") or 1)
        epochs = int(effective_train_args.get("epochs") or 0)

        gpu_devices_requested = 1 if gpu_available and gpu_count > 0 else 0
        cpu_processes_estimated = max(1, min(cpu_count, num_workers + 1))

        return {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "ensemble_size": ensemble_size,
            "num_replicates": num_replicates,
            "epochs": epochs,
            "cpu_cores_available": cpu_count,
            "cpu_processes_estimated": cpu_processes_estimated,
            "gpu_available": gpu_available,
            "gpu_count_available": gpu_count,
            "gpu_devices_requested": gpu_devices_requested,
            "gpu_name": compute_env.get("gpu_name"),
            "memory_gb_total": compute_env.get("memory_gb_total"),
            "disk_gb_free": compute_env.get("disk_gb_free"),
            "disk_gb_total": compute_env.get("disk_gb_total"),
            "execution_env": compute_env.get("execution_env"),
        }

    def _summarize_training_durations(
        self,
        *,
        split_results: List[Dict[str, Any]],
        total_started_at: datetime,
        total_completed_at: datetime,
    ) -> Dict[str, Any]:
        return summarize_training_durations(
            split_results=split_results,
            total_started_at=total_started_at,
            total_completed_at=total_completed_at,
        )

    def _aggregate_split_families(self, split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        families: Dict[str, List[Dict[str, Any]]] = {}
        for item in split_results:
            family = item.get("strategy_family") or item.get("strategy")
            metrics = (item.get("metrics") or {}).get("test") or {}
            if not family or not metrics:
                continue
            families.setdefault(family, []).append(item)

        aggregated: Dict[str, Any] = {}
        metric_names = ("mse", "mae", "rae", "rmse", "r2", "spearman", "kendall")
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
                    {
                        "label": item.get("strategy_label"),
                        "seed": item.get("seed"),
                        "metrics": metrics,
                    }
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

    def _assess_protocol_results(self, split_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return assess_protocol_results(split_results)

    def _materialize_primary_protocol_artifacts(
        self,
        *,
        root_output_dir: Path,
        primary_output_dir: Path,
    ) -> Dict[str, Optional[str]]:
        root_output_dir.mkdir(parents=True, exist_ok=True)
        root_model_dir = root_output_dir / "model_0"
        root_model_dir.mkdir(parents=True, exist_ok=True)

        copied: Dict[str, Optional[str]] = {
            "best_model_path": None,
            "validation_predictions_path": None,
            "test_predictions_path": None,
            "config_path": None,
            "splits_path": None,
        }

        resolved_artifacts = self._resolve_chemprop_run_artifacts(primary_output_dir)
        file_map = {
            resolved_artifacts["best_model_path"]: root_model_dir / "best.pt",
            resolved_artifacts["validation_predictions_path"]: root_model_dir
            / "validation_predictions.csv",
            resolved_artifacts["test_predictions_path"]: root_model_dir / "test_predictions.csv",
            resolved_artifacts["config_path"]: root_output_dir / "config.toml",
            resolved_artifacts["splits_path"]: root_output_dir / "splits.json",
        }

        for source_path, target_path in file_map.items():
            if source_path and source_path.exists():
                if source_path.resolve() != target_path.resolve():
                    shutil.copy2(source_path, target_path)
                if target_path.name == "best.pt":
                    copied["best_model_path"] = str(target_path)
                elif target_path.name == "validation_predictions.csv":
                    copied["validation_predictions_path"] = str(target_path)
                elif target_path.name == "test_predictions.csv":
                    copied["test_predictions_path"] = str(target_path)
                elif target_path.name == "config.toml":
                    copied["config_path"] = str(target_path)
                elif target_path.name == "splits.json":
                    copied["splits_path"] = str(target_path)

        return copied

    def _build_applicability_domain(
        self,
        *,
        train_csv: str,
        primary_run: Dict[str, Any],
        primary_output_dir: Path,
        model_id_hint: str,
        task: PredictionTaskSpec,
        prediction_artifact_paths: Optional[Dict[str, Any]] = None,
        applicability_domain_methods: Optional[List[str] | str] = None,
        similarity_top_k_neighbors: int | str | None = None,
        similarity_threshold_percentile: float | str | None = None,
    ) -> Dict[str, Any]:
        artifacts = self._resolve_chemprop_run_artifacts(primary_output_dir)
        model_path = (
            primary_run.get("best_model_path")
            or primary_run.get("model_path")
            or (str(artifacts["best_model_path"]) if artifacts.get("best_model_path") else None)
        )
        chemprop_train_csv = str(primary_run.get("chemprop_training_input_csv") or train_csv)
        if model_path:
            try:
                fingerprints = self.backend.fingerprint_from_csv(
                    input_csv=chemprop_train_csv,
                    model_path=str(model_path),
                    output_csv=str(
                        primary_output_dir
                        / "applicability_domain"
                        / "chemprop_embeddings_train.csv"
                    ),
                    smiles_columns=task.smiles_columns or ["smiles"],
                    ffn_block_index=DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX,
                )
                feature_frame = pd.read_csv(fingerprints["fingerprints_path"])
                return build_applicability_domain_for_training(
                    train_csv=chemprop_train_csv,
                    primary_run=primary_run,
                    primary_output_dir=primary_output_dir,
                    task=task,
                    model_id_hint=model_id_hint,
                    feature_columns=fingerprints["feature_columns"],
                    feature_frame=feature_frame,
                    feature_space="chemprop_embedding",
                    feature_metadata={
                        "chemprop_fingerprint": {
                            "ffn_block_index": fingerprints["ffn_block_index"],
                            "feature_count": fingerprints["feature_count"],
                        }
                    },
                    prediction_artifact_paths=prediction_artifact_paths,
                    applicability_domain_methods=applicability_domain_methods,
                    similarity_top_k_neighbors=similarity_top_k_neighbors,
                    similarity_threshold_percentile=similarity_threshold_percentile,
                )
            except Exception as exc:
                return {
                    "available": False,
                    "method": "bounding_box",
                    "feature_space": "chemprop_embedding",
                    "reason": f"Chemprop embedding extraction failed: {exc}",
                }
        return build_applicability_domain_for_training(
            train_csv=train_csv,
            primary_run=primary_run,
            primary_output_dir=primary_output_dir,
            task=task,
            model_id_hint=model_id_hint,
            feature_columns=[],
            feature_space="chemprop_embedding",
            prediction_artifact_paths=prediction_artifact_paths,
            applicability_domain_methods=applicability_domain_methods,
            similarity_top_k_neighbors=similarity_top_k_neighbors,
            similarity_threshold_percentile=similarity_threshold_percentile,
        )

    def describe_backend(
        self,
        backend_name: Optional[str] = None,
        __name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Describe Chemprop backend availability and version information.

        The agent sometimes passes lightweight selector kwargs such as
        `backend_name` or `__name`. We accept and ignore them here so a simple
        backend inspection never fails on argument noise.
        """
        return self.backend.describe_environment()

    def _compute_training_metrics(
        self,
        *,
        train_csv: str,
        output_dir: str,
        task: PredictionTaskSpec,
        splits_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir).expanduser()
        target_column = task.target_columns[0] if task.target_columns else None
        if not target_column:
            return {}

        normalized_predictions = self._write_normalized_test_predictions(
            train_csv=train_csv,
            output_dir=output_path,
            task=task,
            splits_file=splits_file,
        )
        resolved_artifacts = self._resolve_chemprop_run_artifacts(output_path)
        explicit_splits_path = Path(str(splits_file)).expanduser() if splits_file else None
        splits_path = (
            explicit_splits_path
            if explicit_splits_path is not None and explicit_splits_path.exists()
            else resolved_artifacts["splits_path"]
        )
        preds_path = Path(
            str(
                normalized_predictions.get("test_predictions_path")
                or resolved_artifacts["test_predictions_path"]
            )
        )

        if (
            splits_path is None
            or preds_path is None
            or not splits_path.exists()
            or not preds_path.exists()
        ):
            return {}

        dataset = _strip_unnamed_columns(
            pd.read_csv(Path(_agent_local_path(train_csv)).expanduser())
        )
        predictions = _strip_unnamed_columns(pd.read_csv(preds_path))

        split_payload = json.loads(splits_path.read_text())
        if not split_payload or "test" not in split_payload[0]:
            return {}

        test_indices = split_payload[0]["test"]
        true_column = f"{target_column}_true"
        is_classification = is_classification_task(task.task_type)
        if true_column in predictions.columns and "prediction" in predictions.columns:
            actual_values = predictions[true_column]
            predicted_values = predictions["prediction"]
        else:
            actual = dataset.iloc[test_indices].reset_index(drop=True)
            if target_column not in actual.columns or target_column not in predictions.columns:
                return {}
            if len(actual) != len(predictions):
                return {}
            actual_values = actual[target_column]
            predicted_values = predictions[target_column]
        if is_classification:
            class_labels = None
            manifest_path = output_path / "chemprop_inputs" / "chemprop_input_manifest.json"
            classification_targets: Dict[str, Any] = {}
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    classification_targets = manifest.get("classification_targets") or {}
                    class_labels = (classification_targets.get(target_column) or {}).get(
                        "class_labels"
                    )
                except Exception:
                    class_labels = None
            metric_values = compute_classification_metrics(
                actual_values,
                predicted_values,
                class_labels=class_labels,
                positive_scores=predictions.get("positive_probability"),
                target_column=target_column,
            )
        else:
            metric_values = compute_regression_metrics(
                pd.to_numeric(actual_values, errors="coerce"),
                pd.to_numeric(predicted_values, errors="coerce"),
                target_column=target_column,
            )
        if not metric_values:
            return {}
        target_metrics: Dict[str, Any] = {target_column: metric_values}
        for extra_target in list(task.target_columns or [])[1:]:
            true_col = f"{extra_target}_true"
            pred_col = f"{extra_target}_prediction"
            if true_col not in predictions.columns or pred_col not in predictions.columns:
                continue
            if is_classification:
                target_metrics[extra_target] = compute_classification_metrics(
                    predictions[true_col],
                    predictions[pred_col],
                    class_labels=(classification_targets.get(extra_target) or {}).get(
                        "class_labels"
                    ),
                    positive_scores=predictions.get(f"{extra_target}_positive_probability"),
                    target_column=extra_target,
                )
            else:
                target_metrics[extra_target] = compute_regression_metrics(
                    pd.to_numeric(predictions[true_col], errors="coerce"),
                    pd.to_numeric(predictions[pred_col], errors="coerce"),
                    target_column=extra_target,
                )

        return {
            "best_model_path": (
                str(resolved_artifacts["best_model_path"])
                if resolved_artifacts.get("best_model_path")
                else None
            ),
            "test_predictions_path": str(preds_path),
            "raw_test_prediction_paths": normalized_predictions.get("raw_test_prediction_paths")
            or [],
            "splits_path": str(splits_path),
            "train_size": len(split_payload[0].get("train") or []),
            "val_size": len(
                split_payload[0].get("val") or split_payload[0].get("validation") or []
            ),
            "test_size": len(test_indices),
            "replicate_artifacts": normalized_predictions.get("replicate_artifacts")
            or self._replicate_artifacts(output_path),
            "replicate_count": normalized_predictions.get("replicate_count") or 1,
            "detected_replicate_count": normalized_predictions.get("detected_replicate_count"),
            "excluded_replicate_count": normalized_predictions.get("excluded_replicate_count"),
            "prediction_aggregation": normalized_predictions.get("prediction_aggregation")
            or "single_replicate",
            "replicate_alignment_policy": normalized_predictions.get("replicate_alignment_policy"),
            "prediction_column": normalized_predictions.get("prediction_column") or target_column,
            "target_true_column": normalized_predictions.get("target_true_column") or target_column,
            "target_prediction_column": normalized_predictions.get("target_prediction_column")
            or target_column,
            "metrics": {"test": metric_values},
            "target_metrics": target_metrics,
        }

    def train_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_columns: Optional[List[str] | str] = None,
        target_columns: Optional[List[str] | str] = None,
        reaction_columns: Optional[List[str] | str] = None,
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        validation_strategy: Optional[Dict[str, Any]] = None,
        applicability_domain_methods: Optional[List[str] | str] = None,
        similarity_top_k_neighbors: int | str | None = None,
        similarity_threshold_percentile: float | str | None = None,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Launch Chemprop training and persist a lightweight training record."""
        source_train_csv = _agent_storage_path(train_csv)
        local_train_csv = _agent_local_path(train_csv)
        smiles_columns = normalize_json_list_argument(
            smiles_columns,
            argument_name="smiles_columns",
        )
        target_columns = normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        )
        reaction_columns = normalize_json_list_argument(
            reaction_columns,
            argument_name="reaction_columns",
        )

        resolved_output_dir = str(Path(output_dir).expanduser().resolve())
        root_output_path = Path(resolved_output_dir)
        active_marker_path = root_output_path / ".training_in_progress"
        trained_at = project_now()
        cleaned_extra_args, extra_activity_args = split_activity_cliff_args(extra_args)
        requested_validation_strategy = (
            validation_strategy
            if validation_strategy is not None
            else cleaned_extra_args.pop("validation_strategy", None)
        )
        requested_ad_methods = (
            applicability_domain_methods
            if applicability_domain_methods is not None
            else cleaned_extra_args.pop("applicability_domain_methods", None)
        )
        requested_similarity_top_k = (
            similarity_top_k_neighbors
            if similarity_top_k_neighbors is not None
            else cleaned_extra_args.pop("similarity_top_k_neighbors", None)
        )
        requested_similarity_percentile = (
            similarity_threshold_percentile
            if similarity_threshold_percentile is not None
            else cleaned_extra_args.pop("similarity_threshold_percentile", None)
        )
        activity_args = {
            "activity_cliff_index": activity_cliff_index,
            "activity_cliff_feedback": activity_cliff_feedback,
            "activity_cliff_feedback_loops": activity_cliff_feedback_loops,
            "activity_cliff_similarity_threshold": activity_cliff_similarity_threshold,
            "activity_cliff_top_k_neighbors": activity_cliff_top_k_neighbors,
            "activity_cliff_flag_threshold": activity_cliff_flag_threshold,
            **extra_activity_args,
        }
        training_policy = self._apply_training_profile(cleaned_extra_args)
        protocol_policy = self._resolve_validation_protocol(
            requested_protocol=training_policy.get("validation_protocol"),
            training_profile=training_policy["training_profile"],
            seed_policy=training_policy["extra_args"].get("seed_policy"),
            base_seed=training_policy["extra_args"].get("data_seed")
            or training_policy["extra_args"].get("random_state"),
            validation_strategy=requested_validation_strategy,
        )
        protocol_override_note = self._apply_protocol_training_overrides(
            training_policy=training_policy,
            protocol_policy=protocol_policy,
        )
        training_policy["extra_args"]["data_seed"] = protocol_policy["seed_policy"]["model_seed"]
        task = PredictionTaskSpec(
            task_type=task_type,
            smiles_columns=smiles_columns or ["smiles"],
            target_columns=target_columns or [],
            reaction_columns=reaction_columns or [],
        )
        activity_cliffs: Dict[str, Any] = {}
        if task.task_type == "regression" and len(task.target_columns) == 1:
            try:
                activity_cliffs = prepare_activity_cliff_context(
                    train_csv=local_train_csv,
                    output_dir=resolved_output_dir,
                    smiles_column=task.smiles_columns[0] if task.smiles_columns else "smiles",
                    target_column=task.target_columns[0],
                    **activity_args,
                )
            except Exception as exc:
                if activity_args.get("activity_cliff_index") != "sali":
                    raise
                activity_cliffs = {
                    "enabled": False,
                    "mode": "skipped",
                    "index_name": activity_args.get("activity_cliff_index", "sali"),
                    "warnings": [f"Activity-cliff annotation skipped: {exc}"],
                }
        prediction_state = None
        qsar_training_state = None
        active_run_record = {
            "status": "running",
            "train_csv": source_train_csv,
            "output_dir": resolved_output_dir,
            "validation_protocol": protocol_policy["protocol"],
            "training_profile": training_policy["training_profile"],
            "created_at": trained_at.isoformat(),
            "active_marker_path": str(active_marker_path),
            "current_split_label": None,
        }

        if agent is not None:
            prediction_state = get_prediction_state(agent)
            prediction_state["active_training_run"] = dict(active_run_record)
            qsar_training_state = agent.session_state.setdefault("qsar_training", {})
            qsar_training_state["active_run"] = dict(active_run_record)

        write_active_training_marker(active_marker_path, active_run_record)

        split_results: List[Dict[str, Any]] = []
        primary_run: Optional[Dict[str, Any]] = None
        primary_output_dir: Optional[Path] = None
        total_started_at = project_now()

        multi_run_protocol = len(protocol_policy["split_runs"]) > 1
        with S3.open(source_train_csv, "r") as fh:
            split_source_df = strip_unnamed_columns(pd.read_csv(fh))
        is_cv_protocol = protocol_policy.get("validation_strategy_type") == "cross_validation"
        cv_split_payloads: Dict[str, List[Dict[str, Any]]] = {}
        if is_cv_protocol:
            cv_strategy = protocol_policy.get("validation_strategy") or {}
            cv_split_payloads = build_repeated_kfold_split_payloads(
                df=split_source_df,
                n_splits=int(cv_strategy.get("n_folds") or cv_strategy.get("n_splits") or 5),
                n_repeats=int(cv_strategy.get("n_repeats") or 1),
                random_state=int(
                    cv_strategy.get("seed") or protocol_policy["seed_policy"].get("model_seed") or 0
                ),
            )

        try:
            for split_run in protocol_policy["split_runs"]:
                label = split_run["label"]
                run_output_dir = (
                    root_output_path / f"{safe_slug(label)}_split"
                    if multi_run_protocol
                    else root_output_path
                )
                run_args = {
                    **{
                        key: value
                        for key, value in training_policy["extra_args"].items()
                        if key != "seed_policy"
                    },
                    "split_type": split_run["backend_split_type"],
                    "split_sizes": split_run.get("split_sizes")
                    or training_policy["extra_args"].get("split_sizes"),
                    "data_seed": split_run["seed"],
                }
                if label in cv_split_payloads:
                    split_payload = cv_split_payloads[label]
                elif split_run["backend_split_type"] == "final_refit":
                    split_payload = build_full_train_split_payload(df=split_source_df)
                    run_args["split_sizes"] = [1.0]
                    run_args["final_refit"] = True
                else:
                    split_payload = self._build_split_payload(
                        train_csv=local_train_csv,
                        task=task,
                        split_type=split_run["backend_split_type"],
                        split_sizes=run_args["split_sizes"],
                        seed=int(split_run["seed"]),
                    )

                active_run_record["current_split_label"] = label
                if prediction_state is not None:
                    prediction_state["active_training_run"] = dict(active_run_record)
                if qsar_training_state is not None:
                    qsar_training_state["active_run"] = dict(active_run_record)
                write_active_training_marker(active_marker_path, active_run_record)

                single_result = self._train_single_run(
                    train_csv=local_train_csv,
                    task=task,
                    output_dir=str(run_output_dir),
                    train_args=run_args,
                    split_payload=split_payload,
                    split_label=label,
                    seed=split_run["seed"],
                )
                if label.startswith("cv_repeat_"):
                    strategy_name = label
                    strategy_family = "cross_validation"
                elif "scaffold" in label:
                    strategy_name = "scaffold"
                    strategy_family = "scaffold"
                elif "kmeans" in label or "cluster" in label:
                    strategy_name = "cluster_kmeans"
                    strategy_family = "cluster_kmeans"
                elif "kennard" in label:
                    strategy_name = "distance_kennard_stone"
                    strategy_family = "distance_kennard_stone"
                elif "random_seed_" in label:
                    strategy_name = label
                    strategy_family = "random"
                else:
                    strategy_name = "random"
                    strategy_family = "random"
                single_result["strategy"] = strategy_name
                single_result["strategy_family"] = strategy_family
                single_result["strategy_label"] = label
                single_result["backend_split_type"] = split_run["backend_split_type"]
                single_result["seed"] = split_run["seed"]
                single_result["repeat_index"] = split_run.get("repeat_index")
                single_result["fold_index"] = split_run.get("fold_index")
                single_result["n_folds"] = split_run.get("n_folds")
                single_result["n_repeats"] = split_run.get("n_repeats")
                single_result["output_dir"] = str(run_output_dir)
                single_result["validation_protocol"] = protocol_policy["protocol"]
                single_result["split_payload"] = split_payload
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
                    primary_output_dir = run_output_dir

            if primary_run is None or primary_output_dir is None:
                raise ValueError("Training protocol did not produce a primary run.")

            cross_validation_artifacts: Dict[str, Any] = {}
            final_refit_run: Optional[Dict[str, Any]] = None
            final_refit_output_dir: Optional[Path] = None
            if is_cv_protocol and task.target_columns:
                cross_validation_artifacts = build_cross_validation_artifacts(
                    split_results=split_results,
                    output_dir=root_output_path / "cross_validation",
                    target_column=task.target_columns[0],
                )
            if is_cv_protocol and protocol_policy.get("final_refit", True):
                final_refit_output_dir = root_output_path / "final_refit"
                final_split_payload = build_full_train_split_payload(df=split_source_df)
                final_args = {
                    **{
                        key: value
                        for key, value in training_policy["extra_args"].items()
                        if key != "seed_policy"
                    },
                    "split_type": "final_refit",
                    "split_sizes": [1.0],
                    "data_seed": protocol_policy["seed_policy"]["model_seed"],
                    "final_refit": True,
                }
                final_refit_run = self._train_single_run(
                    train_csv=local_train_csv,
                    task=task,
                    output_dir=str(final_refit_output_dir),
                    train_args=final_args,
                    split_payload=final_split_payload,
                    split_label="final_refit",
                    seed=protocol_policy["seed_policy"].get("model_seed"),
                )
                final_refit_run["strategy"] = "final_refit"
                final_refit_run["strategy_family"] = "final_refit"
                final_refit_run["strategy_label"] = "final_refit"
                final_refit_run["backend_split_type"] = "final_refit"
                final_refit_run["seed"] = protocol_policy["seed_policy"].get("model_seed")
                final_refit_run["output_dir"] = str(final_refit_output_dir)
                final_refit_run["validation_protocol"] = protocol_policy["protocol"]
                final_refit_run["split_payload"] = final_split_payload

            final_primary_run = final_refit_run or primary_run
            final_primary_output_dir = final_refit_output_dir or primary_output_dir

            if prediction_state is not None:
                prediction_state["training_runs"].append(
                    {
                        "train_csv": source_train_csv,
                        "output_dir": resolved_output_dir,
                        "task_type": task_type,
                        "smiles_columns": task.smiles_columns,
                        "target_columns": task.target_columns,
                        "validation_protocol": protocol_policy["protocol"],
                        "seed_policy": protocol_policy["seed_policy"],
                        "split_runs": [
                            {
                                "label": item["strategy_label"],
                                "strategy": item["strategy"],
                                "strategy_family": item.get("strategy_family"),
                                "output_dir": item["output_dir"],
                                "seed": item["seed"],
                            }
                            for item in split_results
                        ],
                    }
                )

            root_artifacts = self._materialize_primary_protocol_artifacts(
                root_output_dir=root_output_path,
                primary_output_dir=final_primary_output_dir,
            )
            validation_assessment = self._assess_protocol_results(split_results)
            ad_summary = self._build_applicability_domain(
                train_csv=local_train_csv,
                primary_run=final_primary_run,
                primary_output_dir=final_primary_output_dir,
                model_id_hint=Path(resolved_output_dir).name,
                task=task,
                prediction_artifact_paths={
                    "validation": root_artifacts.get("validation_predictions_path"),
                "test": root_artifacts.get("test_predictions_path"),
            },
            applicability_domain_methods=requested_ad_methods,
            similarity_top_k_neighbors=requested_similarity_top_k,
            similarity_threshold_percentile=requested_similarity_percentile,
        )
            plot_artifacts: Dict[str, str] = {}
            target_column = task.target_columns[0] if task.target_columns else None
            if protocol_policy.get("validation_strategy_type") != "full_train":
                plot_artifacts = build_training_plots_if_possible(
                    train_csv=local_train_csv,
                    split_results=split_results,
                    primary_run=final_primary_run,
                    root_artifacts=root_artifacts,
                    root_output_dir=root_output_path,
                    target_column=target_column,
                    task_type=task.task_type,
                )

            result = dict(final_primary_run)
            result["backend_name"] = self.backend.backend_name
            result["output_dir"] = resolved_output_dir
            result["validation_protocol"] = protocol_policy["protocol"]
            if protocol_policy.get("validation_strategy_type") == "full_train":
                result["metrics"] = {}
                result["target_metrics"] = {}
                result["validation_predictions_path"] = None
                result["test_predictions_path"] = None
                result["test_predictions_file_ref"] = None
                result["metrics_status"] = "not_evaluated"
                result["evaluation_required"] = True
            result["validation_protocol_reason"] = protocol_policy["reason"]
            result["validation_strategy"] = protocol_policy.get("validation_strategy")
            result["validation_strategy_type"] = protocol_policy.get("validation_strategy_type")
            result["validation_aggregation"] = protocol_policy.get("aggregation")
            result["selection_metric"] = protocol_policy.get("selection_metric")
            result["final_refit"] = protocol_policy.get("final_refit")
            result["seed_policy"] = protocol_policy["seed_policy"]
            result["seed_policy_report"] = seed_policy_reporting_text(
                protocol_policy["seed_policy"]
            )
            result["reproducibility"] = seed_policy_reproducibility_metadata(
                protocol_policy["seed_policy"]
            )
            result["split_results"] = split_results
            result["cross_validation"] = cross_validation_artifacts
            result["cv_artifacts"] = cross_validation_artifacts
            result["final_refit_result"] = final_refit_run
            result["catalog_model_policy"] = (
                "final_refit_only_fold_models_are_artifacts"
                if is_cv_protocol
                else result.get("catalog_model_policy")
            )
            result["validation_assessment"] = validation_assessment
            result["compute_environment"] = training_policy["compute_environment"]
            result["training_profile"] = training_policy["training_profile"]
            result["profile_reason"] = training_policy["profile_reason"]
            result["effective_train_args"] = {
                key: value
                for key, value in training_policy["extra_args"].items()
                if key != "seed_policy"
            }
            result["effective_train_args"]["model_seed"] = protocol_policy["seed_policy"].get(
                "model_seed"
            )
            if primary_run.get("seed") is not None:
                result["effective_train_args"]["data_seed"] = primary_run.get("seed")
                result["effective_train_args"]["data_seed_scope"] = "primary_split"
            result["replicate_policy"] = {
                "num_replicates_requested": int(
                    result["effective_train_args"].get("num_replicates") or 1
                ),
                "protocol_override_note": protocol_override_note,
                "prediction_aggregation": primary_run.get("prediction_aggregation"),
                "catalog_primary_replicate_index": 0,
                "catalog_primary_model_policy": (
                    "The catalog model artifact points to replicate_0/model_0/best.pt. "
                    "Validation prediction CSVs aggregate only replicate outputs whose SMILES "
                    "exactly align to the split test rows; non-aligned Chemprop replicate outputs "
                    "are recorded but excluded from validation metrics."
                ),
                "split_replicate_counts": [
                    {
                        "label": item.get("strategy_label"),
                        "strategy_family": item.get("strategy_family"),
                        "replicate_count": item.get("replicate_count"),
                        "detected_replicate_count": item.get("detected_replicate_count"),
                        "excluded_replicate_count": item.get("excluded_replicate_count"),
                        "prediction_aggregation": item.get("prediction_aggregation"),
                        "replicate_alignment_policy": item.get("replicate_alignment_policy"),
                    }
                    for item in split_results
                ],
            }
            result["training_resources"] = self._summarize_training_resources(
                compute_env=training_policy["compute_environment"],
                effective_train_args=result["effective_train_args"],
            )
            total_completed_at = project_now()
            result["training_durations"] = self._summarize_training_durations(
                split_results=split_results,
                total_started_at=total_started_at,
                total_completed_at=total_completed_at,
            )
            result["applicability_domain"] = ad_summary
            result["activity_cliffs"] = activity_cliffs
            result["plot_artifacts"] = plot_artifacts
            result["trained_at"] = trained_at.isoformat()
            result["trained_date"] = trained_at.strftime("%d/%m/%Y")
            result["trained_time"] = trained_at.strftime("%H:%M:%S")

            training_summary_path = Path(resolved_output_dir) / "cs_copilot_training_summary.json"

            resolved_primary_artifacts = self._resolve_chemprop_run_artifacts(
                Path(resolved_output_dir)
            )
            best_model_path = Path(
                root_artifacts.get("best_model_path")
                or resolved_primary_artifacts.get("best_model_path")
                or (Path(resolved_output_dir) / "model_0" / "best.pt")
            )
            config_path = Path(
                root_artifacts.get("config_path") or resolved_primary_artifacts["config_path"]
            )
            splits_path = Path(
                root_artifacts.get("splits_path") or resolved_primary_artifacts["splits_path"]
            )
            result["summary_path"] = str(training_summary_path)
            if best_model_path.exists():
                result["best_model_path"] = str(best_model_path)
                result["model_path"] = str(best_model_path)
                result["download_file_ref"] = str(best_model_path)
            result["summary_file_ref"] = str(training_summary_path)
            if root_artifacts.get("validation_predictions_path"):
                result["validation_predictions_path"] = root_artifacts[
                    "validation_predictions_path"
                ]
            elif primary_run.get("validation_predictions_path"):
                result["validation_predictions_path"] = primary_run["validation_predictions_path"]
            if root_artifacts.get("test_predictions_path"):
                result["test_predictions_file_ref"] = root_artifacts["test_predictions_path"]
                result["test_predictions_path"] = root_artifacts["test_predictions_path"]
            elif primary_run.get("test_predictions_path"):
                result["test_predictions_file_ref"] = primary_run["test_predictions_path"]
                result["test_predictions_path"] = primary_run["test_predictions_path"]
            bundle_path = (
                Path(".files")
                / "prediction_outputs"
                / f"{Path(resolved_output_dir).name}_training_bundle.zip"
            ).resolve()
            bundle_files = collect_training_bundle_files(
                train_csv=local_train_csv,
                summary_path=training_summary_path,
                result={
                    **result,
                    "best_model_path": str(best_model_path),
                    "config_path": str(config_path),
                    "splits_path": str(splits_path),
                },
                split_results=split_results,
                ad_summary=ad_summary,
                plot_artifacts=plot_artifacts,
                activity_cliffs=activity_cliffs,
                extra_files=[Path(resolved_output_dir)],
            )
            bundle = bundle_artifacts(
                bundle_path,
                bundle_files,
            )
            result["bundle_file_ref"] = str(bundle)
            result["training_bundle"] = str(bundle)
            result["bundle_download_tag"] = f"<file>{bundle}</file>"
            write_training_summary(training_summary_path, result)
            return result
        except Exception as exc:
            active_run_record["status"] = "failed"
            active_run_record["error"] = str(exc)
            if prediction_state is not None:
                prediction_state["active_training_run"] = dict(active_run_record)
            if qsar_training_state is not None:
                qsar_training_state["active_run"] = dict(active_run_record)
            write_active_training_marker(active_marker_path, active_run_record)
            raise
        finally:
            if active_marker_path.exists():
                active_marker_path.unlink()
            if prediction_state is not None:
                prediction_state["active_training_run"] = None
            if qsar_training_state is not None:
                qsar_training_state["active_run"] = None
