#!/usr/bin/env python
# coding: utf-8
"""
Internal toolkit for Chemprop-specific QSAR training flows.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import shutil
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

from cs_copilot.storage.client import S3
from cs_copilot.tools.activity_cliffs import (
    prepare_activity_cliff_context,
    split_activity_cliff_args,
)

from .backend import PredictionModelRecord, PredictionTaskSpec
from .chemprop_adapter import (
    materialize_chemprop_inputs,
    normalize_chemprop_classification_predictions,
)
from .chemprop_backend import DEFAULT_CHEMPROP_FINGERPRINT_FFN_BLOCK_INDEX, ChempropBackend
from .hyperparameter_tuning import (
    HYPERPARAMETER_CONTRACT_VERSION,
    ChempropHpoptAdapter,
    HyperparameterTuningError,
    normalize_tuning_config,
    tuning_metadata_for_catalog,
)
from .outlier_analysis import (
    attach_activity_cliff_annotations,
    attach_ad_annotations,
    normalize_outlier_analysis_config,
    select_outliers,
    selection_predictions_from_frame,
    write_outlier_analysis_artifacts,
    write_outlier_variant_comparison,
)
from .qsar_progress import apply_progress_update
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
    build_classification_prediction_frame,
    build_cross_validation_artifacts,
    build_training_plots_if_possible,
    collect_training_bundle_files,
    compute_classification_metrics,
    compute_regression_metrics,
    decode_classification_labels,
    is_classification_task,
    is_multiclass_task,
    normalize_json_list_argument,
    strip_unnamed_columns,
    write_training_summary,
)

logger = logging.getLogger(__name__)


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
        is_classification = is_classification_task(task.task_type)
        is_multiclass = is_multiclass_task(task.task_type)
        classification_targets: Dict[str, Any] = {}
        if is_classification:
            manifest_path = output_path / "chemprop_inputs" / "chemprop_input_manifest.json"
            if manifest_path.exists():
                try:
                    classification_targets = (
                        json.loads(manifest_path.read_text()).get("classification_targets") or {}
                    )
                except Exception:
                    classification_targets = {}
        prediction_series_by_target: Dict[str, List[pd.Series]] = {
            column: [] for column in target_columns
        }
        probability_frames_by_target: Dict[str, List[pd.DataFrame]] = {
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
            if is_classification:
                try:
                    predictions = normalize_chemprop_classification_predictions(
                        predictions,
                        task=task,
                        classification_targets=classification_targets,
                    )
                except Exception as exc:
                    artifact["aligned_for_validation"] = False
                    artifact["exclusion_reason"] = f"invalid_classification_predictions: {exc}"
                    normalized_replicate_artifacts.append(artifact)
                    continue
            replicate_index = item.get("replicate_index")
            if not isinstance(replicate_index, int):
                replicate_index = len(included_replicate_indices)
            artifact["aligned_for_validation"] = True
            artifact["exclusion_reason"] = None
            normalized_replicate_artifacts.append(artifact)
            for column_index, column in enumerate(target_columns):
                if is_multiclass:
                    prefix = "probability_" if column_index == 0 else f"{column}_probability_"
                    probability_columns = [
                        name for name in predictions.columns if str(name).startswith(prefix)
                    ]
                    probability_frames_by_target[column].append(
                        predictions[probability_columns].apply(pd.to_numeric, errors="coerce")
                    )
                    prediction_series_by_target[column].append(
                        pd.to_numeric(
                            predictions[
                                (
                                    "prediction_class_code"
                                    if column_index == 0
                                    else f"{column}_prediction_class_code"
                                )
                            ],
                            errors="coerce",
                        )
                    )
                elif is_classification:
                    probability_column = (
                        "positive_probability"
                        if column_index == 0
                        else f"{column}_positive_probability"
                    )
                    prediction_series_by_target[column].append(
                        pd.to_numeric(predictions[probability_column], errors="coerce")
                    )
                else:
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
            class_info: Dict[str, Any] = classification_targets.get(target_column) or {}
            class_labels = list(class_info.get("class_labels") or [0, 1])
            if is_multiclass:
                probability_frames = probability_frames_by_target[target_column]
                mean_probabilities = sum(probability_frames[1:], probability_frames[0].copy())
                mean_probabilities = mean_probabilities / float(len(probability_frames))
                predicted_codes = pd.Series(mean_probabilities.to_numpy().argmax(axis=1), dtype=int)
                selected_probabilities = pd.DataFrame(
                    {
                        replicate_index: [
                            frame.iloc[row_index, int(code)]
                            for row_index, code in enumerate(predicted_codes)
                        ]
                        for replicate_index, frame in enumerate(probability_frames)
                    }
                )
                prediction_std = selected_probabilities.std(axis=1, ddof=0).fillna(0.0)
                probabilities = mean_probabilities
            else:
                positive_probability = y_pred_numeric.clip(lower=0.0, upper=1.0)
                predicted_codes = (positive_probability >= 0.5).astype(int)
                probabilities = pd.DataFrame(
                    {0: 1.0 - positive_probability, 1: positive_probability}
                )
            true_labels = decode_classification_labels(
                y_true_numeric.fillna(-1).astype(int).tolist(), class_labels
            )
            normalized = pd.DataFrame(
                {
                    "source_row_index": test_indices,
                    "smiles": smiles_values,
                    f"{target_column}_true": true_labels,
                    "true_class_code": y_true_numeric,
                    "prediction_std": prediction_std,
                    "replicate_count": len(prediction_series),
                    "detected_replicate_count": len(replicate_artifacts),
                }
            )
            normalized = pd.concat(
                [
                    normalized,
                    build_classification_prediction_frame(
                        predicted_codes=predicted_codes,
                        class_labels=class_labels,
                        target_column=target_column,
                        probabilities=probabilities,
                        primary_target=True,
                    ),
                ],
                axis=1,
            )
            for extra_target in target_columns[1:]:
                extra_series = prediction_series_by_target.get(extra_target) or []
                if not extra_series:
                    continue
                extra_frame = pd.concat(extra_series, axis=1)
                extra_labels = list(
                    (classification_targets.get(extra_target) or {}).get("class_labels") or [0, 1]
                )
                if is_multiclass:
                    extra_probability_frames = probability_frames_by_target[extra_target]
                    extra_probabilities = sum(
                        extra_probability_frames[1:], extra_probability_frames[0].copy()
                    ) / float(len(extra_probability_frames))
                    extra_codes = pd.Series(
                        extra_probabilities.to_numpy().argmax(axis=1), dtype=int
                    )
                else:
                    extra_positive_probability = extra_frame.mean(axis=1, skipna=True).clip(
                        lower=0.0, upper=1.0
                    )
                    extra_probabilities = pd.DataFrame(
                        {0: 1.0 - extra_positive_probability, 1: extra_positive_probability}
                    )
                    extra_codes = (extra_positive_probability >= 0.5).astype(int)
                normalized[f"{extra_target}_true"] = decode_classification_labels(
                    pd.to_numeric(actual[extra_target], errors="coerce")
                    .fillna(-1)
                    .astype(int)
                    .tolist(),
                    extra_labels,
                )
                normalized = pd.concat(
                    [
                        normalized,
                        build_classification_prediction_frame(
                            predicted_codes=extra_codes,
                            class_labels=extra_labels,
                            target_column=extra_target,
                            probabilities=extra_probabilities,
                            primary_target=False,
                        ),
                    ],
                    axis=1,
                )
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

        is_classification = is_classification_task(task.task_type)
        classification_targets: Dict[str, Any] = {}
        manifest_path = output_path / "chemprop_inputs" / "chemprop_input_manifest.json"
        if is_classification and manifest_path.exists():
            try:
                classification_targets = (
                    json.loads(manifest_path.read_text()).get("classification_targets") or {}
                )
            except Exception:
                classification_targets = {}

        record = PredictionModelRecord(
            model_id=f"{output_path.name}_validation",
            backend_name=self.backend.backend_name,
            model_path=str(model_artifact),
            task=task,
            inference_profile={"classification_targets": classification_targets},
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
            predicted_codes = pd.to_numeric(predictions["prediction_class_code"], errors="coerce")
            if is_multiclass_task(task.task_type):
                probability_columns = [
                    column
                    for column in predictions.columns
                    if str(column).startswith("probability_")
                ]
                probabilities = predictions[probability_columns]
                positive_probability = None
            else:
                positive_probability = pd.to_numeric(
                    predictions["positive_probability"], errors="coerce"
                ).clip(lower=0.0, upper=1.0)
                probabilities = pd.DataFrame(
                    {0: 1.0 - positive_probability, 1: positive_probability}
                )
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
                    "true_class_code": true_codes,
                    "replicate_count": 1,
                    "detected_replicate_count": 1,
                }
            )
            normalized = pd.concat(
                [
                    normalized,
                    build_classification_prediction_frame(
                        predicted_codes=predicted_codes,
                        class_labels=class_labels,
                        target_column=target_column,
                        probabilities=probabilities,
                        primary_target=True,
                    ),
                ],
                axis=1,
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
                extra_codes = pd.to_numeric(
                    predictions[f"{extra_target}_prediction_class_code"], errors="coerce"
                )
                if is_multiclass_task(task.task_type):
                    extra_probability_columns = [
                        column
                        for column in predictions.columns
                        if str(column).startswith(f"{extra_target}_probability_")
                    ]
                    extra_probabilities = predictions[extra_probability_columns]
                    extra_probability = None
                else:
                    extra_probability = pd.to_numeric(
                        predictions[f"{extra_target}_positive_probability"], errors="coerce"
                    ).clip(lower=0.0, upper=1.0)
                    extra_probabilities = pd.DataFrame(
                        {0: 1.0 - extra_probability, 1: extra_probability}
                    )
                extra_true_codes = pd.to_numeric(actual[extra_target], errors="coerce")
                extra_true = decode_classification_labels(
                    extra_true_codes.fillna(-1).astype(int).tolist(), extra_labels
                )
                extra_pred = decode_classification_labels(extra_codes.tolist(), extra_labels)
                normalized[f"{extra_target}_true"] = extra_true
                normalized = pd.concat(
                    [
                        normalized,
                        build_classification_prediction_frame(
                            predicted_codes=extra_codes,
                            class_labels=extra_labels,
                            target_column=extra_target,
                            probabilities=extra_probabilities,
                            primary_target=False,
                        ),
                    ],
                    axis=1,
                )
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
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        run_output_dir = Path(output_dir).expanduser().resolve()
        chemprop_input = materialize_chemprop_inputs(
            source_csv=_agent_local_path(train_csv),
            output_dir=run_output_dir / "chemprop_inputs",
            task=task,
            split_payload=split_payload,
            split_label=split_label,
            seed=seed,
            allow_empty_test=bool(train_args.get("final_refit")),
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
        classification_targets = chemprop_input.get("classification_targets") or {}
        if is_multiclass_task(task.task_type):
            class_counts = {
                int(metadata.get("class_count") or 0)
                for metadata in classification_targets.values()
            }
            if len(class_counts) != 1 or 0 in class_counts:
                raise ValueError(
                    "Chemprop multiclass training requires one shared positive class count."
                )
            backend_train_args["multiclass_num_classes"] = class_counts.pop()
        backend_kwargs: Dict[str, Any] = {
            "train_csv": chemprop_input["chemprop_training_input_csv"],
            "output_dir": output_dir,
            "task": task,
            "extra_args": backend_train_args,
        }
        if progress_callback is not None:
            backend_kwargs["progress_callback"] = progress_callback
        result = self.backend.train_model(**backend_kwargs)
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
        if classification_targets:
            result["classification_targets"] = classification_targets
            primary_target = task.target_columns[0]
            primary_metadata = classification_targets.get(primary_target) or {}
            for key in (
                "task_kind",
                "class_labels",
                "class_count",
                "label_mapping",
                "positive_class_label",
            ):
                if primary_metadata.get(key) is not None:
                    result[key] = primary_metadata[key]
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
        feature_columns = [
            column
            for column in dataset.columns
            if column not in set((task.smiles_columns or []) + (task.target_columns or []))
            and pd.api.types.is_numeric_dtype(dataset[column])
        ]
        if split_type == "kmeans" and not feature_columns:
            smiles_column = (task.smiles_columns or ["smiles"])[0]
            if smiles_column not in dataset.columns:
                raise ValueError("Chemprop cluster holdout requires a valid SMILES column.")

            def _cluster_descriptors(smiles: Any) -> List[float]:
                molecule = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
                if molecule is None:
                    return [0.0] * 8
                return [
                    float(Descriptors.MolWt(molecule)),
                    float(Descriptors.MolLogP(molecule)),
                    float(Descriptors.TPSA(molecule)),
                    float(Lipinski.NumHDonors(molecule)),
                    float(Lipinski.NumHAcceptors(molecule)),
                    float(Lipinski.RingCount(molecule)),
                    float(Lipinski.NumRotatableBonds(molecule)),
                    float(Descriptors.FractionCSP3(molecule)),
                ]

            cluster_columns = [f"__qsaria_cluster_descriptor_{index}" for index in range(8)]
            descriptor_frame = pd.DataFrame(
                [_cluster_descriptors(value) for value in dataset[smiles_column]],
                columns=cluster_columns,
                index=dataset.index,
            )
            dataset = pd.concat([dataset, descriptor_frame], axis=1)
            feature_columns = cluster_columns
        return build_qsar_split_payload(
            df=dataset,
            split_type=split_type,
            split_sizes=normalized_split_sizes,
            random_state=seed,
            smiles_column=(task.smiles_columns or ["smiles"])[0],
            feature_columns=feature_columns,
        )

    @staticmethod
    def _hpo_development_payload(
        split_payload: List[Dict[str, List[int]]],
    ) -> tuple[List[int], List[Dict[str, List[int]]]]:
        """Return an isolated train/validation data set for Chemprop HPO.

        The returned source-row indices deliberately contain no test index.
        Keeping this transformation here makes the test exclusion auditable in
        both the transient manifest and the final tuning summary.
        """
        if not split_payload:
            raise HyperparameterTuningError("Chemprop tuning requires a holdout split payload.")
        split_map = split_payload[0]
        train_indices = [int(index) for index in split_map.get("train") or []]
        validation_indices = [
            int(index) for index in (split_map.get("val") or split_map.get("validation") or [])
        ]
        test_indices = [int(index) for index in split_map.get("test") or []]
        if not train_indices or not validation_indices or not test_indices:
            raise HyperparameterTuningError(
                "Chemprop tuning V1 requires non-empty train, validation, and test holdout sets."
            )
        development_indices = [*train_indices, *validation_indices]
        return development_indices, [
            {
                "train": list(range(len(train_indices))),
                "val": list(range(len(train_indices), len(development_indices))),
                "test": [],
            }
        ]

    @staticmethod
    def _extract_hpopt_parameters(
        hpopt_output_dir: Path,
        requested_parameters: List[str],
        backend_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Read the winning native Chemprop configuration without retaining it.

        Chemprop has used both TOML and JSON names across releases.  We accept
        the documented forms and then keep only Qsaria's declared architecture
        parameters.  No candidate checkpoint or Ray state crosses this method.
        """

        def select_declared_parameters(payload: Dict[str, Any]) -> Dict[str, Any]:
            # `chemprop hpopt` serializes its CLI configuration using kebab-case
            # (`message-hidden-dim`), whereas Qsaria's public contract uses
            # Python-style snake_case (`message_hidden_dim`).
            normalized = {str(key).replace("-", "_"): value for key, value in payload.items()}
            return {name: normalized[name] for name in requested_parameters if name in normalized}

        for key in ("best_params", "best_parameters", "best_hyperparameters"):
            value = backend_result.get(key)
            if isinstance(value, dict):
                selected = select_declared_parameters(value)
                if selected:
                    return selected

        def load_candidate(path: Path) -> Optional[Dict[str, Any]]:
            """Read either strict TOML/JSON or ConfigArgParse's native config.

            Chemprop calls ConfigArgParse's ``write_config_file`` for
            ``best_config.toml``.  That output is a flat ``key = value``
            configuration file and can contain unquoted paths/lists, so its
            filename does not guarantee strict TOML syntax.  The fallback only
            reads declared scalar/list values; it never evaluates arbitrary
            code or imports the temporary configuration.
            """
            try:
                with path.open("rb") as fh:
                    payload = tomllib.load(fh) if path.suffix == ".toml" else json.load(fh)
            except (OSError, ValueError, tomllib.TOMLDecodeError):
                payload = None
            if isinstance(payload, dict):
                return payload

            if path.suffix != ".toml":
                return None
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return None

            flat_payload: Dict[str, Any] = {}
            for raw_line in lines:
                line = raw_line.strip()
                if not line or line.startswith(("#", ";")) or "=" not in line:
                    continue
                raw_key, raw_value = line.split("=", 1)
                key = raw_key.strip()
                if not key or not key.replace("_", "").replace("-", "").isalnum():
                    continue
                value = raw_value.strip()
                try:
                    flat_payload[key] = ast.literal_eval(value)
                except (SyntaxError, ValueError):
                    # The native format permits bare strings such as data
                    # paths.  They are irrelevant to selection but retaining
                    # them lets the same parser handle all flat config fields.
                    flat_payload[key] = value.strip("\"'")
            return flat_payload or None

        explicit_best_config = backend_result.get("best_config_path")
        candidates: List[Path] = []
        if isinstance(explicit_best_config, (str, Path)):
            hinted_path = Path(explicit_best_config).expanduser()
            if hinted_path.is_file():
                candidates.append(hinted_path)

        discovered_candidates = sorted(
            [
                *hpopt_output_dir.rglob("*best*.toml"),
                *hpopt_output_dir.rglob("*best*.json"),
                *hpopt_output_dir.rglob("*hyperparameter*.toml"),
                *hpopt_output_dir.rglob("*hyperparameter*.json"),
            ],
            key=lambda path: (len(path.parts), path.name),
        )
        for path in discovered_candidates:
            if path not in candidates:
                candidates.append(path)

        unreadable_candidates: List[str] = []
        for path in candidates:
            payload = load_candidate(path)
            if not isinstance(payload, dict):
                unreadable_candidates.append(path.name)
                continue
            containers = [payload]
            for key in ("best_config", "best_params", "best_hyperparameters", "config"):
                nested = payload.get(key)
                if isinstance(nested, dict):
                    containers.append(nested)
            for item in containers:
                selected = select_declared_parameters(item)
                if selected:
                    return selected
        requested_display = ", ".join(requested_parameters)
        checked_display = ", ".join(path.name for path in candidates) or "no candidate config file"
        unreadable_display = (
            f" Unreadable files: {', '.join(unreadable_candidates)}."
            if unreadable_candidates
            else ""
        )
        raise HyperparameterTuningError(
            "Chemprop hpopt completed, but its winning configuration did not contain the "
            f"requested Qsaria architecture parameters ({requested_display}). Checked: "
            f"{checked_display}.{unreadable_display} No final model was trained, so the "
            "test set remains untouched."
        )

    def _run_chemprop_hpopt(
        self,
        *,
        source_df: pd.DataFrame,
        task: PredictionTaskSpec,
        split_payload: List[Dict[str, List[int]]],
        config: Any,
        fixed_parameters: Dict[str, Any],
        train_args: Dict[str, Any],
        output_dir: Path,
        seed: int,
        compute_environment: Optional[Mapping[str, Any]] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """Execute native Chemprop HPO on train/validation only and clean it up."""
        ChempropHpoptAdapter().validate(config)
        output_dir.mkdir(parents=True, exist_ok=True)
        if config.search_space:
            raise HyperparameterTuningError(
                "Chemprop native hpopt V1 owns its `basic` search space; custom search_space "
                "is not yet supported."
            )
        resolved_compute_environment = dict(
            compute_environment or self.describe_compute_environment()
        )
        gpu_available = bool(resolved_compute_environment.get("gpu_available"))
        gpu_count = int(resolved_compute_environment.get("gpu_count") or 0)
        development_indices, development_split = self._hpo_development_payload(split_payload)
        with tempfile.TemporaryDirectory(prefix="qsaria-chemprop-hpopt-") as temporary_dir:
            temporary_path = Path(temporary_dir)
            development_source = temporary_path / "development_source.csv"
            source_df.iloc[development_indices].reset_index(drop=True).to_csv(
                development_source,
                index=False,
            )
            chemprop_input = materialize_chemprop_inputs(
                source_csv=str(development_source),
                output_dir=temporary_path / "inputs",
                task=task,
                split_payload=development_split,
                split_label="hyperparameter_selection",
                seed=seed,
                allow_empty_test=True,
            )
            hpopt_args = {
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
            hpopt_args.update(
                {
                    "splits_file": chemprop_input["chemprop_splits_file"],
                    "raytune_num_samples": config.n_trials,
                    "raytune_search_algorithm": "hyperopt",
                    "raytune_trial_scheduler": "FIFO",
                    "raytune_num_workers": 1,
                    "raytune_max_concurrent_trials": 1,
                    "hyperopt_random_state_seed": seed,
                    "tracking_metric": "val_loss",
                    # Chemprop accepts every member of `basic` as an individual
                    # keyword.  Passing only the non-fixed names makes a direct
                    # user value genuinely fixed throughout the HPO campaign.
                    "search_parameter_keywords": list(config.parameters),
                    "data_seed": seed,
                }
            )
            requested_accelerator = str(hpopt_args.get("accelerator") or "").strip().lower()
            explicit_cpu = requested_accelerator == "cpu"
            use_gpu = gpu_available and gpu_count > 0 and not explicit_cpu
            if use_gpu:
                # Chemprop's native HPO uses Ray Train.  Lightning's accelerator alone is
                # insufficient: Ray must reserve a GPU for its worker, otherwise it starts
                # CPU-only trials even when Qsaria has detected CUDA successfully.
                hpopt_args["raytune_use_gpu"] = True
                hpopt_args["raytune_num_gpus"] = 1
                if requested_accelerator in {"", "auto"}:
                    hpopt_args["accelerator"] = "gpu"
                if hpopt_args.get("devices") in {None, "", "auto"}:
                    hpopt_args["devices"] = 1
            else:
                # A direct CPU request is authoritative, and a CPU-only machine must never
                # carry stale Ray GPU resource settings into the Chemprop command.
                hpopt_args.pop("raytune_use_gpu", None)
                hpopt_args.pop("raytune_num_gpus", None)
            hpopt_execution = {
                "gpu_available": gpu_available,
                "gpu_count_available": gpu_count,
                "raytune_use_gpu": use_gpu,
                "raytune_num_gpus": 1 if use_gpu else 0,
                "accelerator": hpopt_args.get("accelerator") or "auto",
                "devices": hpopt_args.get("devices") or "auto",
            }
            backend_kwargs: Dict[str, Any] = {
                "train_csv": chemprop_input["chemprop_training_input_csv"],
                "output_dir": str(temporary_path / "ray_output"),
                "task": task,
                "extra_args": hpopt_args,
            }
            if progress_callback is not None:
                backend_kwargs["progress_callback"] = progress_callback
            backend_result = self.backend.hpopt_model(**backend_kwargs)
            best_parameters = self._extract_hpopt_parameters(
                temporary_path,
                list(config.parameters),
                backend_result,
            )
            best_parameters = {**fixed_parameters, **best_parameters}

        summary = {
            "contract_version": HYPERPARAMETER_CONTRACT_VERSION,
            "backend_name": "chemprop",
            "engine": "chemprop_hpopt_hyperopt",
            "engine_version": getattr(self.backend, "_package_version", lambda: None)(),
            "status": "completed",
            "seed": seed,
            "objective": config.objective.as_dict(),
            "requested_trials": config.n_trials,
            "completed_trials": config.n_trials,
            "failed_trials": 0,
            "best_trial": {
                "params": best_parameters,
                "objective": "val_loss",
            },
            "selection_protocol": {
                "tracking_metric": "val_loss",
                "search_parameter_keywords": list(config.parameters),
                "basic_architecture_space": True,
                "raytune_search_algorithm": "hyperopt",
                "raytune_trial_scheduler": "FIFO",
                "raytune_num_workers": 1,
                "raytune_max_concurrent_trials": 1,
                "execution_resources": hpopt_execution,
                "hyperopt_random_state_seed": seed,
                "test_rows_provided_to_hpopt": 0,
                "applicability_domain_used_for_selection": False,
            },
            "note": (
                "Chemprop selection used its global native val_loss. Applicability-domain diagnostics "
                "are computed only for the final refit and never rank a Chemprop HPO trial."
            ),
        }
        summary_path = output_dir / "hyperparameter_tuning_summary.json"
        summary["summary_path"] = str(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        return summary

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

    def _run_outlier_refits(
        self,
        *,
        train_csv: str,
        source_df: pd.DataFrame,
        task: PredictionTaskSpec,
        split_payload: List[Dict[str, Any]],
        selection_run: Dict[str, Any],
        output_dir: Path,
        train_args: Dict[str, Any],
        selected_parameters: Dict[str, Any],
        activity_args: Dict[str, Any],
        applicability_domain_methods: Optional[List[str] | str],
        similarity_top_k_neighbors: int | str | None,
        similarity_threshold_percentile: float | str | None,
        selection_fraction: float,
        fold_label: str,
        repeat_index: Optional[int] = None,
        baseline_run: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, Optional[Dict[str, Any]]], None]] = None,
    ) -> Dict[str, Any]:
        """Apply the shared outlier policy to one Chemprop validation fit.

        Chemprop supplies its own temporary prediction CSV and embedding-based
        AD.  The common module still owns row alignment, eligibility, APE
        ordering and artifacts, so the scientific decision matches tabular
        backends exactly.
        """
        split = split_payload[0]
        train_indices = [int(value) for value in split.get("train") or []]
        validation_indices = [
            int(value) for value in (split.get("val") or split.get("validation") or [])
        ]
        test_indices = [int(value) for value in split.get("test") or []]
        if not train_indices or not validation_indices:
            raise ValueError("Outlier analysis requires a non-empty train and validation split.")
        target_column = task.target_columns[0]
        validation_path = selection_run.get("validation_predictions_path")
        if not validation_path or not Path(str(validation_path)).exists():
            raise ValueError("Chemprop selection fit did not produce validation predictions.")
        selection = selection_predictions_from_frame(
            pd.read_csv(Path(str(validation_path))),
            source_row_indices=validation_indices,
            target_column=target_column,
            fold_label=fold_label,
            repeat_index=repeat_index,
        )
        analysis_dir = output_dir / "outlier_analysis"
        development_indices = [*train_indices, *validation_indices]
        development_frame = source_df.iloc[development_indices].copy()
        development_frame.index = development_indices

        # This selection model was fitted only on its train indices.  Reuse its
        # temporary embedding AD to score the validation rows, never the test.
        selection_ad = self._build_applicability_domain(
            train_csv=train_csv,
            primary_run=selection_run,
            primary_output_dir=Path(str(selection_run.get("output_dir") or output_dir)),
            model_id_hint="chemprop_outlier_selection",
            task=task,
            prediction_artifact_paths={"validation": validation_path},
            applicability_domain_methods=applicability_domain_methods,
            similarity_top_k_neighbors=similarity_top_k_neighbors,
            similarity_threshold_percentile=similarity_threshold_percentile,
        )
        statuses: Optional[pd.Series] = None
        scores_path = selection_ad.get("scores_validation_path")
        if scores_path and Path(str(scores_path)).exists():
            scores = pd.read_csv(Path(str(scores_path)))
            if "ad_status" in scores.columns and len(scores) == len(validation_indices):
                statuses = scores["ad_status"]
        selection = attach_ad_annotations(
            selection,
            ad_statuses=statuses.tolist() if statuses is not None else None,
        )

        ac_annotations: Optional[pd.DataFrame] = None
        if task.task_type == "regression":
            ac_input = development_frame.drop(columns=["source_row_index"], errors="ignore").copy()
            ac_input.insert(0, "source_row_index", ac_input.index)
            ac_source = analysis_dir / "development_for_selection.csv"
            ac_source.parent.mkdir(parents=True, exist_ok=True)
            ac_input.to_csv(ac_source, index=False)
            try:
                ac_context = prepare_activity_cliff_context(
                    train_csv=str(ac_source),
                    output_dir=str(analysis_dir / "activity_cliffs_development"),
                    smiles_column=task.smiles_columns[0] if task.smiles_columns else "smiles",
                    target_column=target_column,
                    **activity_args,
                )
                annotated_path = ac_context.get("annotated_training_csv")
                if annotated_path and Path(str(annotated_path)).exists():
                    ac_annotations = pd.read_csv(annotated_path)
            except Exception as exc:
                logger.warning("Development-only Activity Cliff annotations unavailable: %s", exc)
        selection = attach_activity_cliff_annotations(selection, annotations=ac_annotations)
        selected_rows, selection_summary = select_outliers(
            selection,
            task_type=task.task_type,
            selection_fraction=selection_fraction,
        )
        artifacts = write_outlier_analysis_artifacts(
            output_dir=analysis_dir,
            selection_frame=selected_rows,
            selection_summary=selection_summary,
            development_frame=development_frame,
            extra_summary={
                "fold_label": fold_label,
                "repeat_index": repeat_index,
                "selection_model": "temporary_train_only_fit",
                "activity_cliff_scope": "development_only",
                "test_rows_used_for_selection": 0,
            },
        )
        selected_indices = [
            int(value)
            for value in selected_rows.loc[
                selected_rows["selected_for_removal"].astype(bool), "source_row_index"
            ].tolist()
        ]

        def _fit_variant(variant_id: str, train_indices_for_variant: List[int]) -> Dict[str, Any]:
            variant_payload = [
                {
                    "train": train_indices_for_variant,
                    **({"test": test_indices} if test_indices else {}),
                    "metadata": {
                        **dict(split.get("metadata") or {}),
                        "refit_on_train_validation": True,
                        "outlier_variant": variant_id,
                        "removed_source_row_indices": (
                            selected_indices if variant_id == "outlier_filtered" else []
                        ),
                    },
                }
            ]
            run = self._train_single_run(
                train_csv=train_csv,
                task=task,
                output_dir=str(output_dir / "outlier_variants" / variant_id),
                train_args={
                    **train_args,
                    **selected_parameters,
                    "split_type": "final_refit",
                    "split_sizes": [1.0] if not test_indices else [0.9, 0.1],
                    "final_refit": True,
                },
                split_payload=variant_payload,
                split_label=variant_id,
                seed=int(train_args.get("data_seed") or train_args.get("random_state") or 0),
            )
            run["outlier_variant"] = variant_id
            run["outlier_selected_count"] = len(selected_indices)
            run["split_payload"] = variant_payload
            return run

        variants: List[Dict[str, Any]] = []
        if baseline_run is None:
            if progress_callback is not None:
                progress_callback("Training baseline refit", {"detail": "train + validation"})
            baseline_run = _fit_variant("baseline", development_indices)
        baseline_run = dict(baseline_run)
        baseline_run["outlier_variant"] = "baseline"
        baseline_run["outlier_selected_count"] = len(selected_indices)
        variants.append({"variant_id": "baseline", "run": baseline_run})
        if selected_indices:
            filtered_indices = [
                value for value in development_indices if value not in set(selected_indices)
            ]
            try:
                if progress_callback is not None:
                    progress_callback(
                        "Training filtered refit",
                        {"detail": f"{len(selected_indices)} rows removed"},
                    )
                variants.append(
                    {
                        "variant_id": "outlier_filtered",
                        "run": _fit_variant("outlier_filtered", filtered_indices),
                    }
                )
            except Exception as exc:
                variants.append(
                    {"variant_id": "outlier_filtered", "status": "failed", "reason": str(exc)}
                )
        comparison_path = write_outlier_variant_comparison(
            output_dir=analysis_dir,
            variants=variants,
            selected_count=len(selected_indices),
        )
        return {
            "summary": {
                **selection_summary,
                "enabled": True,
                "artifacts": artifacts,
                "summary_path": artifacts.get("summary_path"),
                "selection_predictions_path": artifacts.get("selection_predictions_path"),
                "filtered_development_path": artifacts.get("filtered_development_path"),
                "plot_artifacts": {
                    key: value
                    for key, value in artifacts.items()
                    if key.startswith("outlier_selection_")
                },
                "comparison_path": comparison_path,
                "test_comparison_policy": "descriptive_only_no_automatic_winner",
                "selected_source_row_indices": selected_indices,
                "variants": [
                    {
                        "variant_id": item["variant_id"],
                        "status": item.get("status", "completed"),
                        "reason": item.get("reason"),
                    }
                    for item in variants
                ],
            },
            "variants": variants,
        }

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
        hyperparameter_tuning: Optional[Dict[str, Any]] = None,
        outlier_analysis: Optional[Dict[str, Any]] = None,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
        bundle_path: Optional[str] = None,
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
        requested_hyperparameter_tuning = (
            hyperparameter_tuning
            if hyperparameter_tuning is not None
            else cleaned_extra_args.pop("hyperparameter_tuning", None)
        )
        requested_outlier_analysis = (
            outlier_analysis
            if outlier_analysis is not None
            else cleaned_extra_args.pop("outlier_analysis", None)
        )
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
        has_validation = bool(protocol_policy.get("validation_strategy_type") == "cross_validation")
        if not has_validation:
            has_validation = any(
                len(run.get("split_sizes") or []) == 3
                for run in protocol_policy.get("split_runs") or []
            )
        outlier_config, outlier_skip_reason = normalize_outlier_analysis_config(
            requested_outlier_analysis,
            has_validation=has_validation,
            target_count=len(target_columns or []),
            activity_cliff_feedback=bool(activity_args.get("activity_cliff_feedback")),
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
        if (
            is_classification_task(task.task_type)
            and training_policy["extra_args"].get("metric") == "rmse"
        ):
            training_policy["extra_args"].pop("metric")
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
            "backend_name": "chemprop",
            "train_csv": source_train_csv,
            "output_dir": resolved_output_dir,
            "validation_protocol": protocol_policy["protocol"],
            "training_profile": training_policy["training_profile"],
            "created_at": trained_at.isoformat(),
            "active_marker_path": str(active_marker_path),
            "current_split_label": None,
            "phase": "Preparing Chemprop training",
            "progress_message": None,
        }

        def publish_active_progress(phase: str, payload: Optional[Dict[str, Any]] = None) -> None:
            """Persist compact Chemprop telemetry for the shared QSAR status card."""
            apply_progress_update(active_run_record, phase, payload)
            if prediction_state is not None:
                prediction_state["active_training_run"] = dict(active_run_record)
            if qsar_training_state is not None:
                qsar_training_state["active_run"] = dict(active_run_record)
            write_active_training_marker(active_marker_path, active_run_record)

        def publish_chemprop_progress(payload: Dict[str, Any]) -> None:
            """Project native Chemprop and Ray telemetry onto the shared status card."""

            phase = str(active_run_record.get("phase") or "Training model")
            if payload.get("event") == "ray_tune_trial_status":
                candidate_index = payload.get("candidate_index")
                total_trials = payload.get("total_trials")
                if phase != "Native hyperparameter optimization" or candidate_index is None:
                    return
                detail = (
                    f"Candidate {candidate_index} of {total_trials}"
                    if total_trials
                    else f"Candidate {candidate_index}"
                )
                publish_active_progress(
                    phase,
                    {
                        "detail": detail,
                        "candidate_index": candidate_index,
                        "total_trials": total_trials,
                        "completed_trials": payload.get("completed_trials"),
                        "failed_trials": payload.get("failed_trials"),
                    },
                )
                return

            epoch = payload.get("epoch")
            total_epochs = payload.get("total_epochs")
            if epoch is None:
                return
            if phase == "Native hyperparameter optimization":
                candidate_index = active_run_record.get("candidate_index")
                total_trials = active_run_record.get("total_trials")
                candidate_prefix = (
                    f"Candidate {candidate_index} of {total_trials} — "
                    if candidate_index is not None and total_trials
                    else "Current candidate — "
                )
                detail = (
                    f"{candidate_prefix}epoch {epoch} of {total_epochs}"
                    if total_epochs
                    else f"{candidate_prefix}epoch {epoch}"
                )
            else:
                detail = f"epoch {epoch} of {total_epochs}" if total_epochs else f"epoch {epoch}"
            publish_active_progress(
                phase,
                {"detail": detail, "epoch": epoch, "total_epochs": total_epochs},
            )

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
                outer_test_size=cv_strategy.get("outer_test_size"),
            )

        tunable_architecture_parameters = {
            "depth",
            "message_hidden_dim",
            "ffn_hidden_dim",
            "ffn_num_layers",
            "dropout",
        }
        direct_architecture_parameters = {
            key: value
            for key, value in cleaned_extra_args.items()
            if key in tunable_architecture_parameters
        }
        hpo_eligible = has_validation
        tuning_config = normalize_tuning_config(
            requested_hyperparameter_tuning,
            backend_name="chemprop",
            task_type=task.task_type,
            eligible=hpo_eligible,
            fixed_parameters=direct_architecture_parameters,
        )
        tuning_summary: Optional[Dict[str, Any]] = None

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

                selection_split_payload = split_payload
                selected_parameters: Dict[str, Any] = {}
                selection_train_args = dict(run_args)
                if tuning_config is not None:
                    if tuning_config.parameters:
                        publish_active_progress(
                            "Native hyperparameter optimization",
                            {"detail": f"{tuning_config.n_trials} candidates requested"},
                        )
                        tuning_summary = self._run_chemprop_hpopt(
                            source_df=split_source_df,
                            task=task,
                            split_payload=split_payload,
                            config=tuning_config,
                            fixed_parameters=direct_architecture_parameters,
                            train_args=run_args,
                            output_dir=root_output_path,
                            seed=int(tuning_config.seed or split_run["seed"]),
                            compute_environment=training_policy["compute_environment"],
                            progress_callback=publish_chemprop_progress,
                        )
                        selected_parameters = dict(
                            (tuning_summary.get("best_trial") or {}).get("params") or {}
                        )
                        selection_train_args.update(selected_parameters)
                    else:
                        tuning_summary = {
                            "contract_version": HYPERPARAMETER_CONTRACT_VERSION,
                            "backend_name": "chemprop",
                            "engine": "chemprop_hpopt_hyperopt",
                            "status": "skipped",
                            "objective": tuning_config.objective.as_dict(),
                            "requested_trials": tuning_config.n_trials,
                            "completed_trials": 0,
                            "failed_trials": 0,
                            "reason": "All selected Chemprop tuning parameters were fixed directly.",
                            "selection_protocol": {
                                "tracking_metric": "val_loss",
                                "applicability_domain_used_for_selection": False,
                                "test_rows_provided_to_hpopt": 0,
                            },
                        }
                        summary_path = root_output_path / "hyperparameter_tuning_summary.json"
                        tuning_summary["summary_path"] = str(summary_path)
                        summary_path.write_text(json.dumps(tuning_summary, indent=2) + "\n")
                        selected_parameters = {}

                    base_split = split_payload[0]
                    split_payload = [
                        {
                            "train": [
                                *[int(index) for index in base_split.get("train") or []],
                                *[
                                    int(index)
                                    for index in (
                                        base_split.get("val") or base_split.get("validation") or []
                                    )
                                ],
                            ],
                            "test": [int(index) for index in base_split.get("test") or []],
                        }
                    ]
                    run_args.update(selected_parameters)
                    run_args["split_sizes"] = [0.9, 0.1]
                    run_args["final_refit"] = True
                    label = "hyperparameter_final_refit"

                active_run_record["current_split_label"] = label
                publish_active_progress(
                    (
                        "Refitting final model"
                        if label == "hyperparameter_final_refit"
                        else "Training model"
                    ),
                    {
                        "detail": (
                            "train + validation" if label == "hyperparameter_final_refit" else label
                        )
                    },
                )

                single_result = self._train_single_run(
                    train_csv=local_train_csv,
                    task=task,
                    output_dir=str(run_output_dir),
                    train_args=run_args,
                    split_payload=split_payload,
                    split_label=label,
                    seed=split_run["seed"],
                    progress_callback=publish_chemprop_progress,
                )
                publish_active_progress("Evaluating final test set", {})
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
                single_result["selection_split_payload"] = selection_split_payload
                single_result["selected_hyperparameters"] = selected_parameters
                single_result["selection_train_args"] = selection_train_args
                split_results.append(single_result)

                if split_run.get("primary") or primary_run is None:
                    primary_run = single_result
                    primary_output_dir = run_output_dir

            outlier_study: Dict[str, Any] = {
                "enabled": bool(outlier_config.enabled and outlier_skip_reason is None),
                "status": "skipped" if outlier_skip_reason else "pending",
                "reason": outlier_skip_reason,
                "config": outlier_config.as_dict(),
                "studies": [],
            }
            outlier_variants: List[Dict[str, Any]] = []
            if outlier_config.enabled and outlier_skip_reason is None:
                for split_result in split_results:
                    selection_payload = split_result.get("selection_split_payload") or []
                    selection_split = selection_payload[0] if selection_payload else {}
                    validation_indices = (
                        selection_split.get("val") or selection_split.get("validation") or []
                    )
                    if not validation_indices:
                        outlier_study["studies"].append(
                            {
                                "split_label": split_result.get("strategy_label"),
                                "status": "skipped",
                                "reason": "No validation rows are available for selection.",
                            }
                        )
                        continue
                    publish_active_progress(
                        "Identifying validation outliers",
                        {"detail": str(split_result.get("strategy_label") or "holdout")},
                    )
                    selection_run = split_result
                    selection_path = selection_run.get("validation_predictions_path")
                    # Native hpopt returns only the winning parameter set, not
                    # a durable best-trial model. Fit that selected architecture
                    # once on train-only data solely to obtain row-level
                    # validation predictions and train-only AD scores.
                    if not selection_path or not Path(str(selection_path)).exists():
                        with tempfile.TemporaryDirectory(
                            prefix="qsaria_chemprop_outlier_selection_"
                        ) as temporary_dir:
                            selection_run = self._train_single_run(
                                train_csv=local_train_csv,
                                task=task,
                                output_dir=temporary_dir,
                                train_args=dict(split_result.get("selection_train_args") or {}),
                                split_payload=selection_payload,
                                split_label="outlier_selection",
                                seed=split_result.get("seed"),
                                progress_callback=publish_chemprop_progress,
                            )
                            study = self._run_outlier_refits(
                                train_csv=local_train_csv,
                                source_df=split_source_df,
                                task=task,
                                split_payload=selection_payload,
                                selection_run=selection_run,
                                output_dir=Path(
                                    str(split_result.get("output_dir") or root_output_path)
                                ),
                                train_args=dict(split_result.get("selection_train_args") or {}),
                                selected_parameters=dict(
                                    split_result.get("selected_hyperparameters") or {}
                                ),
                                activity_args=activity_args,
                                applicability_domain_methods=requested_ad_methods,
                                similarity_top_k_neighbors=requested_similarity_top_k,
                                similarity_threshold_percentile=requested_similarity_percentile,
                                selection_fraction=outlier_config.selection_fraction,
                                fold_label=str(split_result.get("strategy_label") or "holdout"),
                                repeat_index=split_result.get("repeat_index"),
                                baseline_run=split_result if tuning_config is not None else None,
                                progress_callback=publish_active_progress,
                            )
                    else:
                        study = self._run_outlier_refits(
                            train_csv=local_train_csv,
                            source_df=split_source_df,
                            task=task,
                            split_payload=selection_payload,
                            selection_run=selection_run,
                            output_dir=Path(
                                str(split_result.get("output_dir") or root_output_path)
                            ),
                            train_args=dict(split_result.get("selection_train_args") or {}),
                            selected_parameters=dict(
                                split_result.get("selected_hyperparameters") or {}
                            ),
                            activity_args=activity_args,
                            applicability_domain_methods=requested_ad_methods,
                            similarity_top_k_neighbors=requested_similarity_top_k,
                            similarity_threshold_percentile=requested_similarity_percentile,
                            selection_fraction=outlier_config.selection_fraction,
                            fold_label=str(split_result.get("strategy_label") or "holdout"),
                            repeat_index=split_result.get("repeat_index"),
                            baseline_run=None,
                            progress_callback=publish_active_progress,
                        )
                    study_summary = dict(study["summary"])
                    study_summary["split_label"] = split_result.get("strategy_label")
                    outlier_study["studies"].append(study_summary)
                    for variant in study["variants"]:
                        run = variant.get("run")
                        if isinstance(run, dict):
                            run_output_dir = Path(str(run.get("output_dir") or root_output_path))
                            run["applicability_domain"] = self._build_applicability_domain(
                                train_csv=local_train_csv,
                                primary_run=run,
                                primary_output_dir=run_output_dir,
                                model_id_hint=f"{Path(resolved_output_dir).name}_{variant.get('variant_id')}",
                                task=task,
                                prediction_artifact_paths={
                                    "test": run.get("test_predictions_path")
                                },
                                applicability_domain_methods=requested_ad_methods,
                                similarity_top_k_neighbors=requested_similarity_top_k,
                                similarity_threshold_percentile=requested_similarity_percentile,
                            )
                        variant["split_label"] = split_result.get("strategy_label")
                        variant["repeat_index"] = split_result.get("repeat_index")
                        variant["variant_id"] = (
                            f"{safe_slug(str(split_result.get('strategy_label') or 'holdout'))}_"
                            f"{variant.get('variant_id')}"
                        )
                        outlier_variants.append(variant)
                if outlier_variants:
                    outlier_study["status"] = "completed"
                elif not outlier_study["studies"]:
                    outlier_study["status"] = "skipped"
                    outlier_study["reason"] = "Selection validation predictions are unavailable."

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
                cv_outer_test_indices = list(
                    ((next(iter(cv_split_payloads.values()), [{}]) or [{}])[0]).get("test") or []
                )
                final_split_payload = build_full_train_split_payload(
                    df=split_source_df,
                    test_indices=cv_outer_test_indices or None,
                )
                final_args = {
                    **{
                        key: value
                        for key, value in training_policy["extra_args"].items()
                        if key != "seed_policy"
                    },
                    "split_type": "final_refit",
                    "split_sizes": [1.0] if not cv_outer_test_indices else [0.9, 0.1],
                    "data_seed": protocol_policy["seed_policy"]["model_seed"],
                    "final_refit": True,
                }
                publish_active_progress("Refitting final model", {"detail": "all training rows"})
                final_refit_run = self._train_single_run(
                    train_csv=local_train_csv,
                    task=task,
                    output_dir=str(final_refit_output_dir),
                    train_args=final_args,
                    split_payload=final_split_payload,
                    split_label="final_refit",
                    seed=protocol_policy["seed_policy"].get("model_seed"),
                    progress_callback=publish_chemprop_progress,
                )
                publish_active_progress("Evaluating final test set", {})
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
            if outlier_variants and not is_cv_protocol:
                primary_label = primary_run.get("strategy_label")
                baseline_variant = next(
                    (
                        item
                        for item in outlier_variants
                        if item.get("split_label") == primary_label
                        and str(item.get("variant_id", "")).endswith("_baseline")
                        and isinstance(item.get("run"), dict)
                    ),
                    None,
                )
                if baseline_variant is not None:
                    final_primary_run = baseline_variant["run"]
                    final_primary_output_dir = Path(
                        str(final_primary_run.get("output_dir") or final_primary_output_dir)
                    )

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
            publish_active_progress("Assessing applicability domain", {})
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
            if tuning_summary is not None:
                result["hyperparameter_tuning"] = tuning_summary
                result["hyperparameter_tuning_metadata"] = tuning_metadata_for_catalog(
                    tuning_summary
                )
                result["selection_validation"] = {
                    "label": "best Chemprop native hpopt trial",
                    "objective": tuning_summary.get("objective"),
                    "best_trial": tuning_summary.get("best_trial"),
                    "applicability_domain_used": False,
                }
                result["test_final"] = {
                    "label": "single refit on train + validation",
                    "metrics": result.get("metrics") or {},
                }
            result["activity_cliffs"] = activity_cliffs
            result["plot_artifacts"] = {
                **plot_artifacts,
                **{
                    key: value
                    for study in outlier_study.get("studies") or []
                    for key, value in (study.get("plot_artifacts") or {}).items()
                },
            }
            result["outlier_analysis"] = outlier_study
            result["outlier_model_variants"] = outlier_variants
            if outlier_variants:
                result["catalog_model_policy"] = "outlier_variants_no_test_winner"
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
            bundle_destination = (
                Path(bundle_path).expanduser()
                if bundle_path
                else Path(".files")
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
                bundle_destination,
                bundle_files,
            )
            result["bundle_file_ref"] = str(bundle)
            result["training_bundle"] = str(bundle)
            result["bundle_download_tag"] = f"<file>{bundle}</file>"
            publish_active_progress("Writing training artifacts", {})
            write_training_summary(training_summary_path, result)
            active_run_record["status"] = "completed"
            active_run_record["completed_at"] = project_now().isoformat()
            if prediction_state is not None:
                prediction_state["active_training_run"] = dict(active_run_record)
            if qsar_training_state is not None:
                qsar_training_state["active_run"] = dict(active_run_record)
            write_active_training_marker(active_marker_path, active_run_record)
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
