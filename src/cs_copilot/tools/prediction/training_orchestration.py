#!/usr/bin/env python
# coding: utf-8
"""Shared orchestration helpers for QSAR training toolkits.

This module intentionally keeps backend-specific training logic out of the
common layer. It centralizes the small, repeated orchestration pieces that all
training backends need: argument normalization, profile application, primary
artifact materialization, applicability-domain construction, plotting, summary
writing, and bundle file collection.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from .ad_builder import build_applicability_domain_from_training_data as build_legacy_similarity_ad
from .applicability_domain import (
    QSAR_ROW_ID_COLUMN,
    append_ad_scores_to_csv,
    build_modern_ad_plots,
    fit_modern_applicability_domain,
    metrics_by_ad_status,
    score_modern_applicability_domain,
)
from .backend import PredictionTaskSpec
from .qsar_plots import build_qsar_training_plots
from .qsar_training_policy import describe_compute_environment, resolve_training_profile

CLASSIFICATION_TASK_TYPES = {
    "classification",
    "binary_classification",
    "multiclass",
    "multiclass_classification",
}
MULTICLASS_TASK_TYPES = {"multiclass", "multiclass_classification"}


def strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without CSV index columns such as ``Unnamed: 0``."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


def normalize_task_type(task_type: str) -> str:
    normalized = str(task_type or "").strip().lower()
    if normalized in {"binary_classification", "classification"}:
        return "classification"
    if normalized in MULTICLASS_TASK_TYPES:
        return "multiclass_classification"
    return normalized or "regression"


def is_classification_task(task_type: str) -> bool:
    return str(task_type or "").strip().lower() in CLASSIFICATION_TASK_TYPES


def is_multiclass_task(task_type: str) -> bool:
    return str(task_type or "").strip().lower() in MULTICLASS_TASK_TYPES


def classification_task_kind(task_type: str, class_count: Optional[int] = None) -> str:
    if is_multiclass_task(task_type) or (class_count is not None and class_count > 2):
        return "multiclass_classification"
    if is_classification_task(task_type):
        return "binary_classification"
    return "regression"


def normalize_classification_label(value: Any) -> Any:
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
    return value


def label_key(value: Any) -> str:
    return json.dumps(normalize_classification_label(value), sort_keys=True, default=str)


def json_safe_label(value: Any) -> Any:
    value = normalize_classification_label(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sort_class_labels(labels: Sequence[Any]) -> List[Any]:
    unique = {label_key(label): normalize_classification_label(label) for label in labels}
    values = list(unique.values())
    if len(values) == 2:
        negative_tokens = {"0", "false", "f", "no", "n", "inactive", "negative", "neg", "control"}
        positive_tokens = {"1", "true", "t", "yes", "y", "active", "positive", "pos"}

        def polarity(value: Any) -> Optional[int]:
            token = str(json_safe_label(value)).strip().lower()
            if token in negative_tokens:
                return 0
            if token in positive_tokens:
                return 1
            return None

        polarities = [polarity(value) for value in values]
        if set(polarities) == {0, 1}:
            return [
                value
                for _, value in sorted(
                    zip(polarities, values, strict=False), key=lambda item: item[0]
                )
            ]
    return sorted(values, key=lambda value: (str(type(value)), str(value)))


def resolve_class_labels(series: pd.Series) -> List[Any]:
    labels = [normalize_classification_label(value) for value in series.tolist()]
    return sort_class_labels([label for label in labels if label is not None])


def encode_classification_labels(
    series: pd.Series,
    class_labels: Sequence[Any],
) -> Tuple[pd.Series, Dict[str, int]]:
    mapping = {label_key(label): index for index, label in enumerate(class_labels)}

    def _encode(value: Any) -> Optional[int]:
        key = label_key(value)
        return mapping.get(key)

    encoded = series.map(_encode)
    return encoded, {str(json_safe_label(label)): index for index, label in enumerate(class_labels)}


def decode_classification_labels(codes: Sequence[Any], class_labels: Sequence[Any]) -> pd.Series:
    labels = list(class_labels)

    def _decode(code: Any) -> Any:
        try:
            index = int(code)
        except (TypeError, ValueError):
            return None
        return json_safe_label(labels[index]) if 0 <= index < len(labels) else None

    return pd.Series([_decode(code) for code in codes])


def normalize_json_list_argument(
    value: Optional[Sequence[Any] | str],
    *,
    argument_name: str,
    coerce_numbers: bool = False,
    allow_scalar: bool = True,
    allow_comma_separated: bool = True,
) -> Optional[List[Any]]:
    """Normalize a list-like tool argument.

    Agents often pass list arguments as real lists, JSON strings, scalar
    strings, or comma-separated strings. This helper accepts those safe forms
    and raises a clear error for unsupported shapes.
    """
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed: Any = json.loads(stripped)
        except json.JSONDecodeError:
            if allow_comma_separated and "," in stripped:
                items = [item.strip() for item in stripped.split(",") if item.strip()]
            elif allow_scalar:
                items = [stripped]
            else:
                raise ValueError(
                    f"{argument_name} must be a list or a JSON-encoded list."
                ) from None
        else:
            if isinstance(parsed, list):
                items = parsed
            elif allow_scalar and isinstance(parsed, (str, int, float, bool)):
                items = [parsed]
            else:
                raise ValueError(
                    f"{argument_name} must be a list, scalar string, or JSON-encoded list."
                )
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError(f"{argument_name} must be a list, scalar string, or JSON-encoded list.")

    if coerce_numbers:
        return [float(item) for item in items]
    return items


TrainingDefaultsProvider = Callable[[str], Dict[str, Any]]
TrainingProfileLimiter = Callable[[str, Dict[str, Any], bool], Dict[str, Any]]


def apply_training_profile(
    extra_args: Optional[Mapping[str, Any]],
    *,
    defaults_for_profile: TrainingDefaultsProvider,
    limit_profile_args: Optional[TrainingProfileLimiter] = None,
    compute_environment: Optional[Dict[str, Any]] = None,
    protected_profiles: Iterable[str] = ("heavy_validation", "benchmark"),
) -> Dict[str, Any]:
    """Apply shared training-profile resolution around backend-specific defaults."""
    requested = dict(extra_args or {})
    requested_profile = requested.pop("training_profile", None)
    requested_validation_protocol = requested.pop("validation_protocol", None)
    allow_heavy_compute = bool(requested.pop("allow_heavy_compute", False))

    compute_env = compute_environment or describe_compute_environment()
    resolved = resolve_training_profile(compute_env)
    profile = requested_profile or resolved["profile"]

    if not allow_heavy_compute and profile in set(protected_profiles):
        profile = resolved["profile"]

    merged = {
        **defaults_for_profile(profile),
        **requested,
    }
    if limit_profile_args is not None:
        merged = limit_profile_args(profile, merged, allow_heavy_compute)

    return {
        "compute_environment": compute_env,
        "training_profile": profile,
        "profile_reason": resolved["reason"],
        "validation_protocol": requested_validation_protocol,
        "extra_args": merged,
    }


def materialize_primary_protocol_artifacts(
    *,
    root_output_dir: Path,
    primary_run: Mapping[str, Any],
    model_filename: str,
) -> Dict[str, Optional[str]]:
    """Copy primary-run artifacts to canonical root-level locations."""
    root_model_dir = root_output_dir / "model_0"
    root_model_dir.mkdir(parents=True, exist_ok=True)

    file_map = {
        primary_run.get("model_path")
        or primary_run.get("best_model_path"): root_model_dir / model_filename,
        primary_run.get("validation_predictions_path"): root_model_dir
        / "validation_predictions.csv",
        primary_run.get("test_predictions_path"): root_model_dir / "test_predictions.csv",
        primary_run.get("config_path"): root_output_dir / "config.toml",
        primary_run.get("splits_path"): root_output_dir / "splits.json",
    }
    copied: Dict[str, Optional[str]] = {
        "best_model_path": None,
        "validation_predictions_path": None,
        "test_predictions_path": None,
        "config_path": None,
        "splits_path": None,
    }

    for source_raw, target_path in file_map.items():
        if not source_raw:
            continue
        source_path = Path(str(source_raw)).expanduser()
        if not source_path.exists():
            continue
        if source_path.resolve() != target_path.resolve():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
        if target_path == root_model_dir / model_filename:
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


def compute_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute the standard QSAR regression metrics on aligned values."""
    actual_values = pd.to_numeric(y_true, errors="coerce")
    predicted_values = pd.to_numeric(y_pred, errors="coerce")
    valid_mask = actual_values.notna() & predicted_values.notna()
    if not valid_mask.any():
        return {}

    actual = actual_values[valid_mask].astype(float)
    predicted = predicted_values[valid_mask].astype(float)
    residuals = actual - predicted
    mse = float((residuals.pow(2)).mean())
    mae = float(residuals.abs().mean())
    rae_denom = float((actual - float(actual.mean())).abs().sum())
    rae_num = float(residuals.abs().sum())
    centered = actual - float(actual.mean())
    ss_tot = float((centered.pow(2)).sum())
    ss_res = float((residuals.pow(2)).sum())
    spearman = None
    kendall = None
    try:
        spearman_stat = spearmanr(actual.to_numpy(), predicted.to_numpy(), nan_policy="omit")
        spearman = float(spearman_stat.statistic) if spearman_stat.statistic is not None else None
    except Exception:
        pass
    try:
        kendall_stat = kendalltau(actual.to_numpy(), predicted.to_numpy(), nan_policy="omit")
        kendall = float(kendall_stat.statistic) if kendall_stat.statistic is not None else None
    except Exception:
        pass
    return {
        "mse": mse,
        "mae": mae,
        "rae": float(rae_num / rae_denom) if rae_denom > 0 else None,
        "rmse": float(math.sqrt(mse)),
        "r2": float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None,
        "spearman": spearman,
        "kendall": kendall,
        "n": int(valid_mask.sum()),
        **({"target_column": target_column} if target_column else {}),
    }


def _binary_roc_auc(y_true_codes: pd.Series, positive_scores: pd.Series) -> Optional[float]:
    aligned = pd.DataFrame({"y": y_true_codes, "score": positive_scores}).dropna()
    if aligned.empty:
        return None
    positives = int((aligned["y"] == 1).sum())
    negatives = int((aligned["y"] == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = aligned["score"].rank(method="average")
    rank_sum_pos = float(ranks[aligned["y"] == 1].sum())
    return float((rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives))


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    class_labels: Optional[Sequence[Any]] = None,
    positive_scores: Optional[pd.Series] = None,
    target_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute standard QSAR classification metrics for binary or multiclass labels."""
    truth = y_true.map(normalize_classification_label)
    pred = y_pred.map(normalize_classification_label)
    mask = truth.notna() & pred.notna()
    truth = truth[mask].reset_index(drop=True)
    pred = pred[mask].reset_index(drop=True)
    if truth.empty:
        return {}

    labels = list(class_labels or resolve_class_labels(pd.concat([truth, pred], ignore_index=True)))
    if len(labels) < 2:
        return {}

    true_codes, _ = encode_classification_labels(truth, labels)
    pred_codes, _ = encode_classification_labels(pred, labels)
    valid = true_codes.notna() & pred_codes.notna()
    true_codes = true_codes[valid].astype(int).reset_index(drop=True)
    pred_codes = pred_codes[valid].astype(int).reset_index(drop=True)
    if true_codes.empty:
        return {}

    rows: List[Dict[str, Any]] = []
    recalls: List[float] = []
    precisions: List[float] = []
    f1_values: List[float] = []
    for class_index, class_label in enumerate(labels):
        tp = int(((true_codes == class_index) & (pred_codes == class_index)).sum())
        fp = int(((true_codes != class_index) & (pred_codes == class_index)).sum())
        fn = int(((true_codes == class_index) & (pred_codes != class_index)).sum())
        support = int((true_codes == class_index).sum())
        precision = float(tp / (tp + fp)) if tp + fp else 0.0
        recall = float(tp / (tp + fn)) if tp + fn else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_values.append(f1)
        rows.append(
            {
                "class_label": json_safe_label(class_label),
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    accuracy = float((true_codes == pred_codes).mean())
    metrics: Dict[str, Any] = {
        "accuracy": accuracy,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else None,
        "precision_macro": float(sum(precisions) / len(precisions)) if precisions else None,
        "recall_macro": float(sum(recalls) / len(recalls)) if recalls else None,
        "f1_macro": float(sum(f1_values) / len(f1_values)) if f1_values else None,
        "n": int(len(true_codes)),
        "num_classes": len(labels),
        "class_count": len(labels),
        "class_labels": [json_safe_label(label) for label in labels],
        "class_counts": {
            str(json_safe_label(labels[index])): int((true_codes == index).sum())
            for index in range(len(labels))
        },
        "per_class": rows,
        **({"target_column": target_column} if target_column else {}),
    }
    if len(labels) == 2:
        pos_label = labels[1]
        binary_row = rows[1]
        metrics.update(
            {
                "positive_class": json_safe_label(pos_label),
                "negative_class": json_safe_label(labels[0]),
                "precision": binary_row["precision"],
                "recall": binary_row["recall"],
                "f1": binary_row["f1"],
            }
        )
        if positive_scores is not None:
            scores = positive_scores.reset_index(drop=True)
            if len(scores) == len(mask):
                scores = scores[mask.to_numpy()].reset_index(drop=True)
            if len(scores) == len(valid):
                scores = scores[valid.to_numpy()].reset_index(drop=True)
            auc = _binary_roc_auc(true_codes, pd.to_numeric(scores, errors="coerce"))
            if auc is not None:
                metrics["roc_auc"] = auc
    return metrics


def build_applicability_domain_for_training(
    *,
    train_csv: str,
    primary_run: Mapping[str, Any],
    primary_output_dir: Path,
    task: PredictionTaskSpec,
    model_id_hint: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    feature_frame: Optional[pd.DataFrame] = None,
    feature_space: Optional[str] = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    prediction_artifact_paths: Optional[Mapping[str, Any]] = None,
    applicability_domain_methods: Optional[Sequence[str] | str] = None,
) -> Dict[str, Any]:
    """Build the modern AD from the train split and keep legacy AD."""
    splits_path = Path(str(primary_run.get("splits_path") or primary_output_dir / "splits.json"))
    if not splits_path.exists():
        return {}

    split_payload = json.loads(splits_path.read_text())
    if not split_payload or "train" not in split_payload[0]:
        return {}

    dataset = strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
    modern_feature_frame = (
        strip_unnamed_columns(feature_frame.copy()) if feature_frame is not None else dataset
    )
    train_indices = split_payload[0].get("train") or []
    if not train_indices:
        return {}
    smiles_column = task.smiles_columns[0] if task.smiles_columns else "smiles"
    target_column = task.target_columns[0] if task.target_columns else None
    ad_output_dir = primary_output_dir / "applicability_domain"
    ad_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_feature_columns = [
        str(column)
        for column in (
            feature_columns
            or primary_run.get("feature_columns")
            or primary_run.get("features")
            or []
        )
        if str(column) in modern_feature_frame.columns
    ]
    if not resolved_feature_columns:
        excluded = {
            smiles_column,
            "smiles",
            QSAR_ROW_ID_COLUMN,
            *list(task.target_columns or []),
        }
        resolved_feature_columns = [
            column
            for column in modern_feature_frame.columns
            if column not in excluded and pd.api.types.is_numeric_dtype(modern_feature_frame[column])
        ]

    resolved_feature_space = (
        feature_space
        or primary_run.get("representation_name")
        or primary_run.get("feature_space")
        or "tabular_features"
    )

    if str(resolved_feature_space) == "chemprop_embedding" and not feature_columns:
        resolved_feature_columns = []

    train_frame = modern_feature_frame.iloc[list(train_indices)].copy()
    ad_random_state = (
        primary_run.get("random_state")
        or primary_run.get("data_seed")
        or primary_run.get("seed")
        or 0
    )
    ad_summary = fit_modern_applicability_domain(
        feature_frame=train_frame,
        feature_columns=resolved_feature_columns,
        output_dir=ad_output_dir,
        model_id=model_id_hint or primary_output_dir.name,
        feature_space=str(resolved_feature_space),
        representation_name=str(resolved_feature_space),
        feature_metadata=feature_metadata,
        methods=applicability_domain_methods,
        random_state=int(ad_random_state),
    )

    if ad_summary.get("available"):
        split_score_summaries: Dict[str, Any] = {}
        score_paths: Dict[str, str] = {}
        split_item = split_payload[0]
        split_aliases = {
            "train": ["train"],
            "validation": ["validation", "val", "valid"],
            "test": ["test"],
        }
        for label, keys in split_aliases.items():
            indices: List[int] = []
            for key in keys:
                if split_item.get(key):
                    indices = list(split_item.get(key) or [])
                    break
            if not indices:
                continue
            scored = score_modern_applicability_domain(
                feature_frame=modern_feature_frame.iloc[indices].copy(),
                applicability_domain=ad_summary,
                output_dir=ad_output_dir,
                score_label=label,
            )
            split_score_summaries[label] = scored.get("summary") or {}
            if scored.get("scores_path"):
                score_paths[f"scores_{label}_path"] = str(scored["scores_path"])
            if scored.get("scores") is not None:
                predictions_path = primary_run.get(f"{label}_predictions_path")
                if label == "test":
                    predictions_path = predictions_path or primary_run.get("test_predictions_path")
                ad_metrics = attach_ad_scores_and_metrics_to_predictions(
                    predictions_path=str(predictions_path) if predictions_path else None,
                    scores=scored["scores"],
                    task=task,
                    target_column=target_column,
                    class_labels=primary_run.get("class_labels"),
                )
                if ad_metrics:
                    split_score_summaries[label].update(ad_metrics)
                elif label in {"validation", "test"}:
                    split_score_summaries[label]["metrics_unavailable_reason"] = (
                        f"No {label}_predictions_path artifact was available."
                    )
                canonical_path = (prediction_artifact_paths or {}).get(label)
                if canonical_path and str(canonical_path) != str(predictions_path or ""):
                    try:
                        append_ad_scores_to_csv(str(canonical_path), scored["scores"])
                        split_score_summaries[label][
                            "ad_enriched_canonical_predictions_path"
                        ] = str(canonical_path)
                    except Exception as exc:
                        split_score_summaries[label][
                            "ad_canonical_sync_warning"
                        ] = str(exc)
            if label in {"validation", "test"} and scored.get("scores") is not None:
                plot_dir = ad_output_dir / "plots" / label
                plot_artifacts = build_modern_ad_plots(scored["scores"], plot_dir)
                if plot_artifacts:
                    split_score_summaries[label]["plots"] = plot_artifacts
        ad_summary["split_score_summaries"] = split_score_summaries
        ad_summary.update(score_paths)

    legacy_summary: Dict[str, Any] = {}
    if smiles_column in dataset.columns or "smiles" in dataset.columns:
        try:
            legacy_summary = build_legacy_similarity_ad(
                dataset=dataset,
                train_indices=train_indices,
                smiles_column=smiles_column if smiles_column in dataset.columns else "smiles",
                output_dir=str(ad_output_dir / "legacy_similarity_ad"),
                model_id=model_id_hint or primary_output_dir.name,
            )
            if legacy_summary:
                legacy_summary["legacy"] = True
                legacy_summary["method"] = "legacy_similarity_ad"
        except Exception as exc:
            legacy_summary = {
                "available": False,
                "legacy": True,
                "method": "legacy_similarity_ad",
                "error": str(exc),
            }
    if legacy_summary:
        ad_summary["legacy_similarity_ad"] = legacy_summary
    return ad_summary


def build_training_plots_if_possible(
    *,
    train_csv: str,
    split_results: List[Dict[str, Any]],
    primary_run: Dict[str, Any],
    root_artifacts: Mapping[str, Optional[str]],
    root_output_dir: Path,
    target_column: Optional[str],
    task_type: str = "regression",
) -> Dict[str, str]:
    """Build standard QSAR plots when the required split artifacts are present."""
    if (
        not target_column
        or not root_artifacts.get("splits_path")
        or not root_artifacts.get("test_predictions_path")
    ):
        return {}

    plots_output_dir = root_output_dir / "artifacts" / "plots"
    try:
        return build_qsar_training_plots(
            train_csv=train_csv,
            split_results=[
                {
                    **item,
                    "splits_path": (
                        root_artifacts["splits_path"]
                        if item is primary_run
                        else item.get("splits_path")
                    ),
                    "test_predictions_path": (
                        root_artifacts["test_predictions_path"]
                        if item is primary_run
                        else item.get("test_predictions_path")
                    ),
                }
                for item in split_results
            ],
            primary_run={
                **primary_run,
                "splits_path": root_artifacts.get("splits_path") or primary_run.get("splits_path"),
                "test_predictions_path": root_artifacts.get("test_predictions_path")
                or primary_run.get("test_predictions_path"),
            },
            output_dir=str(plots_output_dir),
            target_column=target_column,
            task_type=task_type,
        )
    except Exception:
        return {}


def _prediction_columns(
    frame: pd.DataFrame, target_column: str
) -> tuple[Optional[str], Optional[str]]:
    true_candidates = ["y_true", f"{target_column}_true", target_column]
    pred_candidates = ["y_pred", f"{target_column}_prediction", "prediction", target_column]
    true_col = next((column for column in true_candidates if column in frame.columns), None)
    pred_col = next(
        (column for column in pred_candidates if column in frame.columns and column != true_col),
        None,
    )
    return true_col, pred_col


def _positive_score_column(frame: pd.DataFrame) -> Optional[str]:
    for column in ("positive_probability", "probability"):
        if column in frame.columns:
            return column
    return next(
        (
            column
            for column in frame.columns
            if str(column).endswith("_positive_probability")
            or str(column).endswith("_probability")
        ),
        None,
    )


def attach_ad_scores_and_metrics_to_predictions(
    *,
    predictions_path: Optional[str],
    scores: pd.DataFrame,
    task: PredictionTaskSpec,
    target_column: Optional[str],
    class_labels: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Append AD scores to predictions and compute all/in/out metrics."""
    if not predictions_path or not target_column:
        return {}
    path = Path(str(predictions_path)).expanduser()
    if not path.exists():
        return {}
    try:
        append_ad_scores_to_csv(path, scores)
        frame = pd.read_csv(path)
        true_col, pred_col = _prediction_columns(frame, target_column)
        if not true_col or not pred_col:
            return {}
        metric_frame = frame.copy()
        metric_frame["__ad_true"] = metric_frame[true_col]
        metric_frame["__ad_pred"] = metric_frame[pred_col]
        score_col = _positive_score_column(metric_frame) if is_classification_task(task.task_type) else None

        if is_classification_task(task.task_type):
            def metric_func(
                y_true: pd.Series,
                y_pred: pd.Series,
                *,
                positive_scores: Optional[pd.Series] = None,
                target_column: Optional[str] = None,
            ) -> Dict[str, Any]:
                return compute_classification_metrics(
                    y_true,
                    y_pred,
                    class_labels=class_labels,
                    positive_scores=positive_scores,
                    target_column=target_column,
                )

        else:
            metric_func = compute_regression_metrics

        metrics = metrics_by_ad_status(
            frame=metric_frame,
            metric_func=metric_func,
            target_column="__ad_true",
            prediction_column="__ad_pred",
            score_column=score_col,
        )
        for section in ("metrics_all", "metrics_in_domain", "metrics_out_of_domain"):
            if isinstance(metrics.get(section), dict) and metrics[section].get("target_column"):
                metrics[section]["target_column"] = target_column
        return metrics
    except Exception as exc:
        return {"metrics_unavailable_reason": str(exc)}


def build_cross_validation_artifacts(
    *,
    split_results: List[Dict[str, Any]],
    output_dir: Path,
    target_column: str,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for split_result in split_results:
        predictions_path = split_result.get("test_predictions_path")
        split_payload = split_result.get("split_payload") or []
        if not predictions_path or not split_payload:
            continue
        path = Path(str(predictions_path)).expanduser()
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        true_col, pred_col = _prediction_columns(frame, target_column)
        test_indices = [int(idx) for idx in ((split_payload[0] or {}).get("test") or [])]
        if not true_col or not pred_col or len(frame) != len(test_indices):
            continue
        metadata = (split_payload[0] or {}).get("metadata") or {}
        repeat_index = int(metadata.get("cv_repeat") or split_result.get("repeat_index") or 1)
        fold_index = int(metadata.get("cv_fold") or split_result.get("fold_index") or 1)
        for row_index, source_row_index in enumerate(test_indices):
            y_true = pd.to_numeric(
                pd.Series([frame.iloc[row_index][true_col]]), errors="coerce"
            ).iloc[0]
            y_pred = pd.to_numeric(
                pd.Series([frame.iloc[row_index][pred_col]]), errors="coerce"
            ).iloc[0]
            if pd.isna(y_true) or pd.isna(y_pred):
                continue
            residual = float(y_true) - float(y_pred)
            rows.append(
                {
                    "repeat": repeat_index,
                    "fold": fold_index,
                    "fold_label": split_result.get("strategy_label"),
                    "source_row_index": source_row_index,
                    "y_true": float(y_true),
                    "y_pred": float(y_pred),
                    "residual": residual,
                    "absolute_error": abs(residual),
                }
            )

    if not rows:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    fold_predictions = pd.DataFrame(rows)
    fold_predictions_path = output_dir / "cv_fold_predictions.csv"
    fold_predictions.to_csv(fold_predictions_path, index=False)

    repeat_rows: List[Dict[str, Any]] = []
    for repeat, group in fold_predictions.groupby("repeat", sort=True):
        metrics = compute_regression_metrics(
            group["y_true"],
            group["y_pred"],
            target_column=target_column,
        )
        metrics["repeat"] = int(repeat)
        metrics["q2"] = metrics.get("r2")
        repeat_rows.append(metrics)
    repeat_metrics = pd.DataFrame(repeat_rows)
    repeat_metrics_path = output_dir / "cv_repeat_metrics.csv"
    repeat_metrics.to_csv(repeat_metrics_path, index=False)

    metric_names = ("q2", "r2", "rmse", "mae", "mse", "rae", "spearman", "kendall")
    summary: Dict[str, Any] = {
        "n_repeats": int(fold_predictions["repeat"].nunique()),
        "n_folds": int(fold_predictions["fold"].nunique()),
        "n_predictions": int(len(fold_predictions)),
        "metrics": {},
    }
    for metric_name in metric_names:
        values = pd.to_numeric(repeat_metrics.get(metric_name), errors="coerce").dropna()
        if values.empty:
            continue
        summary["metrics"][metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    mean_rmse = (summary["metrics"].get("rmse") or {}).get("mean")
    molecule_stats = fold_predictions.groupby("source_row_index", as_index=False).agg(
        y_true=("y_true", "first"), avg_pred=("y_pred", "mean")
    )
    molecule_stats["abs_error"] = (molecule_stats["avg_pred"] - molecule_stats["y_true"]).abs()
    if mean_rmse is not None:
        molecule_stats["outlier"] = molecule_stats["abs_error"] >= (2.0 * float(mean_rmse))
        summary["outlier_threshold_abs_error"] = 2.0 * float(mean_rmse)
        summary["outlier_count"] = int(molecule_stats["outlier"].sum())
    molecule_stats_path = output_dir / "cv_molecule_stats.csv"
    molecule_stats.to_csv(molecule_stats_path, index=False)

    summary_path = output_dir / "cv_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "fold_predictions_path": str(fold_predictions_path),
        "repeat_metrics_path": str(repeat_metrics_path),
        "summary_path": str(summary_path),
        "molecule_stats_path": str(molecule_stats_path),
        "summary": summary,
    }


def write_training_summary(summary_path: Path, payload: Mapping[str, Any]) -> Path:
    """Write a canonical JSON training summary."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dict(payload), indent=2) + "\n")
    return summary_path


def collect_training_bundle_files(
    *,
    train_csv: str,
    summary_path: Path,
    result: Mapping[str, Any],
    split_results: Iterable[Mapping[str, Any]],
    ad_summary: Mapping[str, Any],
    plot_artifacts: Mapping[str, str],
    curation_artifacts: Optional[Mapping[str, Any]] = None,
    activity_cliffs: Optional[Mapping[str, Any]] = None,
    extra_files: Optional[Iterable[Path]] = None,
) -> List[Path]:
    """Collect common training artifacts for a downloadable bundle."""
    files: List[Path] = [Path(train_csv).expanduser(), summary_path]
    for key in (
        "model_path",
        "best_model_path",
        "config_path",
        "splits_path",
        "validation_predictions_path",
        "test_predictions_path",
    ):
        if result.get(key):
            files.append(Path(str(result[key])).expanduser())
    for split_result in split_results:
        for key in (
            "summary_path",
            "model_path",
            "best_model_path",
            "validation_predictions_path",
            "test_predictions_path",
            "splits_path",
        ):
            if split_result.get(key):
                files.append(Path(str(split_result[key])).expanduser())
    for key in (
        "manifest_path",
        "bounds_path",
        "scores_train_path",
        "scores_validation_path",
        "scores_test_path",
        "reference_store_path",
        "reference_manifest_path",
        "applicability_domain_path",
    ):
        if ad_summary.get(key):
            files.append(Path(str(ad_summary[key])).expanduser())
    for nested in ((ad_summary.get("methods") or {}).values()):
        if isinstance(nested, Mapping):
            for key in ("manifest_path", "bounds_path"):
                if nested.get(key):
                    files.append(Path(str(nested[key])).expanduser())
    legacy_ad = ad_summary.get("legacy_similarity_ad") or {}
    if isinstance(legacy_ad, Mapping):
        for key in ("reference_store_path", "reference_manifest_path", "applicability_domain_path"):
            if legacy_ad.get(key):
                files.append(Path(str(legacy_ad[key])).expanduser())
    for artifact_path in plot_artifacts.values():
        files.append(Path(str(artifact_path)).expanduser())
    for artifact_path in ((curation_artifacts or {}).get("artifacts") or {}).values():
        if artifact_path:
            files.append(Path(str(artifact_path)).expanduser())
    curated_dataset_path = (curation_artifacts or {}).get("curated_dataset_path")
    if curated_dataset_path:
        files.append(Path(str(curated_dataset_path)).expanduser())

    cliffs = activity_cliffs or {}
    for key in ("annotated_training_csv", "summary_path", "clean_training_csv"):
        if cliffs.get(key):
            files.append(Path(str(cliffs[key])).expanduser())
    for variant in cliffs.get("variants") or []:
        if variant.get("filtered_training_csv"):
            files.append(Path(str(variant["filtered_training_csv"])).expanduser())
        training_result = variant.get("training_result") or {}
        for split_result in training_result.get("split_results") or []:
            for key in (
                "model_path",
                "best_model_path",
                "validation_predictions_path",
                "test_predictions_path",
                "splits_path",
            ):
                if split_result.get(key):
                    files.append(Path(str(split_result[key])).expanduser())
    for artifact_path in (cliffs.get("plot_artifacts") or {}).values():
        files.append(Path(str(artifact_path)).expanduser())
    for artifact_path in (cliffs.get("loop_comparison_plot_artifacts") or {}).values():
        files.append(Path(str(artifact_path)).expanduser())
    if extra_files:
        files.extend(Path(path).expanduser() for path in extra_files)
    return files
