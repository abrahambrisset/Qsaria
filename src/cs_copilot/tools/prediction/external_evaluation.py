#!/usr/bin/env python
# coding: utf-8
"""Append-only external evaluation helpers for catalogued prediction models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import (
    resolve_smiles_column_name,
    standardize_smiles_column,
)

from .backend import PredictionModelRecord
from .qsar_training_policy import project_now, safe_slug
from .training_orchestration import (
    compute_classification_metrics,
    compute_regression_metrics,
    is_classification_task,
    is_multiclass_task,
    json_safe_label,
    resolve_class_labels,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def _read_csv(path: str) -> pd.DataFrame:
    local_path = Path(path).expanduser()
    if local_path.exists():
        return pd.read_csv(local_path)
    with S3.open(path, "r") as fh:
        return pd.read_csv(fh)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2) + "\n")


def _resolve_target_columns(
    record: PredictionModelRecord, target_columns: Optional[Sequence[str]]
) -> List[str]:
    resolved = list(target_columns or record.task.target_columns or [])
    if not resolved:
        raise ValueError(
            "External evaluation requires target_columns or model metadata target columns."
        )
    return resolved


def _prediction_column(predictions: pd.DataFrame, target: str, *, single_target: bool) -> str:
    candidates = [f"{target}_prediction"]
    if single_target:
        candidates.append("prediction")
    candidates.extend([f"{target}_pred", f"predicted_{target}"])
    for column in candidates:
        if column in predictions.columns:
            return column
    raise ValueError(
        f"External evaluation could not find a prediction column for target `{target}`. "
        f"Expected one of {candidates}; available columns: {list(predictions.columns)}"
    )


def _score_column(predictions: pd.DataFrame, target: str, *, single_target: bool) -> Optional[str]:
    candidates = [f"{target}_positive_probability", f"{target}_probability"]
    if single_target:
        candidates.extend(["positive_probability", "probability"])
    for column in candidates:
        if column in predictions.columns:
            return column
    return None


def _normalize_prediction_columns(
    predictions: pd.DataFrame,
    target_columns: Sequence[str],
) -> pd.DataFrame:
    """Avoid collisions when a backend reuses target names for predictions."""
    normalized = predictions.copy()
    rename: Dict[str, str] = {}
    for target in target_columns:
        explicit_name = f"{target}_prediction"
        if target in normalized.columns and explicit_name not in normalized.columns:
            rename[target] = explicit_name
    if rename:
        normalized = normalized.rename(columns=rename)
    return normalized


def _plot_regression(y_true: pd.Series, y_pred: pd.Series, output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    actual = pd.to_numeric(y_true, errors="coerce")
    predicted = pd.to_numeric(y_pred, errors="coerce")
    frame = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if frame.empty:
        return {}
    residuals = frame["actual"] - frame["predicted"]
    errors = residuals.abs()
    artifacts: Dict[str, str] = {}

    def save(name: str) -> str:
        path = output_dir / f"{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        artifacts[name] = str(path)
        return str(path)

    plt.figure(figsize=(5, 5))
    plt.scatter(frame["actual"], frame["predicted"], alpha=0.75)
    low = float(min(frame["actual"].min(), frame["predicted"].min()))
    high = float(max(frame["actual"].max(), frame["predicted"].max()))
    plt.plot([low, high], [low, high], color="black", linewidth=1)
    plt.xlabel("Observed")
    plt.ylabel("Predicted")
    plt.title("Observed vs predicted")
    save("observed_vs_predicted")

    plt.figure(figsize=(6, 4))
    plt.scatter(frame["predicted"], residuals, alpha=0.75)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals")
    save("residuals")

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual histogram")
    save("residual_histogram")

    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=30)
    plt.xlabel("Absolute error")
    plt.ylabel("Count")
    plt.title("Error distribution")
    save("error_distribution")
    return artifacts


def _confusion_counts(y_true: pd.Series, y_pred: pd.Series, labels: Sequence[Any]) -> np.ndarray:
    normalized_labels = [json_safe_label(label) for label in labels]
    label_to_index = {str(label): index for index, label in enumerate(normalized_labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=float)
    for truth, pred in zip(y_true, y_pred, strict=False):
        truth_key = str(json_safe_label(truth))
        pred_key = str(json_safe_label(pred))
        if truth_key in label_to_index and pred_key in label_to_index:
            matrix[label_to_index[truth_key], label_to_index[pred_key]] += 1.0
    return matrix


def _plot_matrix(matrix: np.ndarray, labels: Sequence[Any], path: Path, title: str) -> None:
    plt.figure(figsize=(max(5, len(labels) * 0.7), max(4, len(labels) * 0.6)))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    ticks = list(range(len(labels)))
    display_labels = [str(json_safe_label(label)) for label in labels]
    plt.xticks(ticks, display_labels, rotation=45, ha="right")
    plt.yticks(ticks, display_labels)
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            text = f"{value:.2f}" if float(value) % 1 else str(int(value))
            plt.text(col, row, text, ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_binary_curves(
    y_true: pd.Series,
    scores: pd.Series,
    labels: Sequence[Any],
    output_dir: Path,
    artifacts: Dict[str, str],
) -> None:
    try:
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception:
        return
    encoded = y_true.map(
        lambda value: 1 if json_safe_label(value) == json_safe_label(labels[1]) else 0
    )
    frame = pd.DataFrame({"y": encoded, "score": pd.to_numeric(scores, errors="coerce")}).dropna()
    if frame.empty or frame["y"].nunique() < 2:
        return

    fpr, tpr, _ = roc_curve(frame["y"], frame["score"])
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    path = output_dir / "roc_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["roc_curve"] = str(path)

    precision, recall, _ = precision_recall_curve(frame["y"], frame["score"])
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    path = output_dir / "precision_recall_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["precision_recall_curve"] = str(path)

    prob_true, prob_pred = calibration_curve(frame["y"], frame["score"], n_bins=min(10, len(frame)))
    plt.figure(figsize=(5, 5))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction positives")
    plt.title("Calibration")
    path = output_dir / "calibration_curve.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["calibration_curve"] = str(path)

    plt.figure(figsize=(6, 4))
    for code, label in [(0, labels[0]), (1, labels[1])]:
        subset = frame.loc[frame["y"] == code, "score"]
        if not subset.empty:
            plt.hist(subset, bins=20, alpha=0.55, label=str(json_safe_label(label)))
    plt.xlabel("Positive probability")
    plt.ylabel("Count")
    plt.title("Probability distribution")
    plt.legend()
    path = output_dir / "probability_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["probability_distribution"] = str(path)


def _plot_classification(
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    scores: Optional[pd.Series],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = resolve_class_labels(pd.concat([y_true, y_pred], ignore_index=True))
    if len(labels) < 2:
        return {}
    matrix = _confusion_counts(y_true, y_pred, labels)
    artifacts: Dict[str, str] = {}
    path = output_dir / "confusion_matrix.png"
    _plot_matrix(matrix, labels, path, "Confusion matrix")
    artifacts["confusion_matrix"] = str(path)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    path = output_dir / "confusion_matrix_normalized.png"
    _plot_matrix(normalized, labels, path, "Normalized confusion matrix")
    artifacts["confusion_matrix_normalized"] = str(path)
    if len(labels) == 2 and scores is not None:
        _plot_binary_curves(y_true, scores, labels, output_dir, artifacts)
    return artifacts


def _write_report(
    path: Path,
    *,
    model_id: str,
    evaluation_id: str,
    dataset_path: str,
    task_type: str,
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> None:
    lines = [
        f"# External evaluation: {evaluation_id}",
        "",
        f"- Model: `{model_id}`",
        f"- Dataset: `{dataset_path}`",
        f"- Task type: `{task_type}`",
        f"- Metrics file: `{artifacts.get('metrics')}`",
        f"- Predictions file: `{artifacts.get('predictions')}`",
        "",
        "## Metrics",
        "",
        "```json",
        json.dumps(_json_safe(metrics), indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines))


def evaluate_model_on_external_dataset(
    *,
    record: PredictionModelRecord,
    backend: Any,
    test_csv: str,
    smiles_column: str = "smiles",
    target_columns: Optional[Sequence[str]] = None,
    evaluation_label: Optional[str] = None,
) -> Dict[str, Any]:
    if not record.metadata_path:
        raise ValueError(
            "External evaluation requires a persisted catalog model with metadata_path. "
            "Persist the session model first with `persist_registered_model`, then retry with "
            "the returned canonical catalog model_id."
        )
    metadata_path = Path(record.metadata_path).expanduser()
    if not metadata_path.exists():
        raise ValueError(f"Model metadata_path does not exist: {metadata_path}")

    source_df = _read_csv(test_csv)
    resolved_targets = _resolve_target_columns(record, target_columns)
    missing_targets = [column for column in resolved_targets if column not in source_df.columns]
    if missing_targets:
        raise ValueError(
            "External evaluation dataset is missing required target columns: "
            f"{missing_targets}. No evaluation was written."
        )
    smiles_found = resolve_smiles_column_name(source_df, smiles_column)

    created_at = project_now()
    label_slug = safe_slug(evaluation_label or Path(str(test_csv)).stem or "external_dataset")
    evaluation_id = f"{label_slug}_{created_at.strftime('%Y%m%d_%H%M%S')}"
    model_root = metadata_path.parent
    eval_dir = model_root / "evaluations" / evaluation_id
    eval_dir.mkdir(parents=True, exist_ok=False)

    input_df = source_df.copy()
    input_df = standardize_smiles_column(input_df, smiles_found)
    if smiles_found != "smiles":
        input_df["smiles"] = input_df[smiles_found]
        input_df = input_df.drop(columns=[smiles_found])
    evaluation_input = eval_dir / "evaluation_input.csv"
    prediction_input = input_df.drop(columns=resolved_targets, errors="ignore")
    prediction_input.to_csv(evaluation_input, index=False)

    predictions_path = eval_dir / "predictions.csv"
    backend.predict_from_csv(
        input_csv=str(evaluation_input),
        model_record=record,
        preds_path=str(predictions_path),
        return_uncertainty=False,
    )

    predictions_only = _normalize_prediction_columns(
        pd.read_csv(predictions_path), resolved_targets
    )
    predictions_only.to_csv(predictions_path, index=False)
    prediction_columns = [
        column for column in predictions_only.columns if column not in source_df.columns
    ]
    enriched = pd.concat(
        [
            source_df.reset_index(drop=True),
            predictions_only[prediction_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    enriched.to_csv(predictions_path, index=False)

    single_target = len(resolved_targets) == 1
    task_type = record.task.task_type
    target_metrics: Dict[str, Dict[str, Any]] = {}
    plot_artifacts: Dict[str, Any] = {}
    metrics_by_target_rows: List[Dict[str, Any]] = []
    for target in resolved_targets:
        pred_col = _prediction_column(enriched, target, single_target=single_target)
        score_col = _score_column(enriched, target, single_target=single_target)
        if is_classification_task(task_type):
            scores = enriched[score_col] if score_col else None
            metric_values = compute_classification_metrics(
                enriched[target],
                enriched[pred_col],
                positive_scores=scores,
                target_column=target,
            )
            target_plot_dir = (
                eval_dir / "plots" / target if len(resolved_targets) > 1 else eval_dir / "plots"
            )
            plot_artifacts[target] = _plot_classification(
                enriched[target],
                enriched[pred_col],
                scores=scores,
                output_dir=target_plot_dir,
            )
        else:
            metric_values = compute_regression_metrics(
                enriched[target],
                enriched[pred_col],
                target_column=target,
            )
            target_plot_dir = (
                eval_dir / "plots" / target if len(resolved_targets) > 1 else eval_dir / "plots"
            )
            plot_artifacts[target] = _plot_regression(
                enriched[target], enriched[pred_col], target_plot_dir
            )
        target_metrics[target] = metric_values
        metrics_by_target_rows.append({"target": target, **metric_values})

    if not target_metrics:
        raise ValueError("External evaluation did not produce any metrics.")
    metrics = (
        target_metrics[resolved_targets[0]]
        if single_target
        else {
            "target_count": len(resolved_targets),
            "row_count": int(len(enriched)),
            "task_type": task_type,
            "target_metrics": target_metrics,
        }
    )

    artifacts: Dict[str, Any] = {
        "evaluation_dir": str(eval_dir),
        "predictions": str(predictions_path),
        "metrics": str(eval_dir / "metrics.json"),
        "evaluation_summary": str(eval_dir / "evaluation_summary.json"),
        "evaluation_report": str(eval_dir / "evaluation_report.md"),
        "plots": plot_artifacts,
    }
    if len(resolved_targets) > 1:
        metrics_by_target_path = eval_dir / "metrics_by_target.csv"
        pd.DataFrame(metrics_by_target_rows).to_csv(metrics_by_target_path, index=False)
        artifacts["metrics_by_target"] = str(metrics_by_target_path)

    summary = {
        "evaluation_id": evaluation_id,
        "evaluation_label": evaluation_label or Path(str(test_csv)).stem,
        "evaluation_kind": "external_dataset",
        "model_id": record.model_id,
        "backend_name": record.backend_name,
        "task_type": task_type,
        "dataset_path": test_csv,
        "row_count": int(len(source_df)),
        "smiles_column": smiles_found,
        "target_columns": resolved_targets,
        "metrics": metrics,
        "target_metrics": target_metrics,
        "artifacts": artifacts,
        "created_at": created_at.isoformat(),
    }
    _write_json(eval_dir / "metrics.json", {"metrics": metrics, "target_metrics": target_metrics})
    _write_json(eval_dir / "evaluation_summary.json", summary)
    _write_report(
        eval_dir / "evaluation_report.md",
        model_id=record.model_id,
        evaluation_id=evaluation_id,
        dataset_path=test_csv,
        task_type=task_type,
        metrics=metrics,
        artifacts=artifacts,
    )

    metadata = json.loads(metadata_path.read_text())
    external_evaluations = list(metadata.get("external_evaluations") or [])
    external_evaluations.append(summary)
    metadata["external_evaluations"] = external_evaluations
    training_data_summary = dict(metadata.get("training_data_summary") or {})
    training_data_summary["external_evaluations"] = external_evaluations
    metadata["training_data_summary"] = training_data_summary
    metadata.setdefault("known_metrics", dict(record.known_metrics or {}))
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2) + "\n")

    return {
        **summary,
        "metrics_path": artifacts["metrics"],
        "predictions_path": artifacts["predictions"],
        "evaluation_summary_path": artifacts["evaluation_summary"],
        "evaluation_report_path": artifacts["evaluation_report"],
        "download_file_ref": artifacts["evaluation_report"],
        "download_file_tag": f"<file>{artifacts['evaluation_report']}</file>",
        "external_evaluation_count": len(external_evaluations),
        "multiclass": bool(is_multiclass_task(task_type)),
    }
