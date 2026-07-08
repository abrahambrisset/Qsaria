#!/usr/bin/env python
# coding: utf-8
"""Canonical reporting handoffs for QSAR training and evaluation tools."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

CLASSIFICATION_TASK_TYPES = {
    "classification",
    "binary_classification",
    "multiclass",
    "multiclass_classification",
}
CLASSIFICATION_COLUMNS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("f1_macro", "F1 macro"),
    ("roc_auc", "ROC-AUC"),
)
REGRESSION_COLUMNS = (
    ("r2", "R2"),
    ("rmse", "RMSE"),
    ("mae", "MAE"),
    ("mse", "MSE"),
)


def _is_classification(task_type: str) -> bool:
    return str(task_type or "").strip().lower() in CLASSIFICATION_TASK_TYPES


def _metric_columns(task_type: str) -> tuple[tuple[str, str], ...]:
    return CLASSIFICATION_COLUMNS if _is_classification(task_type) else REGRESSION_COLUMNS


def _display(value: Any) -> str:
    if value is None:
        return "non calculable"
    if isinstance(value, float) and math.isnan(value):
        return "non calculable"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _safe_cell(value: Any) -> str:
    return _display(value).replace("|", "\\|").replace("\n", " ")


def _metric_n(metrics: Mapping[str, Any], fallback: Optional[int]) -> Any:
    value = metrics.get("n")
    if value is not None:
        return value
    return fallback if fallback is not None else "non calculable"


def _metric_rows_markdown(
    *,
    rows: List[Dict[str, Any]],
    task_type: str,
    metrics_status: Optional[str] = None,
) -> str:
    if not rows:
        if metrics_status == "not_evaluated":
            return (
                "### Metriques d'evaluation\n\n"
                "Aucune metrique interne n'a ete calculee : le modele a ete entraine "
                "en full-train et doit etre evalue sur un jeu externe labelise."
            )
        return "### Metriques d'evaluation\n\nAucune metrique disponible."

    metric_columns = _metric_columns(task_type)
    target_values = {row.get("target") for row in rows if row.get("target")}
    include_target = len(target_values) > 1
    headers = ["Source"]
    if include_target:
        headers.append("Cible")
    headers.extend(["AD subset", "n", *[label for _, label in metric_columns]])
    lines = ["### Metriques d'evaluation", "", "| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        cells = [_safe_cell(row.get("source"))]
        if include_target:
            cells.append(_safe_cell(row.get("target") or "-"))
        cells.extend(
            [
                _safe_cell(row.get("ad_subset")),
                _safe_cell(_metric_n(metrics, row.get("n"))),
            ]
        )
        cells.extend(_safe_cell(metrics.get(key)) for key, _ in metric_columns)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _rows_from_ad_split(
    *,
    source: str,
    summary: Mapping[str, Any],
    target: Optional[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "source": source,
            "target": target,
            "ad_subset": "all",
            "n": summary.get("row_count"),
            "metrics": summary.get("metrics_all") or {},
        },
        {
            "source": source,
            "target": target,
            "ad_subset": "in_domain",
            "n": summary.get("n_in_domain"),
            "metrics": summary.get("metrics_in_domain") or {},
        },
        {
            "source": source,
            "target": target,
            "ad_subset": "out_of_domain",
            "n": summary.get("n_out_of_domain"),
            "metrics": summary.get("metrics_out_of_domain") or {},
        },
    ]


def _training_metric_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    target = next(iter(result.get("target_columns") or []), None)
    ad_splits = ((result.get("applicability_domain") or {}).get("split_score_summaries") or {})
    rows: List[Dict[str, Any]] = []
    for split_name, source in (
        ("validation", "validation interne"),
        ("test", "test interne"),
    ):
        summary = ad_splits.get(split_name)
        if isinstance(summary, Mapping):
            rows.extend(_rows_from_ad_split(source=source, summary=summary, target=target))
    if rows:
        return rows

    metrics = result.get("metrics") or {}
    if isinstance(metrics.get("validation"), Mapping):
        rows.append(
            {
                "source": "validation interne",
                "target": target,
                "ad_subset": "all",
                "n": metrics["validation"].get("n"),
                "metrics": metrics["validation"],
            }
        )
    if isinstance(metrics.get("test"), Mapping):
        rows.append(
            {
                "source": "test interne",
                "target": target,
                "ad_subset": "all",
                "n": metrics["test"].get("n"),
                "metrics": metrics["test"],
            }
        )
    return rows


def _external_metric_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    ad_by_target = result.get("ad_metrics_by_target") or {}
    rows: List[Dict[str, Any]] = []
    if isinstance(ad_by_target, Mapping):
        for target, payload in ad_by_target.items():
            if isinstance(payload, Mapping):
                rows.extend(
                    _rows_from_ad_split(
                        source="evaluation externe",
                        summary={
                            **payload,
                            "row_count": (payload.get("metrics_all") or {}).get(
                                "n", result.get("row_count")
                            ),
                            "n_in_domain": (payload.get("metrics_in_domain") or {}).get("n"),
                        },
                        target=str(target),
                    )
                )
    if rows:
        return rows
    metrics = result.get("metrics")
    return [
        {
            "source": "evaluation externe",
            "target": next(iter(result.get("target_columns") or []), None),
            "ad_subset": "all",
            "n": (metrics or {}).get("n") if isinstance(metrics, Mapping) else result.get("row_count"),
            "metrics": metrics or {},
        }
    ]


def _ad_markdown(ad: Mapping[str, Any], *, source_label: str = "training") -> str:
    if not ad:
        return "### Domaine d'applicabilite\n\nAD moderne non disponible."
    lines = [
        "### Domaine d'applicabilite",
        "",
        f"- Methode: `{ad.get('method') or ad.get('primary_method') or 'bounding_box'}`",
        f"- Espace de features: `{ad.get('feature_space') or ad.get('representation_name') or 'non specifie'}`",
        f"- Nombre de features: `{ad.get('feature_count') or 'non specifie'}`",
        f"- Reference train: `{ad.get('fit_row_count') or ad.get('reference_size') or 'non specifie'}`",
    ]
    split_summaries = ad.get("split_score_summaries") if isinstance(ad, Mapping) else None
    coverage_rows: List[tuple[str, Mapping[str, Any]]] = []
    if isinstance(split_summaries, Mapping):
        for split in ("train", "validation", "test"):
            if isinstance(split_summaries.get(split), Mapping):
                coverage_rows.append((split, split_summaries[split]))
    elif ad.get("row_count") is not None:
        coverage_rows.append((source_label, ad))
    if coverage_rows:
        lines.extend(
            [
                "",
                "| Source | n | coverage_in_domain | n_out_of_domain | n_invalid_features | top features violatrices |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for split, summary in coverage_rows:
            top = summary.get("top_violating_features") or []
            top_text = ", ".join(
                f"{item.get('feature')} ({item.get('count')})"
                for item in top[:5]
                if isinstance(item, Mapping)
            ) or "-"
            lines.append(
                "| "
                + " | ".join(
                    [
                        _safe_cell(split),
                        _safe_cell(summary.get("row_count")),
                        _safe_cell(summary.get("coverage_in_domain")),
                        _safe_cell(summary.get("n_out_of_domain")),
                        _safe_cell(summary.get("n_invalid_features")),
                        _safe_cell(top_text),
                    ]
                )
                + " |"
            )
    artifact_lines = [
        ("manifest", ad.get("manifest_path")),
        ("bounds", ad.get("bounds_path")),
        ("scores_train", ad.get("scores_train_path")),
        ("scores_validation", ad.get("scores_validation_path")),
        ("scores_test", ad.get("scores_test_path")),
    ]
    present = [(name, path) for name, path in artifact_lines if path]
    if present:
        lines.extend(["", "Artefacts AD:"])
        lines.extend(f"- {name}: `{path}`" for name, path in present)
    return "\n".join(lines)


def _artifacts_markdown(payload: Mapping[str, Any]) -> str:
    groups = {
        "Archive complete": ("training_bundle", "bundle_file_ref", "bundle_download_tag"),
        "Session training": ("output_dir", "summary_path", "config_path", "splits_path"),
        "Predictions": ("validation_predictions_path", "test_predictions_path", "predictions_path"),
        "Modele catalogue": ("model_root", "model_path", "metadata_path"),
        "Evaluation externe": (
            "evaluation_summary_path",
            "metrics_path",
            "evaluation_report_path",
        ),
    }
    lines = ["### Artefacts generes"]
    wrote = False
    for title, keys in groups.items():
        items = [(key, payload.get(key)) for key in keys if payload.get(key)]
        if not items:
            continue
        wrote = True
        lines.extend(["", f"**{title}**"])
        lines.extend(f"- {key}: `{value}`" for key, value in items)
    if not wrote:
        lines.append("\nAucun artefact liste dans le handoff.")
    return "\n".join(lines)


def governance_markdown(
    *,
    requested_status: Optional[str] = None,
    final_status: Optional[str] = None,
    status_reason: Optional[str] = None,
    metrics_status: Optional[str] = None,
) -> str:
    lines = ["### Gouvernance du modele"]
    if requested_status:
        lines.append(f"- Statut demande: `{requested_status}`")
    if final_status:
        lines.append(f"- Statut final: `{final_status}`")
    if metrics_status:
        lines.append(f"- Statut des metriques: `{metrics_status}`")
    if status_reason:
        lines.append(f"- Raison: {status_reason}")
    if metrics_status == "not_evaluated":
        lines.append("- Manquant pour promotion: evaluation externe labelisee et metriques documentees.")
    return "\n".join(lines)


def build_training_reporting_handoff(result: Mapping[str, Any]) -> Dict[str, Any]:
    metrics_status = result.get("metrics_status") or (
        "not_evaluated" if result.get("evaluation_required") else "evaluated"
    )
    registry_payload = result.get("recommended_registry_payload") or {}
    protocol = result.get("validation_protocol") or result.get("validation_strategy_type")
    decision = (
        f"{result.get('backend_name') or 'QSAR'} {result.get('task_type') or ''} entraine; "
        f"protocole={protocol or 'non specifie'}; metrics_status={metrics_status}."
    )
    if metrics_status == "not_evaluated":
        decision += " Modele full-train pret pour evaluation externe, sans metriques internes."
    protocol_lines = [
        "### Protocole d'entrainement",
        "",
        f"- Backend: `{result.get('backend_name') or 'non specifie'}`",
        f"- Tache: `{result.get('task_type') or 'non specifie'}`",
        f"- Representation: `{result.get('representation_name') or 'non specifie'}`",
        f"- Dataset: `{result.get('train_csv') or result.get('candidate_train_csv') or 'non specifie'}`",
        f"- Cibles: `{', '.join(map(str, result.get('target_columns') or [])) or 'non specifie'}`",
        f"- Strategie: `{protocol or 'non specifie'}`",
    ]
    if any(result.get(key) is not None for key in ("effective_train_count", "validation_count", "test_count")):
        protocol_lines.append(
            "- Split: "
            f"train={result.get('effective_train_count')}, "
            f"validation={result.get('validation_count')}, "
            f"test={result.get('test_count')}"
        )
    curation = result.get("curation") or {}
    curation_summary = (
        "### Dataset et curation\n\n"
        f"- Statut curation: `{curation.get('status') or 'non specifie'}`\n"
        f"- Pret QSAR: `{curation.get('ready_for_qsar') if curation.get('ready_for_qsar') is not None else 'non specifie'}`"
    )
    return {
        "decision_summary": decision,
        "dataset_curation_summary": curation_summary,
        "training_protocol_summary": "\n".join(protocol_lines),
        "evaluation_metrics_markdown": _metric_rows_markdown(
            rows=_training_metric_rows(result),
            task_type=str(result.get("task_type") or ""),
            metrics_status=str(metrics_status or ""),
        ),
        "applicability_domain_markdown": _ad_markdown(result.get("applicability_domain") or {}),
        "robustness_limits_summary": (
            "Validation utilisee pour diagnostic et reporting; aucune nouvelle politique "
            "d'hyperparametres n'est inventee par ce handoff."
        ),
        "governance_markdown": governance_markdown(
            requested_status=registry_payload.get("status"),
            final_status=registry_payload.get("status"),
            metrics_status=str(metrics_status or ""),
        ),
        "artifacts_inventory": _artifacts_markdown(result),
    }


def build_external_evaluation_reporting_handoff(result: Mapping[str, Any]) -> Dict[str, Any]:
    ad = result.get("applicability_domain") or {}
    label = result.get("evaluation_label") or result.get("evaluation_id") or "evaluation externe"
    return {
        "decision_summary": (
            f"Evaluation externe append-only `{result.get('evaluation_id')}` calculee sur "
            f"`{result.get('dataset_path')}`."
        ),
        "evaluation_metrics_markdown": _metric_rows_markdown(
            rows=_external_metric_rows(result),
            task_type=str(result.get("task_type") or ""),
        ),
        "applicability_domain_markdown": _ad_markdown(ad, source_label=str(label)),
        "governance_markdown": governance_markdown(
            final_status=result.get("status"),
            metrics_status=result.get("metrics_status"),
        ),
        "artifacts_inventory": _artifacts_markdown(result),
    }


def build_registry_reporting_handoff(
    *,
    requested_status: Optional[str],
    final_status: Optional[str],
    status_reason: Optional[str],
    metrics_status: Optional[str],
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "governance_markdown": governance_markdown(
            requested_status=requested_status,
            final_status=final_status,
            status_reason=status_reason,
            metrics_status=metrics_status,
        ),
        "artifacts_inventory": _artifacts_markdown(payload),
    }
