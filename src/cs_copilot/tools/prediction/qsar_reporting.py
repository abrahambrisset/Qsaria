#!/usr/bin/env python
# coding: utf-8
"""Canonical reporting handoffs for QSAR training and evaluation tools."""

from __future__ import annotations

import json
import math
from pathlib import Path
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

REPORT_FACTS_SCHEMA_VERSION = "2.0"

# A report handoff deliberately carries facts rather than a second copy of a
# training result.  Keeping this allowlist close to the renderer makes it much
# harder for trial histories, prediction frames, or feature matrices to leak
# back into the agent context after response compaction.
_CURATION_DETAIL_FIELDS = (
    "invalid_smiles_removed",
    "inorganic_rows_removed",
    "organometallic_rows_removed",
    "mixture_rows_removed",
    "salt_or_counterion_rows_processed",
    "duplicate_rows_removed",
    "duplicate_groups_detected",
    "duplicate_groups_aggregated",
    "duplicate_conflicting_groups",
    "duplicate_conflicting_rows_removed",
    "missing_target_removed",
    "non_numeric_target_removed",
    "infinite_target_removed",
    "stereochemistry_markers_removed",
    "constant_target_columns",
    "curation_actions",
    "curation_policy",
    "target_data_quality",
    "warnings",
    "blocking_issues",
)


def _json_safe(value: Any) -> Any:
    """Return a compact, serializable primitive without retaining opaque data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _read_local_json_mapping(value: Any) -> Dict[str, Any]:
    """Best-effort enrichment from a local curation report, never a hard dependency."""
    if not value or str(value).startswith(("s3://", "file://")):
        return {}
    try:
        path = Path(str(value)).expanduser()
        if not path.is_file():
            return {}
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return _mapping(payload)


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
    selection = result.get("selection_validation") or {}
    selection_metrics = selection.get("metrics") if isinstance(selection, Mapping) else None
    if isinstance(selection_metrics, Mapping):
        if any(
            subset in selection_metrics for subset in ("all", "in_domain", "out_of_domain")
        ):
            for subset in ("all", "in_domain", "out_of_domain"):
                metrics = selection_metrics.get(subset)
                if isinstance(metrics, Mapping) and metrics:
                    rows.append(
                        {
                            "source": "validation de sélection",
                            "target": target,
                            "ad_subset": subset,
                            "n": metrics.get("n"),
                            "metrics": metrics,
                        }
                    )
        elif selection_metrics:
            rows.append(
                {
                    "source": "validation de sélection",
                    "target": target,
                    "ad_subset": "all",
                    "n": selection_metrics.get("n"),
                    "metrics": selection_metrics,
                }
            )
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
    methods = ad.get("methods") if isinstance(ad.get("methods"), Mapping) else {}
    lines = [
        "### Domaine d'applicabilite",
        "",
        f"- Methode globale: `{ad.get('method') or ad.get('primary_method') or 'bounding_box'}`",
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
    if methods:
        reference_summary = {}
        if coverage_rows:
            reference_summary = coverage_rows[0][1].get("method_summaries") or coverage_rows[0][1].get(
                "method_status_summaries"
            ) or {}
        lines.extend(
            [
                "",
                "| Methode | Feature space | In-domain | Out-of-domain | Invalid | Coverage | Seuil / regle |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for method_name, method_payload in methods.items():
            method_summary = reference_summary.get(method_name) if isinstance(reference_summary, Mapping) else {}
            counts = (method_summary or {}).get("status_counts") or {}
            rule = (
                method_payload.get("rule")
                or method_payload.get("decision_rule")
                or f"threshold={method_payload.get('threshold')}"
            )
            lines.append(
                "| "
                + " | ".join(
                    [
                        _safe_cell(method_name),
                        _safe_cell(method_payload.get("feature_space") or ad.get("feature_space")),
                        _safe_cell(counts.get("in_domain")),
                        _safe_cell(counts.get("out_of_domain")),
                        _safe_cell(counts.get("invalid_features")),
                        _safe_cell((method_summary or {}).get("coverage_in_domain")),
                        _safe_cell(rule),
                    ]
                )
                + " |"
            )
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
        ("similarity_matrix", ad.get("similarity_matrix_manifest_path")),
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


def _canonical_metric_table(
    *,
    rows: List[Dict[str, Any]],
    task_type: str,
    include_source: bool,
) -> Optional[Dict[str, Any]]:
    """Build one factual metric table without a report-section heading."""
    if not rows:
        return None
    metric_columns = _metric_columns(task_type)
    targets = {row.get("target") for row in rows if row.get("target")}
    include_target = len(targets) > 1
    headers: List[str] = []
    if include_source:
        headers.append("Source")
    if include_target:
        headers.append("Target")
    headers.extend(["AD subset", "n", *[label for _, label in metric_columns]])
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    table_rows: List[Dict[str, Any]] = []
    for row in rows:
        metrics = _mapping(row.get("metrics"))
        cells: List[str] = []
        if include_source:
            cells.append(_safe_cell(row.get("source")))
        if include_target:
            cells.append(_safe_cell(row.get("target") or "-"))
        cells.extend(
            [
                _safe_cell(row.get("ad_subset")),
                _safe_cell(_metric_n(metrics, row.get("n"))),
                *[_safe_cell(metrics.get(key)) for key, _ in metric_columns],
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
        table_rows.append(
            {
                "source": row.get("source"),
                "target": row.get("target"),
                "ad_subset": row.get("ad_subset"),
                "n": _metric_n(metrics, row.get("n")),
                "metrics": _json_safe(metrics),
            }
        )
    return {"markdown": "\n".join(lines), "rows": table_rows}


def _selection_metric_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Selection-only rows, separate from final refit/test metrics."""
    selection = _mapping(result.get("selection_validation"))
    selection_metrics = _mapping(selection.get("metrics"))
    target = next(iter(result.get("target_columns") or []), None)
    rows: List[Dict[str, Any]] = []
    if any(key in selection_metrics for key in ("all", "in_domain", "out_of_domain")):
        for subset in ("all", "in_domain", "out_of_domain"):
            metrics = selection_metrics.get(subset)
            if isinstance(metrics, Mapping):
                rows.append(
                    {
                        "source": "selection_validation",
                        "target": target,
                        "ad_subset": subset,
                        "n": metrics.get("n"),
                        "metrics": metrics,
                    }
                )
    elif selection_metrics:
        rows.append(
            {
                "source": "selection_validation",
                "target": target,
                "ad_subset": "all",
                "n": selection_metrics.get("n"),
                "metrics": selection_metrics,
            }
        )
    return rows


def _variant_test_rows(result: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return final-test rows for persisted variants or multiple final fits only."""
    target = next(iter(result.get("target_columns") or []), None)
    rows: List[Dict[str, Any]] = []
    variants = result.get("outlier_model_variants") or []
    for variant in variants:
        if not isinstance(variant, Mapping):
            continue
        run = _mapping(variant.get("run"))
        label = str(variant.get("variant_id") or run.get("outlier_variant") or "final_variant")
        ad = _mapping(run.get("applicability_domain"))
        if not ad and label.endswith("baseline"):
            ad = _mapping(result.get("applicability_domain"))
        summary = _mapping(_mapping(ad.get("split_score_summaries")).get("test"))
        if summary:
            for row in _rows_from_ad_split(source=label, summary=summary, target=target):
                row["variant"] = label
                rows.append(row)
            continue
        test_metrics = _mapping(_mapping(run.get("metrics")).get("test"))
        if test_metrics:
            rows.append(
                {
                    "source": label,
                    "variant": label,
                    "target": target,
                    "ad_subset": "all",
                    "n": test_metrics.get("n"),
                    "metrics": test_metrics,
                }
            )

    # Cross-validation can create several final configurations without an
    # outlier variant wrapper.  A table is useful only when there is a real
    # external test and more than one configuration to compare.
    if not rows:
        split_results = result.get("split_results") or []
        for index, run in enumerate(split_results, start=1):
            if not isinstance(run, Mapping):
                continue
            label = str(run.get("strategy_label") or run.get("strategy") or f"configuration_{index}")
            ad = _mapping(run.get("applicability_domain"))
            summary = _mapping(_mapping(ad.get("split_score_summaries")).get("test"))
            if summary:
                for row in _rows_from_ad_split(source=label, summary=summary, target=target):
                    row["variant"] = label
                    rows.append(row)
    return rows


def _combined_ad_table(ad: Mapping[str, Any], *, source_label: str = "training") -> Optional[Dict[str, Any]]:
    """One table joining AD methods, flags, and coverage as the V2 canonical block."""
    ad = _mapping(ad)
    if not ad:
        return None
    methods = _mapping(ad.get("methods"))
    split_summaries = _mapping(ad.get("split_score_summaries"))
    split_items = [
        (name, _mapping(split_summaries.get(name)))
        for name in ("train", "validation", "test")
        if _mapping(split_summaries.get(name))
    ]
    if not split_items and ad.get("row_count") is not None:
        split_items = [(source_label, ad)]
    method_items = list(methods.items()) or [
        (str(ad.get("method") or ad.get("primary_method") or "not_specified"), {})
    ]
    if not split_items and not methods:
        return None

    headers = [
        "Method",
        "Feature space",
        "Rule",
        "Split",
        "n",
        "In-domain",
        "Out-of-domain",
        "Invalid features",
        "Coverage",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rows: List[Dict[str, Any]] = []
    for split_name, summary in split_items or [("-", {})]:
        method_summaries = _mapping(
            summary.get("method_summaries") or summary.get("method_status_summaries")
        )
        for method_name, payload in method_items:
            payload = _mapping(payload)
            method_summary = _mapping(method_summaries.get(method_name))
            counts = _mapping(method_summary.get("status_counts"))
            out_of_domain = counts.get("out_of_domain", summary.get("n_out_of_domain"))
            invalid = counts.get("invalid_features", summary.get("n_invalid_features"))
            row_count = summary.get("row_count")
            in_domain = counts.get("in_domain", summary.get("n_in_domain"))
            if in_domain is None and row_count is not None and out_of_domain is not None:
                try:
                    in_domain = max(0, int(row_count) - int(out_of_domain) - int(invalid or 0))
                except (TypeError, ValueError):
                    in_domain = None
            rule = payload.get("rule") or payload.get("decision_rule")
            if rule is None and payload.get("threshold") is not None:
                rule = f"threshold={payload.get('threshold')}"
            coverage = method_summary.get("coverage_in_domain", summary.get("coverage_in_domain"))
            lines.append(
                "| "
                + " | ".join(
                    _safe_cell(value)
                    for value in (
                        method_name,
                        payload.get("feature_space") or ad.get("feature_space"),
                        rule or "-",
                        split_name,
                        row_count,
                        in_domain,
                        out_of_domain,
                        invalid,
                        coverage,
                    )
                )
                + " |"
            )
            rows.append(
                {
                    "method": method_name,
                    "feature_space": payload.get("feature_space") or ad.get("feature_space"),
                    "rule": rule,
                    "split": split_name,
                    "n": row_count,
                    "n_in_domain": in_domain,
                    "n_out_of_domain": out_of_domain,
                    "n_invalid_features": invalid,
                    "coverage_in_domain": coverage,
                }
            )
    return {"markdown": "\n".join(lines), "rows": _json_safe(rows)}


def _normalized_curation_facts(result: Mapping[str, Any]) -> Dict[str, Any]:
    curation = _mapping(result.get("curation"))
    artifacts = _mapping(curation.get("artifacts") or curation.get("curation_artifacts"))
    report_payload = _mapping(curation.get("report") or curation.get("summary"))
    if not report_payload:
        report_payload = _read_local_json_mapping(
            curation.get("report_path") or artifacts.get("curation_report_json")
        )
    merged = {**report_payload, **curation}
    details = _mapping(merged.get("details"))
    facts: Dict[str, Any] = {
        "status": merged.get("status"),
        "ready_for_qsar": merged.get("ready_for_qsar"),
        "dataset_id": merged.get("dataset_id"),
        "source_dataset_path": merged.get("source_dataset_path"),
        "curated_dataset_path": merged.get("curated_dataset_path"),
        "task_type": merged.get("task_type"),
        "smiles_column_original": merged.get("smiles_column_original"),
        "smiles_column_curated": merged.get("smiles_column_curated"),
        "target_columns_original": merged.get("target_columns_original"),
        "target_columns_curated": merged.get("target_columns_curated"),
        "retained_columns": merged.get("retained_columns"),
        "rows_in": merged.get("rows_in"),
        "rows_out": merged.get("rows_out"),
        "rows_removed": merged.get("rows_removed"),
        "curation_backend": merged.get("curation_backend_used") or merged.get("curation_backend"),
        "backend_fallback_used": merged.get("curation_backend_fallback_used"),
        "backend_fallback_reason": merged.get("curation_backend_fallback_reason"),
        "identity_key_type": merged.get("curation_identity_key_type"),
        "artifacts": artifacts,
    }
    for field in _CURATION_DETAIL_FIELDS:
        value = merged.get(field, details.get(field))
        if value is not None:
            facts[field] = value
    return _json_safe({key: value for key, value in facts.items() if value is not None})


def _outlier_selection_annotation_facts(outliers: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the compact selection annotation facts without retaining its CSV."""
    outliers = _mapping(outliers)
    artifacts = _mapping(outliers.get("artifacts"))
    persisted = _read_local_json_mapping(
        outliers.get("summary_path") or artifacts.get("summary_path")
    )
    return _mapping(
        outliers.get("selection_annotations")
        or persisted.get("selection_annotations")
    )


def _normalized_activity_cliff_facts(result: Mapping[str, Any]) -> Dict[str, Any]:
    activity_cliffs = _mapping(result.get("activity_cliffs"))
    if not activity_cliffs:
        selection_annotations = _outlier_selection_annotation_facts(
            _mapping(result.get("outlier_analysis"))
        )
        selection_ac = _mapping(selection_annotations.get("activity_cliffs"))
        if selection_ac.get("executed"):
            return _json_safe(
                {
                    "status": "selection_annotation_only",
                    "enabled": True,
                    "scope": selection_ac.get("scope") or "development_only",
                    "flagged_count": selection_ac.get("flagged_count"),
                    "feedback_loops_requested": 0,
                    "note": (
                        "Activity Cliff annotation was computed automatically on development data "
                        "for outlier selection; no Activity Cliff feedback retraining was requested."
                    ),
                }
            )
        return {"status": "not_run", "reason": "No Activity Cliff output was produced."}
    facts = {
        "status": "completed",
        "enabled": bool(activity_cliffs.get("enabled")),
        "mode": activity_cliffs.get("mode"),
        "index_name": activity_cliffs.get("index_name"),
        "target_column": activity_cliffs.get("target_column"),
        "smiles_column": activity_cliffs.get("smiles_column"),
        "flagged_count": activity_cliffs.get("flagged_count"),
        "priority_counts": activity_cliffs.get("priority_counts"),
        "index_parameters": activity_cliffs.get("index_parameters"),
        "tiering_policy": activity_cliffs.get("tiering_policy"),
        "evaluation_policy": activity_cliffs.get("evaluation_policy"),
        "selection_policy": activity_cliffs.get("selection_policy"),
        "feedback_loops_requested": activity_cliffs.get("feedback_loops_requested"),
        "warnings": activity_cliffs.get("warnings"),
        "summary_path": activity_cliffs.get("summary_path"),
        "annotated_training_csv": activity_cliffs.get("annotated_training_csv"),
        "plot_artifacts": activity_cliffs.get("plot_artifacts"),
    }
    return _json_safe({key: value for key, value in facts.items() if value is not None})


def _normalized_ad_facts(ad: Mapping[str, Any]) -> Dict[str, Any]:
    ad = _mapping(ad)
    if not ad:
        return {"status": "not_available"}
    methods = []
    for name, payload in _mapping(ad.get("methods")).items():
        payload = _mapping(payload)
        methods.append(
            {
                "name": name,
                "feature_space": payload.get("feature_space"),
                "rule": payload.get("rule") or payload.get("decision_rule"),
                "threshold": payload.get("threshold"),
            }
        )
    coverage = []
    for split, payload in _mapping(ad.get("split_score_summaries")).items():
        payload = _mapping(payload)
        coverage.append(
            {
                "split": split,
                "row_count": payload.get("row_count"),
                "n_in_domain": payload.get("n_in_domain"),
                "n_out_of_domain": payload.get("n_out_of_domain"),
                "n_invalid_features": payload.get("n_invalid_features"),
                "coverage_in_domain": payload.get("coverage_in_domain"),
            }
        )
    return _json_safe(
        {
            "status": "available",
            "primary_method": ad.get("method") or ad.get("primary_method"),
            "feature_space": ad.get("feature_space") or ad.get("representation_name"),
            "feature_count": ad.get("feature_count"),
            "fit_row_count": ad.get("fit_row_count") or ad.get("reference_size"),
            "methods": methods,
            "coverage": coverage,
            "artifacts": {
                key: ad.get(key)
                for key in (
                    "manifest_path",
                    "bounds_path",
                    "similarity_matrix_manifest_path",
                    "scores_train_path",
                    "scores_validation_path",
                    "scores_test_path",
                )
                if ad.get(key)
            },
        }
    )


def _normalized_tuning_facts(result: Mapping[str, Any]) -> Dict[str, Any]:
    tuning = _mapping(result.get("hyperparameter_tuning"))
    if not tuning:
        return {"status": "not_run", "reason": "No hyperparameter tuning output was produced."}
    best_trial = _mapping(tuning.get("best_trial"))
    objective = _mapping(tuning.get("objective"))
    return _json_safe(
        {
            "status": tuning.get("status") or "completed",
            "engine": tuning.get("engine"),
            "engine_version": tuning.get("engine_version"),
            "requested_trials": tuning.get("requested_trials"),
            "completed_trials": tuning.get("completed_trials"),
            "failed_trials": tuning.get("failed_trials"),
            "seed": tuning.get("seed"),
            "objective": objective,
            "selection_protocol": tuning.get("selection_protocol"),
            "sampler": tuning.get("sampler"),
            "pruner": tuning.get("pruner"),
            "best_trial": {
                key: best_trial.get(key)
                for key in ("number", "value", "params", "metrics")
                if best_trial.get(key) is not None
            },
            "summary_path": tuning.get("summary_path")
            or result.get("hyperparameter_tuning_summary_path"),
            "plot_artifacts": tuning.get("plot_artifacts"),
        }
    )


def _normalized_outlier_facts(result: Mapping[str, Any]) -> Dict[str, Any]:
    outliers = _mapping(result.get("outlier_analysis"))
    if not outliers:
        return {"status": "not_run", "reason": "No outlier analysis output was produced."}
    studies = []
    for study in outliers.get("studies") or []:
        if not isinstance(study, Mapping):
            continue
        studies.append(
            {
                key: study.get(key)
                for key in (
                    "fold_label",
                    "repeat_index",
                    "task_type",
                    "eligible_count",
                    "selected_count",
                    "selection_fraction",
                    "rmse_multiplier",
                    "reason",
                )
                if study.get(key) is not None
            }
        )
    variants = []
    for variant in result.get("outlier_model_variants") or []:
        if not isinstance(variant, Mapping):
            continue
        variants.append(
            {
                key: variant.get(key)
                for key in ("variant_id", "split_label", "repeat_index", "fold_sources")
                if variant.get(key) is not None
            }
        )
    selection_annotations = _outlier_selection_annotation_facts(outliers)
    return _json_safe(
        {
            "status": outliers.get("status") or ("completed" if outliers.get("enabled") else "skipped"),
            "enabled": outliers.get("enabled"),
            "reason": outliers.get("reason"),
            "config": outliers.get("config"),
            "task_type": outliers.get("task_type"),
            "eligible_count": outliers.get("eligible_count"),
            "selected_count": outliers.get("selected_count"),
            "selection_fraction": outliers.get("selection_fraction"),
            "rmse_multiplier": outliers.get("rmse_multiplier"),
            "studies": studies,
            "final_variants": variants,
            "variant_count": len(variants),
            "selection_annotations": selection_annotations,
            "artifacts": outliers.get("artifacts"),
            "plot_artifacts": outliers.get("plot_artifacts"),
        }
    )


def _compact_fold_and_campaign_facts(result: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep comparable fold/campaign evidence without forwarding run payloads."""
    folds = []
    for index, run in enumerate(result.get("split_results") or [], start=1):
        if not isinstance(run, Mapping):
            continue
        tuning = _mapping(run.get("hyperparameter_tuning"))
        best_trial = _mapping(tuning.get("best_trial"))
        selection = _mapping(run.get("selection_validation"))
        folds.append(
            _json_safe(
                {
                    "label": run.get("strategy_label") or run.get("strategy") or f"split_{index}",
                    "strategy_family": run.get("strategy_family"),
                    "repeat_index": run.get("repeat_index"),
                    "seed": run.get("seed"),
                    "metrics": run.get("metrics"),
                    "selection_validation": {
                        key: selection.get(key)
                        for key in ("label", "objective", "metrics", "applicability_domain_used")
                        if selection.get(key) is not None
                    }
                    if selection
                    else None,
                    "selected_hyperparameters": run.get("selected_hyperparameters"),
                    "tuning": {
                        "engine": tuning.get("engine"),
                        "requested_trials": tuning.get("requested_trials"),
                        "completed_trials": tuning.get("completed_trials"),
                        "failed_trials": tuning.get("failed_trials"),
                        "best_trial": {
                            key: best_trial.get(key)
                            for key in ("number", "value", "params")
                            if best_trial.get(key) is not None
                        },
                    }
                    if tuning
                    else None,
                    "outlier_analysis": _normalized_outlier_facts({"outlier_analysis": run.get("outlier_analysis")})
                    if run.get("outlier_analysis")
                    else None,
                }
            )
        )
    candidates = []
    for candidate in result.get("candidate_results") or result.get("ranking") or []:
        if not isinstance(candidate, Mapping):
            continue
        candidates.append(
            _json_safe(
                {
                    key: candidate.get(key)
                    for key in (
                        "rank",
                        "candidate_id",
                        "backend_name",
                        "representation_name",
                        "strategy",
                        "strategy_label",
                        "random_r2",
                        "scaffold_r2",
                        "hardest_split",
                        "hardest_split_r2",
                        "training_duration_seconds",
                        "feature_preparation_duration_seconds",
                    )
                    if candidate.get(key) is not None
                }
            )
        )
    return {
        "cross_validation": _json_safe(
            {
                key: _mapping(result.get("cross_validation")).get(key)
                for key in ("summary", "n_folds", "n_repeats", "outer_test_size")
                if _mapping(result.get("cross_validation")).get(key) is not None
            }
        ),
        "folds": folds,
        "campaign": _json_safe(
            {
                "type": result.get("campaign_type"),
                "recommended_candidate": result.get("recommended_candidate"),
                "recommended_representation_name": result.get("recommended_representation_name"),
                "candidate_count": len(candidates),
                "duration_seconds": result.get("campaign_duration_seconds"),
                "candidates": candidates,
            }
        )
        if candidates or result.get("campaign_type")
        else {},
    }


def _evaluation_scope(result: Mapping[str, Any], *, metrics_status: str) -> Dict[str, Any]:
    strategy_type = str(result.get("validation_strategy_type") or "").lower()
    strategy = _mapping(result.get("validation_strategy"))
    ad_summaries = _mapping(_mapping(result.get("applicability_domain")).get("split_score_summaries"))
    has_external_test = bool(
        result.get("test_count")
        or _mapping(ad_summaries.get("test"))
        or strategy.get("outer_test_size")
    )
    if metrics_status == "not_evaluated" or str(result.get("validation_protocol") or "").lower() == "full_train":
        return {
            "scope": "none",
            "heading_key": "no_internal_evaluation",
            "statement": "The model was trained in full-train mode; no internal evaluation is available.",
        }
    if strategy_type in {"cross_validation", "repeated_cross_validation"} and not has_external_test:
        return {
            "scope": "internal",
            "heading_key": "internal_test_results",
            "statement": (
                "Results are internal out-of-fold evidence. No external test set was used, and "
                "final refits are not independently evaluated."
            ),
        }
    return {
        "scope": "external",
        "heading_key": "external_test_results",
        "statement": "Final results use an isolated external test split that was not used for selection.",
    }


def _training_report_facts(result: Mapping[str, Any], *, metrics_status: str) -> Dict[str, Any]:
    protocol = result.get("validation_protocol") or result.get("validation_strategy_type")
    registry = _mapping(result.get("recommended_registry_payload"))
    return _json_safe(
        {
            "schema_version": REPORT_FACTS_SCHEMA_VERSION,
            "report_kind": "training",
            "dataset": {
                "path": result.get("train_csv") or result.get("candidate_train_csv"),
                "smiles_column": result.get("smiles_column"),
                "target_columns": result.get("target_columns") or [],
                "task_type": result.get("task_type"),
                "curation": _normalized_curation_facts(result),
                "activity_cliffs": _normalized_activity_cliff_facts(result),
                "applicability_domain": _normalized_ad_facts(result.get("applicability_domain") or {}),
            },
            "working_environment": {
                "training_profile": result.get("training_profile"),
                "profile_reason": result.get("profile_reason"),
                "compute_environment": result.get("compute_environment"),
                "training_resources": result.get("training_resources"),
                "training_durations": result.get("training_durations"),
                "training_duration_seconds": result.get("training_duration_seconds"),
            },
            "training_protocol": {
                "backend": result.get("backend_name"),
                "task_type": result.get("task_type"),
                "representation": result.get("representation_name"),
                "validation_protocol": protocol,
                "validation_strategy_type": result.get("validation_strategy_type"),
                "validation_strategy": result.get("validation_strategy"),
                "train_count": result.get("effective_train_count"),
                "validation_count": result.get("validation_count"),
                "test_count": result.get("test_count"),
                "seed_policy": result.get("seed_policy"),
                "final_refit": result.get("final_refit"),
                "metrics_status": metrics_status,
            },
            "hyperparameter_optimization": _normalized_tuning_facts(result),
            "outlier_analysis": _normalized_outlier_facts(result),
            "validation_studies": _compact_fold_and_campaign_facts(result),
            "evaluation": _evaluation_scope(result, metrics_status=metrics_status),
            "final_variants": _normalized_outlier_facts(result).get("final_variants") or [],
            "governance": {
                "requested_status": registry.get("status"),
                "final_status": registry.get("status"),
                "metrics_status": metrics_status,
                "catalog_model_policy": result.get("catalog_model_policy"),
                "persistence_plan": result.get("persistence_plan"),
                "recommended_model_id": registry.get("model_id"),
            },
            "artifacts": _json_safe(
                {
                    key: result.get(key)
                    for key in (
                        "training_bundle",
                        "bundle_file_ref",
                        "output_dir",
                        "summary_path",
                        "canonical_summary_path",
                        "config_path",
                        "splits_path",
                        "validation_predictions_path",
                        "test_predictions_path",
                        "model_root",
                        "model_path",
                        "metadata_path",
                        "candidate_manifest_path",
                        "outlier_model_variants_manifest_path",
                    )
                    if result.get(key)
                }
            ),
        }
    )


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
    tuning = result.get("hyperparameter_tuning") or {}
    if isinstance(tuning, Mapping):
        objective = tuning.get("objective") or {}
        protocol_lines.extend(
            [
                "- Tuning: "
                f"moteur=`{tuning.get('engine') or 'non specifie'}`, "
                f"trials={tuning.get('completed_trials')}/{tuning.get('requested_trials')}, "
                f"objectif=`{objective.get('metric') or 'non specifie'}` "
                f"sur `{objective.get('subset') or 'all'}`.",
                "- Sélection: validation de sélection; test final réservé au refit train + validation.",
            ]
        )
        selection_protocol = tuning.get("selection_protocol") or {}
        if selection_protocol.get("applicability_domain_used_for_selection") is False:
            protocol_lines.append(
                "- Note Chemprop: la sélection utilise `val_loss` globale native; l'AD n'influence pas le classement."
            )
    curation = result.get("curation") or {}
    curation_summary = (
        "### Dataset et curation\n\n"
        f"- Statut curation: `{curation.get('status') or 'non specifie'}`\n"
        f"- Pret QSAR: `{curation.get('ready_for_qsar') if curation.get('ready_for_qsar') is not None else 'non specifie'}`"
    )
    selection_rows = _selection_metric_rows(result)
    variant_rows = _variant_test_rows(result)
    variants = {row.get("variant") or row.get("source") for row in variant_rows}
    report_tables: Dict[str, Any] = {}
    ad_table = _combined_ad_table(result.get("applicability_domain") or {})
    if ad_table:
        report_tables["applicability_domain"] = {
            "title": "Applicability Domain methods, flags and coverage",
            "kind": "applicability_domain",
            **ad_table,
        }
    selection_table = _canonical_metric_table(
        rows=selection_rows,
        task_type=str(result.get("task_type") or ""),
        include_source=False,
    )
    if selection_table:
        report_tables["selection_validation_metrics"] = {
            "title": "Selection-validation metrics",
            "kind": "selection_validation_metrics",
            **selection_table,
        }
    # The final comparison is intentionally conditional: a single final
    # baseline does not become a redundant second metrics table.
    if len(variants) > 1:
        variant_table = _canonical_metric_table(
            rows=variant_rows,
            task_type=str(result.get("task_type") or ""),
            include_source=True,
        )
        if variant_table:
            report_tables["final_test_comparison"] = {
                "title": "Final test comparison",
                "kind": "final_test_comparison",
                **variant_table,
            }
    report_facts = _training_report_facts(result, metrics_status=str(metrics_status or ""))
    report_facts["tables_present"] = list(report_tables)
    report_facts["table_policy"] = {
        "maximum": 3,
        "allowed": [
            "applicability_domain",
            "selection_validation_metrics",
            "final_test_comparison",
        ],
    }
    return {
        "report_facts": report_facts,
        "report_tables": report_tables,
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
    metric_rows = _external_metric_rows(result)
    report_tables: Dict[str, Any] = {}
    ad_table = _combined_ad_table(ad, source_label=str(label))
    if ad_table:
        report_tables["applicability_domain"] = {
            "title": "Applicability Domain methods, flags and coverage",
            "kind": "applicability_domain",
            **ad_table,
        }
    metric_table = _canonical_metric_table(
        rows=metric_rows,
        task_type=str(result.get("task_type") or ""),
        include_source=False,
    )
    if metric_table:
        report_tables["external_test_results"] = {
            "title": "External test results",
            "kind": "external_test_results",
            **metric_table,
        }
    report_facts = _json_safe(
        {
            "schema_version": REPORT_FACTS_SCHEMA_VERSION,
            "report_kind": "standalone_labelled_evaluation",
            "evaluation": {
                "evaluation_id": result.get("evaluation_id"),
                "evaluation_label": label,
                "dataset_path": result.get("dataset_path"),
                "row_count": result.get("row_count"),
                "task_type": result.get("task_type"),
                "target_columns": result.get("target_columns") or [],
                "scope": "external",
                "statement": "This is an append-only labelled external evaluation.",
            },
            "applicability_domain": _normalized_ad_facts(ad),
            "governance": {
                "model_id": result.get("model_id"),
                "status": result.get("status"),
                "metrics_status": result.get("metrics_status"),
            },
            "artifacts": result.get("artifacts") or {},
            "tables_present": list(report_tables),
            "table_policy": {
                "maximum": 2,
                "allowed": ["applicability_domain", "external_test_results"],
            },
        }
    )
    return {
        "report_facts": report_facts,
        "report_tables": report_tables,
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
