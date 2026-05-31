#!/usr/bin/env python
# coding: utf-8
"""
Toolkit for canonical QSAR report payload and LaTeX export.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.agents.qsar_report_payload import (
    add_bullets_block,
    add_files_block,
    add_kv_block,
    add_paragraph_block,
    add_section,
    add_table_block,
    init_report_payload,
)
from cs_copilot.tools.prediction.catalog import PredictionModelCatalog

from .qsar_latex import write_latex_report, write_payload_json


def _get_report_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("qsar_report", {})
    state.setdefault("last_request", {})
    state.setdefault("last_result", {})
    return state


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}, ()):
            return value
    return None


def _format_metric_value(value: Any) -> Optional[str]:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return None


def _format_metric_triplet(metrics: Dict[str, Any]) -> Optional[str]:
    r2 = _format_metric_value(metrics.get("r2") or metrics.get("r2_mean"))
    rmse = _format_metric_value(metrics.get("rmse") or metrics.get("rmse_mean"))
    mae = _format_metric_value(metrics.get("mae") or metrics.get("mae_mean"))
    parts: List[str] = []
    if r2 is not None:
        parts.append(f"R² = {r2}")
    if rmse is not None:
        parts.append(f"RMSE = {rmse}")
    if mae is not None:
        parts.append(f"MAE = {mae}")
    return ", ".join(parts) if parts else None


def _extract_validation_summary(
    record: Any,
    metadata_payload: Optional[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    metric_source = {}
    if metadata_payload:
        metric_source = dict(metadata_payload.get("known_metrics") or {})
    if not metric_source and record is not None:
        metric_source = dict(getattr(record, "known_metrics", {}) or {})

    protocol_name = _first_non_empty(
        metadata_payload.get("validation_protocol") if metadata_payload else None,
        (metadata_payload.get("training_data") or {}).get("validation_protocol")
        if metadata_payload
        else None,
        getattr(record, "training_data_summary", {}).get("validation_protocol")
        if record is not None
        else None,
    )

    scaffold_metrics = (
        metric_source.get("scaffold")
        or metric_source.get("scaffold_test")
        or metric_source.get("scaffold_split")
        or {}
    )
    random_metrics = (
        metric_source.get("random")
        or metric_source.get("random_test")
        or metric_source.get("random_split")
        or {}
    )
    cluster_metrics = (
        metric_source.get("cluster")
        or metric_source.get("cluster_test")
        or metric_source.get("cluster_kmeans_split")
        or {}
    )
    test_metrics = metric_source.get("test") or {}

    reference_label = None
    reference_metrics: Dict[str, Any] = {}
    if scaffold_metrics:
        reference_label = "scaffold"
        reference_metrics = scaffold_metrics
    elif cluster_metrics:
        reference_label = "cluster"
        reference_metrics = cluster_metrics
    elif random_metrics:
        reference_label = "random"
        reference_metrics = random_metrics
    elif test_metrics:
        reference_label = "test"
        reference_metrics = test_metrics

    random_summary = None
    if random_metrics:
        random_triplet = _format_metric_triplet(random_metrics)
        random_r2_std = _format_metric_value(random_metrics.get("r2_std"))
        random_rmse_std = _format_metric_value(random_metrics.get("rmse_std"))
        if random_triplet and random_r2_std is not None and random_rmse_std is not None:
            random_summary = (
                f"{random_triplet}, R² std = {random_r2_std}, RMSE std = {random_rmse_std}"
            )
        else:
            random_summary = random_triplet

    return {
        "protocol": str(protocol_name) if protocol_name else None,
        "reference_split": reference_label,
        "reference_metrics": _format_metric_triplet(reference_metrics) if reference_metrics else None,
        "random_metrics": random_summary,
    }


class QSARReportingToolkit(Toolkit):
    def __init__(self):
        super().__init__("qsar_reporting")
        self.register(self.init_qsar_report_payload)
        self.register(self.append_qsar_report_section)
        self.register(self.build_prediction_report_payload)
        self.register(self.export_qsar_latex_report)
        self.register(self.export_latest_prediction_report_bundle)

    def init_qsar_report_payload(
        self,
        *,
        report_type: str,
        title: str,
        intro: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        payload = init_report_payload(
            report_type=report_type,
            title=title,
            intro=intro,
            metadata=metadata or {},
        )
        if agent is not None:
            state = _get_report_state(agent)
            state["last_request"] = {
                "report_type": report_type,
                "title": title,
            }
            state["last_result"]["report_payload"] = payload
        return payload

    def append_qsar_report_section(
        self,
        *,
        payload: Dict[str, Any],
        section_title: str,
        blocks: List[Dict[str, Any]],
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        section = add_section(payload, title=section_title)
        for block in blocks:
            block_type = block.get("type")
            if block_type == "paragraph":
                add_paragraph_block(section, title=block.get("title", ""), text=block.get("text", ""))
            elif block_type == "bullets":
                add_bullets_block(section, title=block.get("title", ""), items=block.get("items", []))
            elif block_type == "table":
                add_table_block(
                    section,
                    title=block.get("title", ""),
                    columns=block.get("columns", []),
                    rows=block.get("rows", []),
                )
            elif block_type == "kv_list":
                add_kv_block(section, title=block.get("title", ""), items=block.get("items", []))
            elif block_type == "files":
                add_files_block(section, title=block.get("title", ""), items=block.get("items", []))
        if agent is not None:
            _get_report_state(agent)["last_result"]["report_payload"] = payload
        return payload

    def build_prediction_report_payload(
        self,
        *,
        max_preview_rows: int = 10,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        if agent is None:
            raise ValueError("Agent is required to build a prediction report payload")

        prediction_state = agent.session_state.get("prediction_models", {})
        history = prediction_state.get("prediction_history") or []
        if not history:
            raise ValueError("No prediction history is available to build a report payload")

        latest = history[-1]
        preds_path = latest.get("preds_path")
        model_id = latest.get("model_id")
        if not preds_path or not Path(preds_path).exists():
            raise ValueError("Latest prediction file is not available")

        catalog = PredictionModelCatalog.load()
        catalog.refresh_from_internal_store(persist=False)
        record_payload = catalog.models.get(model_id)
        record = record_payload if record_payload is not None else None

        df = pd.read_csv(preds_path)
        ad_columns = latest.get("applicability_domain_columns") or []
        ad_summary = latest.get("applicability_domain") or {}

        target_columns = list(getattr(getattr(record, "task", None), "target_columns", []) or [])
        prediction_column_labels: Dict[str, str] = {}
        for target in target_columns:
            explicit_prediction = f"{target}_prediction"
            if explicit_prediction in df.columns:
                label = f"Y predit ({target})" if len(target_columns) > 1 else "Y predit"
                prediction_column_labels[explicit_prediction] = label
            elif target in df.columns:
                label = f"Y predit ({target})" if len(target_columns) > 1 else "Y predit"
                prediction_column_labels[target] = label
        if not prediction_column_labels:
            inferred_prediction_columns = [
                column for column in df.columns if str(column).endswith("_prediction")
            ]
            for column in inferred_prediction_columns:
                target = str(column).removesuffix("_prediction")
                label = f"Y predit ({target})" if len(inferred_prediction_columns) > 1 else "Y predit"
                prediction_column_labels[column] = label
        target_column = next(iter(prediction_column_labels), None)
        if target_column is None:
            auxiliary_columns = {"smiles", "source_row_index", *ad_columns}
            non_aux = [
                column
                for column in df.columns
                if column not in auxiliary_columns
                and not str(column).endswith("_true")
                and not str(column).endswith("_residual")
                and not str(column).endswith("_absolute_error")
                and "replicate" not in str(column)
            ]
            target_column = non_aux[0] if non_aux else None
            if target_column:
                prediction_column_labels[target_column] = "Y predit"

        reliability_map = {
            "in_domain": "Elevee",
            "edge_of_domain": "Moderee",
            "out_of_domain": "Faible",
        }
        recommendation_map = {
            "in_domain": "Prediction fiable",
            "edge_of_domain": "Prediction a interpreter avec prudence",
            "out_of_domain": "Prediction indicative uniquement",
        }

        report_type = "prediction"
        title = "Rapport des predictions QSAR"
        intro = "Voici les resultats des predictions pour les molecules fournies."
        metadata = {
            "model_id": model_id or "",
            "display_name": getattr(record, "display_name", None) or model_id or "",
            "final_status": getattr(record, "status", None) or "",
            "generated_date": pd.Timestamp.now(tz="Europe/Paris").strftime("%d/%m/%Y"),
        }
        payload = init_report_payload(
            report_type=report_type,
            title=title,
            intro=intro,
            metadata=metadata,
        )

        model_section = add_section(payload, title="Modele utilise")
        model_items: List[List[str]] = [
            ["Identifiant", model_id or "non_disponible"],
            ["Nom d'affichage", getattr(record, "display_name", None) or (model_id or "non_disponible")],
            ["Statut de gouvernance", getattr(record, "status", None) or "non_disponible"],
            ["Backend", getattr(record, "backend_name", None) or latest.get("backend_name", "non_disponible")],
        ]
        if record and record.task.task_type:
            model_items.append(["Type de tache", record.task.task_type])
        if target_columns:
            label = "Cibles" if len(target_columns) > 1 else "Cible"
            model_items.append([label, ", ".join(target_columns)])
        elif target_column:
            model_items.append(["Cible", target_column])
        metadata_payload: Dict[str, Any] = {}
        if record and record.metadata_path and Path(record.metadata_path).exists():
            try:
                import json

                metadata_payload = json.loads(Path(record.metadata_path).read_text())
                if metadata_payload.get("trained_date"):
                    model_items.append(["Date de l'entrainement", metadata_payload["trained_date"]])
                if metadata_payload.get("trained_time"):
                    model_items.append(["Heure de l'entrainement", metadata_payload["trained_time"]])
            except Exception:
                metadata_payload = {}

        validation_summary = _extract_validation_summary(record, metadata_payload)
        if validation_summary.get("protocol"):
            model_items.append(["Protocole de validation", validation_summary["protocol"]])
        if validation_summary.get("reference_split"):
            model_items.append(["Split de reference", validation_summary["reference_split"]])
        if validation_summary.get("reference_metrics"):
            model_items.append(["Metriques de reference", validation_summary["reference_metrics"]])
        if validation_summary.get("random_metrics"):
            model_items.append(["Metriques random", validation_summary["random_metrics"]])

        if ad_summary:
            method = ad_summary.get("method", "hybrid_morgan_domain")
            ref_size = ad_summary.get("train_size") or ad_summary.get("reference_size")
            model_items.append(
                [
                    "Domaine d'applicabilite",
                    f"{method} ({ref_size} molecules de reference)" if ref_size else method,
                ]
            )
        add_kv_block(model_section, title="Modele utilise", items=model_items)

        prediction_section = add_section(payload, title="Resultats des predictions")
        preview_df = df.copy()
        if "ad_status" in preview_df.columns:
            preview_df["Fiabilite"] = preview_df["ad_status"].map(reliability_map).fillna("")
        if "smiles" in preview_df.columns:
            preview_df = preview_df.rename(columns={"smiles": "SMILES"})
        if prediction_column_labels:
            preview_df = preview_df.rename(columns=prediction_column_labels)
        if "ad_status" in preview_df.columns:
            preview_df = preview_df.rename(columns={"ad_status": "Statut AD"})
        preview_columns: List[str] = []
        if "SMILES" in preview_df.columns:
            preview_columns.append("SMILES")
        for prediction_label in prediction_column_labels.values():
            if prediction_label in preview_df.columns:
                preview_columns.append(prediction_label)
        if "Statut AD" in preview_df.columns:
            preview_columns.append("Statut AD")
        if "Fiabilite" in preview_df.columns:
            preview_columns.append("Fiabilite")
        if not preview_columns:
            preview_columns = list(preview_df.columns[: min(4, len(preview_df.columns))])
        preview_rows = (
            preview_df[preview_columns]
            .head(max_preview_rows)
            .fillna("")
            .astype(str)
            .values.tolist()
        )
        add_table_block(
            prediction_section,
            title="Resultats des predictions",
            columns=preview_columns,
            rows=preview_rows,
        )

        ad_section = add_section(payload, title="Evaluation de fiabilite par statut AD")
        ad_rows: List[List[str]] = []
        if "ad_status" in df.columns:
            counts = df["ad_status"].fillna("non_disponible").value_counts().to_dict()
            for status in ["in_domain", "edge_of_domain", "out_of_domain"]:
                count = counts.get(status, 0)
                percentage = (count / len(df) * 100.0) if len(df) else 0.0
                ad_rows.append(
                    [
                        status,
                        str(count),
                        f"{percentage:.0f}%",
                        reliability_map.get(status, ""),
                        recommendation_map.get(status, ""),
                    ]
                )
        else:
            ad_rows.append(
                ["non_disponible", str(len(df)), "100%", "Non disponible", "Statut AD non disponible"]
            )
        add_table_block(
            ad_section,
            title="Evaluation de fiabilite par statut AD",
            columns=["Statut AD", "Nombre", "Fraction", "Fiabilite", "Interpretation"],
            rows=ad_rows,
        )
        if ad_summary.get("thresholds"):
            thresholds = ad_summary["thresholds"]
            threshold_items: List[str] = []
            if thresholds.get("in_domain_min") is not None:
                threshold_items.append(f"Seuil in_domain : {thresholds['in_domain_min']}")
            if thresholds.get("edge_domain_min") is not None:
                threshold_items.append(f"Seuil edge_of_domain : {thresholds['edge_domain_min']}")
            if threshold_items:
                add_bullets_block(
                    ad_section,
                    title="Seuils de similarite",
                    items=threshold_items,
                )

        summary_section = add_section(payload, title="Resume statistique")
        summary_items = [f"Nombre de molecules : {len(df)}"]
        if target_column and target_column in df.columns:
            values = pd.to_numeric(df[target_column], errors="coerce").dropna()
            if not values.empty:
                summary_items.extend(
                    [
                        f"Valeur minimale : {values.min():.3f}",
                        f"Valeur maximale : {values.max():.3f}",
                        f"Moyenne : {values.mean():.3f}",
                        f"Ecart-type : {values.std(ddof=0):.3f}",
                    ]
                )
        if "ad_status" in df.columns:
            counts = df["ad_status"].value_counts(dropna=False).to_dict()
            for status in ["in_domain", "edge_of_domain", "out_of_domain"]:
                if status in counts:
                    summary_items.append(f"{status} : {counts[status]}")
        add_bullets_block(summary_section, title="Resume statistique", items=summary_items)

        interpretation_section = add_section(payload, title="Interpretation des valeurs Y")
        interpretation_items = [
            "Echelle : unitless_log_scale",
            "Valeurs positives : lipophilicite plus elevee.",
            "Valeurs negatives : lipophilicite plus faible.",
        ]
        if target_column:
            interpretation_items.append(f"Colonne de prediction : {target_column}")
        add_bullets_block(
            interpretation_section,
            title="Interpretation des valeurs Y",
            items=interpretation_items,
        )

        files_section = add_section(payload, title="Fichier de resultats complet")
        file_items = [{"label": "CSV des predictions", "path": str(Path(preds_path).resolve()), "kind": "prediction_csv"}]
        add_files_block(files_section, title="Fichier de resultats complet", items=file_items)

        recommendations_section = add_section(payload, title="Recommandations d'utilisation")
        recommendations = [
            "Utilisable pour la priorisation de molecules `in_domain`.",
            "A interpreter avec prudence pour les molecules `edge_of_domain`.",
            "Non recommande pour les molecules `out_of_domain` sans validation supplementaire.",
        ]
        add_bullets_block(
            recommendations_section,
            title="Recommandations d'utilisation",
            items=recommendations,
        )

        state = _get_report_state(agent)
        state["last_request"] = {"report_type": report_type, "title": title}
        state["last_result"]["report_payload"] = payload
        return payload

    def export_qsar_latex_report(
        self,
        *,
        payload: Dict[str, Any],
        output_dir: Optional[str] = None,
        basename: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        root = (
            Path(output_dir).expanduser().resolve()
            if output_dir
            else (Path(".files") / "qsar_reports").resolve()
        )
        root.mkdir(parents=True, exist_ok=True)
        name = basename or f"{payload.get('report_type', 'qsar_report')}_report"
        tex_path = root / f"{name}.tex"
        json_path = root / f"{name}.payload.json"
        latex_result = write_latex_report(payload, str(tex_path))
        payload_result = write_payload_json(payload, str(json_path))
        result = {
            **latex_result,
            **payload_result,
        }
        if agent is not None:
            state = _get_report_state(agent)
            state["last_result"]["report_payload"] = payload
            state["last_result"]["latex_report_path"] = result["report_path"]
            state["last_result"]["report_payload_path"] = result["payload_path"]
        return result

    def export_latest_prediction_report_bundle(
        self,
        *,
        output_dir: Optional[str] = None,
        basename: Optional[str] = None,
        max_preview_rows: int = 10,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        if agent is None:
            raise ValueError("Agent is required to export the latest prediction report bundle")

        payload = self.build_prediction_report_payload(
            max_preview_rows=max_preview_rows,
            agent=agent,
        )
        model_id = payload.get("metadata", {}).get("model_id") or "prediction"
        safe_name = str(model_id).replace("/", "_").replace(" ", "_")
        name = basename or f"{safe_name}_prediction_report"
        result = self.export_qsar_latex_report(
            payload=payload,
            output_dir=output_dir,
            basename=name,
            agent=agent,
        )
        result["report_payload"] = payload
        return result
