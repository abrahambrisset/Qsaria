#!/usr/bin/env python
# coding: utf-8
"""Backend-neutral prediction inference facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import (
    resolve_smiles_column_name,
    standardize_smiles_column,
)

from .model_registry_toolkit import ModelRegistryToolkit
from .session_state import get_prediction_state


def prediction_output_path(model_id: str, preds_path: Optional[str] = None) -> Path:
    if preds_path:
        return Path(preds_path).expanduser()
    return (Path(".files") / "prediction_outputs" / f"{model_id}_predictions.csv").resolve()


class PredictionInferenceToolkit(Toolkit):
    """Backend-neutral prediction execution and prediction-history export."""

    def __init__(
        self,
        *,
        backends: Mapping[str, Any],
        registry_toolkit: ModelRegistryToolkit,
        register_tools: bool = True,
    ):
        super().__init__("prediction_inference")
        self.backends = dict(backends)
        self.registry_toolkit = registry_toolkit

        if register_tools:
            self.register(self.predict_from_csv)
            self.register(self.predict_from_smiles)
            self.register(self.export_prediction_summary)

    def get_backend(self, backend_name: str):
        backend = self.backends.get(backend_name)
        if backend is None:
            raise ValueError(f"Unsupported prediction backend: {backend_name}")
        return backend

    def predict_from_csv(
        self,
        model_id: str,
        input_csv: str,
        smiles_column: str = "smiles",
        preds_path: Optional[str] = None,
        return_uncertainty: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run prediction from a CSV file and persist the result path in session state."""
        if agent is None:
            raise ValueError("Agent is required for prediction")

        record = self.registry_toolkit.resolve_record(model_id, agent)
        output_path = prediction_output_path(model_id, preds_path)
        backend = self.get_backend(record.backend_name)
        local_input = Path(input_csv).expanduser()
        if local_input.exists():
            source_df = pd.read_csv(local_input)
        else:
            with S3.open(input_csv, "r") as fh:
                source_df = pd.read_csv(fh)

        df = source_df.copy()
        smiles_found = resolve_smiles_column_name(df, smiles_column)
        df = standardize_smiles_column(df, smiles_found)
        if smiles_found != "smiles":
            df["smiles"] = df[smiles_found]
            df = df.drop(columns=[smiles_found])

        local_input = (Path(".files") / "prediction_inputs" / f"{model_id}_input.csv").resolve()
        local_input.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(local_input, index=False)

        result = backend.predict_from_csv(
            input_csv=str(local_input),
            model_record=record,
            preds_path=str(output_path),
            return_uncertainty=return_uncertainty,
        )

        predictions_only_df = pd.read_csv(output_path)
        prediction_columns = [
            column for column in predictions_only_df.columns if column not in source_df.columns
        ]
        if prediction_columns:
            enriched_output_df = pd.concat(
                [
                    source_df.reset_index(drop=True),
                    predictions_only_df[prediction_columns].reset_index(drop=True),
                ],
                axis=1,
            )
            enriched_output_df.to_csv(output_path, index=False)

        preview_df = pd.read_csv(output_path)
        preview_columns = list(preview_df.columns)
        preview = preview_df.head(5).to_dict(orient="records")
        num_rows = int(len(preview_df))

        prediction_state = get_prediction_state(agent)
        prediction_state["last_prediction"] = {
            "model_id": model_id,
            "input_csv": str(local_input),
            "preds_path": str(output_path),
            "return_uncertainty": return_uncertainty,
            "applicability_domain": result.get("applicability_domain") or {},
            "applicability_domain_columns": result.get("applicability_domain_columns") or [],
            "ensemble_inference_summary": result.get("ensemble_inference_summary") or {},
        }
        history_entry = {
            "model_id": model_id,
            "backend_name": record.backend_name,
            "task_type": record.task.task_type,
            "input_csv": str(local_input),
            "preds_path": str(output_path),
            "download_file_ref": str(output_path),
            "preview_columns": preview_columns,
            "preview": preview,
            "num_rows": num_rows,
            "applicability_domain": result.get("applicability_domain") or {},
            "applicability_domain_columns": result.get("applicability_domain_columns") or [],
            "ensemble_inference_summary": result.get("ensemble_inference_summary") or {},
        }
        prediction_state["prediction_history"].append(history_entry)
        result["preds_path"] = str(output_path)
        result["download_file_ref"] = str(output_path)
        result["download_file_tag"] = f"<file>{output_path}</file>"
        result["preview_columns"] = preview_columns
        result["preview"] = preview
        result["num_rows"] = num_rows
        return result

    def predict_from_smiles(
        self,
        model_id: str,
        smiles: List[str],
        preds_path: Optional[str] = None,
        return_uncertainty: bool = False,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Run prediction from an in-memory list of SMILES by materializing a temporary CSV."""
        if agent is None:
            raise ValueError("Agent is required for prediction")

        if not smiles:
            raise ValueError("At least one SMILES string is required")

        input_path = Path(".files") / "prediction_inputs" / f"{model_id}_smiles_input.csv"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"smiles": smiles})
        df = standardize_smiles_column(df, "smiles")
        df.to_csv(input_path, index=False)

        result = self.predict_from_csv(
            model_id=model_id,
            input_csv=str(input_path),
            smiles_column="smiles",
            preds_path=preds_path,
            return_uncertainty=return_uncertainty,
            agent=agent,
        )
        result["num_smiles"] = len(smiles)
        return result

    def export_prediction_summary(
        self,
        summary_csv: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Export a consolidated CSV summary from prediction history."""
        if agent is None:
            raise ValueError("Agent is required to export a prediction summary")

        prediction_state = get_prediction_state(agent)
        history = prediction_state.get("prediction_history") or []
        if not history:
            raise ValueError("No prediction history is available for summary export")

        summary_path = (
            Path(summary_csv).expanduser()
            if summary_csv
            else (Path(".files") / "prediction_outputs" / "prediction_summary.csv").resolve()
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        frames: List[pd.DataFrame] = []
        for item in history:
            preds_path = item.get("preds_path")
            model_id = item.get("model_id")
            task_type = item.get("task_type")
            if not preds_path or not Path(preds_path).exists():
                continue

            df = pd.read_csv(preds_path).copy()
            if df.empty:
                continue

            value_columns = [col for col in df.columns if col.lower() != "smiles"]
            if not value_columns:
                continue

            melted = df.melt(
                id_vars=["smiles"] if "smiles" in df.columns else None,
                value_vars=value_columns,
                var_name="prediction_column",
                value_name="predicted_value",
            )
            melted.insert(0, "model_id", model_id)
            melted.insert(1, "task_type", task_type)
            frames.append(melted)

        if not frames:
            raise ValueError("No readable prediction files were found for summary export")

        summary_df = pd.concat(frames, ignore_index=True)
        summary_df.to_csv(summary_path, index=False)

        prediction_outputs = agent.session_state.setdefault("prediction_outputs", {})
        prediction_outputs["latest_summary"] = str(summary_path)

        preview_columns = list(summary_df.columns)
        preview = summary_df.head(10).to_dict(orient="records")

        return {
            "summary_csv": str(summary_path),
            "download_file_ref": str(summary_path),
            "num_rows": int(len(summary_df)),
            "num_files": len(frames),
            "preview_columns": preview_columns,
            "preview": preview,
        }
