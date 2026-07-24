#!/usr/bin/env python
# coding: utf-8
"""Post-hoc consensus ensemble backend for catalogued QSAR models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from .backend import (
    BackendNotAvailableError,
    InvalidPredictionInputError,
    PredictionBackend,
    PredictionExecutionError,
    PredictionModelRecord,
    PredictionTaskSpec,
)
from .backend_capabilities import (
    BACKEND_CAPABILITIES,
    BackendCapabilities,
    backend_requires_feature_preparation,
    enrich_backend_environment,
)
from .chemprop_backend import ChempropBackend
from .lightgbm_backend import LightGBMBackend
from .qsar_training_policy import safe_slug
from .tabicl_backend import TabICLBackend
from .tabular_feature_preparation import TabularFeaturePreparationService
from .training_orchestration import is_classification_task, is_multiclass_task, json_safe_label


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser())


def _prediction_column(frame: pd.DataFrame, target_columns: list[str]) -> str:
    if "prediction" in frame.columns:
        return "prediction"
    for column in target_columns:
        if column in frame.columns:
            return column
    numeric = [column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
    if numeric:
        return numeric[-1]
    raise InvalidPredictionInputError(
        "Component prediction output does not contain a numeric prediction column."
    )


def _component_representation(component: Dict[str, Any]) -> str:
    contract = component.get("tabular_representation_contract") or {}
    if contract.get("representation_name"):
        return str(contract["representation_name"])
    if component.get("backend_name") == "chemprop":
        return "molecular_graph"
    return "unknown"


def _finite_float(value: Any) -> Optional[float]:
    if pd.isna(value):
        return None
    return float(value)


def _numeric_summary(series: pd.Series) -> Dict[str, Optional[float]]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "max": None, "mean": None, "median": None, "q90": None, "q95": None}
    return {
        "min": _finite_float(values.min()),
        "max": _finite_float(values.max()),
        "mean": _finite_float(values.mean()),
        "median": _finite_float(values.median()),
        "q90": _finite_float(values.quantile(0.90)),
        "q95": _finite_float(values.quantile(0.95)),
    }


class EnsembleBackend(PredictionBackend):
    """Aggregate predictions from already-persisted component models."""

    backend_name = "ensemble"

    def __init__(
        self,
        backends: Optional[Dict[str, PredictionBackend]] = None,
        *,
        backend_capabilities: Optional[Mapping[str, BackendCapabilities]] = None,
        tabular_feature_service: Optional[TabularFeaturePreparationService] = None,
    ) -> None:
        self.backends = backends or {
            "chemprop": ChempropBackend(),
            "lightgbm": LightGBMBackend(),
            "tabicl": TabICLBackend(),
        }
        self.backend_capabilities: Dict[str, BackendCapabilities] = {
            **BACKEND_CAPABILITIES,
            **dict(backend_capabilities or {}),
        }
        self.tabular_feature_service = tabular_feature_service or TabularFeaturePreparationService()

    def is_available(self) -> bool:
        return True

    def describe_environment(self) -> Dict[str, Any]:
        return enrich_backend_environment(
            self.backend_name,
            {
                "backend_name": self.backend_name,
                "available": True,
                "component_backends": {
                    name: backend.describe_environment() for name, backend in self.backends.items()
                },
            },
            registry=self.backend_capabilities,
        )

    def _requires_feature_preparation(self, backend_name: str) -> bool:
        try:
            return backend_requires_feature_preparation(
                backend_name,
                registry=self.backend_capabilities,
            )
        except KeyError as exc:
            raise InvalidPredictionInputError(
                f"Backend `{backend_name}` is configured but has no registered capabilities."
            ) from exc

    def validate_model_path(self, model_path: str) -> Path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise InvalidPredictionInputError(f"Ensemble model path does not exist: {model_path}")
        if path.name != "ensemble.json":
            raise InvalidPredictionInputError(
                "Ensemble model artifact must be named ensemble.json."
            )
        payload = self._load_payload(path)
        if int(payload.get("schema_version", 0)) != 1:
            raise InvalidPredictionInputError("Unsupported ensemble schema_version.")
        if payload.get("ensemble_kind") not in {
            "catalog_consensus_regression",
            "catalog_consensus_classification",
        }:
            raise InvalidPredictionInputError("Unsupported ensemble_kind for this V1 backend.")
        if (
            payload.get("ensemble_kind") == "catalog_consensus_regression"
            and payload.get("aggregation_strategy") != "median"
        ):
            raise InvalidPredictionInputError(
                "Only median aggregation is supported for regression ensemble V1."
            )
        if (
            payload.get("ensemble_kind") == "catalog_consensus_classification"
            and payload.get("aggregation_strategy") != "majority_vote"
        ):
            raise InvalidPredictionInputError(
                "Only majority_vote aggregation is supported for classification ensemble V1."
            )
        components = payload.get("components")
        if not isinstance(components, list) or not components:
            raise InvalidPredictionInputError("Ensemble must contain at least one component.")
        return path.resolve()

    def _load_payload(self, path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            raise InvalidPredictionInputError(f"Could not read ensemble.json: {exc}") from exc
        if not isinstance(payload, dict):
            raise InvalidPredictionInputError("ensemble.json must contain a JSON object.")
        return payload

    def _component_record(
        self, component: Dict[str, Any], ensemble_record: PredictionModelRecord
    ) -> PredictionModelRecord:
        task_payload = component.get("task") or {}
        return PredictionModelRecord(
            model_id=str(component.get("model_id")),
            backend_name=str(component.get("backend_name")),
            model_path=str(component.get("model_path")),
            metadata_path=component.get("metadata_path"),
            display_name=component.get("display_name"),
            description=component.get("description"),
            version=component.get("version"),
            status=component.get("status", "workflow_demo"),
            known_metrics=dict(component.get("known_metrics") or {}),
            training_data_summary=dict(component.get("training_data_summary") or {}),
            inference_profile=dict(component.get("inference_profile") or {}),
            selection_hints=dict(component.get("selection_hints") or {}),
            applicability_domain=dict(component.get("applicability_domain") or {}),
            tabular_representation_contract=dict(
                component.get("tabular_representation_contract") or {}
            ),
            task=PredictionTaskSpec(
                task_type=task_payload.get("task_type") or ensemble_record.task.task_type,
                smiles_columns=task_payload.get("smiles_columns")
                or ensemble_record.task.smiles_columns,
                target_columns=task_payload.get("target_columns")
                or ensemble_record.task.target_columns,
            ),
        )

    def _prepare_component_input_csv(
        self,
        *,
        backend_name: str,
        record: PredictionModelRecord,
        source_csv: str,
        work_dir: Path,
        slug: str,
    ) -> str:
        if not self._requires_feature_preparation(backend_name):
            return source_csv
        prepared = self.tabular_feature_service.prepare_for_model(
            input_csv=source_csv,
            output_dir=str(work_dir / "_feature_inputs" / slug),
            record=record,
            purpose="ensemble",
            smiles_column="smiles",
        )
        return prepared.prepared_csv

    def _top_disagreement_rows(
        self,
        *,
        source_df: pd.DataFrame,
        aggregate: pd.DataFrame,
        limit: int = 5,
    ) -> list[Dict[str, Any]]:
        if "ensemble_prediction_std" not in aggregate.columns:
            return []
        rows: list[Dict[str, Any]] = []
        sorted_indices = (
            aggregate["ensemble_prediction_std"].sort_values(ascending=False).head(limit).index
        )
        id_columns = [
            column
            for column in ("Molecule Name", "molecule_name", "OCNT_ID", "id", "smiles")
            if column in source_df.columns
        ]
        for index in sorted_indices:
            item: Dict[str, Any] = {"row_index": int(index)}
            for column in id_columns:
                item[column] = source_df.iloc[int(index)][column]
            item["ensemble_prediction_median"] = _finite_float(
                aggregate.iloc[int(index)]["ensemble_prediction_median"]
            )
            item["ensemble_prediction_std"] = _finite_float(
                aggregate.iloc[int(index)]["ensemble_prediction_std"]
            )
            item["ensemble_prediction_min"] = _finite_float(
                aggregate.iloc[int(index)]["ensemble_prediction_min"]
            )
            item["ensemble_prediction_max"] = _finite_float(
                aggregate.iloc[int(index)]["ensemble_prediction_max"]
            )
            rows.append(item)
        return rows

    def predict_from_csv(
        self,
        input_csv: str,
        model_record: PredictionModelRecord,
        preds_path: str,
        *,
        return_uncertainty: bool = False,
    ) -> Dict[str, Any]:
        ensemble_path = self.validate_model_path(model_record.model_path)
        payload = self._load_payload(ensemble_path)
        ensemble_kind = str(payload.get("ensemble_kind") or "")
        task_is_classification = (
            ensemble_kind == "catalog_consensus_classification"
            or is_classification_task(model_record.task.task_type)
        )
        if task_is_classification and is_multiclass_task(model_record.task.task_type):
            raise InvalidPredictionInputError(
                "Ensemble multiclass classification is not enabled in this QSARIA version."
            )
        components = payload.get("components") or []
        output_path = Path(preds_path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        work_dir = output_path.parent / f"{output_path.stem}_components"
        work_dir.mkdir(parents=True, exist_ok=True)

        input_path = Path(input_csv).expanduser()
        component_input_csv = str(input_path.resolve()) if input_path.exists() else input_csv
        source_df = _read_csv(component_input_csv)
        component_columns: Dict[str, pd.Series] = {}
        component_probability_columns: Dict[str, pd.Series] = {}
        component_paths: Dict[str, str] = {}
        component_input_paths: Dict[str, str] = {}
        component_summaries: list[Dict[str, Any]] = []
        failures: list[str] = []

        for component in components:
            backend_name = str(component.get("backend_name") or "")
            backend = self.backends.get(backend_name)
            if backend is None:
                failures.append(
                    f"{component.get('model_id')}: unsupported backend `{backend_name}`"
                )
                continue
            record = self._component_record(component, model_record)
            slug = (
                safe_slug(
                    str(
                        component.get("component_slug") or component.get("model_id") or backend_name
                    )
                )
                or backend_name
            )
            component_path = work_dir / f"{slug}_predictions.csv"
            try:
                prepared_input_csv = self._prepare_component_input_csv(
                    backend_name=backend_name,
                    record=record,
                    source_csv=component_input_csv,
                    work_dir=work_dir,
                    slug=slug,
                )
                backend.predict_from_csv(
                    input_csv=prepared_input_csv,
                    model_record=record,
                    preds_path=str(component_path),
                    return_uncertainty=False,
                )
                frame = _read_csv(component_path)
                if task_is_classification:
                    column = (
                        "prediction"
                        if "prediction" in frame.columns
                        else next(
                            (
                                target
                                for target in record.task.target_columns
                                if target in frame.columns
                            ),
                            None,
                        )
                    )
                    if column is None:
                        raise InvalidPredictionInputError(
                            f"Component `{record.model_id}` did not return a classification prediction column."
                        )
                    values = frame[column].map(json_safe_label)
                    if "positive_probability" in frame.columns:
                        component_probability_columns[f"positive_probability_{slug}"] = (
                            pd.to_numeric(
                                frame["positive_probability"],
                                errors="coerce",
                            ).reset_index(drop=True)
                        )
                else:
                    column = _prediction_column(frame, record.task.target_columns)
                    values = pd.to_numeric(frame[column], errors="coerce")
                if len(values) != len(source_df):
                    raise InvalidPredictionInputError(
                        f"Component `{record.model_id}` returned {len(values)} predictions for {len(source_df)} inputs."
                    )
                component_columns[f"prediction_{slug}"] = values.reset_index(drop=True)
                component_paths[slug] = str(component_path)
                component_input_paths[slug] = prepared_input_csv
                component_summaries.append(
                    {
                        "model_id": record.model_id,
                        "component_slug": slug,
                        "backend_name": backend_name,
                        "representation_name": _component_representation(component) or "unknown",
                        "prediction_column": f"prediction_{slug}",
                        "predictions_path": str(component_path),
                        "input_csv": prepared_input_csv,
                        "status": "ok",
                    }
                )
            except Exception as exc:
                failures.append(f"{component.get('model_id')}: {exc}")

        if failures:
            raise PredictionExecutionError(
                "Ensemble component inference failed: " + " | ".join(failures)
            )
        if not component_columns:
            raise PredictionExecutionError("No component predictions were produced.")

        component_df = pd.DataFrame(component_columns)
        if task_is_classification:
            modes = component_df.mode(axis=1, dropna=True)
            majority = modes[0] if 0 in modes.columns else pd.Series([None] * len(component_df))
            vote_fraction = component_df.eq(majority, axis=0).sum(axis=1) / float(
                len(component_columns)
            )
            aggregate = pd.DataFrame(
                {
                    "prediction": majority,
                    "ensemble_prediction_class": majority,
                    "ensemble_vote_fraction": vote_fraction,
                    "ensemble_component_count": len(component_columns),
                }
            )
            if component_probability_columns:
                probability_df = pd.DataFrame(component_probability_columns)
                aggregate["positive_probability"] = probability_df.mean(axis=1, skipna=True)
                aggregate["ensemble_positive_probability_std"] = probability_df.std(
                    axis=1, ddof=0
                ).fillna(0.0)
                component_df = pd.concat([component_df, probability_df], axis=1)
        else:
            aggregate = pd.DataFrame(
                {
                    "prediction": component_df.median(axis=1),
                    "ensemble_prediction_median": component_df.median(axis=1),
                    "ensemble_prediction_mean": component_df.mean(axis=1),
                    "ensemble_prediction_std": component_df.std(axis=1, ddof=0),
                    "ensemble_prediction_min": component_df.min(axis=1),
                    "ensemble_prediction_max": component_df.max(axis=1),
                    "ensemble_component_count": len(component_columns),
                }
            )
        result_df = pd.concat([aggregate, component_df], axis=1)
        result_df.to_csv(output_path, index=False)
        ensemble_inference_summary = {
            "report_kind": "ensemble_inference",
            "model_id": model_record.model_id,
            "target_columns": list(model_record.task.target_columns),
            "task_type": model_record.task.task_type,
            "rows_in": int(len(source_df)),
            "rows_predicted": int(len(result_df)),
            "rows_failed": 0,
            "aggregation_strategy": "majority_vote" if task_is_classification else "median",
            "official_prediction_column": (
                "ensemble_prediction_class"
                if task_is_classification
                else "ensemble_prediction_median"
            ),
            "uncertainty_strategy": (
                "vote_fraction" if task_is_classification else "component_disagreement_std"
            ),
            "uncertainty_note": (
                "ensemble_vote_fraction is component agreement, not calibrated predictive uncertainty."
                if task_is_classification
                else "ensemble_prediction_std is inter-component disagreement, not calibrated predictive uncertainty."
            ),
            "component_count": len(component_columns),
            "components": component_summaries,
            "output_columns": list(result_df.columns),
            "prediction_summary": (
                aggregate["ensemble_prediction_class"].value_counts(dropna=False).to_dict()
                if task_is_classification
                else _numeric_summary(aggregate["ensemble_prediction_median"])
            ),
            "disagreement_summary": (
                _numeric_summary(aggregate["ensemble_vote_fraction"])
                if task_is_classification
                else _numeric_summary(aggregate["ensemble_prediction_std"])
            ),
            "top_disagreement_rows": (
                []
                if task_is_classification
                else self._top_disagreement_rows(
                    source_df=source_df,
                    aggregate=aggregate,
                )
            ),
            "applicability_domain_applied": False,
        }
        return {
            "backend_name": self.backend_name,
            "predictions_path": str(output_path),
            "component_prediction_paths": component_paths,
            "component_input_paths": component_input_paths,
            "components": component_summaries,
            "component_count": len(component_columns),
            "aggregation_strategy": "majority_vote" if task_is_classification else "median",
            "uncertainty_strategy": (
                "vote_fraction" if task_is_classification else "component_disagreement_std"
            ),
            "ensemble_inference_summary": ensemble_inference_summary,
            "download_file_ref": str(output_path),
            "download_file_tag": f"<file>{output_path}</file>",
            "return_uncertainty": return_uncertainty,
        }

    def train_model(self, request) -> Dict[str, Any]:
        raise BackendNotAvailableError(
            "EnsembleBackend is post-hoc only and does not train component models."
        )
