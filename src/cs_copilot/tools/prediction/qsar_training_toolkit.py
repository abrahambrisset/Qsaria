#!/usr/bin/env python
# coding: utf-8
"""Agent-facing QSAR training facade."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import (
    resolve_smiles_column_name,
    standardize_smiles_column,
)
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit

from .chemprop_toolkit import ChempropToolkit
from .lightgbm_toolkit import LightGBMToolkit
from .qsar_training_policy import describe_compute_environment
from .session_state import (
    bundle_artifacts,
    discover_curation_artifacts_near_dataset,
    latest_curation_artifacts,
)
from .tabicl_toolkit import TabICLToolkit
from .training_orchestration import normalize_json_list_argument, write_training_summary


TABULAR_REPRESENTATION_SPECS: Dict[str, Dict[str, Any]] = {
    "morgan_only": {
        "use_morgan": True,
        "use_rdkit": False,
        "descriptor_set": None,
    },
    "rdkit_basic_only": {
        "use_morgan": False,
        "use_rdkit": True,
        "descriptor_set": "basic",
    },
    "morgan_rdkit_basic": {
        "use_morgan": True,
        "use_rdkit": True,
        "descriptor_set": "basic",
    },
    "morgan_rdkit_all": {
        "use_morgan": True,
        "use_rdkit": True,
        "descriptor_set": "all",
    },
}


def _default_representation_for_profile(extra_args: Optional[Dict[str, Any]]) -> str:
    profile = (extra_args or {}).get("training_profile")
    return "morgan_rdkit_all" if profile == "heavy_validation" else "morgan_rdkit_basic"


def _feature_columns_from_csv(path: str, target_columns: List[str]) -> List[str]:
    with S3.open(path, "r") as fh:
        columns = list(pd.read_csv(fh, nrows=0).columns)
    excluded = {"smiles", *target_columns}
    return [column for column in columns if column not in excluded]


class QSARTrainingToolkit(Toolkit):
    """Single public training facade for QSAR agents."""

    def __init__(
        self,
        *,
        chemprop_toolkit: Optional[ChempropToolkit] = None,
        lightgbm_toolkit: Optional[LightGBMToolkit] = None,
        tabicl_toolkit: Optional[TabICLToolkit] = None,
        molecular_feature_toolkit: Optional[MolecularFeatureToolkit] = None,
    ):
        super().__init__("qsar_training")
        self.chemprop_toolkit = chemprop_toolkit or ChempropToolkit(register_tools=False)
        self.lightgbm_toolkit = lightgbm_toolkit or LightGBMToolkit(register_tools=False)
        self.tabicl_toolkit = tabicl_toolkit or TabICLToolkit(register_tools=False)
        self.molecular_feature_toolkit = molecular_feature_toolkit or MolecularFeatureToolkit()

        self.register(self.describe_qsar_training_environment)
        self.register(self.prepare_training_dataset)
        self.register(self.train_qsar_model)
        self.register(self.train_chemprop_model)
        self.register(self.train_lightgbm_model)
        self.register(self.train_tabicl_model)

    def describe_qsar_training_environment(self) -> Dict[str, Any]:
        """Describe compute and backend training availability."""
        return {
            "compute_environment": describe_compute_environment(),
            "backends": {
                "chemprop": self.chemprop_toolkit.backend.describe_environment(),
                "lightgbm": self.lightgbm_toolkit.backend.describe_environment(),
                "tabicl": self.tabicl_toolkit.backend.describe_environment(),
            },
            "toolkit": "QSARTrainingToolkit",
        }

    def backend_mapping(self) -> Dict[str, Any]:
        """Return the backend instances used by the training facade."""
        return {
            self.chemprop_toolkit.backend.backend_name: self.chemprop_toolkit.backend,
            self.lightgbm_toolkit.backend.backend_name: self.lightgbm_toolkit.backend,
            self.tabicl_toolkit.backend.backend_name: self.tabicl_toolkit.backend,
        }

    def prepare_training_dataset(
        self,
        input_csv: str,
        smiles_column: str,
        target_columns: List[str] | str,
        output_csv: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize a QSAR training CSV into canonical `smiles` + target columns."""
        with S3.open(input_csv, "r") as fh:
            df = pd.read_csv(fh)

        normalized_target_columns = normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []

        resolved_smiles_column = resolve_smiles_column_name(df, smiles_column)
        df = standardize_smiles_column(df, resolved_smiles_column)
        if resolved_smiles_column != "smiles":
            df["smiles"] = df[resolved_smiles_column]
            df = df.drop(columns=[resolved_smiles_column])
        missing_targets = [column for column in normalized_target_columns if column not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        standardized = df[["smiles", *normalized_target_columns]].copy()
        destination = output_csv or "training/qsar_training_dataset.csv"
        with S3.open(destination, "w") as fh:
            standardized.to_csv(fh, index=False)

        return {
            "output_csv": destination,
            "rows": int(len(standardized)),
            "columns": list(standardized.columns),
            "smiles_column": "smiles",
            "source_smiles_column": resolved_smiles_column,
        }

    def _prepare_tabular_training_dataset(
        self,
        *,
        train_csv: str,
        output_dir: str,
        smiles_column: str,
        target_columns: List[str],
        representation_name: str,
    ) -> Dict[str, Any]:
        spec = TABULAR_REPRESENTATION_SPECS.get(representation_name)
        if spec is None:
            raise ValueError(
                f"Unsupported representation_name={representation_name!r}. "
                f"Expected one of {sorted(TABULAR_REPRESENTATION_SPECS)}."
            )

        total_started_at = time.monotonic()
        duration_steps: List[Dict[str, Any]] = []
        output_path = Path(output_dir).expanduser().resolve()
        features_dir = output_path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        base_csv_for_features = train_csv
        feature_smiles_column = smiles_column
        if smiles_column != "smiles":
            step_started_at = time.monotonic()
            with S3.open(train_csv, "r") as fh:
                source_df = pd.read_csv(fh)
            resolved_smiles_column = resolve_smiles_column_name(source_df, smiles_column)
            missing_targets = [column for column in target_columns if column not in source_df.columns]
            if missing_targets:
                raise ValueError(f"Missing target columns: {missing_targets}")
            normalized_df = source_df.copy()
            if resolved_smiles_column != "smiles":
                normalized_df["smiles"] = normalized_df[resolved_smiles_column]
                normalized_df = normalized_df.drop(columns=[resolved_smiles_column])
            normalized_csv = features_dir / f"{Path(train_csv).stem}_canonical_smiles.csv"
            with S3.open(str(normalized_csv), "w") as fh:
                normalized_df[["smiles", *target_columns]].to_csv(fh, index=False)
            base_csv_for_features = str(normalized_csv)
            feature_smiles_column = "smiles"
            duration_steps.append(
                {
                    "step": "canonical_smiles_dataset",
                    "duration_seconds": round(time.monotonic() - step_started_at, 3),
                    "output_csv": base_csv_for_features,
                }
            )

        feature_csvs: List[str] = []
        if spec["use_morgan"]:
            step_started_at = time.monotonic()
            morgan = self.molecular_feature_toolkit.smiles_to_morgan_fingerprints(
                input_csv=base_csv_for_features,
                smiles_column=feature_smiles_column,
                output_csv=str(features_dir / "morgan.csv"),
                radius=2,
                n_bits=2048,
                include_input_columns=True,
                input_columns_to_keep=[feature_smiles_column, *target_columns],
            )
            feature_csvs.append(morgan["output_csv"])
            duration_steps.append(
                {
                    "step": "morgan_fingerprints",
                    "duration_seconds": float(
                        morgan.get("duration_seconds")
                        or round(time.monotonic() - step_started_at, 3)
                    ),
                    "output_csv": morgan["output_csv"],
                    "num_features": morgan.get("num_features"),
                    "radius": morgan.get("radius"),
                    "n_bits": morgan.get("n_bits"),
                }
            )

        if spec["use_rdkit"]:
            descriptor_set = str(spec["descriptor_set"])
            step_started_at = time.monotonic()
            rdkit = self.molecular_feature_toolkit.smiles_to_rdkit_descriptors(
                input_csv=base_csv_for_features,
                smiles_column=feature_smiles_column,
                output_csv=str(features_dir / f"rdkit_{descriptor_set}.csv"),
                descriptor_set=descriptor_set,
                include_input_columns=True,
                input_columns_to_keep=[feature_smiles_column, *target_columns],
            )
            feature_csvs.append(rdkit["output_csv"])
            duration_steps.append(
                {
                    "step": "rdkit_descriptors",
                    "duration_seconds": float(
                        rdkit.get("duration_seconds")
                        or round(time.monotonic() - step_started_at, 3)
                    ),
                    "output_csv": rdkit["output_csv"],
                    "descriptor_set": rdkit.get("descriptor_set"),
                    "num_descriptors": rdkit.get("num_descriptors"),
                }
            )

        step_started_at = time.monotonic()
        tabular = self.molecular_feature_toolkit.build_tabular_qsar_dataset(
            base_csv=base_csv_for_features,
            output_csv=str(output_path / f"{Path(train_csv).stem}_{representation_name}.csv"),
            feature_csvs=feature_csvs,
            join_on=["smiles", *target_columns],
            base_columns_to_keep=["smiles", *target_columns],
            drop_duplicate_feature_columns=True,
            canonicalize_smiles_join=False,
        )
        duration_steps.append(
            {
                "step": "tabular_dataset_assembly",
                "duration_seconds": float(
                    tabular.get("duration_seconds")
                    or round(time.monotonic() - step_started_at, 3)
                ),
                "output_csv": tabular["output_csv"],
                "num_added_feature_columns": tabular.get("num_added_feature_columns"),
                "final_column_count": tabular.get("final_column_count"),
                "canonicalize_smiles_join": tabular.get("canonicalize_smiles_join"),
            }
        )
        feature_columns = _feature_columns_from_csv(tabular["output_csv"], target_columns)
        feature_preparation = {
            "mode": "generated_tabular_features",
            "representation_name": representation_name,
            "input_csv": train_csv,
            "base_csv_for_features": base_csv_for_features,
            "prepared_train_csv": tabular["output_csv"],
            "feature_csvs": feature_csvs,
            "feature_count": len(feature_columns),
            "durations": {
                "total_duration_seconds": round(time.monotonic() - total_started_at, 3),
                "steps": duration_steps,
            },
        }
        return {
            "train_csv": tabular["output_csv"],
            "representation_name": representation_name,
            "feature_csvs": feature_csvs,
            "feature_columns": feature_columns,
            "tabular_dataset": tabular,
            "feature_preparation": feature_preparation,
            "feature_preparation_durations": feature_preparation["durations"],
        }

    def _recommended_registry_payload(
        self,
        *,
        backend_name: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "backend_name": backend_name,
            "model_path": result.get("best_model_path") or result.get("model_path"),
            "task_type": task_type,
            "smiles_columns": [smiles_column],
            "target_columns": list(target_columns),
            "known_metrics": result.get("metrics") or {},
            "training_data_summary": {
                "validation_protocol": result.get("validation_protocol"),
                "training_profile": result.get("training_profile"),
                "seed_policy": result.get("seed_policy"),
                "representation_name": result.get("representation_name"),
                "feature_preparation": result.get("feature_preparation") or {},
            },
            "inference_profile": {
                "feature_columns": list(result.get("feature_columns") or []),
                "representation_name": result.get("representation_name"),
            },
            "applicability_domain": result.get("applicability_domain") or {},
        }

    def _refresh_enriched_training_artifacts(
        self,
        *,
        result: Dict[str, Any],
        output_dir: str,
    ) -> None:
        """Rewrite summary and bundle after facade-level enrichment."""
        summary_path = result.get("summary_path") or result.get("canonical_summary_path")
        if summary_path:
            write_training_summary(Path(str(summary_path)), result)

        bundle_path = result.get("bundle_file_ref") or result.get("training_bundle")
        if bundle_path:
            resolved_output_dir = result.get("output_dir") or output_dir
            bundle_inputs = [Path(str(resolved_output_dir)).expanduser().resolve()]
            for key in ("candidate_train_csv", "train_csv"):
                if result.get(key):
                    bundle_inputs.append(Path(str(result[key])))
            curation = result.get("curation") or {}
            if curation.get("curated_dataset_path"):
                bundle_inputs.append(Path(str(curation["curated_dataset_path"])))
            for artifact_path in (curation.get("artifacts") or {}).values():
                if artifact_path:
                    bundle_inputs.append(Path(str(artifact_path)))
            bundle = bundle_artifacts(Path(str(bundle_path)), bundle_inputs)
            result["bundle_file_ref"] = str(bundle)
            result["training_bundle"] = str(bundle)
            result["bundle_download_tag"] = f"<file>{bundle}</file>"

    def train_qsar_model(
        self,
        train_csv: str,
        backend_name: str,
        task_type: str,
        output_dir: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        representation_name: Optional[str] = None,
        feature_columns: Optional[List[str] | str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a QSAR model with the requested backend."""
        normalized_backend = backend_name.strip().lower()
        normalized_target_columns = normalize_json_list_argument(
            target_columns,
            argument_name="target_columns",
        ) or []
        normalized_feature_columns = normalize_json_list_argument(
            feature_columns,
            argument_name="feature_columns",
        )
        normalized_categorical_feature_columns = normalize_json_list_argument(
            categorical_feature_columns,
            argument_name="categorical_feature_columns",
        )
        requested_extra_args = dict(extra_args or {})
        requested_extra_args.setdefault("validation_protocol", validation_protocol)

        if normalized_backend == "chemprop":
            result = self.chemprop_toolkit.train_model(
                train_csv=train_csv,
                task_type=task_type,
                output_dir=output_dir,
                smiles_columns=[smiles_column],
                target_columns=list(normalized_target_columns),
                activity_cliff_index=activity_cliff_index,
                activity_cliff_feedback=activity_cliff_feedback,
                activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                extra_args=requested_extra_args,
                agent=agent,
            )
            result["backend_name"] = "chemprop"
            result.setdefault("representation_name", "molecular_graph")
            result["candidate_train_csv"] = train_csv
        elif normalized_backend in {"lightgbm", "tabicl"}:
            working_train_csv = train_csv
            resolved_representation = representation_name
            if not normalized_feature_columns:
                resolved_representation = resolved_representation or _default_representation_for_profile(
                    requested_extra_args
                )
                prepared = self._prepare_tabular_training_dataset(
                    train_csv=train_csv,
                    output_dir=output_dir,
                    smiles_column=smiles_column,
                    target_columns=list(normalized_target_columns),
                    representation_name=resolved_representation,
                )
                working_train_csv = prepared["train_csv"]
                normalized_feature_columns = prepared["feature_columns"]
                feature_preparation = prepared["feature_preparation"]
            else:
                resolved_representation = resolved_representation or "precomputed_tabular"
                feature_preparation = {
                    "mode": "precomputed_tabular_features",
                    "representation_name": resolved_representation,
                    "input_csv": train_csv,
                    "prepared_train_csv": working_train_csv,
                    "feature_count": len(normalized_feature_columns or []),
                    "durations": {"total_duration_seconds": 0.0, "steps": []},
                }

            if normalized_backend == "lightgbm":
                result = self.lightgbm_toolkit.train_lightgbm_model(
                    train_csv=working_train_csv,
                    task_type=task_type,
                    output_dir=output_dir,
                    target_columns=list(normalized_target_columns),
                    feature_columns=normalized_feature_columns,
                    categorical_feature_columns=normalized_categorical_feature_columns,
                    validation_protocol=validation_protocol,
                    activity_cliff_index=activity_cliff_index,
                    activity_cliff_feedback=activity_cliff_feedback,
                    activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                    activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                    activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                    activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                    extra_args=requested_extra_args,
                    agent=agent,
                )
            else:
                result = self.tabicl_toolkit.train_tabicl_model(
                    train_csv=working_train_csv,
                    task_type=task_type,
                    output_dir=output_dir,
                    target_columns=list(normalized_target_columns),
                    feature_columns=normalized_feature_columns,
                    validation_protocol=validation_protocol,
                    activity_cliff_index=activity_cliff_index,
                    activity_cliff_feedback=activity_cliff_feedback,
                    activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                    activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                    activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                    activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                    extra_args=requested_extra_args,
                    agent=agent,
                )
            result["backend_name"] = normalized_backend
            result["representation_name"] = resolved_representation
            result["candidate_train_csv"] = working_train_csv
            result["feature_columns"] = list(normalized_feature_columns or [])
            result["feature_preparation"] = feature_preparation
            result["feature_preparation_durations"] = feature_preparation["durations"]
        else:
            raise ValueError(
                "Unsupported backend_name. Expected one of ['chemprop', 'lightgbm', 'tabicl']."
            )

        if not ((result.get("curation") or {}).get("artifacts")):
            curation_artifacts = latest_curation_artifacts(agent) if agent is not None else {}
            if not (curation_artifacts.get("artifacts") if curation_artifacts else None):
                curation_artifacts = discover_curation_artifacts_near_dataset(train_csv)
            if curation_artifacts:
                result["curation"] = curation_artifacts

        if normalized_backend in {"lightgbm", "tabicl"}:
            self._refresh_enriched_training_artifacts(
                result=result,
                output_dir=output_dir,
            )

        result["recommended_registry_payload"] = self._recommended_registry_payload(
            backend_name=normalized_backend,
            task_type=task_type,
            smiles_column=smiles_column,
            target_columns=list(normalized_target_columns),
            result=result,
        )
        return result

    def train_chemprop_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train Chemprop through the unified QSAR facade."""
        return self.train_qsar_model(
            train_csv=train_csv,
            backend_name="chemprop",
            task_type=task_type,
            output_dir=output_dir,
            smiles_column=smiles_column,
            target_columns=target_columns,
            validation_protocol=validation_protocol,
            activity_cliff_index=activity_cliff_index,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
            activity_cliff_flag_threshold=activity_cliff_flag_threshold,
            extra_args=extra_args,
            agent=agent,
        )

    def train_lightgbm_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        smiles_column: str = "smiles",
        representation_name: Optional[str] = None,
        feature_columns: Optional[List[str] | str] = None,
        categorical_feature_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train LightGBM through the unified QSAR facade."""
        return self.train_qsar_model(
            train_csv=train_csv,
            backend_name="lightgbm",
            task_type=task_type,
            output_dir=output_dir,
            smiles_column=smiles_column,
            target_columns=target_columns,
            validation_protocol=validation_protocol,
            representation_name=representation_name,
            feature_columns=feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            activity_cliff_index=activity_cliff_index,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
            activity_cliff_flag_threshold=activity_cliff_flag_threshold,
            extra_args=extra_args,
            agent=agent,
        )

    def train_tabicl_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        target_columns: List[str] | str,
        smiles_column: str = "smiles",
        representation_name: Optional[str] = None,
        feature_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        activity_cliff_index: str = "sali",
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = 0.70,
        activity_cliff_top_k_neighbors: int = 10,
        activity_cliff_flag_threshold: float = 0.35,
        extra_args: Optional[Dict[str, Any]] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train TabICL through the unified QSAR facade."""
        return self.train_qsar_model(
            train_csv=train_csv,
            backend_name="tabicl",
            task_type=task_type,
            output_dir=output_dir,
            smiles_column=smiles_column,
            target_columns=target_columns,
            validation_protocol=validation_protocol,
            representation_name=representation_name,
            feature_columns=feature_columns,
            activity_cliff_index=activity_cliff_index,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
            activity_cliff_flag_threshold=activity_cliff_flag_threshold,
            extra_args=extra_args,
            agent=agent,
        )
