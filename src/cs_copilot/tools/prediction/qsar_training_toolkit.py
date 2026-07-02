#!/usr/bin/env python
# coding: utf-8
"""Agent-facing QSAR training facade."""

from __future__ import annotations

import time
import hashlib
import json
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
from .tabular_representations import (
    AUTOMATIC_TABULAR_REPRESENTATION_NAMES,
    default_tabular_representation_for_protocol,
    describe_tabular_representations,
    get_tabular_representation,
)
from .training_orchestration import normalize_json_list_argument, write_training_summary

FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT = 20
QSAR_ROW_ID_COLUMN = "__qsar_row_id"


def _feature_columns_from_csv(path: str, target_columns: List[str]) -> List[str]:
    with S3.open(path, "r") as fh:
        columns = list(pd.read_csv(fh, nrows=0).columns)
    excluded = {"smiles", QSAR_ROW_ID_COLUMN, *target_columns}
    return [column for column in columns if column not in excluded]


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with S3.open(path, "rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with S3.open(str(path), "r") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _storage_path_exists(path: Path | str) -> bool:
    try:
        with S3.open(str(path), "rb") as fh:
            fh.read(1)
        return True
    except FileNotFoundError:
        return False


def _resolve_feature_n_jobs(raw: Optional[Any] = None) -> int:
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1
    return max(1, min(int(describe_compute_environment().get("cpu_count") or 1), 16))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with S3.open(str(path), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _feature_step_name(component_name: str) -> str:
    return {
        "morgan_binary": "morgan_binary_fingerprints",
        "morgan_count": "morgan_count_fingerprints",
        "rdkit_all": "rdkit_all_descriptors",
        "rdkit_basic": "rdkit_basic_descriptors",
    }.get(component_name, component_name)


def _feature_columns_summary(
    feature_columns: Optional[List[str]],
    *,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    columns = list(feature_columns or [])
    omitted_count = max(0, len(columns) - FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT)
    payload: Dict[str, Any] = {
        "feature_columns_count": len(columns),
        "feature_columns_sample": columns[:FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT],
        "feature_columns_omitted_count": omitted_count,
    }
    if source:
        payload["feature_columns_source"] = source
    if omitted_count:
        payload["feature_columns_note"] = (
            "Full feature_columns are stored in the training summary and restored "
            "during catalog persistence; they are omitted from the tool response "
            "to keep agent context bounded."
        )
    return payload


def _feature_columns_summary_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    feature_columns = result.get("feature_columns")
    if isinstance(feature_columns, list):
        return _feature_columns_summary(
            feature_columns,
            source=result.get("summary_path") or result.get("canonical_summary_path"),
        )
    payload = {
        key: result[key]
        for key in (
            "feature_columns_count",
            "feature_columns_sample",
            "feature_columns_omitted_count",
            "feature_columns_source",
            "feature_columns_note",
        )
        if key in result
    }
    if "feature_columns_source" not in payload:
        source = result.get("summary_path") or result.get("canonical_summary_path")
        if source:
            payload["feature_columns_source"] = source
    return payload


def _compact_registry_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return payload
    compacted = dict(payload)
    inference_profile = dict(compacted.get("inference_profile") or {})
    feature_columns = inference_profile.pop("feature_columns", None)
    if isinstance(feature_columns, list):
        inference_profile.update(
            _feature_columns_summary(
                feature_columns,
                source=inference_profile.get("feature_columns_source")
                or (compacted.get("training_data_summary") or {}).get("training_summary_path"),
            )
        )
    compacted["inference_profile"] = inference_profile
    return compacted


def _compact_feature_preparation_durations(feature_preparation: Dict[str, Any]) -> Dict[str, Any]:
    durations = (feature_preparation or {}).get("durations") or {}
    return {
        "total_duration_seconds": durations.get("total_duration_seconds"),
        "steps": [
            {
                "step": step.get("step"),
                "cache_status": step.get("cache_status"),
                "cache_hit": step.get("cache_hit"),
                "duration_seconds": step.get("duration_seconds"),
                "num_features": step.get("num_features"),
                "num_descriptors": step.get("num_descriptors"),
                "num_added_feature_columns": step.get("num_added_feature_columns"),
            }
            for step in durations.get("steps") or []
        ],
    }


def _compact_feature_preparation(feature_preparation: Dict[str, Any]) -> Dict[str, Any]:
    if not feature_preparation:
        return {}
    return {
        key: feature_preparation.get(key)
        for key in (
            "mode",
            "representation_name",
            "representation_display_name",
            "representation_legacy",
            "prepared_train_csv",
            "feature_cache_key",
            "feature_cache_status",
            "feature_n_jobs",
            "cache_hits",
            "cache_misses",
            "feature_count",
        )
        if feature_preparation.get(key) is not None
    } | {"durations": _compact_feature_preparation_durations(feature_preparation)}


def _compact_split_result_for_response(split_result: Dict[str, Any]) -> Dict[str, Any]:
    metrics = split_result.get("metrics") or {}
    return {
        key: split_result.get(key)
        for key in (
            "strategy_label",
            "strategy",
            "strategy_family",
            "backend_split_type",
            "seed",
            "model_path",
            "best_model_path",
            "test_predictions_path",
            "splits_path",
            "chemprop_training_input_csv",
            "chemprop_splits_file",
            "chemprop_input_manifest_path",
            "output_dir",
            "source_train_count",
            "effective_train_count",
            "validation_count",
            "test_count",
            "has_validation_split",
            "split_metadata",
            "duration_seconds",
        )
        if split_result.get(key) is not None
    } | {"metrics": metrics}


def _compact_activity_cliffs(activity_cliffs: Dict[str, Any]) -> Dict[str, Any]:
    if not activity_cliffs:
        return {}
    return {
        key: activity_cliffs.get(key)
        for key in (
            "enabled",
            "mode",
            "index_name",
            "flagged_count",
            "priority_counts",
            "recommended_variant",
            "summary_path",
            "annotated_training_csv",
            "plot_artifacts",
            "reporting_handoff",
        )
        if activity_cliffs.get(key) is not None
    }


def _compact_training_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    keep_keys = (
        "campaign_started",
        "campaign_type",
        "backend_name",
        "task_type",
        "representation_name",
        "validation_protocol",
        "validation_protocol_reason",
        "validation_strategy",
        "validation_strategy_type",
        "validation_aggregation",
        "selection_metric",
        "final_refit",
        "training_profile",
        "profile_reason",
        "compute_environment",
        "training_resources",
        "training_durations",
        "training_duration_seconds",
        "metrics",
        "validation_assessment",
        "model_path",
        "best_model_path",
        "summary_path",
        "canonical_summary_path",
        "config_path",
        "splits_path",
        "test_predictions_path",
        "bundle_file_ref",
        "training_bundle",
        "bundle_download_tag",
        "train_csv",
        "candidate_train_csv",
        "target_columns",
        "trained_at",
        "trained_date",
        "trained_time",
        "applicability_domain",
        "plot_artifacts",
        "recommended_registry_payload",
        "candidate_registry_payloads",
        "recommended_registry_payloads",
        "persistence_plan",
        "candidate_results",
        "ranking",
        "recommended_candidate",
        "recommended_representation_name",
        "representations",
        "campaign_duration_seconds",
        "feature_cache",
        "feature_cache_dir",
        "chemprop_training_input_csv",
        "chemprop_splits_file",
        "chemprop_input_manifest_path",
    )
    compact = {key: result.get(key) for key in keep_keys if result.get(key) is not None}
    compact.update(
        {
            key: result[key]
            for key in (
                "feature_columns_count",
                "feature_columns_sample",
                "feature_columns_omitted_count",
                "feature_columns_source",
                "feature_columns_note",
            )
            if key in result
        }
    )
    if result.get("seed_policy"):
        compact["seed_policy"] = result["seed_policy"]
    if result.get("split_results"):
        compact["split_results"] = [
            _compact_split_result_for_response(item) for item in result.get("split_results") or []
        ]
    if result.get("feature_preparation"):
        compact["feature_preparation"] = _compact_feature_preparation(result["feature_preparation"])
        compact["feature_preparation_durations"] = compact["feature_preparation"].get("durations")
    if result.get("curation"):
        compact["curation"] = {
            key: (result["curation"] or {}).get(key)
            for key in ("dataset_id", "status", "ready_for_qsar", "artifacts")
            if (result["curation"] or {}).get(key) is not None
        }
    if result.get("activity_cliffs"):
        compact["activity_cliffs"] = _compact_activity_cliffs(result["activity_cliffs"])
    return compact


def _candidate_registry_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    payload = _compact_registry_payload(result.get("recommended_registry_payload")) or {}
    if not payload.get("model_id"):
        suffix = _cache_key(
            {
                "candidate_id": result.get("candidate_id"),
                "model_path": result.get("best_model_path") or result.get("model_path"),
            }
        )
        payload["model_id"] = f"{result.get('candidate_id') or 'qsar_candidate'}_{suffix}"
    return payload


def _rank_training_campaign_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> tuple:
        validation = item.get("validation_assessment") or {}
        aggregated = validation.get("aggregated_split_metrics") or {}
        hardest = validation.get("hardest_split")
        hardest_family = aggregated.get(hardest) if hardest else None
        random_family = aggregated.get("random") or {}
        hardest_r2 = (hardest_family or {}).get("r2_mean", (hardest_family or {}).get("r2"))
        random_r2 = random_family.get("r2_mean", random_family.get("r2"))
        duration = ((item.get("training_durations") or {}).get("total_duration_seconds"))
        return (
            float(hardest_r2) if hardest_r2 is not None else float("-inf"),
            float(random_r2) if random_r2 is not None else float("-inf"),
            -float(duration) if duration is not None else 0.0,
        )

    return sorted(results, key=score, reverse=True)


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
            "tabular_representations": describe_tabular_representations(),
            "automatic_tabular_representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
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
        feature_cache_dir: Optional[str] = None,
        feature_n_jobs: Optional[int] = None,
    ) -> Dict[str, Any]:
        spec = get_tabular_representation(representation_name)
        resolved_feature_n_jobs = _resolve_feature_n_jobs(feature_n_jobs)

        total_started_at = time.monotonic()
        duration_steps: List[Dict[str, Any]] = []
        output_path = Path(output_dir).expanduser().resolve()
        features_dir = output_path / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        cache_root = Path(feature_cache_dir).expanduser().resolve() if feature_cache_dir else features_dir / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)

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
        normalized_df[QSAR_ROW_ID_COLUMN] = range(len(normalized_df))
        normalized_csv = features_dir / f"{Path(train_csv).stem}_feature_base.csv"
        with S3.open(str(normalized_csv), "w") as fh:
            normalized_df[[QSAR_ROW_ID_COLUMN, "smiles", *target_columns]].to_csv(fh, index=False)
        base_csv_for_features = str(normalized_csv)
        feature_smiles_column = "smiles"
        duration_steps.append(
            {
                "step": "feature_base_dataset",
                "duration_seconds": round(time.monotonic() - step_started_at, 3),
                "output_csv": base_csv_for_features,
                "row_id_column": QSAR_ROW_ID_COLUMN,
                "source_rows": int(len(normalized_df)),
            }
        )

        dataset_hash = _hash_file(base_csv_for_features)

        def component_from_cache(
            *,
            component_name: str,
            generator_payload: Dict[str, Any],
        ) -> Optional[Dict[str, Any]]:
            component_key_payload = {
                "kind": "feature_component",
                "component_name": component_name,
                "dataset_hash": dataset_hash,
                "smiles_column": feature_smiles_column,
                "target_columns": list(target_columns),
                **generator_payload,
            }
            cache_key = _cache_key(component_key_payload)
            output_csv = cache_root / f"{component_name}_{cache_key}.csv"
            metadata_path = cache_root / f"{component_name}_{cache_key}.json"
            cached = _read_json_if_exists(metadata_path)
            if cached and cached.get("cache_key") == cache_key and _storage_path_exists(output_csv):
                return {
                    "cache_key": cache_key,
                    "output_csv": str(output_csv),
                    "metadata_path": str(metadata_path),
                    "cache_status": "reused_from_cache",
                    "result": cached.get("result") or {"output_csv": str(output_csv)},
                    "key_payload": component_key_payload,
                }
            return {
                "cache_key": cache_key,
                "output_csv": str(output_csv),
                "metadata_path": str(metadata_path),
                "cache_status": "generated",
                "result": None,
                "key_payload": component_key_payload,
            }

        def persist_component_cache(component: Dict[str, Any], result: Dict[str, Any]) -> None:
            _write_json(
                Path(component["metadata_path"]),
                {
                    "cache_key": component["cache_key"],
                    "cache_status": "generated",
                    "key_payload": component["key_payload"],
                    "result": result,
                },
            )

        feature_csvs: List[str] = []
        component_records: List[Dict[str, Any]] = []

        if spec.use_morgan_binary:
            component = component_from_cache(
                component_name="morgan_binary",
                generator_payload={"radius": 2, "n_bits": 2048, "fingerprint_kind": "binary"},
            )
            if component["cache_status"] == "generated":
                step_started_at = time.monotonic()
                result = self.molecular_feature_toolkit.smiles_to_morgan_fingerprints(
                    input_csv=base_csv_for_features,
                    smiles_column=feature_smiles_column,
                    output_csv=component["output_csv"],
                    radius=2,
                    n_bits=2048,
                    include_input_columns=True,
                    input_columns_to_keep=[QSAR_ROW_ID_COLUMN, feature_smiles_column, *target_columns],
                    feature_prefix="fp_",
                    fingerprint_kind="binary",
                    n_jobs=resolved_feature_n_jobs,
                )
                persist_component_cache(component, result)
                duration_seconds = float(
                    result.get("duration_seconds") or round(time.monotonic() - step_started_at, 3)
                )
            else:
                result = component["result"]
                duration_seconds = 0.0
            feature_csvs.append(component["output_csv"])
            component_records.append(component)
            duration_steps.append(
                {
                    "step": _feature_step_name("morgan_binary"),
                    "cache_status": component["cache_status"],
                    "cache_hit": component["cache_status"] == "reused_from_cache",
                    "cache_key": component["cache_key"],
                    "duration_seconds": duration_seconds,
                    "output_csv": component["output_csv"],
                    "num_features": result.get("num_features"),
                    "radius": result.get("radius", 2),
                    "n_bits": result.get("n_bits", 2048),
                    "fingerprint_kind": "binary",
                    "n_jobs": result.get("n_jobs", resolved_feature_n_jobs),
                }
            )

        if spec.use_morgan_count:
            component = component_from_cache(
                component_name="morgan_count",
                generator_payload={"radius": 2, "n_bits": 2048, "fingerprint_kind": "count"},
            )
            if component["cache_status"] == "generated":
                step_started_at = time.monotonic()
                result = self.molecular_feature_toolkit.smiles_to_morgan_fingerprints(
                    input_csv=base_csv_for_features,
                    smiles_column=feature_smiles_column,
                    output_csv=component["output_csv"],
                    radius=2,
                    n_bits=2048,
                    include_input_columns=True,
                    input_columns_to_keep=[QSAR_ROW_ID_COLUMN, feature_smiles_column, *target_columns],
                    feature_prefix="cfp_",
                    fingerprint_kind="count",
                    n_jobs=resolved_feature_n_jobs,
                )
                persist_component_cache(component, result)
                duration_seconds = float(
                    result.get("duration_seconds") or round(time.monotonic() - step_started_at, 3)
                )
            else:
                result = component["result"]
                duration_seconds = 0.0
            feature_csvs.append(component["output_csv"])
            component_records.append(component)
            duration_steps.append(
                {
                    "step": _feature_step_name("morgan_count"),
                    "cache_status": component["cache_status"],
                    "cache_hit": component["cache_status"] == "reused_from_cache",
                    "cache_key": component["cache_key"],
                    "duration_seconds": duration_seconds,
                    "output_csv": component["output_csv"],
                    "num_features": result.get("num_features"),
                    "radius": result.get("radius", 2),
                    "n_bits": result.get("n_bits", 2048),
                    "fingerprint_kind": "count",
                    "n_jobs": result.get("n_jobs", resolved_feature_n_jobs),
                }
            )

        if spec.use_rdkit:
            descriptor_set = str(spec.descriptor_set)
            component_name = f"rdkit_{descriptor_set}"
            component = component_from_cache(
                component_name=component_name,
                generator_payload={"descriptor_set": descriptor_set},
            )
            if component["cache_status"] == "generated":
                step_started_at = time.monotonic()
                result = self.molecular_feature_toolkit.smiles_to_rdkit_descriptors(
                    input_csv=base_csv_for_features,
                    smiles_column=feature_smiles_column,
                    output_csv=component["output_csv"],
                    descriptor_set=descriptor_set,
                    include_input_columns=True,
                    input_columns_to_keep=[QSAR_ROW_ID_COLUMN, feature_smiles_column, *target_columns],
                    n_jobs=resolved_feature_n_jobs,
                )
                persist_component_cache(component, result)
                duration_seconds = float(
                    result.get("duration_seconds") or round(time.monotonic() - step_started_at, 3)
                )
            else:
                result = component["result"]
                duration_seconds = 0.0
            feature_csvs.append(component["output_csv"])
            component_records.append(component)
            duration_steps.append(
                {
                    "step": _feature_step_name(component_name),
                    "cache_status": component["cache_status"],
                    "cache_hit": component["cache_status"] == "reused_from_cache",
                    "cache_key": component["cache_key"],
                    "duration_seconds": duration_seconds,
                    "output_csv": component["output_csv"],
                    "descriptor_set": result.get("descriptor_set", descriptor_set),
                    "num_descriptors": result.get("num_descriptors"),
                    "n_jobs": result.get("n_jobs", resolved_feature_n_jobs),
                }
            )

        tabular_key_payload = {
            "kind": "assembled_tabular_representation",
            "representation_name": representation_name,
            "dataset_hash": dataset_hash,
            "smiles_column": feature_smiles_column,
            "row_id_column": QSAR_ROW_ID_COLUMN,
            "target_columns": list(target_columns),
            "component_cache_keys": [component["cache_key"] for component in component_records],
        }
        tabular_cache_key = _cache_key(tabular_key_payload)
        tabular_output_csv = cache_root / f"tabular_{representation_name}_{tabular_cache_key}.csv"
        tabular_metadata_path = cache_root / f"tabular_{representation_name}_{tabular_cache_key}.json"
        cached_tabular = _read_json_if_exists(tabular_metadata_path)
        tabular_cache_status = "generated"
        if cached_tabular and cached_tabular.get("cache_key") == tabular_cache_key and _storage_path_exists(tabular_output_csv):
            tabular_cache_status = "reused_from_cache"
            tabular = cached_tabular.get("result") or {"output_csv": str(tabular_output_csv)}
            tabular["output_csv"] = str(tabular_output_csv)
            duration_steps.append(
                {
                    "step": "tabular_dataset_assembly",
                    "cache_status": "reused_from_cache",
                    "cache_hit": True,
                    "cache_key": tabular_cache_key,
                    "duration_seconds": 0.0,
                    "output_csv": str(tabular_output_csv),
                    "num_added_feature_columns": tabular.get("num_added_feature_columns"),
                    "final_column_count": tabular.get("final_column_count"),
                    "canonicalize_smiles_join": tabular.get("canonicalize_smiles_join"),
                }
            )
        else:
            step_started_at = time.monotonic()
            tabular = self.molecular_feature_toolkit.build_tabular_qsar_dataset(
                base_csv=base_csv_for_features,
                output_csv=str(tabular_output_csv),
                feature_csvs=feature_csvs,
                join_on=[QSAR_ROW_ID_COLUMN],
                base_columns_to_keep=[QSAR_ROW_ID_COLUMN, "smiles", *target_columns],
                drop_duplicate_feature_columns=True,
                canonicalize_smiles_join=False,
            )
            _write_json(
                tabular_metadata_path,
                {
                    "cache_key": tabular_cache_key,
                    "cache_status": "generated",
                    "key_payload": tabular_key_payload,
                    "result": tabular,
                },
            )
            duration_steps.append(
                {
                    "step": "tabular_dataset_assembly",
                    "cache_status": "generated",
                    "cache_hit": False,
                    "cache_key": tabular_cache_key,
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
        cache_hits = [step for step in duration_steps if step.get("cache_hit") is True]
        cache_misses = [step for step in duration_steps if step.get("cache_hit") is False]
        feature_preparation = {
            "mode": "generated_tabular_features",
            "representation_name": representation_name,
            "representation_display_name": spec.display_name,
            "representation_legacy": spec.legacy,
            "input_csv": train_csv,
            "base_csv_for_features": base_csv_for_features,
            "prepared_train_csv": tabular["output_csv"],
            "feature_csvs": feature_csvs,
            "feature_cache_dir": str(cache_root),
            "feature_cache_key": tabular_cache_key,
            "feature_cache_status": tabular_cache_status,
            "feature_n_jobs": resolved_feature_n_jobs,
            "cache_hits": len(cache_hits),
            "cache_misses": len(cache_misses),
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
        summary_path = result.get("summary_path") or result.get("canonical_summary_path")
        feature_columns = list(result.get("feature_columns") or [])
        model_path = result.get("best_model_path") or result.get("model_path")
        model_id = (
            f"{backend_name}_{result.get('representation_name') or 'model'}_"
            f"{_cache_key({'model_path': model_path, 'validation_protocol': result.get('validation_protocol')})}"
        )
        return {
            "model_id": model_id,
            "backend_name": backend_name,
            "model_path": model_path,
            "task_type": task_type,
            "smiles_columns": [smiles_column],
            "target_columns": list(target_columns),
            "known_metrics": result.get("metrics") or {},
            "training_data_summary": {
                "validation_protocol": result.get("validation_protocol"),
                "training_profile": result.get("training_profile"),
                "seed_policy": result.get("seed_policy"),
                "representation_name": result.get("representation_name"),
                "feature_preparation": _compact_feature_preparation(
                    result.get("feature_preparation") or {}
                ),
                "training_summary_path": summary_path,
            },
            "inference_profile": {
                "representation_name": result.get("representation_name"),
                **_feature_columns_summary(feature_columns, source=summary_path),
            },
            "applicability_domain": result.get("applicability_domain") or {},
        }

    def _split_registry_payloads(
        self,
        *,
        backend_name: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        result: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for index, split_result in enumerate(result.get("split_results") or [], start=1):
            model_path = split_result.get("best_model_path") or split_result.get("model_path")
            if not model_path:
                continue
            label = split_result.get("strategy_label") or split_result.get("strategy") or f"split_{index}"
            split_context = {
                **result,
                **split_result,
                "model_path": model_path,
                "best_model_path": model_path,
                "feature_columns": result.get("feature_columns") or [],
                "feature_preparation": result.get("feature_preparation") or {},
                "representation_name": result.get("representation_name"),
                "validation_protocol": f"{result.get('validation_protocol')}_{label}",
            }
            payloads.append(
                {
                    "rank": index,
                    "candidate_id": f"{backend_name}_{result.get('representation_name')}_{label}",
                    "backend_name": backend_name,
                    "representation_name": result.get("representation_name"),
                    "split_label": label,
                    "registry_payload": self._recommended_registry_payload(
                        backend_name=backend_name,
                        task_type=task_type,
                        smiles_column=smiles_column,
                        target_columns=target_columns,
                        result=split_context,
                    ),
                }
            )
        return payloads

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

    def _compact_campaign_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        validation = result.get("validation_assessment") or {}
        aggregated = validation.get("aggregated_split_metrics") or {}
        hardest = validation.get("hardest_split")
        hardest_family = aggregated.get(hardest) if hardest else None
        random_family = aggregated.get("random") or {}
        scaffold_family = aggregated.get("scaffold") or {}
        feature_prep = result.get("feature_preparation") or {}
        return {
            "backend_name": result.get("backend_name"),
            "representation_name": result.get("representation_name"),
            "model_path": result.get("best_model_path") or result.get("model_path"),
            "candidate_train_csv": result.get("candidate_train_csv"),
            "summary_path": result.get("summary_path") or result.get("canonical_summary_path"),
            "random_r2": (random_family or {}).get("r2_mean", (random_family or {}).get("r2")),
            "scaffold_r2": (scaffold_family or {}).get("r2_mean", (scaffold_family or {}).get("r2")),
            "hardest_split": hardest,
            "hardest_split_r2": (hardest_family or {}).get("r2_mean", (hardest_family or {}).get("r2"))
            if hardest_family
            else None,
            **_feature_columns_summary_from_result(result),
            "feature_cache_key": feature_prep.get("feature_cache_key"),
            "feature_cache_status": feature_prep.get("feature_cache_status"),
            "cache_hits": feature_prep.get("cache_hits"),
            "cache_misses": feature_prep.get("cache_misses"),
            "feature_preparation_duration_seconds": (
                (feature_prep.get("durations") or {}).get("total_duration_seconds")
            ),
            "feature_preparation_durations": _compact_feature_preparation_durations(feature_prep),
            "training_duration_seconds": (result.get("training_durations") or {}).get(
                "total_duration_seconds"
            ),
        }

    def _train_tabular_representation_campaign(
        self,
        *,
        train_csv: str,
        backend_name: str,
        task_type: str,
        output_dir: str,
        smiles_column: str,
        target_columns: List[str],
        validation_protocol: str,
        validation_strategy: Optional[Dict[str, Any]],
        activity_cliff_index: str,
        activity_cliff_feedback: bool,
        activity_cliff_feedback_loops: int,
        activity_cliff_similarity_threshold: float,
        activity_cliff_top_k_neighbors: int,
        activity_cliff_flag_threshold: float,
        extra_args: Dict[str, Any],
        agent: Optional[Agent],
    ) -> Dict[str, Any]:
        campaign_started_at = time.monotonic()
        campaign_root = Path(output_dir).expanduser().resolve()
        campaign_root.mkdir(parents=True, exist_ok=True)
        feature_cache_dir = str(Path(extra_args.get("feature_cache_dir") or campaign_root / "feature_cache"))
        candidate_results: List[Dict[str, Any]] = []

        for representation_name in AUTOMATIC_TABULAR_REPRESENTATION_NAMES:
            candidate_dir = campaign_root / f"{backend_name}_{representation_name}"
            candidate_extra_args = dict(extra_args)
            candidate_extra_args["feature_cache_dir"] = feature_cache_dir
            result = self.train_qsar_model(
                train_csv=train_csv,
                backend_name=backend_name,
                task_type=task_type,
                output_dir=str(candidate_dir),
                smiles_column=smiles_column,
                target_columns=list(target_columns),
                validation_protocol=validation_protocol,
                validation_strategy=validation_strategy,
                representation_name=representation_name,
                activity_cliff_index=activity_cliff_index,
                activity_cliff_feedback=activity_cliff_feedback,
                activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                extra_args=candidate_extra_args,
                agent=agent,
            )
            result["candidate_id"] = f"{backend_name}_{representation_name}"
            candidate_results.append(result)

        ranked_results = _rank_training_campaign_results(candidate_results)
        best_result = ranked_results[0]
        rows = [self._compact_campaign_row(result) for result in ranked_results]
        payload_by_candidate_id = {
            result.get("candidate_id"): _candidate_registry_payload(result)
            for result in candidate_results
        }
        candidate_registry_payloads = [
            {
                "rank": index,
                "candidate_id": row.get("backend_name")
                and f"{row.get('backend_name')}_{row.get('representation_name')}",
                "backend_name": row.get("backend_name"),
                "representation_name": row.get("representation_name"),
                "registry_payload": payload_by_candidate_id.get(
                    f"{row.get('backend_name')}_{row.get('representation_name')}"
                ),
            }
            for index, row in enumerate(rows, start=1)
        ]
        cache_hits = sum(
            int((result.get("feature_preparation") or {}).get("cache_hits") or 0)
            for result in candidate_results
        )
        cache_misses = sum(
            int((result.get("feature_preparation") or {}).get("cache_misses") or 0)
            for result in candidate_results
        )
        campaign_result = {
            "campaign_started": True,
            "campaign_type": "tabular_representation_campaign",
            "backend_name": backend_name,
            "task_type": task_type,
            "validation_protocol": validation_protocol,
            "validation_strategy": validation_strategy,
            "representations": list(AUTOMATIC_TABULAR_REPRESENTATION_NAMES),
            "feature_cache_dir": feature_cache_dir,
            "campaign_duration_seconds": round(time.monotonic() - campaign_started_at, 3),
            "feature_cache": {
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "feature_cache_dir": feature_cache_dir,
            },
            "candidate_results": rows,
            "ranking": rows,
            "recommended_candidate": best_result.get("candidate_id")
            or f"{backend_name}_{best_result.get('representation_name')}",
            "recommended_representation_name": best_result.get("representation_name"),
            "recommended_registry_payload": _candidate_registry_payload(best_result),
            "candidate_registry_payloads": candidate_registry_payloads,
            "recommended_registry_payloads": [
                item["registry_payload"] for item in candidate_registry_payloads
            ],
            "persistence_plan": {
                "persist_all_candidates": True,
                "candidate_count": len(candidate_registry_payloads),
                "candidate_registry_payloads_key": "candidate_registry_payloads",
                "required_tool_sequence": (
                    "For each candidate_registry_payloads item: call register_model with "
                    "`registry_payload`, then persist_registered_model with the returned "
                    "temporary model_id. Report every persisted canonical catalog model_id."
                ),
                "recommended_candidate": best_result.get("candidate_id")
                or f"{backend_name}_{best_result.get('representation_name')}",
            },
            "best_model_path": best_result.get("best_model_path") or best_result.get("model_path"),
            "model_path": best_result.get("best_model_path") or best_result.get("model_path"),
            "train_csv": train_csv,
            "candidate_train_csv": best_result.get("candidate_train_csv"),
            **_feature_columns_summary_from_result(best_result),
            "feature_preparation": _compact_feature_preparation(
                best_result.get("feature_preparation") or {}
            ),
            "feature_preparation_durations": _compact_feature_preparation_durations(
                best_result.get("feature_preparation") or {}
            ),
            "training_duration_seconds": (
                (best_result.get("training_durations") or {}).get("total_duration_seconds")
            ),
            "validation_assessment": best_result.get("validation_assessment") or {},
        }
        summary_path = campaign_root / "qsar_training_campaign_summary.json"
        write_training_summary(summary_path, campaign_result)
        campaign_result["summary_path"] = str(summary_path)
        return _compact_training_tool_result(campaign_result)

    def train_qsar_model(
        self,
        train_csv: str,
        backend_name: str,
        task_type: str,
        output_dir: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        validation_strategy: Optional[Dict[str, Any]] = None,
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
        requested_validation_strategy = (
            validation_strategy if validation_strategy is not None else requested_extra_args.pop("validation_strategy", None)
        )
        requested_extra_args.setdefault("validation_protocol", validation_protocol)

        if normalized_backend == "chemprop":
            if requested_validation_strategy is not None:
                requested_extra_args["validation_strategy"] = requested_validation_strategy
            result = self.chemprop_toolkit.train_model(
                train_csv=train_csv,
                task_type=task_type,
                output_dir=output_dir,
                smiles_columns=[smiles_column],
                target_columns=list(normalized_target_columns),
                validation_strategy=requested_validation_strategy,
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
            if (
                not representation_name
                and not normalized_feature_columns
                and (
                    requested_validation_strategy is not None
                    or validation_protocol in {"standard_qsar", "robust_qsar"}
                )
            ):
                return self._train_tabular_representation_campaign(
                    train_csv=train_csv,
                    backend_name=normalized_backend,
                    task_type=task_type,
                    output_dir=output_dir,
                    smiles_column=smiles_column,
                    target_columns=list(normalized_target_columns),
                    validation_protocol=validation_protocol,
                    validation_strategy=requested_validation_strategy,
                    activity_cliff_index=activity_cliff_index,
                    activity_cliff_feedback=activity_cliff_feedback,
                    activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                    activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                    activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                    activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                    extra_args=requested_extra_args,
                    agent=agent,
                )
            working_train_csv = train_csv
            resolved_representation = representation_name
            if not normalized_feature_columns:
                resolved_representation = resolved_representation or default_tabular_representation_for_protocol(
                    validation_protocol,
                    training_profile=requested_extra_args.get("training_profile"),
                )
                prepared = self._prepare_tabular_training_dataset(
                    train_csv=train_csv,
                    output_dir=output_dir,
                    smiles_column=smiles_column,
                    target_columns=list(normalized_target_columns),
                    representation_name=resolved_representation,
                    feature_cache_dir=requested_extra_args.get("feature_cache_dir"),
                    feature_n_jobs=requested_extra_args.get("feature_n_jobs") or requested_extra_args.get("n_jobs"),
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
                    validation_strategy=requested_validation_strategy,
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
                    validation_strategy=requested_validation_strategy,
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

        result["recommended_registry_payload"] = self._recommended_registry_payload(
            backend_name=normalized_backend,
            task_type=task_type,
            smiles_column=smiles_column,
            target_columns=list(normalized_target_columns),
            result=result,
        )
        split_registry_payloads = self._split_registry_payloads(
            backend_name=normalized_backend,
            task_type=task_type,
            smiles_column=smiles_column,
            target_columns=list(normalized_target_columns),
            result=result,
        )
        if len(split_registry_payloads) > 1:
            result["candidate_registry_payloads"] = split_registry_payloads
            result["recommended_registry_payloads"] = [
                item["registry_payload"] for item in split_registry_payloads
            ]
            result["persistence_plan"] = {
                "persist_all_candidates": True,
                "candidate_count": len(split_registry_payloads),
                "candidate_registry_payloads_key": "candidate_registry_payloads",
                "required_tool_sequence": (
                    "For each candidate_registry_payloads item: call register_model with "
                    "`registry_payload`, then persist_registered_model with the returned "
                    "temporary model_id. Report every persisted canonical catalog model_id."
                ),
            }
        self._refresh_enriched_training_artifacts(
            result=result,
            output_dir=output_dir,
        )
        if normalized_backend in {"lightgbm", "tabicl"} and isinstance(result.get("feature_columns"), list):
            full_feature_columns = list(result.pop("feature_columns") or [])
            result.update(
                _feature_columns_summary(
                    full_feature_columns,
                    source=result.get("summary_path") or result.get("canonical_summary_path"),
                )
            )
        return _compact_training_tool_result(result)

    def train_chemprop_model(
        self,
        train_csv: str,
        task_type: str,
        output_dir: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str] | str] = None,
        validation_protocol: str = "standard_qsar",
        validation_strategy: Optional[Dict[str, Any]] = None,
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
            validation_strategy=validation_strategy,
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
        validation_strategy: Optional[Dict[str, Any]] = None,
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
            validation_strategy=validation_strategy,
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
        validation_strategy: Optional[Dict[str, Any]] = None,
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
            validation_strategy=validation_strategy,
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
