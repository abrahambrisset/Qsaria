#!/usr/bin/env python
# coding: utf-8
"""Agent-facing QSAR training facade."""

from __future__ import annotations

import hashlib
import json
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
from .hyperparameter_tuning import describe_backend_hyperparameters, describe_tuning_engines
from .lightgbm_toolkit import LightGBMToolkit
from .outlier_analysis import describe_outlier_analysis_policy
from .qsar_contracts import (
    ChempropTrainingRequest,
    GeneratedRepresentation,
    LightGBMTrainingRequest,
    PrecomputedRepresentation,
    QsariaTrainingRequest,
    ResolvedTrainingPlan,
    RuntimePaths,
    StandardQsarValidation,
    TabICLTrainingRequest,
    canonical_validation_strategy,
)
from .qsar_reporting import build_training_reporting_handoff
from .qsar_response_compaction import (
    compact_applicability_domain_for_response as _compact_applicability_domain_for_response,
)
from .qsar_response_compaction import (
    compact_registry_payload_for_response,
    feature_columns_summary_for_response,
)
from .qsar_training_policy import describe_compute_environment, resolve_training_profile
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

QSAR_ROW_ID_COLUMN = "__qsar_row_id"
PERSISTENCE_MANIFEST_SCHEMA_VERSION = "1.0"
PERSISTENCE_MANIFEST_FILENAME = "catalog_candidates_manifest.json"
OUTLIER_VARIANTS_MANIFEST_FILENAME = "outlier_model_variants.json"


def _agent_storage_path(path: str | Path) -> str:
    """Normalize agent-returned storage paths before passing them to S3.open."""
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


def _feature_columns_from_csv(path: str, target_columns: List[str]) -> List[str]:
    with S3.open(_agent_storage_path(path), "r") as fh:
        columns = list(pd.read_csv(fh, nrows=0).columns)
    excluded = {"smiles", QSAR_ROW_ID_COLUMN, *target_columns}
    return [column for column in columns if column not in excluded]


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with S3.open(_agent_storage_path(path), "rb") as fh:
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
        with S3.open(_agent_storage_path(path), "r") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _storage_path_exists(path: Path | str) -> bool:
    try:
        with S3.open(_agent_storage_path(path), "rb") as fh:
            fh.read(1)
        return True
    except FileNotFoundError:
        return False


def _resolve_existing_training_csv(train_csv: str, agent: Optional[Agent] = None) -> str:
    """Return an existing training CSV, preferring explicit paths then latest curation."""
    normalized = _agent_storage_path(train_csv)
    if _storage_path_exists(normalized):
        return normalized

    if agent is not None:
        latest = latest_curation_artifacts(agent)
        candidates = [
            latest.get("curated_dataset_path"),
            (latest.get("artifacts") or {}).get("curated_dataset_csv"),
        ]
        for candidate in candidates:
            if candidate and _storage_path_exists(str(candidate)):
                return str(candidate)

    discovered = discover_curation_artifacts_near_dataset(train_csv)
    candidates = [
        discovered.get("curated_dataset_path"),
        (discovered.get("artifacts") or {}).get("curated_dataset_csv"),
    ]
    for candidate in candidates:
        if candidate and _storage_path_exists(str(candidate)):
            return str(candidate)

    return normalized


def _resolve_feature_n_jobs(raw: Optional[Any] = None) -> int:
    if raw is not None:
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1
    return max(1, min(int(describe_compute_environment().get("cpu_count") or 1), 16))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with S3.open(_agent_storage_path(path), "w") as fh:
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
    return feature_columns_summary_for_response(feature_columns, source=source)


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
    if inference_profile.get("feature_columns") and not inference_profile.get(
        "feature_columns_source"
    ):
        inference_profile["feature_columns_source"] = (
            compacted.get("training_data_summary") or {}
        ).get("training_summary_path")
    compacted["inference_profile"] = inference_profile
    return compact_registry_payload_for_response(compacted)


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
            "validation_predictions_path",
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
    } | {"metrics": metrics, "target_metrics": split_result.get("target_metrics") or {}}


def _compact_outlier_variant_for_response(variant: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the agent handoff descriptive without embedding a complete refit run.

    The exact run, including split index arrays and detailed AD data, remains in
    the outlier variants manifest.  Returning it verbatim is particularly
    costly for CV because each final candidate repeats the same development
    split provenance.
    """
    run = variant.get("run") if isinstance(variant, dict) else None
    run = run if isinstance(run, dict) else {}
    compact_run = {
        key: run.get(key)
        for key in (
            "outlier_variant",
            "model_path",
            "best_model_path",
            "output_dir",
            "config_path",
            "splits_path",
            "validation_predictions_path",
            "test_predictions_path",
            "metrics",
            "metrics_status",
            "effective_train_count",
            "validation_count",
            "test_count",
            "removed_from_train_count",
            "requested_exclusion_count",
            "outlier_selected_count",
            "selected_hyperparameters",
            "source_folds",
            "repeat_index",
            "fold_index",
            "source_split_label",
        )
        if run.get(key) is not None
    }
    applicability_domain = run.get("applicability_domain")
    if isinstance(applicability_domain, dict):
        compact_run["applicability_domain"] = {
            key: applicability_domain.get(key)
            for key in (
                "available",
                "primary_method",
                "method",
                "manifest_path",
                "plots_dir",
                "scores_validation_path",
                "scores_test_path",
            )
            if applicability_domain.get(key) is not None
        }
    return {
        key: variant.get(key)
        for key in (
            "variant_id",
            "configuration_id",
            "repeat_index",
            "fold_index",
            "source_split_label",
            "source_folds",
        )
        if variant.get(key) is not None
    } | {"run": compact_run}


def _manifest_root(result: Dict[str, Any], output_dir: str) -> Path:
    summary_path = result.get("summary_path") or result.get("canonical_summary_path")
    if summary_path:
        return Path(str(summary_path)).expanduser().parent
    return Path(output_dir).expanduser()


def _candidate_manifest_index(candidate: Dict[str, Any], index: int) -> Dict[str, Any]:
    registry_payload = candidate.get("registry_payload") if isinstance(candidate, dict) else None
    registry_payload = registry_payload if isinstance(registry_payload, dict) else {}
    return {
        "rank": candidate.get("rank", index),
        "candidate_id": candidate.get("candidate_id"),
        "split_label": candidate.get("split_label"),
        "backend_name": candidate.get("backend_name"),
        "representation_name": candidate.get("representation_name"),
        "model_id": registry_payload.get("model_id"),
        "outlier_variant": (registry_payload.get("training_data_summary") or {}).get(
            "outlier_variant"
        ),
    }


def _materialize_candidate_persistence_manifest(
    *,
    result: Dict[str, Any],
    output_dir: str,
) -> None:
    """Persist lossless registry inputs outside the conversational handoff."""
    candidates = result.get("candidate_registry_payloads")
    if not isinstance(candidates, list) or not candidates:
        return

    root = _manifest_root(result, output_dir)
    manifest_path = root / PERSISTENCE_MANIFEST_FILENAME
    write_training_summary(
        manifest_path,
        {
            "schema_version": PERSISTENCE_MANIFEST_SCHEMA_VERSION,
            "training_summary_path": result.get("summary_path")
            or result.get("canonical_summary_path"),
            "candidate_registry_payloads": candidates,
        },
    )
    candidate_index = [
        _candidate_manifest_index(candidate, index)
        for index, candidate in enumerate(candidates, start=1)
        if isinstance(candidate, dict)
    ]
    plan = dict(result.get("persistence_plan") or {})
    plan.update(
        {
            "persist_all_candidates": True,
            "candidate_count": len(candidate_index),
            "candidate_manifest_path": str(manifest_path),
            "candidate_manifest_schema_version": PERSISTENCE_MANIFEST_SCHEMA_VERSION,
            "required_tool_sequence": (
                "Call register_and_persist_candidates with the exact "
                "candidate_manifest_path and report every returned canonical catalog model_id."
            ),
        }
    )
    plan.pop("candidate_registry_payloads_key", None)
    result["persistence_plan"] = plan
    result["candidate_manifest_path"] = str(manifest_path)
    result["candidate_persistence_manifest"] = {
        "path": str(manifest_path),
        "schema_version": PERSISTENCE_MANIFEST_SCHEMA_VERSION,
        "candidate_count": len(candidate_index),
        "candidates": candidate_index,
    }
    # The manifest is the authoritative single copy.  Keeping both payload
    # lists in the summary and the tool response multiplies CV context size.
    result.pop("candidate_registry_payloads", None)
    result.pop("recommended_registry_payloads", None)


def _materialize_outlier_variants_manifest(
    *,
    result: Dict[str, Any],
    output_dir: str,
    compact_for_response: bool = True,
) -> None:
    """Move detailed final-refit runs to a dedicated audit artifact."""
    variants = result.get("outlier_model_variants")
    if not isinstance(variants, list) or not variants:
        return

    root = _manifest_root(result, output_dir)
    manifest_path = root / "artifacts" / "outlier_analysis" / OUTLIER_VARIANTS_MANIFEST_FILENAME
    write_training_summary(
        manifest_path,
        {
            "schema_version": PERSISTENCE_MANIFEST_SCHEMA_VERSION,
            "training_summary_path": result.get("summary_path")
            or result.get("canonical_summary_path"),
            "variants": variants,
        },
    )
    result["outlier_model_variants_manifest_path"] = str(manifest_path)
    if compact_for_response:
        result["outlier_model_variants"] = [
            _compact_outlier_variant_for_response(variant)
            for variant in variants
            if isinstance(variant, dict)
        ]


def _compact_activity_cliffs(activity_cliffs: Dict[str, Any]) -> Dict[str, Any]:
    if not activity_cliffs:
        return {}
    return {
        key: activity_cliffs.get(key)
        for key in (
            "enabled",
            "mode",
            "index_name",
            "target_column",
            "flagged_count",
            "priority_counts",
            "index_parameters",
            "tiering_policy",
            "evaluation_policy",
            "selection_policy",
            "warnings",
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
        "training_contract_version",
        "requested_contract",
        "resolved_plan",
        "effective_parameters",
        "runtime",
        "workflow",
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
        "target_metrics",
        "validation_assessment",
        "model_path",
        "best_model_path",
        "summary_path",
        "canonical_summary_path",
        "config_path",
        "splits_path",
        "validation_predictions_path",
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
        "plot_artifacts",
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
        "reporting_handoff",
        "outlier_analysis",
        "outlier_model_variants",
        "outlier_model_variants_manifest_path",
        "candidate_manifest_path",
        "candidate_persistence_manifest",
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
    if result.get("applicability_domain"):
        compact["applicability_domain"] = _compact_applicability_domain_for_response(
            result["applicability_domain"]
        )
    if result.get("recommended_registry_payload"):
        compact["recommended_registry_payload"] = _compact_registry_payload(
            result["recommended_registry_payload"]
        )
    if result.get("recommended_registry_payloads") and not result.get(
        "candidate_persistence_manifest"
    ):
        compact["recommended_registry_payloads"] = [
            _compact_registry_payload(item) or {}
            for item in result.get("recommended_registry_payloads") or []
        ]
    if result.get("candidate_registry_payloads") and not result.get(
        "candidate_persistence_manifest"
    ):
        compact["candidate_registry_payloads"] = []
        for item in result.get("candidate_registry_payloads") or []:
            row = dict(item or {})
            if row.get("registry_payload"):
                row["registry_payload"] = _compact_registry_payload(row["registry_payload"])
            compact["candidate_registry_payloads"].append(row)
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
    if result.get("outlier_model_variants"):
        compact["outlier_model_variants"] = [
            _compact_outlier_variant_for_response(variant)
            for variant in result.get("outlier_model_variants") or []
            if isinstance(variant, dict)
        ]
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
        duration = (item.get("training_durations") or {}).get("total_duration_seconds")
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
        self.chemprop_toolkit = chemprop_toolkit or ChempropToolkit()
        self.lightgbm_toolkit = lightgbm_toolkit or LightGBMToolkit()
        self.tabicl_toolkit = tabicl_toolkit or TabICLToolkit()
        self.molecular_feature_toolkit = molecular_feature_toolkit or MolecularFeatureToolkit()

        self.register(self.describe_qsar_training_environment)
        self.register(self.describe_backend_hyperparameters)
        self.register(self.describe_tuning_engines)
        self.register(self.describe_outlier_analysis)
        self.register(self._agno_train_qsar_model, name="train_qsar_model")
        self.register(self._agno_train_chemprop_model, name="train_chemprop_model")
        self.register(self._agno_train_lightgbm_model, name="train_lightgbm_model")
        self.register(self._agno_train_tabicl_model, name="train_tabicl_model")
        for function in self.functions.values():
            function.process_entrypoint()
            function.parameters["additionalProperties"] = False
            function.skip_entrypoint_processing = True

    @staticmethod
    def _managed_agno_paths(backend_name: str) -> tuple[str, str]:
        """Allocate session-scoped Agno paths without exposing them to the model."""

        token = f"{time.time_ns():x}"
        output_dir = S3.path(f"training/{backend_name}_{token}")
        bundle_dir = S3.path("training_bundles")
        return output_dir, bundle_dir

    def _agno_train_qsar_model(
        self,
        train_csv: str,
        request: QsariaTrainingRequest,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        output_dir, bundle_dir = self._managed_agno_paths(request.backend.name)
        return self.train_qsar_model(
            train_csv=train_csv,
            request=request,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

    def _agno_train_chemprop_model(
        self,
        train_csv: str,
        request: ChempropTrainingRequest,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        output_dir, bundle_dir = self._managed_agno_paths("chemprop")
        return self.train_chemprop_model(
            train_csv=train_csv,
            request=request,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

    def _agno_train_lightgbm_model(
        self,
        train_csv: str,
        request: LightGBMTrainingRequest,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        output_dir, bundle_dir = self._managed_agno_paths("lightgbm")
        return self.train_lightgbm_model(
            train_csv=train_csv,
            request=request,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

    def _agno_train_tabicl_model(
        self,
        train_csv: str,
        request: TabICLTrainingRequest,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        output_dir, bundle_dir = self._managed_agno_paths("tabicl")
        return self.train_tabicl_model(
            train_csv=train_csv,
            request=request,
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

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
            "backend_hyperparameters": describe_backend_hyperparameters(),
            "tuning_engines": describe_tuning_engines(),
            "outlier_analysis": describe_outlier_analysis_policy(),
            "toolkit": "QSARTrainingToolkit",
        }

    def describe_backend_hyperparameters(
        self, backend_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Describe Qsaria-supported direct and tunable backend hyperparameters."""
        return describe_backend_hyperparameters(backend_name)

    def describe_tuning_engines(self, engine_name: Optional[str] = None) -> Dict[str, Any]:
        """Describe shared tuning engines and their real backend availability."""
        return describe_tuning_engines(engine_name)

    def describe_outlier_analysis(self) -> Dict[str, Any]:
        """Describe the shared post-selection outlier analysis policy."""
        return describe_outlier_analysis_policy()

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
        confirm_explicit_export_request: bool = False,
    ) -> Dict[str, Any]:
        """Normalize a QSAR training CSV into canonical `smiles` + target columns."""
        if not confirm_explicit_export_request:
            raise ValueError(
                "prepare_training_dataset is an export/debug helper, not a required training step. "
                "For backend training, call train_lightgbm_model, train_tabicl_model, or "
                "train_chemprop_model directly with the curated dataset. Retry this helper with "
                "confirm_explicit_export_request=True only when the user explicitly asked for a "
                "separate training-ready CSV artifact."
            )

        with S3.open(_agent_storage_path(input_csv), "r") as fh:
            df = pd.read_csv(fh)

        normalized_target_columns = (
            normalize_json_list_argument(
                target_columns,
                argument_name="target_columns",
            )
            or []
        )

        resolved_smiles_column = resolve_smiles_column_name(df, smiles_column)
        df = standardize_smiles_column(df, resolved_smiles_column)
        if resolved_smiles_column != "smiles":
            df["smiles"] = df[resolved_smiles_column]
            df = df.drop(columns=[resolved_smiles_column])
        missing_targets = [
            column for column in normalized_target_columns if column not in df.columns
        ]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")

        standardized = df[["smiles", *normalized_target_columns]].copy()
        destination = output_csv or "training/qsar_training_dataset.csv"
        with S3.open(_agent_storage_path(destination), "w") as fh:
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
        cache_root = (
            Path(feature_cache_dir).expanduser().resolve()
            if feature_cache_dir
            else features_dir / "cache"
        )
        cache_root.mkdir(parents=True, exist_ok=True)

        step_started_at = time.monotonic()
        with S3.open(_agent_storage_path(train_csv), "r") as fh:
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
                    input_columns_to_keep=[
                        QSAR_ROW_ID_COLUMN,
                        feature_smiles_column,
                        *target_columns,
                    ],
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
                    input_columns_to_keep=[
                        QSAR_ROW_ID_COLUMN,
                        feature_smiles_column,
                        *target_columns,
                    ],
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
                    input_columns_to_keep=[
                        QSAR_ROW_ID_COLUMN,
                        feature_smiles_column,
                        *target_columns,
                    ],
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
        tabular_metadata_path = (
            cache_root / f"tabular_{representation_name}_{tabular_cache_key}.json"
        )
        cached_tabular = _read_json_if_exists(tabular_metadata_path)
        tabular_cache_status = "generated"
        if (
            cached_tabular
            and cached_tabular.get("cache_key") == tabular_cache_key
            and _storage_path_exists(tabular_output_csv)
        ):
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
        cross_validation = result.get("cross_validation") or {}
        known_metrics = result.get("metrics") or {}
        metrics_status = result.get("metrics_status") or (
            "not_evaluated" if result.get("evaluation_required") else "evaluated"
        )
        if cross_validation.get("summary"):
            known_metrics = {
                "cross_validation": cross_validation["summary"],
                "final_refit": known_metrics,
            }
        if metrics_status == "not_evaluated":
            known_metrics = {}
        # Persisting outlier variants registers each final fit independently.
        # Keep the artifact provenance with that exact fit instead of relying
        # on the campaign-level summary, whose primary run is the baseline.
        # The model registry consumes these paths when materializing the
        # catalog entry.
        artifact_sources = {
            "training_summary_path": summary_path,
            "config_path": result.get("config_path"),
            "splits_path": result.get("splits_path"),
            "validation_predictions_path": result.get("validation_predictions_path"),
            "test_predictions_path": result.get("test_predictions_path"),
            "hyperparameter_tuning_summary_path": result.get("hyperparameter_tuning_summary_path")
            or (result.get("hyperparameter_tuning") or {}).get("summary_path"),
            "applicability_domain": result.get("applicability_domain") or {},
            "plot_artifacts": result.get("plot_artifacts") or {},
            "activity_cliffs": result.get("activity_cliffs") or {},
            "curation": result.get("curation") or {},
            "feature_preparation": result.get("feature_preparation") or {},
            "outlier_analysis": result.get("outlier_analysis") or {},
        }
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
            "known_metrics": known_metrics,
            "status": "workflow_demo" if metrics_status == "not_evaluated" else "experimental",
            "training_data_summary": {
                "validation_protocol": result.get("validation_protocol"),
                "validation_strategy_type": result.get("validation_strategy_type"),
                "validation_strategy": result.get("validation_strategy"),
                "metrics_status": metrics_status,
                "evaluation_required": bool(result.get("evaluation_required")),
                "external_evaluations": [],
                "training_profile": result.get("training_profile"),
                "seed_policy": result.get("seed_policy"),
                "representation_name": result.get("representation_name"),
                "task_kind": result.get("task_kind"),
                "class_labels": result.get("class_labels") or [],
                "class_count": result.get("class_count"),
                "label_mapping": result.get("label_mapping") or {},
                "positive_class_label": result.get("positive_class_label"),
                "feature_preparation": _compact_feature_preparation(
                    result.get("feature_preparation") or {}
                ),
                "training_summary_path": summary_path,
                "artifact_sources": artifact_sources,
                "hyperparameter_tuning": result.get("catalog_hyperparameter_tuning")
                or result.get("hyperparameter_tuning_metadata")
                or {},
                "hyperparameter_tuning_summary_path": result.get(
                    "hyperparameter_tuning_summary_path"
                )
                or (result.get("hyperparameter_tuning") or {}).get("summary_path"),
                "cross_validation": cross_validation,
                "catalog_model_policy": result.get("catalog_model_policy"),
                "outlier_analysis": result.get("outlier_analysis") or {},
                "outlier_variant": result.get("outlier_variant"),
            },
            "inference_profile": {
                "representation_name": result.get("representation_name"),
                "task_kind": result.get("task_kind"),
                "class_labels": result.get("class_labels") or [],
                "class_count": result.get("class_count"),
                "label_mapping": result.get("label_mapping") or {},
                "positive_class_label": result.get("positive_class_label"),
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
        variant_models = result.get("outlier_model_variants") or []
        if variant_models:
            for index, variant in enumerate(variant_models, start=1):
                run = variant.get("run") if isinstance(variant, dict) else None
                if not isinstance(run, dict):
                    continue
                model_path = run.get("best_model_path") or run.get("model_path")
                if not model_path:
                    continue
                variant_id = str(variant.get("variant_id") or run.get("outlier_variant") or index)
                context = {
                    **result,
                    **run,
                    "model_path": model_path,
                    "best_model_path": model_path,
                    "outlier_variant": variant_id,
                    "applicability_domain": run.get("applicability_domain")
                    or result.get("applicability_domain")
                    or {},
                    "validation_protocol": f"{result.get('validation_protocol')}_{variant_id}",
                }
                payloads.append(
                    {
                        "rank": index,
                        "candidate_id": f"{backend_name}_{result.get('representation_name')}_{variant_id}",
                        "backend_name": backend_name,
                        "representation_name": result.get("representation_name"),
                        "split_label": variant_id,
                        "registry_payload": self._recommended_registry_payload(
                            backend_name=backend_name,
                            task_type=task_type,
                            smiles_column=smiles_column,
                            target_columns=target_columns,
                            result=context,
                        ),
                    }
                )
            if payloads:
                return payloads
        for index, split_result in enumerate(result.get("split_results") or [], start=1):
            model_path = split_result.get("best_model_path") or split_result.get("model_path")
            if not model_path:
                continue
            label = (
                split_result.get("strategy_label")
                or split_result.get("strategy")
                or f"split_{index}"
            )
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
            "scaffold_r2": (scaffold_family or {}).get(
                "r2_mean", (scaffold_family or {}).get("r2")
            ),
            "hardest_split": hardest,
            "hardest_split_r2": (
                (hardest_family or {}).get("r2_mean", (hardest_family or {}).get("r2"))
                if hardest_family
                else None
            ),
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

    def train_qsar_model(
        self,
        train_csv: str,
        request: QsariaTrainingRequest,
        output_dir: str,
        bundle_dir: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train a QSAR model from the strict public training contract."""
        if isinstance(request, dict):
            request = QsariaTrainingRequest.model_validate(request)
        train_csv = _resolve_existing_training_csv(train_csv, agent)
        normalized_backend = request.backend.name
        task_type = request.task_type
        smiles_column = request.smiles_column
        normalized_target_columns = list(request.target_columns)
        validation_protocol = "standard_qsar"
        requested_validation_strategy = canonical_validation_strategy(request.validation)
        representation_name: Optional[str]
        normalized_feature_columns: Optional[List[str]]
        normalized_categorical_feature_columns: Optional[List[str]]
        if isinstance(request.representation, GeneratedRepresentation):
            representation_name = request.representation.name
            normalized_feature_columns = None
            normalized_categorical_feature_columns = None
        elif isinstance(request.representation, PrecomputedRepresentation):
            representation_name = "precomputed_tabular"
            normalized_feature_columns = list(request.representation.feature_columns)
            normalized_categorical_feature_columns = list(
                request.representation.categorical_feature_columns
            )
        else:
            representation_name = "molecular_graph"
            normalized_feature_columns = None
            normalized_categorical_feature_columns = None

        activity_cliff_index = request.activity_cliffs.index
        activity_cliff_feedback = request.activity_cliffs.feedback
        activity_cliff_feedback_loops = request.activity_cliffs.feedback_loops
        activity_cliff_similarity_threshold = request.activity_cliffs.similarity_threshold
        activity_cliff_top_k_neighbors = request.activity_cliffs.top_k_neighbors
        activity_cliff_flag_threshold = request.activity_cliffs.flag_threshold
        requested_ad_methods = list(request.applicability_domain.methods)
        requested_similarity_top_k = request.applicability_domain.similarity_top_k_neighbors
        requested_similarity_percentile = (
            request.applicability_domain.similarity_threshold_percentile
        )
        requested_hyperparameter_tuning = request.tuning.model_dump(exclude_none=True)
        search_space = requested_hyperparameter_tuning.get("search_space")
        if isinstance(search_space, dict):
            search_space.pop("backend", None)
            requested_hyperparameter_tuning["search_space"] = {
                name: value for name, value in search_space.items() if value is not None
            }
        requested_outlier_analysis = request.outlier_analysis.model_dump()
        requested_resolved_parameters = request.backend.model_dump(
            exclude_none=True,
            exclude_unset=True,
        )
        requested_resolved_parameters.pop("name", None)
        if normalized_backend == "lightgbm":
            device = requested_resolved_parameters.pop("device", "auto")
            if device != "auto":
                requested_resolved_parameters["device_type"] = device
        compute_profile = request.compute.profile
        if compute_profile == "auto":
            compute_profile = resolve_training_profile(describe_compute_environment())["profile"]
        requested_resolved_parameters["training_profile"] = compute_profile
        requested_resolved_parameters["allow_heavy_compute"] = request.compute.allow_heavy_compute
        requested_resolved_parameters["feature_cache_dir"] = str(
            (Path(output_dir).expanduser().resolve() / "feature_cache")
        )
        requested_resolved_parameters["feature_n_jobs"] = _resolve_feature_n_jobs()
        requested_resolved_parameters["validation_protocol"] = validation_protocol
        if (
            isinstance(request.validation, StandardQsarValidation)
            and request.validation.seed is not None
        ):
            requested_resolved_parameters["random_state"] = request.validation.seed
        bundle_path = (
            str(
                Path(bundle_dir).expanduser().resolve()
                / f"{Path(output_dir).name}_training_bundle.zip"
            )
            if bundle_dir
            else None
        )
        backend_resolved_parameters = dict(requested_resolved_parameters)
        backend_resolved_parameters.pop("feature_cache_dir", None)
        backend_resolved_parameters.pop("feature_n_jobs", None)

        if normalized_backend == "chemprop":
            if requested_validation_strategy is not None:
                requested_resolved_parameters["validation_strategy"] = requested_validation_strategy
            chemprop_kwargs = {"bundle_path": bundle_path} if bundle_path else {}
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
                applicability_domain_methods=requested_ad_methods,
                similarity_top_k_neighbors=requested_similarity_top_k,
                similarity_threshold_percentile=requested_similarity_percentile,
                resolved_parameters=backend_resolved_parameters,
                hyperparameter_tuning=requested_hyperparameter_tuning,
                outlier_analysis=requested_outlier_analysis,
                agent=agent,
                **chemprop_kwargs,
            )
            result["backend_name"] = "chemprop"
            result.setdefault("representation_name", "molecular_graph")
            result["candidate_train_csv"] = train_csv
        elif normalized_backend in {"lightgbm", "tabicl"}:
            working_train_csv = train_csv
            resolved_representation = representation_name
            if not normalized_feature_columns:
                resolved_representation = (
                    resolved_representation
                    or default_tabular_representation_for_protocol(
                        validation_protocol,
                        training_profile=requested_resolved_parameters.get("training_profile"),
                    )
                )
                prepared = self._prepare_tabular_training_dataset(
                    train_csv=train_csv,
                    output_dir=output_dir,
                    smiles_column=smiles_column,
                    target_columns=list(normalized_target_columns),
                    representation_name=resolved_representation,
                    feature_cache_dir=requested_resolved_parameters.get("feature_cache_dir"),
                    feature_n_jobs=requested_resolved_parameters.get("feature_n_jobs")
                    or requested_resolved_parameters.get("n_jobs"),
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
                lightgbm_kwargs = {"bundle_path": bundle_path} if bundle_path else {}
                result = self.lightgbm_toolkit.train_lightgbm_model(
                    train_csv=working_train_csv,
                    task_type=task_type,
                    output_dir=output_dir,
                    target_columns=list(normalized_target_columns),
                    feature_columns=normalized_feature_columns,
                    representation_name=resolved_representation,
                    categorical_feature_columns=normalized_categorical_feature_columns,
                    validation_protocol=validation_protocol,
                    validation_strategy=requested_validation_strategy,
                    activity_cliff_index=activity_cliff_index,
                    activity_cliff_feedback=activity_cliff_feedback,
                    activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                    activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                    activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                    activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                    resolved_parameters=backend_resolved_parameters,
                    hyperparameter_tuning=requested_hyperparameter_tuning,
                    outlier_analysis=requested_outlier_analysis,
                    agent=agent,
                    **lightgbm_kwargs,
                )
            else:
                result = self.tabicl_toolkit.train_tabicl_model(
                    train_csv=working_train_csv,
                    task_type=task_type,
                    output_dir=output_dir,
                    target_columns=list(normalized_target_columns),
                    feature_columns=normalized_feature_columns,
                    representation_name=resolved_representation,
                    validation_protocol=validation_protocol,
                    validation_strategy=requested_validation_strategy,
                    activity_cliff_index=activity_cliff_index,
                    activity_cliff_feedback=activity_cliff_feedback,
                    activity_cliff_feedback_loops=activity_cliff_feedback_loops,
                    activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
                    activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
                    activity_cliff_flag_threshold=activity_cliff_flag_threshold,
                    applicability_domain_methods=requested_ad_methods,
                    similarity_top_k_neighbors=requested_similarity_top_k,
                    similarity_threshold_percentile=requested_similarity_percentile,
                    resolved_parameters=backend_resolved_parameters,
                    hyperparameter_tuning=requested_hyperparameter_tuning,
                    outlier_analysis=requested_outlier_analysis,
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
                "Unsupported backend. Expected one of ['chemprop', 'lightgbm', 'tabicl']."
            )

        resolved_plan = ResolvedTrainingPlan(
            requested_contract=request,
            backend_name=normalized_backend,
            representation_name=str(result.get("representation_name") or representation_name),
            feature_columns=list(result.get("feature_columns") or normalized_feature_columns or []),
            categorical_feature_columns=list(normalized_categorical_feature_columns or []),
            validation_protocol=(
                "standard_qsar" if requested_validation_strategy is None else "custom"
            ),
            validation_strategy=requested_validation_strategy or {},
            split_runs=list(result.get("split_runs") or []),
            seed_policy=dict(result.get("seed_policy") or {}),
            compute_profile=compute_profile,
            tuning=request.tuning,
            outlier_analysis=request.outlier_analysis,
            activity_cliffs=request.activity_cliffs,
            applicability_domain=request.applicability_domain,
            effective_parameters=request.backend.model_dump(mode="json"),
            runtime_paths=RuntimePaths(
                output_dir=str(Path(output_dir).expanduser().resolve()),
                bundle_path=bundle_path,
                feature_cache_dir=requested_resolved_parameters.get("feature_cache_dir"),
            ),
        )
        result["training_contract_version"] = "2.0"
        result["requested_contract"] = request.model_dump(mode="json")
        result["resolved_plan"] = resolved_plan.model_dump(mode="json")
        result["effective_parameters"] = request.backend.model_dump(mode="json")
        result["runtime"] = {
            "compute_profile": compute_profile,
            "allow_heavy_compute": request.compute.allow_heavy_compute,
        }
        result["workflow"] = {
            "kind": "single_training",
            "validation": request.validation.model_dump(mode="json"),
        }

        if not ((result.get("curation") or {}).get("artifacts")):
            curation_artifacts = latest_curation_artifacts(agent) if agent is not None else {}
            if not (curation_artifacts.get("artifacts") if curation_artifacts else None):
                curation_artifacts = discover_curation_artifacts_near_dataset(train_csv)
            if curation_artifacts:
                result["curation"] = curation_artifacts

        # Preserve the user-visible dataset contract in the compact factual
        # report, even when a backend does not echo these inputs verbatim.
        result.setdefault("smiles_column", smiles_column)
        result.setdefault("target_columns", list(normalized_target_columns))
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
        if len(split_registry_payloads) > 1 and (
            result.get("validation_strategy_type") != "cross_validation"
            or result.get("outlier_model_variants")
        ):
            result["candidate_registry_payloads"] = split_registry_payloads
            result["recommended_registry_payloads"] = [
                item["registry_payload"] for item in split_registry_payloads
            ]
            result["persistence_plan"] = {
                "persist_all_candidates": True,
                "candidate_count": len(split_registry_payloads),
                "candidate_registry_payloads_key": "candidate_registry_payloads",
                "required_tool_sequence": (
                    "Call register_and_persist_candidates with the exact "
                    "candidate_registry_payloads list and report every returned canonical catalog model_id."
                ),
            }
        _materialize_candidate_persistence_manifest(result=result, output_dir=output_dir)
        _materialize_outlier_variants_manifest(
            result=result,
            output_dir=output_dir,
            compact_for_response=False,
        )
        # The factual report handoff must be created after variant/candidate
        # manifests have been materialized, then written into the canonical
        # summary before the facade response is compacted.
        result["reporting_handoff"] = build_training_reporting_handoff(result)
        if result.get("outlier_model_variants"):
            result["outlier_model_variants"] = [
                _compact_outlier_variant_for_response(variant)
                for variant in result["outlier_model_variants"]
                if isinstance(variant, dict)
            ]
        self._refresh_enriched_training_artifacts(
            result=result,
            output_dir=output_dir,
        )
        if normalized_backend in {"lightgbm", "tabicl"} and isinstance(
            result.get("feature_columns"), list
        ):
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
        request: ChempropTrainingRequest,
        output_dir: str,
        bundle_dir: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train Chemprop through the strict unified QSAR facade."""
        if isinstance(request, dict):
            request = ChempropTrainingRequest.model_validate(request)
        return self.train_qsar_model(
            train_csv=train_csv,
            request=QsariaTrainingRequest.model_validate(request.model_dump()),
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

    def train_lightgbm_model(
        self,
        train_csv: str,
        request: LightGBMTrainingRequest,
        output_dir: str,
        bundle_dir: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train LightGBM through the strict unified QSAR facade."""
        if isinstance(request, dict):
            request = LightGBMTrainingRequest.model_validate(request)
        return self.train_qsar_model(
            train_csv=train_csv,
            request=QsariaTrainingRequest.model_validate(request.model_dump()),
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )

    def train_tabicl_model(
        self,
        train_csv: str,
        request: TabICLTrainingRequest,
        output_dir: str,
        bundle_dir: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Train TabICL through the strict unified QSAR facade."""
        if isinstance(request, dict):
            request = TabICLTrainingRequest.model_validate(request)
        return self.train_qsar_model(
            train_csv=train_csv,
            request=QsariaTrainingRequest.model_validate(request.model_dump()),
            output_dir=output_dir,
            bundle_dir=bundle_dir,
            agent=agent,
        )
