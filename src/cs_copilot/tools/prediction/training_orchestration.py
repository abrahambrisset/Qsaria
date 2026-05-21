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
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import pandas as pd

from .ad_builder import build_applicability_domain_from_training_data
from .backend import PredictionTaskSpec
from .qsar_plots import build_qsar_training_plots
from .qsar_training_policy import describe_compute_environment, resolve_training_profile


def strip_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy without CSV index columns such as ``Unnamed: 0``."""
    return df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()


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
                raise ValueError(f"{argument_name} must be a list or a JSON-encoded list.") from None
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
        primary_run.get("model_path") or primary_run.get("best_model_path"): root_model_dir / model_filename,
        primary_run.get("test_predictions_path"): root_model_dir / "test_predictions.csv",
        primary_run.get("config_path"): root_output_dir / "config.toml",
        primary_run.get("splits_path"): root_output_dir / "splits.json",
    }
    copied: Dict[str, Optional[str]] = {
        "best_model_path": None,
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
        elif target_path.name == "test_predictions.csv":
            copied["test_predictions_path"] = str(target_path)
        elif target_path.name == "config.toml":
            copied["config_path"] = str(target_path)
        elif target_path.name == "splits.json":
            copied["splits_path"] = str(target_path)
    return copied


def build_applicability_domain_for_training(
    *,
    train_csv: str,
    primary_run: Mapping[str, Any],
    primary_output_dir: Path,
    task: PredictionTaskSpec,
    model_id_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the standard hybrid Morgan applicability domain from the train split."""
    splits_path = Path(str(primary_run.get("splits_path") or primary_output_dir / "splits.json"))
    if not splits_path.exists():
        return {}

    split_payload = json.loads(splits_path.read_text())
    if not split_payload or "train" not in split_payload[0]:
        return {}

    dataset = strip_unnamed_columns(pd.read_csv(Path(train_csv).expanduser()))
    train_indices = split_payload[0].get("train") or []
    smiles_column = task.smiles_columns[0] if task.smiles_columns else "smiles"
    ad_output_dir = primary_output_dir / "applicability_domain"
    return build_applicability_domain_from_training_data(
        dataset=dataset,
        train_indices=train_indices,
        smiles_column=smiles_column,
        output_dir=str(ad_output_dir),
        model_id=model_id_hint or primary_output_dir.name,
    )


def build_training_plots_if_possible(
    *,
    train_csv: str,
    split_results: List[Dict[str, Any]],
    primary_run: Dict[str, Any],
    root_artifacts: Mapping[str, Optional[str]],
    root_output_dir: Path,
    target_column: Optional[str],
) -> Dict[str, str]:
    """Build standard QSAR plots when the required split artifacts are present."""
    if not target_column or not root_artifacts.get("splits_path") or not root_artifacts.get("test_predictions_path"):
        return {}

    plots_output_dir = root_output_dir / "artifacts" / "plots"
    try:
        return build_qsar_training_plots(
            train_csv=train_csv,
            split_results=[
                {
                    **item,
                    "splits_path": (
                        root_artifacts["splits_path"] if item is primary_run else item.get("splits_path")
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
        )
    except Exception:
        return {}


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
    for key in ("model_path", "best_model_path", "config_path", "splits_path", "test_predictions_path"):
        if result.get(key):
            files.append(Path(str(result[key])).expanduser())
    for split_result in split_results:
        for key in ("summary_path", "model_path", "best_model_path", "test_predictions_path", "splits_path"):
            if split_result.get(key):
                files.append(Path(str(split_result[key])).expanduser())
    for key in ("reference_store_path", "reference_manifest_path", "applicability_domain_path"):
        if ad_summary.get(key):
            files.append(Path(str(ad_summary[key])).expanduser())
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
            for key in ("model_path", "best_model_path", "test_predictions_path", "splits_path"):
                if split_result.get(key):
                    files.append(Path(str(split_result[key])).expanduser())
    for artifact_path in (cliffs.get("plot_artifacts") or {}).values():
        files.append(Path(str(artifact_path)).expanduser())
    for artifact_path in (cliffs.get("loop_comparison_plot_artifacts") or {}).values():
        files.append(Path(str(artifact_path)).expanduser())
    if extra_files:
        files.extend(Path(path).expanduser() for path in extra_files)
    return files
