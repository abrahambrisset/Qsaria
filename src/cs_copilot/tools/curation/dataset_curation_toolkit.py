#!/usr/bin/env python
# coding: utf-8
"""
Toolkit dedicated to QSAR dataset curation.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from agno.agent import Agent
from agno.tools.toolkit import Toolkit
from rdkit import Chem

from cs_copilot.tools.curation.backends import (
    standardize_with_chembl_structure_v1,
    standardize_with_legacy_rdkit_v1,
)
from cs_copilot.tools.curation.policies import (
    CHEMBL_QSAR_POLICY,
    DEFAULT_CURATION_BACKEND,
    DEFAULT_DUPLICATE_CONFLICT_THRESHOLD,
    LEGACY_CURATION_BACKEND,
    LEGACY_QSAR_POLICY,
)
from cs_copilot.storage import S3
from cs_copilot.tools.prediction.training_orchestration import (
    is_classification_task,
    normalize_classification_label,
    resolve_class_labels,
)

from .backend import CurationRequest, CurationResult, TargetSummary


def _get_curation_state(agent: Agent) -> Dict[str, Any]:
    state = agent.session_state.setdefault("qsar_curation", {})
    state.setdefault("last_request", {})
    state.setdefault("last_result", {})
    state.setdefault("history", [])
    return state


def _load_dataset(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path).expanduser()
    if path.exists():
        return pd.read_csv(path)
    with S3.open(dataset_path, "r") as fh:
        return pd.read_csv(fh)


def _absolute_storage_path(path_value: Optional[str | Path]) -> Optional[str]:
    """Return a real absolute storage path for local or session-relative artifacts."""
    if path_value is None:
        return None
    path_text = str(path_value)
    if not path_text:
        return None
    expanded = Path(path_text).expanduser()
    if expanded.is_absolute():
        return str(expanded.resolve())
    return S3.path(path_text)


def _bundle_files(bundle_path: Path, files: List[Path]) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as zf:
        for file_path in files:
            if file_path.exists():
                zf.write(file_path, arcname=file_path.name)
    return bundle_path


def _count_stereo_markers_removed(original: pd.Series, standardized: pd.Series) -> int:
    """Approximate how many rows lost explicit stereochemistry markers during standardization."""
    count = 0
    for raw_smiles, std_smiles in zip(original.fillna(""), standardized.fillna(""), strict=False):
        if not isinstance(raw_smiles, str) or not isinstance(std_smiles, str):
            continue
        if ("@" in raw_smiles or "/" in raw_smiles or "\\" in raw_smiles) and (
            "@" not in std_smiles and "/" not in std_smiles and "\\" not in std_smiles
        ):
            count += 1
    return count


_UNIT_COLUMN_PATTERN = re.compile(r"(?:^|_)(unit|units|uom|measurement_unit|conc_unit)(?:$|_)", re.I)
_ASSAY_CONTEXT_PATTERNS = {
    "assay_columns": re.compile(r"(assay|protocol|experiment|screen)", re.I),
    "ph_columns": re.compile(r"(^|_)(ph|p_h)(_|$)", re.I),
    "temperature_columns": re.compile(r"(temp|temperature)", re.I),
    "replicate_columns": re.compile(r"(replicate|replicates|repeat|n_repl|nrep)", re.I),
    "fit_quality_columns": re.compile(r"(fit_quality|fitscore|fit_score|curve_quality|r2_fit|hill_fit|dose_response_quality)", re.I),
    "cytotoxicity_columns": re.compile(r"(cytotox|cytotoxicity|cell_viability|viability)", re.I),
    "interference_columns": re.compile(r"(interference|artifact|fluorescence|quench|signal)", re.I),
    "time_columns": re.compile(r"(time|incubation)", re.I),
}


def _normalize_unit_value(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.lower().replace(" ", "")


def _infer_target_unit_from_context(
    *,
    dataset_path: str,
    dataset_id: Optional[str],
    target_columns: List[str],
) -> Optional[str]:
    context = " ".join(
        [
            Path(dataset_path).stem.lower(),
            (dataset_id or "").lower(),
            " ".join(column.lower() for column in target_columns),
        ]
    )
    if any(token in context for token in ("lipophilicity", "logp", "logd")):
        return "unitless_log_scale"
    return None


def _detect_target_unit_quality(
    df: pd.DataFrame,
    *,
    dataset_path: str,
    dataset_id: Optional[str],
    target_columns: List[str],
) -> Dict[str, Any]:
    unit_columns = [column for column in df.columns if _UNIT_COLUMN_PATTERN.search(column)]
    unit_column_value_map: Dict[str, List[str]] = {}
    distinct_units: List[str] = []

    for column in unit_columns:
        values = [
            normalized
            for normalized in (_normalize_unit_value(v) for v in df[column].tolist())
            if normalized is not None
        ]
        unique_values = sorted(set(values))
        unit_column_value_map[column] = unique_values
        distinct_units.extend(unique_values)

    distinct_units = sorted(set(distinct_units))
    inferred_unit = _infer_target_unit_from_context(
        dataset_path=dataset_path,
        dataset_id=dataset_id,
        target_columns=target_columns,
    )

    if distinct_units:
        return {
            "unit_columns_detected": unit_columns,
            "unit_values_detected": distinct_units,
            "target_unit_detected": distinct_units[0] if len(distinct_units) == 1 else None,
            "target_unit_source": "dataset_unit_column",
            "target_unit_homogeneous": len(distinct_units) <= 1,
            "unit_conflicts_detected": max(len(distinct_units) - 1, 0),
            "unit_column_value_map": unit_column_value_map,
        }

    return {
        "unit_columns_detected": [],
        "unit_values_detected": [inferred_unit] if inferred_unit else [],
        "target_unit_detected": inferred_unit,
        "target_unit_source": "context_inference" if inferred_unit else None,
        "target_unit_homogeneous": True if inferred_unit else None,
        "unit_conflicts_detected": 0,
        "unit_column_value_map": {},
    }


def _detect_target_outliers(
    df: pd.DataFrame,
    *,
    target_columns: List[str],
) -> Dict[str, Any]:
    outlier_counts: Dict[str, int] = {}
    outlier_bounds: Dict[str, Dict[str, float]] = {}
    total_flagged = 0

    for column in target_columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 4:
            outlier_counts[column] = 0
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            outlier_counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = int(((series < lower) | (series > upper)).sum())
        outlier_counts[column] = flagged
        total_flagged += flagged
        outlier_bounds[column] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
        }

    return {
        "outlier_method": "iqr_1.5",
        "outliers_flagged_total": total_flagged,
        "outliers_flagged_by_target": outlier_counts,
        "outlier_bounds_by_target": outlier_bounds,
        "outlier_policy": "flag_only",
    }


def _detect_measurement_context(df: pd.DataFrame) -> Dict[str, Any]:
    detected: Dict[str, List[str]] = {}
    for key, pattern in _ASSAY_CONTEXT_PATTERNS.items():
        columns = [column for column in df.columns if pattern.search(column)]
        if columns:
            detected[key] = columns

    all_detected_columns = sorted({column for cols in detected.values() for column in cols})
    return {
        "measurement_quality_available": bool(
            detected.get("fit_quality_columns")
            or detected.get("replicate_columns")
            or detected.get("cytotoxicity_columns")
            or detected.get("interference_columns")
        ),
        "experimental_context_columns_detected": all_detected_columns,
        "measurement_quality_columns": sorted(
            {
                *detected.get("fit_quality_columns", []),
                *detected.get("replicate_columns", []),
                *detected.get("cytotoxicity_columns", []),
                *detected.get("interference_columns", []),
            }
        ),
        "context_column_groups": detected,
    }


_METAL_ATOMIC_NUMBERS = {
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56,
    57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
    74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
}


def _structure_flags(smiles: str) -> Dict[str, bool]:
    """Classify raw structures for early structural curation decisions."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "valid": False,
            "is_inorganic": False,
            "is_organometallic": False,
            "is_mixture": False,
            "has_salt_or_counterion_pattern": False,
        }

    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    organic_fragment_count = 0
    has_metal = False
    has_carbon = False

    for frag in fragments:
        frag_has_carbon = any(atom.GetAtomicNum() == 6 for atom in frag.GetAtoms())
        if frag_has_carbon:
            organic_fragment_count += 1
        has_carbon = has_carbon or frag_has_carbon
        if any(atom.GetAtomicNum() in _METAL_ATOMIC_NUMBERS for atom in frag.GetAtoms()):
            has_metal = True

    is_organometallic = has_metal and has_carbon
    is_inorganic = not has_carbon
    is_mixture = organic_fragment_count > 1
    has_salt_or_counterion_pattern = len(fragments) > 1 and organic_fragment_count <= 1

    return {
        "valid": True,
        "is_inorganic": is_inorganic,
        "is_organometallic": is_organometallic,
        "is_mixture": is_mixture,
        "has_salt_or_counterion_pattern": has_salt_or_counterion_pattern,
    }


def _resolve_regression_duplicates(
    df: pd.DataFrame,
    target_columns: List[str],
    conflict_threshold: float,
    identity_column: str = "smiles",
) -> Dict[str, Any]:
    """Resolve duplicate standardized structures for regression datasets."""
    grouped_rows: List[Dict[str, Any]] = []
    duplicate_group_records: List[Dict[str, Any]] = []
    duplicate_groups_detected = 0
    duplicate_groups_aggregated = 0
    duplicate_conflicting_groups = 0
    duplicate_conflicting_rows_removed = 0

    for identity_key, group in df.groupby(identity_column, dropna=False, sort=False):
        if len(group) == 1:
            grouped_rows.append(group.iloc[0].to_dict())
            continue

        duplicate_groups_detected += 1
        is_conflicting = False
        target_spreads: Dict[str, float] = {}
        target_values: Dict[str, List[float]] = {}
        for target in target_columns:
            series = pd.to_numeric(group[target], errors="coerce").dropna()
            target_values[target] = [float(value) for value in series.tolist()]
            if len(series) <= 1:
                target_spreads[target] = 0.0
                continue
            spread = float(series.max() - series.min())
            target_spreads[target] = spread
            if spread > conflict_threshold:
                is_conflicting = True
                break

        group_record = {
            "identity_key": identity_key,
            "identity_column": identity_column,
            "group_size": int(len(group)),
            "resolution": "removed_conflict" if is_conflicting else "aggregated_mean",
            "target_spreads": json.dumps(target_spreads, sort_keys=True),
            "target_values": json.dumps(target_values, sort_keys=True),
            "row_indices": ",".join(str(idx) for idx in group.index.tolist()),
            "raw_smiles": " | ".join(str(v) for v in group.get("raw_smiles", pd.Series()).tolist()),
            "standardized_smiles": " | ".join(
                str(v) for v in group.get("standardized_smiles", group["smiles"]).tolist()
            ),
        }
        duplicate_group_records.append(group_record)

        if is_conflicting:
            duplicate_conflicting_groups += 1
            duplicate_conflicting_rows_removed += int(len(group))
            continue

        aggregated = group.iloc[0].to_dict()
        for target in target_columns:
            aggregated[target] = float(pd.to_numeric(group[target], errors="coerce").mean())
        grouped_rows.append(aggregated)
        duplicate_groups_aggregated += 1

    resolved_df = pd.DataFrame(grouped_rows, columns=df.columns)
    return {
        "dataframe": resolved_df,
        "duplicate_groups_detected": duplicate_groups_detected,
        "duplicate_groups_aggregated": duplicate_groups_aggregated,
        "duplicate_conflicting_groups": duplicate_conflicting_groups,
        "duplicate_conflicting_rows_removed": duplicate_conflicting_rows_removed,
        "duplicate_group_records": duplicate_group_records,
    }


def _resolve_classification_duplicates(
    df: pd.DataFrame,
    target_columns: List[str],
    identity_column: str = "smiles",
) -> Dict[str, Any]:
    """Resolve duplicate standardized structures for classification datasets."""
    grouped_rows: List[Dict[str, Any]] = []
    duplicate_group_records: List[Dict[str, Any]] = []
    duplicate_groups_detected = 0
    duplicate_groups_aggregated = 0
    duplicate_conflicting_groups = 0
    duplicate_conflicting_rows_removed = 0

    for identity_key, group in df.groupby(identity_column, dropna=False, sort=False):
        if len(group) == 1:
            grouped_rows.append(group.iloc[0].to_dict())
            continue

        duplicate_groups_detected += 1
        target_values: Dict[str, List[Any]] = {}
        is_conflicting = False
        for target in target_columns:
            labels = [normalize_classification_label(value) for value in group[target].tolist()]
            labels = [label for label in labels if label is not None]
            unique = resolve_class_labels(pd.Series(labels))
            target_values[target] = [str(value) for value in labels]
            if len(unique) > 1:
                is_conflicting = True

        duplicate_group_records.append(
            {
                "identity_key": identity_key,
                "identity_column": identity_column,
                "group_size": int(len(group)),
                "resolution": "removed_conflict" if is_conflicting else "aggregated_label",
                "target_spreads": "{}",
                "target_values": json.dumps(target_values, sort_keys=True),
                "row_indices": ",".join(str(idx) for idx in group.index.tolist()),
                "raw_smiles": " | ".join(str(v) for v in group.get("raw_smiles", pd.Series()).tolist()),
                "standardized_smiles": " | ".join(
                    str(v) for v in group.get("standardized_smiles", group["smiles"]).tolist()
                ),
            }
        )
        if is_conflicting:
            duplicate_conflicting_groups += 1
            duplicate_conflicting_rows_removed += int(len(group))
            continue

        grouped_rows.append(group.iloc[0].to_dict())
        duplicate_groups_aggregated += 1

    return {
        "dataframe": pd.DataFrame(grouped_rows, columns=df.columns),
        "duplicate_groups_detected": duplicate_groups_detected,
        "duplicate_groups_aggregated": duplicate_groups_aggregated,
        "duplicate_conflicting_groups": duplicate_conflicting_groups,
        "duplicate_conflicting_rows_removed": duplicate_conflicting_rows_removed,
        "duplicate_group_records": duplicate_group_records,
    }


def _select_curation_backend(curation_backend: str):
    if curation_backend == DEFAULT_CURATION_BACKEND:
        return standardize_with_chembl_structure_v1
    if curation_backend == LEGACY_CURATION_BACKEND:
        return standardize_with_legacy_rdkit_v1
    available = [DEFAULT_CURATION_BACKEND, LEGACY_CURATION_BACKEND]
    raise ValueError(
        f"Unknown curation_backend: {curation_backend}. Available backends: {available}"
    )


class DatasetCurationToolkit(Toolkit):
    """Tools for preparing QSAR-ready datasets before training."""

    def __init__(self):
        super().__init__("qsar_dataset_curation")
        self.register(self.inspect_dataset_schema)
        self.register(self.identify_qsar_columns)
        self.register(self.curate_qsar_dataset)
        self.register(self.summarize_curated_dataset)
        self.register(self.write_curation_report)

    def inspect_dataset_schema(self, dataset_path: str) -> Dict[str, Any]:
        """Inspect columns, size, and a small preview for a dataset."""
        try:
            df = _load_dataset(dataset_path)
        except FileNotFoundError as exc:
            requested_name = Path(str(dataset_path)).name.lower()
            target_like_names = {
                "pec50.csv",
                "pic50.csv",
                "ec50.csv",
                "ic50.csv",
                "pki.csv",
                "ki.csv",
            }
            if requested_name in target_like_names:
                raise FileNotFoundError(
                    f"Dataset path `{dataset_path}` does not exist. `{Path(str(dataset_path)).stem}` looks like a "
                    "target name, not an uploaded dataset file. For a request such as `cree un ensemble QSAR sur "
                    "pEC50`, do not inspect a dataset schema; route to the model-registry ensemble workflow "
                    "(`inspect_ensemble_candidates` then `create_ensemble_from_catalog`)."
                ) from exc
            raise
        return {
            "dataset_path": dataset_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "preview": df.head(5).to_dict(orient="records"),
        }

    def identify_qsar_columns(
        self,
        dataset_path: str,
        task_type: str,
        preferred_smiles_column: Optional[str] = None,
        preferred_target_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Identify the likely SMILES and target columns for QSAR work."""
        df = _load_dataset(dataset_path)
        columns = list(df.columns)

        smiles_column = None
        if preferred_smiles_column and preferred_smiles_column in columns:
            smiles_column = preferred_smiles_column
        else:
            for candidate in ("smiles", "SMILES", "Drug", "smi", "structure"):
                if candidate in columns:
                    smiles_column = candidate
                    break

        target_columns: List[str] = []
        preferred_target_columns = preferred_target_columns or []
        for column in preferred_target_columns:
            if column in columns:
                target_columns.append(column)

        if not target_columns:
            excluded = {smiles_column} if smiles_column else set()
            excluded.update({"Drug_ID", "compound_id", "id", "ID"})
            numeric_candidates = [
                column
                for column in columns
                if column not in excluded and pd.api.types.is_numeric_dtype(df[column])
            ]
            if task_type == "regression":
                target_columns = numeric_candidates[:1]
            elif is_classification_task(task_type):
                max_reasonable_classes = max(20, int(len(df) * 0.2))
                classification_candidates = []
                for column in columns:
                    if column in excluded:
                        continue
                    labels = df[column].map(normalize_classification_label).dropna()
                    class_count = int(labels.nunique(dropna=True))
                    if 2 <= class_count <= max_reasonable_classes:
                        classification_candidates.append(column)
                target_columns = classification_candidates[:1]
            else:
                target_columns = numeric_candidates[:1]

        return {
            "dataset_path": dataset_path,
            "task_type": task_type,
            "smiles_column": smiles_column,
            "target_columns": target_columns,
            "all_columns": columns,
            "ready": bool(smiles_column and target_columns),
        }

    def curate_qsar_dataset(
        self,
        dataset_path: str,
        task_type: str,
        smiles_column: str,
        target_columns: List[str],
        output_csv: Optional[str] = None,
        dataset_id: Optional[str] = None,
        duplicate_conflict_threshold: float = DEFAULT_DUPLICATE_CONFLICT_THRESHOLD,
        curation_backend: str = DEFAULT_CURATION_BACKEND,
        report_path: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Produce a QSAR-ready curated dataset and structured curation result."""
        request = CurationRequest(
            dataset_path=dataset_path,
            task_type=task_type,
            dataset_id=dataset_id,
            preferred_smiles_column=smiles_column,
            preferred_target_columns=target_columns,
            duplicate_conflict_threshold=duplicate_conflict_threshold,
            curation_backend=curation_backend,
        )
        df = _load_dataset(dataset_path)
        rows_in = int(len(df))
        working = df.copy()

        warnings: List[str] = []
        blocking_issues: List[str] = []
        actions: List[str] = []
        backend_policy = (
            CHEMBL_QSAR_POLICY
            if curation_backend == DEFAULT_CURATION_BACKEND
            else LEGACY_QSAR_POLICY
        )
        classification_task = is_classification_task(task_type)
        curation_policy = {
            **backend_policy,
            "duplicate_policy": "aggregate_mean_if_spread_within_threshold_else_drop_conflicts",
            "duplicate_conflict_threshold": duplicate_conflict_threshold,
            "target_policy": (
                "preserve_class_labels -> remove_missing_labels -> require_at_least_two_classes"
                if classification_task
                else "coerce_numeric -> remove_non_numeric -> remove_infinite -> remove_missing -> flag_constant_targets"
            ),
            "unit_policy": "detect explicit unit columns -> block on unresolved heterogeneous units -> infer unit context only when no explicit unit column exists",
            "outlier_policy": "not_applicable_for_classification" if classification_task else "detect_iqr_1.5 -> flag_only",
            "measurement_context_policy": "detect assay/context/fit-quality columns -> report availability without filtering rows",
        }

        if smiles_column not in working.columns:
            blocking_issues.append(f"SMILES column not found: {smiles_column}")
        missing_targets = [column for column in target_columns if column not in working.columns]
        if missing_targets:
            blocking_issues.append(f"Target columns not found: {missing_targets}")

        if blocking_issues:
            result = CurationResult(
                status="blocked",
                ready_for_qsar=False,
                dataset_id=dataset_id,
                source_dataset_path=dataset_path,
                curated_dataset_path=None,
                smiles_column_original=smiles_column,
                smiles_column_curated=None,
                target_columns_original=list(target_columns),
                target_columns_curated=[],
                task_type=task_type,
                rows_in=rows_in,
                rows_out=0,
                blocking_issues=blocking_issues,
            )
            if agent is not None:
                state = _get_curation_state(agent)
                state["last_request"] = request.as_dict()
                state["last_result"] = result.as_dict()
                state["history"].append(result.as_dict())
            return result.as_dict()

        if smiles_column != "smiles":
            working = working.rename(columns={smiles_column: "smiles"})
            actions.append(f"rename {smiles_column} -> smiles")
        else:
            actions.append("use existing smiles column")

        raw_smiles = working["smiles"].copy()
        flags = raw_smiles.apply(
            lambda s: _structure_flags(s) if isinstance(s, str) else _structure_flags("")
        )
        flags_df = pd.DataFrame(list(flags))

        inorganic_rows_removed = int(flags_df["is_inorganic"].sum())
        organometallic_rows_removed = int(flags_df["is_organometallic"].sum())
        mixture_rows_removed = int(flags_df["is_mixture"].sum())
        salt_or_counterion_rows_processed = int(flags_df["has_salt_or_counterion_pattern"].sum())

        structural_remove_mask = (
            flags_df["is_inorganic"] | flags_df["is_organometallic"] | flags_df["is_mixture"]
        )
        structural_removed_records: List[Dict[str, Any]] = []
        if structural_remove_mask.any():
            removed_structures = working.loc[structural_remove_mask].copy()
            for row_index, row in removed_structures.iterrows():
                flag_row = flags_df.loc[row_index]
                reasons = [
                    reason
                    for reason, active in {
                        "inorganic": flag_row["is_inorganic"],
                        "organometallic": flag_row["is_organometallic"],
                        "mixture_multiple_organic_fragments": flag_row["is_mixture"],
                    }.items()
                    if active
                ]
                structural_removed_records.append(
                    {
                        "row_index": row_index,
                        "raw_smiles": row.get("smiles"),
                        "standardized_smiles": None,
                        "curation_identity_key": None,
                        "removal_reason": ",".join(reasons),
                    }
                )
            working = working.loc[~structural_remove_mask].copy()
            actions.append("remove inorganic structures")
            actions.append("remove organometallic structures")
            actions.append("remove mixtures with multiple organic fragments")
        else:
            actions.append("check inorganic / organometallic / mixture structures")

        original_smiles = working["smiles"].copy()
        backend_standardizer = _select_curation_backend(curation_backend)
        backend_result = backend_standardizer(original_smiles)
        curation_policy["curation_backend_requested"] = backend_result.get("backend_name")
        curation_policy["curation_backend_used"] = backend_result.get("used_backend_name")
        curation_policy["curation_backend_fallback_used"] = bool(
            backend_result.get("fallback_used")
        )
        if backend_result.get("fallback_reason"):
            curation_policy["curation_backend_fallback_reason"] = backend_result[
                "fallback_reason"
            ]
        standardization_map = backend_result["standardization_map"].copy()
        standardization_map = standardization_map.set_index("row_index", drop=False)
        working["raw_smiles"] = standardization_map.reindex(working.index)["raw_smiles"].values
        if "chembl_input_smiles" in standardization_map.columns:
            working["chembl_input_smiles"] = standardization_map.reindex(working.index)[
                "chembl_input_smiles"
            ].values
        working["standardized_smiles"] = standardization_map.reindex(working.index)[
            "standardized_smiles"
        ].values
        working["qsar_identity_smiles"] = standardization_map.reindex(working.index)[
            "qsar_identity_smiles"
        ].values
        working["curation_identity_key"] = standardization_map.reindex(working.index)[
            "curation_identity_key"
        ].values
        working["curation_identity_key_type"] = standardization_map.reindex(working.index)[
            "curation_identity_key_type"
        ].values
        working["curation_backend_status"] = standardization_map.reindex(working.index)[
            "curation_backend_status"
        ].values
        working["checker_issues"] = standardization_map.reindex(working.index)["checker_issues"].values
        working["checker_max_penalty"] = standardization_map.reindex(working.index)[
            "checker_max_penalty"
        ].values
        working["parent_structure_changed"] = standardization_map.reindex(working.index)[
            "parent_structure_changed"
        ].values
        working["stereochemistry_removed_for_identity"] = standardization_map.reindex(working.index)[
            "stereochemistry_removed_for_identity"
        ].values
        working["smiles"] = working["standardized_smiles"]
        checker_warning_mask = pd.to_numeric(
            working["checker_max_penalty"], errors="coerce"
        ).fillna(0) > 0
        invalid_or_rejected_mask = working["smiles"].isna()
        invalid_before_drop = int(invalid_or_rejected_mask.sum())
        stereochemistry_markers_removed = _count_stereo_markers_removed(
            original_smiles, working["smiles"]
        )
        if invalid_before_drop:
            actions.append("standardize smiles")
            actions.append("remove invalid smiles")
        else:
            actions.append("standardize smiles")
        invalid_removed_records = [
            {
                "row_index": row_index,
                "raw_smiles": row.get("raw_smiles"),
                "standardized_smiles": row.get("standardized_smiles"),
                "curation_identity_key": row.get("curation_identity_key"),
                "removal_reason": (
                    "chembl_checker_penalty_ge_6"
                    if pd.to_numeric(row.get("checker_max_penalty"), errors="coerce") >= 6
                    else row.get("curation_backend_status") or "invalid_smiles"
                ),
            }
            for row_index, row in working.loc[invalid_or_rejected_mask].iterrows()
        ]
        working = working.loc[~invalid_or_rejected_mask].copy()

        missing_target_removed = 0
        non_numeric_target_removed = 0
        infinite_target_removed = 0
        curated_targets: List[str] = []
        classification_target_summary: Dict[str, Any] = {}
        if classification_task:
            for column in target_columns:
                normalized = working[column].map(normalize_classification_label)
                missing_target_removed += int(normalized.isna().sum())
                working[column] = normalized
                curated_targets.append(column)
            if not curated_targets:
                blocking_issues.append("No classification target column was retained after curation.")
            working = working.dropna(subset=curated_targets).copy()
            for column in curated_targets:
                labels = resolve_class_labels(working[column])
                counts = working[column].value_counts(dropna=False).to_dict()
                classification_target_summary[column] = {
                    "class_count": len(labels),
                    "classes": [str(value) for value in labels],
                    "class_counts": {str(key): int(value) for key, value in counts.items()},
                }
                if len(labels) < 2:
                    blocking_issues.append(
                        f"Classification target `{column}` has fewer than two classes after curation."
                    )
            actions.append("preserve classification target labels")
            actions.append("remove missing classification target rows")
        else:
            for column in target_columns:
                numeric = pd.to_numeric(working[column], errors="coerce")
                raw_missing = int(working[column].isna().sum())
                coerced_missing = int(numeric.isna().sum())
                non_numeric_target_removed += max(coerced_missing - raw_missing, 0)
                infinite_mask = numeric.apply(lambda v: isinstance(v, (int, float)) and not math.isfinite(v))
                infinite_target_removed += int(infinite_mask.sum())
                numeric = numeric.mask(infinite_mask, other=pd.NA)
                missing_target_removed += int(numeric.isna().sum())
                working[column] = numeric
                curated_targets.append(column)

            if not curated_targets:
                blocking_issues.append("No regression target column was retained after curation.")
            working = working.dropna(subset=curated_targets).copy()
            actions.append("coerce target columns to numeric")
            actions.append("remove missing target rows")

        duplicate_identity_column = backend_result.get("identity_column") or "curation_identity_key"
        duplicate_rows_removed = int(working.duplicated(subset=[duplicate_identity_column]).sum())
        duplicate_groups_detected = 0
        duplicate_groups_aggregated = 0
        duplicate_conflicting_groups = 0
        duplicate_conflicting_rows_removed = 0
        duplicate_group_records: List[Dict[str, Any]] = []
        if duplicate_rows_removed:
            duplicate_resolution = None
            if task_type == "regression" and curated_targets:
                duplicate_resolution = _resolve_regression_duplicates(
                    working,
                    curated_targets,
                    conflict_threshold=duplicate_conflict_threshold,
                    identity_column=duplicate_identity_column,
                )
            elif classification_task and curated_targets:
                duplicate_resolution = _resolve_classification_duplicates(
                    working,
                    curated_targets,
                    identity_column=duplicate_identity_column,
                )
            if duplicate_resolution is not None:
                working = duplicate_resolution["dataframe"]
                duplicate_groups_detected = duplicate_resolution["duplicate_groups_detected"]
                duplicate_groups_aggregated = duplicate_resolution["duplicate_groups_aggregated"]
                duplicate_conflicting_groups = duplicate_resolution["duplicate_conflicting_groups"]
                duplicate_conflicting_rows_removed = duplicate_resolution[
                    "duplicate_conflicting_rows_removed"
                ]
                duplicate_group_records = duplicate_resolution["duplicate_group_records"]
                actions.append(
                    "resolve duplicate QSAR identities using classification labels"
                    if classification_task
                    else "resolve duplicate QSAR identities using conflict-threshold aggregation"
                )
            else:
                duplicate_group_records = [
                    {
                        "identity_key": key,
                        "identity_column": duplicate_identity_column,
                        "group_size": int(len(group)),
                        "resolution": "kept_first",
                        "target_spreads": "{}",
                        "target_values": json.dumps(
                            {
                                target: [
                                    float(value)
                                    for value in pd.to_numeric(
                                        group[target], errors="coerce"
                                    )
                                    .dropna()
                                    .tolist()
                                ]
                                for target in curated_targets
                            },
                            sort_keys=True,
                        ),
                        "row_indices": ",".join(str(idx) for idx in group.index.tolist()),
                        "raw_smiles": " | ".join(str(v) for v in group["raw_smiles"].tolist()),
                        "standardized_smiles": " | ".join(
                            str(v) for v in group["standardized_smiles"].tolist()
                        ),
                    }
                    for key, group in working.groupby(duplicate_identity_column, dropna=False)
                    if len(group) > 1
                ]
                working = working.drop_duplicates(subset=[duplicate_identity_column], keep="first").copy()
                actions.append("remove duplicate QSAR identities")
        else:
            actions.append("check duplicate QSAR identities")

        rows_out = int(len(working))
        if rows_out == 0:
            blocking_issues.append("Dataset is empty after curation.")

        target_summaries: List[TargetSummary] = []
        constant_target_columns: List[str] = []
        for column in curated_targets:
            if classification_task:
                labels = resolve_class_labels(working[column])
                if len(labels) <= 1:
                    constant_target_columns.append(column)
                continue
            else:
                series = pd.to_numeric(working[column], errors="coerce").dropna()
                if series.empty:
                    warnings.append(f"Target column `{column}` is empty after curation.")
                    continue
                if series.nunique(dropna=True) <= 1:
                    constant_target_columns.append(column)
                target_summaries.append(
                    TargetSummary(
                        column=column,
                        mean=float(series.mean()),
                        std=float(series.std()) if len(series) > 1 else 0.0,
                        minimum=float(series.min()),
                        maximum=float(series.max()),
                        median=float(series.median()),
                    )
                )

        if task_type == "regression" and len(constant_target_columns) == len(curated_targets):
            blocking_issues.append("All retained target columns are constant after curation.")
        if classification_task and len(constant_target_columns) == len(curated_targets):
            blocking_issues.append("All retained classification targets have fewer than two classes after curation.")

        unit_quality = _detect_target_unit_quality(
            df=df,
            dataset_path=dataset_path,
            dataset_id=dataset_id or Path(dataset_path).stem,
            target_columns=curated_targets or target_columns,
        )
        outlier_quality = _detect_target_outliers(
            working,
            target_columns=curated_targets,
        ) if not classification_task else {
            "outlier_method": None,
            "outliers_flagged_total": 0,
            "outliers_flagged_by_target": {},
            "outlier_bounds_by_target": {},
            "outlier_policy": "not_applicable_for_classification",
        }
        measurement_context = _detect_measurement_context(df)

        target_data_quality = {
            "numeric_targets_required": task_type == "regression",
            "classification_targets_allowed": classification_task,
            "classification_target_summary": classification_target_summary,
            "non_numeric_target_removed": non_numeric_target_removed,
            "infinite_target_removed": infinite_target_removed,
            "missing_target_removed": missing_target_removed,
            "constant_target_columns": list(constant_target_columns),
            "target_ready_for_qsar": not bool(
                blocking_issues
                or (
                    (task_type == "regression" or classification_task)
                    and len(constant_target_columns) == len(curated_targets)
                )
            ),
        }
        target_data_quality.update(unit_quality)
        target_data_quality.update(outlier_quality)
        target_data_quality.update(measurement_context)

        if unit_quality.get("unit_conflicts_detected", 0):
            blocking_issues.append(
                "Conflicting target units detected in the dataset; unit harmonization is required before QSAR training."
            )
        elif unit_quality.get("target_unit_detected"):
            actions.append(
                f"detect target unit context: {unit_quality['target_unit_detected']}"
            )
        if measurement_context.get("experimental_context_columns_detected"):
            actions.append("detect experimental context columns")

        if invalid_before_drop == 0:
            warnings.append("No invalid SMILES were removed during curation.")
        if backend_result.get("fallback_used"):
            warnings.append(
                "Requested ChEMBL curation backend fell back to legacy RDKit backend: "
                + str(backend_result.get("fallback_reason"))
            )
        parent_changed_count = int(standardization_map["parent_structure_changed"].fillna(False).sum())
        if parent_changed_count:
            warnings.append(
                f"{parent_changed_count} structures changed during parent/salt standardization."
            )
        checker_warning_count = int(checker_warning_mask.sum())
        if checker_warning_count:
            max_checker_penalty = int(
                pd.to_numeric(working["checker_max_penalty"], errors="coerce")
                .fillna(0)
                .max()
            )
            warnings.append(
                f"ChEMBL checker flagged {checker_warning_count} standardized parent structures; max penalty={max_checker_penalty}. Rows were retained because checker policy is warn_only."
            )
        row_fallback_count = int(
            (
                standardization_map["curation_backend_status"]
                == "chembl_row_fallback_legacy_rdkit"
            ).sum()
        )
        if row_fallback_count:
            warnings.append(
                f"{row_fallback_count} rows used legacy RDKit fallback after ChEMBL row-level standardization errors."
            )
        stereo_identity_removed_count = int(
            standardization_map["stereochemistry_removed_for_identity"].fillna(False).sum()
        )
        if stereo_identity_removed_count:
            warnings.append(
                f"Stereochemistry was stripped for QSAR identity on {stereo_identity_removed_count} rows."
            )
        if inorganic_rows_removed:
            warnings.append(f"{inorganic_rows_removed} inorganic structures were removed.")
        if organometallic_rows_removed:
            warnings.append(f"{organometallic_rows_removed} organometallic structures were removed.")
        if mixture_rows_removed:
            warnings.append(f"{mixture_rows_removed} multi-component organic mixtures were removed.")
        if salt_or_counterion_rows_processed:
            warnings.append(
                f"{salt_or_counterion_rows_processed} rows were processed as salt/counterion cases via fragment-parent standardization."
            )
        if duplicate_rows_removed == 0:
            warnings.append("No duplicate QSAR identities were removed.")
        elif duplicate_conflicting_groups:
            warnings.append(
                f"{duplicate_conflicting_groups} duplicate QSAR identity groups exceeded the conflict threshold and were removed."
            )
        if duplicate_groups_aggregated:
            warnings.append(
                f"{duplicate_groups_aggregated} duplicate QSAR identity groups were aggregated by mean target."
            )
        if stereochemistry_markers_removed:
            warnings.append(
                f"Standardization removed explicit stereochemistry markers for {stereochemistry_markers_removed} rows."
            )
        if non_numeric_target_removed:
            warnings.append(
                f"{non_numeric_target_removed} target values were non-numeric and removed during coercion."
            )
        if infinite_target_removed:
            warnings.append(
                f"{infinite_target_removed} target values were infinite and removed during curation."
            )
        if constant_target_columns:
            warnings.append(
                f"Constant target columns detected after curation: {', '.join(constant_target_columns)}."
            )
        if unit_quality.get("unit_conflicts_detected", 0):
            warnings.append(
                "Multiple target units were detected; the dataset was blocked pending unit harmonization."
            )
        elif unit_quality.get("target_unit_detected"):
            unit_source = unit_quality.get("target_unit_source") or "unknown_source"
            warnings.append(
                f"Target unit context detected as `{unit_quality['target_unit_detected']}` via {unit_source}."
            )
        else:
            warnings.append("No explicit target unit column was detected.")
        if outlier_quality.get("outliers_flagged_total", 0):
            warnings.append(
                f"{outlier_quality['outliers_flagged_total']} potential target outliers were flagged using the IQR rule."
            )
        if measurement_context.get("experimental_context_columns_detected"):
            warnings.append(
                "Experimental context columns detected: "
                + ", ".join(measurement_context["experimental_context_columns_detected"])
                + "."
            )
        else:
            warnings.append(
                "No explicit experimental context or measurement-quality columns were detected."
            )

        target_data_quality["target_ready_for_qsar"] = not bool(
            blocking_issues
            or (task_type == "regression" and len(constant_target_columns) == len(curated_targets))
        )

        if output_csv:
            curated_path = Path(_absolute_storage_path(output_csv)).expanduser()
        else:
            stem = dataset_id or Path(dataset_path).stem or "qsar_dataset"
            curated_path = (Path(".files") / "qsar_curation" / f"{stem}_curated.csv").resolve()
        curated_path.parent.mkdir(parents=True, exist_ok=True)
        working.to_csv(curated_path, index=False)

        artifact_dir = curated_path.parent / f"{curated_path.stem}_curation_artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        standardization_map_path = artifact_dir / "curation_standardization_map.csv"
        duplicate_groups_path = artifact_dir / "curation_duplicate_identity_groups.csv"
        removed_rows_path = artifact_dir / "curation_removed_rows.csv"
        identity_diagnostics_path = artifact_dir / "curation_identity_diagnostics.json"
        manifest_path = artifact_dir / "curation_manifest.json"

        standardization_map.to_csv(standardization_map_path, index=False)
        pd.DataFrame(duplicate_group_records).to_csv(duplicate_groups_path, index=False)
        duplicate_conflict_removed_records = [
            {
                "row_index": record["row_indices"],
                "row_indices": record["row_indices"],
                "group_size": record["group_size"],
                "removed_row_count": record["group_size"],
                "raw_smiles": record["raw_smiles"],
                "standardized_smiles": record["standardized_smiles"],
                "curation_identity_key": record["identity_key"],
                "target_spreads": record.get("target_spreads", "{}"),
                "target_values": record.get("target_values", "{}"),
                "removal_reason": "duplicate_identity_conflict",
            }
            for record in duplicate_group_records
            if record.get("resolution") == "removed_conflict"
        ]
        removed_rows = pd.DataFrame(
            [
                *structural_removed_records,
                *invalid_removed_records,
                *duplicate_conflict_removed_records,
            ]
        )
        removed_reason_counts = (
            removed_rows["removal_reason"].value_counts(dropna=False).to_dict()
            if "removal_reason" in removed_rows.columns
            else {}
        )
        removed_rows.to_csv(removed_rows_path, index=False)
        identity_diagnostics = {
            "curation_backend_requested": backend_result.get("backend_name"),
            "curation_backend_used": backend_result.get("used_backend_name"),
            "curation_backend_fallback_used": bool(backend_result.get("fallback_used")),
            "curation_backend_fallback_reason": backend_result.get("fallback_reason"),
            "duplicate_identity_column": duplicate_identity_column,
            "duplicate_conflict_threshold": duplicate_conflict_threshold,
            "duplicate_groups_detected": duplicate_groups_detected,
            "duplicate_groups_aggregated": duplicate_groups_aggregated,
            "duplicate_conflicting_groups": duplicate_conflicting_groups,
            "checker_policy": curation_policy.get("checker_policy"),
            "checker_flagged_rows": checker_warning_count,
            "checker_rejected_rows": 0,
            "backend_status_counts": standardization_map[
                "curation_backend_status"
            ].value_counts(dropna=False).to_dict(),
            "chembl_row_fallback_legacy_rdkit_rows": row_fallback_count,
            "removed_reason_counts": removed_reason_counts,
            "parent_structure_changed_rows": parent_changed_count,
            "stereochemistry_stripped_for_identity_rows": stereo_identity_removed_count,
        }
        identity_diagnostics_path.write_text(json.dumps(identity_diagnostics, indent=2) + "\n")
        manifest = {
            "dataset_id": dataset_id or Path(dataset_path).stem,
            "curation_backend": backend_result.get("used_backend_name"),
            "curation_policy": {
                "stereochemistry_policy": curation_policy.get("stereochemistry_policy"),
                "duplicate_identity_policy": curation_policy.get("duplicate_identity_policy"),
                "duplicate_conflict_threshold": duplicate_conflict_threshold,
                "checker_policy": curation_policy.get("checker_policy"),
            },
            "files": {
                "curated_dataset_csv": {
                    "path": str(curated_path),
                    "description": "QSAR-ready curated dataset.",
                },
                "curation_report_json": {
                    "path": "<written by write_curation_report>",
                    "description": "Canonical machine-readable curation summary, written by write_curation_report.",
                },
                "standardization_map_csv": {
                    "path": str(standardization_map_path),
                    "description": "Per-row raw, ChEMBL input, standardized, and QSAR identity mapping.",
                },
                "duplicate_identity_groups_csv": {
                    "path": str(duplicate_groups_path),
                    "description": "Duplicate QSAR identity groups with row indices, target values, spreads, and resolution.",
                },
                "removed_rows_csv": {
                    "path": str(removed_rows_path),
                    "description": "Rows or duplicate groups removed during curation with explicit removal reasons.",
                },
                "identity_diagnostics_json": {
                    "path": str(identity_diagnostics_path),
                    "description": "Compact diagnostics for backend, checker, duplicate identity, and identity stripping.",
                },
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        curation_artifacts = {
            "standardization_map_csv": str(standardization_map_path),
            "duplicate_identity_groups_csv": str(duplicate_groups_path),
            "removed_rows_csv": str(removed_rows_path),
            "identity_diagnostics_json": str(identity_diagnostics_path),
            "manifest_json": str(manifest_path),
        }

        result = CurationResult(
            status="ready" if not blocking_issues else "blocked",
            ready_for_qsar=not blocking_issues,
            dataset_id=dataset_id or Path(dataset_path).stem,
            source_dataset_path=dataset_path,
            curated_dataset_path=str(curated_path),
            smiles_column_original=smiles_column,
            smiles_column_curated="smiles",
            target_columns_original=list(target_columns),
            target_columns_curated=curated_targets,
            task_type=task_type,
            rows_in=rows_in,
            rows_out=rows_out,
            retained_columns=list(working.columns),
            invalid_smiles_removed=invalid_before_drop,
            inorganic_rows_removed=inorganic_rows_removed,
            organometallic_rows_removed=organometallic_rows_removed,
            mixture_rows_removed=mixture_rows_removed,
            salt_or_counterion_rows_processed=salt_or_counterion_rows_processed,
            duplicate_rows_removed=duplicate_rows_removed,
            duplicate_groups_detected=duplicate_groups_detected,
            duplicate_groups_aggregated=duplicate_groups_aggregated,
            duplicate_conflicting_groups=duplicate_conflicting_groups,
            duplicate_conflicting_rows_removed=duplicate_conflicting_rows_removed,
            curation_backend=backend_result.get("backend_name"),
            curation_backend_used=backend_result.get("used_backend_name"),
            curation_backend_fallback_used=bool(backend_result.get("fallback_used")),
            curation_backend_fallback_reason=backend_result.get("fallback_reason"),
            curation_identity_key_type=(
                str(working["curation_identity_key_type"].dropna().iloc[0])
                if "curation_identity_key_type" in working.columns
                and not working["curation_identity_key_type"].dropna().empty
                else None
            ),
            curation_artifacts=curation_artifacts,
            curation_diagnostics=identity_diagnostics,
            missing_target_removed=missing_target_removed,
            non_numeric_target_removed=non_numeric_target_removed,
            infinite_target_removed=infinite_target_removed,
            constant_target_columns=constant_target_columns,
            stereochemistry_markers_removed=stereochemistry_markers_removed,
            target_summaries=target_summaries,
            curation_actions=actions,
            curation_policy=curation_policy,
            target_data_quality=target_data_quality,
            warnings=warnings,
            blocking_issues=blocking_issues,
            report_path=report_path,
        )

        if report_path:
            report_payload = result.as_dict()
            report_file = Path(_absolute_storage_path(report_path)).expanduser()
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(json.dumps(report_payload, indent=2) + "\n")
            result.report_path = str(report_file)

        result_payload = result.as_dict()

        if agent is not None:
            state = _get_curation_state(agent)
            state["last_request"] = request.as_dict()
            state["last_result"] = result_payload
            state["history"].append(result_payload)

        return result_payload

    def summarize_curated_dataset(
        self,
        curated_dataset_path: str,
        smiles_column: str = "smiles",
        target_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Summarize a curated dataset that is ready for QSAR training."""
        df = _load_dataset(curated_dataset_path)
        target_columns = target_columns or [
            column for column in df.columns if column != smiles_column
        ]

        target_summaries = []
        for column in target_columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                continue
            series = pd.to_numeric(df[column], errors="coerce").dropna()
            if series.empty:
                continue
            target_summaries.append(
                TargetSummary(
                    column=column,
                    mean=float(series.mean()),
                    std=float(series.std()) if len(series) > 1 else 0.0,
                    minimum=float(series.min()),
                    maximum=float(series.max()),
                    median=float(series.median()),
                ).as_dict()
            )

        return {
            "curated_dataset_path": curated_dataset_path,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "smiles_column": smiles_column,
            "target_columns": target_columns,
            "target_summaries": target_summaries,
        }

    def write_curation_report(
        self,
        curation_result: Dict[str, Any],
        report_path: Optional[str] = None,
        agent: Optional[Agent] = None,
    ) -> Dict[str, Any]:
        """Persist a curation result as a JSON report artifact."""
        dataset_id = curation_result.get("dataset_id") or "qsar_dataset"
        destination = (
            Path(_absolute_storage_path(report_path)).expanduser()
            if report_path
            else (Path(".files") / "qsar_curation" / f"{dataset_id}_curation_report.json").resolve()
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(curation_result, indent=2) + "\n")

        payload = {
            "report_path": str(destination),
            "download_file_ref": str(destination),
            "dataset_id": dataset_id,
        }
        curated_dataset_path = curation_result.get("curated_dataset_path")
        if curated_dataset_path:
            curated_path = Path(curated_dataset_path).expanduser()
            manifest_ref = (curation_result.get("curation_artifacts") or {}).get(
                "manifest_json"
            )
            if manifest_ref:
                manifest_path = Path(manifest_ref).expanduser()
                if manifest_path.exists():
                    try:
                        manifest = json.loads(manifest_path.read_text())
                        manifest.setdefault("files", {}).setdefault(
                            "curation_report_json", {}
                        )["path"] = str(destination)
                        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                    except Exception:
                        pass
            artifact_files = [
                Path(path).expanduser()
                for path in (curation_result.get("curation_artifacts") or {}).values()
                if path
            ]
            bundle_path = (
                Path(".files")
                / "qsar_curation"
                / f"{dataset_id}_curation_bundle.zip"
            ).resolve()
            bundle = _bundle_files(bundle_path, [curated_path, destination, *artifact_files])
            payload["bundle_file_ref"] = str(bundle)
            payload["bundle_download_tag"] = f"<file>{bundle}</file>"
        if agent is not None:
            state = _get_curation_state(agent)
            if state.get("last_result"):
                state["last_result"]["report_path"] = str(destination)
                if payload.get("bundle_file_ref"):
                    state["last_result"]["bundle_file_ref"] = payload["bundle_file_ref"]
                    state["last_result"]["bundle_download_tag"] = payload[
                        "bundle_download_tag"
                    ]
        return payload
