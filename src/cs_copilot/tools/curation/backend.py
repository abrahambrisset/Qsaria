#!/usr/bin/env python
# coding: utf-8
"""
Structured contracts for the QSAR dataset curation workflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CurationRequest:
    """Declarative request for preparing a QSAR-ready dataset."""

    dataset_path: str
    task_type: str
    endpoint_name: Optional[str] = None
    dataset_id: Optional[str] = None
    preferred_smiles_column: Optional[str] = None
    preferred_target_columns: List[str] = field(default_factory=list)
    curation_policy: Optional[str] = None
    duplicate_conflict_threshold: Optional[float] = None
    curation_backend: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "task_type": self.task_type,
            "endpoint_name": self.endpoint_name,
            "dataset_id": self.dataset_id,
            "preferred_smiles_column": self.preferred_smiles_column,
            "preferred_target_columns": list(self.preferred_target_columns),
            "curation_policy": self.curation_policy,
            "duplicate_conflict_threshold": self.duplicate_conflict_threshold,
            "curation_backend": self.curation_backend,
        }


@dataclass
class TargetSummary:
    """Compact numeric summary for target columns after curation."""

    column: str
    mean: Optional[float] = None
    std: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    median: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "median": self.median,
        }


@dataclass
class CurationResult:
    """Structured output from the curation workflow."""

    status: str
    ready_for_qsar: bool
    dataset_id: Optional[str]
    source_dataset_path: str
    curated_dataset_path: Optional[str]
    smiles_column_original: Optional[str]
    smiles_column_curated: Optional[str]
    target_columns_original: List[str]
    target_columns_curated: List[str]
    task_type: str
    rows_in: int
    rows_out: int
    retained_columns: List[str] = field(default_factory=list)
    invalid_smiles_removed: int = 0
    inorganic_rows_removed: int = 0
    organometallic_rows_removed: int = 0
    mixture_rows_removed: int = 0
    salt_or_counterion_rows_processed: int = 0
    duplicate_rows_removed: int = 0
    duplicate_groups_detected: int = 0
    duplicate_groups_aggregated: int = 0
    duplicate_conflicting_groups: int = 0
    duplicate_conflicting_rows_removed: int = 0
    curation_backend: Optional[str] = None
    curation_backend_used: Optional[str] = None
    curation_backend_fallback_used: bool = False
    curation_backend_fallback_reason: Optional[str] = None
    curation_identity_key_type: Optional[str] = None
    curation_artifacts: Dict[str, Any] = field(default_factory=dict)
    curation_diagnostics: Dict[str, Any] = field(default_factory=dict)
    missing_target_removed: int = 0
    non_numeric_target_removed: int = 0
    infinite_target_removed: int = 0
    constant_target_columns: List[str] = field(default_factory=list)
    stereochemistry_markers_removed: int = 0
    target_summaries: List[TargetSummary] = field(default_factory=list)
    curation_actions: List[str] = field(default_factory=list)
    curation_policy: Dict[str, Any] = field(default_factory=dict)
    target_data_quality: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    blocking_issues: List[str] = field(default_factory=list)
    report_path: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "ready_for_qsar": self.ready_for_qsar,
            "dataset_id": self.dataset_id,
            "source_dataset_path": self.source_dataset_path,
            "curated_dataset_path": self.curated_dataset_path,
            "smiles_column_original": self.smiles_column_original,
            "smiles_column_curated": self.smiles_column_curated,
            "target_columns_original": list(self.target_columns_original),
            "target_columns_curated": list(self.target_columns_curated),
            "task_type": self.task_type,
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "rows_removed": self.rows_in - self.rows_out,
            "retained_columns": list(self.retained_columns),
            "invalid_smiles_removed": self.invalid_smiles_removed,
            "inorganic_rows_removed": self.inorganic_rows_removed,
            "organometallic_rows_removed": self.organometallic_rows_removed,
            "mixture_rows_removed": self.mixture_rows_removed,
            "salt_or_counterion_rows_processed": self.salt_or_counterion_rows_processed,
            "duplicate_rows_removed": self.duplicate_rows_removed,
            "duplicate_groups_detected": self.duplicate_groups_detected,
            "duplicate_groups_aggregated": self.duplicate_groups_aggregated,
            "duplicate_conflicting_groups": self.duplicate_conflicting_groups,
            "duplicate_conflicting_rows_removed": self.duplicate_conflicting_rows_removed,
            "curation_backend": self.curation_backend,
            "curation_backend_used": self.curation_backend_used,
            "curation_backend_fallback_used": self.curation_backend_fallback_used,
            "curation_backend_fallback_reason": self.curation_backend_fallback_reason,
            "curation_identity_key_type": self.curation_identity_key_type,
            "curation_artifacts": dict(self.curation_artifacts),
            "curation_diagnostics": dict(self.curation_diagnostics),
            "missing_target_removed": self.missing_target_removed,
            "non_numeric_target_removed": self.non_numeric_target_removed,
            "infinite_target_removed": self.infinite_target_removed,
            "constant_target_columns": list(self.constant_target_columns),
            "stereochemistry_markers_removed": self.stereochemistry_markers_removed,
            "target_summaries": [summary.as_dict() for summary in self.target_summaries],
            "curation_actions": list(self.curation_actions),
            "curation_policy": dict(self.curation_policy),
            "target_data_quality": dict(self.target_data_quality),
            "warnings": list(self.warnings),
            "blocking_issues": list(self.blocking_issues),
            "report_path": self.report_path,
        }
