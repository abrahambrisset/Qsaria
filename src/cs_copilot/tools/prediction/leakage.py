"""Audit molecular identity and similarity across QSAR partitions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from cs_copilot.tools.chemistry.standardize import resolve_smiles_column_name, standardize_smiles

from .backend import PredictionModelRecord

REFERENCE_FILENAME = "leakage_reference.csv"
SIMILARITY_THRESHOLDS = (0.95, 0.99)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2)


@lru_cache(maxsize=16_384)
def _fingerprint(smiles: str) -> Any:
    return _MORGAN.GetSparseCountFingerprint(Chem.MolFromSmiles(smiles))


def audit_molecular_overlap(
    training_smiles: Iterable[Any], evaluation_smiles: Iterable[Any]
) -> dict[str, Any]:
    """Compare evaluation rows to fit rows without materializing a similarity matrix."""
    training_keys = [standardize_smiles(value) for value in training_smiles]
    evaluation_keys = [standardize_smiles(value) for value in evaluation_smiles]
    valid_training = [key for key in training_keys if key]
    valid_evaluation = [key for key in evaluation_keys if key]
    reference = dict.fromkeys(valid_training)
    report: dict[str, Any] = {
        "status": "unavailable",
        "identity": "qsaria_standardized_smiles_without_stereochemistry",
        "similarity": "sparse_count_morgan_radius_2_tanimoto",
        "training_rows": len(training_keys),
        "training_valid_rows": len(valid_training),
        "evaluation_rows": len(evaluation_keys),
        "evaluation_valid_rows": len(valid_evaluation),
        "exact_overlap_count": None,
        "near_duplicate_counts": None,
        "nearest_neighbor_similarity": None,
    }
    if not reference or not valid_evaluation:
        report["reason"] = "No usable molecular structures in the fit or evaluation reference."
        return report

    # Repeated evaluation rows count separately. Cached nearest matches also
    # avoid recomputing similarities for replicate measurements.
    fingerprints = [_fingerprint(key) for key in reference]
    nearest_by_identity = {
        key: max(DataStructs.BulkTanimotoSimilarity(_fingerprint(key), fingerprints))
        for key in dict.fromkeys(valid_evaluation)
        if key not in reference
    }
    nonidentical = [nearest_by_identity[key] for key in valid_evaluation if key not in reference]
    exact_matches = len(valid_evaluation) - len(nonidentical)
    nearest = nonidentical + [1.0] * exact_matches
    complete = len(training_keys) == len(valid_training) and len(evaluation_keys) == len(
        valid_evaluation
    )
    report.update(
        {
            "status": "complete" if complete else "partial",
            "exact_overlap_count": exact_matches,
            "near_duplicate_counts": {
                f"{threshold:.2f}": sum(value >= threshold for value in nonidentical)
                for threshold in SIMILARITY_THRESHOLDS
            },
            "nearest_neighbor_similarity": {"max": max(nearest), "median": median(nearest)},
        }
    )
    return report


def audit_training_split(
    frame: pd.DataFrame,
    smiles_column: str,
    split: Mapping[str, Sequence[int]],
    output_dir: Path,
) -> dict[str, Any]:
    """Preserve the actual fit reference for future labelled evaluations."""
    try:
        column = resolve_smiles_column_name(frame, smiles_column)
    except KeyError:
        structures = pd.Series([None] * len(frame), dtype=object)
    else:
        structures = frame[column]

    train_indices = list(split.get("train") or [])
    fitted = structures.iloc[train_indices].map(lambda value: standardize_smiles(value) or None)
    result: dict[str, Any] = {
        "leakage": {
            label: audit_molecular_overlap(fitted, structures.iloc[list(indices)])
            for label, indices in (
                ("validation", split.get("val") or split.get("validation")),
                ("test", split.get("test")),
            )
            if indices
        }
    }
    if fitted.notna().any():
        reference_path = output_dir / REFERENCE_FILENAME
        fitted.rename("smiles").to_csv(reference_path, index=False)
        result["leakage_reference_path"] = str(reference_path)
    return result


def audit_record_overlap(
    record: PredictionModelRecord, evaluation_smiles: Iterable[Any]
) -> dict[str, Any]:
    """Resolve a persisted fit reference or identify a coarser legacy check."""
    evaluation = list(evaluation_smiles)
    if record.metadata_path:
        metadata_path = Path(record.metadata_path).expanduser().resolve()
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, ValueError):
            metadata = {}
        artifacts = metadata.get("artifacts") or {}
        for key, scope, column in (
            ("leakage_reference_path", "fitted_rows", "smiles"),
            (
                "training_dataset_path",
                "legacy_training_dataset",
                (record.task.smiles_columns or ["smiles"])[0],
            ),
        ):
            if not artifacts.get(key):
                continue
            path = (metadata_path.parent / artifacts[key]).resolve()
            if not path.is_relative_to(metadata_path.parent):
                continue
            try:
                actual_column = resolve_smiles_column_name(pd.read_csv(path, nrows=0), column)
                training = pd.read_csv(path, usecols=[actual_column])[actual_column]
            except (OSError, ValueError, KeyError):
                continue
            return {
                **audit_molecular_overlap(training, evaluation),
                "reference_scope": scope,
            }
    return {
        **audit_molecular_overlap([], evaluation),
        "reference_scope": "unavailable",
        "reason": "No persisted training structures were available for this model.",
    }


def overlap_markdown(report: Mapping[str, Any]) -> str:
    """Render compact overlap evidence without interpreting it as a performance metric."""
    lines = [f"- Overlap audit `{report.get('status')}`"]
    for model_id, component_report in (report.get("components") or {}).items():
        lines.extend(["", f"#### `{model_id}`", "", overlap_markdown(component_report)])
    if report.get("reference_scope") == "legacy_training_dataset":
        lines.append(
            "- The legacy reference covers the recorded dataset, including possible holdout rows."
        )
    if report.get("reason"):
        lines.append(f"- {report['reason']}")
    if report.get("exact_overlap_count") is not None:
        near = report["near_duplicate_counts"]
        nearest = report["nearest_neighbor_similarity"]
        lines.extend(
            [
                f"- Exact standardized matches **{report['exact_overlap_count']}**",
                f"- Nonidentical neighbours at Tanimoto ≥ 0.95 **{near['0.95']}** and ≥ 0.99 **{near['0.99']}**",
                f"- Nearest-neighbour Tanimoto median **{nearest['median']:.4f}** and maximum **{nearest['max']:.4f}**",
                f"- Structures audited **{report['evaluation_valid_rows']} of {report['evaluation_rows']}** evaluation rows",
            ]
        )
    return "\n".join(lines)
