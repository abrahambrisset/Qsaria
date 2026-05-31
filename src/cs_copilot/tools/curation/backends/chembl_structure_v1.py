"""Adapter for the official ChEMBL Structure Pipeline package."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from rdkit import Chem

from cs_copilot.tools.chemistry.standardize import standardize_smiles
from cs_copilot.tools.curation.backends.legacy_rdkit_v1 import (
    standardize_with_legacy_rdkit_v1,
)
from cs_copilot.tools.curation.identity import (
    has_explicit_stereochemistry,
    strip_stereochemistry_from_smiles,
)


def _format_checker_issues(issues: List[Tuple[Any, Any]]) -> Tuple[str, int]:
    if not issues:
        return "", 0
    normalized = []
    for first, second in issues:
        if isinstance(first, int):
            penalty, message = int(first), str(second)
        else:
            penalty, message = int(second), str(first)
        normalized.append((penalty, message))
    max_penalty = max(penalty for penalty, _message in normalized)
    text = "; ".join(f"{penalty}:{message}" for penalty, message in normalized)
    return text, max_penalty


def _mol_to_molblock(mol: Chem.Mol) -> str:
    return Chem.MolToMolBlock(mol, kekulize=False)


def _legacy_row_fallback(raw: str | None) -> Tuple[str | None, str | None, bool, bool]:
    if not raw:
        return None, None, False, False
    standardized = standardize_smiles(raw)
    qsar_identity = strip_stereochemistry_from_smiles(standardized) if standardized else None
    parent_structure_changed = bool(standardized and standardized != raw)
    stereo_removed = bool(
        standardized
        and qsar_identity
        and has_explicit_stereochemistry(standardized)
        and qsar_identity != standardized
    )
    return standardized, qsar_identity, parent_structure_changed, stereo_removed


def _apply_legacy_row_fallback(
    raw: str | None,
    checker_issues: str,
    reason: str,
) -> Tuple[str | None, str | None, bool, bool, str, str]:
    standardized, qsar_identity, parent_structure_changed, stereo_removed = (
        _legacy_row_fallback(raw)
    )
    checker_issues = f"{checker_issues}; {reason}" if checker_issues else reason
    if standardized and qsar_identity:
        checker_issues += "; legacy_rdkit_row_fallback_applied"
        status = "chembl_row_fallback_legacy_rdkit"
    else:
        status = "standardization_failed"
    return (
        standardized,
        qsar_identity,
        parent_structure_changed,
        stereo_removed,
        checker_issues,
        status,
    )


def standardize_with_chembl_structure_v1(raw_smiles: pd.Series) -> Dict[str, Any]:
    """Standardize a SMILES series with ChEMBL, then apply QSAR identity policy."""

    try:
        from chembl_structure_pipeline import checker, standardizer
    except Exception as exc:
        legacy = standardize_with_legacy_rdkit_v1(raw_smiles)
        legacy.update(
            {
                "backend_name": "chembl_structure_v1",
                "used_backend_name": "legacy_rdkit_v1",
                "fallback_used": True,
                "fallback_reason": f"chembl_structure_pipeline unavailable: {exc}",
            }
        )
        return legacy

    rows = []
    for row_index, smiles in raw_smiles.items():
        raw = smiles if isinstance(smiles, str) else None
        mol = Chem.MolFromSmiles(raw) if raw else None
        if mol is None:
            rows.append(
                {
                    "row_index": row_index,
                    "raw_smiles": raw,
                    "chembl_input_smiles": None,
                    "standardized_smiles": None,
                    "qsar_identity_smiles": None,
                    "curation_identity_key": None,
                    "curation_identity_key_type": "qsar_identity_smiles",
                    "curation_backend_status": "invalid_smiles",
                    "checker_issues": "",
                    "checker_max_penalty": 0,
                    "parent_structure_changed": False,
                    "stereochemistry_removed_for_identity": False,
                }
            )
            continue

        checker_issues = ""
        checker_max_penalty = 0
        chembl_input_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True, kekuleSmiles=False
        )
        try:
            standardized_mol = standardizer.standardize_mol(mol)
            parent_mol, _exclude = standardizer.get_parent_mol(standardized_mol)
            standardized = (
                Chem.MolToSmiles(parent_mol, canonical=True, isomericSmiles=True)
                if parent_mol is not None
                else None
            )
            if parent_mol is not None:
                issues = checker.check_molblock(_mol_to_molblock(parent_mol))
                checker_issues, checker_max_penalty = _format_checker_issues(issues)
            qsar_identity = (
                strip_stereochemistry_from_smiles(standardized) if standardized else None
            )
            parent_structure_changed = bool(standardized and raw and standardized != raw)
            stereo_removed = bool(
                standardized
                and qsar_identity
                and has_explicit_stereochemistry(standardized)
                and qsar_identity != standardized
            )
            if standardized and qsar_identity:
                status = "ok"
            else:
                (
                    standardized,
                    qsar_identity,
                    parent_structure_changed,
                    stereo_removed,
                    checker_issues,
                    status,
                ) = _apply_legacy_row_fallback(
                    raw,
                    checker_issues,
                    "chembl_empty_standardized_or_identity",
                )
        except Exception as exc:
            (
                standardized,
                qsar_identity,
                parent_structure_changed,
                stereo_removed,
                checker_issues,
                status,
            ) = _apply_legacy_row_fallback(
                raw,
                checker_issues,
                f"standardization_error:{exc}",
            )

        rows.append(
            {
                "row_index": row_index,
                "raw_smiles": raw,
                "chembl_input_smiles": chembl_input_smiles if mol is not None else None,
                "standardized_smiles": standardized,
                "qsar_identity_smiles": qsar_identity,
                "curation_identity_key": qsar_identity,
                "curation_identity_key_type": "qsar_identity_smiles",
                "curation_backend_status": status,
                "checker_issues": checker_issues,
                "checker_max_penalty": checker_max_penalty,
                "parent_structure_changed": parent_structure_changed,
                "stereochemistry_removed_for_identity": stereo_removed,
            }
        )

    return {
        "backend_name": "chembl_structure_v1",
        "used_backend_name": "chembl_structure_v1",
        "fallback_used": False,
        "fallback_reason": None,
        "identity_column": "curation_identity_key",
        "standardization_map": pd.DataFrame(rows),
    }
