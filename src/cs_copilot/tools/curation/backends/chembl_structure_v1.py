"""Adapter for the official ChEMBL Structure Pipeline package."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
from rdkit import Chem

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


def ensure_chembl_structure_pipeline_available() -> Tuple[Any, Any]:
    """Import the mandatory ChEMBL pipeline or fail with an actionable error."""
    try:
        from chembl_structure_pipeline import checker, standardizer
    except Exception as exc:
        raise RuntimeError(
            "The mandatory `chembl_structure_pipeline` dependency is unavailable. "
            "Install the Qsaria project dependencies before initializing dataset curation."
        ) from exc
    return checker, standardizer


def standardize_with_chembl_structure_v1(raw_smiles: pd.Series) -> Dict[str, Any]:
    """Standardize a SMILES series with ChEMBL, then apply QSAR identity policy."""

    checker, standardizer = ensure_chembl_structure_pipeline_available()

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
                standardized = None
                qsar_identity = None
                parent_structure_changed = False
                stereo_removed = False
                reason = "chembl_empty_standardized_or_identity"
                checker_issues = f"{checker_issues}; {reason}" if checker_issues else reason
                status = "standardization_failed"
        except Exception as exc:
            standardized = None
            qsar_identity = None
            parent_structure_changed = False
            stereo_removed = False
            reason = f"standardization_error:{exc}"
            checker_issues = f"{checker_issues}; {reason}" if checker_issues else reason
            status = "standardization_failed"

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
        "identity_column": "curation_identity_key",
        "standardization_map": pd.DataFrame(rows),
    }
