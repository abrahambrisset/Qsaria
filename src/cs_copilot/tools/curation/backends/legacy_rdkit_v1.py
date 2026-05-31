"""Legacy RDKit curation backend.

This preserves the historical ChemSpace Copilot standardization behavior so it
can be selected explicitly while newer backends are introduced.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from cs_copilot.tools.chemistry.standardize import standardize_smiles


def standardize_with_legacy_rdkit_v1(raw_smiles: pd.Series) -> Dict[str, Any]:
    """Standardize a SMILES series using the existing RDKit helper."""

    rows = []
    for row_index, smiles in raw_smiles.items():
        raw = smiles if isinstance(smiles, str) else None
        standardized = standardize_smiles(raw) if raw else None
        rows.append(
            {
                "row_index": row_index,
                "raw_smiles": raw,
                "chembl_input_smiles": None,
                "standardized_smiles": standardized,
                "qsar_identity_smiles": standardized,
                "curation_identity_key": standardized,
                "curation_identity_key_type": "standardized_smiles",
                "curation_backend_status": "ok" if standardized else "invalid_smiles",
                "checker_issues": "",
                "checker_max_penalty": 0,
                "parent_structure_changed": False,
                "stereochemistry_removed_for_identity": False,
            }
        )

    return {
        "backend_name": "legacy_rdkit_v1",
        "used_backend_name": "legacy_rdkit_v1",
        "fallback_used": False,
        "fallback_reason": None,
        "identity_column": "curation_identity_key",
        "standardization_map": pd.DataFrame(rows),
    }
