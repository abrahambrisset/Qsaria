from __future__ import annotations

from typing import Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

_UNCHARGER = rdMolStandardize.Uncharger()
_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
_SMILES_COLUMN_CANDIDATES = ("smiles", "SMILES", "canonical_smiles", "smi", "structure")


def standardize_smiles(smiles: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        clean_mol = rdMolStandardize.Cleanup(mol)
        parent = rdMolStandardize.FragmentParent(clean_mol)
        uncharged = _UNCHARGER.uncharge(parent)
        tautomer = _TAUTOMER_ENUMERATOR.Canonicalize(uncharged)

        return Chem.MolToSmiles(tautomer, canonical=True)
    except Exception:
        return None


def standardize_smiles_column(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """Apply SMILES standardization to a DataFrame column in-place.

    Rows where standardization fails (invalid SMILES) will have the column value
    set to None/NaN so callers can drop them with dropna(subset=[col_name]).

    Args:
        df: DataFrame containing the SMILES column.
        col_name: Name of the column holding SMILES strings.

    Returns:
        The same DataFrame with the column values replaced by standardized SMILES.
    """

    df[col_name] = df[col_name].apply(
        lambda s: standardize_smiles(s) if isinstance(s, str) else None
    )
    return df


def resolve_smiles_column_name(df: pd.DataFrame, requested_column: str = "smiles") -> str:
    """Resolve a requested SMILES column against common QSAR column aliases.

    Curated QSAR datasets use canonical lowercase ``smiles`` even when the
    source dataset used ``SMILES``. This helper lets downstream tools recover
    from that harmless source/curated naming drift without guessing via the LLM.
    """
    columns = list(df.columns)
    if requested_column in columns:
        return requested_column

    requested_lower = str(requested_column).lower()
    by_lower = {str(column).lower(): str(column) for column in columns}
    if requested_lower in by_lower:
        return by_lower[requested_lower]

    for candidate in _SMILES_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
        candidate_lower = candidate.lower()
        if candidate_lower in by_lower:
            return by_lower[candidate_lower]

    raise KeyError(
        f"SMILES column '{requested_column}' not found. Available columns: {columns}"
    )
