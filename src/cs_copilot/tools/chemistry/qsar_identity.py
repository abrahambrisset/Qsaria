"""Reuse the molecular identity of curated Qsaria datasets during featurization."""

from __future__ import annotations

import pandas as pd

from .standardize import standardize_smiles_column

QSAR_STRUCTURE_POLICY_COLUMN = "qsar_structure_policy"
QSAR_STRUCTURE_POLICY = "rdkit_parent_tautomer_nonisomeric_v1"
QSAR_REMOVE_STEREOCHEMISTRY = True


def prepare_qsar_smiles(
    frame: pd.DataFrame,
    smiles_column: str,
    *,
    max_workers: int | None = None,
    reject_collisions: bool = False,
) -> pd.DataFrame:
    """Use the recorded model SMILES as-is, or standardize an unmarked source once."""
    original = frame[smiles_column]
    legacy_identity = frame.get("curation_identity_key")
    if QSAR_STRUCTURE_POLICY_COLUMN in frame:
        marked = frame[QSAR_STRUCTURE_POLICY_COLUMN].eq(QSAR_STRUCTURE_POLICY).fillna(False)
        if not marked.all() or (
            legacy_identity is not None and not original.equals(legacy_identity)
        ):
            raise ValueError(
                "The recorded QSAR structure policy does not match this dataset. Recurate it before featurization."
            )
        if legacy_identity is not None and original.duplicated().any():
            raise ValueError(
                "The curated dataset has duplicate model structures. Recurate it before splitting."
            )
        return frame.copy()

    prepared = standardize_smiles_column(
        frame.copy(),
        smiles_column,
        remove_stereochemistry=QSAR_REMOVE_STEREOCHEMISTRY,
        max_workers=max_workers,
    )
    if reject_collisions or legacy_identity is not None:
        pairs = pd.DataFrame({"input": original, "identity": prepared[smiles_column]}).dropna()
        collapsed = pairs.groupby("identity", sort=False)["input"].nunique().gt(1)
        if collapsed.any():
            raise ValueError(
                f"Standardization merges distinct inputs in {int(collapsed.sum())} molecular identity group(s). "
                "Recurate the labelled dataset before splitting so its target values can be resolved together."
            )
    return prepared
