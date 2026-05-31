"""Structure identity helpers for QSAR curation."""

from __future__ import annotations

from typing import Optional

from rdkit import Chem


def strip_stereochemistry_from_smiles(smiles: str) -> Optional[str]:
    """Return a canonical SMILES after removing atom and bond stereochemistry."""

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except Exception:
        return None


def has_explicit_stereochemistry(smiles: str) -> bool:
    """Detect explicit stereochemical information in a SMILES string."""

    if not isinstance(smiles, str):
        return False
    if "@" in smiles or "/" in smiles or "\\" in smiles:
        return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    if chiral_centers:
        return True
    return any(
        bond.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
        for bond in mol.GetBonds()
    )

