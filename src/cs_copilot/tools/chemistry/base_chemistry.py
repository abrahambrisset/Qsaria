#!/usr/bin/env python
# coding: utf-8
"""
Base chemistry toolkit providing general molecular operations and utilities.

This module contains fundamental molecular operations that can be reused
across different chemistry-related tools and applications.
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np
from agno.tools.toolkit import Toolkit
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Fingerprints import FingerprintMols

from .standardize import standardize_smiles

logger = logging.getLogger(__name__)


def _smiles_to_mol_or_none(smiles: str) -> "Chem.Mol | None":
    """Standardize *smiles* and return the corresponding RDKit mol, or None on failure.

    Centralises the repeated ``standardize → MolFromSmiles`` pattern used by
    the standalone fingerprint helpers and the toolkit's ``smiles_to_mol``.
    """
    smiles_std = standardize_smiles(smiles)
    if smiles_std is None:
        return None
    return Chem.MolFromSmiles(smiles_std)


class ChemistryError(Exception):
    """Base exception for chemistry operations."""

    pass


class InvalidSMILESError(ChemistryError):
    """Invalid SMILES string provided."""

    pass


class BaseChemistryToolkit(Toolkit):
    """
    Base toolkit for general molecular operations and utilities.

    This class provides fundamental molecular operations that can be extended
    by more specialized chemistry toolkits.
    """

    def __init__(self, name: str = "base_chemistry"):
        """Initialize the BaseChemistryToolkit.

        Args:
            name: Name of the toolkit (used for identification)
        """
        super().__init__(name)
        # Register all chemistry tools
        self.register(self.validate_smiles)
        self.register(self.smiles_to_mol)
        self.register(self.mol_to_smiles)
        self.register(self.get_molecular_weight)
        self.register(self.get_molecular_formula)
        self.register(self.get_lipinski_descriptors)
        self.register(self.get_basic_descriptors)
        self.register(self.generate_fingerprint)
        self.register(self.batch_validate_smiles)
        self.register(self.filter_valid_smiles)
        self.register(self.get_smiles_statistics)

    def validate_smiles(self, smiles: str) -> bool:
        """
        Validate if a SMILES string is valid.

        Args:
            smiles: SMILES string to validate

        Returns:
            True if valid, False otherwise
        """
        if not smiles or not isinstance(smiles, str):
            return False

        try:
            return standardize_smiles(smiles) is not None
        except Exception:
            return False

    def smiles_to_mol(self, smiles: str) -> Chem.Mol:
        """
        Convert SMILES string to RDKit molecule object.

        Standardizes the SMILES (cleanup, largest fragment, uncharge, canonical
        tautomer) before conversion so downstream tools always operate on a
        consistent molecular representation.

        Args:
            smiles: SMILES string

        Returns:
            RDKit molecule object built from the standardized SMILES

        Raises:
            InvalidSMILESError: If SMILES string is invalid or cannot be standardized
        """
        if not smiles or not isinstance(smiles, str):
            raise InvalidSMILESError("SMILES must be a non-empty string")

        mol = _smiles_to_mol_or_none(smiles)
        if mol is None:
            raise InvalidSMILESError(f"Invalid or unstandardizable SMILES string: {smiles}")

        return mol

    def mol_to_smiles(self, mol: Chem.Mol, canonical: bool = True) -> str:
        """
        Convert RDKit molecule object to SMILES string.

        Args:
            mol: RDKit molecule object
            canonical: Whether to return canonical SMILES

        Returns:
            SMILES string
        """
        if mol is None:
            raise InvalidSMILESError("Molecule object is None")

        if canonical:
            return Chem.MolToSmiles(mol)
        else:
            return Chem.MolToSmiles(mol, canonical=False)

    def get_molecular_weight(self, smiles: str) -> float:
        """
        Calculate molecular weight from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Molecular weight in g/mol
        """
        mol = self.smiles_to_mol(smiles)
        return Descriptors.MolWt(mol)

    def get_molecular_formula(self, smiles: str) -> str:
        """
        Get molecular formula from SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Molecular formula string
        """
        mol = self.smiles_to_mol(smiles)
        return Descriptors.rdMolDescriptors.CalcMolFormula(mol)

    def get_lipinski_descriptors(self, smiles: str) -> Dict[str, Union[int, float]]:
        """
        Calculate Lipinski's Rule of Five descriptors.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary containing Lipinski descriptors
        """
        mol = self.smiles_to_mol(smiles)

        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "hbd": Descriptors.NumHDonors(mol),
            "hba": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "aromatic_rings": Descriptors.NumAromaticRings(mol),
            "heavy_atoms": Descriptors.HeavyAtomCount(mol),
        }

    def get_basic_descriptors(self, smiles: str) -> Dict[str, Union[int, float]]:
        """
        Calculate basic molecular descriptors.

        Args:
            smiles: SMILES string

        Returns:
            Dictionary containing basic descriptors
        """
        mol = self.smiles_to_mol(smiles)

        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": Descriptors.RingCount(mol),
            "num_aromatic_rings": Descriptors.NumAromaticRings(mol),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_hbd": Descriptors.NumHDonors(mol),
            "num_hba": Descriptors.NumHAcceptors(mol),
            "formal_charge": Chem.rdmolops.GetFormalCharge(mol),
        }

    def generate_fingerprint(self, smiles: str, fp_type: str = "morgan") -> Any:
        """
        Generate molecular fingerprint from SMILES string.

        Args:
            smiles: SMILES string
            fp_type: Type of fingerprint ('morgan', 'rdkit', 'maccs')

        Returns:
            Molecular fingerprint object
        """
        mol = self.smiles_to_mol(smiles)

        if fp_type.lower() == "morgan":
            from rdkit.Chem import rdMolDescriptors

            return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
        elif fp_type.lower() == "rdkit":
            return FingerprintMols.FingerprintMol(mol)
        elif fp_type.lower() == "maccs":
            from rdkit.Chem import MACCSkeys

            return MACCSkeys.GenMACCSKeys(mol)
        else:
            raise ValueError(f"Unsupported fingerprint type: {fp_type}")

    def batch_validate_smiles(self, smiles_list: List[str]) -> List[bool]:
        """
        Validate multiple SMILES strings in batch.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of validation results (True/False)
        """
        return [self.validate_smiles(smiles) for smiles in smiles_list]

    def filter_valid_smiles(self, smiles_list: List[str]) -> List[str]:
        """
        Filter out invalid SMILES strings from a list.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of valid SMILES strings
        """
        return [smiles for smiles in smiles_list if self.validate_smiles(smiles)]

    def get_smiles_statistics(self, smiles_list: List[str]) -> Dict[str, Any]:
        """
        Calculate statistics for a list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dictionary containing statistics
        """
        valid_smiles = self.filter_valid_smiles(smiles_list)

        if not valid_smiles:
            return {
                "total_count": len(smiles_list),
                "valid_count": 0,
                "invalid_count": len(smiles_list),
                "validity_rate": 0.0,
            }

        # Calculate molecular weights for valid molecules
        try:
            mol_weights = [self.get_molecular_weight(smiles) for smiles in valid_smiles]
            avg_mw = sum(mol_weights) / len(mol_weights)
            min_mw = min(mol_weights)
            max_mw = max(mol_weights)
        except Exception as e:
            logger.warning(f"Error calculating molecular weights: {e}")
            avg_mw = min_mw = max_mw = None

        return {
            "total_count": len(smiles_list),
            "valid_count": len(valid_smiles),
            "invalid_count": len(smiles_list) - len(valid_smiles),
            "validity_rate": len(valid_smiles) / len(smiles_list),
            "avg_molecular_weight": avg_mw,
            "min_molecular_weight": min_mw,
            "max_molecular_weight": max_mw,
        }


# Standalone fingerprint functions (not part of toolkit, used in tools)
def calc_morgan_fp(smiles: str, nbits: int) -> np.ndarray:
    """
    Calculate the Morgan fingerprint for a given SMILES string using a specified bit size.

    Args:
        smiles (str): SMILES string of the molecule.
        nbits (int): Number of bits for the fingerprint.

    Returns:
        np.ndarray: Morgan fingerprint as a numpy array, or None if molecule creation fails.
    """
    mol = _smiles_to_mol_or_none(smiles)
    if mol is None:
        return None
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    return fp_generator.GetCountFingerprintAsNumPy(mol)


def calc_morgan_bit_fp(smiles: str, nbits: int) -> np.ndarray:
    """
    Calculate a binary Morgan fingerprint for a given SMILES string.

    Args:
        smiles (str): SMILES string of the molecule.
        nbits (int): Number of bits for the fingerprint.

    Returns:
        np.ndarray: Binary Morgan fingerprint as a numpy array, or None if
            molecule creation fails.
    """
    mol = _smiles_to_mol_or_none(smiles)
    if mol is None:
        return None
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nbits)
    return fp_generator.GetFingerprintAsNumPy(mol)


def calc_fp(smiles: str, fp_generator) -> np.ndarray:
    """
    Calculate a fingerprint for a given SMILES string using the provided fingerprint generator.

    Args:
        smiles (str): SMILES string of the molecule.
        fp_generator: A fingerprint generator object.

    Returns:
        np.ndarray: Fingerprint as a numpy array, or None if molecule is invalid.
    """
    mol = _smiles_to_mol_or_none(smiles)
    if mol is None:
        return None
    return fp_generator.GetCountFingerprintAsNumPy(mol)


def calc_maccs_fp(smiles: str) -> np.ndarray:
    """
    Calculate the MACCS keys fingerprint for a given SMILES string.

    Args:
        smiles (str): SMILES string of the molecule.

    Returns:
        np.ndarray: MACCS keys as a numpy array, or None if molecule is invalid.
    """
    mol = _smiles_to_mol_or_none(smiles)
    if mol is None:
        return None
    return np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=np.float64)
