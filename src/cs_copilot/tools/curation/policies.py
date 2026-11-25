"""Named curation policies used by QSAR dataset curation backends."""

from __future__ import annotations

from cs_copilot.tools.chemistry.qsar_identity import (
    QSAR_REMOVE_STEREOCHEMISTRY,
    QSAR_STRUCTURE_POLICY,
)

DEFAULT_CURATION_BACKEND = "chembl_structure_v1"

STEREOCHEMISTRY_POLICY_STRIP_THEN_DEDUPLICATE = "strip_then_deduplicate"

DEFAULT_DUPLICATE_CONFLICT_THRESHOLD = 0.5

CHEMBL_QSAR_POLICY = {
    "curation_backend": DEFAULT_CURATION_BACKEND,
    "structure_pipeline": [
        "chembl standardize_mol",
        "chembl get_parent_mol",
        "chembl checker diagnostics on standardized parent",
        "Qsaria modeling parent and canonical tautomer with stereochemistry removed",
        "duplicate resolution by QSAR identity",
    ],
    "fragment_handling": "chembl_get_parent",
    "smiles_standardization": "ChEMBL standardized parent followed by Qsaria model identity",
    "stereochemistry_policy": STEREOCHEMISTRY_POLICY_STRIP_THEN_DEDUPLICATE,
    "remove_stereochemistry": QSAR_REMOVE_STEREOCHEMISTRY,
    "structure_policy": QSAR_STRUCTURE_POLICY,
    "duplicate_identity_policy": "qsar_identity_after_tautomer_normalization_and_stereo_strip",
    "checker_policy": "warn_only_after_standardized_parent",
}
