"""Named curation policies used by QSAR dataset curation backends."""

from __future__ import annotations

DEFAULT_CURATION_BACKEND = "chembl_structure_v1"

STEREOCHEMISTRY_POLICY_STRIP_THEN_DEDUPLICATE = "strip_then_deduplicate"

DEFAULT_DUPLICATE_CONFLICT_THRESHOLD = 0.5

CHEMBL_QSAR_POLICY = {
    "curation_backend": DEFAULT_CURATION_BACKEND,
    "structure_pipeline": [
        "chembl standardize_mol",
        "chembl get_parent_mol",
        "chembl checker diagnostics on standardized parent",
        "ChemSpace Copilot QSAR stereo strip",
        "duplicate resolution by QSAR identity",
    ],
    "fragment_handling": "chembl_get_parent",
    "smiles_standardization": "chembl_structure_pipeline standardize -> get_parent",
    "stereochemistry_policy": STEREOCHEMISTRY_POLICY_STRIP_THEN_DEDUPLICATE,
    "duplicate_identity_policy": "qsar_identity_after_stereo_strip",
    "checker_policy": "warn_only_after_standardized_parent",
}
