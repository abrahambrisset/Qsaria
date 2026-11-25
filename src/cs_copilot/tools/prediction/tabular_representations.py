#!/usr/bin/env python
# coding: utf-8
"""Canonical tabular representation registry for QSAR training."""

from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

#: Morgan defaults shared by every representation that does not override them.  They are the
#: historical hardcoded values, so leaving them alone keeps existing models reproducible.
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_N_BITS = 2048


@dataclass(frozen=True)
class TabularRepresentationSpec:
    """Declarative description of a reusable tabular molecular representation."""

    name: str
    display_name: str
    use_morgan_binary: bool = False
    use_morgan_count: bool = False
    use_rdkit: bool = False
    descriptor_set: Optional[str] = None
    automatic: bool = False
    description: str = ""
    morgan_radius: int = DEFAULT_MORGAN_RADIUS
    morgan_n_bits: int = DEFAULT_MORGAN_N_BITS
    morgan_use_features: bool = False
    #: Non-Morgan structural fingerprints built by
    #: ``MolecularFeatureToolkit.smiles_to_structural_fingerprints``.
    fingerprints: Tuple[str, ...] = field(default_factory=tuple)
    #: Optional import path this representation needs; used to report availability without
    #: importing the module.
    requires_module: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["fingerprints"] = list(self.fingerprints)
        payload["available"] = representation_unavailable_reason(self.name) is None
        payload["unavailable_reason"] = representation_unavailable_reason(self.name)
        return payload


TABULAR_REPRESENTATIONS: Dict[str, TabularRepresentationSpec] = {
    "rdkit_all": TabularRepresentationSpec(
        name="rdkit_all",
        display_name="RDKit all descriptors",
        use_rdkit=True,
        descriptor_set="all",
        automatic=True,
        description="Full RDKit descriptor table.",
    ),
    "morgan_only": TabularRepresentationSpec(
        name="morgan_only",
        display_name="Morgan binary fingerprint",
        use_morgan_binary=True,
        automatic=True,
        description="Binary ECFP/Morgan fingerprint bits.",
    ),
    "morgan_count_only": TabularRepresentationSpec(
        name="morgan_count_only",
        display_name="Morgan count fingerprint",
        use_morgan_count=True,
        automatic=True,
        description="Count-based ECFP/Morgan fingerprint bins.",
    ),
    "morgan_binary_count_rdkit_all": TabularRepresentationSpec(
        name="morgan_binary_count_rdkit_all",
        display_name="Morgan binary + count + RDKit all",
        use_morgan_binary=True,
        use_morgan_count=True,
        use_rdkit=True,
        descriptor_set="all",
        automatic=False,
        description="Explicit advanced combined representation. Use only when requested.",
    ),
    "morgan_rdkit_all": TabularRepresentationSpec(
        name="morgan_rdkit_all",
        display_name="Morgan binary + RDKit all",
        use_morgan_binary=True,
        use_rdkit=True,
        descriptor_set="all",
        automatic=False,
        description="Historical high-validation combined representation retained for compatibility and explicit use.",
    ),
    "maccs": TabularRepresentationSpec(
        name="maccs",
        display_name="MACCS structural keys",
        fingerprints=("maccs",),
        automatic=False,
        description="166 MACCS keys as a 167-bit vector; small, interpretable substructure keys.",
    ),
    "avalon": TabularRepresentationSpec(
        name="avalon",
        display_name="Avalon fingerprint",
        fingerprints=("avalon",),
        automatic=False,
        description="Avalon substructure fingerprint (1024 bits).",
        requires_module="rdkit.Avalon.pyAvalonTools",
    ),
    "atom_pair": TabularRepresentationSpec(
        name="atom_pair",
        display_name="Atom-pair fingerprint",
        fingerprints=("atom_pair",),
        automatic=False,
        description="Hashed atom-pair fingerprint (2048 bits); encodes topological distances.",
    ),
    "topological_torsion": TabularRepresentationSpec(
        name="topological_torsion",
        display_name="Topological-torsion fingerprint",
        fingerprints=("topological_torsion",),
        automatic=False,
        description="Hashed topological-torsion fingerprint (2048 bits) over four-atom paths.",
    ),
}

AUTOMATIC_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.automatic
)
SUPPORTED_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(TABULAR_REPRESENTATIONS)


@lru_cache(maxsize=None)
def representation_unavailable_reason(name: str) -> Optional[str]:
    """Return why a representation cannot be built here, or None when it is usable.

    Availability is decided from ``requires_module`` with ``importlib.util.find_spec`` so an
    optional RDKit component (for example Avalon) never turns into an import-time crash.
    """
    spec = TABULAR_REPRESENTATIONS.get(str(name or "").strip().lower())
    if spec is None or not spec.requires_module:
        return None
    try:
        found = importlib.util.find_spec(spec.requires_module) is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return None
    return (
        f"Representation {spec.name!r} needs {spec.requires_module!r}, which this build does not "
        "provide. Install an RDKit build that includes it or pick another representation."
    )


def get_tabular_representation(name: str) -> TabularRepresentationSpec:
    """Return a registered tabular representation spec."""
    normalized = str(name or "").strip().lower()
    try:
        return TABULAR_REPRESENTATIONS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported representation_name={name!r}. "
            f"Expected one of {sorted(TABULAR_REPRESENTATIONS)}."
        ) from exc


def default_tabular_representation_for_protocol(
    protocol: str, *, training_profile: Optional[str] = None
) -> str:
    """Return the single representation used when a protocol is not comparative."""
    normalized = str(protocol or "").strip().lower()
    if normalized == "standard_qsar":
        return "rdkit_all"
    # Keep the historical strong default for explicit single-model tabular training.
    if training_profile == "heavy_validation":
        return "morgan_rdkit_all"
    return "morgan_only"


def automatic_tabular_representations() -> List[TabularRepresentationSpec]:
    """Return representation specs used by modern comparative tabular campaigns."""
    return [get_tabular_representation(name) for name in AUTOMATIC_TABULAR_REPRESENTATION_NAMES]


def tabular_candidate_id(backend_name: str, representation_name: str) -> str:
    """Return the canonical candidate identifier for a backend/representation pair."""
    return f"{str(backend_name).strip().lower()}_{str(representation_name).strip().lower()}"


def tabular_candidates_for_backend(
    backend_name: str,
    *,
    representation_names: Optional[List[str]] = None,
    single_default_protocol: Optional[str] = None,
    training_profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build candidate dictionaries for a tabular backend."""
    normalized_backend = str(backend_name).strip().lower()
    if representation_names:
        specs = [get_tabular_representation(name) for name in representation_names]
    elif single_default_protocol:
        specs = [
            get_tabular_representation(
                default_tabular_representation_for_protocol(
                    single_default_protocol,
                    training_profile=training_profile,
                )
            )
        ]
    else:
        specs = automatic_tabular_representations()

    return [
        {
            "candidate_id": tabular_candidate_id(normalized_backend, spec.name),
            "backend_name": normalized_backend,
            "representation_name": spec.name,
            "representation_display_name": spec.display_name,
            "representation_automatic": spec.automatic,
        }
        for spec in specs
    ]


def describe_tabular_representations() -> Dict[str, Dict[str, Any]]:
    """Return all registered representation specs as dictionaries."""
    return {name: spec.as_dict() for name, spec in TABULAR_REPRESENTATIONS.items()}
