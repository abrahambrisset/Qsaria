#!/usr/bin/env python
# coding: utf-8
"""Canonical tabular representation registry for QSAR training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


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

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
}

AUTOMATIC_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.automatic
)
SUPPORTED_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(TABULAR_REPRESENTATIONS)


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
