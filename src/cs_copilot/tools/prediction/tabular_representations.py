#!/usr/bin/env python
# coding: utf-8
"""Canonical tabular representation registry for QSAR training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class TabularRepresentationSpec:
    """Declarative description of a reusable tabular molecular representation."""

    name: str
    display_name: str
    use_morgan_binary: bool = False
    use_morgan_count: bool = False
    use_chemeleon: bool = False
    use_rdkit: bool = False
    descriptor_set: Optional[str] = None
    automatic: bool = False
    tabicl_automatic: bool = False
    tabicl_compatible: bool = True
    tabicl_preferred: bool = False
    high_dimensional: bool = False
    legacy: bool = False
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


TABULAR_REPRESENTATIONS: Dict[str, TabularRepresentationSpec] = {
    "chemeleon_rdkit_all": TabularRepresentationSpec(
        name="chemeleon_rdkit_all",
        display_name="CheMeleon fingerprint + RDKit all descriptors",
        use_chemeleon=True,
        use_rdkit=True,
        descriptor_set="all",
        tabicl_automatic=True,
        tabicl_preferred=True,
        description=(
            "Preferred TabICL representation combining CheMeleon learned molecular "
            "embeddings with the full RDKit 2D descriptor table."
        ),
    ),
    "rdkit_all": TabularRepresentationSpec(
        name="rdkit_all",
        display_name="RDKit all descriptors",
        use_rdkit=True,
        descriptor_set="all",
        automatic=True,
        tabicl_automatic=True,
        description="Full RDKit descriptor table.",
    ),
    "morgan_only": TabularRepresentationSpec(
        name="morgan_only",
        display_name="Morgan binary fingerprint",
        use_morgan_binary=True,
        automatic=True,
        tabicl_compatible=False,
        high_dimensional=True,
        description="Binary ECFP/Morgan fingerprint bits.",
    ),
    "morgan_count_only": TabularRepresentationSpec(
        name="morgan_count_only",
        display_name="Morgan count fingerprint",
        use_morgan_count=True,
        automatic=True,
        tabicl_compatible=False,
        high_dimensional=True,
        description="Count-based ECFP/Morgan fingerprint bins.",
    ),
    "morgan_binary_count_rdkit_all": TabularRepresentationSpec(
        name="morgan_binary_count_rdkit_all",
        display_name="Morgan binary + count + RDKit all",
        use_morgan_binary=True,
        use_morgan_count=True,
        use_rdkit=True,
        descriptor_set="all",
        automatic=True,
        tabicl_compatible=False,
        high_dimensional=True,
        description="Modern complete tabular pack combining binary fingerprints, count fingerprints, and RDKit descriptors.",
    ),
    "morgan_rdkit_all": TabularRepresentationSpec(
        name="morgan_rdkit_all",
        display_name="Morgan binary + RDKit all",
        use_morgan_binary=True,
        use_rdkit=True,
        descriptor_set="all",
        automatic=False,
        tabicl_compatible=False,
        high_dimensional=True,
        description="Historical high-validation combined representation retained for compatibility and explicit use.",
    ),
    "rdkit_basic_only": TabularRepresentationSpec(
        name="rdkit_basic_only",
        display_name="Legacy RDKit basic descriptors",
        use_rdkit=True,
        descriptor_set="basic",
        legacy=True,
        description="Legacy lightweight descriptor set. Use only when explicitly requested.",
    ),
    "morgan_rdkit_basic": TabularRepresentationSpec(
        name="morgan_rdkit_basic",
        display_name="Legacy Morgan binary + RDKit basic",
        use_morgan_binary=True,
        use_rdkit=True,
        descriptor_set="basic",
        tabicl_compatible=False,
        high_dimensional=True,
        legacy=True,
        description="Legacy local-light combined representation. Use only when explicitly requested.",
    ),
}

AUTOMATIC_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.automatic
)
LEGACY_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.legacy
)
TABICL_AUTOMATIC_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.tabicl_automatic
)
TABICL_SUPPORTED_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.tabicl_compatible
)
HIGH_DIMENSIONAL_TABULAR_REPRESENTATION_NAMES: Tuple[str, ...] = tuple(
    name for name, spec in TABULAR_REPRESENTATIONS.items() if spec.high_dimensional
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


def _normalized_backend_name(backend_name: Optional[str]) -> str:
    return str(backend_name or "").strip().lower()


def validate_tabular_representation_for_backend(
    backend_name: str,
    representation_name: str,
) -> TabularRepresentationSpec:
    """Return a representation spec after enforcing backend-specific policy."""
    spec = get_tabular_representation(representation_name)
    if _normalized_backend_name(backend_name) == "tabicl" and not spec.tabicl_compatible:
        raise ValueError(
            f"TabICL should not be used with high-dimensional representation "
            f"{representation_name!r}. Use 'chemeleon_rdkit_all' as the preferred "
            "TabICL representation, or 'rdkit_all' for an RDKit-only low-dimensional "
            "fallback."
        )
    return spec


def default_tabular_representation_for_protocol(
    protocol: str,
    *,
    training_profile: Optional[str] = None,
    backend_name: Optional[str] = None,
) -> str:
    """Return the single representation used when a protocol is not comparative."""
    if _normalized_backend_name(backend_name) == "tabicl":
        return "chemeleon_rdkit_all"
    normalized = str(protocol or "").strip().lower()
    if normalized == "fast_local":
        return "rdkit_all"
    # Keep the historical strong default for explicit single-model tabular training.
    if training_profile == "heavy_validation":
        return "morgan_rdkit_all"
    return "morgan_only"


def automatic_tabular_representations(
    *,
    include_legacy: bool = False,
    backend_name: Optional[str] = None,
) -> List[TabularRepresentationSpec]:
    """Return representation specs used by modern comparative tabular campaigns."""
    names: Iterable[str]
    if _normalized_backend_name(backend_name) == "tabicl":
        names = TABICL_AUTOMATIC_TABULAR_REPRESENTATION_NAMES
    elif include_legacy:
        names = [*AUTOMATIC_TABULAR_REPRESENTATION_NAMES, *LEGACY_TABULAR_REPRESENTATION_NAMES]
    else:
        names = AUTOMATIC_TABULAR_REPRESENTATION_NAMES
    return [get_tabular_representation(name) for name in names]


def tabular_candidate_id(backend_name: str, representation_name: str) -> str:
    """Return the canonical candidate identifier for a backend/representation pair."""
    return f"{str(backend_name).strip().lower()}_{str(representation_name).strip().lower()}"


def tabular_candidates_for_backend(
    backend_name: str,
    *,
    include_legacy: bool = False,
    representation_names: Optional[List[str]] = None,
    single_default_protocol: Optional[str] = None,
    training_profile: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build candidate dictionaries for a tabular backend."""
    normalized_backend = str(backend_name).strip().lower()
    if representation_names:
        specs = [
            validate_tabular_representation_for_backend(normalized_backend, name)
            for name in representation_names
        ]
    elif single_default_protocol:
        specs = [
            validate_tabular_representation_for_backend(
                normalized_backend,
                default_tabular_representation_for_protocol(
                    single_default_protocol,
                    training_profile=training_profile,
                    backend_name=normalized_backend,
                ),
            )
        ]
    else:
        specs = automatic_tabular_representations(
            include_legacy=include_legacy,
            backend_name=normalized_backend,
        )

    return [
        {
            "candidate_id": tabular_candidate_id(normalized_backend, spec.name),
            "backend_name": normalized_backend,
            "representation_name": spec.name,
            "representation_display_name": spec.display_name,
            "representation_legacy": spec.legacy,
            "representation_automatic": (
                spec.tabicl_automatic if normalized_backend == "tabicl" else spec.automatic
            ),
        }
        for spec in specs
    ]


def describe_tabular_representations() -> Dict[str, Dict[str, Any]]:
    """Return all registered representation specs as dictionaries."""
    return {name: spec.as_dict() for name, spec in TABULAR_REPRESENTATIONS.items()}
