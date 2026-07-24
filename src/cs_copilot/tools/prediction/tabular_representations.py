#!/usr/bin/env python
# coding: utf-8
"""Canonical tabular representation registry for QSAR training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

TABULAR_REPRESENTATION_CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class RDKitDescriptors:
    """One deterministic RDKit descriptor component."""

    kind: str = "rdkit_descriptors"
    descriptor_set: str = "all"


@dataclass(frozen=True)
class MorganBinaryFingerprint:
    """One deterministic binary Morgan fingerprint component."""

    kind: str = "morgan_binary"
    radius: int = 2
    n_bits: int = 2048
    feature_prefix: str = "fp_"


@dataclass(frozen=True)
class MorganCountFingerprint:
    """One deterministic count-based Morgan fingerprint component."""

    kind: str = "morgan_count"
    radius: int = 2
    n_bits: int = 2048
    feature_prefix: str = "cfp_"


@dataclass(frozen=True)
class PrecomputedFeatures:
    """Explicit precomputed tabular features."""

    feature_columns: Tuple[str, ...]
    categorical_feature_columns: Tuple[str, ...] = ()
    kind: str = "precomputed"


TabularRepresentationComponent = Union[
    RDKitDescriptors,
    MorganBinaryFingerprint,
    MorganCountFingerprint,
    PrecomputedFeatures,
]


@dataclass(frozen=True)
class TabularRepresentationSpec:
    """Declarative description of a reusable tabular molecular representation."""

    name: str
    display_name: str
    components: Tuple[TabularRepresentationComponent, ...] = ()
    automatic: bool = False

    @property
    def use_morgan_binary(self) -> bool:
        return any(isinstance(component, MorganBinaryFingerprint) for component in self.components)

    @property
    def use_morgan_count(self) -> bool:
        return any(isinstance(component, MorganCountFingerprint) for component in self.components)

    @property
    def use_rdkit(self) -> bool:
        return any(isinstance(component, RDKitDescriptors) for component in self.components)

    @property
    def descriptor_set(self) -> Optional[str]:
        component = next(
            (component for component in self.components if isinstance(component, RDKitDescriptors)),
            None,
        )
        return component.descriptor_set if component is not None else None

    @property
    def description(self) -> str:
        descriptions = {
            RDKitDescriptors: "Full RDKit descriptor table",
            MorganBinaryFingerprint: "Binary ECFP/Morgan fingerprint bits",
            MorganCountFingerprint: "Count-based ECFP/Morgan fingerprint bins",
            PrecomputedFeatures: "Explicit precomputed tabular features",
        }
        return " + ".join(descriptions[type(component)] for component in self.components) + "."

    def as_dict(self) -> Dict[str, Any]:
        # Preserve the existing public capability shape while exposing the new
        # composable recipe additively.
        return {
            "name": self.name,
            "display_name": self.display_name,
            "use_morgan_binary": self.use_morgan_binary,
            "use_morgan_count": self.use_morgan_count,
            "use_rdkit": self.use_rdkit,
            "descriptor_set": self.descriptor_set,
            "automatic": self.automatic,
            "description": self.description,
            "components": [asdict(component) for component in self.components],
        }


TABULAR_REPRESENTATIONS: Dict[str, TabularRepresentationSpec] = {
    "rdkit_all": TabularRepresentationSpec(
        name="rdkit_all",
        display_name="RDKit all descriptors",
        components=(RDKitDescriptors(descriptor_set="all"),),
        automatic=True,
    ),
    "morgan_only": TabularRepresentationSpec(
        name="morgan_only",
        display_name="Morgan binary fingerprint",
        components=(MorganBinaryFingerprint(),),
        automatic=True,
    ),
    "morgan_count_only": TabularRepresentationSpec(
        name="morgan_count_only",
        display_name="Morgan count fingerprint",
        components=(MorganCountFingerprint(),),
        automatic=True,
    ),
    "morgan_binary_count_rdkit_all": TabularRepresentationSpec(
        name="morgan_binary_count_rdkit_all",
        display_name="Morgan binary + count + RDKit all",
        components=(
            MorganBinaryFingerprint(),
            MorganCountFingerprint(),
            RDKitDescriptors(descriptor_set="all"),
        ),
        automatic=False,
    ),
    "morgan_rdkit_all": TabularRepresentationSpec(
        name="morgan_rdkit_all",
        display_name="Morgan binary + RDKit all",
        components=(
            MorganBinaryFingerprint(),
            RDKitDescriptors(descriptor_set="all"),
        ),
        automatic=False,
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
