#!/usr/bin/env python
# coding: utf-8
"""Deterministic preparation of tabular molecular representations.

This module is the only orchestration layer allowed to turn molecular CSVs into
feature matrices for tabular prediction backends.  Backends consume prepared
matrices and never generate molecular features themselves.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Annotated, Any, Dict, Iterator, List, Literal, Optional, Sequence, Union

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rdkit import rdBase

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import (
    resolve_smiles_column_name,
    standardize_smiles_column,
)
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit

from .backend import InvalidPredictionInputError, PredictionModelRecord
from .tabular_representations import (
    TABULAR_REPRESENTATION_CONTRACT_VERSION,
    MorganBinaryFingerprint,
    MorganCountFingerprint,
    PrecomputedFeatures,
    RDKitDescriptors,
    TabularRepresentationComponent,
    get_tabular_representation,
)

FEATURE_GENERATOR_VERSION = "1.0"
SMILES_NORMALIZATION_CONTRACT = "qsaria_standardized_smiles_v1"
QSAR_ROW_ID_COLUMN = "__qsar_row_id"
DEFAULT_TABULAR_CACHE_ROOT = (Path(".files") / "cache" / "tabular_representations" / "v1").resolve()


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class RDKitDescriptorsContract(_StrictModel):
    kind: Literal["rdkit_descriptors"] = "rdkit_descriptors"
    descriptor_set: Literal["all"] = "all"


class MorganBinaryFingerprintContract(_StrictModel):
    kind: Literal["morgan_binary"] = "morgan_binary"
    radius: Literal[2] = 2
    n_bits: Literal[2048] = 2048
    feature_prefix: Literal["fp_"] = "fp_"


class MorganCountFingerprintContract(_StrictModel):
    kind: Literal["morgan_count"] = "morgan_count"
    radius: Literal[2] = 2
    n_bits: Literal[2048] = 2048
    feature_prefix: Literal["cfp_"] = "cfp_"


class PrecomputedFeaturesContract(_StrictModel):
    kind: Literal["precomputed"] = "precomputed"
    feature_columns: List[str] = Field(min_length=1)
    categorical_feature_columns: List[str] = Field(default_factory=list)


TabularComponentContract = Annotated[
    Union[
        RDKitDescriptorsContract,
        MorganBinaryFingerprintContract,
        MorganCountFingerprintContract,
        PrecomputedFeaturesContract,
    ],
    Field(discriminator="kind"),
]


def _recipe_payload(
    *,
    kind: str,
    representation_name: str,
    components: Sequence[TabularComponentContract | Dict[str, Any]],
) -> Dict[str, Any]:
    serialized = [
        component.model_dump(mode="json") if isinstance(component, BaseModel) else dict(component)
        for component in components
    ]
    return {
        "schema_version": TABULAR_REPRESENTATION_CONTRACT_VERSION,
        "kind": kind,
        "representation_name": representation_name,
        "components": serialized,
        "smiles_normalization_contract": SMILES_NORMALIZATION_CONTRACT,
        "feature_generator_version": (
            FEATURE_GENERATOR_VERSION if kind == "generated" else "not_applicable"
        ),
    }


def representation_recipe_signature(
    *,
    kind: str,
    representation_name: str,
    components: Sequence[TabularComponentContract | Dict[str, Any]],
) -> str:
    encoded = json.dumps(
        _recipe_payload(
            kind=kind,
            representation_name=representation_name,
            components=components,
        ),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class TabularRepresentationContract(_StrictModel):
    """The only accepted representation source for a tabular model."""

    schema_version: Literal["1.0"] = TABULAR_REPRESENTATION_CONTRACT_VERSION
    kind: Literal["generated", "precomputed"]
    representation_name: str
    components: List[TabularComponentContract] = Field(min_length=1)
    recipe_signature: str
    feature_columns: List[str] = Field(min_length=1)
    categorical_feature_columns: List[str] = Field(default_factory=list)
    smiles_normalization_contract: Literal["qsaria_standardized_smiles_v1"] = (
        SMILES_NORMALIZATION_CONTRACT
    )
    feature_generator_version: str = FEATURE_GENERATOR_VERSION
    rdkit_version: str
    qsaria_version: str

    @field_validator(
        "schema_version",
        "representation_name",
        "recipe_signature",
        "feature_generator_version",
        "rdkit_version",
        "qsaria_version",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("Tabular representation contract strings cannot be empty.")
        return normalized

    @field_validator("feature_columns", "categorical_feature_columns")
    @classmethod
    def _unique_columns(cls, values: List[str]) -> List[str]:
        normalized = [str(value).strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Feature column names cannot be empty.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Feature column names must be unique.")
        return normalized

    @model_validator(mode="after")
    def _validate_contract(self) -> "TabularRepresentationContract":
        expected_signature = representation_recipe_signature(
            kind=self.kind,
            representation_name=self.representation_name,
            components=self.components,
        )
        if self.recipe_signature != expected_signature:
            raise ValueError(
                "Tabular representation recipe_signature does not match its components."
            )
        categorical = set(self.categorical_feature_columns)
        if not categorical.issubset(set(self.feature_columns)):
            raise ValueError("categorical_feature_columns must be a subset of feature_columns.")
        if self.kind == "generated":
            if any(isinstance(item, PrecomputedFeaturesContract) for item in self.components):
                raise ValueError("Generated representations cannot contain precomputed features.")
            if self.categorical_feature_columns:
                raise ValueError(
                    "Generated molecular representations cannot contain categorical features."
                )
        else:
            if len(self.components) != 1 or not isinstance(
                self.components[0], PrecomputedFeaturesContract
            ):
                raise ValueError(
                    "Precomputed representations require exactly one precomputed component."
                )
            if self.feature_generator_version != "not_applicable":
                raise ValueError(
                    "Precomputed representations require "
                    "feature_generator_version='not_applicable'."
                )
            component = self.components[0]
            if component.feature_columns != self.feature_columns:
                raise ValueError("Precomputed component feature_columns must match the contract.")
            if component.categorical_feature_columns != self.categorical_feature_columns:
                raise ValueError(
                    "Precomputed component categorical_feature_columns must match the contract."
                )
        return self


class TabularPreparationRequest(_StrictModel):
    input_csv: str
    output_dir: str
    kind: Literal["generated", "precomputed"]
    representation_name: str
    smiles_column: str = "smiles"
    target_columns: List[str] = Field(default_factory=list)
    base_columns_to_keep: List[str] = Field(default_factory=list)
    expected_feature_columns: List[str] = Field(default_factory=list)
    categorical_feature_columns: List[str] = Field(default_factory=list)
    cache_root: Optional[str] = None
    n_jobs: int = Field(default=1, ge=1)
    purpose: Literal[
        "training",
        "inference",
        "external_evaluation",
        "ensemble",
        "applicability_domain",
    ] = "training"


class TabularPreparationResult(_StrictModel):
    prepared_csv: str
    representation_name: str
    feature_columns: List[str]
    categorical_feature_columns: List[str]
    row_count: int
    source_hash: str
    recipe_signature: str
    component_artifacts: List[Dict[str, Any]]
    cache_root: str
    cache_key: str
    cache_status: Literal["generated", "reused_from_cache", "not_applicable"]
    cache_hits: int
    cache_misses: int
    duration_seconds: float
    contract: TabularRepresentationContract


def _current_qsaria_version() -> str:
    try:
        return importlib_metadata.version("cs_copilot")
    except importlib_metadata.PackageNotFoundError:
        return "0.4.0"


def current_rdkit_version() -> str:
    return str(rdBase.rdkitVersion)


def _component_contract(
    component: TabularRepresentationComponent,
) -> TabularComponentContract:
    if isinstance(component, RDKitDescriptors):
        return RDKitDescriptorsContract(descriptor_set=component.descriptor_set)
    if isinstance(component, MorganBinaryFingerprint):
        return MorganBinaryFingerprintContract(
            radius=component.radius,
            n_bits=component.n_bits,
            feature_prefix=component.feature_prefix,
        )
    if isinstance(component, MorganCountFingerprint):
        return MorganCountFingerprintContract(
            radius=component.radius,
            n_bits=component.n_bits,
            feature_prefix=component.feature_prefix,
        )
    if isinstance(component, PrecomputedFeatures):
        return PrecomputedFeaturesContract(
            feature_columns=list(component.feature_columns),
            categorical_feature_columns=list(component.categorical_feature_columns),
        )
    raise TypeError(f"Unsupported tabular representation component: {type(component)!r}")


def _read_csv(path: str) -> pd.DataFrame:
    local = Path(path).expanduser()
    if local.exists():
        return pd.read_csv(local)
    with S3.open(path, "r") as fh:
        return pd.read_csv(fh)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_key(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


_LOCK_GUARD = threading.Lock()
_PROCESS_LOCKS: Dict[str, threading.RLock] = {}


@contextmanager
def _cache_lock(cache_root: Path, key: str) -> Iterator[None]:
    with _LOCK_GUARD:
        process_lock = _PROCESS_LOCKS.setdefault(key, threading.RLock())
    with process_lock:
        lock_path = cache_root / "locks" / f"{key}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):
                fcntl = None  # type: ignore[assignment]
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def require_tabular_model_contract(
    record: PredictionModelRecord,
) -> TabularRepresentationContract:
    payload = record.tabular_representation_contract
    if not payload:
        raise InvalidPredictionInputError(
            f"Tabular model `{record.model_id}` has no tabular_representation_contract. "
            "This pre-0.4.0 model format is unsupported; retrain and persist the model."
        )
    schema_version = str(payload.get("schema_version") or "")
    if schema_version != TABULAR_REPRESENTATION_CONTRACT_VERSION:
        raise InvalidPredictionInputError(
            "Tabular representation contract is incompatible with this Qsaria "
            f"runtime: expected schema {TABULAR_REPRESENTATION_CONTRACT_VERSION}, "
            f"model provides {schema_version or 'missing'}. Retrain the model."
        )
    if (
        payload.get("kind") == "generated"
        and payload.get("feature_generator_version") != FEATURE_GENERATOR_VERSION
    ):
        raise InvalidPredictionInputError(
            "Tabular representation generator is incompatible with the persisted "
            f"model: expected {payload.get('feature_generator_version') or 'missing'}, "
            f"installed {FEATURE_GENERATOR_VERSION}. Retrain the model."
        )
    try:
        return TabularRepresentationContract.model_validate(payload)
    except Exception as exc:
        raise InvalidPredictionInputError(
            f"Tabular model `{record.model_id}` has an invalid "
            f"tabular_representation_contract: {exc}"
        ) from exc


def validate_tabular_runtime(contract: TabularRepresentationContract) -> None:
    if contract.kind == "precomputed":
        return
    runtime_rdkit = current_rdkit_version()
    if contract.rdkit_version != runtime_rdkit:
        raise InvalidPredictionInputError(
            "Tabular representation runtime is incompatible with the persisted model: "
            f"expected RDKit {contract.rdkit_version}, installed {runtime_rdkit}. "
            "Retrain the model with the installed runtime."
        )
    if contract.feature_generator_version != FEATURE_GENERATOR_VERSION:
        raise InvalidPredictionInputError(
            "Tabular representation generator is incompatible with the persisted model: "
            f"expected {contract.feature_generator_version}, installed "
            f"{FEATURE_GENERATOR_VERSION}. Retrain the model."
        )
    try:
        runtime_spec = get_tabular_representation(contract.representation_name)
    except ValueError as exc:
        raise InvalidPredictionInputError(
            "The persisted tabular representation is not registered in this Qsaria "
            f"runtime: {contract.representation_name!r}. Retrain the model."
        ) from exc
    runtime_components = [_component_contract(component) for component in runtime_spec.components]
    runtime_signature = representation_recipe_signature(
        kind="generated",
        representation_name=runtime_spec.name,
        components=runtime_components,
    )
    if contract.recipe_signature != runtime_signature:
        raise InvalidPredictionInputError(
            "Tabular representation recipe is incompatible with the persisted model: "
            f"expected {contract.recipe_signature}, installed {runtime_signature}. "
            "Retrain the model."
        )


class TabularFeaturePreparationService:
    """Prepare generated or precomputed tabular matrices for every consumer."""

    def __init__(
        self,
        *,
        cache_root: str | Path | None = None,
        feature_toolkit: MolecularFeatureToolkit | None = None,
    ) -> None:
        self.cache_root = Path(cache_root or DEFAULT_TABULAR_CACHE_ROOT).expanduser().resolve()
        self.feature_toolkit = feature_toolkit or MolecularFeatureToolkit()

    def _cache_is_valid(
        self,
        *,
        csv_path: Path,
        metadata_path: Path,
        cache_key: str,
        expected_source_hash: str,
    ) -> bool:
        if not csv_path.exists() or not metadata_path.exists():
            return False
        try:
            metadata = json.loads(metadata_path.read_text())
            cached_frame = pd.read_csv(csv_path)
            header = list(cached_frame.columns)
            row_count = len(cached_frame)
        except Exception:
            return False
        return bool(
            metadata.get("schema_version") == "1.0"
            and metadata.get("cache_key") == cache_key
            and metadata.get("source_hash") == expected_source_hash
            and metadata.get("columns") == header
            and int(metadata.get("row_count", -1)) == row_count
            and metadata.get("csv_hash") == _sha256_file(csv_path)
        )

    def _generate_component(
        self,
        *,
        component: TabularComponentContract,
        base_csv: Path,
        source_hash: str,
        columns_to_keep: List[str],
        cache_root: Path,
        n_jobs: int,
    ) -> tuple[Path, Dict[str, Any], bool]:
        component_payload = component.model_dump(mode="json")
        key = _cache_key(
            {
                "kind": "component",
                "source_hash": source_hash,
                "component": component_payload,
                "columns_to_keep": columns_to_keep,
                "rdkit_version": current_rdkit_version(),
                "feature_generator_version": FEATURE_GENERATOR_VERSION,
            }
        )
        component_dir = cache_root / "components"
        csv_path = component_dir / f"{component.kind}_{key}.csv"
        metadata_path = component_dir / f"{component.kind}_{key}.json"
        with _cache_lock(cache_root, f"component-{key}"):
            if self._cache_is_valid(
                csv_path=csv_path,
                metadata_path=metadata_path,
                cache_key=key,
                expected_source_hash=source_hash,
            ):
                return csv_path, json.loads(metadata_path.read_text()), True

            component_dir.mkdir(parents=True, exist_ok=True)
            temporary_csv = component_dir / f".{component.kind}_{key}.{os.getpid()}.csv"
            if isinstance(component, RDKitDescriptorsContract):
                result = self.feature_toolkit.smiles_to_rdkit_descriptors(
                    input_csv=str(base_csv),
                    smiles_column="smiles",
                    output_csv=str(temporary_csv),
                    descriptor_set=component.descriptor_set,
                    include_input_columns=True,
                    input_columns_to_keep=columns_to_keep,
                    n_jobs=n_jobs,
                )
            elif isinstance(component, MorganBinaryFingerprintContract):
                result = self.feature_toolkit.smiles_to_morgan_fingerprints(
                    input_csv=str(base_csv),
                    smiles_column="smiles",
                    output_csv=str(temporary_csv),
                    radius=component.radius,
                    n_bits=component.n_bits,
                    include_input_columns=True,
                    input_columns_to_keep=columns_to_keep,
                    feature_prefix=component.feature_prefix,
                    fingerprint_kind="binary",
                    n_jobs=n_jobs,
                )
            elif isinstance(component, MorganCountFingerprintContract):
                result = self.feature_toolkit.smiles_to_morgan_fingerprints(
                    input_csv=str(base_csv),
                    smiles_column="smiles",
                    output_csv=str(temporary_csv),
                    radius=component.radius,
                    n_bits=component.n_bits,
                    include_input_columns=True,
                    input_columns_to_keep=columns_to_keep,
                    feature_prefix=component.feature_prefix,
                    fingerprint_kind="count",
                    n_jobs=n_jobs,
                )
            else:
                raise TypeError("Precomputed components are not generated.")
            os.replace(temporary_csv, csv_path)
            frame = pd.read_csv(csv_path)
            metadata = {
                "schema_version": "1.0",
                "cache_key": key,
                "source_hash": source_hash,
                "csv_hash": _sha256_file(csv_path),
                "columns": list(frame.columns),
                "row_count": int(len(frame)),
                "component": component_payload,
                "result": result,
            }
            _atomic_write_json(metadata_path, metadata)
            return csv_path, metadata, False

    def prepare(self, request: TabularPreparationRequest) -> TabularPreparationResult:
        started_at = time.monotonic()
        output_dir = Path(request.output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_root = (
            Path(request.cache_root).expanduser().resolve()
            if request.cache_root
            else self.cache_root
        )
        cache_root.mkdir(parents=True, exist_ok=True)

        source = _read_csv(request.input_csv)
        resolved_smiles = resolve_smiles_column_name(source, request.smiles_column)
        normalized = standardize_smiles_column(source.copy(), resolved_smiles)
        if resolved_smiles != "smiles":
            normalized["smiles"] = normalized[resolved_smiles]
            normalized = normalized.drop(columns=[resolved_smiles])
        invalid_rows = int(normalized["smiles"].isna().sum())
        if invalid_rows:
            raise InvalidPredictionInputError(
                f"Cannot prepare tabular representation: {invalid_rows} row(s) contain "
                "invalid or missing standardized SMILES."
            )
        normalized[QSAR_ROW_ID_COLUMN] = range(len(normalized))

        requested_base_columns = list(
            dict.fromkeys(
                [
                    QSAR_ROW_ID_COLUMN,
                    "smiles",
                    *request.base_columns_to_keep,
                    *request.target_columns,
                ]
            )
        )
        missing_base = [column for column in requested_base_columns if column not in normalized]
        if missing_base:
            raise InvalidPredictionInputError(
                f"Tabular preparation input is missing base columns: {missing_base}"
            )
        base = normalized[requested_base_columns].copy()
        base_csv = output_dir / "tabular_feature_base.csv"
        base.to_csv(base_csv, index=False)
        source_hash = _sha256_file(base_csv)

        if request.kind == "precomputed":
            expected = list(request.expected_feature_columns)
            missing = [column for column in expected if column not in normalized.columns]
            if missing:
                raise InvalidPredictionInputError(
                    f"Precomputed tabular input is missing feature columns: {missing}"
                )
            prepared_columns = [
                *requested_base_columns,
                *[column for column in expected if column not in requested_base_columns],
            ]
            prepared = normalized[prepared_columns].copy()
            prepared_csv = output_dir / "precomputed_tabular_features.csv"
            prepared.to_csv(prepared_csv, index=False)
            component = PrecomputedFeaturesContract(
                feature_columns=expected,
                categorical_feature_columns=list(request.categorical_feature_columns),
            )
            signature = representation_recipe_signature(
                kind="precomputed",
                representation_name=request.representation_name,
                components=[component],
            )
            contract = TabularRepresentationContract(
                kind="precomputed",
                representation_name=request.representation_name,
                components=[component],
                recipe_signature=signature,
                feature_columns=expected,
                categorical_feature_columns=list(request.categorical_feature_columns),
                feature_generator_version="not_applicable",
                rdkit_version="not_applicable",
                qsaria_version=_current_qsaria_version(),
            )
            return TabularPreparationResult(
                prepared_csv=str(prepared_csv),
                representation_name=request.representation_name,
                feature_columns=expected,
                categorical_feature_columns=list(request.categorical_feature_columns),
                row_count=int(len(prepared)),
                source_hash=source_hash,
                recipe_signature=signature,
                component_artifacts=[],
                cache_root=str(cache_root),
                cache_key="not_applicable",
                cache_status="not_applicable",
                cache_hits=0,
                cache_misses=0,
                duration_seconds=round(time.monotonic() - started_at, 3),
                contract=contract,
            )

        spec = get_tabular_representation(request.representation_name)
        components = [_component_contract(component) for component in spec.components]
        signature = representation_recipe_signature(
            kind="generated",
            representation_name=spec.name,
            components=components,
        )
        component_paths: List[str] = []
        component_artifacts: List[Dict[str, Any]] = []
        cache_hits = 0
        for component in components:
            component_path, metadata, cache_hit = self._generate_component(
                component=component,
                base_csv=base_csv,
                source_hash=source_hash,
                columns_to_keep=requested_base_columns,
                cache_root=cache_root,
                n_jobs=request.n_jobs,
            )
            component_paths.append(str(component_path))
            component_artifacts.append(
                {
                    "kind": component.kind,
                    "output_csv": str(component_path),
                    "cache_key": metadata["cache_key"],
                    "cache_status": "reused_from_cache" if cache_hit else "generated",
                    "result": dict(metadata.get("result") or {}),
                }
            )
            cache_hits += int(cache_hit)

        assembly_key = _cache_key(
            {
                "kind": "assembled",
                "source_hash": source_hash,
                "recipe_signature": signature,
                "component_keys": [item["cache_key"] for item in component_artifacts],
                "base_columns": requested_base_columns,
            }
        )
        assembled_dir = cache_root / "assembled"
        assembled_csv = assembled_dir / f"{spec.name}_{assembly_key}.csv"
        assembled_metadata = assembled_dir / f"{spec.name}_{assembly_key}.json"
        assembly_hit = False
        with _cache_lock(cache_root, f"assembled-{assembly_key}"):
            if self._cache_is_valid(
                csv_path=assembled_csv,
                metadata_path=assembled_metadata,
                cache_key=assembly_key,
                expected_source_hash=source_hash,
            ):
                assembly_hit = True
            else:
                assembled_dir.mkdir(parents=True, exist_ok=True)
                temporary_csv = assembled_dir / f".{spec.name}_{assembly_key}.{os.getpid()}.csv"
                self.feature_toolkit.build_tabular_qsar_dataset(
                    base_csv=str(base_csv),
                    output_csv=str(temporary_csv),
                    feature_csvs=component_paths,
                    join_on=[QSAR_ROW_ID_COLUMN],
                    base_columns_to_keep=requested_base_columns,
                    drop_duplicate_feature_columns=True,
                    canonicalize_smiles_join=False,
                )
                os.replace(temporary_csv, assembled_csv)
                assembled_frame = pd.read_csv(assembled_csv)
                _atomic_write_json(
                    assembled_metadata,
                    {
                        "schema_version": "1.0",
                        "cache_key": assembly_key,
                        "source_hash": source_hash,
                        "csv_hash": _sha256_file(assembled_csv),
                        "columns": list(assembled_frame.columns),
                        "row_count": int(len(assembled_frame)),
                        "recipe_signature": signature,
                        "component_keys": [item["cache_key"] for item in component_artifacts],
                    },
                )

        frame = pd.read_csv(assembled_csv)
        generated_feature_columns = [
            column for column in frame.columns if column not in requested_base_columns
        ]
        expected = list(request.expected_feature_columns)
        if expected:
            missing = [column for column in expected if column not in frame.columns]
            if missing:
                raise InvalidPredictionInputError(
                    f"Generated tabular representation is missing expected features: {missing}"
                )
            ordered_columns = [
                *requested_base_columns,
                *[column for column in expected if column not in requested_base_columns],
            ]
            frame = frame[ordered_columns]
            prepared_csv = output_dir / "prepared_tabular_features.csv"
            frame.to_csv(prepared_csv, index=False)
            feature_columns = expected
        else:
            prepared_csv = assembled_csv
            feature_columns = generated_feature_columns

        contract = TabularRepresentationContract(
            kind="generated",
            representation_name=spec.name,
            components=components,
            recipe_signature=signature,
            feature_columns=feature_columns,
            categorical_feature_columns=[],
            rdkit_version=current_rdkit_version(),
            qsaria_version=_current_qsaria_version(),
        )
        total_hits = cache_hits + int(assembly_hit)
        total_items = len(components) + 1
        return TabularPreparationResult(
            prepared_csv=str(prepared_csv),
            representation_name=spec.name,
            feature_columns=feature_columns,
            categorical_feature_columns=[],
            row_count=int(len(frame)),
            source_hash=source_hash,
            recipe_signature=signature,
            component_artifacts=component_artifacts,
            cache_root=str(cache_root),
            cache_key=assembly_key,
            cache_status=("reused_from_cache" if total_hits == total_items else "generated"),
            cache_hits=total_hits,
            cache_misses=total_items - total_hits,
            duration_seconds=round(time.monotonic() - started_at, 3),
            contract=contract,
        )

    def prepare_for_model(
        self,
        *,
        input_csv: str,
        output_dir: str,
        record: PredictionModelRecord,
        purpose: Literal[
            "inference",
            "external_evaluation",
            "ensemble",
            "applicability_domain",
        ],
        smiles_column: str = "smiles",
        base_columns_to_keep: Optional[List[str]] = None,
        n_jobs: int = 1,
    ) -> TabularPreparationResult:
        contract = require_tabular_model_contract(record)
        validate_tabular_runtime(contract)
        result = self.prepare(
            TabularPreparationRequest(
                input_csv=input_csv,
                output_dir=output_dir,
                kind=contract.kind,
                representation_name=contract.representation_name,
                smiles_column=smiles_column,
                base_columns_to_keep=list(base_columns_to_keep or []),
                expected_feature_columns=list(contract.feature_columns),
                categorical_feature_columns=list(contract.categorical_feature_columns),
                n_jobs=n_jobs,
                purpose=purpose,
            )
        )
        if result.recipe_signature != contract.recipe_signature:
            raise InvalidPredictionInputError(
                "Generated tabular representation recipe does not match the persisted model. "
                "Retrain the model with the installed Qsaria runtime."
            )
        return result
