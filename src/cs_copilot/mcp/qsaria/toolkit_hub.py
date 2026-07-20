"""Lazy, model-free construction of the scientific Qsaria toolkits.

The MCP Qsaria profile deliberately reuses the production toolkit classes.  It
does not construct an Agno ``Agent`` or ``Team`` and it never attaches a model;
the small :class:`~cs_copilot.mcp.context.MCPAgentContext` used at invocation
time is supplied by :mod:`cs_copilot.mcp.qsaria.experiments`.

Catalog-bearing toolkits share one catalog object.  Several legacy read
methods request ``refresh_from_internal_store(persist=True)`` even though the
operation is conceptually read-only.  ``_ReadSafeCatalog`` preserves the
refresh while suppressing that incidental write unless the dedicated Qsaria
adapter has explicitly opened a catalog-write scope.
"""

from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Literal

ToolkitName = Literal[
    "curation",
    "training",
    "registry",
    "inference",
    "ensemble",
    "benchmark",
    "activity_cliffs",
]

_CATALOG_WRITE_ENABLED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "qsaria_mcp_catalog_write_enabled",
    default=False,
)


def _read_safe_catalog_type():
    """Build the catalog subclass lazily to keep module import lightweight."""

    from cs_copilot.tools.prediction.catalog import PredictionModelCatalog

    class _ReadSafeCatalog(PredictionModelCatalog):
        def refresh_from_internal_store(self, persist: bool = False) -> int:
            return super().refresh_from_internal_store(
                persist=bool(persist and _CATALOG_WRITE_ENABLED.get())
            )

    return _ReadSafeCatalog


class ToolkitHub:
    """Own the lazily-created toolkit graph used by Qsaria MCP tools."""

    def __init__(self) -> None:
        self._instances: dict[str, Any] = {}
        self._catalog: Any | None = None
        self._backends: dict[str, Any] | None = None
        self._lock = threading.RLock()

    def get(self, name: ToolkitName) -> Any:
        """Return one shared toolkit instance, constructing dependencies lazily."""

        with self._lock:
            instance = self._instances.get(name)
            if instance is not None:
                return instance

            builders = {
                "curation": self._build_curation,
                "training": self._build_training,
                "registry": self._build_registry,
                "inference": self._build_inference,
                "ensemble": self._build_ensemble,
                "benchmark": self._build_benchmark,
                "activity_cliffs": self._build_activity_cliffs,
            }
            instance = builders[name]()
            self._instances[name] = instance
            return instance

    def _shared_catalog(self) -> Any:
        if self._catalog is None:
            catalog_cls = _read_safe_catalog_type()
            self._catalog = catalog_cls.load()
        return self._catalog

    def _shared_backends(self) -> dict[str, Any]:
        if self._backends is None:
            from cs_copilot.tools.prediction.backend_factory import (
                build_default_prediction_backends,
            )

            self._backends = build_default_prediction_backends()
        return self._backends

    def _build_curation(self) -> Any:
        from cs_copilot.tools.curation.dataset_curation_toolkit import (
            DatasetCurationToolkit,
        )

        return DatasetCurationToolkit()

    def _build_training(self) -> Any:
        from cs_copilot.tools.prediction.qsar_training_toolkit import QSARTrainingToolkit

        # ``prepare_training_dataset`` remains blocked from the public MCP
        # surface.  The flag also protects against accidental direct use.
        return QSARTrainingToolkit(block_prepare_training_dataset=True)

    def _build_registry(self) -> Any:
        from cs_copilot.tools.prediction.model_registry_toolkit import ModelRegistryToolkit

        return ModelRegistryToolkit(
            backends=self._shared_backends(),
            catalog=self._shared_catalog(),
        )

    def _build_inference(self) -> Any:
        from cs_copilot.tools.prediction.prediction_inference_toolkit import (
            PredictionInferenceToolkit,
        )

        return PredictionInferenceToolkit(
            backends=self._shared_backends(),
            registry_toolkit=self.get("registry"),
        )

    def _build_ensemble(self) -> Any:
        from cs_copilot.tools.prediction.ensemble_toolkit import EnsembleToolkit

        return EnsembleToolkit(catalog=self._shared_catalog())

    def _build_benchmark(self) -> Any:
        from cs_copilot.tools.prediction.benchmark_toolkit import BenchmarkToolkit

        return BenchmarkToolkit(
            training_toolkit=self.get("training"),
            registry_toolkit=self.get("registry"),
        )

    def _build_activity_cliffs(self) -> Any:
        from cs_copilot.tools.activity_cliffs.toolkit import ActivityCliffToolkit

        return ActivityCliffToolkit()

    def reload_catalog(self) -> None:
        """Reload catalog records from disk without persisting discovery changes."""

        with self._lock:
            if self._catalog is None:
                self._shared_catalog()
                return
            catalog_cls = type(self._catalog)
            fresh = catalog_cls.load(path=str(self._catalog.source_path))
            # Keep object identity: registry, inference, and ensemble all hold
            # references to this object.
            self._catalog.records = fresh.records
            self._catalog.schema_version = fresh.schema_version
            self._catalog.source_path = Path(fresh.source_path)
            self._catalog.refresh_from_internal_store(persist=False)

    def reconcile_catalog(self) -> None:
        """Restore one read-safe catalog after a toolkit replaces its reference.

        ``ModelRegistryToolkit.persist_registered_model`` intentionally reloads
        its catalog by assigning a new base ``PredictionModelCatalog``.  In the
        Qsaria hub that would detach Registry (and therefore Inference) from the
        read-safe catalog still held by Ensemble.  Reload the canonical object
        from disk and rebind every already-created catalog consumer after each
        catalog-write scope, including exceptional exits.
        """

        with self._lock:
            if self._catalog is None:
                return

            # A replacing toolkit may also have changed the configured source
            # path.  Prefer that path, then reload the canonical read-safe
            # object without changing its identity.
            registry = self._instances.get("registry")
            replacement = getattr(registry, "catalog", None)
            replacement_source = getattr(replacement, "source_path", None)
            if replacement_source is not None:
                self._catalog.source_path = Path(replacement_source)
            self.reload_catalog()

            if registry is not None:
                registry.catalog = self._catalog

            inference = self._instances.get("inference")
            inference_registry = getattr(inference, "registry_toolkit", None)
            if inference_registry is not None:
                inference_registry.catalog = self._catalog

            ensemble = self._instances.get("ensemble")
            if ensemble is not None:
                ensemble.catalog = self._catalog

            benchmark = self._instances.get("benchmark")
            benchmark_registry = getattr(benchmark, "registry_toolkit", None)
            if benchmark_registry is not None:
                benchmark_registry.catalog = self._catalog

    @contextmanager
    def catalog_scope(self, *, access: bool, write: bool) -> Iterator[None]:
        """Prepare a fresh catalog view and selectively permit persistence.

        The cross-call/catalog mutation lock is owned by ``ExperimentManager``
        (``manager.run(..., catalog_write=True)``).  This local lock protects
        the shared in-process object while it is refreshed and used.
        """

        if not access:
            yield
            return
        with self._lock:
            self.reconcile_catalog()
            token = _CATALOG_WRITE_ENABLED.set(bool(write))
            try:
                yield
            finally:
                try:
                    if write:
                        self.reconcile_catalog()
                finally:
                    _CATALOG_WRITE_ENABLED.reset(token)


@lru_cache(maxsize=1)
def get_toolkit_hub() -> ToolkitHub:
    """Return the process-wide Qsaria toolkit hub."""

    return ToolkitHub()


def _toolkit(name: ToolkitName) -> Any:
    return get_toolkit_hub().get(name)


def curation_toolkit() -> Any:
    return _toolkit("curation")


def training_toolkit() -> Any:
    return _toolkit("training")


def registry_toolkit() -> Any:
    return _toolkit("registry")


def inference_toolkit() -> Any:
    return _toolkit("inference")


def ensemble_toolkit() -> Any:
    return _toolkit("ensemble")


def benchmark_toolkit() -> Any:
    return _toolkit("benchmark")


def activity_cliffs_toolkit() -> Any:
    return _toolkit("activity_cliffs")


@contextmanager
def catalog_scope(*, access: bool, write: bool) -> Iterator[None]:
    """Public adapter hook for safe catalog reload/write scoping."""

    with get_toolkit_hub().catalog_scope(access=access, write=write):
        yield


__all__ = [
    "ToolkitHub",
    "activity_cliffs_toolkit",
    "benchmark_toolkit",
    "catalog_scope",
    "curation_toolkit",
    "ensemble_toolkit",
    "get_toolkit_hub",
    "inference_toolkit",
    "registry_toolkit",
    "training_toolkit",
]
