#!/usr/bin/env python
# coding: utf-8
"""Backend construction helpers for QSAR prediction workflows."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_default_prediction_backends(
    *,
    chemprop_backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the standard backend set without making any toolkit the hub."""
    from .chemprop_backend import ChempropBackend
    from .ensemble_backend import EnsembleBackend
    from .lightgbm_backend import LightGBMBackend
    from .tabicl_backend import TabICLBackend

    primary_backend = chemprop_backend or ChempropBackend()
    tabicl_backend = TabICLBackend()
    lightgbm_backend = LightGBMBackend()
    component_backends = {
        primary_backend.backend_name: primary_backend,
        tabicl_backend.backend_name: tabicl_backend,
        lightgbm_backend.backend_name: lightgbm_backend,
    }
    ensemble_backend = EnsembleBackend(backends=component_backends)
    return {
        **component_backends,
        ensemble_backend.backend_name: ensemble_backend,
    }
