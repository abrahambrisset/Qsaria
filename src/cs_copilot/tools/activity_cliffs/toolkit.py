#!/usr/bin/env python
# coding: utf-8
"""Agent-facing Activity Cliff toolkit."""

from __future__ import annotations

from typing import Any, Dict

from agno.tools.toolkit import Toolkit

from .service import (
    DEFAULT_ACTIVITY_CLIFF_INDEX,
    DEFAULT_FLAG_THRESHOLD,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_NEIGHBORS,
    default_registry,
    prepare_activity_cliff_context,
)


class ActivityCliffToolkit(Toolkit):
    """Expose activity-cliff annotation and variant planning to QSAR training."""

    def __init__(self):
        super().__init__("activity_cliffs")
        self.register(self.list_activity_cliff_indexes)
        self.register(self.prepare_activity_cliff_context)

    def list_activity_cliff_indexes(self) -> Dict[str, Any]:
        registry = default_registry()
        return {
            "default_index": DEFAULT_ACTIVITY_CLIFF_INDEX,
            "available_indexes": registry.list_indexes(),
        }

    def prepare_activity_cliff_context(
        self,
        train_csv: str,
        output_dir: str,
        target_column: str,
        smiles_column: str = "smiles",
        activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
        activity_cliff_feedback: bool = False,
        activity_cliff_feedback_loops: int = 0,
        activity_cliff_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        activity_cliff_top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
        activity_cliff_flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    ) -> Dict[str, Any]:
        """Annotate a QSAR-ready dataset and produce Activity Cliff artifacts."""
        return prepare_activity_cliff_context(
            train_csv=train_csv,
            output_dir=output_dir,
            smiles_column=smiles_column,
            target_column=target_column,
            activity_cliff_index=activity_cliff_index,
            activity_cliff_feedback=activity_cliff_feedback,
            activity_cliff_feedback_loops=activity_cliff_feedback_loops,
            activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
            activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
            activity_cliff_flag_threshold=activity_cliff_flag_threshold,
        )
