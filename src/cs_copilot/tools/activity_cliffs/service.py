#!/usr/bin/env python
# coding: utf-8
"""Core Activity Cliff services.

This module intentionally treats SALI as the first registered index, not as the
framework itself. Training code should depend on the generic
``activity_cliff_*`` contract so later indexes can be added without changing
backend toolkits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from cs_copilot.storage import S3

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTIVITY_CLIFF_ANNOTATION_PREFIX = "activity_cliff_"
DEFAULT_ACTIVITY_CLIFF_INDEX = "sali"
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_TOP_K_NEIGHBORS = 10
DEFAULT_FLAG_THRESHOLD = 0.35
DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_KIND = "morgan_count"
DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_RADIUS = 2
DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_SIZE = 2048
DEFAULT_ACTIVITY_CLIFF_SIMILARITY_METRIC = "count_tanimoto"
DEFAULT_ACTIVITY_CLIFF_REPRESENTATION_NOTE = (
    "Count fingerprints preserve repeated local-environment multiplicity."
)
DEFAULT_ACTIVITY_GAP_REFERENCE = 1.0
MAX_ACTIVITY_CLIFF_LOOPS = 3
MIN_VARIANT_TRAINING_ROWS = 10
SALI_EPSILON = 1e-6
PLOT_DPI = 300
ACTIVITY_CLIFF_COLORS = {
    "none": "#2F80ED",
    "low": "#4CAF50",
    "medium": "#E08D2D",
    "high": "#C44E52",
}
ACTIVITY_CLIFF_ARG_KEYS = {
    "activity_cliff_index",
    "activity_cliff_feedback",
    "activity_cliff_feedback_loops",
    "activity_cliff_similarity_threshold",
    "activity_cliff_top_k_neighbors",
    "activity_cliff_k_neighbors",
    "activity_cliff_flag_threshold",
}


@dataclass(frozen=True)
class ActivityCliffConfig:
    index_name: str = DEFAULT_ACTIVITY_CLIFF_INDEX
    mode: str = "standard"
    feedback_loops: int = 0
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS
    flag_threshold: float = DEFAULT_FLAG_THRESHOLD
    fingerprint_kind: str = DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_KIND
    fingerprint_radius: int = DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_RADIUS
    fingerprint_size: int = DEFAULT_ACTIVITY_CLIFF_FINGERPRINT_SIZE
    similarity_metric: str = DEFAULT_ACTIVITY_CLIFF_SIMILARITY_METRIC

    @property
    def loops_enabled(self) -> bool:
        return self.mode == "with_feedback_loops" and self.feedback_loops > 0


class ActivityCliffIndex(Protocol):
    index_name: str

    def compute(
        self,
        *,
        dataset: pd.DataFrame,
        smiles_column: str,
        target_column: str,
        config: ActivityCliffConfig,
    ) -> pd.DataFrame:
        """Return index-specific and generic activity-cliff annotations."""


class ActivityCliffIndexRegistry:
    """Registry for pluggable activity-cliff indexes."""

    def __init__(self):
        self._indexes: Dict[str, ActivityCliffIndex] = {}

    def register(self, index: ActivityCliffIndex) -> None:
        self._indexes[index.index_name.lower()] = index

    def list_indexes(self) -> List[str]:
        return sorted(self._indexes)

    def get(self, index_name: str) -> ActivityCliffIndex:
        normalized = str(index_name or DEFAULT_ACTIVITY_CLIFF_INDEX).lower()
        if normalized not in self._indexes:
            raise ValueError(
                "Unsupported activity_cliff_index "
                f"`{index_name}`. Available indexes: {self.list_indexes()}."
            )
        return self._indexes[normalized]


class SALIIndex:
    """Structure-Activity Landscape Index implementation."""

    index_name = DEFAULT_ACTIVITY_CLIFF_INDEX

    def compute(
        self,
        *,
        dataset: pd.DataFrame,
        smiles_column: str,
        target_column: str,
        config: ActivityCliffConfig,
    ) -> pd.DataFrame:
        if smiles_column not in dataset.columns:
            raise ValueError(f"Missing smiles column for activity cliffs: {smiles_column}")
        if target_column not in dataset.columns:
            raise ValueError(f"Missing target column for activity cliffs: {target_column}")

        working = dataset.copy().reset_index(drop=True)
        y_true = pd.to_numeric(working[target_column], errors="coerce")
        if y_true.dropna().empty:
            raise ValueError("Cannot compute activity cliffs: target column has no numeric values.")

        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=config.fingerprint_radius,
            fpSize=config.fingerprint_size,
        )
        fps: List[Any] = []
        for smiles in working[smiles_column].tolist():
            try:
                mol = Chem.MolFromSmiles(str(smiles)) if pd.notna(smiles) else None
                fps.append(generator.GetCountFingerprint(mol) if mol is not None else None)
            except Exception:
                fps.append(None)

        raw_scores = np.zeros(len(working), dtype=float)
        neighbor_counts = np.zeros(len(working), dtype=int)
        max_similarity = np.zeros(len(working), dtype=float)
        max_activity_gap = np.zeros(len(working), dtype=float)
        pair_scores: List[float] = []
        exact_similarity_nonidentical_pairs: set[tuple[int, int]] = set()
        max_similarity_observed = 0.0

        for idx, fp in enumerate(fps):
            if fp is None or pd.isna(y_true.iloc[idx]):
                continue
            sims = DataStructs.BulkTanimotoSimilarity(fp, fps)
            candidates: List[tuple[float, float, float]] = []
            for other_idx, sim in enumerate(sims):
                if other_idx == idx or fps[other_idx] is None or pd.isna(y_true.iloc[other_idx]):
                    continue
                sim_value = float(sim)
                max_similarity_observed = max(max_similarity_observed, sim_value)
                if sim_value < config.similarity_threshold:
                    continue
                gap = abs(float(y_true.iloc[idx]) - float(y_true.iloc[other_idx]))
                if sim_value >= 1.0 and gap <= 0.0:
                    continue
                if sim_value >= 1.0 and gap > 0.0:
                    exact_similarity_nonidentical_pairs.add(tuple(sorted((idx, other_idx))))
                sali = gap / max(1.0 - sim_value, SALI_EPSILON)
                pair_scores.append(float(sali))
                candidates.append((sim_value, gap, float(sali)))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            top_candidates = candidates[: config.top_k_neighbors]
            neighbor_counts[idx] = len(top_candidates)
            max_similarity[idx] = max(item[0] for item in top_candidates)
            max_activity_gap[idx] = max(item[1] for item in top_candidates)
            raw_scores[idx] = max(item[2] for item in top_candidates)

        p95 = float(np.percentile(pair_scores, 95)) if pair_scores else 0.0
        if p95 <= 0:
            p95 = 1.0
        norm_scores = pd.Series(raw_scores, index=working.index).div(p95).clip(0.0, 1.0)

        working["activity_cliff_index_name"] = self.index_name
        working["activity_cliff_score_raw"] = pd.Series(raw_scores, index=working.index).astype(float)
        working["activity_cliff_score_norm"] = norm_scores.astype(float)
        working["activity_cliff_neighbor_count"] = pd.Series(
            neighbor_counts, index=working.index
        ).astype(int)
        working["activity_cliff_sali_raw"] = working["activity_cliff_score_raw"]
        working["activity_cliff_sali_norm"] = working["activity_cliff_score_norm"]
        working["activity_cliff_max_similarity"] = pd.Series(
            max_similarity, index=working.index
        ).astype(float)
        working["activity_cliff_max_activity_gap"] = pd.Series(
            max_activity_gap, index=working.index
        ).astype(float)
        working.attrs["activity_cliff_similarity_diagnostics"] = {
            "exact_similarity_nonidentical_pair_count": len(exact_similarity_nonidentical_pairs),
            "max_similarity_observed": float(max_similarity_observed),
            "exact_similarity_policy": "reported_not_silently_corrected",
        }
        return working


def default_registry() -> ActivityCliffIndexRegistry:
    registry = ActivityCliffIndexRegistry()
    registry.register(SALIIndex())
    return registry


def strip_activity_cliff_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.astype(str).str.startswith(ACTIVITY_CLIFF_ANNOTATION_PREFIX)].copy()


def split_activity_cliff_args(extra_args: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    raw = dict(extra_args or {})
    activity_args: Dict[str, Any] = {}
    cleaned: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in ACTIVITY_CLIFF_ARG_KEYS:
            normalized_key = (
                "activity_cliff_top_k_neighbors"
                if key == "activity_cliff_k_neighbors"
                else key
            )
            activity_args[normalized_key] = value
        else:
            cleaned[key] = value
    return cleaned, activity_args


def parse_activity_cliff_config(
    *,
    activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 0,
    activity_cliff_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    activity_cliff_top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
    activity_cliff_flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    registry: Optional[ActivityCliffIndexRegistry] = None,
) -> ActivityCliffConfig:
    registry = registry or default_registry()
    registry.get(activity_cliff_index)
    loops = int(activity_cliff_feedback_loops or 0)
    if loops < 0:
        raise ValueError("activity_cliff_feedback_loops must be >= 0.")
    if loops > MAX_ACTIVITY_CLIFF_LOOPS:
        raise ValueError("activity_cliff_feedback_loops cannot exceed 3.")
    if bool(activity_cliff_feedback) and loops < 1:
        raise ValueError(
            "activity_cliff_feedback_loops must be between 1 and 3 when feedback loops are requested."
        )
    if not (0.0 < float(activity_cliff_similarity_threshold) <= 1.0):
        raise ValueError("activity_cliff_similarity_threshold must be in (0, 1].")
    if int(activity_cliff_top_k_neighbors) < 1:
        raise ValueError("activity_cliff_top_k_neighbors must be >= 1.")
    if not (0.0 < float(activity_cliff_flag_threshold) <= 1.0):
        raise ValueError("activity_cliff_flag_threshold must be in (0, 1].")
    return ActivityCliffConfig(
        index_name=str(activity_cliff_index or DEFAULT_ACTIVITY_CLIFF_INDEX).lower(),
        mode="with_feedback_loops" if loops > 0 or activity_cliff_feedback else "standard",
        feedback_loops=loops,
        similarity_threshold=float(activity_cliff_similarity_threshold),
        top_k_neighbors=int(activity_cliff_top_k_neighbors),
        flag_threshold=float(activity_cliff_flag_threshold),
    )


def _assign_tiers(score_norm: pd.Series, flagged: pd.Series) -> pd.Series:
    tiers = pd.Series("none", index=score_norm.index, dtype=object)
    ranked = score_norm[flagged].sort_values(ascending=False, kind="mergesort")
    n_flagged = len(ranked)
    if n_flagged == 0:
        return tiers
    high_end = max(1, int(math.ceil(n_flagged * 0.20)))
    medium_end = max(high_end, int(math.ceil(n_flagged * 0.40)))
    for rank, idx in enumerate(ranked.index.tolist()):
        if rank < high_end:
            tiers.at[idx] = "high"
        elif rank < medium_end:
            tiers.at[idx] = "medium"
        else:
            tiers.at[idx] = "low"
    return tiers


def _reason_codes(row: pd.Series) -> str:
    codes: List[str] = []
    if bool(row.get("activity_cliff_flag", False)):
        codes.append("HIGH_ACTIVITY_DISCONTINUITY")
    if float(row.get("activity_cliff_score_norm", 0.0) or 0.0) >= 0.8:
        codes.append("HIGH_NORMALIZED_INDEX_SCORE")
    if int(row.get("activity_cliff_neighbor_count", 0) or 0) >= 3:
        codes.append("MULTIPLE_SIMILAR_NEIGHBORS")
    return "|".join(codes) if codes else "NOT_FLAGGED"


def _justification(row: pd.Series) -> str:
    tier = str(row.get("activity_cliff_priority_tier") or "none")
    if tier == "none":
        return "Not flagged by the selected activity-cliff index under the current policy."
    return (
        f"{tier.title()} priority from the selected activity-cliff index: "
        f"normalized score={float(row.get('activity_cliff_score_norm', 0.0) or 0.0):.3f}, "
        f"qualified_neighbors={int(row.get('activity_cliff_neighbor_count', 0) or 0)}."
    )


def _priority_counts(series: pd.Series) -> Dict[str, int]:
    counts = series.fillna("none").astype(str).value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in ("none", "low", "medium", "high")}


def _plot_score_histogram(
    annotated: pd.DataFrame,
    output_path: Path,
    *,
    flag_threshold: float,
) -> Optional[str]:
    values = pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce").dropna()
    if values.empty:
        return None
    flagged_values = values[values >= flag_threshold]
    tiers = annotated.loc[values.index, "activity_cliff_priority_tier"].fillna("none").astype(str)
    tier_counts = _priority_counts(tiers)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 4.5),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    ax = axes[0]
    ax.hist(values, bins=40, color=ACTIVITY_CLIFF_COLORS["none"], edgecolor="white", alpha=0.86)
    ax.axvline(
        flag_threshold,
        color=ACTIVITY_CLIFF_COLORS["high"],
        linewidth=2,
        linestyle="--",
        label=f"flag threshold {flag_threshold:g}",
    )
    if not flagged_values.empty:
        tier_values_all = [
            values[(tiers == tier) & (values >= flag_threshold)]
            for tier in ("low", "medium", "high")
        ]
        ax.hist(
            tier_values_all,
            bins=10,
            stacked=True,
            color=[ACTIVITY_CLIFF_COLORS[tier] for tier in ("low", "medium", "high")],
            edgecolor="white",
            alpha=0.92,
        )
    ax.set_yscale("log")
    ax.set_title("All compounds (log scale)")
    ax.set_xlabel("Normalized activity-cliff score")
    ax.set_ylabel("Compound count (log)")
    ax.grid(alpha=0.2, linestyle="--")
    ax.text(
        0.98,
        0.94,
        (
            f"none={tier_counts['none']}, flagged={len(flagged_values)}\n"
            f"low={tier_counts['low']}, medium={tier_counts['medium']}, high={tier_counts['high']}\n"
            f"flag threshold={flag_threshold:g}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.88, edgecolor="#d0d0d0"),
    )

    ax_zoom = axes[1]
    if not flagged_values.empty:
        lower_edge = math.floor(float(min(flagged_values.min(), flag_threshold)) * 10.0) / 10.0
        bins = np.arange(max(0.0, lower_edge), 1.11, 0.1)
        tier_values = [
            values[(tiers == tier) & (values >= flag_threshold)]
            for tier in ("low", "medium", "high")
        ]
        ax_zoom.hist(
            tier_values,
            bins=bins,
            stacked=True,
            color=[ACTIVITY_CLIFF_COLORS[tier] for tier in ("low", "medium", "high")],
            edgecolor="white",
            alpha=0.95,
            label=[
                f"low ({tier_counts['low']})",
                f"medium ({tier_counts['medium']})",
                f"high ({tier_counts['high']})",
            ],
        )
        ax_zoom.set_xlim(max(0.0, min(flagged_values.min(), flag_threshold) - 0.04), 1.02)
        ax_zoom.legend(frameon=False, fontsize=8)
    else:
        ax_zoom.text(0.5, 0.5, "No flagged compounds", ha="center", va="center", transform=ax_zoom.transAxes)
        ax_zoom.set_xlim(flag_threshold, 1.02)
    ax_zoom.axvline(
        flag_threshold,
        color=ACTIVITY_CLIFF_COLORS["high"],
        linewidth=2,
        linestyle="--",
    )
    ax_zoom.set_title("Flagged score distribution by tier")
    ax_zoom.set_xlabel("Normalized score")
    ax_zoom.set_ylabel("Count")
    ax_zoom.grid(alpha=0.2, linestyle="--")
    fig.suptitle("SALI normalized score distribution by priority tier", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_tier_distribution(annotated: pd.DataFrame, output_path: Path) -> Optional[str]:
    counts = _priority_counts(annotated["activity_cliff_priority_tier"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    labels = ["low", "medium", "high"]
    values = [counts[label] for label in labels]
    bars = ax.bar(labels, values, color=[ACTIVITY_CLIFF_COLORS[label] for label in labels])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values + [1]) * 0.03,
            str(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title("Flagged activity-cliff tier distribution")
    ax.set_xlabel("Tier")
    ax.set_ylabel("Flagged compound count")
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.text(
        0.99,
        0.94,
        f"Not flagged: {counts['none']}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#5f6972",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_gap_vs_similarity(
    annotated: pd.DataFrame,
    output_path: Path,
    *,
    similarity_threshold: float,
    activity_gap_reference: float = DEFAULT_ACTIVITY_GAP_REFERENCE,
) -> Optional[str]:
    if "activity_cliff_max_similarity" not in annotated.columns:
        return None
    x = pd.to_numeric(annotated["activity_cliff_max_similarity"], errors="coerce")
    y = pd.to_numeric(annotated["activity_cliff_max_activity_gap"], errors="coerce")
    valid = x.notna() & y.notna()
    if not valid.any():
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    tiers = annotated.loc[valid, "activity_cliff_priority_tier"].fillna("none").astype(str)
    tier_counts = annotated["activity_cliff_priority_tier"].fillna("none").astype(str).value_counts()
    x_plot = x[valid].copy().astype(float)
    stacked_at_one = x_plot >= 0.999
    if stacked_at_one.any():
        jitter = np.linspace(-0.006, 0.006, int(stacked_at_one.sum()))
        x_plot.loc[stacked_at_one] = (x_plot.loc[stacked_at_one].to_numpy() + jitter).clip(0.0, 1.01)
    none_mask = tiers == "none"
    if none_mask.any():
        ax.scatter(
            x_plot[none_mask],
            y[valid][none_mask],
            s=24,
            c=ACTIVITY_CLIFF_COLORS["none"],
            alpha=0.46,
            edgecolor="none",
            label=f"none ({int(tier_counts.get('none', 0))})",
        )
    for tier, color, size in (
        ("low", ACTIVITY_CLIFF_COLORS["low"], 58),
        ("medium", ACTIVITY_CLIFF_COLORS["medium"], 72),
        ("high", ACTIVITY_CLIFF_COLORS["high"], 88),
    ):
        tier_mask = tiers == tier
        if tier_mask.any():
            ax.scatter(
                x_plot[tier_mask],
                y[valid][tier_mask],
                s=size,
                c=color,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.8,
                label=f"{tier} ({int(tier_counts.get(tier, 0))})",
                zorder=3,
            )
    ax.axvline(
        similarity_threshold,
        color="#5f6972",
        linewidth=1.5,
        linestyle="--",
        label=f"similarity threshold {similarity_threshold:g}",
    )
    if activity_gap_reference > 0:
        ax.axhline(
            activity_gap_reference,
            color="#7a8288",
            linewidth=1.2,
            linestyle=":",
            label=f"activity gap reference {activity_gap_reference:g}",
        )
    positive_x = x_plot[x_plot > 0]
    data_x_min = float(positive_x.min()) - 0.03 if not positive_x.empty else float(similarity_threshold) - 0.04
    x_min = max(0.0, min(float(similarity_threshold) - 0.04, data_x_min))
    ax.set_xlim(x_min, 1.02)
    y_max = float(y[valid].max()) if valid.any() else activity_gap_reference
    annotation_y = min(max(activity_gap_reference + 0.08, y_max * 0.72), y_max * 0.92)
    ax.text(
        min(0.985, max(float(similarity_threshold) + 0.16, 0.84)),
        annotation_y,
        "High-similarity / high-gap cliffs",
        fontsize=8,
        color="#5f6972",
        ha="center",
    )
    ax.text(
        float(similarity_threshold) + 0.012,
        max(0.08, activity_gap_reference * 0.22),
        "Borderline similarity",
        fontsize=8,
        color="#5f6972",
        rotation=90,
        va="bottom",
    )
    ax.text(
        min(0.985, max(float(similarity_threshold) + 0.16, 0.84)),
        max(0.05, activity_gap_reference * 0.18),
        "Low activity gap",
        fontsize=8,
        color="#5f6972",
        ha="center",
    )
    ax.set_title("Activity cliffs: activity gap vs count-Tanimoto similarity")
    ax.set_xlabel("Maximum qualified count-Tanimoto similarity")
    ax.set_ylabel("Maximum activity gap")
    ax.grid(alpha=0.2, linestyle="--")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _write_plots(
    annotated: pd.DataFrame,
    output_dir: Path,
    *,
    flag_threshold: float,
    similarity_threshold: float,
) -> Dict[str, str]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = {
        "activity_cliff_score_histogram": _plot_score_histogram(
            annotated,
            plots_dir / "activity_cliff_sali_histogram.png",
            flag_threshold=flag_threshold,
        ),
        "activity_cliff_tier_distribution": _plot_tier_distribution(
            annotated, plots_dir / "activity_cliff_tier_distribution.png"
        ),
        "activity_gap_vs_similarity": _plot_gap_vs_similarity(
            annotated,
            plots_dir / "activity_gap_vs_similarity.png",
            similarity_threshold=similarity_threshold,
            activity_gap_reference=DEFAULT_ACTIVITY_GAP_REFERENCE,
        ),
    }
    return {key: value for key, value in generated.items() if value}


def _short_variant_label(variant_id: str) -> str:
    if variant_id == "baseline_loop_0":
        return "baseline"
    if "loop_1" in variant_id:
        return "drop high"
    if "loop_2" in variant_id:
        return "drop high+medium"
    if "loop_3" in variant_id:
        return "drop high+medium+low"
    return variant_id


def _loop_axis_label(variant_id: str) -> str:
    if variant_id == "baseline_loop_0":
        return "loop 0\nbaseline"
    if "loop_1" in variant_id:
        return "loop 1\ndrop high"
    if "loop_2" in variant_id:
        return "loop 2\ndrop high+medium"
    if "loop_3" in variant_id:
        return "loop 3\ndrop high+medium+low"
    return variant_id


def _grid_variant_label(variant_id: str) -> str:
    if variant_id == "baseline_loop_0":
        return "Loop 0: baseline"
    if "loop_1" in variant_id:
        return "Loop 1: drop high"
    if "loop_2" in variant_id:
        return "Loop 2: drop high+medium"
    if "loop_3" in variant_id:
        return "Loop 3: drop high+medium+low"
    return variant_id


def _variant_order(variant_id: str) -> int:
    if variant_id == "baseline_loop_0":
        return 0
    for idx in (1, 2, 3):
        if f"loop_{idx}" in variant_id:
            return idx
    return 99


def _plot_loop_metric_comparison(
    table: pd.DataFrame,
    output_path: Path,
    *,
    metric: str,
    ylabel: str,
    recommended_variant: Optional[str] = None,
) -> Optional[str]:
    if table.empty or metric not in table.columns:
        return None
    rows = table.dropna(subset=[metric]).copy()
    if rows.empty:
        return None
    rows["variant_order"] = rows["variant_id"].astype(str).map(_variant_order)
    rows["variant_label"] = rows["variant_id"].astype(str).map(_loop_axis_label)
    rows = rows.sort_values(["variant_order", "split"])
    pivot = rows.pivot_table(index="variant_label", columns="split", values=metric, aggfunc="first")
    if pivot.empty:
        return None

    ordered_labels = rows.drop_duplicates("variant_label").sort_values("variant_order")["variant_label"].tolist()
    pivot = pivot.reindex(ordered_labels)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    colors = {"random": "#224f75", "scaffold": "#d98b36"}
    recommended_label = _loop_axis_label(str(recommended_variant)) if recommended_variant else None
    if recommended_label in ordered_labels:
        ax.axvspan(
            ordered_labels.index(recommended_label) - 0.32,
            ordered_labels.index(recommended_label) + 0.32,
            color="#d8eadf",
            alpha=0.45,
            zorder=0,
            label="recommended",
        )
    for split_name in [column for column in ("random", "scaffold") if column in pivot.columns]:
        baseline_value = pivot.loc[ordered_labels[0], split_name] if ordered_labels else np.nan
        if pd.notna(baseline_value):
            ax.axhline(
                float(baseline_value),
                color=colors.get(split_name, "#5f6972"),
                linewidth=1.1,
                linestyle=":",
                alpha=0.55,
            )
        ax.plot(
            pivot.index,
            pivot[split_name],
            marker="o",
            linewidth=2,
            color=colors.get(split_name, None),
            label=split_name,
        )
        for x_pos, value in enumerate(pivot[split_name].tolist()):
            if pd.notna(value):
                ax.text(x_pos, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=8)
    metric_label = "R²" if metric == "r2" else ylabel
    direction_note = "higher is better" if metric == "r2" else "lower is better"
    ax.set_title(f"Activity-cliff loop comparison - {metric_label}")
    ax.set_xlabel("Variant")
    ax.set_ylabel(metric_label)
    ax.grid(alpha=0.2, linestyle="--", axis="y")
    ax.text(
        0.01,
        0.04,
        f"{direction_note}; dotted lines show split baselines",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#5f6972",
    )
    values = pivot.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size:
        pad = max(0.002, float(values.max() - values.min()) * 0.18)
        ax.set_ylim(float(values.min()) - pad, float(values.max()) + pad)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_loop_delta_comparison(table: pd.DataFrame, output_path: Path) -> Optional[str]:
    if table.empty or "r2" not in table.columns or "rmse" not in table.columns:
        return None
    rows = table.copy()
    rows["variant_order"] = rows["variant_id"].astype(str).map(_variant_order)
    rows["variant_label"] = rows["variant_id"].astype(str).map(_short_variant_label)
    baseline = rows[rows["variant_id"] == "baseline_loop_0"].set_index("split")
    if baseline.empty:
        return None
    plot_rows: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        split = row.get("split")
        if split not in baseline.index:
            continue
        base = baseline.loc[split]
        plot_rows.append(
            {
                "variant_label": row["variant_label"],
                "variant_order": row["variant_order"],
                "split": split,
                "delta_r2": float(row["r2"]) - float(base["r2"]) if pd.notna(row.get("r2")) else np.nan,
                "delta_rmse": float(row["rmse"]) - float(base["rmse"]) if pd.notna(row.get("rmse")) else np.nan,
            }
        )
    delta = pd.DataFrame(plot_rows)
    if delta.empty:
        return None
    delta = delta[delta["variant_label"] != _short_variant_label("baseline_loop_0")].sort_values(["variant_order", "split"])
    if delta.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True)
    colors = {"random": "#224f75", "scaffold": "#d98b36"}
    for ax, metric, title, zero_label in (
        (axes[0], "delta_r2", "Delta R2 vs baseline", "baseline"),
        (axes[1], "delta_rmse", "Delta RMSE vs baseline", "baseline"),
    ):
        pivot = delta.pivot_table(index="variant_label", columns="split", values=metric, aggfunc="first")
        labels = delta.drop_duplicates("variant_label").sort_values("variant_order")["variant_label"].tolist()
        pivot = pivot.reindex(labels)
        vals = pivot.to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        if finite_vals.size:
            pad = max(0.001, float(finite_vals.max() - finite_vals.min()) * 0.20)
            y_min = float(finite_vals.min()) - pad
            y_max = float(finite_vals.max()) + pad
            if metric == "delta_r2":
                ax.axhspan(0.0, y_max, color="#d8eadf", alpha=0.35, zorder=0)
                ax.axhspan(y_min, 0.0, color="#f4d6d2", alpha=0.28, zorder=0)
            else:
                ax.axhspan(y_min, 0.0, color="#d8eadf", alpha=0.35, zorder=0)
                ax.axhspan(0.0, y_max, color="#f4d6d2", alpha=0.28, zorder=0)
            ax.set_ylim(y_min, y_max)
        for split_name in [column for column in ("random", "scaffold") if column in pivot.columns]:
            ax.plot(
                pivot.index,
                pivot[split_name],
                marker="o",
                linewidth=2,
                color=colors.get(split_name, None),
                label=split_name,
            )
            for x_pos, value in enumerate(pivot[split_name].tolist()):
                if pd.notna(value):
                    ax.text(x_pos, float(value), f"{float(value):+.3f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.0, color="#5f6972", linewidth=1, linestyle="--", label=zero_label)
        ax.set_title(title)
        ax.grid(alpha=0.2, linestyle="--", axis="y")
    axes[0].set_ylabel("Delta R²")
    axes[1].set_ylabel("Delta RMSE")
    axes[0].text(0.02, 0.04, "positive is better", transform=axes[0].transAxes, fontsize=8, color="#5f6972")
    axes[1].text(0.02, 0.04, "negative is better", transform=axes[1].transAxes, fontsize=8, color="#5f6972")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _load_variant_prediction_frame(predictions_path: Path) -> Optional[pd.DataFrame]:
    if not predictions_path.exists():
        return None
    predictions = pd.read_csv(predictions_path)
    if "y_true" not in predictions.columns or "y_pred" not in predictions.columns:
        return None
    frame = pd.DataFrame(
        {
            "y_true": pd.to_numeric(predictions["y_true"], errors="coerce"),
            "y_pred": pd.to_numeric(predictions["y_pred"], errors="coerce"),
        }
    ).dropna()
    if frame.empty:
        return None
    frame["residual"] = frame["y_true"] - frame["y_pred"]
    return frame


def _load_test_indices_from_split_result(split_result: Dict[str, Any]) -> List[int]:
    splits_path = split_result.get("splits_path")
    if not splits_path:
        return []
    path = Path(str(splits_path)).expanduser()
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return []
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        return []
    return [int(idx) for idx in (payload[0].get("test") or [])]


def _load_activity_cliff_tier_map(activity_cliffs: Dict[str, Any]) -> Dict[int, str]:
    annotated_path = activity_cliffs.get("annotated_training_csv")
    if not annotated_path:
        return {}
    path = Path(str(annotated_path)).expanduser()
    if not path.exists():
        return {}
    try:
        annotated = pd.read_csv(path)
    except Exception:
        return {}
    if "activity_cliff_priority_tier" not in annotated.columns:
        return {}
    tiers = annotated["activity_cliff_priority_tier"].astype(str).str.lower()
    return {
        int(idx): tier
        for idx, tier in tiers.items()
        if tier in {"low", "medium", "high"}
    }


def _draw_variant_parity(
    ax: Any,
    frame: pd.DataFrame,
    *,
    label: str,
    band_metric: str = "mae",
    recommended: bool = False,
    show_legend: bool = True,
    show_ylabel: bool = True,
    axis_limits: Optional[tuple[float, float]] = None,
    compact_title: bool = False,
    title_fontsize: Optional[int] = None,
    metrics_fontsize: int = 9,
    retained_tiers: Optional[set[str]] = None,
) -> None:
    mae = float(frame["residual"].abs().mean())
    rmse = float((frame["residual"].pow(2).mean()) ** 0.5)
    centered = frame["y_true"] - float(frame["y_true"].mean())
    ss_tot = float((centered.pow(2)).sum())
    ss_res = float((frame["residual"].pow(2)).sum())
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else None

    point_alpha = 0.45 if len(frame) > 250 else 0.65
    ax.scatter(frame["y_true"], frame["y_pred"], s=18, alpha=point_alpha, color=ACTIVITY_CLIFF_COLORS["none"])
    if retained_tiers and "activity_cliff_tier" in frame.columns:
        tier_styles = {
            "low": {"color": ACTIVITY_CLIFF_COLORS["low"], "label": "low retained"},
            "medium": {"color": ACTIVITY_CLIFF_COLORS["medium"], "label": "medium retained"},
            "high": {"color": ACTIVITY_CLIFF_COLORS["high"], "label": "high retained"},
        }
        for tier in ("low", "medium", "high"):
            if tier not in retained_tiers:
                continue
            tier_frame = frame[frame["activity_cliff_tier"] == tier]
            if tier_frame.empty:
                continue
            style = tier_styles[tier]
            ax.scatter(
                tier_frame["y_true"],
                tier_frame["y_pred"],
                s=54,
                c=style["color"],
                edgecolors=style["color"],
                linewidths=0.9,
                alpha=0.96,
                label=style["label"],
                zorder=4,
            )
            ax.scatter(
                tier_frame["y_true"],
                tier_frame["y_pred"],
                s=72,
                facecolors="none",
                edgecolors="white",
                linewidths=1.2,
                alpha=0.98,
                zorder=5,
            )
    if axis_limits is None:
        min_val = min(frame["y_true"].min(), frame["y_pred"].min())
        max_val = max(frame["y_true"].max(), frame["y_pred"].max())
    else:
        min_val, max_val = axis_limits
    x_band = pd.Series([min_val, max_val], dtype=float)
    band_value = mae if band_metric == "mae" else rmse
    band_label = "MAE" if band_metric == "mae" else "RMSE"
    if band_value > 0:
        ax.fill_between(
            x_band,
            x_band - (2.0 * band_value),
            x_band + (2.0 * band_value),
            color="#258d9a",
            alpha=0.10,
            zorder=0,
            label=f"±2x {band_label}",
        )
        ax.fill_between(
            x_band,
            x_band - band_value,
            x_band + band_value,
            color="#258d9a",
            alpha=0.18,
            zorder=1,
            label=f"±1x {band_label}",
        )
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="#b54a4a", linewidth=1.2)
    if compact_title:
        title = label
    else:
        title = f"Scaffold parity - {label} ({band_label} bands)"
    if recommended and not compact_title:
        title = f"{title} - recommended"
    ax.set_title(title, fontsize=title_fontsize)
    if recommended and compact_title:
        ax.text(
            0.97,
            0.03,
            "recommended",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#2f6f4e",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#d8eadf", alpha=0.85, edgecolor="#9fceb3"),
        )
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted" if show_ylabel else "")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_aspect("equal", adjustable="box")
    within_1x = float((frame["residual"].abs() <= band_value).mean() * 100.0) if band_value > 0 else 0.0
    within_2x = float((frame["residual"].abs() <= (2.0 * band_value)).mean() * 100.0) if band_value > 0 else 0.0
    metrics = [f"RMSE = {rmse:.3f}", f"MAE = {mae:.3f}"]
    if r2 is not None:
        metrics.insert(0, f"R² = {r2:.3f}")
    if band_value > 0:
        metrics.extend([f"{within_1x:.1f}% within 1x {band_label}", f"{within_2x:.1f}% within 2x {band_label}"])
    ax.text(
        0.03,
        0.97,
        "\n".join(metrics),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=metrics_fontsize,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )
    ax.grid(alpha=0.2, linestyle="--")
    if show_legend:
        ax.legend(loc="lower right", fontsize=8, frameon=True)


def _plot_variant_parity_from_predictions(
    predictions_path: Path,
    output_path: Path,
    *,
    label: str,
    band_metric: str = "mae",
    recommended: bool = False,
) -> Optional[str]:
    frame = _load_variant_prediction_frame(predictions_path)
    if frame is None:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _draw_variant_parity(
        ax,
        frame,
        label=label,
        band_metric=band_metric,
        recommended=recommended,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def _plot_variant_parity_grid(
    activity_cliffs: Dict[str, Any],
    output_path: Path,
    *,
    band_metric: str,
    recommended_variant: Optional[str] = None,
) -> Optional[str]:
    variants = activity_cliffs.get("variant_training") or []
    tier_map = _load_activity_cliff_tier_map(activity_cliffs)
    frames: List[tuple[str, str, List[str], pd.DataFrame]] = []
    for variant in sorted(variants, key=lambda item: _variant_order(str(item.get("variant_id") or ""))):
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            continue
        scaffold_result = next(
            (
                result
                for result in (variant.get("split_results") or [])
                if result.get("strategy_label") == "scaffold"
            ),
            None,
        )
        if not scaffold_result or not scaffold_result.get("test_predictions_path"):
            continue
        frame = _load_variant_prediction_frame(Path(str(scaffold_result["test_predictions_path"])).expanduser())
        if frame is not None:
            test_indices = _load_test_indices_from_split_result(scaffold_result)
            if tier_map and len(test_indices) == len(frame):
                frame = frame.copy()
                frame["source_row_index"] = test_indices
                frame["activity_cliff_tier"] = [
                    tier_map.get(int(idx), "none")
                    for idx in test_indices
                ]
            frames.append(
                (
                    variant_id,
                    _grid_variant_label(variant_id),
                    [str(tier).lower() for tier in (variant.get("removed_tiers") or [])],
                    frame,
                )
            )
    if not frames:
        return None
    min_val = min(float(min(frame["y_true"].min(), frame["y_pred"].min())) for _, _, _, frame in frames)
    max_val = max(float(max(frame["y_true"].max(), frame["y_pred"].max())) for _, _, _, frame in frames)
    pad = max(0.05, (max_val - min_val) * 0.04)
    axis_limits = (min_val - pad, max_val + pad)
    fig, axes = plt.subplots(1, len(frames), figsize=(4.9 * len(frames), 4.8), sharex=True, sharey=True)
    axes_array = np.atleast_1d(axes)
    for idx, (ax, (variant_id, label, removed_tiers, frame)) in enumerate(zip(axes_array, frames)):
        retained_tiers = {"low", "medium", "high"} - set(removed_tiers)
        _draw_variant_parity(
            ax,
            frame,
            label=label,
            band_metric=band_metric,
            recommended=variant_id == recommended_variant,
            show_legend=idx == len(frames) - 1,
            show_ylabel=idx == 0,
            axis_limits=axis_limits,
            compact_title=True,
            title_fontsize=10,
            metrics_fontsize=8,
            retained_tiers=retained_tiers,
        )
    band_label = "MAE" if band_metric == "mae" else "RMSE"
    fig.suptitle(f"Scaffold parity by activity-cliff loop ({band_label} bands)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.2)
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def build_activity_cliff_loop_comparison_plots(
    activity_cliffs: Dict[str, Any],
    *,
    output_dir: str,
) -> Dict[str, str]:
    table = pd.DataFrame(activity_cliffs.get("variant_comparison_table") or [])
    if table.empty:
        return {}
    comparison_dir = Path(output_dir).expanduser().resolve() / "loop_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Optional[str]] = {
        "loop_r2_comparison": _plot_loop_metric_comparison(
            table,
            comparison_dir / "activity_cliff_loop_r2_comparison.png",
            metric="r2",
            ylabel="R²",
            recommended_variant=activity_cliffs.get("recommended_variant"),
        ),
        "loop_rmse_comparison": _plot_loop_metric_comparison(
            table,
            comparison_dir / "activity_cliff_loop_rmse_comparison.png",
            metric="rmse",
            ylabel="RMSE",
            recommended_variant=activity_cliffs.get("recommended_variant"),
        ),
        "loop_delta_vs_baseline": _plot_loop_delta_comparison(
            table,
            comparison_dir / "activity_cliff_loop_delta_vs_baseline.png",
        ),
    }
    for variant in activity_cliffs.get("variant_training") or []:
        variant_id = str(variant.get("variant_id") or "")
        if not variant_id:
            continue
        scaffold_result = next(
            (
                result
                for result in (variant.get("split_results") or [])
                if result.get("strategy_label") == "scaffold"
            ),
            None,
        )
        if not scaffold_result or not scaffold_result.get("test_predictions_path"):
            continue
        key = f"parity_scaffold_{variant_id}"
        generated[key] = _plot_variant_parity_from_predictions(
            Path(str(scaffold_result["test_predictions_path"])).expanduser(),
            comparison_dir / f"parity_scaffold_{variant_id}.png",
            label=_short_variant_label(variant_id),
            band_metric="mae",
            recommended=variant_id == activity_cliffs.get("recommended_variant"),
        )
        generated[f"{key}_rmse"] = _plot_variant_parity_from_predictions(
            Path(str(scaffold_result["test_predictions_path"])).expanduser(),
            comparison_dir / f"parity_scaffold_{variant_id}_rmse.png",
            label=_short_variant_label(variant_id),
            band_metric="rmse",
            recommended=variant_id == activity_cliffs.get("recommended_variant"),
        )
    generated["parity_scaffold_loop_grid_mae"] = _plot_variant_parity_grid(
        activity_cliffs,
        comparison_dir / "parity_scaffold_loop_grid_mae.png",
        band_metric="mae",
        recommended_variant=activity_cliffs.get("recommended_variant"),
    )
    generated["parity_scaffold_loop_grid_rmse"] = _plot_variant_parity_grid(
        activity_cliffs,
        comparison_dir / "parity_scaffold_loop_grid_rmse.png",
        band_metric="rmse",
        recommended_variant=activity_cliffs.get("recommended_variant"),
    )
    return {key: value for key, value in generated.items() if value}


def _variant_id(loop_index: int, removed_tiers: List[str]) -> str:
    return f"filtered_loop_{loop_index}_drop_{'_'.join(removed_tiers)}"


def _write_variants(
    *,
    annotated: pd.DataFrame,
    output_dir: Path,
    loops: int,
    min_training_rows: int,
) -> tuple[List[Dict[str, Any]], List[str]]:
    variants: List[Dict[str, Any]] = [
        {
            "variant_id": "baseline_loop_0",
            "loop_index": 0,
            "removed_tiers": [],
            "removed_count": 0,
            "removed_row_indices": [],
            "remaining_rows": int(len(annotated)),
            "filtered_training_csv": None,
        }
    ]
    warnings: List[str] = []
    tier_plan = {
        1: ["high"],
        2: ["high", "medium"],
        3: ["high", "medium", "low"],
    }
    clean_source = strip_activity_cliff_columns(annotated)
    for loop_index in range(1, loops + 1):
        removed_tiers = tier_plan[loop_index]
        mask = annotated["activity_cliff_priority_tier"].isin(removed_tiers)
        removed_count = int(mask.sum())
        if removed_count <= 0:
            warnings.append(
                f"Skipped {loop_index}; no compounds belonged to tiers {removed_tiers}."
            )
            continue
        kept = clean_source.loc[~mask].copy().reset_index(drop=True)
        if len(kept) < min_training_rows:
            warnings.append(
                f"Stopped before loop {loop_index}; only {len(kept)} rows would remain."
            )
            break
        variant = _variant_id(loop_index, removed_tiers)
        variant_path = output_dir / f"activity_cliff_{variant}.csv"
        kept.to_csv(variant_path, index=False)
        variants.append(
            {
                "variant_id": variant,
                "loop_index": loop_index,
                "removed_tiers": removed_tiers,
                "removed_count": removed_count,
                "removed_row_indices": [int(idx) for idx in annotated.index[mask].tolist()],
                "remaining_rows": int(len(kept)),
                "filtered_training_csv": str(variant_path),
            }
        )
    return variants, warnings


def prepare_activity_cliff_context(
    *,
    train_csv: str,
    output_dir: str,
    smiles_column: str = "smiles",
    target_column: str,
    activity_cliff_index: str = DEFAULT_ACTIVITY_CLIFF_INDEX,
    activity_cliff_feedback: bool = False,
    activity_cliff_feedback_loops: int = 0,
    activity_cliff_similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    activity_cliff_top_k_neighbors: int = DEFAULT_TOP_K_NEIGHBORS,
    activity_cliff_flag_threshold: float = DEFAULT_FLAG_THRESHOLD,
    min_training_rows: int = MIN_VARIANT_TRAINING_ROWS,
) -> Dict[str, Any]:
    registry = default_registry()
    config = parse_activity_cliff_config(
        activity_cliff_index=activity_cliff_index,
        activity_cliff_feedback=activity_cliff_feedback,
        activity_cliff_feedback_loops=activity_cliff_feedback_loops,
        activity_cliff_similarity_threshold=activity_cliff_similarity_threshold,
        activity_cliff_top_k_neighbors=activity_cliff_top_k_neighbors,
        activity_cliff_flag_threshold=activity_cliff_flag_threshold,
        registry=registry,
    )
    with S3.open(train_csv, "r") as fh:
        dataset = pd.read_csv(fh)
    index = registry.get(config.index_name)
    annotated = index.compute(
        dataset=dataset,
        smiles_column=smiles_column,
        target_column=target_column,
        config=config,
    )
    similarity_diagnostics = dict(
        annotated.attrs.get("activity_cliff_similarity_diagnostics") or {}
    )
    flagged = (
        pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce") >= config.flag_threshold
    ) & (annotated["activity_cliff_neighbor_count"].astype(int) >= 1)
    annotated["activity_cliff_flag"] = flagged.astype(bool)
    annotated["activity_cliff_priority_tier"] = _assign_tiers(
        pd.to_numeric(annotated["activity_cliff_score_norm"], errors="coerce").fillna(0.0),
        annotated["activity_cliff_flag"],
    )
    annotated["activity_cliff_reason_codes"] = annotated.apply(_reason_codes, axis=1)
    annotated["activity_cliff_filter_justification"] = annotated.apply(_justification, axis=1)

    ac_dir = Path(output_dir).expanduser().resolve() / "activity_cliffs"
    ac_dir.mkdir(parents=True, exist_ok=True)
    annotated_path = ac_dir / "activity_cliff_annotated_training.csv"
    annotated.to_csv(annotated_path, index=False)
    clean_training_path = ac_dir / "activity_cliff_training_clean.csv"
    strip_activity_cliff_columns(annotated).to_csv(clean_training_path, index=False)
    plot_artifacts = _write_plots(
        annotated,
        ac_dir,
        flag_threshold=config.flag_threshold,
        similarity_threshold=config.similarity_threshold,
    )
    variants, warnings = _write_variants(
        annotated=annotated,
        output_dir=ac_dir,
        loops=config.feedback_loops,
        min_training_rows=min_training_rows,
    )
    priority_counts = _priority_counts(annotated["activity_cliff_priority_tier"])
    summary = {
        "enabled": True,
        "mode": config.mode,
        "index_name": config.index_name,
        "available_indexes": registry.list_indexes(),
        "source_training_csv": train_csv,
        "clean_training_csv": str(clean_training_path),
        "annotated_training_csv": str(annotated_path),
        "target_column": target_column,
        "smiles_column": smiles_column,
        "ranked_molecule_count": int(len(annotated)),
        "flagged_count": int(annotated["activity_cliff_flag"].sum()),
        "priority_counts": priority_counts,
        "selection_policy": {
            "default_index": DEFAULT_ACTIVITY_CLIFF_INDEX,
            "explicit_index_required_for_non_default": True,
        },
        "index_parameters": {
            "similarity_metric": config.similarity_metric,
            "fingerprint": config.fingerprint_kind,
            "fingerprint_radius": config.fingerprint_radius,
            "fingerprint_dimensions": config.fingerprint_size,
            "fingerprint_bits": None,
            "similarity_threshold": config.similarity_threshold,
            "top_k_neighbors": config.top_k_neighbors,
            "flag_threshold": config.flag_threshold,
            "normalization": "pair_score_p95",
            "representation_note": DEFAULT_ACTIVITY_CLIFF_REPRESENTATION_NOTE,
        },
        "similarity_diagnostics": {
            "exact_similarity_nonidentical_pair_count": int(
                similarity_diagnostics.get("exact_similarity_nonidentical_pair_count") or 0
            ),
            "max_similarity_observed": float(
                similarity_diagnostics.get("max_similarity_observed") or 0.0
            ),
            "exact_similarity_policy": similarity_diagnostics.get(
                "exact_similarity_policy", "reported_not_silently_corrected"
            ),
        },
        "tiering_policy": {
            "source_column": "activity_cliff_score_norm",
            "flag_rule": "score_norm >= flag_threshold and neighbor_count >= 1",
            "high": "top 20 percent of flagged compounds by normalized score",
            "medium": "next 20 percent of flagged compounds by normalized score",
            "low": "remaining flagged compounds",
            "none": "not flagged",
            "oof_residuals_used": False,
        },
        "annotation_columns": [
            column for column in annotated.columns if str(column).startswith("activity_cliff_")
        ],
        "feature_exclusion_prefixes": [ACTIVITY_CLIFF_ANNOTATION_PREFIX],
        "feedback_loops_requested": config.feedback_loops,
        "variants": variants,
        "recommended_variant": None,
        "evaluation_policy": "fixed_holdout_required_for_loop_comparison",
        "plot_artifacts": plot_artifacts,
        "warnings": warnings,
    }
    summary_path = ac_dir / "activity_cliff_summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary
