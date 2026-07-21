#!/usr/bin/env python
# coding: utf-8
"""Common post-selection outlier analysis for QSAR training workflows.

The module deliberately owns the scientific policy but not model fitting.  A
backend supplies one prediction for every validation (or out-of-fold) row;
this module adds Activity Cliff / AD annotations, selects rows reproducibly,
and writes compact audit artifacts.  Keeping this logic independent of the
backend makes LightGBM, Chemprop and TabICL follow the same policy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import pandas as pd

OUTLIER_ANALYSIS_CONTRACT_VERSION = "1.1"
OUTLIER_ANALYSIS_SELECTION_FRACTION = 0.10
OUTLIER_ANALYSIS_RMSE_MULTIPLIER = 2.0

# Keep colour and shape as independent signals: the plots must remain readable
# when colours are close on screen or unavailable to a colour-blind reader.
# This shared mapping is used unchanged by both validation/OOF outlier plots.
OUTLIER_PLOT_MARKER_STYLES: Mapping[str, Mapping[str, Any]] = {
    "unflagged": {"color": "#4c78a8", "marker": "o", "size": 30},
    "AC only": {"color": "#d81b60", "marker": "*", "size": 88},
    "AD only": {"color": "#f28e2b", "marker": "^", "size": 52},
    "AC + AD": {"color": "#d62728", "marker": "D", "size": 48},
}


class OutlierAnalysisError(ValueError):
    """Raised when a public outlier-analysis request is invalid."""


@dataclass(frozen=True)
class OutlierAnalysisConfig:
    """The deliberately small public configuration for post-selection removal."""

    enabled: bool = True
    selection_fraction: float = OUTLIER_ANALYSIS_SELECTION_FRACTION

    def as_dict(self) -> Dict[str, Any]:
        return {
            "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
            "enabled": self.enabled,
            "selection_fraction": self.selection_fraction,
            "rmse_multiplier": OUTLIER_ANALYSIS_RMSE_MULTIPLIER,
        }


def normalize_outlier_analysis_config(
    raw: Optional[Mapping[str, Any]],
    *,
    has_validation: bool,
    target_count: int,
    activity_cliff_feedback: bool = False,
) -> tuple[OutlierAnalysisConfig, Optional[str]]:
    """Validate V1 config and return a deterministic skip reason when needed."""
    if raw is None:
        config = OutlierAnalysisConfig()
    elif not isinstance(raw, Mapping):
        raise OutlierAnalysisError(
            "outlier_analysis must be an object such as "
            "{'enabled': false} or {'selection_fraction': 1.0}."
        )
    else:
        unknown = set(raw) - {"enabled", "selection_fraction"}
        if unknown:
            raise OutlierAnalysisError(
                "Unknown outlier_analysis option(s): " + ", ".join(sorted(map(str, unknown)))
            )
        value = raw.get("enabled", True)
        if not isinstance(value, bool):
            raise OutlierAnalysisError("outlier_analysis.enabled must be a boolean.")
        config = OutlierAnalysisConfig(
            enabled=value,
            selection_fraction=_normalize_selection_fraction(
                raw.get("selection_fraction", OUTLIER_ANALYSIS_SELECTION_FRACTION)
            ),
        )

    if not config.enabled:
        return config, "Disabled explicitly by outlier_analysis.enabled=false."
    if not has_validation:
        return config, "No validation set is available for post-selection outlier analysis."
    if target_count != 1:
        return config, "Outlier analysis V1 supports single-target training only."
    if activity_cliff_feedback:
        raise OutlierAnalysisError(
            "Activity-cliff feedback loops cannot be combined with outlier analysis in V1."
        )
    return config, None


def _normalize_selection_fraction(value: Any) -> float:
    """Return a safe fraction for eligible regression rows.

    ``1.0`` is deliberately meaningful: it removes every eligible validation
    outlier.  Zero is rejected rather than silently turning the analysis into
    a no-op; callers can use ``enabled=false`` for that explicit choice.
    """
    if isinstance(value, bool):
        raise OutlierAnalysisError(
            "outlier_analysis.selection_fraction must be a number in (0, 1]."
        )
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise OutlierAnalysisError(
            "outlier_analysis.selection_fraction must be a number in (0, 1]."
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise OutlierAnalysisError(
            "outlier_analysis.selection_fraction must be a number in (0, 1]."
        )
    return fraction


def describe_outlier_analysis_policy() -> Dict[str, Any]:
    """Expose the stable V1 policy for reports and backend identity cards."""
    return {
        "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
        "supported": True,
        "single_target_only": True,
        "activation": "automatic_when_validation_available",
        "explicit_opt_out": {"enabled": False},
        "explicit_configuration": {
            "selection_fraction": {
                "default": OUTLIER_ANALYSIS_SELECTION_FRACTION,
                "valid_range": "(0, 1]",
                "all_eligible": 1.0,
            }
        },
        "regression": {
            "eligibility": "absolute_error > 2 * fold_rmse and (activity_cliff_flag or out_of_domain)",
            "ranking": "absolute_percentage_error_descending",
            "selection_fraction": OUTLIER_ANALYSIS_SELECTION_FRACTION,
            "selection_rounding": "ceil",
        },
        "classification": {"selection": "all_validation_misclassifications"},
    }


def _prediction_columns(frame: pd.DataFrame, target_column: str) -> tuple[str, str]:
    true_candidates = ("y_true", f"{target_column}_true", target_column)
    pred_candidates = ("y_pred", f"{target_column}_prediction", "prediction", target_column)
    true_column = next((name for name in true_candidates if name in frame.columns), None)
    prediction_column = next(
        (name for name in pred_candidates if name in frame.columns and name != true_column), None
    )
    if not true_column or not prediction_column:
        raise OutlierAnalysisError(
            "Validation predictions must expose true and predicted target columns."
        )
    return true_column, prediction_column


def selection_predictions_from_frame(
    prediction_frame: pd.DataFrame,
    *,
    source_row_indices: Sequence[int],
    target_column: str,
    fold_label: str,
    repeat_index: Optional[int] = None,
) -> pd.DataFrame:
    """Normalize one backend validation frame into the shared row-level shape."""
    if len(prediction_frame) != len(source_row_indices):
        raise OutlierAnalysisError(
            "Validation prediction row count does not match its fixed split indices."
        )
    true_column, prediction_column = _prediction_columns(prediction_frame, target_column)
    output = pd.DataFrame(
        {
            "source_row_index": [int(index) for index in source_row_indices],
            "y_true": prediction_frame[true_column].reset_index(drop=True),
            "y_pred": prediction_frame[prediction_column].reset_index(drop=True),
            "fold_label": str(fold_label),
            "repeat_index": repeat_index,
        }
    )
    return output


def attach_ad_annotations(
    selection_frame: pd.DataFrame,
    *,
    ad_statuses: Optional[Sequence[Any]],
) -> pd.DataFrame:
    """Attach AD statuses in validation-row order without making unknown rows OOD."""
    result = selection_frame.copy()
    if ad_statuses is None or len(ad_statuses) != len(result):
        result["ad_status"] = None
        result["ad_out_of_domain"] = False
        return result
    statuses = pd.Series(list(ad_statuses), index=result.index, dtype="object")
    result["ad_status"] = statuses
    result["ad_out_of_domain"] = statuses.notna() & statuses.ne("in_domain")
    return result


def attach_activity_cliff_annotations(
    selection_frame: pd.DataFrame,
    *,
    annotations: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Attach development-only Activity Cliff annotations by source row index."""
    result = selection_frame.copy()
    result["activity_cliff_flag"] = False
    result["activity_cliff_priority_tier"] = "none"
    if annotations is None or annotations.empty:
        return result
    if "source_row_index" not in annotations.columns:
        raise OutlierAnalysisError("Activity Cliff annotations require source_row_index.")
    columns = ["source_row_index"]
    for name in ("activity_cliff_flag", "activity_cliff_priority_tier"):
        if name in annotations.columns:
            columns.append(name)
    lookup = annotations.loc[:, columns].drop_duplicates("source_row_index").copy()
    lookup = lookup.rename(
        columns={
            "activity_cliff_flag": "__ac_flag",
            "activity_cliff_priority_tier": "__ac_tier",
        }
    )
    result = result.merge(lookup, on="source_row_index", how="left", sort=False)
    if "__ac_flag" in result:
        result["activity_cliff_flag"] = result["__ac_flag"].map(
            lambda value: bool(value) if pd.notna(value) else False
        )
        result = result.drop(columns="__ac_flag")
    if "__ac_tier" in result:
        result["activity_cliff_priority_tier"] = result["__ac_tier"].fillna("none")
        result = result.drop(columns="__ac_tier")
    return result


def _regression_selection(
    frame: pd.DataFrame, *, selection_fraction: float
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    result = frame.copy()
    result["y_true"] = pd.to_numeric(result["y_true"], errors="coerce")
    result["y_pred"] = pd.to_numeric(result["y_pred"], errors="coerce")
    result = result.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    result["residual"] = result["y_true"] - result["y_pred"]
    result["absolute_error"] = result["residual"].abs()

    fold_rmse: Dict[str, float] = {}
    for fold_label, rows in result.groupby("fold_label", sort=False):
        rmse = float((rows["residual"].pow(2).mean()) ** 0.5)
        fold_rmse[str(fold_label)] = rmse
    result["fold_rmse"] = result["fold_label"].map(fold_rmse)
    result["error_threshold"] = OUTLIER_ANALYSIS_RMSE_MULTIPLIER * result["fold_rmse"]
    result["error_exceeds_threshold"] = result["absolute_error"] > result["error_threshold"]
    if "activity_cliff_flag" not in result:
        result["activity_cliff_flag"] = False
    if "ad_out_of_domain" not in result:
        result["ad_out_of_domain"] = False
    result["flagged_by_ac_or_ad"] = result["activity_cliff_flag"].fillna(False).astype(
        bool
    ) | result["ad_out_of_domain"].fillna(False).astype(bool)
    result["eligible"] = result["error_exceeds_threshold"] & result["flagged_by_ac_or_ad"]

    denominator = result["y_true"].abs()
    result["ape_percent"] = pd.NA
    result.loc[result["eligible"], "ape_percent"] = (
        result.loc[result["eligible"], "absolute_error"]
        / denominator.loc[result["eligible"]]
        * 100.0
    )
    result["ape_zero_actual"] = (
        result["eligible"] & denominator.eq(0.0) & result["absolute_error"].gt(0.0)
    )
    result.loc[result["ape_zero_actual"], "ape_percent"] = pd.NA
    result["__ape_sort"] = pd.to_numeric(result["ape_percent"], errors="coerce")
    result.loc[result["ape_zero_actual"], "__ape_sort"] = math.inf

    eligible = result.loc[result["eligible"]].sort_values(
        ["__ape_sort", "source_row_index"], ascending=[False, True], kind="stable"
    )
    selected_count = int(math.ceil(len(eligible) * selection_fraction))
    selected_indices = set(eligible.head(selected_count).index.tolist())
    result["ape_rank"] = pd.NA
    if not eligible.empty:
        result.loc[eligible.index, "ape_rank"] = list(range(1, len(eligible) + 1))
    result["selected_for_removal"] = result.index.isin(selected_indices)
    result = result.drop(columns="__ape_sort")
    summary = {
        "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
        "task_type": "regression",
        "fold_rmse": fold_rmse,
        "rmse_multiplier": OUTLIER_ANALYSIS_RMSE_MULTIPLIER,
        "selection_fraction": selection_fraction,
        "eligible_count": int(result["eligible"].sum()),
        "selected_count": int(result["selected_for_removal"].sum()),
    }
    return result, summary


def _classification_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
    result = frame.copy()
    result["misclassified"] = result["y_true"].astype(str) != result["y_pred"].astype(str)
    result["eligible"] = result["misclassified"]
    result["selected_for_removal"] = result["misclassified"]
    result["ape_percent"] = pd.NA
    result["ape_zero_actual"] = False
    result["ape_rank"] = pd.NA
    return result, {
        "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
        "task_type": "classification",
        "eligible_count": int(result["eligible"].sum()),
        "selected_count": int(result["selected_for_removal"].sum()),
    }


def select_outliers(
    selection_frame: pd.DataFrame,
    *,
    task_type: str,
    selection_fraction: float = OUTLIER_ANALYSIS_SELECTION_FRACTION,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the shared policy to normalized validation/OOF predictions."""
    selection_fraction = _normalize_selection_fraction(selection_fraction)
    required = {"source_row_index", "y_true", "y_pred", "fold_label"}
    missing = sorted(required - set(selection_frame.columns))
    if missing:
        raise OutlierAnalysisError("Selection predictions are missing: " + ", ".join(missing))
    if selection_frame.empty:
        empty = selection_frame.copy()
        empty["eligible"] = pd.Series(dtype=bool)
        empty["selected_for_removal"] = pd.Series(dtype=bool)
        return empty, {
            "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
            "task_type": task_type,
            "eligible_count": 0,
            "selected_count": 0,
        }
    if str(task_type).lower() in {
        "classification",
        "binary_classification",
        "multiclass",
        "multiclass_classification",
    }:
        return _classification_selection(selection_frame)
    return _regression_selection(selection_frame, selection_fraction=selection_fraction)


def filtered_development_frame(
    development_frame: pd.DataFrame,
    *,
    selected_source_indices: Iterable[int],
) -> pd.DataFrame:
    """Return the reproducible development set after only selected rows are removed."""
    selected = {int(index) for index in selected_source_indices}
    return development_frame.loc[~development_frame.index.isin(selected)].copy()


def deduplicate_parameter_configurations(
    configurations: Sequence[Mapping[str, Any]],
) -> list[Dict[str, Any]]:
    """Group equivalent effective parameter sets while retaining all originating folds."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for item in configurations:
        params = dict(item.get("parameters") or item.get("params") or {})
        key = json.dumps(params, sort_keys=True, default=str, separators=(",", ":"))
        entry = grouped.setdefault(
            key,
            {
                "configuration_id": f"config_{len(grouped) + 1}",
                "parameters": params,
                "source_folds": [],
            },
        )
        source = {
            key: item.get(key)
            for key in ("fold_label", "repeat_index", "trial_number", "objective")
            if item.get(key) is not None
        }
        entry["source_folds"].append(source)
    return list(grouped.values())


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def build_outlier_analysis_plots(
    selection_frame: pd.DataFrame,
    *,
    output_dir: str | Path,
    title_suffix: str = "validation selection",
) -> Dict[str, Any]:
    """Write the two requested regression selection plots when data is available."""
    if selection_frame.empty or "residual" not in selection_frame.columns:
        return {}
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a project dependency
        return {}

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frame = selection_frame.copy()

    def _bool_column(name: str) -> pd.Series:
        if name not in frame.columns:
            return pd.Series(False, index=frame.index, dtype=bool)
        return frame[name].fillna(False).astype(bool)

    frame["marker_group"] = "unflagged"
    ac_flags = _bool_column("activity_cliff_flag")
    ad_flags = _bool_column("ad_out_of_domain")
    frame.loc[ac_flags, "marker_group"] = "AC only"
    frame.loc[ad_flags, "marker_group"] = "AD only"
    both = ac_flags & ad_flags
    frame.loc[both, "marker_group"] = "AC + AD"
    rmse_values = (
        pd.to_numeric(frame["fold_rmse"], errors="coerce")
        if "fold_rmse" in frame.columns
        else pd.Series(dtype=float)
    )
    max_rmse = float(rmse_values.max()) if not rmse_values.dropna().empty else 0.0

    parity_path = output / "outlier_selection_observed_vs_predicted.png"
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for group, style in OUTLIER_PLOT_MARKER_STYLES.items():
        rows = frame.loc[frame["marker_group"] == group]
        if not rows.empty:
            ax.scatter(
                rows["y_true"],
                rows["y_pred"],
                s=style["size"],
                marker=style["marker"],
                alpha=0.82,
                color=style["color"],
                edgecolors="white",
                linewidths=0.45,
                label=group,
            )
    selected = frame.loc[_bool_column("selected_for_removal")]
    if not selected.empty:
        ax.scatter(
            selected["y_true"],
            selected["y_pred"],
            s=78,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.6,
            label="selected outlier",
            zorder=4,
        )
    low = float(min(frame["y_true"].min(), frame["y_pred"].min()))
    high = float(max(frame["y_true"].max(), frame["y_pred"].max()))
    ax.plot([low, high], [low, high], "--", color="#444444", linewidth=1.1, label="ideal")
    if max_rmse > 0:
        x = pd.Series([low, high])
        ax.fill_between(
            x,
            x - 2 * max_rmse,
            x + 2 * max_rmse,
            color="#9db7c9",
            alpha=0.15,
            label="±2 RMSE (max fold)",
        )
    ax.set_title(f"Observed vs predicted — {title_suffix}")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(parity_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    residual_path = output / "outlier_selection_residuals_vs_observed.png"
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for group, style in OUTLIER_PLOT_MARKER_STYLES.items():
        rows = frame.loc[frame["marker_group"] == group]
        if not rows.empty:
            ax.scatter(
                rows["y_true"],
                rows["residual"],
                s=style["size"],
                marker=style["marker"],
                alpha=0.82,
                color=style["color"],
                edgecolors="white",
                linewidths=0.45,
                label=group,
            )
    if not selected.empty:
        ax.scatter(
            selected["y_true"],
            selected["residual"],
            s=78,
            facecolors="none",
            edgecolors="#d62728",
            linewidths=1.6,
            label="selected outlier",
            zorder=4,
        )
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.1)
    if max_rmse > 0:
        ax.axhline(2 * max_rmse, color="#9db7c9", linestyle=":", label="±2 RMSE (max fold)")
        ax.axhline(-2 * max_rmse, color="#9db7c9", linestyle=":")
    ax.set_title(f"Residuals vs observed — {title_suffix}")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Residual (observed − predicted)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(residual_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "outlier_selection_observed_vs_predicted": str(parity_path),
        "outlier_selection_residuals_vs_observed": str(residual_path),
    }


def write_outlier_analysis_artifacts(
    *,
    output_dir: str | Path,
    selection_frame: pd.DataFrame,
    selection_summary: Mapping[str, Any],
    development_frame: pd.DataFrame,
    extra_summary: Optional[Mapping[str, Any]] = None,
) -> Dict[str, str]:
    """Persist the minimal reproducibility bundle for one outlier study."""
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates_path = output / "outlier_selection_predictions.csv"
    selection_frame.drop(columns=["__ape_sort"], errors="ignore").to_csv(
        candidates_path, index=False
    )
    selected_mask = (
        selection_frame["selected_for_removal"].fillna(False).astype(bool)
        if "selected_for_removal" in selection_frame.columns
        else pd.Series(False, index=selection_frame.index, dtype=bool)
    )
    selected = selection_frame.loc[selected_mask]
    filtered = filtered_development_frame(
        development_frame,
        selected_source_indices=selected.get("source_row_index", pd.Series(dtype=int)).tolist(),
    )
    filtered_path = output / "outlier_filtered_development.csv"
    filtered.to_csv(filtered_path, index=True, index_label="source_row_index")
    plots = build_outlier_analysis_plots(selection_frame, output_dir=output / "plots")

    def _count_true(column: str) -> int:
        if column not in selection_frame.columns:
            return 0
        return int(selection_frame[column].fillna(False).astype(bool).sum())

    # The Activity Cliff annotation used for outlier selection is deliberately
    # a development-only analysis.  It is distinct from optional feedback
    # loops, so persist it explicitly rather than leaving reports to infer it
    # from the presence of AC plot markers.
    selection_annotations = {
        "activity_cliffs": {
            "executed": "activity_cliff_flag" in selection_frame.columns,
            "flagged_count": _count_true("activity_cliff_flag"),
            "scope": "development_only",
        },
        "applicability_domain": {
            "available": "ad_out_of_domain" in selection_frame.columns,
            "out_of_domain_count": _count_true("ad_out_of_domain"),
            "scope": "train_fit_validation_score",
        },
    }
    summary = {
        "contract_version": OUTLIER_ANALYSIS_CONTRACT_VERSION,
        **dict(selection_summary),
        "selected_source_row_indices": [
            int(value) for value in selected.get("source_row_index", [])
        ],
        "selection_predictions_path": str(candidates_path),
        "filtered_development_path": str(filtered_path),
        "plot_artifacts": plots,
        "selection_annotations": selection_annotations,
        **dict(extra_summary or {}),
    }
    summary_path = output / "outlier_analysis_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n")
    return {
        "summary_path": str(summary_path),
        "selection_predictions_path": str(candidates_path),
        "filtered_development_path": str(filtered_path),
        "selection_annotations": selection_annotations,
        **plots,
    }


def write_outlier_variant_comparison(
    *,
    output_dir: str | Path,
    variants: Sequence[Mapping[str, Any]],
    selected_count: int,
) -> str:
    """Persist a descriptive-only baseline/filtered comparison table.

    The test set can be present in these metrics, but the table deliberately
    stores no ranking/recommendation field: it is an audit, never a model
    selector.
    """
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, Any]] = []
    for item in variants:
        run = item.get("run") if isinstance(item, Mapping) else None
        rows.append(
            {
                "variant_id": item.get("variant_id"),
                "status": item.get("status", "completed"),
                "reason": item.get("reason"),
                "selected_outlier_count": int(selected_count),
                "test_metrics": json.dumps(
                    _json_safe(((run or {}).get("metrics") or {}).get("test") or {}),
                    sort_keys=True,
                ),
                "test_evaluation": "descriptive_only",
            }
        )
    path = output / "outlier_variant_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)
