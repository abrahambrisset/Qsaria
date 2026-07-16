from __future__ import annotations

import json

import pandas as pd
import pytest

from cs_copilot.tools.prediction.outlier_analysis import (
    OUTLIER_PLOT_MARKER_STYLES,
    OutlierAnalysisError,
    attach_activity_cliff_annotations,
    deduplicate_parameter_configurations,
    normalize_outlier_analysis_config,
    select_outliers,
    selection_predictions_from_frame,
    write_outlier_analysis_artifacts,
)


def _regression_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "source_row_index": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "y_true": [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "y_pred": [1.0, 1.2, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "fold_label": ["fold_1"] * 10,
            "activity_cliff_flag": [False, True, True, True, False, False, False, False, False, False],
            "ad_out_of_domain": [False, False, False, False, False, False, False, False, False, False],
        }
    )


def test_regression_selection_requires_error_and_ac_or_ad_and_rounds_up():
    selected, summary = select_outliers(_regression_rows(), task_type="regression")

    # RMSE is ~0.43; rows 12 and 13 exceed 2xRMSE and are AC flagged.  The
    # top 10% of two eligible rows rounds up to one selected row.
    assert summary["eligible_count"] == 2
    assert summary["selected_count"] == 1
    assert selected.loc[selected["source_row_index"] == 10, "eligible"].item() is False
    assert selected.loc[selected["source_row_index"] == 11, "eligible"].item() is False
    assert selected.loc[selected["source_row_index"] == 13, "ape_zero_actual"].item() is True
    assert selected.loc[selected["source_row_index"] == 13, "selected_for_removal"].item() is True


def test_regression_selection_can_remove_every_eligible_outlier():
    selected, summary = select_outliers(
        _regression_rows(),
        task_type="regression",
        selection_fraction=1.0,
    )

    assert summary["eligible_count"] == 2
    assert summary["selection_fraction"] == 1.0
    assert summary["selected_count"] == 2
    assert selected.loc[selected["eligible"], "selected_for_removal"].all()


def test_regression_selection_uses_per_fold_rmse():
    frame = pd.DataFrame(
        {
            "source_row_index": list(range(12)),
            "y_true": [1.0] * 12,
            "y_pred": [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0],
            "fold_label": ["easy"] * 6 + ["hard"] * 6,
            "activity_cliff_flag": [True] + [False] * 5 + [True] + [False] * 5,
            "ad_out_of_domain": [False] * 12,
        }
    )
    selected, summary = select_outliers(frame, task_type="regression")

    assert set(summary["fold_rmse"]) == {"easy", "hard"}
    assert selected.loc[selected["source_row_index"] == 0, "eligible"].item() is True
    assert selected.loc[selected["source_row_index"] == 6, "eligible"].item() is True
    assert selected.loc[selected["source_row_index"] == 6, "fold_rmse"].item() != selected.loc[
        selected["source_row_index"] == 0, "fold_rmse"
    ].item()


def test_classification_removes_every_misclassification():
    frame = pd.DataFrame(
        {
            "source_row_index": [1, 2, 3],
            "y_true": [0, 1, 1],
            "y_pred": [0, 0, 1],
            "fold_label": ["fold_1"] * 3,
        }
    )
    selected, summary = select_outliers(frame, task_type="classification")

    assert summary["selected_count"] == 1
    assert selected["selected_for_removal"].tolist() == [False, True, False]


def test_classification_still_removes_every_misclassification_with_custom_fraction():
    frame = pd.DataFrame(
        {
            "source_row_index": [1, 2, 3],
            "y_true": [0, 1, 1],
            "y_pred": [1, 0, 1],
            "fold_label": ["fold_1"] * 3,
        }
    )

    selected, summary = select_outliers(
        frame,
        task_type="classification",
        selection_fraction=0.10,
    )

    assert summary["selected_count"] == 2
    assert selected["selected_for_removal"].tolist() == [True, True, False]


def test_ape_is_recorded_only_for_eligible_regression_rows():
    selected, _ = select_outliers(_regression_rows(), task_type="regression")

    assert pd.isna(selected.loc[selected["source_row_index"] == 10, "ape_percent"]).item()
    assert selected.loc[selected["source_row_index"] == 12, "ape_percent"].item() > 0


def test_activity_cliff_annotations_join_by_source_row_index():
    predictions = pd.DataFrame(
        {"source_row_index": [5, 7], "y_true": [1.0, 2.0], "y_pred": [1.0, 2.0], "fold_label": ["f", "f"]}
    )
    annotated = attach_activity_cliff_annotations(
        predictions,
        annotations=pd.DataFrame(
            {
                "source_row_index": [7],
                "activity_cliff_flag": [True],
                "activity_cliff_priority_tier": ["high"],
            }
        ),
    )

    assert annotated["activity_cliff_flag"].tolist() == [False, True]
    assert annotated["activity_cliff_priority_tier"].tolist() == ["none", "high"]


def test_config_skips_or_rejects_incompatible_requests():
    config, reason = normalize_outlier_analysis_config(
        {"enabled": False}, has_validation=True, target_count=1
    )
    assert config.enabled is False
    assert "Disabled explicitly" in reason

    config, reason = normalize_outlier_analysis_config(
        {"selection_fraction": 1.0}, has_validation=True, target_count=1
    )
    assert reason is None
    assert config.selection_fraction == 1.0
    assert config.as_dict()["selection_fraction"] == 1.0

    _, reason = normalize_outlier_analysis_config(None, has_validation=True, target_count=2)
    assert "single-target" in reason

    with pytest.raises(OutlierAnalysisError, match="feedback loops"):
        normalize_outlier_analysis_config(
            None, has_validation=True, target_count=1, activity_cliff_feedback=True
        )

    with pytest.raises(OutlierAnalysisError, match="selection_fraction"):
        normalize_outlier_analysis_config(
            {"selection_fraction": 0}, has_validation=True, target_count=1
        )

    with pytest.raises(OutlierAnalysisError, match="selection_fraction"):
        select_outliers(_regression_rows(), task_type="regression", selection_fraction=1.01)


def test_configuration_deduplication_retains_originating_folds():
    configurations = deduplicate_parameter_configurations(
        [
            {"fold_label": "fold_1", "parameters": {"depth": 6}},
            {"fold_label": "fold_2", "parameters": {"depth": 6}},
            {"fold_label": "fold_3", "parameters": {"depth": 8}},
        ]
    )

    assert len(configurations) == 2
    assert configurations[0]["source_folds"] == [{"fold_label": "fold_1"}, {"fold_label": "fold_2"}]


def test_outlier_plot_styles_use_distinct_colours_and_markers():
    """Both plot types share a visually distinguishable flag encoding."""
    styles = OUTLIER_PLOT_MARKER_STYLES

    assert set(styles) == {"unflagged", "AC only", "AD only", "AC + AD"}
    assert len({style["color"] for style in styles.values()}) == len(styles)
    assert len({style["marker"] for style in styles.values()}) == len(styles)
    assert styles["AC only"]["marker"] == "*"


def test_selection_artifacts_are_compact_and_json_safe(tmp_path):
    selected, summary = select_outliers(_regression_rows(), task_type="regression")
    artifacts = write_outlier_analysis_artifacts(
        output_dir=tmp_path,
        selection_frame=selected,
        selection_summary=summary,
        development_frame=pd.DataFrame({"value": range(20)}, index=range(20)),
    )

    payload = json.loads((tmp_path / "outlier_analysis_summary.json").read_text())
    assert payload["selected_count"] == 1
    assert (tmp_path / "outlier_selection_predictions.csv").exists()
    assert (tmp_path / "outlier_filtered_development.csv").exists()
    assert artifacts["summary_path"].endswith("outlier_analysis_summary.json")


def test_selection_prediction_frame_requires_fixed_index_alignment():
    with pytest.raises(OutlierAnalysisError, match="row count"):
        selection_predictions_from_frame(
            pd.DataFrame({"y_true": [1.0], "y_pred": [1.0]}),
            source_row_indices=[1, 2],
            target_column="Y",
            fold_label="fold_1",
        )
