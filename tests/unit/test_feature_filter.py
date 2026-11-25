from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cs_copilot.tools.prediction import feature_filter as ff_module
from cs_copilot.tools.prediction.feature_filter import FeatureFilter, FeatureFilterError

FEATURES = ["desc_a", "desc_const", "desc_nan", "desc_b"]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "desc_a": [1.0, 2.0, 3.0, 4.0],
            "desc_const": [7.0, 7.0, 7.0, 7.0],
            "desc_nan": [1.0, np.nan, np.nan, 4.0],
            "desc_b": [0.5, 9.0, -3.0, 2.0],
            "smiles": ["C", "CC", "CCC", "CCCC"],
        }
    )


def test_constant_column_is_the_only_one_dropped():
    # NaN rule disabled so this isolates the variance rule.
    report = FeatureFilter(nan_fraction_max=None).fit(_frame(), FEATURES).report()

    assert report["dropped"]["zero_variance"] == ["desc_const"]
    assert report["dropped"]["high_nan"] == []
    assert report["dropped"]["correlated"] == []
    assert report["kept_columns"] == ["desc_a", "desc_nan", "desc_b"]
    assert (report["n_input"], report["n_kept"], report["n_dropped"]) == (4, 3, 1)


def test_variance_threshold_matches_closed_form_population_variance():
    # desc_a = [1, 2, 3, 4]: mean 2.5, population variance = 5/4 = 1.25 exactly.
    frame = _frame()
    assert frame["desc_a"].var(ddof=0) == pytest.approx(1.25)

    just_below = FeatureFilter(variance_min=1.24).fit(frame, ["desc_a", "desc_b"])
    just_above = FeatureFilter(variance_min=1.25).fit(frame, ["desc_a", "desc_b"])

    assert "desc_a" in just_below.kept_columns
    assert just_above.report()["dropped"]["zero_variance"] == ["desc_a"]


def test_nan_fraction_above_threshold_is_dropped():
    # desc_nan is 50% NaN; the default threshold is 0.2.
    report = FeatureFilter(nan_fraction_max=0.2).fit(_frame(), FEATURES).report()
    assert report["dropped"]["high_nan"] == ["desc_nan"]

    relaxed = FeatureFilter(nan_fraction_max=0.5).fit(_frame(), FEATURES).report()
    assert relaxed["dropped"]["high_nan"] == []


def test_correlated_pair_loses_exactly_the_lower_variance_column():
    frame = pd.DataFrame(
        {
            "desc_low": [1.0, 2.0, 3.0, 4.0],  # population variance 1.25
            "desc_high": [10.0, 20.0, 30.0, 40.0],  # perfectly correlated, variance 125.0
            "desc_other": [3.0, -1.0, 4.0, 0.5],
        }
    )

    report = (
        FeatureFilter(correlation_max=0.95)
        .fit(frame, ["desc_low", "desc_high", "desc_other"])
        .report()
    )

    assert report["dropped"]["correlated"] == ["desc_low"]
    assert report["kept_columns"] == ["desc_high", "desc_other"]


def test_correlation_tie_break_is_the_column_name():
    # Identical variance, perfect correlation: the alphabetically first name survives.
    frame = pd.DataFrame({"desc_z": [1.0, 2.0, 3.0], "desc_a": [1.0, 2.0, 3.0]})

    report = FeatureFilter(correlation_max=0.95).fit(frame, ["desc_z", "desc_a"]).report()

    assert report["kept_columns"] == ["desc_a"]
    assert report["dropped"]["correlated"] == ["desc_z"]


def test_keep_columns_protects_a_column_that_would_otherwise_be_dropped():
    report = FeatureFilter(keep_columns=("desc_const", "desc_nan")).fit(_frame(), FEATURES).report()

    assert report["dropped"]["zero_variance"] == []
    assert report["dropped"]["high_nan"] == []
    assert report["kept_columns"] == FEATURES


def test_protected_column_still_evicts_its_correlated_partner():
    frame = pd.DataFrame(
        {
            "desc_low": [1.0, 2.0, 3.0, 4.0],
            "desc_high": [10.0, 20.0, 30.0, 40.0],
        }
    )

    report = (
        FeatureFilter(correlation_max=0.95, keep_columns=("desc_low",))
        .fit(frame, ["desc_low", "desc_high"])
        .report()
    )

    assert report["kept_columns"] == ["desc_low"]
    assert report["dropped"]["correlated"] == ["desc_high"]


def test_transform_raises_when_a_learned_column_is_missing():
    fitted = FeatureFilter().fit(_frame(), FEATURES)

    with pytest.raises(FeatureFilterError, match="missing at transform time"):
        fitted.transform(_frame().drop(columns=["desc_a"]))


def test_transform_ignores_extra_columns_and_preserves_learned_order():
    fitted = FeatureFilter().fit(_frame(), FEATURES)
    incoming = _frame().assign(desc_new=1.0)
    # Shuffle the column order to prove transform projects rather than passes through.
    incoming = incoming[list(reversed(incoming.columns))]

    out = fitted.transform(incoming)

    assert list(out.columns) == fitted.kept_columns
    assert "smiles" not in out.columns
    assert "desc_new" not in out.columns


def test_transform_before_fit_raises():
    with pytest.raises(FeatureFilterError, match="before fit"):
        FeatureFilter().transform(_frame())


def test_round_trip_determinism():
    first = FeatureFilter(correlation_max=0.9).fit(_frame(), FEATURES).report()
    second = FeatureFilter(correlation_max=0.9).fit(_frame(), FEATURES).report()

    assert first == second


def test_non_numeric_columns_are_never_coerced_or_variance_filtered():
    frame = pd.DataFrame(
        {
            "label": ["a", "a", "a", "a"],  # constant, but not numeric
            "desc_a": [1.0, 2.0, 3.0, 4.0],
        }
    )

    out = FeatureFilter(correlation_max=0.95).fit_transform(frame, ["label", "desc_a"])

    assert list(out.columns) == ["label", "desc_a"]
    assert out["label"].tolist() == ["a", "a", "a", "a"]


def test_all_nan_column_is_dropped_by_the_nan_rule_then_by_variance():
    frame = pd.DataFrame({"desc_a": [1.0, 2.0], "desc_void": [np.nan, np.nan]})

    by_nan = FeatureFilter(nan_fraction_max=0.2).fit(frame, ["desc_a", "desc_void"]).report()
    assert by_nan["dropped"]["high_nan"] == ["desc_void"]
    assert by_nan["dropped"]["zero_variance"] == []

    by_variance = FeatureFilter(nan_fraction_max=None).fit(frame, ["desc_a", "desc_void"]).report()
    assert by_variance["dropped"]["high_nan"] == []
    assert by_variance["dropped"]["zero_variance"] == ["desc_void"]


def test_single_row_skips_variance_and_correlation_and_warns():
    frame = _frame().head(1)

    report = (
        FeatureFilter(nan_fraction_max=None, correlation_max=0.95).fit(frame, FEATURES).report()
    )

    assert report["kept_columns"] == FEATURES
    assert any("variance is undefined" in message for message in report["warnings"])
    assert any("correlation is undefined" in message for message in report["warnings"])


def test_correlation_step_is_reported_not_silently_skipped_above_the_cap(monkeypatch):
    monkeypatch.setattr(ff_module, "MAX_CORRELATION_COLUMNS", 1)
    frame = pd.DataFrame({"desc_a": [1.0, 2.0, 3.0], "desc_b": [2.0, 4.0, 6.0]})

    report = FeatureFilter(correlation_max=0.95).fit(frame, ["desc_a", "desc_b"]).report()

    assert report["dropped"]["correlated"] == []
    assert any("correlation filtering skipped" in message for message in report["warnings"])


def test_invalid_configuration_and_inputs_raise():
    with pytest.raises(FeatureFilterError):
        FeatureFilter(nan_fraction_max=1.5)
    with pytest.raises(FeatureFilterError):
        FeatureFilter(correlation_max=-0.1)
    with pytest.raises(FeatureFilterError):
        FeatureFilter(variance_min=-1.0)
    with pytest.raises(FeatureFilterError, match="duplicate"):
        FeatureFilter().fit(_frame(), ["desc_a", "desc_a"])
    with pytest.raises(FeatureFilterError, match="absent"):
        FeatureFilter().fit(_frame(), ["desc_a", "desc_absent"])
    with pytest.raises(FeatureFilterError, match="must not be empty"):
        FeatureFilter().fit(_frame(), [])
