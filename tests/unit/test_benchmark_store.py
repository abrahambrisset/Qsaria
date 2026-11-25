"""Tests for the long-format benchmark results store."""

from __future__ import annotations

import pandas as pd
import pytest

from cs_copilot.tools.benchmarks.store import (
    RESULT_COLUMNS,
    RESULT_DTYPES,
    BenchmarkStore,
    ResultRow,
)

CREATED_AT = "2026-01-02T03:04:05+00:00"


def _row(**overrides: object) -> ResultRow:
    payload: dict[str, object] = {
        "run_id": "run-001",
        "campaign_id": "campaign-a",
        "dataset": "esol",
        "split": "scaffold",
        "seed": 0,
        "model": "random_forest",
        "representation": "morgan_2_2048",
        "metric": "r2",
        "value": 0.5,
        "created_at": CREATED_AT,
    }
    payload.update(overrides)
    return ResultRow(**payload)  # type: ignore[arg-type]


def _store(tmp_path, name: str = "results") -> BenchmarkStore:
    return BenchmarkStore(tmp_path / name)


# --------------------------------------------------------------------- schema


def test_empty_store_returns_empty_frame_with_full_schema(tmp_path):
    store = _store(tmp_path)

    frame = store.read()

    assert frame.empty
    assert list(frame.columns) == list(RESULT_COLUMNS)
    assert {name: str(dtype) for name, dtype in frame.dtypes.items()} == {
        name: str(pd.Series(dtype=RESULT_DTYPES[name]).dtype) for name in RESULT_COLUMNS
    }
    assert store.datasets() == []
    assert store.models() == []
    assert store.runs() == []
    assert store.summary()["n_rows"] == 0
    assert store.summary()["n_parts"] == 0
    assert not (tmp_path / "results").exists()


def test_column_order_matches_the_declared_schema():
    assert RESULT_COLUMNS == (
        "run_id",
        "campaign_id",
        "dataset",
        "dataset_version",
        "task_type",
        "split",
        "split_family",
        "seed",
        "model",
        "backend",
        "representation",
        "tuning_enabled",
        "tuning_trials",
        "metric",
        "value",
        "ci_low",
        "ci_high",
        "fit_time_s",
        "predict_time_s",
        "n_train",
        "n_test",
        "ad_coverage",
        "conformal_coverage",
        "env_hash",
        "recipe_hash",
        "created_at",
    )


# ------------------------------------------------------------- round-tripping


def test_append_then_read_round_trips_values_dtypes_and_missing_optionals(tmp_path):
    store = _store(tmp_path)
    row = _row(
        dataset_version="v1.3",
        task_type="regression",
        split_family="scaffold",
        backend="sklearn",
        tuning_enabled=True,
        tuning_trials=40,
        value=0.8125,
        ci_low=0.75,
        ci_high=0.875,
        fit_time_s=12.5,
        predict_time_s=0.25,
        n_train=800,
        n_test=228,
        ad_coverage=0.9375,
        env_hash="deadbeef",
        recipe_hash="cafebabe",
    )

    assert store.append([row]) == 1

    frame = store.read()
    assert len(frame) == 1
    stored = frame.iloc[0]

    assert stored["run_id"] == "run-001"
    assert stored["dataset_version"] == "v1.3"
    assert stored["seed"] == 0
    assert bool(stored["tuning_enabled"]) is True
    assert stored["tuning_trials"] == 40
    assert stored["value"] == 0.8125
    assert stored["ci_low"] == 0.75
    assert stored["ci_high"] == 0.875
    assert stored["n_train"] == 800
    assert stored["ad_coverage"] == 0.9375
    assert stored["created_at"] == CREATED_AT

    # Optional fields left unset stay missing instead of becoming 0 / NaN / "".
    assert stored["n_test"] == 228
    assert frame["conformal_coverage"].isna().all()
    assert frame["task_type"].tolist() == ["regression"]

    missing_only = _row(seed=1, run_id="run-002")
    store.append([missing_only])
    second = store.read(run_id="run-002").iloc[0]
    for column in ("backend", "dataset_version", "tuning_enabled", "tuning_trials", "n_train"):
        assert pd.isna(second[column]), column

    # dtypes survive the Parquet round trip exactly.
    assert {name: str(dtype) for name, dtype in store.read().dtypes.items()} == {
        name: str(pd.Series(dtype=RESULT_DTYPES[name]).dtype) for name in RESULT_COLUMNS
    }


def test_appending_twice_neither_loses_nor_duplicates_rows(tmp_path):
    store = _store(tmp_path)

    first = [_row(seed=seed) for seed in (0, 1)]
    second = [_row(seed=seed, run_id="run-002") for seed in (0, 1, 2)]
    assert store.append(first) == 2
    assert store.append(second) == 3

    frame = store.read()
    assert len(frame) == 5
    assert len(store.parts()) == 2
    assert sorted(frame["run_id"].tolist()) == ["run-001"] * 2 + ["run-002"] * 3
    assert sorted(frame[frame["run_id"] == "run-002"]["seed"].tolist()) == [0, 1, 2]
    assert store.append([]) == 0
    assert len(store.read()) == 5


def test_concurrent_appenders_do_not_overwrite_each_others_parts(tmp_path):
    """A second writer on the same directory is a second BenchmarkStore."""

    writer_a = _store(tmp_path)
    writer_b = BenchmarkStore(tmp_path / "results")

    writer_a.append([_row(seed=0)])
    writer_b.append([_row(seed=1)])
    writer_a.append([_row(seed=2)])

    assert len(writer_a.parts()) == 3
    assert sorted(writer_a.read()["seed"].tolist()) == [0, 1, 2]
    assert list((tmp_path / "results").glob(".*tmp")) == []


def test_append_rejects_non_result_rows(tmp_path):
    store = _store(tmp_path)

    with pytest.raises(TypeError):
        store.append([{"dataset": "esol"}])  # type: ignore[list-item]


# ------------------------------------------------------------------- filters


def test_filters_compose_and_unknown_columns_are_refused(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(model="random_forest", split="random", seed=0, value=0.1),
            _row(model="random_forest", split="scaffold", seed=0, value=0.2),
            _row(model="lightgbm", split="scaffold", seed=0, value=0.3),
            _row(model="lightgbm", split="scaffold", seed=1, metric="rmse", value=0.4),
        ]
    )

    assert len(store.read(model="lightgbm")) == 2
    assert len(store.read(model="lightgbm", split="scaffold")) == 2
    assert len(store.read(model="lightgbm", split="scaffold", metric="r2")) == 1
    assert store.read(model="lightgbm", split="scaffold", metric="r2")["value"].iloc[0] == 0.3
    assert len(store.read(model=["lightgbm", "random_forest"], split="scaffold")) == 3
    # A None filter is a no-op so optional filters need no branching.
    assert len(store.read(model=None)) == 4
    assert len(store.read(dataset="not-there")) == 0

    with pytest.raises(ValueError, match="Unknown result column"):
        store.read(mdoel="lightgbm")


def test_inventory_helpers_return_sorted_unique_values(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(run_id="run-b", dataset="lipo", model="lightgbm"),
            _row(run_id="run-a", dataset="esol", model="random_forest"),
            _row(run_id="run-a", dataset="esol", model="lightgbm", seed=1),
        ]
    )

    assert store.datasets() == ["esol", "lipo"]
    assert store.models() == ["lightgbm", "random_forest"]
    assert store.runs() == ["run-a", "run-b"]
    summary = store.summary()
    assert summary["n_rows"] == 3
    assert summary["n_parts"] == 1
    assert summary["n_datasets"] == 2
    assert summary["n_models"] == 2
    assert summary["n_runs"] == 2


# --------------------------------------------------------------- leaderboard


def test_leaderboard_aggregates_three_seeds_with_closed_form_mean_and_std(tmp_path):
    store = _store(tmp_path)
    # 0.25, 0.5, 0.75 are exact in binary floating point:
    #   mean = 1.5 / 3            = 0.5
    #   var  = (0.0625 + 0 + 0.0625) / (3 - 1) = 0.0625  (sample, ddof=1)
    #   std  = sqrt(0.0625)       = 0.25
    store.append(
        [
            _row(seed=0, value=0.25, ci_low=0.0, ci_high=0.5),
            _row(seed=1, value=0.50, ci_low=0.25, ci_high=0.75),
            _row(seed=2, value=0.75, ci_low=0.5, ci_high=1.0),
        ]
    )

    board = store.leaderboard("esol", "r2")

    assert len(board) == 1
    best = board.iloc[0]
    assert best["dataset"] == "esol"
    assert best["split"] == "scaffold"
    assert best["model"] == "random_forest"
    assert best["metric"] == "r2"
    assert float(best["mean"]) == 0.5
    assert float(best["std"]) == 0.25
    assert int(best["n_seeds"]) == 3
    assert float(best["ci_low"]) == 0.25
    assert float(best["ci_high"]) == 0.75


def test_leaderboard_sorts_by_higher_is_better_in_both_directions(tmp_path):
    store = _store(tmp_path)
    for model, values in (("worst", (0.1, 0.2)), ("middle", (0.4, 0.6)), ("best", (0.9, 0.9))):
        store.append(
            [_row(model=model, seed=seed, value=value) for seed, value in enumerate(values)]
        )

    descending = store.leaderboard("esol", "r2", higher_is_better=True)
    ascending = store.leaderboard("esol", "r2", higher_is_better=False)

    assert descending["model"].tolist() == ["best", "middle", "worst"]
    assert ascending["model"].tolist() == ["worst", "middle", "best"]
    assert descending["mean"].tolist() == pytest.approx([0.9, 0.5, 0.15])


def test_leaderboard_keeps_split_and_single_seed_rows_honest(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(split="random", seed=0, value=0.9),
            _row(split="scaffold", seed=0, value=0.4),
            _row(split="scaffold", seed=1, value=0.6),
        ]
    )

    both = store.leaderboard("esol", "r2")
    assert set(both["split"]) == {"random", "scaffold"}
    single = both[both["split"] == "random"].iloc[0]
    # One seed gives no sample standard deviation; it must not be reported as 0.
    assert pd.isna(single["std"])
    assert int(single["n_seeds"]) == 1
    # Missing confidence bounds stay missing rather than becoming 0.
    assert pd.isna(single["ci_low"])

    only_scaffold = store.leaderboard("esol", "r2", split="scaffold")
    assert len(only_scaffold) == 1
    assert float(only_scaffold.iloc[0]["mean"]) == pytest.approx(0.5)


def test_leaderboard_on_an_empty_match_returns_an_empty_frame(tmp_path):
    store = _store(tmp_path)
    store.append([_row()])

    board = store.leaderboard("not-a-dataset", "r2")

    assert board.empty
    assert "mean" in board.columns
    assert "n_seeds" in board.columns


# ------------------------------------------------------------------- compare


def test_compare_returns_per_seed_deltas_aligned_on_seed(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(model="a", seed=0, value=0.75),
            _row(model="a", seed=1, value=0.50),
            _row(model="a", seed=2, value=0.25),
            _row(model="b", seed=2, value=0.125),
            _row(model="b", seed=1, value=0.25),
            _row(model="b", seed=0, value=0.50),
        ]
    )

    result = store.compare("esol", "r2", "a", "b")

    assert result["model_a"] == "a"
    assert result["model_b"] == "b"
    assert result["seeds"] == [0, 1, 2]
    assert result["values_a"] == [0.75, 0.50, 0.25]
    assert result["values_b"] == [0.50, 0.25, 0.125]
    assert result["deltas"] == pytest.approx([0.25, 0.25, 0.125])
    assert result["n_paired"] == 3
    assert result["mean_delta"] == pytest.approx(0.625 / 3)
    assert result["std_delta"] == pytest.approx(0.0721687836487, rel=1e-9)
    assert result["unpaired_seeds"] == []


def test_compare_reports_unpaired_seeds_instead_of_dropping_them(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(model="a", seed=0, value=0.5),
            _row(model="a", seed=1, value=0.5),
            _row(model="b", seed=1, value=0.25),
            _row(model="b", seed=7, value=0.25),
        ]
    )

    result = store.compare("esol", "r2", "a", "b")

    assert result["seeds"] == [1]
    assert result["deltas"] == pytest.approx([0.25])
    assert result["n_paired"] == 1
    assert result["std_delta"] is None
    assert result["unpaired_seeds"] == [0, 7]


def test_compare_refuses_to_average_over_unspecified_configurations(tmp_path):
    store = _store(tmp_path)
    store.append(
        [
            _row(model="a", representation="morgan", seed=0, value=0.9),
            _row(model="a", representation="rdkit_descriptors", seed=0, value=0.1),
            _row(model="b", representation="morgan", seed=0, value=0.5),
        ]
    )

    with pytest.raises(ValueError, match="several rows for seed"):
        store.compare("esol", "r2", "a", "b")

    narrowed = store.compare("esol", "r2", "a", "b", representation="morgan")
    assert narrowed["deltas"] == pytest.approx([0.4])


def test_compare_without_overlap_returns_no_deltas(tmp_path):
    store = _store(tmp_path)
    store.append([_row(model="a", seed=0, value=0.5)])

    result = store.compare("esol", "r2", "a", "missing_model")

    assert result["n_paired"] == 0
    assert result["deltas"] == []
    assert result["mean_delta"] is None


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "field", ["run_id", "dataset", "split", "seed", "model", "representation", "metric", "value"]
)
def test_result_row_requires_the_identity_of_a_measurement(field):
    with pytest.raises(ValueError, match=field):
        _row(**{field: None})


def test_result_row_requires_a_caller_supplied_timestamp():
    with pytest.raises(ValueError, match="created_at"):
        _row(created_at=None)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_result_row_refuses_a_non_finite_measurement(bad_value):
    with pytest.raises(ValueError, match="finite"):
        _row(value=bad_value)


def test_result_row_refuses_a_non_finite_optional_metric_field():
    with pytest.raises(ValueError, match="finite"):
        _row(ad_coverage=float("nan"))


@pytest.mark.parametrize(
    ("field", "bad"),
    [("seed", "0"), ("seed", 1.5), ("dataset", 7), ("tuning_enabled", "yes"), ("value", "0.5")],
)
def test_result_row_refuses_wrongly_typed_fields(field, bad):
    with pytest.raises(TypeError):
        _row(**{field: bad})


def test_result_row_normalises_integer_metrics_to_float():
    row = _row(value=1, fit_time_s=3)

    assert isinstance(row.value, float)
    assert isinstance(row.fit_time_s, float)
    assert row.as_dict()["value"] == 1.0
    assert list(row.as_dict()) == list(RESULT_COLUMNS)
