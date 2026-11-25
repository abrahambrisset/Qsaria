#!/usr/bin/env python
# coding: utf-8
"""
Long-format store for benchmark results.

One row per ``(run, dataset, split, seed, model, representation, metric)``.
Everything a campaign normally writes as a separate artefact -- leaderboards,
regression checks, comparison plots -- is a *query* over this one table, so
results can be aggregated across datasets and across dates.

The store is append-only.  Each :meth:`BenchmarkStore.append` writes its own
Parquet part file under the store directory; existing parts are never
rewritten, which makes concurrent appends from separate processes safe.  A part
is written to a temporary file in the same directory and moved into place with
:func:`os.replace`, so a reader never observes a partial part.

Reproducibility: the store never calls ``datetime.now()``.  ``created_at`` is
supplied by the caller on every row.  A campaign can therefore be replayed from
its own manifest and produce byte-identical rows, and a replayed run is
distinguishable from a fresh one only by the values the caller chose to change.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = [
    "RESULT_COLUMNS",
    "RESULT_DTYPES",
    "REQUIRED_FIELDS",
    "ResultRow",
    "BenchmarkStore",
    "empty_result_frame",
]


@dataclass
class ResultRow:
    """One measured value of one metric for one model on one split and seed.

    Every field defaults to ``None`` so callers only name what they measured,
    but the identity of a measurement is mandatory: ``run_id``, ``dataset``,
    ``split``, ``seed``, ``model``, ``representation``, ``metric``, ``value``
    and ``created_at`` must all be supplied, and ``value`` must be finite.  A
    metric that could not be computed is not a row -- do not store ``NaN`` as
    if it were a measurement.

    ``created_at`` is an ISO-8601 timestamp **passed in by the caller**; the
    store never generates one, so replaying a campaign reproduces its rows
    exactly.
    """

    run_id: str | None = None
    campaign_id: str | None = None
    dataset: str | None = None
    dataset_version: str | None = None
    task_type: str | None = None
    split: str | None = None
    split_family: str | None = None
    seed: int | None = None
    model: str | None = None
    backend: str | None = None
    representation: str | None = None
    tuning_enabled: bool | None = None
    tuning_trials: int | None = None
    metric: str | None = None
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    fit_time_s: float | None = None
    predict_time_s: float | None = None
    n_train: int | None = None
    n_test: int | None = None
    ad_coverage: float | None = None
    conformal_coverage: float | None = None
    env_hash: str | None = None
    recipe_hash: str | None = None
    created_at: str | None = None

    def __post_init__(self) -> None:
        for name in REQUIRED_FIELDS:
            if getattr(self, name) is None:
                raise ValueError(f"ResultRow requires `{name}`; got None.")
        for name in _STRING_COLUMNS:
            raw = getattr(self, name)
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise TypeError(f"ResultRow.{name} must be a string, got {type(raw).__name__}.")
            if name in REQUIRED_FIELDS and not raw.strip():
                raise ValueError(f"ResultRow.{name} must not be blank.")
        for name in _INT_COLUMNS:
            raw = getattr(self, name)
            if raw is not None and (isinstance(raw, bool) or not isinstance(raw, int)):
                raise TypeError(f"ResultRow.{name} must be an int, got {type(raw).__name__}.")
        if not isinstance(self.tuning_enabled, (bool, type(None))):
            raise TypeError("ResultRow.tuning_enabled must be a bool or None.")
        for name in _FLOAT_COLUMNS:
            raw = getattr(self, name)
            if raw is None:
                continue
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise TypeError(f"ResultRow.{name} must be a number, got {type(raw).__name__}.")
            if not math.isfinite(float(raw)):
                raise ValueError(f"ResultRow.{name} must be finite; got {raw!r}.")
            setattr(self, name, float(raw))

    def as_dict(self) -> dict[str, Any]:
        """Return the row as a plain mapping in canonical column order."""

        payload = asdict(self)
        return {column: payload[column] for column in RESULT_COLUMNS}


RESULT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(ResultRow))

_STRING_COLUMNS: tuple[str, ...] = (
    "run_id",
    "campaign_id",
    "dataset",
    "dataset_version",
    "task_type",
    "split",
    "split_family",
    "model",
    "backend",
    "representation",
    "metric",
    "env_hash",
    "recipe_hash",
    "created_at",
)
_INT_COLUMNS: tuple[str, ...] = ("seed", "tuning_trials", "n_train", "n_test")
_FLOAT_COLUMNS: tuple[str, ...] = (
    "value",
    "ci_low",
    "ci_high",
    "fit_time_s",
    "predict_time_s",
    "ad_coverage",
    "conformal_coverage",
)
_BOOL_COLUMNS: tuple[str, ...] = ("tuning_enabled",)

#: Fields without which a row is not a measurement.
REQUIRED_FIELDS: tuple[str, ...] = (
    "run_id",
    "dataset",
    "split",
    "seed",
    "model",
    "representation",
    "metric",
    "value",
    "created_at",
)

#: Nullable pandas dtypes, so a missing optional value round-trips as ``pd.NA``
#: instead of degrading to ``NaN`` in an integer column or to ``0``.
RESULT_DTYPES: dict[str, str] = {
    **dict.fromkeys(_STRING_COLUMNS, "string"),
    **dict.fromkeys(_INT_COLUMNS, "Int64"),
    **dict.fromkeys(_FLOAT_COLUMNS, "Float64"),
    **dict.fromkeys(_BOOL_COLUMNS, "boolean"),
}

_LEADERBOARD_KEYS: tuple[str, ...] = ("dataset", "split", "model", "representation", "backend")
_LEADERBOARD_COLUMNS: tuple[str, ...] = (
    *_LEADERBOARD_KEYS,
    "metric",
    "mean",
    "std",
    "n_seeds",
    "ci_low",
    "ci_high",
)
_LEADERBOARD_DTYPES: dict[str, str] = {
    **{column: RESULT_DTYPES[column] for column in _LEADERBOARD_KEYS},
    "metric": "string",
    "mean": "Float64",
    "std": "Float64",
    "n_seeds": "Int64",
    "ci_low": "Float64",
    "ci_high": "Float64",
}
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def empty_result_frame() -> pd.DataFrame:
    """Return an empty frame carrying the full schema and its dtypes."""

    return pd.DataFrame(
        {column: pd.Series(dtype=RESULT_DTYPES[column]) for column in RESULT_COLUMNS}
    )


def _conform(frame: pd.DataFrame) -> pd.DataFrame:
    """Reindex to the canonical columns and apply the nullable dtypes."""

    return frame.reindex(columns=list(RESULT_COLUMNS)).astype(RESULT_DTYPES)


def _as_filter_values(wanted: Any) -> list[Any]:
    if isinstance(wanted, (str, bytes)) or not isinstance(wanted, Iterable):
        return [wanted]
    return list(wanted)


class BenchmarkStore:
    """A directory of append-only Parquet parts holding long-format results.

    Parameters
    ----------
    path:
        Store directory.  It is created on the first :meth:`append`; reading a
        store that does not exist yet yields an empty frame rather than an
        error.

    Notes
    -----
    The store never stamps a timestamp itself -- see ``ResultRow.created_at``.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path).expanduser()
        self._counter = itertools.count()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BenchmarkStore({str(self.path)!r})"

    # ------------------------------------------------------------------ write

    def append(self, rows: Iterable[ResultRow]) -> int:
        """Write ``rows`` as one new Parquet part and return how many landed.

        The part is written to a hidden temporary file in the store directory
        and moved into place with :func:`os.replace`, so a concurrent reader
        sees either no part or the whole part.  Part names are unique, so two
        processes appending at the same time cannot overwrite each other.
        """

        payloads = []
        for row in rows:
            if not isinstance(row, ResultRow):
                raise TypeError(f"append() expects ResultRow instances, got {type(row).__name__}.")
            payloads.append(row.as_dict())
        if not payloads:
            return 0

        frame = _conform(pd.DataFrame(payloads))
        self.path.mkdir(parents=True, exist_ok=True)
        target = self.path / self._part_name(str(payloads[0]["run_id"]))
        tmp_path = self.path / f".{target.name}.tmp"
        try:
            with tmp_path.open("wb") as handle:
                frame.to_parquet(handle, index=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            tmp_path = None  # type: ignore[assignment]
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return len(frame)

    def _part_name(self, run_id: str) -> str:
        """Build a part name from the run id, an append counter and a nonce.

        The counter orders the parts one store instance writes; the nonce keeps
        parts from colliding when several processes append concurrently.
        """

        slug = _UNSAFE_NAME.sub("-", run_id).strip("-")[:64] or "run"
        return f"{slug}-{next(self._counter):06d}-{uuid.uuid4().hex[:12]}.parquet"

    def parts(self) -> list[Path]:
        """Return the Parquet parts of this store, in stable name order."""

        if not self.path.is_dir():
            return []
        return sorted(self.path.glob("*.parquet"))

    # ------------------------------------------------------------------- read

    def read(self, **filters: Any) -> pd.DataFrame:
        """Return every stored row, optionally filtered column by column.

        Each keyword names a schema column -- ``dataset``, ``model``, ``split``,
        ``metric``, ``run_id`` and the rest -- and takes either one value or an
        iterable of accepted values.  Filters compose with AND.  A keyword whose
        value is ``None`` is ignored, so optional filters need no branching at
        the call site.
        """

        unknown = sorted(set(filters) - set(RESULT_COLUMNS))
        if unknown:
            raise ValueError(f"Unknown result column(s) in filter: {', '.join(unknown)}")

        parts = self.parts()
        if not parts:
            return empty_result_frame()
        frame = _conform(pd.concat([pd.read_parquet(part) for part in parts], ignore_index=True))

        for column, wanted in filters.items():
            if wanted is None:
                continue
            frame = frame[frame[column].isin(_as_filter_values(wanted))]
        return frame.reset_index(drop=True)

    def leaderboard(
        self,
        dataset: str,
        metric: str,
        split: str | None = None,
        higher_is_better: bool = True,
    ) -> pd.DataFrame:
        """Aggregate one metric across seeds, one row per model configuration.

        Returns ``mean``, sample ``std`` (``ddof=1``, so a single seed gives
        ``NA`` rather than a fake ``0``), ``n_seeds`` as the number of distinct
        seeds, and the mean confidence bounds when the rows carry them.  Rows
        are sorted best-first according to ``higher_is_better``, ties broken by
        model name so the order is deterministic.
        """

        frame = self.read(dataset=dataset, metric=metric, split=split)
        if frame.empty:
            return pd.DataFrame(
                {
                    column: pd.Series(dtype=_LEADERBOARD_DTYPES[column])
                    for column in _LEADERBOARD_COLUMNS
                }
            )

        grouped = frame.groupby(list(_LEADERBOARD_KEYS), dropna=False, sort=False).agg(
            mean=("value", "mean"),
            std=("value", "std"),
            n_seeds=("seed", "nunique"),
            ci_low=("ci_low", "mean"),
            ci_high=("ci_high", "mean"),
        )
        board = grouped.reset_index()
        board["metric"] = metric
        board = board[list(_LEADERBOARD_COLUMNS)].astype(_LEADERBOARD_DTYPES)
        return board.sort_values(
            ["mean", "model"],
            ascending=[not higher_is_better, True],
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)

    def compare(
        self,
        dataset: str,
        metric: str,
        model_a: str,
        model_b: str,
        split: str | None = None,
        *,
        representation: str | None = None,
    ) -> dict[str, Any]:
        """Pair two models seed by seed and return the per-seed deltas.

        Only seeds both models were run on are compared; seeds held by one
        model alone are reported under ``unpaired_seeds`` instead of being
        silently dropped.  ``mean_delta`` is a descriptive summary of
        ``value(model_a) - value(model_b)``, **not** a significance verdict.

        Raises
        ------
        ValueError
            If either model has more than one row per seed after filtering --
            narrow the query with ``split`` or ``representation`` rather than
            letting the store average over configurations you did not choose.
        """

        frame = self.read(
            dataset=dataset,
            metric=metric,
            split=split,
            representation=representation,
            model=[model_a, model_b],
        )
        series_a = _seed_series(frame, model_a)
        series_b = _seed_series(frame, model_b)

        seeds = sorted(set(series_a.index) & set(series_b.index))
        values_a = [float(series_a.loc[seed]) for seed in seeds]
        values_b = [float(series_b.loc[seed]) for seed in seeds]
        deltas = [a - b for a, b in zip(values_a, values_b, strict=True)]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        std_delta = None
        if mean_delta is not None and len(deltas) > 1:
            variance = sum((delta - mean_delta) ** 2 for delta in deltas) / (len(deltas) - 1)
            std_delta = math.sqrt(variance)

        return {
            "dataset": dataset,
            "metric": metric,
            "split": split,
            "representation": representation,
            "model_a": model_a,
            "model_b": model_b,
            "seeds": [int(seed) for seed in seeds],
            "values_a": values_a,
            "values_b": values_b,
            "deltas": deltas,
            "n_paired": len(seeds),
            "mean_delta": mean_delta,
            "std_delta": std_delta,
            "unpaired_seeds": sorted(
                int(seed) for seed in set(series_a.index) ^ set(series_b.index)
            ),
        }

    # ------------------------------------------------------------- inventory

    def datasets(self) -> list[str]:
        """Return the sorted distinct dataset names in the store."""

        return self._distinct("dataset")

    def models(self) -> list[str]:
        """Return the sorted distinct model names in the store."""

        return self._distinct("model")

    def runs(self) -> list[str]:
        """Return the sorted distinct run ids in the store."""

        return self._distinct("run_id")

    def _distinct(self, column: str) -> list[str]:
        values = self.read()[column].dropna().unique().tolist()
        return sorted(str(value) for value in values)

    def summary(self) -> dict[str, Any]:
        """Return row and cardinality counts for the whole store."""

        frame = self.read()
        counts = {
            "n_runs": "run_id",
            "n_campaigns": "campaign_id",
            "n_datasets": "dataset",
            "n_models": "model",
            "n_representations": "representation",
            "n_splits": "split",
            "n_metrics": "metric",
            "n_seeds": "seed",
        }
        summary: dict[str, Any] = {
            "path": str(self.path),
            "n_parts": len(self.parts()),
            "n_rows": int(len(frame)),
        }
        for key, column in counts.items():
            summary[key] = int(frame[column].nunique(dropna=True))
        return summary


def _seed_series(frame: pd.DataFrame, model: str) -> pd.Series:
    """Return ``seed -> value`` for one model, refusing duplicated seeds."""

    subset = frame[frame["model"] == model]
    duplicated = subset["seed"][subset["seed"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(
            f"Model {model!r} has several rows for seed(s) {sorted(duplicated)}; "
            "narrow the comparison with split= or representation=."
        )
    return subset.set_index("seed")["value"]
