#!/usr/bin/env python
# coding: utf-8
"""Deterministic feature filtering for tabular QSAR descriptor matrices.

``rdkit_all`` emits the full RDKit descriptor table (~210 columns), many of which
are constant, mutually collinear, or NaN for whole chemotypes.  Tree ensembles
tolerate that; linear and kernel baselines do not, and the bounding-box
applicability domain is degraded by constant columns.

Schema contract (read this before wiring the filter into a pipeline)
--------------------------------------------------------------------
The kept column list becomes part of the model feature schema that the
applicability domain binds to via
:func:`cs_copilot.tools.prediction.applicability_domain.schema_hash`, so
inference must see exactly the training columns.  Therefore:

* :meth:`FeatureFilter.fit` is fit on **training rows only**.
* :meth:`FeatureFilter.transform` is a strict projection: it returns only the
  kept columns, in the order learned at fit time.
* Extra columns present at transform time are **ignored**, never passed through.
* A kept column missing at transform time raises :class:`FeatureFilterError`
  rather than silently yielding a narrower matrix.
* No filtering decision is taken at transform time; two frames with the same
  columns always produce the same output schema.

Filter order and precedence
---------------------------
Rules are applied in a fixed order and a dropped column is attributed to the
first rule that removed it: NaN fraction, then variance, then correlation.
Columns listed in ``keep_columns`` are never dropped by any rule, but they do
take part in correlation filtering as survivors and can evict their collinear
partners.

Documented edge-case behaviour
------------------------------
single row (or empty frame)
    Variance and Pearson correlation are undefined for fewer than two rows.
    Both steps are skipped, every candidate column survives them, and a note is
    recorded in ``report()["warnings"]``.  The filter never guesses.
all-NaN column
    ``nan_fraction`` is ``1.0`` so it is dropped by the NaN rule under any
    ``nan_fraction_max < 1.0``.  If the NaN rule is disabled (or set to
    ``1.0``), its variance is NaN, which counts as zero variance and is dropped
    by the variance rule when ``drop_zero_variance`` is enabled.
non-numeric columns
    Never coerced and never inspected for variance or correlation.  The NaN
    rule still applies, because missingness is dtype-agnostic.
wide matrices
    Correlation filtering is skipped entirely above
    :data:`MAX_CORRELATION_COLUMNS` surviving numeric columns and the skip is
    recorded in ``report()["warnings"]`` -- it is never silent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
from pandas.api.types import is_numeric_dtype

#: Correlation filtering is O(n^2) in columns; above this many surviving
#: numeric columns the step is skipped and reported instead of run.
MAX_CORRELATION_COLUMNS = 1000

DROP_REASONS = ("zero_variance", "high_nan", "correlated")


class FeatureFilterError(ValueError):
    """Raised for an invalid filter configuration or a feature schema mismatch."""


def _check_unit_interval(name: str, value: Optional[float]) -> Optional[float]:
    """Validate an optional threshold constrained to ``[0, 1]``."""
    if value is None:
        return None
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise FeatureFilterError(f"{name} must lie in [0, 1] or be None, got {value!r}.")
    return numeric


class FeatureFilter:
    """Learn a deterministic drop list for descriptor columns and apply it.

    scikit-learn-like ``fit``/``transform`` semantics with plain pandas IO.

    Parameters
    ----------
    drop_zero_variance
        Drop numeric columns whose population variance (``ddof=0``, computed
        over non-NaN entries) is at most ``variance_min``.  NaN variance counts
        as zero variance.
    nan_fraction_max
        Drop columns whose fraction of missing values is strictly greater than
        this value.  ``None`` disables the rule.
    correlation_max
        Drop one column of every pair whose absolute Pearson correlation is
        strictly greater than this value, keeping the higher-variance column.
        ``None`` disables the rule.
    variance_min
        Variance threshold used when ``drop_zero_variance`` is enabled.  The
        default ``0.0`` removes exactly the constant columns.
    keep_columns
        Columns that are never dropped, whatever the rules say.
    """

    def __init__(
        self,
        drop_zero_variance: bool = True,
        nan_fraction_max: Optional[float] = 0.2,
        correlation_max: Optional[float] = None,
        variance_min: float = 0.0,
        keep_columns: Sequence[str] = (),
    ) -> None:
        self.drop_zero_variance = bool(drop_zero_variance)
        self.nan_fraction_max = _check_unit_interval("nan_fraction_max", nan_fraction_max)
        self.correlation_max = _check_unit_interval("correlation_max", correlation_max)
        self.variance_min = float(variance_min)
        if self.variance_min < 0.0:
            raise FeatureFilterError(f"variance_min must be >= 0, got {variance_min!r}.")
        self.keep_columns = tuple(str(column) for column in keep_columns)

        self._kept_columns: Optional[List[str]] = None
        self._dropped: Dict[str, List[str]] = {reason: [] for reason in DROP_REASONS}
        self._warnings: List[str] = []
        self._n_input = 0

    # -- fitting ---------------------------------------------------------

    def fit(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> "FeatureFilter":
        """Learn which of ``feature_columns`` to drop from the rows in ``df``.

        Fit on training rows only; the surviving column list is part of the
        model feature schema.
        """
        columns = [str(column) for column in feature_columns]
        if not columns:
            raise FeatureFilterError("feature_columns must not be empty.")
        duplicates = sorted({column for column in columns if columns.count(column) > 1})
        if duplicates:
            raise FeatureFilterError(f"feature_columns contains duplicate names: {duplicates}.")
        missing = [column for column in columns if column not in set(df.columns)]
        if missing:
            raise FeatureFilterError(
                f"{len(missing)} feature column(s) absent from the fit frame: {missing[:10]}."
            )

        frame = df.loc[:, columns]
        n_rows = int(len(frame.index))
        protected = set(self.keep_columns)
        numeric = {column for column in columns if is_numeric_dtype(frame[column])}
        warnings: List[str] = []
        dropped: Dict[str, Set[str]] = {reason: set() for reason in DROP_REASONS}

        surviving = list(columns)

        if self.nan_fraction_max is not None:
            if n_rows == 0:
                warnings.append("empty frame: NaN-fraction filtering skipped.")
            else:
                fractions = frame.isna().sum() / n_rows
                dropped["high_nan"] = {
                    column
                    for column in surviving
                    if column not in protected and float(fractions[column]) > self.nan_fraction_max
                }
                surviving = [column for column in surviving if column not in dropped["high_nan"]]

        if self.drop_zero_variance:
            if n_rows < 2:
                warnings.append(
                    f"{n_rows} row(s): variance is undefined, variance filtering skipped."
                )
            else:
                candidates = [column for column in surviving if column in numeric]
                variances = frame.loc[:, candidates].var(ddof=0)
                dropped["zero_variance"] = {
                    column
                    for column in candidates
                    if column not in protected
                    and (
                        bool(pd.isna(variances[column]))
                        or float(variances[column]) <= self.variance_min
                    )
                }
                surviving = [
                    column for column in surviving if column not in dropped["zero_variance"]
                ]

        if self.correlation_max is not None:
            candidates = [column for column in surviving if column in numeric]
            if n_rows < 2:
                warnings.append(
                    f"{n_rows} row(s): correlation is undefined, correlation filtering skipped."
                )
            elif len(candidates) > MAX_CORRELATION_COLUMNS:
                warnings.append(
                    f"correlation filtering skipped: {len(candidates)} surviving numeric columns "
                    f"exceed MAX_CORRELATION_COLUMNS={MAX_CORRELATION_COLUMNS}."
                )
            elif len(candidates) > 1:
                dropped["correlated"] = self._correlated_drops(
                    frame.loc[:, candidates], protected=protected
                )
                surviving = [column for column in surviving if column not in dropped["correlated"]]

        self._n_input = len(columns)
        self._kept_columns = surviving
        # Emit every drop list in the caller's feature_columns order so repeated
        # fits on the same data report identical, reproducible lists.
        self._dropped = {
            reason: [column for column in columns if column in dropped[reason]]
            for reason in DROP_REASONS
        }
        self._warnings = warnings
        return self

    def _correlated_drops(self, frame: pd.DataFrame, *, protected: Set[str]) -> Set[str]:
        """Return the columns to drop for collinearity, keeping the higher-variance one.

        Greedy pass in priority order: protected columns first, then descending
        variance, then ascending column name.  Pairs whose correlation is NaN
        (no overlapping observations, or a column constant on the overlap) are
        treated as uncorrelated.
        """
        threshold = float(self.correlation_max or 0.0)
        correlations = frame.corr(method="pearson").abs()
        variances = frame.var(ddof=0)

        def priority(column: str) -> tuple[bool, float, str]:
            variance = variances[column]
            magnitude = 0.0 if bool(pd.isna(variance)) else float(variance)
            return (column not in protected, -magnitude, column)

        order = sorted(frame.columns.astype(str).tolist(), key=priority)
        drops: Set[str] = set()
        for index, keeper in enumerate(order):
            if keeper in drops:
                continue
            for other in order[index + 1 :]:
                if other in drops or other in protected:
                    continue
                value = correlations.at[keeper, other]
                if not bool(pd.isna(value)) and float(value) > threshold:
                    drops.add(other)
        return drops

    # -- applying --------------------------------------------------------

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Project ``df`` onto the learned columns, in the learned order.

        Extra columns are ignored.  Raises :class:`FeatureFilterError` if a
        learned column is missing, so a schema drift can never reach a model or
        the applicability domain unnoticed.
        """
        if self._kept_columns is None:
            raise FeatureFilterError("FeatureFilter.transform() called before fit().")
        counts = df.columns.value_counts()
        missing = [column for column in self._kept_columns if int(counts.get(column, 0)) == 0]
        if missing:
            raise FeatureFilterError(
                f"Feature schema mismatch: {len(missing)} learned column(s) missing at transform "
                f"time: {missing[:10]}."
            )
        duplicated = [column for column in self._kept_columns if int(counts.get(column, 0)) > 1]
        if duplicated:
            raise FeatureFilterError(
                f"Feature schema mismatch: {len(duplicated)} learned column(s) appear more than "
                f"once in the input frame: {duplicated[:10]}."
            )
        return df.loc[:, self._kept_columns].copy()

    def fit_transform(self, df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
        """Fit on ``df`` and return the filtered frame."""
        return self.fit(df, feature_columns).transform(df)

    # -- reporting -------------------------------------------------------

    @property
    def kept_columns(self) -> List[str]:
        """Learned column list, in schema order."""
        if self._kept_columns is None:
            raise FeatureFilterError("FeatureFilter is not fitted.")
        return list(self._kept_columns)

    def report(self) -> Dict[str, Any]:
        """Return a JSON-safe summary of what the fit learned."""
        if self._kept_columns is None:
            raise FeatureFilterError("FeatureFilter.report() called before fit().")
        dropped = {reason: list(self._dropped[reason]) for reason in DROP_REASONS}
        return {
            "n_input": self._n_input,
            "n_kept": len(self._kept_columns),
            "n_dropped": sum(len(items) for items in dropped.values()),
            "dropped": dropped,
            "kept_columns": list(self._kept_columns),
            "warnings": list(self._warnings),
            "params": {
                "drop_zero_variance": self.drop_zero_variance,
                "nan_fraction_max": self.nan_fraction_max,
                "correlation_max": self.correlation_max,
                "variance_min": self.variance_min,
                "keep_columns": list(self.keep_columns),
            },
        }
