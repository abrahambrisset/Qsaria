#!/usr/bin/env python
# coding: utf-8
"""Common applicability-domain helpers for prediction models.

Modern AD methods are fit in the model feature space on training rows only, then
reused for validation, test, inference, and external evaluation. V2 combines a
strict bounding box with a scikit-learn Isolation Forest when both are available.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import IsolationForest

from cs_copilot.tools.chemistry.standardize import resolve_smiles_column_name
from cs_copilot.tools.features.molecular_feature_toolkit import MolecularFeatureToolkit

from .backend import PredictionModelRecord
from .qsar_training_policy import safe_slug
from .tabular_representations import get_tabular_representation

MODERN_AD_VERSION = "3.0"
BOUNDING_BOX_METHOD = "bounding_box"
ISOLATION_FOREST_METHOD = "isolation_forest"
SIMILARITY_MATRIX_METHOD = "similarity_matrix"
COMBINED_METHOD = "combined"
DEFAULT_MODERN_AD_METHODS = (
    BOUNDING_BOX_METHOD,
    ISOLATION_FOREST_METHOD,
    SIMILARITY_MATRIX_METHOD,
)
DEFAULT_SIMILARITY_TOP_K_NEIGHBORS = 5
DEFAULT_SIMILARITY_THRESHOLD_PERCENTILE = 5.0
SUPPORTED_SIMILARITY_TOP_K_NEIGHBORS = {1, 3, 5}
SIMILARITY_SUBSPACES = (
    "morgan_binary",
    "morgan_count",
    "rdkit_descriptors",
    "chemprop_embedding",
    "continuous",
)
QSAR_ROW_ID_COLUMN = "__qsar_row_id"
AD_COLUMNS = [
    "ad_status",
    "ad_method",
    "ad_methods",
    "ad_methods_out",
    "ad_violation_count",
    "ad_violating_features",
    "ad_max_excess",
    "ad_feature_space",
    "ad_bounding_box_status",
    "ad_bounding_box_violation_count",
    "ad_bounding_box_violating_features",
    "ad_bounding_box_max_excess",
    "ad_isolation_forest_status",
    "ad_isolation_forest_score",
    "ad_isolation_forest_decision",
    "ad_isolation_forest_threshold",
    "ad_similarity_status",
    "ad_similarity_subspaces",
    "ad_similarity_subspaces_out",
    "ad_similarity_support_score",
    "ad_similarity_threshold",
    "ad_similarity_nearest_train_index",
    "ad_similarity_top_k_neighbors",
]
for _subspace_name in SIMILARITY_SUBSPACES:
    AD_COLUMNS.extend(
        [
            f"ad_similarity_{_subspace_name}_status",
            f"ad_similarity_{_subspace_name}_support_score",
            f"ad_similarity_{_subspace_name}_threshold",
            f"ad_similarity_{_subspace_name}_nearest_train_index",
            f"ad_similarity_{_subspace_name}_mean_top_k_similarity",
            f"ad_similarity_{_subspace_name}_mean_top_k_distance",
        ]
    )
AD_IN_DOMAIN = "in_domain"
AD_OUT_OF_DOMAIN = "out_of_domain"
AD_INVALID_FEATURES = "invalid_features"
AD_SCHEMA_MISMATCH = "ad_unavailable_schema_mismatch"
AD_UNAVAILABLE = "ad_unavailable"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict)) else False:
        return None
    return value


def feature_kind_for_column(column: str) -> str:
    """Return the AD feature kind for a tabular feature column."""
    name = str(column)
    if name.startswith("fp_"):
        return "morgan_binary"
    if name.startswith("cfp_"):
        return "morgan_count"
    if name.startswith("desc_"):
        return "rdkit_descriptor"
    if name.startswith("chemprop_") or name.startswith("embedding_"):
        return "chemprop_embedding"
    return "continuous"


def feature_kinds_for_columns(columns: Sequence[str]) -> List[str]:
    return [feature_kind_for_column(column) for column in columns]


def schema_hash(feature_names: Sequence[str], feature_kinds: Sequence[str]) -> str:
    payload = {
        "feature_names": [str(item) for item in feature_names],
        "feature_kinds": [str(item) for item in feature_kinds],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_numeric_frame(frame: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    numeric_columns = [
        pd.to_numeric(frame[str(column)], errors="coerce").rename(str(column))
        for column in feature_columns
    ]
    return pd.concat(numeric_columns, axis=1) if numeric_columns else pd.DataFrame(index=frame.index)


def normalize_ad_methods(methods: Optional[Sequence[str] | str]) -> List[str]:
    """Return supported modern AD methods, preserving caller order."""
    if methods is None:
        raw_methods = list(DEFAULT_MODERN_AD_METHODS)
    elif isinstance(methods, str):
        try:
            parsed = json.loads(methods)
            raw_methods = parsed if isinstance(parsed, list) else [methods]
        except json.JSONDecodeError:
            raw_methods = [part.strip() for part in methods.split(",")]
    else:
        raw_methods = list(methods)

    supported = {BOUNDING_BOX_METHOD, ISOLATION_FOREST_METHOD, SIMILARITY_MATRIX_METHOD}
    normalized: List[str] = []
    for method in raw_methods:
        value = str(method).strip().lower()
        if not value:
            continue
        if value not in supported:
            raise ValueError(
                "Unsupported applicability_domain method "
                f"{method!r}; expected one of {sorted(supported)}."
            )
        if value not in normalized:
            normalized.append(value)
    return normalized or list(DEFAULT_MODERN_AD_METHODS)


def _status_summary(scores: pd.DataFrame, status_column: str) -> Dict[str, Any]:
    if status_column not in scores.columns:
        return {}
    counts = scores[status_column].value_counts(dropna=False).to_dict()
    row_count = int(len(scores))
    in_count = int(counts.get(AD_IN_DOMAIN, 0))
    out_count = int(counts.get(AD_OUT_OF_DOMAIN, 0))
    invalid_count = int(counts.get(AD_INVALID_FEATURES, 0))
    return {
        "row_count": row_count,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "coverage_in_domain": float(in_count / row_count) if row_count else None,
        "n_in_domain": in_count,
        "n_out_of_domain": out_count,
        "n_invalid_features": invalid_count,
    }


def _summarize_scores(scores: pd.DataFrame) -> Dict[str, Any]:
    counts = scores["ad_status"].value_counts(dropna=False).to_dict() if not scores.empty else {}
    row_count = int(len(scores))
    in_count = int(counts.get(AD_IN_DOMAIN, 0))
    out_count = int(counts.get(AD_OUT_OF_DOMAIN, 0))
    invalid_count = int(counts.get(AD_INVALID_FEATURES, 0))
    coverage = float(in_count / row_count) if row_count else None
    feature_counter: Counter[str] = Counter()
    for raw in scores.get("ad_violating_features", pd.Series(dtype=str)).fillna(""):
        for feature in str(raw).split(";"):
            if feature:
                feature_counter[feature] += 1
    return {
        "row_count": row_count,
        "status_counts": {str(key): int(value) for key, value in counts.items()},
        "coverage_in_domain": coverage,
        "n_in_domain": in_count,
        "n_out_of_domain": out_count,
        "n_invalid_features": invalid_count,
        "top_violating_features": [
            {"feature": feature, "count": int(count)}
            for feature, count in feature_counter.most_common(20)
        ],
        "method_status_summaries": {
            BOUNDING_BOX_METHOD: _status_summary(scores, "ad_bounding_box_status"),
            ISOLATION_FOREST_METHOD: _status_summary(
                scores, "ad_isolation_forest_status"
            ),
            SIMILARITY_MATRIX_METHOD: _status_summary(scores, "ad_similarity_status"),
        },
    }


def _write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2) + "\n")


def _resolve_path(raw_path: str | Path, *, metadata_path: Optional[str] = None) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if metadata_path:
        return Path(metadata_path).expanduser().parent / path
    return path


def _merge_manifest_overrides(
    manifest: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> Dict[str, Any]:
    merged = dict(manifest or {})
    source = dict(overrides or {})
    if not source:
        return merged
    for key in (
        "manifest_path",
        "bounds_path",
        "isolation_forest_model_path",
        "scores_train_path",
        "scores_validation_path",
        "scores_test_path",
    ):
        if source.get(key):
            merged[key] = source[key]
    methods = dict(merged.get("methods") or {})
    source_methods = source.get("methods") or {}
    for method_name, payload in source_methods.items():
        method_payload = dict(methods.get(method_name) or {})
        method_payload.update({key: value for key, value in dict(payload or {}).items() if value})
        methods[method_name] = method_payload
    merged["methods"] = methods
    return merged


def fit_bounding_box_domain(
    *,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    model_id: str,
    feature_space: str,
    representation_name: Optional[str] = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit and persist a strict bounding-box AD on training features."""
    columns = [str(column) for column in feature_columns if str(column) in feature_frame.columns]
    if not columns:
        return {
            "available": False,
            "method": BOUNDING_BOX_METHOD,
            "reason": "No feature columns were available to fit a bounding-box applicability domain.",
        }

    numeric = _coerce_numeric_frame(feature_frame, columns)
    valid_mask = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    valid_numeric = numeric.loc[valid_mask]
    if valid_numeric.empty:
        return {
            "available": False,
            "method": BOUNDING_BOX_METHOD,
            "reason": "All training feature rows contained NaN or infinite values.",
        }

    kinds = feature_kinds_for_columns(columns)
    resolved_schema_hash = schema_hash(columns, kinds)
    ad_dir = Path(output_dir).expanduser()
    bb_dir = ad_dir / BOUNDING_BOX_METHOD
    plots_dir = ad_dir / "plots"
    bb_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    bounds_path = bb_dir / "bounds.npz"
    manifest_path = ad_dir / "manifest.json"

    np.savez_compressed(
        bounds_path,
        min_values=valid_numeric.min(axis=0).to_numpy(dtype=float),
        max_values=valid_numeric.max(axis=0).to_numpy(dtype=float),
        feature_names=np.asarray(columns, dtype=str),
        feature_kinds=np.asarray(kinds, dtype=str),
        schema_hash=np.asarray([resolved_schema_hash], dtype=str),
    )

    manifest = {
        "available": True,
        "version": MODERN_AD_VERSION,
        "primary_method": BOUNDING_BOX_METHOD,
        "method": BOUNDING_BOX_METHOD,
        "model_id": model_id,
        "feature_space": feature_space,
        "representation_name": representation_name or feature_space,
        "feature_count": len(columns),
        "feature_names": columns,
        "feature_kinds": kinds,
        "fit_row_count": int(len(feature_frame)),
        "fit_valid_row_count": int(len(valid_numeric)),
        "fit_invalid_row_count": int((~valid_mask).sum()),
        "schema_hash": resolved_schema_hash,
        "manifest_path": str(manifest_path),
        "bounds_path": str(bounds_path),
        "plots_dir": str(plots_dir),
        "methods": {
            BOUNDING_BOX_METHOD: {
                "version": MODERN_AD_VERSION,
                "rule": "strict_min_max_any_feature_outside_is_out_of_domain",
                "feature_space": feature_space,
                "representation_name": representation_name or feature_space,
                "feature_count": len(columns),
                "schema_hash": resolved_schema_hash,
                "bounds_path": str(bounds_path),
                "manifest_path": str(manifest_path),
            }
        },
    }
    if feature_metadata:
        manifest.update(dict(feature_metadata))
    _write_manifest(manifest_path, manifest)
    return manifest


def load_bounding_box_manifest(
    applicability_domain: Mapping[str, Any],
    *,
    metadata_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Return the modern bounding-box manifest and its path, if available."""
    ad = dict(applicability_domain or {})
    if not ad:
        return None, None
    manifest_path = ad.get("manifest_path")
    if not manifest_path:
        methods = ad.get("methods") or {}
        method = methods.get(BOUNDING_BOX_METHOD) or {}
        manifest_path = method.get("manifest_path")
    if manifest_path:
        path = _resolve_path(str(manifest_path), metadata_path=metadata_path)
        if path.exists():
            return _merge_manifest_overrides(json.loads(path.read_text()), ad), path
    if BOUNDING_BOX_METHOD in (ad.get("methods") or {}):
        return ad, None
    if ad.get("primary_method") == BOUNDING_BOX_METHOD or ad.get("method") == BOUNDING_BOX_METHOD:
        return ad, None
    return None, None


def _bounds_path_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    metadata_path: Optional[str] = None,
) -> Optional[Path]:
    methods = manifest.get("methods") or {}
    method = methods.get(BOUNDING_BOX_METHOD) or {}
    raw_path = method.get("bounds_path") or manifest.get("bounds_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if metadata_path:
        candidate = Path(metadata_path).expanduser().parent / path
        if candidate.exists():
            return candidate
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    return _resolve_path(str(raw_path), metadata_path=metadata_path)


def score_bounding_box_domain(
    *,
    feature_frame: pd.DataFrame,
    applicability_domain: Mapping[str, Any],
    output_dir: Optional[str | Path] = None,
    score_label: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score rows against a persisted bounding-box AD."""
    manifest, manifest_path = load_bounding_box_manifest(
        applicability_domain,
        metadata_path=metadata_path,
    )
    if not manifest:
        scores = pd.DataFrame(
            {
                "ad_status": [AD_UNAVAILABLE] * len(feature_frame),
                "ad_method": [BOUNDING_BOX_METHOD] * len(feature_frame),
                "ad_methods": [BOUNDING_BOX_METHOD] * len(feature_frame),
                "ad_methods_out": [""] * len(feature_frame),
                "ad_violation_count": [0] * len(feature_frame),
                "ad_violating_features": [""] * len(feature_frame),
                "ad_max_excess": [np.nan] * len(feature_frame),
                "ad_feature_space": [""] * len(feature_frame),
                "ad_bounding_box_status": [AD_UNAVAILABLE] * len(feature_frame),
                "ad_bounding_box_violation_count": [0] * len(feature_frame),
                "ad_bounding_box_violating_features": [""] * len(feature_frame),
                "ad_bounding_box_max_excess": [np.nan] * len(feature_frame),
            }
        )
        return {
            "available": False,
            "reason": "No modern bounding-box applicability domain is available for this model.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    bounds_path = _bounds_path_from_manifest(
        manifest,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    if bounds_path is None or not bounds_path.exists():
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason="Bounding-box bounds.npz is missing.",
        )
        return {
            "available": False,
            "reason": "Bounding-box bounds.npz is missing.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    payload = np.load(bounds_path, allow_pickle=False)
    feature_names = [str(item) for item in payload["feature_names"].tolist()]
    feature_kinds = [str(item) for item in payload["feature_kinds"].tolist()]
    expected_hash = str(payload["schema_hash"].tolist()[0])
    if schema_hash(feature_names, feature_kinds) != expected_hash:
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason="Bounding-box schema hash is inconsistent with stored feature names.",
        )
        return {
            "available": False,
            "reason": "Bounding-box schema hash is inconsistent with stored feature names.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }
    missing = [feature for feature in feature_names if feature not in feature_frame.columns]
    if missing:
        reason = (
            "Prediction feature schema does not match the stored applicability-domain schema. "
            f"Missing {len(missing)} feature(s), for example: {missing[:10]}."
        )
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason=reason,
        )
        return {
            "available": False,
            "reason": reason,
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    values = _coerce_numeric_frame(feature_frame, feature_names).to_numpy(dtype=float)
    min_values = payload["min_values"].astype(float)
    max_values = payload["max_values"].astype(float)
    invalid = ~np.isfinite(values).all(axis=1)
    below = values < min_values
    above = values > max_values
    violations = below | above
    violation_counts = violations.sum(axis=1).astype(int)
    statuses = np.where(
        invalid,
        AD_INVALID_FEATURES,
        np.where(violation_counts > 0, AD_OUT_OF_DOMAIN, AD_IN_DOMAIN),
    )
    ranges = np.maximum(max_values - min_values, 1.0)
    lower_excess = np.maximum(min_values - values, 0.0) / ranges
    upper_excess = np.maximum(values - max_values, 0.0) / ranges
    excess = np.maximum(lower_excess, upper_excess)
    max_excess = np.where(invalid, np.nan, np.nanmax(excess, axis=1))

    violating_features: List[str] = []
    for row_index in range(len(feature_frame)):
        if invalid[row_index]:
            invalid_columns = [
                feature_names[index]
                for index, value in enumerate(values[row_index])
                if not np.isfinite(value)
            ]
            violating_features.append(";".join(invalid_columns[:50]))
            continue
        if violation_counts[row_index] == 0:
            violating_features.append("")
            continue
        features = [
            feature_names[index]
            for index, is_violating in enumerate(violations[row_index])
            if bool(is_violating)
        ]
        violating_features.append(";".join(features[:50]))

    scores = pd.DataFrame(
        {
            "ad_status": statuses,
            "ad_method": BOUNDING_BOX_METHOD,
            "ad_methods": BOUNDING_BOX_METHOD,
            "ad_methods_out": [
                BOUNDING_BOX_METHOD if status == AD_OUT_OF_DOMAIN else ""
                for status in statuses
            ],
            "ad_violation_count": violation_counts,
            "ad_violating_features": violating_features,
            "ad_max_excess": max_excess,
            "ad_feature_space": str(manifest.get("feature_space") or ""),
            "ad_bounding_box_status": statuses,
            "ad_bounding_box_violation_count": violation_counts,
            "ad_bounding_box_violating_features": violating_features,
            "ad_bounding_box_max_excess": max_excess,
        }
    )
    summary = _summarize_scores(scores)
    summary.update(
        {
            "available": True,
            "method": BOUNDING_BOX_METHOD,
            "feature_space": manifest.get("feature_space"),
            "schema_hash": expected_hash,
        }
    )

    score_path = None
    if output_dir:
        label = safe_slug(score_label or "scores") or "scores"
        target = Path(output_dir).expanduser() / f"scores_{label}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target, index=False)
        score_path = str(target)
        summary["scores_path"] = score_path

    return {
        "available": True,
        "method": BOUNDING_BOX_METHOD,
        "scores": scores,
        "summary": summary,
        "scores_path": score_path,
    }


def _schema_mismatch_scores(
    row_count: int,
    *,
    feature_space: str,
    reason: str,
    method: str = BOUNDING_BOX_METHOD,
) -> pd.DataFrame:
    payload: Dict[str, Any] = {
        "ad_status": [AD_SCHEMA_MISMATCH] * row_count,
        "ad_method": [method] * row_count,
        "ad_methods": [method] * row_count,
        "ad_methods_out": [""] * row_count,
        "ad_violation_count": [0] * row_count,
        "ad_violating_features": [reason] * row_count,
        "ad_max_excess": [np.nan] * row_count,
        "ad_feature_space": [feature_space] * row_count,
    }
    if method == BOUNDING_BOX_METHOD:
        payload.update(
            {
                "ad_bounding_box_status": [AD_SCHEMA_MISMATCH] * row_count,
                "ad_bounding_box_violation_count": [0] * row_count,
                "ad_bounding_box_violating_features": [reason] * row_count,
                "ad_bounding_box_max_excess": [np.nan] * row_count,
            }
        )
    if method == ISOLATION_FOREST_METHOD:
        payload.update(
            {
                "ad_isolation_forest_status": [AD_SCHEMA_MISMATCH] * row_count,
                "ad_isolation_forest_score": [np.nan] * row_count,
                "ad_isolation_forest_decision": [np.nan] * row_count,
                "ad_isolation_forest_threshold": [0.0] * row_count,
            }
        )
    if method == SIMILARITY_MATRIX_METHOD:
        payload.update(
            {
                "ad_similarity_status": [AD_SCHEMA_MISMATCH] * row_count,
                "ad_similarity_subspaces": [""] * row_count,
                "ad_similarity_subspaces_out": [""] * row_count,
                "ad_similarity_support_score": [np.nan] * row_count,
                "ad_similarity_threshold": [np.nan] * row_count,
                "ad_similarity_nearest_train_index": [""] * row_count,
                "ad_similarity_top_k_neighbors": [""] * row_count,
            }
        )
    return pd.DataFrame(payload)


def _isolation_forest_model_path_from_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    metadata_path: Optional[str] = None,
) -> Optional[Path]:
    methods = manifest.get("methods") or {}
    method = methods.get(ISOLATION_FOREST_METHOD) or {}
    raw_path = method.get("model_path") or manifest.get("isolation_forest_model_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if metadata_path:
        candidate = Path(metadata_path).expanduser().parent / path
        if candidate.exists():
            return candidate
    if manifest_path is not None:
        candidate = manifest_path.parent / path
        if candidate.exists():
            return candidate
    return _resolve_path(str(raw_path), metadata_path=metadata_path)


def fit_isolation_forest_domain(
    *,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    model_id: str,
    feature_space: str,
    representation_name: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Fit and persist an Isolation Forest AD on training features."""
    columns = [str(column) for column in feature_columns if str(column) in feature_frame.columns]
    if not columns:
        return {
            "available": False,
            "method": ISOLATION_FOREST_METHOD,
            "reason": "No feature columns were available to fit an Isolation Forest applicability domain.",
        }

    numeric = _coerce_numeric_frame(feature_frame, columns)
    values = numeric.to_numpy(dtype=float)
    valid_mask = np.isfinite(values).all(axis=1)
    valid_numeric = numeric.loc[valid_mask]
    if valid_numeric.empty:
        return {
            "available": False,
            "method": ISOLATION_FOREST_METHOD,
            "reason": "All training feature rows contained NaN or infinite values.",
        }

    kinds = feature_kinds_for_columns(columns)
    resolved_schema_hash = schema_hash(columns, kinds)
    ad_dir = Path(output_dir).expanduser()
    if_dir = ad_dir / ISOLATION_FOREST_METHOD
    if_dir.mkdir(parents=True, exist_ok=True)
    model_path = if_dir / "model.joblib"
    estimator = IsolationForest(random_state=random_state)
    estimator.fit(valid_numeric.to_numpy(dtype=float))
    joblib.dump(estimator, model_path)

    params = estimator.get_params(deep=False)
    return {
        "available": True,
        "version": MODERN_AD_VERSION,
        "method": ISOLATION_FOREST_METHOD,
        "model_id": model_id,
        "feature_space": feature_space,
        "representation_name": representation_name or feature_space,
        "feature_count": len(columns),
        "feature_names": columns,
        "feature_kinds": kinds,
        "schema_hash": resolved_schema_hash,
        "fit_row_count": int(len(feature_frame)),
        "fit_valid_row_count": int(len(valid_numeric)),
        "fit_invalid_row_count": int((~valid_mask).sum()),
        "model_path": str(model_path),
        "params": _json_safe(params),
        "isolation_forest_params": _json_safe(params),
        "isolation_forest_offset": float(getattr(estimator, "offset_", np.nan)),
        "threshold": 0.0,
        "decision_rule": "decision_function_less_than_zero_is_out_of_domain",
    }


def score_isolation_forest_domain(
    *,
    feature_frame: pd.DataFrame,
    applicability_domain: Mapping[str, Any],
    output_dir: Optional[str | Path] = None,
    score_label: Optional[str] = None,
    metadata_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score rows against a persisted Isolation Forest AD."""
    manifest, manifest_path = load_modern_ad_manifest(
        applicability_domain,
        metadata_path=metadata_path,
    )
    if not manifest or ISOLATION_FOREST_METHOD not in (manifest.get("methods") or {}):
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str((manifest or {}).get("feature_space") or ""),
            reason="No Isolation Forest applicability domain is available for this model.",
            method=ISOLATION_FOREST_METHOD,
        )
        return {
            "available": False,
            "reason": "No Isolation Forest applicability domain is available for this model.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    method = (manifest.get("methods") or {}).get(ISOLATION_FOREST_METHOD) or {}
    feature_names = [str(item) for item in method.get("feature_names") or manifest.get("feature_names") or []]
    feature_kinds = [str(item) for item in method.get("feature_kinds") or manifest.get("feature_kinds") or []]
    expected_hash = str(method.get("schema_hash") or manifest.get("schema_hash") or "")
    if not feature_names:
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason="Isolation Forest feature schema is missing from the manifest.",
            method=ISOLATION_FOREST_METHOD,
        )
        return {
            "available": False,
            "reason": "Isolation Forest feature schema is missing from the manifest.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }
    if feature_kinds and schema_hash(feature_names, feature_kinds) != expected_hash:
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason="Isolation Forest schema hash is inconsistent with stored feature names.",
            method=ISOLATION_FOREST_METHOD,
        )
        return {
            "available": False,
            "reason": "Isolation Forest schema hash is inconsistent with stored feature names.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    missing = [feature for feature in feature_names if feature not in feature_frame.columns]
    if missing:
        reason = (
            "Prediction feature schema does not match the stored Isolation Forest schema. "
            f"Missing {len(missing)} feature(s), for example: {missing[:10]}."
        )
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason=reason,
            method=ISOLATION_FOREST_METHOD,
        )
        return {"available": False, "reason": reason, "scores": scores, "summary": _summarize_scores(scores)}

    model_path = _isolation_forest_model_path_from_manifest(
        manifest,
        manifest_path=manifest_path,
        metadata_path=metadata_path,
    )
    if model_path is None or not model_path.exists():
        reason = "Isolation Forest model.joblib is missing."
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str(manifest.get("feature_space") or ""),
            reason=reason,
            method=ISOLATION_FOREST_METHOD,
        )
        return {"available": False, "reason": reason, "scores": scores, "summary": _summarize_scores(scores)}

    estimator = joblib.load(model_path)
    values = _coerce_numeric_frame(feature_frame, feature_names).to_numpy(dtype=float)
    invalid = ~np.isfinite(values).all(axis=1)
    score_samples = np.full(len(feature_frame), np.nan, dtype=float)
    decisions = np.full(len(feature_frame), np.nan, dtype=float)
    statuses = np.asarray([AD_INVALID_FEATURES if item else AD_IN_DOMAIN for item in invalid], dtype=object)
    if (~invalid).any():
        valid_values = values[~invalid]
        score_samples[~invalid] = estimator.score_samples(valid_values)
        decisions[~invalid] = estimator.decision_function(valid_values)
        statuses[~invalid] = np.where(decisions[~invalid] < 0.0, AD_OUT_OF_DOMAIN, AD_IN_DOMAIN)

    invalid_features: List[str] = []
    for row_index in range(len(feature_frame)):
        if not invalid[row_index]:
            invalid_features.append("")
            continue
        columns = [
            feature_names[index]
            for index, value in enumerate(values[row_index])
            if not np.isfinite(value)
        ]
        invalid_features.append(";".join(columns[:50]))

    scores = pd.DataFrame(
        {
            "ad_status": statuses,
            "ad_method": ISOLATION_FOREST_METHOD,
            "ad_methods": ISOLATION_FOREST_METHOD,
            "ad_methods_out": [
                ISOLATION_FOREST_METHOD if status == AD_OUT_OF_DOMAIN else ""
                for status in statuses
            ],
            "ad_violation_count": [0] * len(feature_frame),
            "ad_violating_features": invalid_features,
            "ad_max_excess": np.where(decisions < 0.0, -decisions, 0.0),
            "ad_feature_space": str(manifest.get("feature_space") or ""),
            "ad_isolation_forest_status": statuses,
            "ad_isolation_forest_score": score_samples,
            "ad_isolation_forest_decision": decisions,
            "ad_isolation_forest_threshold": [0.0] * len(feature_frame),
        }
    )
    summary = _summarize_scores(scores)
    summary.update(
        {
            "available": True,
            "method": ISOLATION_FOREST_METHOD,
            "feature_space": manifest.get("feature_space"),
            "schema_hash": expected_hash,
            "threshold": 0.0,
            "offset": float(getattr(estimator, "offset_", np.nan)),
        }
    )

    score_path = None
    if output_dir:
        label = safe_slug(score_label or "scores") or "scores"
        target = Path(output_dir).expanduser() / f"scores_{label}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target, index=False)
        score_path = str(target)
        summary["scores_path"] = score_path

    return {
        "available": True,
        "method": ISOLATION_FOREST_METHOD,
        "scores": scores,
        "summary": summary,
        "scores_path": score_path,
    }


def _normalize_similarity_top_k(value: int | str | None) -> int:
    try:
        top_k = int(value or DEFAULT_SIMILARITY_TOP_K_NEIGHBORS)
    except (TypeError, ValueError) as exc:
        raise ValueError("similarity_top_k_neighbors must be one of 1, 3, or 5.") from exc
    if top_k not in SUPPORTED_SIMILARITY_TOP_K_NEIGHBORS:
        raise ValueError("similarity_top_k_neighbors must be one of 1, 3, or 5.")
    return top_k


def _normalize_similarity_percentile(value: float | str | None) -> float:
    try:
        percentile = float(
            DEFAULT_SIMILARITY_THRESHOLD_PERCENTILE if value is None else value
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("similarity_threshold_percentile must be a number in [0, 100].") from exc
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("similarity_threshold_percentile must be in [0, 100].")
    return percentile


def _similarity_subspaces(feature_columns: Sequence[str]) -> Dict[str, List[str]]:
    columns = [str(column) for column in feature_columns]
    groups = {
        "morgan_binary": [column for column in columns if column.startswith("fp_")],
        "morgan_count": [column for column in columns if column.startswith("cfp_")],
        "rdkit_descriptors": [column for column in columns if column.startswith("desc_")],
        "chemprop_embedding": [
            column
            for column in columns
            if column.startswith("chemprop_") or column.startswith("embedding_")
        ],
    }
    groups = {name: cols for name, cols in groups.items() if cols}
    if not groups and columns:
        groups["continuous"] = columns
    return groups


def _subspace_metric(subspace: str) -> str:
    if subspace == "morgan_binary":
        return "tanimoto"
    if subspace == "morgan_count":
        return "count_tanimoto"
    return "euclidean"


def _active_subspace_values(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    subspace: str,
    fit_stats: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, List[str], np.ndarray, Dict[str, Any]]:
    numeric = _coerce_numeric_frame(frame, columns)
    values = numeric.to_numpy(dtype=float)
    active_columns = [str(column) for column in columns]
    stats: Dict[str, Any] = dict(fit_stats or {})
    if subspace in {"rdkit_descriptors", "chemprop_embedding", "continuous"}:
        if fit_stats:
            mean = np.asarray(fit_stats.get("mean"), dtype=float)
            std = np.asarray(fit_stats.get("std"), dtype=float)
            active_columns = [str(item) for item in fit_stats.get("feature_names") or active_columns]
            numeric = _coerce_numeric_frame(frame, active_columns)
            values = numeric.to_numpy(dtype=float)
        else:
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            finite_stats = np.isfinite(mean) & np.isfinite(std) & (std > 0.0)
            active_columns = [
                column for column, keep in zip(active_columns, finite_stats) if bool(keep)
            ]
            values = values[:, finite_stats] if values.size else values
            mean = mean[finite_stats]
            std = std[finite_stats]
        if not active_columns:
            return np.empty((len(frame), 0), dtype=float), [], np.zeros(len(frame), dtype=bool), stats
        values = (values - mean) / std
        stats = {
            "feature_names": active_columns,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    invalid = ~np.isfinite(values).all(axis=1) if values.size else np.ones(len(frame), dtype=bool)
    return values.astype(float), active_columns, invalid, stats


def _pairwise_similarity_or_distance(
    query: np.ndarray,
    reference: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    if query.size == 0 or reference.size == 0:
        return np.empty((len(query), len(reference)), dtype=np.float32)
    left = query.astype(float, copy=False)
    right = reference.astype(float, copy=False)
    dot = left @ right.T
    if metric in {"tanimoto", "count_tanimoto"}:
        left_sq = np.sum(left * left, axis=1)[:, None]
        right_sq = np.sum(right * right, axis=1)[None, :]
        denom = left_sq + right_sq - dot
        values = np.divide(dot, denom, out=np.zeros_like(dot, dtype=float), where=denom > 0)
        return np.clip(values, 0.0, 1.0).astype(np.float32)
    left_sq = np.sum(left * left, axis=1)[:, None]
    right_sq = np.sum(right * right, axis=1)[None, :]
    distances = np.sqrt(np.maximum(left_sq + right_sq - 2.0 * dot, 0.0))
    return distances.astype(np.float32)


def _write_pairwise_matrix(
    *,
    values: np.ndarray,
    invalid: np.ndarray,
    metric: str,
    output_path: Path,
    chunk_size: int = 512,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(values), len(values)),
    )
    safe_values = np.where(invalid[:, None], 0.0, values)
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        block = _pairwise_similarity_or_distance(
            safe_values[start:stop],
            safe_values,
            metric=metric,
        )
        if invalid.any():
            block[:, invalid] = np.nan
            block[invalid[start:stop], :] = np.nan
        matrix[start:stop, :] = block
    matrix.flush()


def _support_from_vector(values: np.ndarray, *, metric: str, top_k: int) -> Tuple[float, Any, float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, "", np.nan, np.nan
    k = min(top_k, finite.size)
    if metric in {"tanimoto", "count_tanimoto"}:
        top = np.sort(finite)[-k:]
        support = float(np.mean(top))
        return support, int(np.nanargmax(values)), support, np.nan
    top = np.sort(finite)[:k]
    mean_distance = float(np.mean(top))
    support = float(1.0 / (1.0 + mean_distance))
    return support, int(np.nanargmin(values)), np.nan, mean_distance


def _train_support_scores(
    matrix: np.ndarray,
    *,
    train_positions: Sequence[int],
    metric: str,
    top_k: int,
) -> np.ndarray:
    supports: List[float] = []
    train_positions = [int(item) for item in train_positions]
    for row_pos in train_positions:
        row = np.asarray(matrix[row_pos, train_positions], dtype=float).copy()
        self_index = train_positions.index(row_pos)
        row[self_index] = np.nan
        support, _, _, _ = _support_from_vector(row, metric=metric, top_k=top_k)
        supports.append(support)
    return np.asarray(supports, dtype=float)


def fit_similarity_matrix_domain(
    *,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    model_id: str,
    feature_space: str,
    train_indices: Optional[Sequence[int]] = None,
    split_indices: Optional[Mapping[str, Sequence[int]]] = None,
    representation_name: Optional[str] = None,
    top_k_neighbors: int | str | None = DEFAULT_SIMILARITY_TOP_K_NEIGHBORS,
    threshold_percentile: float | str | None = DEFAULT_SIMILARITY_THRESHOLD_PERCENTILE,
) -> Dict[str, Any]:
    """Fit a structural/embedding similarity AD and persist full all/all matrices."""
    columns = [str(column) for column in feature_columns if str(column) in feature_frame.columns]
    if not columns:
        return {
            "available": False,
            "method": SIMILARITY_MATRIX_METHOD,
            "reason": "No feature columns were available to fit a similarity-matrix applicability domain.",
        }
    top_k = _normalize_similarity_top_k(top_k_neighbors)
    percentile = _normalize_similarity_percentile(threshold_percentile)
    row_count = int(len(feature_frame))
    resolved_train_indices = [int(item) for item in (train_indices or range(row_count))]
    resolved_train_indices = [
        item for item in resolved_train_indices if 0 <= int(item) < row_count
    ]
    if len(resolved_train_indices) < 2:
        return {
            "available": False,
            "method": SIMILARITY_MATRIX_METHOD,
            "reason": "At least two train rows are required to fit a similarity-matrix applicability domain.",
        }

    ad_dir = Path(output_dir).expanduser()
    sim_dir = ad_dir / SIMILARITY_MATRIX_METHOD
    sim_dir.mkdir(parents=True, exist_ok=True)
    subspaces: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    for subspace, subspace_columns in _similarity_subspaces(columns).items():
        metric = _subspace_metric(subspace)
        subspace_dir = sim_dir / subspace
        try:
            if _subspace_metric(subspace) == "euclidean":
                _, active_columns, _, fit_stats = _active_subspace_values(
                    feature_frame.iloc[resolved_train_indices].copy(),
                    subspace_columns,
                    subspace=subspace,
                )
                values, active_columns, invalid, _ = _active_subspace_values(
                    feature_frame,
                    active_columns,
                    subspace=subspace,
                    fit_stats=fit_stats,
                )
            else:
                values, active_columns, invalid, fit_stats = _active_subspace_values(
                    feature_frame,
                    subspace_columns,
                    subspace=subspace,
                )
            valid_train_indices = [
                index for index in resolved_train_indices if not bool(invalid[index])
            ]
            if len(valid_train_indices) < 2:
                errors[subspace] = "Fewer than two valid train rows were available."
                continue
            matrix_path = subspace_dir / "matrix_all.npy"
            _write_pairwise_matrix(
                values=values,
                invalid=invalid,
                metric=metric,
                output_path=matrix_path,
            )
            matrix = np.load(matrix_path, mmap_mode="r")
            train_supports = _train_support_scores(
                matrix,
                train_positions=valid_train_indices,
                metric=metric,
                top_k=top_k,
            )
            finite_supports = train_supports[np.isfinite(train_supports)]
            if finite_supports.size == 0:
                errors[subspace] = "No finite train support scores could be computed."
                continue
            threshold = float(np.percentile(finite_supports, percentile))
            feature_kinds = feature_kinds_for_columns(active_columns)
            resolved_hash = schema_hash(active_columns, feature_kinds)
            reference_features = values[valid_train_indices].astype(np.float32)
            reference_path = subspace_dir / "reference_features.npz"
            np.savez_compressed(
                reference_path,
                reference_features=reference_features,
                train_indices=np.asarray(valid_train_indices, dtype=np.int32),
                feature_names=np.asarray(active_columns, dtype=str),
                feature_kinds=np.asarray(feature_kinds, dtype=str),
                schema_hash=np.asarray([resolved_hash], dtype=str),
                metric=np.asarray([metric], dtype=str),
                threshold=np.asarray([threshold], dtype=float),
                top_k_neighbors=np.asarray([top_k], dtype=np.int32),
                threshold_percentile=np.asarray([percentile], dtype=float),
                mean=np.asarray(fit_stats.get("mean") or [], dtype=float),
                std=np.asarray(fit_stats.get("std") or [], dtype=float),
            )
            subspaces[subspace] = {
                "version": MODERN_AD_VERSION,
                "method": SIMILARITY_MATRIX_METHOD,
                "subspace": subspace,
                "metric": metric,
                "feature_space": feature_space,
                "representation_name": representation_name or feature_space,
                "feature_count": len(active_columns),
                "feature_names": active_columns,
                "feature_kinds": feature_kinds,
                "schema_hash": resolved_hash,
                "matrix_all_path": str(matrix_path),
                "reference_features_path": str(reference_path),
                "row_count": row_count,
                "train_size": len(resolved_train_indices),
                "valid_train_size": len(valid_train_indices),
                "invalid_row_count": int(invalid.sum()),
                "top_k_neighbors": top_k,
                "threshold_percentile": percentile,
                "threshold": threshold,
                "support_score": "mean_top_k_tanimoto" if metric != "euclidean" else "inverse_one_plus_mean_top_k_distance",
                "decision_rule": "support_score_less_than_threshold_is_out_of_domain",
                "standardization": fit_stats if fit_stats else None,
            }
        except Exception as exc:
            errors[subspace] = str(exc)

    if not subspaces:
        return {
            "available": False,
            "method": SIMILARITY_MATRIX_METHOD,
            "errors": errors,
            "reason": "No similarity subspace could be fit.",
        }

    manifest_path = sim_dir / "manifest.json"
    manifest = {
        "available": True,
        "version": MODERN_AD_VERSION,
        "method": SIMILARITY_MATRIX_METHOD,
        "model_id": model_id,
        "feature_space": feature_space,
        "representation_name": representation_name or feature_space,
        "activity_cliffs_reusable": True,
        "top_k_neighbors": top_k,
        "threshold_percentile": percentile,
        "row_count": row_count,
        "train_indices": resolved_train_indices,
        "split_indices": {
            str(key): [int(item) for item in (value or [])]
            for key, value in dict(split_indices or {}).items()
        },
        "subspaces": subspaces,
        "errors": errors,
        "manifest_path": str(manifest_path),
    }
    _write_manifest(manifest_path, manifest)
    return manifest


def _similarity_method_payload(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    methods = manifest.get("methods") or {}
    return methods.get(SIMILARITY_MATRIX_METHOD) or {}


def _reference_path_from_similarity_payload(
    subspace_payload: Mapping[str, Any],
    *,
    manifest_path: Optional[Path] = None,
    metadata_path: Optional[str] = None,
) -> Optional[Path]:
    raw_path = subspace_payload.get("reference_features_path")
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    if metadata_path:
        candidate = Path(metadata_path).expanduser().parent / path
        if candidate.exists():
            return candidate
    if manifest_path is not None:
        candidate = manifest_path.parent.parent / raw_path
        if candidate.exists():
            return candidate
    return _resolve_path(str(raw_path), metadata_path=metadata_path)


def score_similarity_matrix_domain(
    *,
    feature_frame: pd.DataFrame,
    applicability_domain: Mapping[str, Any],
    output_dir: Optional[str | Path] = None,
    score_label: Optional[str] = None,
    metadata_path: Optional[str] = None,
    row_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Score rows against persisted similarity-matrix AD reference features."""
    manifest, manifest_path = load_modern_ad_manifest(
        applicability_domain,
        metadata_path=metadata_path,
    )
    method = _similarity_method_payload(manifest or {})
    if not manifest or not method:
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space=str((manifest or {}).get("feature_space") or ""),
            reason="No similarity-matrix applicability domain is available for this model.",
            method=SIMILARITY_MATRIX_METHOD,
        )
        return {
            "available": False,
            "reason": "No similarity-matrix applicability domain is available for this model.",
            "scores": scores,
            "summary": _summarize_scores(scores),
        }

    subspaces = method.get("subspaces") or {}
    row_ids = [int(item) for item in row_indices] if row_indices is not None else [None] * len(feature_frame)
    per_subspace_scores: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Any] = {}
    for subspace, subspace_payload in subspaces.items():
        reference_path = _reference_path_from_similarity_payload(
            subspace_payload,
            manifest_path=manifest_path,
            metadata_path=metadata_path,
        )
        if reference_path is None or not reference_path.exists():
            per_subspace_scores[str(subspace)] = _schema_mismatch_scores(
                len(feature_frame),
                feature_space=str(manifest.get("feature_space") or ""),
                reason=f"Similarity reference_features.npz is missing for {subspace}.",
                method=SIMILARITY_MATRIX_METHOD,
            )
            continue
        payload = np.load(reference_path, allow_pickle=False)
        feature_names = [str(item) for item in payload["feature_names"].tolist()]
        feature_kinds = [str(item) for item in payload["feature_kinds"].tolist()]
        expected_hash = str(payload["schema_hash"].tolist()[0])
        if schema_hash(feature_names, feature_kinds) != expected_hash:
            per_subspace_scores[str(subspace)] = _schema_mismatch_scores(
                len(feature_frame),
                feature_space=str(manifest.get("feature_space") or ""),
                reason=f"Similarity schema hash is inconsistent for {subspace}.",
                method=SIMILARITY_MATRIX_METHOD,
            )
            continue
        missing = [feature for feature in feature_names if feature not in feature_frame.columns]
        if missing:
            reason = (
                "Prediction feature schema does not match the stored similarity schema. "
                f"Missing {len(missing)} feature(s), for example: {missing[:10]}."
            )
            per_subspace_scores[str(subspace)] = _schema_mismatch_scores(
                len(feature_frame),
                feature_space=str(manifest.get("feature_space") or ""),
                reason=reason,
                method=SIMILARITY_MATRIX_METHOD,
            )
            continue
        metric = str(payload["metric"].tolist()[0])
        threshold = float(payload["threshold"].tolist()[0])
        top_k = int(payload["top_k_neighbors"].tolist()[0])
        fit_stats = None
        if str(subspace) in {"rdkit_descriptors", "chemprop_embedding", "continuous"}:
            fit_stats = {
                "feature_names": feature_names,
                "mean": payload["mean"].tolist(),
                "std": payload["std"].tolist(),
            }
        values, _, invalid, _ = _active_subspace_values(
            feature_frame,
            feature_names,
            subspace=str(subspace),
            fit_stats=fit_stats,
        )
        reference_features = payload["reference_features"].astype(float)
        reference_indices = payload["train_indices"].astype(int)
        safe_values = np.where(invalid[:, None], 0.0, values)
        matrix = _pairwise_similarity_or_distance(
            safe_values,
            reference_features,
            metric=metric,
        ).astype(float)
        statuses: List[str] = []
        supports: List[float] = []
        nearest_indices: List[Any] = []
        mean_sims: List[float] = []
        mean_distances: List[float] = []
        for row_pos in range(len(feature_frame)):
            if invalid[row_pos]:
                statuses.append(AD_INVALID_FEATURES)
                supports.append(np.nan)
                nearest_indices.append("")
                mean_sims.append(np.nan)
                mean_distances.append(np.nan)
                continue
            row = matrix[row_pos].copy()
            if row_ids[row_pos] is not None:
                row[reference_indices == int(row_ids[row_pos])] = np.nan
            support, nearest_pos, mean_sim, mean_distance = _support_from_vector(
                row,
                metric=metric,
                top_k=top_k,
            )
            nearest_index = (
                int(reference_indices[int(nearest_pos)])
                if nearest_pos != "" and np.isfinite(support)
                else ""
            )
            status = AD_OUT_OF_DOMAIN if support < threshold else AD_IN_DOMAIN
            statuses.append(status)
            supports.append(support)
            nearest_indices.append(nearest_index)
            mean_sims.append(mean_sim)
            mean_distances.append(mean_distance)
        subspace_scores = pd.DataFrame(
            {
                "ad_similarity_status": statuses,
                "ad_similarity_subspaces": str(subspace),
                "ad_similarity_subspaces_out": [
                    str(subspace) if status == AD_OUT_OF_DOMAIN else "" for status in statuses
                ],
                "ad_similarity_support_score": supports,
                "ad_similarity_threshold": [threshold] * len(feature_frame),
                "ad_similarity_nearest_train_index": nearest_indices,
                "ad_similarity_top_k_neighbors": [top_k] * len(feature_frame),
                f"ad_similarity_{subspace}_status": statuses,
                f"ad_similarity_{subspace}_support_score": supports,
                f"ad_similarity_{subspace}_threshold": [threshold] * len(feature_frame),
                f"ad_similarity_{subspace}_nearest_train_index": nearest_indices,
                f"ad_similarity_{subspace}_mean_top_k_similarity": mean_sims,
                f"ad_similarity_{subspace}_mean_top_k_distance": mean_distances,
                "ad_feature_space": str(manifest.get("feature_space") or ""),
            }
        )
        per_subspace_scores[str(subspace)] = subspace_scores
        summaries[str(subspace)] = _status_summary(subspace_scores, "ad_similarity_status")

    scores = pd.DataFrame(index=range(len(feature_frame)))
    scores["ad_similarity_subspaces"] = ";".join(per_subspace_scores)
    subspace_status_columns: List[str] = []
    for subspace, subspace_scores in per_subspace_scores.items():
        for column in subspace_scores.columns:
            if column not in scores:
                scores[column] = subspace_scores[column].reset_index(drop=True)
        status_column = f"ad_similarity_{subspace}_status"
        if status_column not in scores and "ad_similarity_status" in subspace_scores:
            scores[status_column] = subspace_scores["ad_similarity_status"].reset_index(drop=True)
        if status_column in scores:
            subspace_status_columns.append(status_column)

    statuses: List[str] = []
    out_subspaces: List[str] = []
    for row_pos in range(len(feature_frame)):
        row_statuses = [str(scores.loc[row_pos, column]) for column in subspace_status_columns]
        if AD_SCHEMA_MISMATCH in row_statuses:
            status = AD_SCHEMA_MISMATCH
        elif AD_INVALID_FEATURES in row_statuses:
            status = AD_INVALID_FEATURES
        elif AD_OUT_OF_DOMAIN in row_statuses:
            status = AD_OUT_OF_DOMAIN
        elif row_statuses and all(item == AD_IN_DOMAIN for item in row_statuses):
            status = AD_IN_DOMAIN
        else:
            status = AD_UNAVAILABLE
        statuses.append(status)
        out_subspaces.append(
            ";".join(
                subspace
                for subspace in per_subspace_scores
                if scores.get(f"ad_similarity_{subspace}_status", pd.Series(dtype=object)).get(row_pos)
                == AD_OUT_OF_DOMAIN
            )
        )
    scores["ad_status"] = statuses
    scores["ad_method"] = SIMILARITY_MATRIX_METHOD
    scores["ad_methods"] = SIMILARITY_MATRIX_METHOD
    scores["ad_methods_out"] = [
        SIMILARITY_MATRIX_METHOD if status == AD_OUT_OF_DOMAIN else ""
        for status in statuses
    ]
    scores["ad_similarity_status"] = statuses
    scores["ad_similarity_subspaces_out"] = out_subspaces
    support_columns = [
        f"ad_similarity_{subspace}_support_score" for subspace in per_subspace_scores
        if f"ad_similarity_{subspace}_support_score" in scores.columns
    ]
    threshold_columns = [
        f"ad_similarity_{subspace}_threshold" for subspace in per_subspace_scores
        if f"ad_similarity_{subspace}_threshold" in scores.columns
    ]
    if support_columns:
        scores["ad_similarity_support_score"] = (
            scores[support_columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
        )
    if threshold_columns:
        scores["ad_similarity_threshold"] = (
            scores[threshold_columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
        )
    scores["ad_violation_count"] = [0] * len(feature_frame)
    scores["ad_violating_features"] = out_subspaces
    scores["ad_max_excess"] = np.maximum(
        pd.to_numeric(scores.get("ad_similarity_threshold"), errors="coerce")
        - pd.to_numeric(scores.get("ad_similarity_support_score"), errors="coerce"),
        0.0,
    )
    for column in AD_COLUMNS:
        if column not in scores.columns:
            scores[column] = np.nan if column.endswith(("score", "distance", "similarity", "threshold", "excess")) else ""
    scores = scores[AD_COLUMNS]
    summary = _summarize_scores(scores)
    summary.update(
        {
            "available": bool(per_subspace_scores),
            "method": SIMILARITY_MATRIX_METHOD,
            "feature_space": manifest.get("feature_space"),
            "subspace_summaries": summaries,
            "top_k_neighbors": method.get("top_k_neighbors"),
            "threshold_percentile": method.get("threshold_percentile"),
        }
    )
    score_path = None
    if output_dir:
        label = safe_slug(score_label or "scores") or "scores"
        target = Path(output_dir).expanduser() / f"scores_{label}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target, index=False)
        score_path = str(target)
        summary["scores_path"] = score_path
    return {
        "available": bool(per_subspace_scores),
        "method": SIMILARITY_MATRIX_METHOD,
        "scores": scores,
        "summary": summary,
        "scores_path": score_path,
        "subspace_summaries": summaries,
    }


def load_modern_ad_manifest(
    applicability_domain: Mapping[str, Any],
    *,
    metadata_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Return the modern AD manifest and path, including combined V2 manifests."""
    ad = dict(applicability_domain or {})
    if not ad:
        return None, None
    manifest_path = ad.get("manifest_path")
    if manifest_path:
        path = _resolve_path(str(manifest_path), metadata_path=metadata_path)
        if path.exists():
            return _merge_manifest_overrides(json.loads(path.read_text()), ad), path
    if ad.get("methods") or ad.get("primary_method") in {
        BOUNDING_BOX_METHOD,
        ISOLATION_FOREST_METHOD,
        SIMILARITY_MATRIX_METHOD,
        COMBINED_METHOD,
    }:
        return ad, None
    return None, None


def has_modern_applicability_domain(applicability_domain: Mapping[str, Any]) -> bool:
    ad = dict(applicability_domain or {})
    return bool(
        ad.get("manifest_path")
        or ad.get("methods")
        or ad.get("primary_method") in {
            BOUNDING_BOX_METHOD,
            ISOLATION_FOREST_METHOD,
            SIMILARITY_MATRIX_METHOD,
            COMBINED_METHOD,
        }
        or ad.get("method") in {
            BOUNDING_BOX_METHOD,
            ISOLATION_FOREST_METHOD,
            SIMILARITY_MATRIX_METHOD,
            COMBINED_METHOD,
        }
    )


def fit_modern_applicability_domain(
    *,
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    output_dir: str | Path,
    model_id: str,
    feature_space: str,
    representation_name: Optional[str] = None,
    feature_metadata: Optional[Mapping[str, Any]] = None,
    methods: Optional[Sequence[str] | str] = None,
    random_state: Optional[int] = None,
    all_feature_frame: Optional[pd.DataFrame] = None,
    train_indices: Optional[Sequence[int]] = None,
    split_indices: Optional[Mapping[str, Sequence[int]]] = None,
    similarity_top_k_neighbors: int | str | None = DEFAULT_SIMILARITY_TOP_K_NEIGHBORS,
    similarity_threshold_percentile: float | str | None = DEFAULT_SIMILARITY_THRESHOLD_PERCENTILE,
) -> Dict[str, Any]:
    """Fit all requested modern AD methods on the same training feature frame."""
    requested_methods = normalize_ad_methods(methods)
    ad_dir = Path(output_dir).expanduser()
    ad_dir.mkdir(parents=True, exist_ok=True)
    method_payloads: Dict[str, Dict[str, Any]] = {}
    errors: Dict[str, str] = {}

    if BOUNDING_BOX_METHOD in requested_methods:
        bounding = fit_bounding_box_domain(
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            output_dir=ad_dir,
            model_id=model_id,
            feature_space=feature_space,
            representation_name=representation_name,
            feature_metadata=feature_metadata,
        )
        if bounding.get("available"):
            method_payloads[BOUNDING_BOX_METHOD] = dict(
                (bounding.get("methods") or {}).get(BOUNDING_BOX_METHOD) or {}
            )
        else:
            errors[BOUNDING_BOX_METHOD] = str(bounding.get("reason") or bounding.get("error"))

    if ISOLATION_FOREST_METHOD in requested_methods:
        isolation = fit_isolation_forest_domain(
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            output_dir=ad_dir,
            model_id=model_id,
            feature_space=feature_space,
            representation_name=representation_name,
            random_state=random_state,
        )
        if isolation.get("available"):
            method_payloads[ISOLATION_FOREST_METHOD] = {
                "version": MODERN_AD_VERSION,
                "feature_space": feature_space,
                "representation_name": representation_name or feature_space,
                "feature_count": isolation["feature_count"],
                "feature_names": isolation["feature_names"],
                "feature_kinds": isolation["feature_kinds"],
                "schema_hash": isolation["schema_hash"],
                "model_path": isolation["model_path"],
                "params": isolation["params"],
                "isolation_forest_params": isolation["isolation_forest_params"],
                "isolation_forest_offset": isolation["isolation_forest_offset"],
                "threshold": 0.0,
                "decision_rule": isolation["decision_rule"],
            }
        else:
            errors[ISOLATION_FOREST_METHOD] = str(isolation.get("reason") or isolation.get("error"))

    if SIMILARITY_MATRIX_METHOD in requested_methods:
        similarity = fit_similarity_matrix_domain(
            feature_frame=all_feature_frame if all_feature_frame is not None else feature_frame,
            feature_columns=feature_columns,
            output_dir=ad_dir,
            model_id=model_id,
            feature_space=feature_space,
            representation_name=representation_name,
            train_indices=train_indices,
            split_indices=split_indices,
            top_k_neighbors=similarity_top_k_neighbors,
            threshold_percentile=similarity_threshold_percentile,
        )
        if similarity.get("available"):
            similarity_feature_names = [
                feature
                for subspace_payload in (similarity.get("subspaces") or {}).values()
                for feature in (subspace_payload.get("feature_names") or [])
            ]
            similarity_feature_kinds = feature_kinds_for_columns(similarity_feature_names)
            method_payloads[SIMILARITY_MATRIX_METHOD] = {
                "version": MODERN_AD_VERSION,
                "feature_space": feature_space,
                "representation_name": representation_name or feature_space,
                "feature_count": len(similarity_feature_names),
                "feature_names": similarity_feature_names,
                "feature_kinds": similarity_feature_kinds,
                "activity_cliffs_reusable": True,
                "top_k_neighbors": similarity.get("top_k_neighbors"),
                "threshold_percentile": similarity.get("threshold_percentile"),
                "manifest_path": similarity.get("manifest_path"),
                "subspaces": similarity.get("subspaces") or {},
                "errors": similarity.get("errors") or {},
                "decision_rule": "support_score_less_than_p5_train_threshold_is_out_of_domain",
            }
        else:
            errors[SIMILARITY_MATRIX_METHOD] = str(
                similarity.get("reason") or similarity.get("error")
            )

    if not method_payloads:
        return {
            "available": False,
            "method": COMBINED_METHOD,
            "requested_methods": requested_methods,
            "errors": errors,
            "reason": "No modern applicability-domain method could be fit.",
        }

    first_method = method_payloads[next(iter(method_payloads))]
    manifest_path = ad_dir / "manifest.json"
    plots_dir = ad_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "available": True,
        "version": MODERN_AD_VERSION,
        "primary_method": COMBINED_METHOD if len(method_payloads) > 1 else next(iter(method_payloads)),
        "method": COMBINED_METHOD if len(method_payloads) > 1 else next(iter(method_payloads)),
        "model_id": model_id,
        "feature_space": feature_space,
        "representation_name": representation_name or feature_space,
        "feature_count": first_method.get("feature_count"),
        "feature_names": first_method.get("feature_names") or [],
        "feature_kinds": first_method.get("feature_kinds") or [],
        "schema_hash": first_method.get("schema_hash"),
        "manifest_path": str(manifest_path),
        "plots_dir": str(plots_dir),
        "methods": method_payloads,
        "requested_methods": requested_methods,
        "fit_errors": errors,
    }
    if BOUNDING_BOX_METHOD in method_payloads:
        manifest["bounds_path"] = method_payloads[BOUNDING_BOX_METHOD].get("bounds_path")
    if ISOLATION_FOREST_METHOD in method_payloads:
        manifest["isolation_forest_model_path"] = method_payloads[ISOLATION_FOREST_METHOD].get(
            "model_path"
        )
        manifest["isolation_forest_params"] = method_payloads[ISOLATION_FOREST_METHOD].get(
            "isolation_forest_params"
        )
        manifest["isolation_forest_offset"] = method_payloads[ISOLATION_FOREST_METHOD].get(
            "isolation_forest_offset"
        )
    if SIMILARITY_MATRIX_METHOD in method_payloads:
        manifest["similarity_matrix_manifest_path"] = method_payloads[
            SIMILARITY_MATRIX_METHOD
        ].get("manifest_path")
        manifest["activity_cliffs_reusable"] = True
    if feature_metadata:
        manifest.update(dict(feature_metadata))
    _write_manifest(manifest_path, manifest)
    return manifest


def _combine_modern_scores(method_scores: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    if not method_scores:
        return pd.DataFrame()
    methods = list(method_scores)
    row_count = len(next(iter(method_scores.values())))
    combined = pd.DataFrame(index=range(row_count))
    combined["ad_methods"] = ";".join(methods)
    combined["ad_method"] = combined["ad_methods"]

    for method, scores in method_scores.items():
        if method == BOUNDING_BOX_METHOD:
            prefix = "ad_bounding_box"
        elif method == ISOLATION_FOREST_METHOD:
            prefix = "ad_isolation_forest"
        else:
            prefix = "ad_similarity"
        for column in scores.columns:
            if column.startswith(prefix):
                combined[column] = scores[column].reset_index(drop=True)

    status_columns = []
    if "ad_bounding_box_status" in combined:
        status_columns.append("ad_bounding_box_status")
    if "ad_isolation_forest_status" in combined:
        status_columns.append("ad_isolation_forest_status")
    if "ad_similarity_status" in combined:
        status_columns.append("ad_similarity_status")

    statuses: List[str] = []
    methods_out: List[str] = []
    for row_index in range(row_count):
        row_statuses = [str(combined.loc[row_index, column]) for column in status_columns]
        if AD_SCHEMA_MISMATCH in row_statuses:
            status = AD_SCHEMA_MISMATCH
        elif AD_INVALID_FEATURES in row_statuses:
            status = AD_INVALID_FEATURES
        elif AD_OUT_OF_DOMAIN in row_statuses:
            status = AD_OUT_OF_DOMAIN
        elif row_statuses and all(item == AD_IN_DOMAIN for item in row_statuses):
            status = AD_IN_DOMAIN
        else:
            status = AD_UNAVAILABLE
        statuses.append(status)
        out_methods = []
        if combined.get("ad_bounding_box_status", pd.Series(dtype=object)).get(row_index) == AD_OUT_OF_DOMAIN:
            out_methods.append(BOUNDING_BOX_METHOD)
        if combined.get("ad_isolation_forest_status", pd.Series(dtype=object)).get(row_index) == AD_OUT_OF_DOMAIN:
            out_methods.append(ISOLATION_FOREST_METHOD)
        if combined.get("ad_similarity_status", pd.Series(dtype=object)).get(row_index) == AD_OUT_OF_DOMAIN:
            out_methods.append(SIMILARITY_MATRIX_METHOD)
        methods_out.append(";".join(out_methods))

    combined["ad_status"] = statuses
    combined["ad_methods_out"] = methods_out
    combined["ad_violation_count"] = combined.get("ad_bounding_box_violation_count", pd.Series([0] * row_count))
    combined["ad_violating_features"] = combined.get(
        "ad_bounding_box_violating_features", pd.Series([""] * row_count)
    )
    combined["ad_max_excess"] = combined.get("ad_bounding_box_max_excess", pd.Series([np.nan] * row_count))
    feature_space = ""
    for scores in method_scores.values():
        if "ad_feature_space" in scores and not scores["ad_feature_space"].dropna().empty:
            feature_space = str(scores["ad_feature_space"].dropna().iloc[0])
            break
    combined["ad_feature_space"] = feature_space
    for column in AD_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan if column.endswith(("score", "decision", "threshold", "excess")) else ""
    return combined[AD_COLUMNS]


def score_modern_applicability_domain(
    *,
    feature_frame: pd.DataFrame,
    applicability_domain: Mapping[str, Any],
    output_dir: Optional[str | Path] = None,
    score_label: Optional[str] = None,
    metadata_path: Optional[str] = None,
    row_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Score rows against every modern AD method available in the manifest."""
    manifest, _ = load_modern_ad_manifest(
        applicability_domain,
        metadata_path=metadata_path,
    )
    if not manifest:
        scores = _schema_mismatch_scores(
            len(feature_frame),
            feature_space="",
            reason="No modern applicability domain is available for this model.",
        )
        return {"available": False, "reason": "No modern applicability domain is available for this model.", "scores": scores, "summary": _summarize_scores(scores)}

    methods = manifest.get("methods") or {}
    method_scores: Dict[str, pd.DataFrame] = {}
    method_summaries: Dict[str, Any] = {}
    if BOUNDING_BOX_METHOD in methods or manifest.get("method") == BOUNDING_BOX_METHOD:
        scored = score_bounding_box_domain(
            feature_frame=feature_frame,
            applicability_domain=manifest,
            output_dir=(Path(output_dir).expanduser() / BOUNDING_BOX_METHOD) if output_dir else None,
            score_label=score_label,
            metadata_path=metadata_path,
        )
        method_scores[BOUNDING_BOX_METHOD] = scored["scores"]
        method_summaries[BOUNDING_BOX_METHOD] = scored.get("summary") or {}
    if ISOLATION_FOREST_METHOD in methods:
        scored = score_isolation_forest_domain(
            feature_frame=feature_frame,
            applicability_domain=manifest,
            output_dir=(Path(output_dir).expanduser() / ISOLATION_FOREST_METHOD) if output_dir else None,
            score_label=score_label,
            metadata_path=metadata_path,
        )
        method_scores[ISOLATION_FOREST_METHOD] = scored["scores"]
        method_summaries[ISOLATION_FOREST_METHOD] = scored.get("summary") or {}
    if SIMILARITY_MATRIX_METHOD in methods:
        scored = score_similarity_matrix_domain(
            feature_frame=feature_frame,
            applicability_domain=manifest,
            output_dir=(Path(output_dir).expanduser() / SIMILARITY_MATRIX_METHOD) if output_dir else None,
            score_label=score_label,
            metadata_path=metadata_path,
            row_indices=row_indices,
        )
        method_scores[SIMILARITY_MATRIX_METHOD] = scored["scores"]
        method_summaries[SIMILARITY_MATRIX_METHOD] = scored.get("summary") or {}

    scores = _combine_modern_scores(method_scores)
    summary = _summarize_scores(scores)
    summary.update(
        {
            "available": bool(method_scores),
            "method": manifest.get("method") or manifest.get("primary_method"),
            "primary_method": manifest.get("primary_method"),
            "methods": list(method_scores),
            "method_summaries": method_summaries,
            "feature_space": manifest.get("feature_space"),
        }
    )

    score_path = None
    if output_dir:
        label = safe_slug(score_label or "scores") or "scores"
        target = Path(output_dir).expanduser() / f"scores_{label}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(target, index=False)
        score_path = str(target)
        summary["scores_path"] = score_path

    return {
        "available": bool(method_scores),
        "method": summary.get("method"),
        "scores": scores,
        "summary": summary,
        "scores_path": score_path,
        "method_summaries": method_summaries,
    }


def append_ad_scores_to_csv(csv_path: str | Path, scores: pd.DataFrame) -> List[str]:
    """Append standard AD columns to an existing prediction CSV."""
    path = Path(csv_path).expanduser()
    frame = pd.read_csv(path)
    for column in AD_COLUMNS:
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    ad_frame = scores.copy()
    for column in AD_COLUMNS:
        if column not in ad_frame.columns:
            ad_frame[column] = np.nan if column.endswith(("score", "decision", "threshold", "excess")) else ""
    ad_frame = ad_frame[AD_COLUMNS].reset_index(drop=True)
    frame = pd.concat([frame.reset_index(drop=True), ad_frame], axis=1)
    frame.to_csv(path, index=False)
    return list(AD_COLUMNS)


def build_bounding_box_plots(scores: pd.DataFrame, output_dir: str | Path) -> Dict[str, str]:
    """Build lightweight AD diagnostic plots from a score table."""
    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    if scores.empty:
        return artifacts

    status_counts = scores["ad_status"].value_counts()
    plt.figure(figsize=(6, 4))
    status_counts.plot(kind="bar", color=["#2e7d32", "#c62828", "#6a1b9a", "#616161"])
    plt.xlabel("AD status")
    plt.ylabel("Rows")
    plt.title("Applicability-domain status")
    path = target_dir / "ad_status_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["ad_status_distribution"] = str(path)

    plt.figure(figsize=(6, 4))
    pd.to_numeric(scores["ad_violation_count"], errors="coerce").fillna(0).hist(bins=30)
    plt.xlabel("Violation count")
    plt.ylabel("Rows")
    plt.title("Bounding-box violations")
    path = target_dir / "ad_violation_count_histogram.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["ad_violation_count_histogram"] = str(path)

    counter: Counter[str] = Counter()
    for raw in scores["ad_violating_features"].fillna(""):
        for feature in str(raw).split(";"):
            if feature:
                counter[feature] += 1
    top = counter.most_common(20)
    if top:
        labels = [item[0] for item in top]
        values = [item[1] for item in top]
        plt.figure(figsize=(8, max(4, len(labels) * 0.25)))
        plt.barh(range(len(labels)), values)
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Violating rows")
        plt.title("Top violating features")
        plt.gca().invert_yaxis()
        path = target_dir / "ad_top_violating_features.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        artifacts["ad_top_violating_features"] = str(path)
    return artifacts


def build_isolation_forest_plots(scores: pd.DataFrame, output_dir: str | Path) -> Dict[str, str]:
    """Build lightweight Isolation Forest AD diagnostic plots."""
    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    if scores.empty or "ad_isolation_forest_status" not in scores.columns:
        return artifacts

    status_counts = scores["ad_isolation_forest_status"].value_counts()
    plt.figure(figsize=(6, 4))
    status_counts.plot(kind="bar", color=["#2e7d32", "#c62828", "#616161"])
    plt.xlabel("Isolation Forest status")
    plt.ylabel("Rows")
    plt.title("Isolation Forest AD status")
    path = target_dir / "ad_isolation_forest_status_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["ad_isolation_forest_status_distribution"] = str(path)

    decisions = pd.to_numeric(scores.get("ad_isolation_forest_decision"), errors="coerce")
    if decisions.notna().any():
        plt.figure(figsize=(6, 4))
        decisions.dropna().hist(bins=30)
        plt.axvline(0.0, color="#c62828", linestyle="--", linewidth=1.5)
        plt.xlabel("decision_function")
        plt.ylabel("Rows")
        plt.title("Isolation Forest decision scores")
        path = target_dir / "ad_isolation_forest_decision_distribution.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        artifacts["ad_isolation_forest_decision_distribution"] = str(path)

    score_samples = pd.to_numeric(scores.get("ad_isolation_forest_score"), errors="coerce")
    if score_samples.notna().any():
        plt.figure(figsize=(6, 4))
        score_samples.dropna().hist(bins=30)
        plt.xlabel("score_samples")
        plt.ylabel("Rows")
        plt.title("Isolation Forest raw scores")
        path = target_dir / "ad_isolation_forest_score_distribution.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        artifacts["ad_isolation_forest_score_distribution"] = str(path)
    return artifacts


def build_similarity_matrix_plots(scores: pd.DataFrame, output_dir: str | Path) -> Dict[str, str]:
    """Build lightweight similarity-matrix AD diagnostic plots."""
    target_dir = Path(output_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    if scores.empty or "ad_similarity_status" not in scores.columns:
        return artifacts

    status_counts = scores["ad_similarity_status"].value_counts()
    plt.figure(figsize=(6, 4))
    status_counts.plot(kind="bar", color=["#2e7d32", "#c62828", "#616161"])
    plt.xlabel("Similarity AD status")
    plt.ylabel("Rows")
    plt.title("Similarity matrix AD status")
    path = target_dir / "ad_similarity_status_distribution.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    artifacts["ad_similarity_status_distribution"] = str(path)

    support = pd.to_numeric(scores.get("ad_similarity_support_score"), errors="coerce")
    threshold = pd.to_numeric(scores.get("ad_similarity_threshold"), errors="coerce")
    if support.notna().any():
        plt.figure(figsize=(6, 4))
        support.dropna().hist(bins=30)
        if threshold.notna().any():
            plt.axvline(float(threshold.dropna().iloc[0]), color="#c62828", linestyle="--")
        plt.xlabel("Support score")
        plt.ylabel("Rows")
        plt.title("Similarity support scores")
        path = target_dir / "ad_similarity_support_distribution.png"
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        artifacts["ad_similarity_support_distribution"] = str(path)
    return artifacts


def build_modern_ad_plots(scores: pd.DataFrame, output_dir: str | Path) -> Dict[str, str]:
    """Build plots for every modern AD method present in a score table."""
    target_dir = Path(output_dir).expanduser()
    artifacts: Dict[str, str] = {}
    artifacts.update(build_bounding_box_plots(scores, target_dir / BOUNDING_BOX_METHOD))
    artifacts.update(build_isolation_forest_plots(scores, target_dir / ISOLATION_FOREST_METHOD))
    artifacts.update(
        build_similarity_matrix_plots(scores, target_dir / SIMILARITY_MATRIX_METHOD)
    )
    if {"ad_bounding_box_status", "ad_isolation_forest_status"}.issubset(scores.columns):
        crosstab = pd.crosstab(
            scores["ad_bounding_box_status"],
            scores["ad_isolation_forest_status"],
        )
        if not crosstab.empty:
            plt.figure(figsize=(6, 5))
            plt.imshow(crosstab.to_numpy(), cmap="Blues")
            plt.xticks(range(len(crosstab.columns)), crosstab.columns, rotation=30, ha="right")
            plt.yticks(range(len(crosstab.index)), crosstab.index)
            for row_index, row_label in enumerate(crosstab.index):
                for col_index, col_label in enumerate(crosstab.columns):
                    plt.text(col_index, row_index, int(crosstab.loc[row_label, col_label]), ha="center", va="center")
            plt.xlabel("Isolation Forest")
            plt.ylabel("Bounding box")
            plt.title("AD method concordance")
            path = target_dir / "ad_method_concordance.png"
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            plt.close()
            artifacts["ad_method_concordance"] = str(path)
    return artifacts


def _tabular_representation_name(record: PredictionModelRecord) -> Optional[str]:
    return (
        (record.inference_profile or {}).get("representation_name")
        or (record.training_data_summary or {}).get("representation_name")
        or (record.selection_hints or {}).get("representation_name")
    )


def _feature_columns(record: PredictionModelRecord, manifest: Mapping[str, Any]) -> List[str]:
    methods = manifest.get("methods") or {}
    method = methods.get(BOUNDING_BOX_METHOD) or {}
    feature_count = int(method.get("feature_count") or manifest.get("feature_count") or 0)
    columns = list((record.inference_profile or {}).get("feature_columns") or [])
    if not columns:
        columns = list((record.training_data_summary or {}).get("feature_columns") or [])
    if not columns:
        columns = list((record.selection_hints or {}).get("feature_columns") or [])
    if columns and feature_count and len(columns) != feature_count:
        return [str(column) for column in columns[:feature_count]]
    return [str(column) for column in columns]


def prepare_tabular_features_for_ad(
    *,
    input_csv: str,
    record: PredictionModelRecord,
    output_dir: str | Path,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Rebuild the tabular feature frame used by LightGBM/TabICL models."""
    source = pd.read_csv(Path(input_csv).expanduser())
    existing = [column for column in feature_columns if column in source.columns]
    if len(existing) == len(feature_columns):
        return source

    representation_name = _tabular_representation_name(record)
    if not representation_name:
        return source
    try:
        spec = get_tabular_representation(str(representation_name))
    except ValueError:
        return source

    try:
        smiles_column = resolve_smiles_column_name(source, "smiles")
    except ValueError:
        return source

    feature_dir = Path(output_dir).expanduser() / "ad_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    toolkit = MolecularFeatureToolkit()
    feature_frames: List[pd.DataFrame] = []
    if spec.use_morgan_binary:
        output_csv = feature_dir / "morgan_binary.csv"
        toolkit.smiles_to_morgan_fingerprints(
            input_csv=input_csv,
            smiles_column=smiles_column,
            output_csv=str(output_csv),
            include_input_columns=True,
            input_columns_to_keep=["smiles"],
            feature_prefix="fp_",
            fingerprint_kind="binary",
            n_jobs=1,
        )
        feature_frames.append(pd.read_csv(output_csv))
    if spec.use_morgan_count:
        output_csv = feature_dir / "morgan_count.csv"
        toolkit.smiles_to_morgan_fingerprints(
            input_csv=input_csv,
            smiles_column=smiles_column,
            output_csv=str(output_csv),
            include_input_columns=True,
            input_columns_to_keep=["smiles"],
            feature_prefix="cfp_",
            fingerprint_kind="count",
            n_jobs=1,
        )
        feature_frames.append(pd.read_csv(output_csv))
    if spec.use_rdkit:
        output_csv = feature_dir / "rdkit_descriptors.csv"
        toolkit.smiles_to_rdkit_descriptors(
            input_csv=input_csv,
            smiles_column=smiles_column,
            output_csv=str(output_csv),
            descriptor_set=str(spec.descriptor_set or "basic"),
            include_input_columns=True,
            input_columns_to_keep=["smiles"],
            n_jobs=1,
        )
        feature_frames.append(pd.read_csv(output_csv))

    assembled = source.copy()
    additions: List[pd.DataFrame] = []
    for frame in feature_frames:
        columns_to_add = [
            column for column in feature_columns if column in frame.columns and column not in assembled
        ]
        if columns_to_add:
            additions.append(frame[columns_to_add].reset_index(drop=True))
    if additions:
        assembled = pd.concat([assembled.reset_index(drop=True), *additions], axis=1)
    return assembled


def score_record_applicability_domain(
    *,
    record: PredictionModelRecord,
    input_csv: str,
    output_dir: str | Path,
    score_label: str,
    backend: Optional[Any] = None,
) -> Dict[str, Any]:
    """Score a catalog/session model's modern AD on an input CSV."""
    manifest, _ = load_modern_ad_manifest(
        record.applicability_domain or {},
        metadata_path=record.metadata_path,
    )
    input_frame = pd.read_csv(Path(input_csv).expanduser())
    if not manifest:
        scores = pd.DataFrame(
            {
                "ad_status": [AD_UNAVAILABLE] * len(input_frame),
                "ad_method": [COMBINED_METHOD] * len(input_frame),
            }
        )
        for column in AD_COLUMNS:
            if column not in scores.columns:
                scores[column] = np.nan if column.endswith(("score", "decision", "threshold", "excess")) else ""
        return {
            "available": False,
            "reason": "No modern applicability domain is available for this model.",
            "scores": scores[AD_COLUMNS],
            "summary": _summarize_scores(scores),
        }

    bounds_path = _bounds_path_from_manifest(
        manifest,
        metadata_path=record.metadata_path,
    )
    expected_columns: List[str] = []
    if bounds_path is not None and bounds_path.exists():
        payload = np.load(bounds_path, allow_pickle=False)
        expected_columns = [str(item) for item in payload["feature_names"].tolist()]
    else:
        similarity_method = (manifest.get("methods") or {}).get(SIMILARITY_MATRIX_METHOD) or {}
        similarity_columns = [
            str(feature)
            for subspace_payload in (similarity_method.get("subspaces") or {}).values()
            for feature in (subspace_payload.get("feature_names") or [])
        ]
        expected_columns = [
            str(item)
            for item in (
                manifest.get("feature_names")
                or ((manifest.get("methods") or {}).get(ISOLATION_FOREST_METHOD) or {}).get(
                    "feature_names"
                )
                or similarity_method.get("feature_names")
                or similarity_columns
                or _feature_columns(record, manifest)
            )
        ]

    feature_frame = input_frame.copy()
    feature_space = str(manifest.get("feature_space") or "")
    if feature_space == "chemprop_embedding":
        if backend is None or not hasattr(backend, "fingerprint_from_csv"):
            scores = _schema_mismatch_scores(
                len(input_frame),
                feature_space=feature_space,
                reason="Chemprop embedding AD requires a Chemprop backend with fingerprint support.",
            )
            return {
                "available": False,
                "reason": "Chemprop embedding AD could not be scored without fingerprint support.",
                "scores": scores,
                "summary": _summarize_scores(scores),
            }
        fingerprint_meta = (
            manifest.get("chemprop_fingerprint")
            or (record.applicability_domain or {}).get("chemprop_fingerprint")
            or {}
        )
        try:
            smiles_columns = ["smiles"] if "smiles" in input_frame.columns else record.task.smiles_columns
            fingerprints = backend.fingerprint_from_csv(
                input_csv=input_csv,
                model_path=record.model_path,
                output_csv=str(
                    Path(output_dir).expanduser()
                    / "chemprop_embeddings"
                    / f"{safe_slug(score_label) or 'scores'}.csv"
                ),
                smiles_columns=smiles_columns or ["smiles"],
                ffn_block_index=int(fingerprint_meta.get("ffn_block_index") or -1),
            )
            feature_frame = pd.read_csv(fingerprints["fingerprints_path"])
        except Exception as exc:
            reason = f"Chemprop embedding extraction failed for applicability-domain scoring: {exc}"
            scores = _schema_mismatch_scores(
                len(input_frame),
                feature_space=feature_space,
                reason=reason,
            )
            return {
                "available": False,
                "reason": reason,
                "scores": scores,
                "summary": _summarize_scores(scores),
            }
    elif (
        record.backend_name in {"lightgbm", "tabicl"}
        or _tabular_representation_name(record)
        or feature_space in {"rdkit_all", "morgan_only", "morgan_count_only"}
        or feature_space.startswith("morgan")
        or feature_space.startswith("rdkit")
    ):
        feature_frame = prepare_tabular_features_for_ad(
            input_csv=input_csv,
            record=record,
            output_dir=output_dir,
            feature_columns=expected_columns,
        )

    result = score_modern_applicability_domain(
        feature_frame=feature_frame,
        applicability_domain=record.applicability_domain or manifest,
        output_dir=output_dir,
        score_label=score_label,
        metadata_path=record.metadata_path,
    )
    if result.get("available"):
        plots = build_modern_ad_plots(
            result["scores"],
            Path(output_dir).expanduser() / "plots",
        )
        result["summary"]["plots"] = plots
    return result


def metrics_by_ad_status(
    *,
    frame: pd.DataFrame,
    metric_func: Any,
    target_column: str,
    prediction_column: str,
    score_column: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute global/in-domain/out-of-domain metrics for a prediction frame."""
    if "ad_status" not in frame.columns:
        return {}

    def compute(subset: pd.DataFrame) -> Dict[str, Any]:
        if subset.empty:
            return {}
        if score_column and score_column in subset.columns:
            return metric_func(
                subset[target_column],
                subset[prediction_column],
                positive_scores=subset[score_column],
                target_column=target_column,
            )
        return metric_func(
            subset[target_column],
            subset[prediction_column],
            target_column=target_column,
        )

    in_domain = frame.loc[frame["ad_status"] == AD_IN_DOMAIN]
    out_domain = frame.loc[frame["ad_status"] == AD_OUT_OF_DOMAIN]
    return {
        "metrics_all": compute(frame),
        "metrics_in_domain": compute(in_domain),
        "metrics_out_of_domain": compute(out_domain),
        "coverage_in_domain": float(len(in_domain) / len(frame)) if len(frame) else None,
        "n_out_of_domain": int(len(out_domain)),
        "ad_status_counts": {
            str(key): int(value) for key, value in frame["ad_status"].value_counts().items()
        },
    }
