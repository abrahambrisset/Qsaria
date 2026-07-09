#!/usr/bin/env python
# coding: utf-8
"""Small helpers to keep tool responses out of the LLM's context ceiling."""

from __future__ import annotations

from typing import Any, Dict, Optional

FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT = 20


def feature_columns_summary_for_response(
    feature_columns: Optional[list[str]],
    *,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    columns = list(feature_columns or [])
    omitted_count = max(0, len(columns) - FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT)
    payload: Dict[str, Any] = {
        "feature_columns_count": len(columns),
        "feature_columns_sample": columns[:FEATURE_COLUMN_RESPONSE_SAMPLE_LIMIT],
        "feature_columns_omitted_count": omitted_count,
    }
    if source:
        payload["feature_columns_source"] = source
    if omitted_count:
        payload["feature_columns_note"] = (
            "Full feature columns are stored in artifacts/metadata and omitted from "
            "this tool response to keep agent context bounded."
        )
    return payload


def compact_inference_profile_for_response(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = dict(profile or {})
    feature_columns = payload.pop("feature_columns", None)
    if isinstance(feature_columns, list):
        payload.update(
            feature_columns_summary_for_response(
                feature_columns,
                source=payload.get("feature_columns_source"),
            )
        )
    return payload


def compact_training_data_summary_for_response(summary: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    summary = dict(summary or {})
    return {
        key: summary.get(key)
        for key in (
            "trained_at",
            "trained_date",
            "trained_time",
            "endpoint_name",
            "dataset_name",
            "validation_protocol",
            "validation_strategy_type",
            "metrics_status",
            "evaluation_required",
            "seed_policy_report",
            "representation_name",
            "task_kind",
            "class_count",
            "training_summary_path",
            "feature_preparation",
            "feature_preparation_durations",
            "activity_cliffs",
        )
        if summary.get(key) is not None
    }


def _compact_ad_split_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    if not summary:
        return {}
    return {
        key: summary.get(key)
        for key in (
            "row_count",
            "n_in_domain",
            "n_out_of_domain",
            "n_invalid_features",
            "coverage_in_domain",
            "status_counts",
            "method_summaries",
            "method_status_summaries",
            "top_violating_features",
            "metrics_all",
            "metrics_in_domain",
            "metrics_out_of_domain",
        )
        if summary.get(key) is not None
    }


def _compact_ad_subspace(payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "metric",
            "feature_count",
            "schema_hash",
            "matrix_all_path",
            "reference_features_path",
            "row_count",
            "train_size",
            "valid_train_size",
            "invalid_row_count",
            "top_k_neighbors",
            "threshold_percentile",
            "threshold",
            "support_score",
            "decision_rule",
        )
        if payload.get(key) is not None
    }
    standardization = payload.get("standardization")
    if isinstance(standardization, dict):
        compact["standardization"] = {
            key: standardization.get(key)
            for key in ("enabled", "zero_variance_feature_count", "active_feature_count")
            if standardization.get(key) is not None
        }
    return compact


def _compact_ad_method(method_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        key: payload.get(key)
        for key in (
            "version",
            "feature_space",
            "representation_name",
            "feature_count",
            "schema_hash",
            "bounds_path",
            "model_path",
            "manifest_path",
            "params",
            "isolation_forest_params",
            "isolation_forest_offset",
            "threshold",
            "rule",
            "decision_rule",
            "activity_cliffs_reusable",
            "top_k_neighbors",
            "threshold_percentile",
            "errors",
        )
        if payload.get(key) is not None
    }
    subspaces = payload.get("subspaces")
    if method_name == "similarity_matrix" and isinstance(subspaces, dict):
        compact["subspaces"] = {
            name: _compact_ad_subspace(subspace)
            for name, subspace in subspaces.items()
            if isinstance(subspace, dict)
        }
    return compact


def compact_applicability_domain_for_response(ad: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not ad:
        return {}
    compact = {
        key: ad.get(key)
        for key in (
            "available",
            "version",
            "primary_method",
            "method",
            "model_id",
            "feature_space",
            "representation_name",
            "feature_count",
            "schema_hash",
            "manifest_path",
            "plots_dir",
            "requested_methods",
            "fit_errors",
            "bounds_path",
            "isolation_forest_model_path",
            "isolation_forest_params",
            "isolation_forest_offset",
            "similarity_matrix_manifest_path",
            "activity_cliffs_reusable",
            "scores_train_path",
            "scores_validation_path",
            "scores_test_path",
        )
        if ad.get(key) is not None
    }
    methods = ad.get("methods")
    if isinstance(methods, dict):
        compact["methods"] = {
            name: _compact_ad_method(name, payload)
            for name, payload in methods.items()
            if isinstance(payload, dict)
        }
    split_summaries = ad.get("split_score_summaries")
    if isinstance(split_summaries, dict):
        compact["split_score_summaries"] = {
            name: _compact_ad_split_summary(summary)
            for name, summary in split_summaries.items()
            if isinstance(summary, dict)
        }
    legacy = ad.get("legacy_similarity_ad")
    if isinstance(legacy, dict):
        compact["legacy_similarity_ad"] = {
            key: legacy.get(key)
            for key in (
                "available",
                "method",
                "threshold",
                "reference_size",
                "reference_store_path",
                "reference_manifest_path",
                "applicability_domain_path",
            )
            if legacy.get(key) is not None
        }
    return compact


def compact_model_payload_for_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(payload)
    compacted["inference_profile"] = compact_inference_profile_for_response(
        compacted.get("inference_profile")
    )
    if compacted.get("training_data_summary"):
        compacted["training_data_summary"] = compact_training_data_summary_for_response(
            compacted["training_data_summary"]
        )
    if compacted.get("applicability_domain"):
        compacted["applicability_domain"] = compact_applicability_domain_for_response(
            compacted["applicability_domain"]
        )
    return compacted


def compact_registry_payload_for_response(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not payload:
        return payload
    return compact_model_payload_for_response(dict(payload))


def compact_prediction_result_for_response(result: Dict[str, Any]) -> Dict[str, Any]:
    compacted = dict(result)
    feature_columns = compacted.pop("feature_columns", None)
    if isinstance(feature_columns, list):
        compacted.update(feature_columns_summary_for_response(feature_columns))
    if compacted.get("applicability_domain"):
        compacted["applicability_domain"] = compact_applicability_domain_for_response(
            compacted["applicability_domain"]
        )
    return compacted
