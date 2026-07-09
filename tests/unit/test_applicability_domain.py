import pandas as pd

from cs_copilot.tools.prediction.applicability_domain import (
    AD_IN_DOMAIN,
    AD_INVALID_FEATURES,
    AD_OUT_OF_DOMAIN,
    AD_SCHEMA_MISMATCH,
    ISOLATION_FOREST_METHOD,
    _combine_modern_scores,
    fit_bounding_box_domain,
    fit_isolation_forest_domain,
    fit_modern_applicability_domain,
    score_bounding_box_domain,
    score_isolation_forest_domain,
)
from cs_copilot.tools.prediction.backend import PredictionTaskSpec
from cs_copilot.tools.prediction.training_orchestration import (
    attach_ad_scores_and_metrics_to_predictions,
    build_applicability_domain_for_training,
)


def _fit(tmp_path, frame, columns):
    return fit_bounding_box_domain(
        feature_frame=frame,
        feature_columns=columns,
        output_dir=tmp_path / "ad",
        model_id="model",
        feature_space="rdkit_all",
        representation_name="rdkit_all",
    )


def test_bounding_box_one_feature_outside_is_out_of_domain(tmp_path):
    manifest = _fit(
        tmp_path,
        pd.DataFrame({"desc_a": [0.0, 1.0], "desc_b": [5.0, 6.0]}),
        ["desc_a", "desc_b"],
    )

    scored = score_bounding_box_domain(
        feature_frame=pd.DataFrame({"desc_a": [0.5, 0.5], "desc_b": [5.5, 7.0]}),
        applicability_domain=manifest,
    )

    scores = scored["scores"]
    assert list(scores["ad_status"]) == ["in_domain", AD_OUT_OF_DOMAIN]
    assert int(scores.loc[1, "ad_violation_count"]) == 1
    assert scores.loc[1, "ad_violating_features"] == "desc_b"


def test_morgan_binary_uses_strict_min_max(tmp_path):
    manifest = _fit(
        tmp_path,
        pd.DataFrame({"fp_0": [1, 1], "fp_1": [0, 1]}),
        ["fp_0", "fp_1"],
    )

    scored = score_bounding_box_domain(
        feature_frame=pd.DataFrame({"fp_0": [0], "fp_1": [1]}),
        applicability_domain=manifest,
    )

    assert scored["scores"].loc[0, "ad_status"] == AD_OUT_OF_DOMAIN
    assert scored["scores"].loc[0, "ad_violating_features"] == "fp_0"


def test_nan_or_inf_features_are_invalid(tmp_path):
    manifest = _fit(
        tmp_path,
        pd.DataFrame({"desc_a": [0.0, 1.0], "desc_b": [5.0, 6.0]}),
        ["desc_a", "desc_b"],
    )

    scored = score_bounding_box_domain(
        feature_frame=pd.DataFrame({"desc_a": [float("nan")], "desc_b": [5.5]}),
        applicability_domain=manifest,
    )

    assert scored["scores"].loc[0, "ad_status"] == AD_INVALID_FEATURES
    assert scored["scores"].loc[0, "ad_violating_features"] == "desc_a"


def test_schema_mismatch_gives_explicit_unavailable_status(tmp_path):
    manifest = _fit(
        tmp_path,
        pd.DataFrame({"desc_a": [0.0, 1.0], "desc_b": [5.0, 6.0]}),
        ["desc_a", "desc_b"],
    )

    scored = score_bounding_box_domain(
        feature_frame=pd.DataFrame({"desc_a": [0.5]}),
        applicability_domain=manifest,
    )

    assert scored["scores"].loc[0, "ad_status"] == AD_SCHEMA_MISMATCH
    assert "Missing 1 feature" in scored["reason"]


def test_isolation_forest_fit_persists_defaults_and_scores_outlier(tmp_path):
    train = pd.DataFrame({"desc_a": [0.0] * 64 + [1.0] * 64, "desc_b": [0.0] * 128})
    manifest = fit_modern_applicability_domain(
        feature_frame=train,
        feature_columns=["desc_a", "desc_b"],
        output_dir=tmp_path / "ad",
        model_id="model",
        feature_space="rdkit_all",
        methods=[ISOLATION_FOREST_METHOD],
        random_state=13,
    )

    method = manifest["methods"][ISOLATION_FOREST_METHOD]
    assert method["params"]["n_estimators"] == 100
    assert method["params"]["max_samples"] == "auto"
    assert method["params"]["contamination"] == "auto"
    assert method["params"]["random_state"] == 13
    assert (tmp_path / "ad" / "isolation_forest" / "model.joblib").exists()

    scored = score_isolation_forest_domain(
        feature_frame=pd.DataFrame({"desc_a": [0.0, 100.0], "desc_b": [0.0, 100.0]}),
        applicability_domain=manifest,
    )

    scores = scored["scores"]
    assert scores.loc[1, "ad_isolation_forest_decision"] < 0.0
    assert scores.loc[1, "ad_isolation_forest_status"] == AD_OUT_OF_DOMAIN
    assert (
        scores.loc[1, "ad_isolation_forest_status"] == AD_OUT_OF_DOMAIN
    ) == (scores.loc[1, "ad_isolation_forest_decision"] < 0.0)


def test_isolation_forest_invalid_and_schema_mismatch_are_explicit(tmp_path):
    manifest = fit_isolation_forest_domain(
        feature_frame=pd.DataFrame({"desc_a": [0.0, 1.0], "desc_b": [5.0, 6.0]}),
        feature_columns=["desc_a", "desc_b"],
        output_dir=tmp_path / "ad",
        model_id="model",
        feature_space="rdkit_all",
        random_state=0,
    )
    modern_manifest = {
        "available": True,
        "primary_method": ISOLATION_FOREST_METHOD,
        "method": ISOLATION_FOREST_METHOD,
        "feature_space": "rdkit_all",
        "methods": {ISOLATION_FOREST_METHOD: manifest},
    }

    invalid = score_isolation_forest_domain(
        feature_frame=pd.DataFrame({"desc_a": [float("nan")], "desc_b": [5.5]}),
        applicability_domain=modern_manifest,
    )
    mismatch = score_isolation_forest_domain(
        feature_frame=pd.DataFrame({"desc_a": [0.5]}),
        applicability_domain=modern_manifest,
    )

    assert invalid["scores"].loc[0, "ad_isolation_forest_status"] == AD_INVALID_FEATURES
    assert mismatch["scores"].loc[0, "ad_status"] == AD_SCHEMA_MISMATCH


def test_modern_ad_aggregation_is_strict():
    bounding = pd.DataFrame(
        {
            "ad_status": [AD_IN_DOMAIN, AD_OUT_OF_DOMAIN, AD_IN_DOMAIN, AD_IN_DOMAIN],
            "ad_method": ["bounding_box"] * 4,
            "ad_bounding_box_status": [
                AD_IN_DOMAIN,
                AD_OUT_OF_DOMAIN,
                AD_IN_DOMAIN,
                AD_IN_DOMAIN,
            ],
            "ad_bounding_box_violation_count": [0, 1, 0, 0],
            "ad_bounding_box_violating_features": ["", "desc_a", "", ""],
            "ad_bounding_box_max_excess": [0.0, 1.0, 0.0, 0.0],
            "ad_feature_space": ["rdkit_all"] * 4,
        }
    )
    isolation = pd.DataFrame(
        {
            "ad_status": [AD_IN_DOMAIN, AD_IN_DOMAIN, AD_OUT_OF_DOMAIN, AD_INVALID_FEATURES],
            "ad_method": ["isolation_forest"] * 4,
            "ad_isolation_forest_status": [
                AD_IN_DOMAIN,
                AD_IN_DOMAIN,
                AD_OUT_OF_DOMAIN,
                AD_INVALID_FEATURES,
            ],
            "ad_isolation_forest_score": [-0.4, -0.4, -0.6, float("nan")],
            "ad_isolation_forest_decision": [0.1, 0.1, -0.1, float("nan")],
            "ad_isolation_forest_threshold": [0.0] * 4,
            "ad_feature_space": ["rdkit_all"] * 4,
        }
    )

    combined = _combine_modern_scores(
        {"bounding_box": bounding, "isolation_forest": isolation}
    )

    assert combined["ad_status"].tolist() == [
        AD_IN_DOMAIN,
        AD_OUT_OF_DOMAIN,
        AD_OUT_OF_DOMAIN,
        AD_INVALID_FEATURES,
    ]
    assert combined["ad_methods_out"].tolist()[:3] == [
        "",
        "bounding_box",
        "isolation_forest",
    ]


def test_prediction_ad_metrics_are_split_by_status(tmp_path):
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [1.0, 2.0, 3.0, 4.0],
            "y_pred": [1.0, 2.0, 10.0, 12.0],
        }
    ).to_csv(predictions_path, index=False)
    scores = pd.DataFrame(
        {
            "ad_status": [AD_IN_DOMAIN, AD_IN_DOMAIN, AD_OUT_OF_DOMAIN, AD_OUT_OF_DOMAIN],
            "ad_method": ["bounding_box"] * 4,
            "ad_violation_count": [0, 0, 1, 2],
            "ad_violating_features": ["", "", "desc_a", "desc_b"],
            "ad_max_excess": [0.0, 0.0, 0.3, 0.7],
            "ad_feature_space": ["rdkit_all"] * 4,
        }
    )

    metrics = attach_ad_scores_and_metrics_to_predictions(
        predictions_path=str(predictions_path),
        scores=scores,
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        target_column="pEC50",
    )

    assert metrics["metrics_all"]["n"] == 4
    assert metrics["metrics_in_domain"]["n"] == 2
    assert metrics["metrics_out_of_domain"]["n"] == 2
    assert metrics["coverage_in_domain"] == 0.5
    enriched = pd.read_csv(predictions_path)
    assert enriched["ad_status"].tolist() == [
        AD_IN_DOMAIN,
        AD_IN_DOMAIN,
        AD_OUT_OF_DOMAIN,
        AD_OUT_OF_DOMAIN,
    ]


def test_training_ad_syncs_canonical_prediction_artifacts(tmp_path):
    dataset_path = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "smiles": ["CCO", "CCC", "CCN", "CCCl"],
            "pEC50": [1.0, 2.0, 3.0, 4.0],
            "desc_a": [0.0, 1.0, 0.5, 3.0],
        }
    ).to_csv(dataset_path, index=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    splits_path = run_dir / "splits.json"
    splits_path.write_text(
        '[{"train": [0, 1], "validation": [2], "test": [3]}]\n'
    )

    primary_validation = run_dir / "primary_validation.csv"
    canonical_validation = run_dir / "canonical_validation.csv"
    primary_test = run_dir / "primary_test.csv"
    canonical_test = run_dir / "canonical_test.csv"
    for path, y_true, y_pred in (
        (primary_validation, [3.0], [2.8]),
        (canonical_validation, [3.0], [2.8]),
        (primary_test, [4.0], [3.5]),
        (canonical_test, [4.0], [3.5]),
    ):
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(path, index=False)

    summary = build_applicability_domain_for_training(
        train_csv=str(dataset_path),
        primary_run={
            "splits_path": str(splits_path),
            "validation_predictions_path": str(primary_validation),
            "test_predictions_path": str(primary_test),
        },
        primary_output_dir=run_dir,
        task=PredictionTaskSpec(
            task_type="regression",
            smiles_columns=["smiles"],
            target_columns=["pEC50"],
        ),
        feature_columns=["desc_a"],
        prediction_artifact_paths={
            "validation": str(canonical_validation),
            "test": str(canonical_test),
        },
    )

    assert summary["split_score_summaries"]["test"][
        "ad_enriched_canonical_predictions_path"
    ] == str(canonical_test)
    assert "ad_status" in pd.read_csv(canonical_validation).columns
    assert "ad_status" in pd.read_csv(canonical_test).columns
    assert pd.read_csv(canonical_test)["ad_status"].tolist() == [AD_OUT_OF_DOMAIN]
    assert (run_dir / "applicability_domain" / "isolation_forest" / "model.joblib").exists()
