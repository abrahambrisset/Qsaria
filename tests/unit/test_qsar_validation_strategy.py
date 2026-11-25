from __future__ import annotations

import pandas as pd
import pytest

from cs_copilot.tools.prediction.backend import InvalidPredictionInputError, PredictionTaskSpec
from cs_copilot.tools.prediction.qsar_splitters import (
    build_qsar_split_payload,
    build_repeated_kfold_split_payloads,
)
from cs_copilot.tools.prediction.qsar_training_policy import assess_protocol_results
from cs_copilot.tools.prediction.qsar_validation_strategy import resolve_validation_strategy


def test_standard_qsar_resolves_three_reproducible_holdout_families():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy=None,
        training_profile="heavy_validation",
        base_seed=123,
    )

    assert policy["protocol"] == "standard_qsar"
    assert [run["backend_split_type"] for run in policy["split_runs"]] == [
        "random",
        "scaffold_balanced",
        "kmeans",
    ]
    assert [run["split_sizes"] for run in policy["split_runs"]] == [[0.8, 0.1, 0.1]] * 3
    assert [run["primary"] for run in policy["split_runs"]] == [True, False, False]
    assert policy["seed_policy"]["random_split_seeds"] == [123]
    replay = resolve_validation_strategy(
        requested_protocol=None,
        validation_strategy=None,
        training_profile="local_light",
        base_seed=123,
    )
    assert replay["split_runs"] == policy["split_runs"]


def test_explicit_single_random_holdout_resolves_one_run():
    policy = resolve_validation_strategy(
        requested_protocol="single_random_holdout",
        validation_strategy=None,
        training_profile="local_light",
        base_seed=123,
    )

    assert policy["protocol"] == "single_random_holdout"
    assert policy["split_runs"] == [
        {
            "label": "random_seed_123",
            "backend_split_type": "random",
            "primary": True,
            "seed": 123,
            "split_sizes": [0.8, 0.1, 0.1],
        }
    ]


def test_standard_qsar_can_meet_the_validation_gates():
    assessment = assess_protocol_results(
        [
            {
                "strategy_family": family,
                "validation_protocol": "standard_qsar",
                "metrics": {"test": {"r2": r2, "rmse": rmse, "n": 30}},
            }
            for family, r2, rmse in (
                ("random", 0.86, 0.33),
                ("scaffold", 0.83, 0.36),
                ("cluster_kmeans", 0.79, 0.40),
            )
        ]
    )

    assert assessment["hardest_split"] == "cluster_kmeans"
    assert assessment["delta_vs_random"]["cluster_kmeans"]["r2"] == pytest.approx(-0.07)
    assert assessment["governance"]["recommended_status"] == "validated"


def test_custom_holdout_uses_requested_split_sizes():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "holdout",
            "split_family": "random",
            "split_sizes": [0.6, 0.2, 0.2],
            "seed": 7,
        },
        training_profile="heavy_validation",
    )

    assert policy["protocol"] == "random_holdout"
    assert policy["validation_strategy_type"] == "holdout"
    assert policy["split_runs"][0]["split_sizes"] == [0.6, 0.2, 0.2]
    assert policy["split_runs"][0]["backend_split_type"] == "random"


def test_custom_holdout_rejects_backend_style_ratio_payload():
    with pytest.raises(ValueError, match="random_split"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "holdout",
                "split_family": "random",
                "random_split": {"train_ratio": 0.6, "validation_ratio": 0.2, "test_ratio": 0.2},
            },
            training_profile="heavy_validation",
        )


def test_custom_holdout_accepts_train_test_split_sizes_without_validation():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "holdout",
            "split_family": "random",
            "split_sizes": [0.8, 0.2],
        },
        training_profile="heavy_validation",
    )

    assert policy["split_runs"][0]["split_sizes"] == [0.8, 0.2]
    assert policy["validation_strategy"]["split_sizes"] == [0.8, 0.2]


def test_custom_holdout_rejects_ratio_aliases():
    with pytest.raises(ValueError, match="test_ratio"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "holdout",
                "split_family": "random",
                "train_ratio": 0.8,
                "validation_ratio": 0.0,
                "test_ratio": 0.2,
            },
            training_profile="heavy_validation",
        )


def test_custom_validation_rejects_removed_selection_metric():
    with pytest.raises(ValueError, match="selection_metric"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "holdout",
                "split_family": "random",
                "selection_metric": "rmse",
            },
            training_profile="heavy_validation",
        )


def test_resolved_validation_policy_has_no_selection_metric():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={"type": "holdout", "split_family": "random"},
        training_profile="heavy_validation",
    )
    assert "selection_metric" not in policy
    assert "selection_metric" not in policy["validation_strategy"]


def test_custom_holdout_rejects_agent_aliases():
    with pytest.raises(ValueError, match="Expected one of"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={"method": "holdout", "split_type": "scaffold"},
            training_profile="heavy_validation",
        )


@pytest.mark.parametrize("alias", ["random_holdout", "scaffold_holdout", "cluster_holdout"])
def test_custom_holdout_rejects_named_holdout_aliases(alias):
    with pytest.raises(ValueError, match=alias):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={"type": "holdout", alias: {"train_ratio": 0.6}},
            training_profile="heavy_validation",
        )


def test_custom_holdout_rejects_strategy_alias():
    with pytest.raises(ValueError, match="Expected one of"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={"strategy": "holdout", "split_sizes": [0.8, 0.1, 0.1]},
            training_profile="heavy_validation",
        )


def test_repeated_holdout_generates_replayable_runs():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "repeated_holdout",
            "split_family": "scaffold",
            "n_repeats": 3,
            "seed": 11,
        },
        training_profile="heavy_validation",
    )

    assert policy["protocol"] == "repeated_scaffold_holdout"
    assert len(policy["split_runs"]) == 3
    assert all(run["backend_split_type"] == "scaffold_balanced" for run in policy["split_runs"])
    assert policy["seed_policy"]["split_runs"] == policy["split_runs"]


def test_repeated_holdout_without_seed_generates_distinct_random_splits():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "repeated_holdout",
            "split_family": "random",
            "n_repeats": 3,
        },
        training_profile="heavy_validation",
    )

    seeds = [run["seed"] for run in policy["split_runs"]]
    assert len(seeds) == 3
    assert len(set(seeds)) == 3
    assert policy["seed_policy"]["mode"] == "generated_per_run"


def test_repeated_holdout_rejects_holdout_ratio_alias():
    with pytest.raises(ValueError, match="scaffold_holdout"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "repeated_holdout",
                "scaffold_holdout": {"train_ratio": 0.7},
                "n_repeats": 2,
            },
            training_profile="heavy_validation",
        )


def test_cross_validation_generates_requested_fold_runs():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "cross_validation",
            "split_family": "random",
            "n_folds": 5,
            "n_repeats": 1,
            "seed": 17,
        },
        training_profile="heavy_validation",
    )

    assert policy["protocol"] == "cross_validation"
    assert policy["validation_strategy_type"] == "cross_validation"
    assert len(policy["split_runs"]) == 5
    assert [run["label"] for run in policy["split_runs"]] == [
        "cv_repeat_1_fold_1",
        "cv_repeat_1_fold_2",
        "cv_repeat_1_fold_3",
        "cv_repeat_1_fold_4",
        "cv_repeat_1_fold_5",
    ]
    assert all(run["backend_split_type"] == "cross_validation" for run in policy["split_runs"])
    assert all(run["split_family"] == "random" for run in policy["split_runs"])
    assert policy["final_refit"] is True


def test_cross_validation_rejects_fold_aliases():
    with pytest.raises(ValueError, match="Expected one of"):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={"method": "cv", "folds": 5},
            training_profile="heavy_validation",
        )


def test_cross_validation_payload_uses_validation_not_test_for_inner_fold():
    payloads = build_repeated_kfold_split_payloads(
        df=pd.DataFrame({"x": range(20)}),
        n_splits=5,
        n_repeats=1,
        random_state=7,
    )

    first = payloads["cv_repeat_1_fold_1"][0]
    assert first["train"]
    assert first["val"]
    assert "test" not in first
    assert set(first["train"]).isdisjoint(first["val"])


def test_cross_validation_outer_test_is_fixed_and_absent_from_inner_train_validation():
    payloads = build_repeated_kfold_split_payloads(
        df=pd.DataFrame({"x": range(30)}),
        n_splits=3,
        n_repeats=1,
        random_state=11,
        outer_test_size=0.2,
    )

    folds = [payload[0] for payload in payloads.values()]
    outer_test = folds[0]["test"]
    assert all(fold["test"] == outer_test for fold in folds)
    assert all(set(outer_test).isdisjoint(fold["train"]) for fold in folds)
    assert all(set(outer_test).isdisjoint(fold["val"]) for fold in folds)


def test_random_classification_holdout_stratifies_imbalanced_multiclass():
    df = pd.DataFrame({"activity": ["inactive"] * 90 + ["active"] * 5 + ["partial"] * 5})
    task = PredictionTaskSpec(task_type="multiclass_classification", target_columns=["activity"])
    kwargs = {
        "df": df,
        "split_type": "random",
        "split_sizes": [0.8, 0.1, 0.1],
        "random_state": 7,
        "task": task,
    }

    split = build_qsar_split_payload(**kwargs)[0]

    assert split == build_qsar_split_payload(**kwargs)[0]
    assert split["metadata"]["stratified"] is True
    assert df.iloc[split["train"]]["activity"].value_counts().to_dict() == {
        "inactive": 74,
        "active": 3,
        "partial": 3,
    }
    for name in ("val", "test"):
        assert df.iloc[split[name]]["activity"].value_counts().to_dict() == {
            "inactive": 8,
            "active": 1,
            "partial": 1,
        }
    assert sorted(split["train"] + split["val"] + split["test"]) == list(range(100))


def test_random_classification_holdout_reserves_rare_classes_in_training():
    df = pd.DataFrame({"activity": [0] * 17 + [1, 1, 2]})
    task = PredictionTaskSpec(task_type="multiclass_classification", target_columns=["activity"])

    split = build_qsar_split_payload(
        df=df, split_type="random", split_sizes=[0.8, 0.1, 0.1], random_state=3, task=task
    )[0]

    assert set(df.iloc[split["train"]]["activity"]) == {0, 1, 2}
    assert set(df.iloc[split["test"]]["activity"]) == {0, 1}
    assert set(df.iloc[split["val"]]["activity"]) == {0}


def test_impossible_random_classification_training_size_is_a_clear_error():
    df = pd.DataFrame({"activity": [0] * 17 + [1, 1, 2]})
    task = PredictionTaskSpec(task_type="multiclass_classification", target_columns=["activity"])

    with pytest.raises(InvalidPredictionInputError, match="training.*2.*3 classes"):
        build_qsar_split_payload(
            df=df, split_type="random", split_sizes=[0.1, 0.9], random_state=3, task=task
        )


def test_small_classification_holdout_uses_a_feasible_partition():
    df = pd.DataFrame({"activity": [0, 0, 0, 0, 1, 1]})
    task = PredictionTaskSpec(task_type="classification", target_columns=["activity"])

    split = build_qsar_split_payload(
        df=df, split_type="random", split_sizes=[0.8, 0.2], random_state=3, task=task
    )[0]

    assert len(split["train"]) == 5
    assert set(df.iloc[split["train"]]["activity"]) == {0, 1}
    assert len(split["test"]) == 1


def test_repeated_stratified_cv_retains_rare_training_classes_after_outer_test():
    df = pd.DataFrame({"activity": [0] * 30 + [1] * 8 + [2] * 2})
    task = PredictionTaskSpec(task_type="multiclass_classification", target_columns=["activity"])
    kwargs = {
        "df": df,
        "n_splits": 5,
        "n_repeats": 2,
        "random_state": 7,
        "outer_test_size": 0.2,
        "task": task,
    }

    folds = build_repeated_kfold_split_payloads(**kwargs)

    assert folds == build_repeated_kfold_split_payloads(**kwargs)
    outer_test = next(iter(folds.values()))[0]["test"]
    assert set(df.iloc[outer_test]["activity"]) == {0, 1}
    development = set(range(40)) - set(outer_test)
    for repeat in (1, 2):
        validation_indices = []
        for fold in range(1, 6):
            split = folds[f"cv_repeat_{repeat}_fold_{fold}"][0]
            assert split["metadata"]["stratified"] is True
            assert split["test"] == outer_test
            assert set(df.iloc[split["train"]]["activity"]) == {0, 1, 2}
            assert set(split["train"]).isdisjoint(split["val"])
            validation_indices.extend(split["val"])
        assert sorted(validation_indices) == sorted(development)


def test_stratified_cv_can_have_more_folds_than_examples_per_class():
    df = pd.DataFrame({"activity": [0, 1, 2, 0, 1, 2]})
    task = PredictionTaskSpec(task_type="multiclass_classification", target_columns=["activity"])

    folds = build_repeated_kfold_split_payloads(
        df=df, n_splits=4, n_repeats=1, random_state=11, task=task
    )

    assert sorted(len(split[0]["val"]) for split in folds.values()) == [1, 1, 2, 2]
    assert all(set(df.iloc[split[0]["train"]]["activity"]) == {0, 1, 2} for split in folds.values())


def test_stratified_cv_rejects_singleton_class_with_actionable_error():
    df = pd.DataFrame({"activity": [0] * 19 + [1]})
    task = PredictionTaskSpec(task_type="classification", target_columns=["activity"])

    with pytest.raises(InvalidPredictionInputError, match="at least 2.*class.*activity"):
        build_repeated_kfold_split_payloads(
            df=df, n_splits=5, n_repeats=1, random_state=11, task=task
        )


@pytest.mark.parametrize("alias", ["test_size", "test_fold", "test_fraction"])
def test_cross_validation_rejects_ambiguous_outer_test_aliases(alias):
    with pytest.raises(ValueError, match=alias):
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "cross_validation",
                "split_family": "random",
                "n_folds": 3,
                alias: 0.1,
            },
            training_profile="heavy_validation",
        )


def test_full_train_strategy_creates_single_final_refit_run():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={"type": "full_train", "seed": 99},
        training_profile="heavy_validation",
    )

    assert policy["protocol"] == "full_train"
    assert policy["validation_strategy_type"] == "full_train"
    assert policy["final_refit"] is True
    assert policy["aggregation"] == "none"
    assert policy["split_runs"] == [
        {
            "label": "full_train",
            "backend_split_type": "final_refit",
            "seed": 99,
            "primary": True,
            "split_family": "full_train",
            "split_sizes": [1.0],
        }
    ]
    assert policy["validation_strategy"]["split_sizes"] == [1.0]


def test_holdout_one_zero_split_sizes_points_to_full_train():
    try:
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={
                "type": "holdout",
                "split_family": "random",
                "split_sizes": [1.0, 0.0],
            },
            training_profile="heavy_validation",
        )
    except ValueError as exc:
        assert "full_train" in str(exc)
    else:
        raise AssertionError("[1.0, 0.0] should be rejected in favor of full_train")


def test_unknown_validation_strategy_is_rejected():
    try:
        resolve_validation_strategy(
            requested_protocol="standard_qsar",
            validation_strategy={"type": "unsupported_strategy"},
            training_profile="heavy_validation",
        )
    except ValueError as exc:
        assert "holdout, repeated_holdout, cross_validation, full_train" in str(exc)
    else:
        raise AssertionError("unsupported validation strategy should be rejected")
