from __future__ import annotations

from cs_copilot.tools.prediction.qsar_validation_strategy import resolve_validation_strategy


def test_standard_qsar_strategy_delegates_to_existing_protocol():
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
    ]


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


def test_random_kfold_marks_runs_as_payload_required():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "cross_validation",
            "split_family": "random",
            "n_folds": 5,
            "seed": 17,
        },
        training_profile="heavy_validation",
    )

    assert policy["protocol"] == "random_5fold_cv"
    assert policy["aggregation"] == "mean_std"
    assert [run["fold_index"] for run in policy["split_runs"]] == [1, 2, 3, 4, 5]
    assert all(run["requires_split_payload"] for run in policy["split_runs"])
    assert len({run["split_seed"] for run in policy["split_runs"]}) == 1


def test_scaffold_kfold_uses_scaffold_family():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "cross_validation",
            "split_family": "scaffold",
            "n_folds": 3,
        },
        training_profile="heavy_validation",
        base_seed=19,
    )

    assert policy["protocol"] == "scaffold_3fold_cv"
    assert all(run["split_family"] == "scaffold" for run in policy["split_runs"])
    assert all(run["backend_split_type"] == "scaffold_balanced" for run in policy["split_runs"])


def test_cluster_kfold_uses_kmeans_backend_split_type():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "cross_validation",
            "split_family": "cluster",
            "n_folds": 4,
        },
        training_profile="heavy_validation",
        base_seed=23,
    )

    assert policy["protocol"] == "cluster_4fold_cv"
    assert all(run["split_family"] == "cluster" for run in policy["split_runs"])
    assert all(run["backend_split_type"] == "kmeans" for run in policy["split_runs"])


def test_nested_cv_exposes_outer_runs_and_inner_strategy():
    policy = resolve_validation_strategy(
        requested_protocol="standard_qsar",
        validation_strategy={
            "type": "nested_cross_validation",
            "outer": {"split_family": "scaffold", "n_folds": 5},
            "inner": {"split_family": "random", "n_folds": 3},
            "selection_metric": "rmse",
            "final_refit": True,
        },
        training_profile="heavy_validation",
        base_seed=29,
    )

    assert policy["protocol"] == "nested_scaffold_5x3_cv"
    assert policy["final_refit"] is True
    assert len(policy["split_runs"]) == 5
    assert policy["split_runs"][0]["inner_strategy"] == {
        "type": "cross_validation",
        "split_family": "random",
        "n_folds": 3,
        "seed": None,
        "selection_metric": "rmse",
    }
