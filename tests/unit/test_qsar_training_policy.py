from __future__ import annotations

import pytest

from cs_copilot.tools.prediction.qsar_training_policy import assess_protocol_results


def test_assess_protocol_results_uses_classification_primary_metric():
    assessment = assess_protocol_results(
        [
            {
                "strategy_family": "random",
                "strategy_label": "random_seed_1",
                "validation_protocol": "standard_qsar",
                "metrics": {
                    "test": {
                        "accuracy": 0.84,
                        "balanced_accuracy": 0.82,
                        "f1_macro": 0.81,
                        "roc_auc": 0.88,
                        "n": 50,
                    }
                },
            },
            {
                "strategy_family": "scaffold",
                "strategy_label": "scaffold",
                "validation_protocol": "standard_qsar",
                "metrics": {
                    "test": {
                        "accuracy": 0.70,
                        "balanced_accuracy": 0.68,
                        "f1_macro": 0.67,
                        "roc_auc": 0.74,
                        "n": 50,
                    }
                },
            },
        ]
    )

    assert assessment["hardest_split"] == "scaffold"
    assert assessment["governance"]["primary_metric"] == "balanced_accuracy"
    assert assessment["delta_vs_random"]["scaffold"]["balanced_accuracy"] == pytest.approx(-0.14)
    assert assessment["governance"]["hardest_split_metrics"]["balanced_accuracy"] == pytest.approx(
        0.68
    )
    assert assessment["governance"]["passes_hardest_split_gate"] is False
    assert assessment["robustness_warning"]
