#!/usr/bin/env python3
"""Verify Qsaria benchmark predictions and regenerate the paper table and figure."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

METRICS = {"mae": mean_absolute_error, "auroc": roc_auc_score}
DISPLAY_NAMES = {"solubility_aqsoldb": "AqSolDB", "bbb_martins": "BBB_Martins"}
MODELS = {"lightgbm": "Qsaria LightGBM", "dummy": "Naïve", "random_forest": "Random Forest"}


def dataset_rows(directory: Path) -> tuple[list[dict], list[dict], list[dict]]:
    summary = json.loads((directory / "summary.json").read_text())
    predictions = pd.read_csv(directory / "predictions.csv")
    if len(predictions) != summary["evaluated_rows"] or predictions["row_id"].duplicated().any():
        raise ValueError(f"{directory.name}: prediction rows do not match the summary.")
    metric = summary["metric"]
    common = {"dataset": DISPLAY_NAMES.get(directory.name, directory.name), "metric": metric}
    scores = []
    differences = []
    for cohort_name, result in summary["cohorts"].items():
        cohort = (
            predictions.loc[~predictions["identity_seen_in_fit"]]
            if cohort_name == "identity_unseen"
            else predictions
        )
        if len(cohort) != result["n"]:
            raise ValueError(f"{directory.name}: {cohort_name} count does not match.")
        cohort_fields = {**common, "cohort": cohort_name, "n": len(cohort)}
        by_seed = result["scores_by_model_seed"]
        for model, label in MODELS.items():
            values = []
            for seed, benchmark_values in by_seed.items():
                value = METRICS[metric](cohort["Y"], cohort[f"{model}_{seed}"])
                if not math.isclose(value, benchmark_values[model], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"{directory.name}: score mismatch for {model}, seed {seed}.")
                values.append(float(value))
            scores.append(
                {
                    **cohort_fields,
                    "model": label,
                    "mean": float(np.mean(values)),
                    "sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                    "seeds": len(values),
                }
            )
        interval = result["lightgbm_minus_random_forest_first_seed"]
        first_seed = next(iter(by_seed))
        expected_delta = by_seed[first_seed]["lightgbm"] - by_seed[first_seed]["random_forest"]
        if not math.isclose(expected_delta, interval["delta"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{directory.name}: paired difference does not match.")
        differences.append({**cohort_fields, "seed": first_seed, **interval})
    domains = []
    for name, result in summary["bounding_box_by_status"].items():
        group = predictions.loc[predictions["bounding_box_status"] == name]
        if len(group) != result["n"]:
            raise ValueError(f"{directory.name}: {name} domain count does not match.")
        reported = result["lightgbm_first_seed"]
        if reported is not None:
            seed = differences[0]["seed"]
            value = METRICS[metric](group["Y"], group[f"lightgbm_{seed}"])
            if not math.isclose(reported, value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"{directory.name}: {name} domain score does not match.")
        domains.append({**common, "status": name, "n": len(group), "score": reported})
    return scores, differences, domains


def plot_differences(differences: pd.DataFrame, output: Path) -> None:
    grouped = list(differences.groupby("dataset", sort=False))
    figure, axes = plt.subplots(1, len(grouped), figsize=(4.9 * len(grouped), 2.8), squeeze=False)
    for axis, (dataset, rows) in zip(axes.flat, grouped, strict=True):
        labels = [
            "All scored rows" if cohort == "all_scored" else "Unseen identities"
            for cohort in rows["cohort"]
        ]
        y = np.arange(len(rows))
        axis.axvline(0, color="#7B8794", linewidth=1, linestyle="--")
        axis.hlines(y, rows["lower_95"], rows["upper_95"], linewidth=2, color="#225F85")
        axis.scatter(rows["delta"], y, color="#225F85", zorder=3)
        axis.set(yticks=y, yticklabels=labels, title=dataset, ylim=(-0.5, len(rows) - 0.5))
        axis.invert_yaxis()
        metric = rows.iloc[0]["metric"].upper()
        axis.set_xlabel(f"{metric} difference (LightGBM − Random Forest)")
    figure.tight_layout()
    for extension in ("pdf", "png"):
        figure.savefig(output / f"model_differences.{extension}", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    scores, differences, domains = [], [], []
    for dataset in manifest["protocol"]["datasets"]:
        dataset_scores, dataset_differences, dataset_domains = dataset_rows(args.run / dataset)
        scores.extend(dataset_scores)
        differences.extend(dataset_differences)
        domains.extend(dataset_domains)

    args.output.mkdir(parents=True)
    pd.DataFrame(scores).to_csv(args.output / "scores.csv", index=False)
    difference_table = pd.DataFrame(differences)
    difference_table.to_csv(args.output / "paired_differences.csv", index=False)
    pd.DataFrame(domains).to_csv(args.output / "applicability_domain.csv", index=False)
    plot_differences(difference_table, args.output)
    print(f"Verified {len(scores)} model scores and {len(differences)} paired differences.")


if __name__ == "__main__":
    main()
