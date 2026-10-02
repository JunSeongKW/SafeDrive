"""Export small comparison, paired seeds, and recording-cluster uncertainty."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--share-directory", type=Path, required=True)
    arguments = parser.parse_args()
    run = arguments.run_directory
    share = arguments.share_directory
    if share.exists():
        raise RuntimeError("Preserve earlier exported results")
    share.mkdir(parents=True)
    summary = json.loads((run / "comparison_results.json").read_text())
    reports = [
        json.loads(
            (run / f"{row['condition']}_seed{row['seed']}" / "results.json").read_text()
        )
        for row in summary["conditions"]
    ]
    metric = "scene_macro_xy_ade_m"
    aggregated = {}
    for condition in summary["specification"]["conditions"]:
        matched = [
            row for row in summary["conditions"] if row["condition"] == condition
        ]
        aggregated[condition] = {}
        for split in ("train", "development"):
            values = [row["final"][split][metric] for row in matched]
            aggregated[condition][split] = {
                "seed_values": values,
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)),
            }
    final_windows = {}
    rows = []
    selection_descriptions = []
    for report in reports:
        seed, condition = report["seed"], report["condition"]
        final_windows[(condition, seed)] = report["evaluations"]["200"]["development"][
            "windows"
        ]
        development_windows = final_windows[(condition, seed)]
        pair_distances = []
        for window in development_windows:
            identifiers = np.array(window["selected_patch_ids"])
            coordinates = np.column_stack((identifiers % 32, identifiers // 32))
            pair_distances.extend(
                float(np.linalg.norm(coordinates[left] - coordinates[right]))
                for left in range(4)
                for right in range(left + 1, 4)
            )
        selection_descriptions.append(
            {
                "condition": condition,
                "seed": seed,
                "unique_development_selected_sets": len(
                    {
                        tuple(sorted(row["selected_patch_ids"]))
                        for row in development_windows
                    }
                ),
                "mean_selected_pair_distance_grid_cells": float(
                    np.mean(pair_distances)
                ),
                "interpretation": "descriptive_diversity_not_relevance_ground_truth",
            }
        )
        for split in ("train", "development"):
            for window in report["evaluations"]["200"][split]["windows"]:
                rows.append(
                    {
                        "condition": condition,
                        "seed": seed,
                        "split": split,
                        **window,
                        "selected_patch_ids": json.dumps(window["selected_patch_ids"]),
                    }
                )
    with (share / "final_window_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = {}
    for comparison in (
        "fixed_lattice",
        "seeded_random",
        "planning_conditioned_no_auxiliary",
    ):
        per_seed, recording_scene_pairs = [], {}
        for seed in summary["specification"]["seeds"]:
            proposed = final_windows[("planning_conditioned", seed)]
            reference = {row["token"]: row for row in final_windows[(comparison, seed)]}
            scene_differences = {}
            for row in proposed:
                key = (row["recording"], row["scene_token"])
                scene_differences.setdefault(key, []).append(
                    row["xy_ade_m"] - reference[row["token"]]["xy_ade_m"]
                )
            scene_means = {
                key: float(np.mean(values)) for key, values in scene_differences.items()
            }
            per_seed.append(
                {
                    "seed": seed,
                    "learned_minus_comparison_scene_macro_ade_m": float(
                        np.mean(list(scene_means.values()))
                    ),
                }
            )
            for key, value in scene_means.items():
                recording_scene_pairs.setdefault(key, []).append(value)
        groups = sorted({key[0] for key in recording_scene_pairs})
        group_values = [
            np.array(
                [
                    np.mean(values)
                    for key, values in recording_scene_pairs.items()
                    if key[0] == group
                ]
            )
            for group in groups
        ]
        generator = np.random.default_rng(20261002)
        bootstrap = [
            float(
                np.concatenate(
                    [
                        group_values[index]
                        for index in generator.integers(0, len(groups), len(groups))
                    ]
                ).mean()
            )
            for _ in range(2000)
        ]
        paired[comparison] = {
            "paired_seed_differences": per_seed,
            "cluster_bootstrap_95_ci_m": np.quantile(
                bootstrap, [0.025, 0.975]
            ).tolist(),
            "recording_count": len(groups),
            "recording_differences_m": {
                group: float(values.mean())
                for group, values in zip(groups, group_values)
            },
            "interval_scope": "recording_resampling_on_seed_averaged_scene_differences_not_seed_uncertainty",
        }
    summary["seed_aggregates"] = aggregated
    summary["paired_development_comparisons"] = paired
    summary["selection_descriptions"] = selection_descriptions
    summary["raw_run_directory"] = str(run.resolve())
    summary["interpretation"] = (
        "Small 200-update development study, no official safety evaluation or final selection claim"
    )
    for filename, value in (
        ("summary.json", summary),
        (
            "training_curves.json",
            [
                {
                    "condition": report["condition"],
                    "seed": report["seed"],
                    "curve": report["curve"],
                    "evaluations": {
                        step: {
                            split: evaluation["summary"]
                            for split, evaluation in splits.items()
                        }
                        for step, splits in report["evaluations"].items()
                    },
                }
                for report in reports
            ],
        ),
        ("window_manifest.json", json.loads((run / "cache_index.json").read_text())),
        (
            "cache_equivalence.json",
            json.loads((run / "cached_path_equivalence.json").read_text()),
        ),
    ):
        (share / filename).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print(
        json.dumps(
            {
                "baseline": summary["baseline"],
                "aggregates": aggregated,
                "paired": paired,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
