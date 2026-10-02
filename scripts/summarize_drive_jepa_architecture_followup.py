"""Publish bounded architecture results, paired seeds and recording uncertainty."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def paired_recording_comparison(proposed_by_seed, reference_by_seed):
    seed_differences, clustered = [], {}
    for seed, proposed in proposed_by_seed.items():
        reference = {row["token"]: row for row in reference_by_seed[seed]}
        if {row["token"] for row in proposed} != set(reference):
            raise ValueError("Paired comparison requires identical evaluation windows")
        scenes = {}
        for row in proposed:
            key = (row["recording"], row["scene_token"])
            scenes.setdefault(key, []).append(
                row["xy_ade_m"] - reference[row["token"]]["xy_ade_m"]
            )
        scene_means = {key: float(np.mean(values)) for key, values in scenes.items()}
        seed_differences.append(
            {
                "seed": seed,
                "proposed_minus_reference_ade_m": float(
                    np.mean(list(scene_means.values()))
                ),
            }
        )
        for key, value in scene_means.items():
            clustered.setdefault(key, []).append(value)
    groups = sorted({key[0] for key in clustered})
    values = [
        np.array(
            [
                np.mean(differences)
                for key, differences in clustered.items()
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
                    values[index]
                    for index in generator.integers(0, len(groups), len(groups))
                ]
            ).mean()
        )
        for _ in range(2000)
    ]
    return {
        "paired_seed_differences": seed_differences,
        "mean_difference_m": float(
            np.mean([row["proposed_minus_reference_ade_m"] for row in seed_differences])
        ),
        "recording_differences_m": {
            group: float(array.mean()) for group, array in zip(groups, values)
        },
        "recording_cluster_bootstrap_95_ci_m": np.quantile(
            bootstrap, [0.025, 0.975]
        ).tolist(),
        "interval_scope": "recording_resampling_of_seed_averaged_scene_differences_not_seed_uncertainty",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--share-directory", type=Path, required=True)
    parser.add_argument("--append-postflight", action="store_true")
    arguments = parser.parse_args()
    run, share = arguments.run_directory.resolve(), arguments.share_directory.resolve()
    if arguments.append_postflight:
        if not (share / "summary.json").exists():
            raise RuntimeError("Export completed training results first")
        for filename in (
            "trained_checkpoint_postflight.json",
            "postflight_equivalence_failure.json",
        ):
            source, destination = run / filename, share / filename
            if source.exists():
                if destination.exists():
                    raise RuntimeError("Do not overwrite prior postflight export")
                write_json(destination, json.loads(source.read_text()))
        return
    if share.exists():
        raise RuntimeError("Fresh shared output required")
    completion = json.loads((run / "completion.json").read_text())
    execution = json.loads((run / "execution_specification.json").read_text())
    specification = execution["specification"]
    reports = [
        json.loads((run / f"{condition}_seed{seed}" / "results.json").read_text())
        for condition in specification["conditions"]
        for seed in specification["seeds"]
    ]
    workspace = Path(__file__).resolve().parents[1]
    original = json.loads(
        (workspace / specification["reused_cache"])
        .with_name("original_baseline.json")
        .read_text()
    )
    aggregate, final_windows = {}, {}
    for condition in specification["conditions"]:
        matched = [report for report in reports if report["condition"] == condition]
        aggregate[condition] = {}
        final_windows[condition] = {
            report["seed"]: report["evaluations"]["200"]["development"]["windows"]
            for report in matched
        }
        for split in ("train", "development"):
            summaries = [
                report["evaluations"]["200"][split]["summary"] for report in matched
            ]
            aggregate[condition][split] = {
                metric: {
                    "seed_values": [row[metric] for row in summaries],
                    "mean": float(np.mean([row[metric] for row in summaries])),
                    "sample_std": float(
                        np.std([row[metric] for row in summaries], ddof=1)
                    ),
                }
                for metric in (
                    "scene_macro_xy_ade_m",
                    "official_il_loss_whole_split",
                    "future_mse",
                    "persistence_mse",
                )
            }
        aggregate[condition]["trainable_parameters"] = matched[0][
            "trainable_parameters"
        ]
        aggregate[condition]["lora_parameters"] = matched[0]["lora_parameters"]
        aggregate[condition]["wall_seconds_all_seeds"] = sum(
            report["wall_seconds"] for report in matched
        )
        aggregate[condition]["max_peak_allocated_gib"] = max(
            report["peak_allocated_gib"] for report in matched
        )
        aggregate[condition]["command_development_ade"] = {
            command: {
                "count": matched[0]["evaluations"]["200"]["development"]["summary"][
                    "command_window_ade"
                ][command]["count"],
                "seed_values": [
                    report["evaluations"]["200"]["development"]["summary"][
                        "command_window_ade"
                    ][command]["mean"]
                    for report in matched
                ],
            }
            for command in matched[0]["evaluations"]["200"]["development"]["summary"][
                "command_window_ade"
            ]
        }
    final_windows["original_no_branch"] = {
        seed: original["development"]["windows"] for seed in specification["seeds"]
    }
    contrasts = [
        (condition, "original_no_branch") for condition in specification["conditions"]
    ]
    contrasts += [
        ("contextual_residual", "mlp_recipe_control"),
        ("ego_query_residual", "contextual_residual"),
        ("ego_query_residual_lora", "ego_query_residual"),
        ("ego_query_residual_lora", "lora_current_target_control"),
    ]
    paired = {
        f"{proposed}_minus_{reference}": paired_recording_comparison(
            final_windows[proposed], final_windows[reference]
        )
        for proposed, reference in contrasts
    }
    share.mkdir(parents=True)
    rows = []
    for report in reports:
        for split in ("train", "development"):
            for window in report["evaluations"]["200"][split]["windows"]:
                rows.append(
                    {
                        "condition": report["condition"],
                        "seed": report["seed"],
                        "split": split,
                        **{
                            key: json.dumps(value) if isinstance(value, list) else value
                            for key, value in window.items()
                        },
                    }
                )
    with (share / "final_window_metrics.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        share / "summary.json",
        {
            "execution": execution,
            "completion": completion,
            "original": {split: original[split]["summary"] for split in original},
            "aggregates": aggregate,
            "paired_development_comparisons": paired,
            "raw_run_directory": str(run),
            "interpretation": "repeated_small_development_split_exploration_not_independent_generalization_or_safety_evaluation",
        },
    )
    write_json(
        share / "training_curves.json",
        [
            {
                "condition": report["condition"],
                "seed": report["seed"],
                "curve": report["curve"],
                "evaluations": {
                    step: {split: value["summary"] for split, value in splits.items()}
                    for step, splits in report["evaluations"].items()
                },
            }
            for report in reports
        ],
    )
    write_json(
        share / "architecture_contracts.json",
        [
            {
                key: value
                for key, value in report.items()
                if key not in ("curve", "evaluations", "final_parameter_delta_norms")
            }
            for report in reports
        ],
    )
    write_json(
        share / "prefix_cache_verification.json",
        json.loads((run / "prefix_cache_index.json").read_text()),
    )
    print(
        json.dumps(
            {"aggregate": aggregate, "paired": paired, "completion": completion},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
