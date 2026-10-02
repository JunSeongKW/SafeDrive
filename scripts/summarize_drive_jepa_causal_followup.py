"""Export complete controlled runs, window evidence and separate uncertainties."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from summarize_drive_jepa_architecture_followup import paired_recording_comparison


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--share-directory", type=Path, required=True)
    arguments = parser.parse_args()
    run, share = arguments.run_directory.resolve(), arguments.share_directory.resolve()
    completion = json.loads((run / "completion.json").read_text())
    if not completion["complete"]:
        raise RuntimeError("Incomplete run cannot be exported as complete")
    specification = json.loads((run / "specification.json").read_text())
    share.mkdir(parents=True, exist_ok=False)
    reports = [
        json.loads((run / f"{condition}_seed{seed}" / "results.json").read_text())
        for condition in specification["conditions"]
        for seed in specification["seeds"]
    ]
    final_update = str(specification["joint_updates"])
    aggregate, windows_by_condition = {}, {}
    for condition in specification["conditions"]:
        matched = [report for report in reports if report["condition"] == condition]
        windows_by_condition[condition] = {
            report["seed"]: report["evaluations"][final_update]["development"][
                "windows"
            ]
            for report in matched
        }
        summary = {}
        for split in ("train", "development"):
            measurements = [
                report["evaluations"][final_update][split]["summary"]
                for report in matched
            ]
            summary[split] = {
                metric: {
                    "seed_values": [row[metric] for row in measurements],
                    "mean": float(np.mean([row[metric] for row in measurements])),
                    "sample_std": float(
                        np.std([row[metric] for row in measurements], ddof=1)
                    ),
                }
                for metric in (
                    "scene_macro_xy_ade_m",
                    "official_il_loss_whole_split",
                    "future_mse",
                    "persistence_mse",
                )
            }
        summary["development_curve"] = {
            str(update): [
                report["evaluations"][str(update)]["development"]["summary"][
                    "scene_macro_xy_ade_m"
                ]
                for report in matched
            ]
            for update in specification["evaluation_updates"]
        }
        summary["active_parameter_count"] = matched[0]["active_parameter_count"]
        summary["wall_seconds_all_seeds"] = sum(
            report["wall_seconds"] for report in matched
        )
        summary["peak_allocated_gib"] = max(
            report["peak_allocated_gib"] for report in matched
        )
        summary["projection_conflict_fraction"] = {
            str(report["seed"]): float(
                np.mean(
                    [
                        step["projection"]["conflicting"]
                        for step in report["curve"]
                        if step["projection"] is not None
                    ]
                )
            )
            for report in matched
            if any(step["projection"] is not None for step in report["curve"])
        }
        summary["command_development_ade"] = {
            command: {
                "windows": matched[0]["evaluations"][final_update]["development"][
                    "summary"
                ]["command_window_ade"][command]["count"],
                "seed_values": [
                    report["evaluations"][final_update]["development"]["summary"][
                        "command_window_ade"
                    ][command]["mean"]
                    for report in matched
                ],
            }
            for command in matched[0]["evaluations"][final_update]["development"][
                "summary"
            ]["command_window_ade"]
        }
        aggregate[condition] = summary
    baseline = {
        report["seed"]: report["evaluations"]["0"]["development"]["windows"]
        for report in reports
        if report["condition"] == specification["conditions"][0]
    }
    external_reference_provenance = None
    if "ego_reference" not in windows_by_condition:
        workspace = Path(__file__).resolve().parents[1]
        reference_directory = workspace / specification["paired_reference_run"]
        reference_specification = json.loads(
            (reference_directory / "specification.json").read_text()
        )
        for key in (
            "reused_cache",
            "warmup_root",
            "seeds",
            "joint_updates",
            "batch_size",
            "learning_rate",
            "selector_learning_rate",
            "future_auxiliary_weight",
            "weight_decay",
            "gradient_clip_norm",
        ):
            if specification[key] != reference_specification[key]:
                raise RuntimeError(f"External paired reference differs on {key}")
        reference_windows = {}
        for seed in specification["seeds"]:
            reference = json.loads(
                (
                    reference_directory / f"ego_reference_seed{seed}/results.json"
                ).read_text()
            )
            reference_windows[seed] = reference["evaluations"][final_update][
                "development"
            ]["windows"]
            if any(
                report["batch_schedule_sha256"] != reference["batch_schedule_sha256"]
                for report in reports
                if report["seed"] == seed
            ):
                raise RuntimeError("Unpaired training schedule in reference comparison")
        external_reference_provenance = str(reference_directory)
    else:
        reference_windows = windows_by_condition["ego_reference"]
    comparisons = {
        "against_original": {
            condition: paired_recording_comparison(windows, baseline)
            for condition, windows in windows_by_condition.items()
        },
        "against_ego_reference": {
            condition: paired_recording_comparison(windows, reference_windows)
            for condition, windows in windows_by_condition.items()
            if condition != "ego_reference"
        },
    }
    comparisons["against_matched_reference"] = {}
    for condition, windows in windows_by_condition.items():
        default_reference = (
            "mlp_reference"
            if condition.startswith("mlp_") and condition != "mlp_reference"
            else "ego_reference"
        )
        reference_name = (
            specification.get("condition_options", {})
            .get(condition, {})
            .get("reference_condition", default_reference)
        )
        if condition == reference_name:
            continue
        if reference_name in windows_by_condition:
            matched_windows = windows_by_condition[reference_name]
        elif external_reference_provenance is not None:
            matched_windows = {
                seed: json.loads(
                    (
                        Path(external_reference_provenance)
                        / f"{reference_name}_seed{seed}/results.json"
                    ).read_text()
                )["evaluations"][final_update]["development"]["windows"]
                for seed in specification["seeds"]
            }
        else:
            continue
        comparisons["against_matched_reference"][condition] = {
            "reference_condition": reference_name,
            **paired_recording_comparison(windows, matched_windows),
        }
    write_json(
        share / "summary.json",
        {
            "specification": specification,
            "completion": completion,
            "aggregate": aggregate,
            "external_reference_run": external_reference_provenance,
            "paired_comparisons": comparisons,
            "interpretation": "exploratory repeated-development comparison; not independent or safety/PDMS validation; no multiplicity correction",
        },
    )
    write_json(share / "source.json", json.loads((run / "source.json").read_text()))
    write_json(
        share / "gradient_and_dependence.json",
        [
            {
                key: report[key]
                for key in (
                    "condition",
                    "seed",
                    "gradient_contract",
                    "branch_dependence",
                    "initial_module_hashes",
                    "final_module_hashes",
                    "warmup_checkpoint",
                    "warmup_sha256",
                    "batch_schedule_sha256",
                )
            }
            for report in reports
        ],
    )
    fields = [
        "condition",
        "seed",
        "update",
        "split",
        "token",
        "recording",
        "scene_token",
        "command",
        "xy_ade_m",
        "future_mse",
        "persistence_mse",
        "supervised_patch_times",
        "selected_patch_ids",
    ]
    with (share / "window_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            for update, evaluation in report["evaluations"].items():
                for split, evaluated in evaluation.items():
                    for row in evaluated["windows"]:
                        writer.writerow(
                            {
                                "condition": report["condition"],
                                "seed": report["seed"],
                                "update": update,
                                "split": split,
                                **{key: row[key] for key in fields[4:]},
                            }
                        )
    for condition, summary in aggregate.items():
        value = summary["development"]["scene_macro_xy_ade_m"]
        print(f"{condition}: {value['mean']:.6f} +/- {value['sample_std']:.6f} m")


if __name__ == "__main__":
    main()
