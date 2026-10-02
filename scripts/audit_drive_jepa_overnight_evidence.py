"""Independently check fixed-step evidence; never select a checkpoint or train."""

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

BASELINE_PARAMETER_SHA256 = (
    "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a"
)
STUDIES = {
    "overnight_causal_followup_v1_20261003": (21, 400),
    "overnight_coverage_training_v1_20261003": (27, 800),
    "overnight_projection_transfer_v1_20261003": (9, 800),
    "overnight_conservative_adaptation_v1_20261003": (12, 800),
}


def scene_macro_ade(window_rows):
    scene_errors = defaultdict(list)
    seen_tokens = set()
    for row in window_rows:
        if row["token"] in seen_tokens:
            raise ValueError("Duplicate evaluation window")
        seen_tokens.add(row["token"])
        if not math.isfinite(row["xy_ade_m"]) or row["xy_ade_m"] < 0:
            raise ValueError("Invalid trajectory error")
        scene_errors[(row["recording"], row["scene_token"])].append(row["xy_ade_m"])
    if not scene_errors:
        raise ValueError("Missing evaluation windows")
    return mean(mean(errors) for errors in scene_errors.values())


def require_close(measured, reported, label):
    if not math.isclose(measured, reported, rel_tol=0, abs_tol=1e-10):
        raise ValueError(f"{label}: independently computed {measured} != {reported}")


def summarize_patch_spread(window_rows):
    """Descriptive grid coverage, not a test of relevance or collapse."""
    patch_counts = Counter()
    normalized_distances = []
    for row in window_rows:
        selected_ids = row["selected_patch_ids"]
        if len(selected_ids) != 4 or len(set(selected_ids)) != 4:
            raise ValueError("Expected four distinct selected patches")
        if any(patch_id < 0 or patch_id >= 512 for patch_id in selected_ids):
            raise ValueError("Patch outside the official 16 by 32 grid")
        patch_counts.update(selected_ids)
        for position, first in enumerate(selected_ids):
            for second in selected_ids[position + 1 :]:
                normalized_distances.append(
                    math.hypot(
                        (first % 32 - second % 32) / 31,
                        (first // 32 - second // 32) / 15,
                    )
                    / math.sqrt(2)
                )
    total = sum(patch_counts.values())
    return {
        "unique_patch_ids": len(patch_counts),
        "marginal_entropy_over_log512": -sum(
            count / total * math.log(count / total) for count in patch_counts.values()
        )
        / math.log(512),
        "mean_pairwise_grid_distance_over_diagonal": mean(normalized_distances),
        "most_frequent_patch_counts": patch_counts.most_common(4),
        "interpretation": "concentrated spatial choices may be relevant or redundant; neither uniformity nor diversity establishes planning utility",
    }


def audit_study(workspace, study_name, expected_runs, final_update):
    raw_directory = workspace / "outputs/drive_jepa_selective_future" / study_name
    shared_directory = workspace / "results/drive_jepa_selective_future" / study_name
    summary_path = shared_directory / "summary.json"
    summary = json.loads(summary_path.read_text())
    specification = json.loads((raw_directory / "specification.json").read_text())
    completion = json.loads((raw_directory / "completion.json").read_text())
    if summary["specification"] != specification or summary["completion"] != completion:
        raise ValueError("Export differs from raw experiment specification/completion")
    if (
        not completion["complete"]
        or completion["baseline_hash_unchanged"] != BASELINE_PARAMETER_SHA256
    ):
        raise ValueError("Incomplete run or changed original parameters")
    if specification["joint_updates"] != final_update:
        raise ValueError("Unregistered final update")
    if len(specification["conditions"]) * len(specification["seeds"]) != expected_runs:
        raise ValueError("Unexpected experiment count")
    if len(completion["results"]) != expected_runs:
        raise ValueError("Missing completed experiment")
    schedule_by_seed_and_pool = {}
    original_by_token = {}
    condition_rows = {}
    for condition in specification["conditions"]:
        development_values = []
        selection_spread_by_seed = {}
        for seed in specification["seeds"]:
            run_directory = raw_directory / f"{condition}_seed{seed}"
            report = json.loads((run_directory / "results.json").read_text())
            if report["condition"] != condition or report["seed"] != seed:
                raise ValueError("Experiment identity mismatch")
            if [row["update"] for row in report["curve"]] != list(
                range(1, final_update + 1)
            ):
                raise ValueError("Missing/duplicated training updates")
            if not (run_directory / "complete.pt").is_file():
                raise ValueError("Missing recovery checkpoint")
            for split in ("train", "development"):
                evaluated = report["evaluations"][str(final_update)][split]
                independently_computed = scene_macro_ade(evaluated["windows"])
                require_close(
                    independently_computed,
                    evaluated["summary"]["scene_macro_xy_ade_m"],
                    "scene aggregation",
                )
                if split == "development":
                    development_values.append(independently_computed)
                    selection_spread_by_seed[str(seed)] = summarize_patch_spread(
                        evaluated["windows"]
                    )
            original = {
                row["token"]: row["xy_ade_m"]
                for row in report["evaluations"]["0"]["development"]["windows"]
            }
            if original_by_token and original != original_by_token:
                raise ValueError("Unpaired baseline outputs or evaluation population")
            original_by_token = original
            # The first immutable study predates the explicit count field.
            training_window_count = report.get(
                "actual_training_window_count",
                len(report["evaluations"][str(final_update)]["train"]["windows"]),
            )
            schedule_key = (seed, training_window_count)
            schedule_hash = report["batch_schedule_sha256"]
            if (
                schedule_key in schedule_by_seed_and_pool
                and schedule_by_seed_and_pool[schedule_key] != schedule_hash
            ):
                raise ValueError("Unpaired batch order at matched training pool size")
            schedule_by_seed_and_pool[schedule_key] = schedule_hash
            auxiliary = report["gradient_contract"]["auxiliary"]
            if auxiliary["patch_selector"] != 0 or auxiliary["future_bridge"] != 0:
                raise ValueError("Future auxiliary gradient leaked to selector/bridge")
        exported = summary["aggregate"][condition]["development"][
            "scene_macro_xy_ade_m"
        ]
        for measured, reported in zip(development_values, exported["seed_values"]):
            require_close(measured, reported, "paired seed metric")
        require_close(mean(development_values), exported["mean"], "seed mean")
        condition_rows[condition] = {
            "development_ade_m": exported,
            "versus_original": summary["paired_comparisons"]["against_original"][
                condition
            ],
            "actual_training_windows": training_window_count,
            "selection_spread_by_seed": selection_spread_by_seed,
        }
    return {
        "passed": True,
        "completed_runs": expected_runs,
        "joint_updates": expected_runs * final_update,
        "series_wall_seconds": completion["wall_seconds"],
        "peak_allocated_gib": max(
            row["peak_allocated_gib"] for row in summary["aggregate"].values()
        ),
        "baseline_development_ade_m": scene_macro_ade(
            report["evaluations"]["0"]["development"]["windows"]
        ),
        "development_windows": len(original_by_token),
        "summary_path": str(summary_path.relative_to(workspace)),
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "conditions": condition_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    audited = {
        name: audit_study(workspace, name, *limits) for name, limits in STUDIES.items()
    }
    result = {
        "all_passed": True,
        "completed_runs": sum(row["completed_runs"] for row in audited.values()),
        "joint_updates": sum(row["joint_updates"] for row in audited.values()),
        "scope": "CPU artifact and independent aggregation checks; no model re-execution or held-out evaluation",
        "studies": audited,
    }
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("all_passed", "completed_runs", "joint_updates")
            }
        )
    )


if __name__ == "__main__":
    main()
