"""Share compact measured results without publishing private cache/checkpoints."""

import json

import numpy as np
import train_target_supervision_ablation as exploration

PROJECT_ROOT = exploration.PROJECT_ROOT


def final_evaluation(run, split):
    return next(
        row
        for row in run["evaluations"]
        if row["update"] == run["end_update"] and row["split"] == split
    )


def main():
    raw_directory = (
        PROJECT_ROOT / "outputs/future_prediction_diagnostics/bounded_followup_1000_v1"
    )
    report = json.loads((raw_directory / "comparison_report.json").read_text())
    if report.get("status") != "bounded_followup_complete":
        raise RuntimeError("bounded execution not complete")
    old_path = (
        PROJECT_ROOT
        / "outputs/target_supervision_exploration/seed29_updates200_v1/comparison_report.json"
    )
    old = json.loads(old_path.read_text())
    cheap_path = (
        PROJECT_ROOT
        / "outputs/future_prediction_diagnostics/seed29_update200_v2/diagnostics.json"
    )
    cheap = json.loads(cheap_path.read_text())
    compact = {key: value for key, value in report.items() if key != "runs"}
    compact["runs"] = [
        {key: value for key, value in run.items() if key != "new_training_updates"}
        for run in report["runs"]
    ]
    compact["existing_200_update_prediction_diagnostics"] = cheap
    compact["old_report_sha256"] = exploration.file_sha256(old_path)
    compact["original_condition_summary"] = [
        {
            "label": row["condition"]["label"],
            "condition": row["condition"],
            "active_parameters": row["active_trainable_parameters"],
            "evaluations": row["evaluations"],
            "first_update_gradient_contract": row["first_update_gradient_contract"],
            "development_future_branch_dependence": row[
                "development_future_branch_dependence"
            ],
            "supervision_count_sequence_sha256": row[
                "supervision_count_sequence_sha256"
            ],
        }
        for row in old["condition_results"]
    ]
    a_seeds = [
        (
            29,
            next(
                row
                for row in old["condition_results"]
                if row["condition"]["label"] == "A"
            )["evaluations"],
        )
    ]
    a_seeds += [
        (run["seed"], run["evaluations"])
        for run in report["runs"]
        if run["condition"]["label"] == "A" and run["end_update"] == 200
    ]
    baseline_ade = [
        {
            "seed": seed,
            "scene_macro_ade_meters": next(
                row
                for row in evaluations
                if row["update"] == 200 and row["split"] == "development"
            )["scene_macro_ade_meters"],
        }
        for seed, evaluations in a_seeds
    ]
    compact["A_200_update_variability"] = {
        "per_seed": baseline_ade,
        "mean_scene_macro_ade_meters": float(
            np.mean([row["scene_macro_ade_meters"] for row in baseline_ade])
        ),
        "sample_std_scene_macro_ade_meters": float(
            np.std([row["scene_macro_ade_meters"] for row in baseline_ade], ddof=1)
        ),
        "interpretation": "Only A variability, not the variance of other conditions; three seeds, four development recordings.",
    }
    compact["paired_E_F"] = []
    for seed in (29, 11):
        paired = {
            label: next(
                run
                for run in report["runs"]
                if run["seed"] == seed and run["condition"]["label"] == label
            )
            for label in ("E", "F")
        }
        differences = {}
        for split in ("train", "development"):
            coupled = final_evaluation(paired["E"], split)
            detached = final_evaluation(paired["F"], split)
            differences[split] = {
                "E_scene_macro_ade_meters": coupled["scene_macro_ade_meters"],
                "F_scene_macro_ade_meters": detached["scene_macro_ade_meters"],
                "E_minus_F_scene_macro_ade_meters": coupled["scene_macro_ade_meters"]
                - detached["scene_macro_ade_meters"],
            }
        if paired["E"]["initial_state_sha256"] != paired["F"]["initial_state_sha256"]:
            raise RuntimeError("paired initial weights mismatch")
        e_batches = paired["E"]["new_training_updates"]
        f_batches = paired["F"]["new_training_updates"]
        if seed == 29:
            e_batches = (
                json.loads(
                    (
                        PROJECT_ROOT
                        / "outputs/target_supervision_exploration/seed29_updates200_v1/E/training_updates.json"
                    ).read_text()
                )
                + e_batches
            )
            e_counts = [
                row.get(
                    "common_valid_slot_times", row.get("common_valid_slot_time_count")
                )
                for row in e_batches
            ]
        else:
            e_counts = [row["common_valid_slot_times"] for row in e_batches]
        f_counts = [row["common_valid_slot_times"] for row in f_batches]
        if e_counts != f_counts:
            raise RuntimeError("paired supervision sequence differs")
        compact["paired_E_F"].append(
            {
                "seed": seed,
                "shared_initial_weights_verified": True,
                "identical_batch_and_common_supervision_sequence_verified": True,
                "common_supervision_slot_times_over_1000_updates": sum(e_counts),
                "differences": differences,
                "interpretation": "Gradient-coupling control, not standalone proof of future-information causality; seed29 E uses optimizer-state continuation without old RNG snapshots.",
            }
        )
    compact["cached_alignment"] = json.loads(
        (
            PROJECT_ROOT
            / "results/future_prediction_diagnostics/cached_target_alignment_v1.json"
        ).read_text()
    )
    output_path = (
        PROJECT_ROOT / "results/future_prediction_diagnostics/bounded_followup_v1.json"
    )
    if output_path.exists():
        raise RuntimeError("refusing to overwrite shared report")
    output_path.write_text(json.dumps(compact, indent=2) + "\n")
    print("A variability", json.dumps(compact["A_200_update_variability"]))
    for row in compact["paired_E_F"]:
        print("E_F", row["seed"], json.dumps(row["differences"]))
    for run in report["runs"]:
        evaluation = final_evaluation(run, "development")
        print(
            "RESULT",
            run["seed"],
            run["condition"]["label"],
            run["end_update"],
            evaluation["scene_macro_ade_meters"],
            evaluation["planning_loss"],
        )
        if "visual_prediction_diagnostics" in run:
            diagnostic = run["visual_prediction_diagnostics"]["development"]
            print(
                "VISUAL",
                {
                    name: diagnostic[name]["normalized_mse"]
                    for name in ("predictor", "persistence", "train_mean")
                },
                "variance",
                diagnostic["predictor"]["pooled_sample_entity_time_channel_variance"]
                / diagnostic["target"]["pooled_sample_entity_time_channel_variance"],
            )
    print("SHARED_FOLLOWUP_SUMMARY_DONE", output_path)


if __name__ == "__main__":
    main()
