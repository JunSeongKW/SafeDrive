"""Export completed evidence; no training, model selection, or held-out evaluation."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from train_bounded_future_prediction_followup import (
    generate_recording_balanced_batch_sequence,
)
from train_target_supervision_ablation import PROJECT_ROOT, file_sha256

from planning_aware_future_prediction.models.residual_future_supervision import (
    ResidualFutureSupervisionPilot,
)


def scene_macro_from_rows(rows, metric="ade_meters"):
    scenes = defaultdict(list)
    for row in rows:
        scenes[(row["recording_group"], row["scene_token"])].append(row[metric])
    return float(np.mean([np.mean(values) for values in scenes.values()]))


def paired_recording_bootstrap(paired_window_rows, replicates, seed):
    """Resample recordings, retaining all their scenes; not a seed bootstrap.

    Input differences are already averaged over the matched training seeds.
    Each original scene has equal weight, including when its recording is
    sampled multiple times. Unequal recording sizes must not change the
    scene-macro estimand into a recording-macro estimand.
    """
    scenes = defaultdict(list)
    for row in paired_window_rows:
        scenes[(row["recording_group"], row["scene_token"])].append(
            row["mean_matched_seed_difference_meters"]
        )
    recording_scenes = defaultdict(list)
    for (recording, _), differences in scenes.items():
        recording_scenes[recording].append(float(np.mean(differences)))
    recordings = sorted(recording_scenes)
    scene_sums = np.array([sum(recording_scenes[name]) for name in recordings])
    scene_counts = np.array([len(recording_scenes[name]) for name in recordings])
    generator = np.random.default_rng(seed)
    sampled_indices = generator.integers(
        0, len(recordings), size=(replicates, len(recordings))
    )
    sampled_differences = scene_sums[sampled_indices].sum(axis=1) / scene_counts[
        sampled_indices
    ].sum(axis=1)
    return {
        "difference_definition": "visual residual minus absolute, negative ADE difference is better",
        "scene_macro_difference_meters": float(scene_sums.sum() / scene_counts.sum()),
        "recording_cluster_percentile_95_interval_meters": np.quantile(
            sampled_differences, [0.025, 0.975]
        ).tolist(),
        "recordings": len(recordings),
        "scenes": int(scene_counts.sum()),
        "replicates": replicates,
        "bootstrap_seed": seed,
        "conditional_on": "these three matched training seeds, not training-seed uncertainty",
        "per_recording": {
            name: {
                "scenes": len(recording_scenes[name]),
                "scene_mean_difference_meters": float(np.mean(recording_scenes[name])),
            }
            for name in recordings
        },
        "limitations": "only four repeatedly used development recordings; exploratory, not an independent final test",
    }


def seed_statistics(values):
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
    }


def write_new_json(path, content):
    if path.exists():
        raise FileExistsError(f"preserve completed artifact: {path}")
    path.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    directory = arguments.output_directory.resolve()
    if directory.exists() or not directory.is_relative_to(PROJECT_ROOT / "results"):
        raise ValueError("require a NEW shared results directory")
    directory.mkdir(parents=True)
    experiment_root = PROJECT_ROOT / "outputs/pilot_foundation_decision"
    references_path = (
        experiment_root / "physical_ridge_references_v1b/reference_summary.json"
    )
    references = json.loads(references_path.read_text())
    comparison_root = experiment_root / "visual_residual_control_v1"
    comparison_path = comparison_root / "comparison_summary.json"
    comparison = json.loads(comparison_path.read_text())
    manifest_path = (
        experiment_root / "navtrain_three_way_split_v1/recording_split_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    old_root = (
        PROJECT_ROOT / "outputs/future_prediction_diagnostics/bounded_followup_1000_v1"
    )
    cache_index_path = (
        PROJECT_ROOT
        / "outputs/feature_caches/target_supervision_rectified_v1b/cache_index.json"
    )
    cache_index = json.loads(cache_index_path.read_text())
    training_windows = [
        {"metadata": row} for row in cache_index["records"] if row["split"] == "train"
    ]
    paired_rows_by_identity = defaultdict(list)
    paired_seed_results, checkpoint_checks = [], []
    for seed in comparison["registered_comparison"]["seeds"]:
        old_sequence = json.loads(
            (old_root / f"batch_sequence_seed{seed}.json").read_text()
        )
        generated_sequence = generate_recording_balanced_batch_sequence(
            training_windows, seed, 1000, 8
        )
        if generated_sequence != old_sequence:
            raise AssertionError(
                "reused absolute checkpoint used a different batch sequence"
            )
        condition_rows = {}
        for condition in ("C_absolute", "C_visual_residual"):
            run_directory = comparison_root / f"seed{seed}_{condition}"
            rows = json.loads(
                (run_directory / "development_per_window_update1000.json").read_text()
            )
            condition_rows[condition] = {
                (
                    row["recording_group"],
                    row["scene_token"],
                    row["current_frame_token"],
                ): row
                for row in rows
            }
            if len(condition_rows[condition]) != len(rows):
                raise AssertionError("duplicate evaluation window identity")
            if not (seed == 29 and condition == "C_absolute"):
                checkpoint_path = run_directory / "checkpoint_update1000.pt"
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True
                )
                model = ResidualFutureSupervisionPilot(
                    residual_visual=condition.endswith("residual")
                )
                model.load_state_dict(checkpoint["model"], strict=True)
                model.set_train_target_normalization(checkpoint["normalization"])
                finite_model = all(
                    torch.isfinite(tensor).all()
                    for tensor in checkpoint["model"].values()
                )
                optimizer_finite = all(
                    torch.isfinite(value).all()
                    for state in checkpoint["optimizer"]["state"].values()
                    for value in state.values()
                    if isinstance(value, torch.Tensor)
                )
                if (
                    not finite_model
                    or not optimizer_finite
                    or checkpoint["optimizer_updates"] != 1000
                ):
                    raise AssertionError("incomplete or nonfinite final checkpoint")
                if (
                    checkpoint["sampler_sequence"] != old_sequence
                    or checkpoint["sampler_next_update"] != 1001
                ):
                    raise AssertionError("sampler recovery state differs")
                if not checkpoint.get("rng_states"):
                    raise AssertionError("RNG recovery state missing")
                checkpoint_checks.append(
                    {
                        "path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
                        "sha256": file_sha256(checkpoint_path),
                        "strict_model_load": True,
                        "model_optimizer_finite": True,
                        "sampler_matches": True,
                        "rng_state_present": True,
                        "optimizer_updates": 1000,
                        "not_performed": "no second training or resume run",
                    }
                )
        if set(condition_rows["C_absolute"]) != set(
            condition_rows["C_visual_residual"]
        ):
            raise AssertionError("paired window sets differ")
        absolute_ade = scene_macro_from_rows(
            list(condition_rows["C_absolute"].values())
        )
        residual_ade = scene_macro_from_rows(
            list(condition_rows["C_visual_residual"].values())
        )
        paired_seed_results.append(
            {
                "seed": seed,
                "absolute_ade_meters": absolute_ade,
                "visual_residual_ade_meters": residual_ade,
                "residual_minus_absolute_ade_meters": residual_ade - absolute_ade,
                "matched_initial_source_and_batch_sequence_verified": True,
            }
        )
        for identity, absolute in condition_rows["C_absolute"].items():
            residual = condition_rows["C_visual_residual"][identity]
            if (
                absolute["visual_common_valid"]["valid_slot_times"]
                != residual["visual_common_valid"]["valid_slot_times"]
            ):
                raise AssertionError("common supervision count differs")
            paired_rows_by_identity[identity].append(
                {
                    "seed": seed,
                    "absolute_ade_meters": absolute["ade_meters"],
                    "visual_residual_ade_meters": residual["ade_meters"],
                    "difference_meters": residual["ade_meters"]
                    - absolute["ade_meters"],
                }
            )
    paired_rows = [
        {
            "recording_group": identity[0],
            "scene_token": identity[1],
            "current_frame_token": identity[2],
            "matched_seeds": values,
            "mean_matched_seed_difference_meters": float(
                np.mean([row["difference_meters"] for row in values])
            ),
        }
        for identity, values in sorted(paired_rows_by_identity.items())
    ]
    bootstrap = paired_recording_bootstrap(paired_rows, 2000, 20261002)
    compact_runs = [
        {key: value for key, value in run.items() if key != "training_curve"}
        for run in comparison["runs"]
    ]
    original_curve_hashes = {
        str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
        for path in sorted(comparison_root.glob("seed*/condition_result.json"))
    }
    statistics = {
        "baseline_commit": "607da52",
        "same_cache_index_sha256": comparison["cache_index_sha256"],
        "paired_seed_results": paired_seed_results,
        "absolute_ade_meters": seed_statistics(
            [row["absolute_ade_meters"] for row in paired_seed_results]
        ),
        "visual_residual_ade_meters": seed_statistics(
            [row["visual_residual_ade_meters"] for row in paired_seed_results]
        ),
        "paired_seed_difference_meters": seed_statistics(
            [row["residual_minus_absolute_ade_meters"] for row in paired_seed_results]
        ),
        "recording_cluster_bootstrap": bootstrap,
        "final_checkpoint_checks": checkpoint_checks,
        "raw_training_curve_source_sha256": original_curve_hashes,
        "interpretation": "visual forecasting improves over persistence slightly; planning improvement not established; pilot remains weaker than train-fitted ridge",
        "priority_update": "user paused predictor tuning/data expansion; preserve diagnostic pilot, prioritize released prediction-planning foundation and selection/budget experiments",
        "completed_new_training_updates": comparison["new_optimizer_updates"],
        "cpu_training_comparison_wall_seconds": comparison["wall_seconds"],
        "expanded_cache_or_training_executed": False,
        "held_out_model_evaluation_performed": False,
    }
    write_new_json(directory / "physical_and_ridge_references.json", references)
    write_new_json(directory / "visual_residual_paired_statistics.json", statistics)
    write_new_json(
        directory / "visual_residual_runs.json", {**comparison, "runs": compact_runs}
    )
    write_new_json(directory / "visual_residual_per_window_paired.json", paired_rows)
    # Small JSON of every ego prediction error, reusable for future paired checks.
    ego_windows = {}
    for split in ("train", "development"):
        raw = json.loads(
            (references_path.parent / f"{split}_per_window.json").read_text()
        )
        ego_windows[split] = {
            method: [
                {
                    key: value
                    for key, value in row.items()
                    if key
                    in (
                        "recording_group",
                        "scene_token",
                        "current_frame_token",
                        "ade_meters",
                        "fde_meters",
                        "xy_position_errors_meters",
                    )
                }
                for row in rows
            ]
            for method, rows in raw.items()
            if "ade_meters" in rows[0]
        }
    write_new_json(directory / "ego_references_per_window.json", ego_windows)
    write_new_json(directory / "navtrain_three_way_manifest.json", manifest)
    for source, destination in (
        (
            experiment_root
            / "preprocessing_sensitivity_v1/preprocessing_evidence.json",
            "preprocessing_evidence.json",
        ),
        (
            experiment_root / "navtrain_profile200_gate_v1/execution_gate.json",
            "expanded_cache_execution_gate.json",
        ),
        (
            experiment_root / "wa_jepa_attention_gradient_v1.json",
            "official_wa_jepa_attention_gradient.json",
        ),
    ):
        write_new_json(directory / destination, json.loads(source.read_text()))
    print(
        json.dumps(
            {
                key: statistics[key]
                for key in (
                    "absolute_ade_meters",
                    "visual_residual_ade_meters",
                    "paired_seed_difference_meters",
                    "recording_cluster_bootstrap",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
