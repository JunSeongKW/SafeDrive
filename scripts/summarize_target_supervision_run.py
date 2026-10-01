"""Share actual small-run records; cluster intervals are descriptive, not tests."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def paired_recording_cluster_interval(condition_rows, reference_rows):
    reference = {row["current_frame_token"]: row for row in reference_rows}
    scenes = defaultdict(list)
    for row in condition_rows:
        matching_row = reference[row["current_frame_token"]]
        if (row["recording_group"], row["scene_token"]) != (
            matching_row["recording_group"],
            matching_row["scene_token"],
        ):
            raise RuntimeError("paired metadata differs")
        scenes[(row["recording_group"], row["scene_token"])].append(
            row["ade_meters"] - matching_row["ade_meters"]
        )
    recording_scene_deltas = defaultdict(list)
    for (recording, _), values in scenes.items():
        recording_scene_deltas[recording].append(float(np.mean(values)))
    recordings = sorted(recording_scene_deltas)
    generator = np.random.default_rng(29)
    estimates = []
    for _ in range(1000):
        drawn_recordings = generator.choice(
            recordings, size=len(recordings), replace=True
        )
        estimates.append(
            float(
                np.mean(
                    [
                        delta
                        for recording in drawn_recordings
                        for delta in recording_scene_deltas[recording]
                    ]
                )
            )
        )
    return {
        "reference": "B",
        "metric": "paired scene-macro ADE difference meters; negative favors condition",
        "point_estimate_meters": float(
            np.mean([np.mean(values) for values in scenes.values()])
        ),
        "percentile_95_interval_meters": np.quantile(
            estimates, [0.025, 0.975]
        ).tolist(),
        "resamples": 1000,
        "recording_clusters": len(recordings),
        "interpretation": "Descriptive only: four development recordings and one training seed; not significance or generalization evidence.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    run_directory = arguments.run_directory.resolve()
    output_path = arguments.output_file.resolve()
    if not output_path.is_relative_to(PROJECT_ROOT / "results") or output_path.exists():
        raise RuntimeError("new shared result path required, no overwrite")
    run_report_path = run_directory / "comparison_report.json"
    report = json.loads(run_report_path.read_text())
    if report.get("status") != "all_five_conditions_completed_200_updates":
        raise RuntimeError("cannot summarize unfinished run as complete")
    cache_directory = PROJECT_ROOT / report["configuration"]["cache_directory"]
    cache_index = json.loads((cache_directory / "cache_index.json").read_text())
    cache_report = json.loads((cache_directory / "cache_report.json").read_text())
    cache_profile = json.loads((cache_directory / "profile_report.json").read_text())
    # Independently recheck source files after cache/training, not only an
    # assertion that our implementation did not issue shared-dataset writes.
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    split_manifest = json.loads(
        (PROJECT_ROOT / report["configuration"]["split_manifest"]).read_text()
    )
    for group in split_manifest["groups"]:
        if (
            sha256(dataset_root / "navsim_logs/mini" / group["segment_filename"])
            != group["source_sha256"]
        ):
            raise RuntimeError("shared source log hash changed")
    image_hashes = json.loads(
        (cache_directory / "source_image_sha256.json").read_text()
    )
    for relative_path, expected_sha in image_hashes.items():
        image_path = (dataset_root / "sensor_blobs/mini" / relative_path).resolve()
        if (
            not image_path.is_relative_to(
                (dataset_root / "sensor_blobs/mini").resolve()
            )
            or sha256(image_path) != expected_sha
        ):
            raise RuntimeError("source image hash/path changed")
    for source_path, expected_sha in report["source_sha256"].items():
        if sha256(PROJECT_ROOT / source_path) != expected_sha:
            raise RuntimeError("training source changed during run")
    report["post_run_shared_source_verification"] = {
        "log_files": len(split_manifest["groups"]),
        "image_files": len(image_hashes),
        "all_pre_post_sha256_equal": True,
        "training_source_matches_recorded_sha256": True,
    }
    reference_rows = json.loads(
        (run_directory / "B/development_update_200.json").read_text()
    )
    for condition_result in report["condition_results"]:
        label = condition_result["condition"]["label"]
        condition_rows = json.loads(
            (run_directory / label / "development_update_200.json").read_text()
        )
        condition_result["paired_recording_cluster_interval_vs_B"] = (
            paired_recording_cluster_interval(condition_rows, reference_rows)
        )
        training_updates_path = run_directory / label / "training_updates.json"
        updates = json.loads(training_updates_path.read_text())
        condition_result["training_curve_record"] = {
            "source_sha256": sha256(training_updates_path),
            "first_25_update_mean_losses": {
                key: float(np.mean([row[key] for row in updates[:25]]))
                for key in ("planning_loss", "total_loss")
            },
            "last_25_update_mean_losses": {
                key: float(np.mean([row[key] for row in updates[-25:]]))
                for key in ("planning_loss", "total_loss")
            },
            "common_supervised_slot_times_across_training_draws": sum(
                row["common_valid_slot_time_count"] for row in updates
            ),
            "zero_common_target_sample_draws": sum(
                row["zero_common_target_samples"] for row in updates
            ),
        }
        condition_result["local_checkpoint"] = {
            "path": str(
                (run_directory / label / "last_update_200.pt").relative_to(PROJECT_ROOT)
            ),
            "sha256": sha256(run_directory / label / "last_update_200.pt"),
        }
    report["local_run_directory"] = str(run_directory.relative_to(PROJECT_ROOT))
    report["original_run_report_sha256"] = sha256(run_report_path)
    report["cache_generation"] = cache_report
    report["cache_train_eight_window_profile"] = cache_profile
    normalization = torch.load(
        run_directory / "train_only_normalization.pt",
        weights_only=True,
        map_location="cpu",
    )
    report["normalization_statistics"] = {
        name: {
            "observations": values["observations"],
            "channels_at_std_floor": values["channels_at_std_floor"],
            "std_min": float(values["std"].min()),
            "std_median": float(values["std"].median()),
            "std_max": float(values["std"].max()),
            "mean_vector": values["mean"].tolist() if name == "spatial" else None,
            "std_vector": values["std"].tolist() if name == "spatial" else None,
        }
        for name, values in normalization.items()
    }
    report["normalization_tensor_file_sha256"] = sha256(
        run_directory / "train_only_normalization.pt"
    )
    projection_audit_path = (
        PROJECT_ROOT
        / "outputs/projection_audit/front_camera_rectified_verified_20261001/projection_audit.json"
    )
    report["projection_audit"] = json.loads(projection_audit_path.read_text())
    report["projection_audit_file_sha256"] = sha256(projection_audit_path)
    if (
        sha256(PROJECT_ROOT / "scripts/audit_front_camera_projection.py")
        != report["projection_audit"]["source_script_sha256"]
    ):
        raise RuntimeError("projection audit source changed since verification")
    report["summarizer_source_sha256"] = sha256(Path(__file__))
    report["cache_coverage"] = {}
    for split in ("train", "development"):
        records = [
            record for record in cache_index["records"] if record["split"] == split
        ]
        report["cache_coverage"][split] = {
            "windows": len(records),
            "recordings": len({record["recording_group"] for record in records}),
            "current_front_valid_count_gt4_windows": sum(
                record["current_front_valid_count"] > 4 for record in records
            ),
            "zero_current_front_valid_windows": sum(
                record["current_front_valid_count"] == 0 for record in records
            ),
            "future_validity_used_for_selection": False,
        }
    report["limits"] = [
        "GT boxes/tracks/current spatial state are privileged; front camera only.",
        "Stored JPEG export history not independently byte-verified; explicit stored-K/D rectification is the operational convention.",
        "Projection validity does not prove object visibility; occlusion and sensor synchronization remain imperfect.",
        "Frozen encoder and newly initialized planner/predictor, not full official Drive-JEPA reproduction or visual JEPA pretraining.",
        "B-E have two heads, identical structure, whitened predicted future input; only auxiliary loss type/weights differ.",
        "A has different active capacity. B is not guaranteed to predict future semantics.",
        "Common-mask supervision answers a conditional question; native-mask forecast evaluation is separate.",
        "E uses 0.05+0.05 versus C/D 0.1; total coefficient matching does not match gradient energy.",
        "200 updates, one seed, small navmini development split. No target elimination/winner/novelty/H1-H2 decision.",
        "Imitation ADE/FDE/loss only, no official collision/comfort/PDMS evaluator.",
        "Cached-head update timing is not full current-image pipeline latency or an end-to-end efficiency claim.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"SHARED_RESULT_WRITTEN {output_path}")
    for condition in report["condition_results"]:
        final_train = next(
            row
            for row in condition["evaluations"]
            if row["split"] == "train" and row["update"] == 200
        )
        final_dev = next(
            row
            for row in condition["evaluations"]
            if row["split"] == "development" and row["update"] == 200
        )
        print(
            condition["condition"]["label"],
            "train_ADE",
            final_train["scene_macro_ade_meters"],
            "dev_ADE",
            final_dev["scene_macro_ade_meters"],
            "dev_FDE",
            final_dev["fde_meters"],
        )


if __name__ == "__main__":
    main()
