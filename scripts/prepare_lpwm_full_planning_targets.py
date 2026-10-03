"""Build official NAVSIM ego targets without feature caching or privileged inputs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from pyquaternion import Quaternion
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from prepare_lpwm_navsim_posttraining import file_digest


def ego_supervision(frames):
    current = frames[3]
    current_yaw = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
    current_xy = np.asarray(current["ego2global_translation"][:2])
    rotation = np.array([[np.cos(current_yaw), -np.sin(current_yaw)], [np.sin(current_yaw), np.cos(current_yaw)]])
    trajectory = []
    for frame in frames[4:12]:
        local_xy = (np.asarray(frame["ego2global_translation"][:2]) - current_xy) @ rotation
        yaw_difference = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0] - current_yaw
        trajectory.append([*local_xy, np.arctan2(np.sin(yaw_difference), np.cos(yaw_difference))])
    return np.r_[current["driving_command"], current["ego_dynamic_state"]].astype(np.float32), np.asarray(trajectory, dtype=np.float32)


def main(config_path):
    specification = json.loads(config_path.read_text())
    output_root = PROJECT_ROOT / specification["output_directory"]
    destination = output_root / "planning_targets.npz"
    if destination.exists():
        print("PLANNING_TARGETS_ALREADY_READY", flush=True)
        return
    records = json.loads((output_root / "planning_records.json").read_text())["records"]
    by_segment = defaultdict(list)
    for record in records:
        by_segment[record["segment_filename"]].append(record)
    prepared, statuses, trajectories, discarded = [], [], [], []
    reference_index_path = PROJECT_ROOT / "outputs/drive_jepa_selective_future/overnight_recording_coverage_v1_20261003/cache_index.json"
    reference_by_token = {row["current_frame_token"]: row for row in json.loads(reference_index_path.read_text())["records"]}
    maximum_reference_error, compared_references = 0., 0
    for segment_index, (filename, segment_records) in enumerate(by_segment.items()):
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
            frames = pickle.load(stream)
        for record in segment_records:
            sequence = frames[record["start_index"]:record["start_index"] + 12]
            assert sequence[3]["token"] == record["current_frame_token"]
            if not np.allclose(np.diff([frame["timestamp"] for frame in sequence]) / 1e6, .5, atol=.1):
                discarded.append(record["current_frame_token"])
                continue
            status, trajectory = ego_supervision(sequence)
            if record["current_frame_token"] in reference_by_token:
                cached = torch.load(reference_by_token[record["current_frame_token"]]["cache_file"], map_location="cpu", weights_only=True, mmap=True)
                np.testing.assert_allclose(status, cached["current_ego_status"].numpy(), atol=1e-6)
                error = float(np.max(np.abs(trajectory - cached["ego_trajectory_target"].numpy())))
                np.testing.assert_allclose(trajectory, cached["ego_trajectory_target"].numpy(), atol=1e-5)
                maximum_reference_error = max(maximum_reference_error, error)
                compared_references += 1
            prepared.append(record)
            statuses.append(status)
            trajectories.append(trajectory)
        if segment_index % 50 == 0:
            print("PLANNING_TARGETS", len(prepared), flush=True)
    assert compared_references > 0
    np.savez(destination, ego_status=np.stack(statuses), ego_trajectory_target=np.stack(trajectories))
    manifest = {"records": prepared, "counts": dict(Counter(row["split"] for row in prepared)),
        "recording_counts": {split: len({row["recording_group"] for row in prepared if row["split"] == split}) for split in ("train", "development")},
        "discarded_irregular_timestamp_tokens": discarded, "compared_official_cached_targets": compared_references,
        "maximum_official_target_difference": maximum_reference_error, "targets_sha256": file_digest(destination),
        "observed_input": "four front RGB images and current command/velocity/acceleration only",
        "source_planning_records_sha256": file_digest(output_root / "planning_records.json")}
    (output_root / "planning_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("PLANNING_TARGETS_READY", manifest["counts"], manifest["recording_counts"], maximum_reference_error, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/full_posttraining_v2.json")
    arguments = parser.parse_args()
    main(arguments.config.resolve())
