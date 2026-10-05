"""Cache the exact official DrivoR navtrain/navval scene lists and four cameras.

Raw shared data are read-only. Missing official scenes fail the completion gate;
there is no replacement with the old internal LPWM recording split.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import time

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.interpolate import CubicSpline
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAMERA_NAMES = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def process_segment(job):
    log_name, expected_tokens, split, destination = job
    destination = Path(destination)
    metadata_path = destination / "segments" / (log_name + ".json")
    if metadata_path.exists():
        cached = json.loads(metadata_path.read_text())
        if cached["expected_tokens"] == sorted(expected_tokens):
            for name in ("images.npy", "ego.npy", "trajectory.npy", "trajectory_long.npy"):
                assert (metadata_path.parent / log_name / name).exists()
            return cached
        raise RuntimeError("Existing segment cache has a different official token set")
    source = PROJECT_ROOT / "dataset/navsim_logs/trainval" / (log_name + ".pkl")
    with source.open("rb") as stream:
        frames = pickle.load(stream)
    expected_tokens = set(expected_tokens)
    selected = [(index, frame) for index, frame in enumerate(frames)
                if frame["token"] in expected_tokens]
    record_directory = metadata_path.parent / log_name
    record_directory.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(record_directory / "images.npy", mode="w+", dtype=np.uint8,
                                      shape=(len(selected), 4, 128, 128, 3))
    ego_status, trajectories, longer_trajectories, records = [], [], [], []
    for row_index, (current_index, current) in enumerate(selected):
        assert current_index >= 3 and current_index + 10 < len(frames)
        assert len(current["roadblock_ids"]) > 0
        camera_paths = []
        for camera_index, name in enumerate(CAMERA_NAMES):
            path = PROJECT_ROOT / "dataset/sensor_blobs/trainval" / current["cams"][name]["data_path"]
            with Image.open(path) as image:
                images[row_index, camera_index] = np.asarray(image.convert("RGB").resize((128, 128), Image.Resampling.BICUBIC))
            camera_paths.append(str(path))
        current_yaw = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
        origin = np.asarray(current["ego2global_translation"][:2])
        rotation = np.array([[np.cos(current_yaw), -np.sin(current_yaw)],
                             [np.sin(current_yaw), np.cos(current_yaw)]])
        future_states = []
        for future_frame in frames[current_index + 1:current_index + 11]:
            position = (np.asarray(future_frame["ego2global_translation"][:2]) - origin) @ rotation
            heading = Quaternion(*future_frame["ego2global_rotation"]).yaw_pitch_roll[0] - current_yaw
            future_states.append([*position, np.arctan2(np.sin(heading), np.cos(heading))])
        future_states = np.asarray(future_states, dtype=np.float32)
        # Same additional-two-poses CubicSpline target as DrivoRTargetBuilder.
        sample_times = np.arange(8, dtype=np.float32)
        sample_times += np.cumsum((sample_times + 1) * (4 / (8 * 9)))
        longer = np.stack([CubicSpline(np.arange(10, dtype=np.float32), future_states[:, axis])(sample_times)
                           for axis in range(3)], -1).astype(np.float32)
        ego_status.append(np.r_[np.zeros(3), current["ego_dynamic_state"], current["driving_command"]].astype(np.float32))
        assert ego_status[-1].shape == (11,)
        trajectories.append(future_states[:8])
        longer_trajectories.append(longer)
        records.append({"token": current["token"], "log_name": log_name,
            "recording_group": "_".join(log_name.split("_")[:-2]), "split": split,
            "current_frame_index": current_index, "cache_directory": str(record_directory),
            "cache_row": row_index, "current_camera_paths": camera_paths})
    images.flush()
    np.save(record_directory / "ego.npy", np.asarray(ego_status, dtype=np.float32))
    np.save(record_directory / "trajectory.npy", np.asarray(trajectories, dtype=np.float32))
    np.save(record_directory / "trajectory_long.npy", np.asarray(longer_trajectories, dtype=np.float32))
    assert {record["token"] for record in records} == expected_tokens
    result = {"expected_tokens": sorted(expected_tokens), "records": records, "source_sha256": digest(source)}
    write_json(metadata_path, result)
    return result


def prepare(arguments):
    started = time.time()
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    configuration_root = PROJECT_ROOT / "reference_repositories/DrivoR/navsim/planning/script/config"
    official_filter_path = configuration_root / "common/train_test_split/scene_filter/navtrain.yaml"
    official_split_path = configuration_root / "training/default_train_val_test_log_split.yaml"
    official_filter = yaml.safe_load(official_filter_path.read_text())
    official_split = yaml.safe_load(official_split_path.read_text())
    expected_tokens = set(official_filter["tokens"])
    train_logs, validation_logs = set(official_split["train_logs"]), set(official_split["val_logs"])
    jobs, discovered_tokens = [], set()
    # A light first pass determines exact membership before any training starts.
    for log_name in sorted(official_filter["log_names"]):
        path = PROJECT_ROOT / "dataset/navsim_logs/trainval" / (log_name + ".pkl")
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        tokens = [frame["token"] for index, frame in enumerate(frames)
                  if frame["token"] in expected_tokens and index >= 3 and index + 10 < len(frames)
                  and len(frame["roadblock_ids"]) > 0]
        assert log_name in train_logs or log_name in validation_logs
        split = "navtrain" if log_name in train_logs else "navval"
        discovered_tokens.update(tokens)
        jobs.append((log_name, tokens, split, str(destination)))
        if arguments.engineering_scene_count and tokens:
            tokens = tokens[:arguments.engineering_scene_count]
            jobs = [(log_name, tokens, split, str(destination))]
            discovered_tokens = expected_tokens = set(tokens)
            break
    assert discovered_tokens == expected_tokens, f"Missing {len(expected_tokens - discovered_tokens)} official training scenes"
    write_json(destination / "official_membership.json", {"tokens": sorted(expected_tokens),
        "scene_filter_sha256": digest(official_filter_path), "split_sha256": digest(official_split_path)})
    records, completed = [], 0
    with ProcessPoolExecutor(max_workers=arguments.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(process_segment, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            records.extend(result["records"])
            completed += 1
            if completed % 10 == 0 or completed == len(jobs):
                status = {"completed_segments": completed, "total_segments": len(jobs),
                          "completed_scenes": len(records), "elapsed_seconds": time.time() - started}
                write_json(destination / "progress.json", status)
                print(json.dumps(status), flush=True)
    records.sort(key=lambda record: (record["log_name"], record["current_frame_index"]))
    assert len(records) == len(expected_tokens) == len({record["token"] for record in records})
    for index, record in enumerate(records):
        record["index"] = index
    manifest = {"complete": True, "records": records, "counts": dict(Counter(record["split"] for record in records)),
        "scene_filter_sha256": digest(official_filter_path), "split_sha256": digest(official_split_path),
        "camera_order": CAMERA_NAMES, "observation_frames_per_camera": 1,
        "image_shape": [4, 128, 128, 3], "future_images_used": False,
        "no_internal_recording_resplit": True, "engineering_only": bool(arguments.engineering_scene_count),
        "elapsed_seconds": time.time() - started}
    write_json(destination / "manifest.json", manifest)
    print("OFFICIAL_DRIVOR_SCENES_READY", manifest["counts"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--engineering-scene-count", type=int, default=0)
    prepare(parser.parse_args())
