"""Seal identical held-out scene membership and preserve each model's input recipe."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pickle
import time

import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def prepare(configuration_path):
    configuration = read_json(configuration_path)
    output = ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "inputs_complete.json").exists():
        assert read_json(output / "inputs_complete.json")["configuration_sha256"] == digest(configuration_path)
        return
    source = ROOT / "outputs/lpwm_drivor_joint_v1/evaluation_inputs/navtest"
    input_manifest = read_json(source / "manifest.json")
    official_filter = ROOT / "reference_repositories/DrivoR/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"
    official_tokens = set(yaml.safe_load(official_filter.read_text())["tokens"])
    assert input_manifest["complete"] and set(input_manifest["tokens"]) == official_tokens and len(official_tokens) == 12146
    cache_root = ROOT / "outputs/official_drive_jepa_reproduction/downloaded_metric_cache/metric_cache"
    caches = {path.parent.name: path for path in cache_root.glob("*/*/*/metric_cache.pkl")}
    assert not official_tokens - set(caches), "Do not silently change the panel for missing metric caches"
    primary_training_path = ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json"
    adapter_training_path = ROOT / "outputs/lpwm_navsim_full_posttraining_v2/planning_manifest.json"
    stage1_training_path = ROOT / "outputs/lpwm_navsim_full_posttraining_v2/manifest.json"
    primary_training = read_json(primary_training_path)["records"]
    adapter_training = [row for row in read_json(adapter_training_path)["records"] if row["split"] == "train"]
    stage1_training = [row for row in read_json(stage1_training_path)["records"] if row["split"] == "train"]
    training_tokens = {row["token"] for row in primary_training} | {row["current_frame_token"] for row in adapter_training + stage1_training}
    training_recordings = {row["recording_group"] for row in primary_training + adapter_training + stage1_training}
    assert not official_tokens & training_tokens
    grouped = defaultdict(list)
    for token in official_tokens:
        recording = "_".join(caches[token].parts[-4].split("_")[:-2])
        grouped[recording].append(token)
    assert not set(grouped) & training_recordings, "Recording-level training overlap"
    ordering = lambda value: hashlib.sha256((configuration["selection_seed"] + "/" + value).encode()).hexdigest()
    groups = sorted(grouped, key=ordering)
    for group in groups:
        grouped[group].sort(key=ordering)
    tokens, offset = [], 0
    while len(tokens) < configuration["scene_count"]:
        for group in groups:
            if offset < len(grouped[group]):
                tokens.append(grouped[group][offset])
                if len(tokens) == configuration["scene_count"]:
                    break
        offset += 1
    records = [{"token": token, "log_name": caches[token].parts[-4],
                "recording_group": "_".join(caches[token].parts[-4].split("_")[:-2]),
                "metric_cache_file": str(caches[token])} for token in tokens]
    manifest = {"tokens": tokens, "records": records, "selection": configuration["selection"],
                "selection_seed": configuration["selection_seed"], "full_navtest": False,
                "training_token_overlap": 0, "training_recording_overlap": 0,
                "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                    (official_filter, primary_training_path, adapter_training_path, stage1_training_path)},
                "sealed_before_predictions_unix": time.time()}
    panel = output / "panel.json"
    if panel.exists():
        existing = read_json(panel)
        assert existing["tokens"] == tokens and existing["source_sha256"] == manifest["source_sha256"]
        manifest = existing
    else:
        write_json(panel, manifest)
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    input_row = {token: index for index, token in enumerate(input_manifest["tokens"])}
    primary_images = np.load(source / "images.npy", mmap_mode="r")
    primary_ego = np.load(source / "ego.npy", mmap_mode="r")
    selected_rows = [input_row[token] for token in tokens]
    np.save(inputs / "primary_current_images.npy", np.array(primary_images[selected_rows]))
    np.save(inputs / "primary_ego.npy", np.array(primary_ego[selected_rows]))
    adapter_images = np.lib.format.open_memmap(inputs / "adapter_observed_images.npy", mode="w+", dtype=np.uint8,
                                              shape=(len(tokens), 4, 128, 128, 3))
    adapter_ego = np.empty((len(tokens), 8), dtype=np.float32)
    ground_truth = np.empty((len(tokens), 8, 3), dtype=np.float32)
    by_log = defaultdict(list)
    for index, record in enumerate(records):
        by_log[record["log_name"]].append((index, record))

    def prepare_log(item):
        log_name, selected = item
        with (ROOT / "dataset/navsim_logs/test" / (log_name + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        lookup = {frame["token"]: index for index, frame in enumerate(frames)}
        for index, record in selected:
            current_index = lookup[record["token"]]
            assert current_index >= 3 and current_index + 8 < len(frames)
            observed = frames[current_index - 3:current_index + 1]
            current = observed[-1]
            for frame_index, frame in enumerate(observed):
                path = ROOT / "dataset/sensor_blobs/test" / frame["cams"]["CAM_F0"]["data_path"]
                with Image.open(path) as image:
                    assert image.size == (1920, 1080)
                    adapter_images[index, frame_index] = cv2.resize(np.asarray(image.convert("RGB"))[28:-28],
                        (128, 128), interpolation=cv2.INTER_AREA)
            adapter_ego[index] = np.r_[current["driving_command"], current["ego_dynamic_state"]]
            np.testing.assert_allclose(adapter_ego[index], np.r_[primary_ego[input_row[record["token"]], 7:11],
                                                               primary_ego[input_row[record["token"]], 3:7]], atol=1e-6)
            yaw = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
            origin = np.asarray(current["ego2global_translation"][:2])
            rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            for future_index, frame in enumerate(frames[current_index + 1:current_index + 9]):
                xy = (np.asarray(frame["ego2global_translation"][:2]) - origin) @ rotation
                delta = Quaternion(*frame["ego2global_rotation"]).yaw_pitch_roll[0] - yaw
                ground_truth[index, future_index] = [*xy, np.arctan2(np.sin(delta), np.cos(delta))]
            record["scene_type"] = "left_turn" if ground_truth[index, -1, 2] > .25 else "right_turn" if ground_truth[index, -1, 2] < -.25 else "straight"
            record["ego_speed_mps"] = float(np.linalg.norm(adapter_ego[index, 4:6]))
            record["observed_timestamps"] = [int(frame["timestamp"]) for frame in observed]
        return len(selected)

    completed = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        for count in pool.map(prepare_log, sorted(by_log.items())):
            completed += count
            write_json(output / "prepare_progress.json", {"completed_scenes": completed, "scene_count": len(tokens)})
    adapter_images.flush()
    np.save(inputs / "adapter_ego.npy", adapter_ego)
    np.save(inputs / "ground_truth_trajectory.npy", ground_truth)
    write_json(output / "scene_metadata.json", {"records": records,
        "by_scene_type": dict(Counter(record["scene_type"] for record in records)),
        "ground_truth_use": "Scoring and post-hoc scenario labels only; never passed into either model"})
    write_json(output / "inputs_complete.json", {"complete": True, "configuration_sha256": digest(configuration_path),
        "panel_sha256": digest(panel), "scene_count": len(tokens), "recording_count": len(grouped),
        "training_token_overlap": 0, "training_recording_overlap": 0,
        "files_sha256": {str(path.relative_to(output)): digest(path) for path in inputs.glob("*.npy")},
        "metadata_sha256": digest(output / "scene_metadata.json"),
        "preprocessing": {"primary": "Four current cameras, full-image128x128 BICUBIC, original11Dego",
                          "adapter": "Four historical/current front frames, crop28top/bottom,128x128 INTER_AREA, original8Dego"}})
    print(json.dumps(read_json(output / "inputs_complete.json")), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/lpwm_shared_navtest_comparison/epoch3.json")
    prepare(parser.parse_args().config.resolve())
