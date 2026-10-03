"""Prepare strict-navtrain RGB sequences; keep annotations out of training inputs."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from prepare_lpwm_navsim_clips import projected_objects, overlap_proxy


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_recording_balanced(records, maximum_clips):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["recording_group"]].append(record)
    for group_records in grouped.values():
        group_records.sort(key=lambda row: hashlib.sha256(("lpwm-posttraining-v1:" + row["current_frame_token"]).encode()).hexdigest())
    ordered_groups = sorted(grouped, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    selected = []
    for row_index in range(max((len(rows) for rows in grouped.values()), default=0)):
        for group_name in ordered_groups:
            if row_index < len(grouped[group_name]):
                selected.append(grouped[group_name][row_index])
                if len(selected) == maximum_clips:
                    return selected
    return selected


def main(arguments):
    started = time.monotonic()
    cv2.setNumThreads(1)
    specification = json.loads((PROJECT_ROOT / arguments.config).read_text())
    output_root = PROJECT_ROOT / specification["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "manifest.json").exists():
        print("DATA_ALREADY_READY", flush=True)
        return
    filter_path = PROJECT_ROOT / specification["data"]["official_scene_filter"]
    official_filter = yaml.safe_load(filter_path.read_text())
    allowed_logs, allowed_tokens = set(official_filter["log_names"]), set(official_filter["tokens"])
    assignment_path = PROJECT_ROOT / specification["data"]["recording_assignment"]
    assignments = json.loads(assignment_path.read_text())["assignments"]["split_by_recording"]
    assignment_policy = specification["data"].get("assignment_policy", "existing_train_development_only")
    candidate_path = output_root / "eligible_sequences.json"
    if candidate_path.exists():
        survey = json.loads(candidate_path.read_text())
        assert survey["official_filter_sha256"] == file_digest(filter_path)
        assert survey["assignment_sha256"] == file_digest(assignment_path)
        assert survey["sequence_frames"] == specification["data"]["sequence_frames"]
        assert survey.get("assignment_policy", "existing_train_development_only") == assignment_policy
    else:
        candidates, planning_candidates, source_hashes, rejected = [], [], {}, Counter()
        image_available = {}
        for segment_index, log_name in enumerate(sorted(allowed_logs)):
            recording = "_".join(log_name.split("_")[:-2])
            split = assignments.get(recording)
            if assignment_policy == "all_navtrain_except_existing_development_recordings":
                split = "development" if split == "development" else "train"
            if split not in ("train", "development"):
                continue
            log_path = PROJECT_ROOT / "dataset/navsim_logs/trainval" / (log_name + ".pkl")
            if not log_path.exists():
                rejected["missing_log"] += 1
                continue
            source_hashes[log_path.name] = file_digest(log_path)
            with log_path.open("rb") as stream:
                frames = pickle.load(stream)
            for current_index in range(3, len(frames) - 10):
                current = frames[current_index]
                if current["token"] not in allowed_tokens or not len(current["roadblock_ids"]):
                    continue
                sequence = frames[current_index - 3:current_index + 9]
                image_paths = [frame["cams"]["CAM_F0"]["data_path"] for frame in sequence]
                for image_path in image_paths:
                    if image_path not in image_available:
                        image_available[image_path] = (PROJECT_ROOT / "dataset/sensor_blobs/trainval" / image_path).is_file()
                common_record = {"segment_filename": log_path.name, "start_index": current_index - 3,
                    "current_frame_token": str(current["token"]), "recording_group": recording, "split": split,
                    "ego_speed_meters_per_second": float(np.linalg.norm(current["ego_dynamic_state"][:2])),
                    "image_paths": image_paths}
                if all(image_available[path] for path in image_paths[:4]):
                    planning_candidates.append({**common_record, "image_available": [image_available[path] for path in image_paths]})
                else:
                    rejected[split + "/missing_observed_RGB"] += 1
                if not all(image_available[path] for path in image_paths):
                    rejected[split + "/missing_sequence_RGB"] += 1
                    continue
                time_intervals = np.diff([frame["timestamp"] for frame in sequence]) / 1e6
                if not np.allclose(time_intervals, .5, atol=.1):
                    rejected[split + "/irregular_timestamps"] += 1
                    continue
                candidates.append(common_record)
            if segment_index % 50 == 0:
                print("SURVEY", segment_index, dict(Counter(row["split"] for row in candidates)), flush=True)
        survey = {"records": candidates, "source_log_sha256": source_hashes,
            "official_filter_sha256": file_digest(filter_path), "assignment_sha256": file_digest(assignment_path),
            "sequence_frames": specification["data"]["sequence_frames"], "rejected": dict(rejected),
            "assignment_policy": assignment_policy, "planning_records": planning_candidates}
        candidate_path.write_text(json.dumps(survey))
    selected = []
    for split, count_key in (("train", "train_clips"), ("development", "development_clips")):
        eligible = [row for row in survey["records"] if row["split"] == split]
        requested_count = specification["data"][count_key]
        if requested_count is None:
            requested_count = len(eligible)
        if len(eligible) < requested_count:
            raise RuntimeError(f"Only {len(eligible)} eligible {split} clips, requested {requested_count}; revise and register actual data size explicitly")
        selected.extend(select_recording_balanced(eligible, requested_count))
    train_recordings = {row["recording_group"] for row in selected if row["split"] == "train"}
    development_recordings = {row["recording_group"] for row in selected if row["split"] == "development"}
    assert not train_recordings & development_recordings
    assert len({row["current_frame_token"] for row in selected}) == len(selected)
    # Labels are produced for reporting only; the training dataset reads RGB indices exclusively.
    by_segment = defaultdict(list)
    for row in selected:
        by_segment[row["segment_filename"]].append(row)
    for filename, rows in by_segment.items():
        if all(row["split"] == "train" for row in rows):
            for row in rows:
                row["scenario"] = "not_annotated_for_unsupervised_training"
            continue
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
            frames = pickle.load(stream)
        projected_by_frame = {}
        for row in rows:
            sequence = frames[row["start_index"]:row["start_index"] + 12]
            yaw = np.unwrap([np.arctan2(frame["ego2global"][1, 0], frame["ego2global"][0, 0]) for frame in sequence])
            yaw_change = float(np.rad2deg(np.max(yaw) - np.min(yaw)))
            for frame in sequence:
                if frame["token"] not in projected_by_frame:
                    projected_by_frame[frame["token"]] = projected_objects(frame)
            objects_by_frame = [projected_by_frame[frame["token"]] for frame in sequence]
            overlap = max(overlap_proxy(objects) for objects in objects_by_frame[2:7])
            scenario = "projected_overlap" if overlap >= .35 else "turn" if yaw_change >= 15 else "straight" if yaw_change <= 5 and row["ego_speed_meters_per_second"] >= 1 else "other"
            row.update(scenario=scenario, yaw_range_degrees=yaw_change, projected_overlap_fraction=overlap, objects_by_frame=objects_by_frame)
        print("DEVELOPMENT_LABELS", filename, len(rows), flush=True)
    unique_images = sorted({path for row in selected for path in row["image_paths"]})
    planning_records = survey.get("planning_records", []) if specification["data"].get("prepare_planning_records") else []
    if planning_records:
        unique_images = sorted(set(unique_images) | {path for row in planning_records for path, available in zip(row["image_paths"], row["image_available"]) if available})
    image_index = {path: index for index, path in enumerate(unique_images)}
    frame_cache_path = output_root / "rgb_frames.npy"
    frame_cache = np.lib.format.open_memmap(frame_cache_path, mode="w+", dtype=np.uint8,
        shape=(len(unique_images), 128, 128, 3))
    reusable_cache, reusable_indices = None, {}
    if specification["data"].get("reuse_rgb_cache"):
        reusable_root = PROJECT_ROOT / specification["data"]["reuse_rgb_cache"]
        reusable_manifest = json.loads((reusable_root / "manifest.json").read_text())
        reusable_cache = np.load(reusable_root / "rgb_frames.npy", mmap_mode="r")
        reusable_indices = {path: index for index, path in enumerate(reusable_manifest["unique_image_paths"])}
    def resize_image(index_and_path):
        frame_index, image_path = index_and_path
        if image_path in reusable_indices:
            frame_cache[frame_index] = reusable_cache[reusable_indices[image_path]]
            return frame_index
        with Image.open(PROJECT_ROOT / "dataset/sensor_blobs/trainval" / image_path) as image:
            if image.size != (1920, 1080):
                raise ValueError(f"Unexpected camera resolution: {image.size}")
            resized = cv2.resize(np.asarray(image.convert("RGB"))[28:-28], (128, 128), interpolation=cv2.INTER_AREA)
        frame_cache[frame_index] = resized
        return frame_index
    with ThreadPoolExecutor(max_workers=8) as executor:
        for completed, _ in enumerate(executor.map(resize_image, enumerate(unique_images)), start=1):
            if completed % 1000 == 0:
                print("RGB_CACHE", completed, len(unique_images), flush=True)
    frame_cache.flush()
    for row in selected:
        row["frame_cache_indices"] = [image_index[path] for path in row.pop("image_paths")]
    if planning_records:
        for row in planning_records:
            row["frame_cache_indices"] = [image_index.get(path, -1) for path in row.pop("image_paths")]
        (output_root / "planning_records.json").write_text(json.dumps({"records": planning_records,
            "counts": dict(Counter(row["split"] for row in planning_records)), "input": "only first four RGB frames and current ego status; future frames are world-model targets only"}))
    manifest = {"records": selected, "unique_image_paths": unique_images,
        "counts": dict(Counter(row["split"] for row in selected)),
        "recording_counts": {"train": len(train_recordings), "development": len(development_recordings)},
        "scenario_counts": dict(Counter(row["split"] + "/" + row["scenario"] for row in selected)),
        "official_filter_sha256": survey["official_filter_sha256"], "assignment_sha256": survey["assignment_sha256"],
        "candidate_counts": dict(Counter(row["split"] for row in survey["records"])),
        "source_log_sha256": survey["source_log_sha256"], "frame_cache_sha256": file_digest(frame_cache_path),
        "configuration_sha256": file_digest(PROJECT_ROOT / arguments.config),
        "unique_rgb_frames": len(unique_images), "cache_bytes": frame_cache_path.stat().st_size,
        "source_data_writes": False, "official_navtest_usage": False,
        "previous_navtrain_heldout_now_training": assignment_policy == "all_navtrain_except_existing_development_recordings",
        "assignment_policy": assignment_policy, "rejected": survey["rejected"],
        "development_is_independent_test": False,
        "preprocessing": "front camera 1920x1080, crop 28 top/bottom, resize 128x128, RGB uint8",
        "selection_caveat": "Complete 12-frame RGB availability required for video post-training; projected overlap is an occlusion proxy only",
        "seconds": time.monotonic() - started}
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DATA_READY", {key: manifest[key] for key in ("counts", "recording_counts", "scenario_counts", "unique_rgb_frames", "seconds")}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lpwm_navsim_adaptation/posttraining_v1.json")
    main(parser.parse_args())
