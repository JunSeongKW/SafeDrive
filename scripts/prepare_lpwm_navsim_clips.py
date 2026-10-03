"""Prepare read-only NAVSIM clips and evaluation annotations for the LPWM pilot."""
import hashlib
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.adapters.navsim_visual_entities import project_lidar_box_to_front_roi

ARTIFACT_ROOT = PROJECT_ROOT / "outputs/lpwm_navsim_adaptation_v1"


def projected_objects(frame):
    objects = []
    camera = frame["cams"]["CAM_F0"]
    for box, category, track in zip(frame["anns"]["gt_boxes"], frame["anns"]["gt_names"], frame["anns"]["track_tokens"]):
        if category not in ("vehicle", "pedestrian", "bicycle"):
            continue
        roi, valid = project_lidar_box_to_front_roi(box, camera)
        roi = roi * np.array([.25, .5, .25, .5])
        if valid and np.min(roi[2:] - roi[:2]) >= 3:
            camera_center = (box[:3] - camera["sensor2lidar_translation"]) @ np.linalg.inv(camera["sensor2lidar_rotation"]).T
            objects.append({"track": str(track), "category": str(category), "box": roi.tolist(), "depth": float(camera_center[2])})
    return objects


def overlap_proxy(objects):
    maximum_overlap = 0.0
    for farther in objects:
        farther_box = np.array(farther["box"])
        for nearer in objects:
            if nearer["depth"] >= farther["depth"] - 1:
                continue
            nearer_box = np.array(nearer["box"])
            intersection = np.maximum(0, np.minimum(farther_box[2:], nearer_box[2:]) - np.maximum(farther_box[:2], nearer_box[:2])).prod()
            maximum_overlap = max(maximum_overlap, intersection / np.prod(farther_box[2:] - farther_box[:2]))
    return float(maximum_overlap)


def clip_metadata(record, frames):
    yaw = np.unwrap([np.arctan2(frame["ego2global"][1, 0], frame["ego2global"][0, 0]) for frame in frames])
    yaw_change = float(np.rad2deg(np.max(yaw) - np.min(yaw)))
    objects_by_frame = [projected_objects(frame) for frame in frames]
    overlap = max(overlap_proxy(objects) for objects in objects_by_frame[2:7])
    if overlap >= .35:
        scenario = "projected_overlap"
    elif yaw_change >= 15:
        scenario = "turn"
    elif yaw_change <= 5 and record["ego_speed_meters_per_second"] >= 1:
        scenario = "straight"
    else:
        scenario = "other"
    return {**{key: record[key] for key in ("segment_filename", "start_index", "current_frame_token", "recording_group", "split")},
            "scenario": scenario, "yaw_range_degrees": yaw_change, "projected_overlap_fraction": overlap,
            "objects_by_frame": objects_by_frame, "ego_speed_meters_per_second": record["ego_speed_meters_per_second"]}


def prepare_clips():
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata_path = ARTIFACT_ROOT / "candidate_metadata.json"
    index_path = PROJECT_ROOT / "outputs/drive_jepa_selective_future/overnight_recording_coverage_v1_20261003/cache_index.json"
    if metadata_path.exists():
        candidates = json.loads(metadata_path.read_text())
    else:
        grouped_records = defaultdict(list)
        for record in json.loads(index_path.read_text())["records"]:
            grouped_records[record["segment_filename"]].append(record)
        candidates = []
        for filename, records in grouped_records.items():
            with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
                log_frames = pickle.load(stream)
            for record in records:
                frames = log_frames[record["start_index"]:record["start_index"] + 12]
                assert len(frames) == 12
                candidates.append(clip_metadata(record, frames))
        metadata_path.write_text(json.dumps(candidates))
    if any("available_first_eight_frames" not in record for record in candidates):
        grouped_candidates = defaultdict(list)
        for record in candidates:
            grouped_candidates[record["segment_filename"]].append(record)
        for filename, records in grouped_candidates.items():
            with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
                log_frames = pickle.load(stream)
            for record in records:
                record["available_first_eight_frames"] = all(
                    (PROJECT_ROOT / "dataset/sensor_blobs/trainval" / frame["cams"]["CAM_F0"]["data_path"]).is_file()
                    for frame in log_frames[record["start_index"]:record["start_index"] + 8])
        metadata_path.write_text(json.dumps(candidates))
    candidates = [record for record in candidates if record["available_first_eight_frames"]]
    print("candidate_counts", Counter((record["split"], record["scenario"]) for record in candidates), flush=True)
    selected_records = []
    for split, scenario_quota in (("train", 40), ("development", 12)):
        for scenario in ("straight", "turn", "projected_overlap"):
            eligible = [record for record in candidates if record["split"] == split and record["scenario"] == scenario]
            eligible.sort(key=lambda record: hashlib.sha256(("lpwm-selection-v1:" + record["current_frame_token"]).encode()).hexdigest())
            # Round-robin across recordings prevents a long log from filling a stratum.
            recording_counts = Counter()
            while eligible and sum(recording_counts.values()) < scenario_quota:
                chosen = min(eligible, key=lambda record: recording_counts[record["recording_group"]])
                eligible.remove(chosen)
                recording_counts[chosen["recording_group"]] += 1
                selected_records.append(chosen)
    train_recordings = {record["recording_group"] for record in selected_records if record["split"] == "train"}
    dev_recordings = {record["recording_group"] for record in selected_records if record["split"] == "development"}
    assert not train_recordings & dev_recordings
    clip_root = ARTIFACT_ROOT / "clips"
    clip_root.mkdir(exist_ok=True)
    for record in selected_records:
        output_path = clip_root / (record["current_frame_token"] + ".npz")
        record["clip_path"] = str(output_path.relative_to(PROJECT_ROOT))
        if output_path.exists():
            continue
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / record["segment_filename"]).open("rb") as stream:
            frames = pickle.load(stream)[record["start_index"]:record["start_index"] + 8]
        camera_rotations, camera_intrinsics, raw_images = [], [], []
        for frame in frames:
            camera = frame["cams"]["CAM_F0"]
            image_path = PROJECT_ROOT / "dataset/sensor_blobs/trainval" / camera["data_path"]
            with Image.open(image_path) as image:
                assert image.size == (1920, 1080)
                rgb_image = np.array(image.convert("RGB"))[28:-28]
            raw_images.append(cv2.resize(rgb_image, (128, 128), interpolation=cv2.INTER_AREA))
            camera_rotations.append(np.array(frame["ego2global"])[:3, :3] @ np.array(frame["lidar2ego"])[:3, :3] @ camera["sensor2lidar_rotation"])
            crop_resize = np.array([[128 / 1920, 0, 0], [0, 128 / 1024, -28 * 128 / 1024], [0, 0, 1]])
            camera_intrinsics.append(crop_resize @ camera["cam_intrinsic"])
        homographies = [camera_intrinsics[0] @ camera_rotations[0].T @ rotation @ np.linalg.inv(intrinsics)
                        for rotation, intrinsics in zip(camera_rotations, camera_intrinsics)]
        stabilized_images = [cv2.warpPerspective(image, transform, (128, 128), flags=cv2.INTER_LINEAR)
                             for image, transform in zip(raw_images, homographies)]
        np.savez_compressed(output_path, raw_images=np.array(raw_images), stabilized_images=np.array(stabilized_images),
                            observed_rotation_homographies=np.array(homographies), camera_rotations=np.array(camera_rotations),
                            camera_intrinsics=np.array(camera_intrinsics))
    manifest = {"selection_rule": "hash then round-robin recordings; thresholds fixed before any model evaluation",
                "source_index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                "raw_image_preprocessing": "1920x1080 -> crop 28 rows each side -> anisotropic resize 128x128",
                "occlusion_caveat": "projected box overlap with depth ordering is a proxy, not a visibility annotation",
                "train_recordings": len(train_recordings), "dev_recordings": len(dev_recordings),
                "counts": {f"{split}/{scenario}": count for (split, scenario), count in Counter((record["split"], record["scenario"]) for record in selected_records).items()},
                "records": selected_records}
    (ARTIFACT_ROOT / "clip_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print({key: value for key, value in manifest.items() if key != "records"}, flush=True)


if __name__ == "__main__":
    cv2.setNumThreads(1)
    prepare_clips()
