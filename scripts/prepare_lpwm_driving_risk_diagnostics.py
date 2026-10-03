"""Evaluation-only annotations for ego motion, disappearance and small/far objects."""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.adapters.navsim_visual_entities import project_lidar_box_to_front_roi


def projected_objects_including_small(frame, thresholds):
    camera = frame["cams"]["CAM_F0"]
    objects = []
    for box, category, track in zip(frame["anns"]["gt_boxes"], frame["anns"]["gt_names"], frame["anns"]["track_tokens"]):
        if category not in ("vehicle", "pedestrian", "bicycle"):
            continue
        pixel_box, valid = project_lidar_box_to_front_roi(box, camera)
        pixel_box = pixel_box * np.array([.25, .5, .25, .5])
        size = pixel_box[2:] - pixel_box[:2]
        if not valid or size.min() < thresholds["minimum_projected_object_side_pixels"]:
            continue
        camera_center = (box[:3] - camera["sensor2lidar_translation"]) @ np.linalg.inv(camera["sensor2lidar_rotation"]).T
        objects.append({"track": str(track), "category": str(category), "box": pixel_box.tolist(),
            "depth": float(camera_center[2]),
            "small": bool(size.prod() < thresholds["small_projected_box_area_pixels"]),
            "far": bool(camera_center[2] >= thresholds["far_object_camera_depth_meters"])})
    return objects


def main(config_path):
    cv2.setNumThreads(1)
    specification = json.loads(config_path.read_text())
    risk_config_path = PROJECT_ROOT / "configs/lpwm_navsim_adaptation/driving_risks_v1.json"
    thresholds = json.loads(risk_config_path.read_text())
    output_root = PROJECT_ROOT / specification["output_directory"]
    destination = output_root / "driving_risk_metadata.json"
    if destination.exists():
        print("RISK_METADATA_ALREADY_READY", flush=True)
        return
    manifest = json.loads((output_root / "manifest.json").read_text())
    frame_cache = np.load(output_root / "rgb_frames.npy", mmap_mode="r")
    by_segment = defaultdict(list)
    for record in manifest["records"]:
        if record["split"] == "development":
            by_segment[record["segment_filename"]].append(record)
    diagnostics, projected_cache = {}, {}
    for segment_index, (filename, records) in enumerate(by_segment.items()):
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
            frames = pickle.load(stream)
        for record in records:
            sequence = frames[record["start_index"]:record["start_index"] + 12]
            objects_by_frame = []
            for frame in sequence:
                if frame["token"] not in projected_cache:
                    projected_cache[frame["token"]] = projected_objects_including_small(frame, thresholds)
                objects_by_frame.append(projected_cache[frame["token"]])
            images = frame_cache[record["frame_cache_indices"]]
            gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images]
            median_flows = []
            for previous, current in zip(gray_images[:-1], gray_images[1:]):
                flow = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 3, 15, 3, 5, 1.2, 0)
                median_flows.append(float(np.median(np.linalg.norm(flow, axis=-1))))
            current_tracks = {obj["track"] for obj in objects_by_frame[3]}
            disappeared = current_tracks - {obj["track"] for obj in objects_by_frame[-1]}
            future_objects = [obj for frame_objects in objects_by_frame[4:] for obj in frame_objects]
            flags = {
                "high_ego_speed": record["ego_speed_meters_per_second"] >= thresholds["high_ego_speed_mps"],
                "large_ego_turn": record["yaw_range_degrees"] >= thresholds["large_yaw_range_degrees"],
                "projected_overlap": record["projected_overlap_fraction"] >= .35,
                "projected_track_disappearance": bool(disappeared),
                "small_future_objects": any(obj["small"] for obj in future_objects),
                "far_future_objects": any(obj["far"] for obj in future_objects),
                "high_interframe_flow": np.mean(median_flows) >= thresholds["high_median_flow_pixels_per_frame"],
            }
            diagnostics[record["current_frame_token"]] = {"risk_flags": {key: bool(value) for key, value in flags.items()},
                "objects_by_frame": objects_by_frame, "mean_median_flow_pixels": float(np.mean(median_flows)),
                "median_flow_pixels_by_transition": median_flows,
                "current_tracks_missing_from_last_projected_frame": len(disappeared),
                "current_projected_tracks": len(current_tracks),
                "frame_intervals_seconds": (np.diff([frame["timestamp"] for frame in sequence]) / 1e6).tolist()}
        if segment_index % 30 == 0:
            print("RISK_METADATA", len(diagnostics), flush=True)
    summary = {"records_by_token": diagnostics, "risk_counts": dict(Counter(name for row in diagnostics.values()
        for name, enabled in row["risk_flags"].items() if enabled)),
        "risk_configuration_sha256": hashlib.sha256(risk_config_path.read_bytes()).hexdigest(),
        "purpose": "evaluation only; never used for model input, training sampling or loss"}
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print("RISK_METADATA_READY", summary["risk_counts"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/posttraining_v1.json")
    arguments = parser.parse_args()
    main(arguments.config.resolve())
