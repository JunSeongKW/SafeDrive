"""Observed RGB inputs and separate privileged supervision for LPWM planning."""
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.adapters.navsim_visual_entities import project_lidar_box_to_front_roi
from planning_aware_future_prediction.object_centric.lpwm_bridge import checkpoint_digest

OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"
CONFIG_PATH = PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json"


def transform_box_center_to_current_ego(box, frame, current_frame):
    point_global = np.asarray(frame["ego2global"]) @ np.asarray(frame["lidar2ego"]) @ np.r_[box[:3], 1.]
    return (np.linalg.inv(np.asarray(current_frame["ego2global"])) @ point_global)[:2]


def object_supervision(frames, ego_trajectory, maximum_objects=32):
    current_frame = frames[3]
    camera = current_frame["cams"]["CAM_F0"]
    categories = {"vehicle": 1, "pedestrian": 2, "bicycle": 3}
    candidates = []
    for box, name, track in zip(current_frame["anns"]["gt_boxes"], current_frame["anns"]["gt_names"], current_frame["anns"]["track_tokens"]):
        if name not in categories:
            continue
        pixel_box, valid = project_lidar_box_to_front_roi(box, camera)
        pixel_box = pixel_box * np.array([.25, .5, .25, .5])
        if not valid or np.min(pixel_box[2:] - pixel_box[:2]) < 3:
            continue
        current_position = transform_box_center_to_current_ego(box, current_frame, current_frame)
        future_positions = np.zeros((8, 2), dtype=np.float32)
        future_valid = np.zeros(8, dtype=bool)
        for future_index, frame in enumerate(frames[4:12]):
            track_lookup = {str(token): index for index, token in enumerate(frame["anns"]["track_tokens"])}
            if str(track) in track_lookup:
                future_positions[future_index] = transform_box_center_to_current_ego(frame["anns"]["gt_boxes"][track_lookup[str(track)]], frame, current_frame)
                future_valid[future_index] = True
        distance = np.linalg.norm(future_positions - ego_trajectory[:, :2], axis=-1)
        minimum_distance = float(distance[future_valid].min()) if future_valid.any() else float(np.linalg.norm(current_position))
        candidates.append((float(np.linalg.norm(current_position)), str(track), pixel_box, categories[name], current_position, future_positions, future_valid, 1 + 4 * np.exp(-minimum_distance**2 / 50)))
    candidates.sort(key=lambda entry: (entry[0], entry[1]))
    result = {"object_boxes": np.zeros((maximum_objects, 4), np.float32), "object_categories": np.zeros(maximum_objects, np.int64),
              "object_current_xy": np.zeros((maximum_objects, 2), np.float32), "object_future_xy": np.zeros((maximum_objects, 8, 2), np.float32),
              "object_future_valid": np.zeros((maximum_objects, 8), bool), "object_valid": np.zeros(maximum_objects, bool),
              "object_risk_weights": np.ones(maximum_objects, np.float32)}
    for object_index, (_, _, pixel_box, category, current_xy, future_xy, future_valid, risk) in enumerate(candidates[:maximum_objects]):
        result["object_boxes"][object_index] = pixel_box / 128
        result["object_categories"][object_index] = category
        result["object_current_xy"][object_index] = current_xy
        result["object_future_xy"][object_index] = future_xy
        result["object_future_valid"][object_index] = future_valid
        result["object_valid"][object_index] = True
        result["object_risk_weights"][object_index] = risk
    return result, len(candidates)


def main():
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    specification = json.loads(CONFIG_PATH.read_text())
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT_ROOT / "supervised_cache.pt"
    if destination.exists():
        raise FileExistsError("Prepared experiment data are immutable")
    source = json.loads((PROJECT_ROOT / specification["source_index"]).read_text())
    metadata = {row["current_frame_token"]: row for row in json.loads((PROJECT_ROOT / "outputs/lpwm_navsim_adaptation_v1/candidate_metadata.json").read_text())}
    by_segment = defaultdict(list)
    for record in source["records"]:
        by_segment[record["segment_filename"]].append(record)
    prepared, tensor_rows, skipped = [], [], []
    for filename, records in by_segment.items():
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
            log_frames = pickle.load(stream)
        for record in records:
            frames = log_frames[record["start_index"]:record["start_index"] + 12]
            assert frames[3]["token"] == record["current_frame_token"]
            image_paths = [PROJECT_ROOT / "dataset/sensor_blobs/trainval" / frames[index]["cams"]["CAM_F0"]["data_path"] for index in (2, 3)]
            if not all(path.is_file() for path in image_paths):
                skipped.append(record["current_frame_token"])
                continue
            observed_images = []
            for path in image_paths:
                with Image.open(path) as image:
                    assert image.size == (1920, 1080)
                    observed_images.append(cv2.resize(np.asarray(image.convert("RGB"))[28:-28], (128, 128), interpolation=cv2.INTER_AREA))
            # These local caches were produced by the official feature/target builder.
            cached = torch.load(record["cache_file"], map_location="cpu", weights_only=True, mmap=True)
            status = cached["current_ego_status"].numpy().copy()
            trajectory = cached["ego_trajectory_target"].numpy().copy()
            np.testing.assert_allclose(status[4:], frames[3]["ego_dynamic_state"], atol=1e-6)
            supervision, uncapped_objects = object_supervision(frames, trajectory, specification["maximum_objects"])
            tensor_rows.append({"observed_images": np.array(observed_images), "ego_status": status, "ego_trajectory_target": trajectory, **supervision})
            meta = metadata[record["current_frame_token"]]
            prepared.append({**record, "scenario": meta["scenario"], "yaw_range_degrees": meta["yaw_range_degrees"],
                             "projected_overlap_fraction": meta["projected_overlap_fraction"], "visible_object_count": int(supervision["object_valid"].sum()),
                             "uncapped_visible_object_count": uncapped_objects, "observed_image_paths": [str(path) for path in image_paths]})
        if len(prepared) % 32 < len(records):
            print("PREPARED", len(prepared), flush=True)
    assert not ({row["recording_group"] for row in prepared if row["split"] == "train"} & {row["recording_group"] for row in prepared if row["split"] == "development"})
    tensors = {key: torch.from_numpy(np.stack([row[key] for row in tensor_rows])) for key in tensor_rows[0]}
    torch.save(tensors, destination)
    manifest = {"records": prepared, "skipped_missing_observed_images": skipped,
                "counts": dict(Counter(row["split"] for row in prepared)), "scenario_counts": dict(Counter(row["split"] + "/" + row["scenario"] for row in prepared)),
                "source_index_sha256": checkpoint_digest(PROJECT_ROOT / specification["source_index"]),
                "cache_sha256": checkpoint_digest(destination), "configuration_sha256": checkpoint_digest(CONFIG_PATH),
                "causality": "Only frames2,3 RGB and current8D status are inference inputs. All boxes/tracks/future poses/trajectories are supervision only.",
                "scope": "Existing recording-disjoint development split, not independent test"}
    (OUTPUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("DATA_READY", manifest["counts"], manifest["scenario_counts"], flush=True)


if __name__ == "__main__":
    main()
