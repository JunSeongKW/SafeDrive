"""Seal a training-only panel with evaluation-only object and road proxies."""
import hashlib
import json
import lzma
import os
from pathlib import Path
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import shapely.vectorized

ROOT = Path(__file__).resolve().parents[1]
os.environ["NUPLAN_MAPS_ROOT"] = str(ROOT / "outputs/lpwm_drivor_joint_v1/maps")
sys.path.insert(0, str(ROOT / "reference_repositories/DrivoR"))
sys.path.insert(0, str(ROOT / "src"))
from planning_aware_future_prediction.object_centric.particle_planning_diagnostics import project_box_full_image
from planning_aware_future_prediction.adapters.navsim_tracked_state import annotation_states_in_current_ego_frame
from nuplan.common.maps.maps_datatypes import SemanticMapLayer

OUTPUT = ROOT / "outputs/lpwm_drivor_representation_monitor_v1"
CAMERAS = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
CATEGORIES = ("vehicle", "pedestrian", "bicycle")


def ordering(value):
    return hashlib.sha256(("particle-monitor-71/" + str(value)).encode()).hexdigest()


def road_projection(frame, camera, image_size, cache, ground_height):
    vertical, horizontal = np.meshgrid(np.arange(128)+.5, np.arange(128)+.5, indexing="ij")
    pixels = np.stack((horizontal*image_size[0]/128, vertical*image_size[1]/128, np.ones_like(vertical)), -1)
    rays = pixels @ np.linalg.inv(camera["cam_intrinsic"]).T @ np.asarray(camera["sensor2lidar_rotation"]).T
    origin = np.asarray(camera["sensor2lidar_translation"])
    distance = (ground_height-origin[2]) / np.where(np.abs(rays[..., 2]) > 1e-8, rays[..., 2], np.nan)
    lidar_points = origin + rays*distance[..., None]
    valid = np.isfinite(lidar_points).all(-1) & (distance > 0) & (np.linalg.norm(lidar_points[..., :2], axis=-1) < 80)
    ego_points = lidar_points @ Quaternion(frame["lidar2ego_rotation"]).rotation_matrix.T + frame["lidar2ego_translation"]
    global_points = ego_points @ Quaternion(frame["ego2global_rotation"]).rotation_matrix.T + frame["ego2global_translation"]
    mask = np.zeros((128, 128), bool)
    layers = {SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA}
    for layer, polygon in zip(cache.drivable_area_map._map_types, cache.drivable_area_map._geometries):
        if layer in layers:
            mask |= valid & shapely.vectorized.contains(polygon, global_points[..., 0], global_points[..., 1])
    return mask


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not (OUTPUT / "panel.json").exists(), "Existing fixed panel must not be resampled"
    manifest_path = ROOT / "outputs/lpwm_drivor_joint_v1/scene_cache/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    by_group = defaultdict(list)
    for record in manifest["records"]:
        by_group[record["recording_group"]].append(record)
    groups = sorted(by_group, key=ordering)[:24]
    probe_evaluation_groups = set(sorted(groups, key=lambda group: ordering("probe/"+group))[:6])
    selected = []
    for group in groups:
        candidates = sorted(by_group[group], key=lambda record: ordering(record["token"]))
        chosen, kinds = [], set()
        for record in candidates:
            trajectory = np.load(Path(record["cache_directory"])/"trajectory.npy", mmap_mode="r")[record["cache_row"]]
            kind = "left_turn" if trajectory[-1, 2] > .25 else "right_turn" if trajectory[-1, 2] < -.25 else "straight"
            if kind not in kinds:
                chosen.append(dict(record, scene_type=kind)); kinds.add(kind)
        used = {record["token"] for record in chosen}
        for record in candidates:
            if len(chosen) >= 4:
                break
            if record["token"] not in used:
                trajectory = np.load(Path(record["cache_directory"])/"trajectory.npy", mmap_mode="r")[record["cache_row"]]
                kind = "left_turn" if trajectory[-1, 2] > .25 else "right_turn" if trajectory[-1, 2] < -.25 else "straight"
                chosen.append(dict(record, scene_type=kind)); used.add(record["token"])
        assert len(chosen) == 4
        selected.extend(chosen)
    selected.sort(key=lambda record: (record["log_name"], record["current_frame_index"]))
    cache_rows = json.loads((ROOT/"outputs/lpwm_candidate_teacher_v1/metric_cache_manifest.json").read_text())
    cache_paths = {record["token"]: record["metric_cache_file"] for record in cache_rows}
    images, road_masks, records, objects = [], [], [], []
    previous_log, frames = None, None
    for scene_index, record in enumerate(selected):
        if previous_log != record["log_name"]:
            with (ROOT/"dataset/navsim_logs/trainval"/(record["log_name"]+".pkl")).open("rb") as stream:
                frames = pickle.load(stream)
            previous_log = record["log_name"]
        current_index = record["current_frame_index"]
        current = frames[current_index]
        current_states = annotation_states_in_current_ego_frame(current, current)
        future_states = [annotation_states_in_current_ego_frame(frames[current_index+offset], current) for offset in (4, 8)]
        future_lookup = [{str(track): index for index, track in enumerate(frames[current_index+offset]["anns"]["track_tokens"])} for offset in (4, 8)]
        expert = np.load(Path(record["cache_directory"])/"trajectory.npy", mmap_mode="r")[record["cache_row"]]
        rgb = np.array(np.load(Path(record["cache_directory"])/"images.npy", mmap_mode="r")[record["cache_row"]])
        images.append(rgb)
        cache = None
        if record["token"] in cache_paths:
            with lzma.open(cache_paths[record["token"]], "rb") as stream:
                cache = pickle.load(stream)
        boxes = np.asarray(current["anns"]["gt_boxes"])
        near_ground = (np.linalg.norm(boxes[:, :2], axis=-1) < 40) & np.isin(current["anns"]["gt_names"], CATEGORIES)
        ground_height = float(np.clip(np.median(boxes[near_ground, 2]-boxes[near_ground, 5]/2), -3, 1)) if near_ground.any() else 0.
        scene_masks = []
        object_start = len(objects)
        camera_overlaps = []
        for camera_index, camera_name in enumerate(CAMERAS):
            camera = current["cams"][camera_name]
            with Image.open(record["current_camera_paths"][camera_index]) as source:
                image_size = source.size
            occupied = np.zeros((128, 128), bool)
            projected_boxes = []
            for annotation_index, (box, category, track) in enumerate(zip(boxes, current["anns"]["gt_names"], current["anns"]["track_tokens"])):
                rectangle = project_box_full_image(box, camera, image_size)
                if rectangle is None:
                    continue
                lower = np.floor(rectangle[:2]).astype(int); upper = np.ceil(rectangle[2:]).astype(int)
                occupied[lower[1]:upper[1], lower[0]:upper[0]] = True
                if category not in CATEGORIES:
                    continue
                projected_boxes.append(rectangle)
                targets, future_valid = [], []
                risk = np.min(np.linalg.norm(expert[:, :2]-current_states[annotation_index, :2], axis=-1)) < 5
                for horizon_index, offset in enumerate((4, 8)):
                    position = future_lookup[horizon_index].get(str(track))
                    future_valid.append(position is not None)
                    targets.append((future_states[horizon_index][position, :2]-current_states[annotation_index, :2]).tolist() if position is not None else [0., 0.])
                    if position is not None:
                        risk = risk or np.linalg.norm(future_states[horizon_index][position, :2]-expert[offset-1, :2]) < 5
                objects.append({"scene_index": scene_index, "camera_index": camera_index, "track": str(track),
                    "category": str(category), "box": rectangle.tolist(), "state": current_states[annotation_index, [0, 1, 6, 7]].tolist(),
                    "future_displacement": targets, "future_valid": future_valid, "near_expert_corridor_proxy": bool(risk),
                    "small_box": bool(np.prod(rectangle[2:]-rectangle[:2]) < 64),
                    "far_object": bool(np.linalg.norm(current_states[annotation_index, :2]) >= 30),
                    "recording_group": record["recording_group"],
                    "probe_partition": "readout_evaluation" if record["recording_group"] in probe_evaluation_groups else "readout_fit"})
            overlap = False
            for index, box in enumerate(projected_boxes):
                for other in projected_boxes[index+1:]:
                    overlap |= bool(np.maximum(np.minimum(box[2:], other[2:])-np.maximum(box[:2], other[:2]), 0).prod() > 0)
            camera_overlaps.append(overlap)
            masks = [road_projection(current, camera, image_size, cache, ground_height+delta) & ~occupied
                     if cache is not None else np.zeros((128, 128), bool) for delta in (-.5, 0, .5)]
            scene_masks.append(masks)
        road_masks.append(scene_masks)
        records.append(dict(record, command=np.asarray(current["driving_command"]).tolist(),
            object_start=object_start, object_end=len(objects), road_proxy_available=cache is not None,
            ground_height_m=ground_height, projected_overlap_proxy=any(camera_overlaps)))
        if scene_index % 16 == 0:
            print("PANEL", scene_index+1, len(selected), len(objects), flush=True)
    np.save(OUTPUT/"images.npy", np.asarray(images))
    np.save(OUTPUT/"road_proxy_masks.npy", np.asarray(road_masks))
    payload = {"records": records, "objects": objects, "scene_count": len(records), "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "probe_evaluation_groups": sorted(probe_evaluation_groups), "categories": CATEGORIES,
        "counts": dict(Counter(obj["category"] for obj in objects)), "scene_types": dict(Counter(record["scene_type"] for record in records)),
        "scope": "Fixed training-distribution diagnostic panel, never an independent planning validation/test set. GT used only by evaluation diagnostics.",
        "road_proxy": "Map drivable area intersected with camera rays at estimated flat ground, excluding projected annotation boxes; heights -0.5/0/+0.5m. Not pixelwise road segmentation.",
        "projection": "Full RGB resize 128x128, no crop; pinhole convention. Visibility/occlusion not guaranteed; overlap is a proxy.",
        "future_targets": "GT track-associated displacement at 2/4s in the current ego frame; never a model input."}
    (OUTPUT/"panel.json").write_text(json.dumps(payload, indent=2)+"\n")
    (OUTPUT/"oracle_manifest.json").write_text(json.dumps({"records": records}, indent=2)+"\n")
    print(json.dumps({"scenes": len(records), "objects": len(objects), "counts": payload["counts"]}), flush=True)


if __name__ == "__main__":
    main()
