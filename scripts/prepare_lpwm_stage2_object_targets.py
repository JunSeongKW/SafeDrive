"""Read-only NAVSIM labels for train-only current/causal-future supervision.

GT boxes, track identities and future ego poses are loss-side targets only.
All currently projected annotated objects are retained; no particle-count cap.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate_lpwm_full_planning import digest, write_json


def prepare_segment(arguments):
    filename, indexed_records, destination, categories, minimum_side = arguments
    from planning_aware_future_prediction.adapters.navsim_tracked_state import (
        annotation_states_in_current_ego_frame, stable_track_identity)
    from planning_aware_future_prediction.adapters.navsim_visual_entities import project_lidar_box_to_front_roi
    import torch
    torch.set_num_threads(1)
    destination = Path(destination)
    if destination.exists():
        return str(destination)
    with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / filename).open("rb") as stream:
        frames = pickle.load(stream)
    projection_cache = {}
    record_indices, offsets = [], [0]
    all_boxes, all_states, all_valid, all_categories, all_tracks = [], [], [], [], []
    for record_index, record in indexed_records:
        sequence = frames[record["start_index"] + 3:record["start_index"] + 12]
        assert len(sequence) == 9 and sequence[0]["token"] == record["current_frame_token"]
        projected_sequence, state_sequence = [], []
        for frame in sequence:
            if frame["token"] not in projection_cache:
                projected = {}
                for annotation_index, (box, category, track) in enumerate(zip(
                        frame["anns"]["gt_boxes"], frame["anns"]["gt_names"], frame["anns"]["track_tokens"])):
                    if category not in categories:
                        continue
                    roi, valid = project_lidar_box_to_front_roi(box, frame["cams"]["CAM_F0"])
                    roi = roi * np.array([.25, .5, .25, .5])
                    if valid and np.isfinite(roi).all() and np.min(roi[2:] - roi[:2]) >= minimum_side:
                        projected[str(track)] = (annotation_index, roi.astype(np.float32), categories.index(category))
                projection_cache[frame["token"]] = projected
            projected_sequence.append(projection_cache[frame["token"]])
            state_sequence.append(annotation_states_in_current_ego_frame(frame, sequence[0])[:, [0, 1, 6, 7]])
        tracks = sorted(projected_sequence[0])
        boxes = np.zeros((len(tracks), 9, 4), np.float32)
        states = np.zeros((len(tracks), 9, 4), np.float32)
        valid = np.zeros((len(tracks), 9), bool)
        for object_index, track in enumerate(tracks):
            for time_index, (projected, current_ego_states) in enumerate(zip(projected_sequence, state_sequence)):
                if track in projected:
                    annotation_index, roi, _ = projected[track]
                    state = current_ego_states[annotation_index]
                    if np.isfinite(state).all():
                        boxes[object_index, time_index] = roi
                        states[object_index, time_index] = state
                        valid[object_index, time_index] = True
        record_indices.append(record_index)
        offsets.append(offsets[-1] + len(tracks))
        all_boxes.append(boxes)
        all_states.append(states)
        all_valid.append(valid)
        all_categories.extend(projected_sequence[0][track][2] for track in tracks)
        all_tracks.extend(stable_track_identity(track) for track in tracks)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.npz")
    np.savez(pending, record_indices=np.asarray(record_indices, np.int64), object_offsets=np.asarray(offsets, np.int64),
        projected_boxes=np.concatenate(all_boxes), current_ego_states=np.concatenate(all_states),
        state_valid=np.concatenate(all_valid), categories=np.asarray(all_categories, np.int16),
        track_identities=np.asarray(all_tracks, np.int64))
    pending.replace(destination)
    return str(destination)


def prepare(config_path):
    specification = json.loads(config_path.read_text())
    auxiliary = specification["object_auxiliary"]
    destination = PROJECT_ROOT / auxiliary["target_directory"]
    destination.mkdir(parents=True, exist_ok=True)
    stage1_specification = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    manifest_path = PROJECT_ROOT / stage1_specification["output_directory"] / "planning_manifest.json"
    records = json.loads(manifest_path.read_text())["records"]
    identity = {"manifest_sha256": digest(manifest_path), "source_sha256": digest(__file__),
        "categories": auxiliary["categories"], "minimum_side_pixels": auxiliary["minimum_projected_side_pixels"]}
    if (destination / "identity.json").exists():
        assert json.loads((destination / "identity.json").read_text()) == identity
    else:
        write_json(destination / "identity.json", identity)
    if (destination / "completion.json").exists():
        return
    grouped = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["segment_filename"]].append((index, record))
    tasks = [(name, rows, str(destination / "segments" / (Path(name).stem + ".npz")),
        auxiliary["categories"], auxiliary["minimum_projected_side_pixels"]) for name, rows in sorted(grouped.items())]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=auxiliary["preparation_workers"]) as executor:
        paths = []
        for path in executor.map(prepare_segment, tasks):
            paths.append(path)
            write_json(destination / "progress.json", {"completed_segments": len(paths), "total_segments": len(tasks),
                "seconds": time.monotonic() - started, "pid": os.getpid()})
            if len(paths) % 20 == 0:
                print("OBJECT_TARGETS", len(paths), len(tasks), flush=True)
    object_counts = np.zeros(len(records), np.int64)
    for path in paths:
        with np.load(path) as values:
            object_counts[values["record_indices"]] = np.diff(values["object_offsets"])
    offsets = np.r_[0, np.cumsum(object_counts)]
    np.save(destination / "object_offsets.npy", offsets)
    arrays = {
        "projected_boxes": (np.float32, (9, 4)), "current_ego_states": (np.float32, (9, 4)),
        "state_valid": (bool, (9,)), "categories": (np.int16, ()), "track_identities": (np.int64, ())}
    mapped = {name: np.lib.format.open_memmap(destination / (name + ".npy"), mode="w+", dtype=dtype,
        shape=(int(offsets[-1]), *shape)) for name, (dtype, shape) in arrays.items()}
    coverage = defaultdict(lambda: np.zeros(9, np.int64))
    for path in paths:
        with np.load(path) as values:
            for local_index, record_index in enumerate(values["record_indices"]):
                source_slice = slice(values["object_offsets"][local_index], values["object_offsets"][local_index + 1])
                destination_slice = slice(offsets[record_index], offsets[record_index + 1])
                for name in arrays:
                    mapped[name][destination_slice] = values[name][source_slice]
                coverage[records[record_index]["split"]] += values["state_valid"][source_slice].sum(0)
    for values in mapped.values():
        values.flush()
    write_json(destination / "completion.json", {"complete": True, "identity": identity, "records": len(records),
        "objects": int(offsets[-1]), "maximum_objects_per_record": int(object_counts.max()),
        "valid_object_observations_by_horizon": {key: value.tolist() for key, value in coverage.items()},
        "seconds": time.monotonic() - started,
        "scope": "Current projected objects and their visible annotated future states in CURRENT ego coordinates; future GT association/poses are LOSS ONLY; unobserved objects are unknown, never negative presence labels",
        "files_sha256": {path.name: digest(path) for path in sorted(destination.glob("*.npy"))}})
    print("OBJECT_TARGETS_READY", len(records), int(offsets[-1]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    prepare(parser.parse_args().config.resolve())
