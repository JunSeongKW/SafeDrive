"""Cache ONLY the committed small split; shared datasets are opened read-only."""

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
)
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    front_image_path,
    front_track_rois,
    load_front_image,
    pool_projected_entity_rois,
)
from planning_aware_future_prediction.models.frozen_driving_video_encoder import (
    FrozenDrivingVideoEncoder,
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def pad_candidate_axis(tensor, candidate_count=32):
    padded = tensor.new_zeros((candidate_count, *tensor.shape[1:]))
    padded[: tensor.shape[0]] = tensor
    return padded


def cache_window(frames, start, sensor_root, encoder, device, source_image_hashes,
                 rectify_stored_image=True):
    adapter_config = TrackedStateAdapterConfig()
    history, future = frames[start : start + 4], frames[start + 4 : start + 12]
    current = history[-1]
    online = build_tracked_state_online_inputs(history, adapter_config)
    targets = build_tracked_state_training_targets(
        current, future, online, adapter_config
    )
    image_frames = [history[-2], current, *future]
    processed_images, image_available = [], []
    for frame_index, frame in enumerate(image_frames):
        image_path = front_image_path(frame, sensor_root)
        available = image_path.is_file()
        if not available and frame_index < 2:
            raise FileNotFoundError(f"current observation unavailable: {image_path}")
        image_available.append(available)
        if available:
            relative_path = str(image_path.relative_to(sensor_root.resolve()))
            if relative_path not in source_image_hashes:
                source_image_hashes[relative_path] = file_sha256(image_path)
            image = (
                torch.from_numpy(
                    load_front_image(
                        frame, sensor_root, rectify_stored_image=rectify_stored_image
                    ).copy()
                )
                .permute(2, 0, 1)
                .float()
                / 255
            )
        else:
            image = torch.zeros(3, 256, 512)
        processed_images.append(image)
    clips = torch.stack(
        [torch.stack(processed_images[index : index + 2], dim=1) for index in range(9)]
    )
    image_mean = clips.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
    image_std = clips.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
    feature_grids = encoder(((clips - image_mean) / image_std).to(device))
    pooled_entities, projected_masks = [], []
    for step, frame in enumerate([current, *future]):
        boxes, valid = front_track_rois(
            frame, online.current_track_tokens, use_opencv_pixel_centers=True
        )
        if not (image_available[step] and image_available[step + 1]):
            valid.zero_()
        pooled_entities.append(
            pool_projected_entity_rois(feature_grids[step : step + 1], boxes, valid)[
                0
            ].cpu()
        )
        projected_masks.append(valid)
    current_features = torch.cat(
        (pooled_entities[0], online.entity_features[0]), dim=-1
    )
    current_features = torch.where(projected_masks[0][:, None], current_features, 0.0)
    future_visual_targets = torch.stack(pooled_entities[1:], dim=1)
    future_visual_valid = (
        torch.stack(projected_masks[1:], dim=1) & targets.future_target_valid_mask[0]
    )
    # No future mask is used to make online inputs or to remove a window.
    cached = {
        "online_inputs": {
            "current_image_grid": feature_grids[0].cpu().half(),
            "current_entity_features": pad_candidate_axis(current_features).half(),
            "current_entity_valid_mask": pad_candidate_axis(projected_masks[0]),
            "stable_entity_ids": pad_candidate_axis(online.stable_entity_ids[0]),
            "current_ego_status": torch.tensor(
                np.concatenate(
                    (current["driving_command"], current["ego_dynamic_state"])
                ),
                dtype=torch.float32,
            ),
        },
        "training_targets": {
            "future_visual_targets": pad_candidate_axis(future_visual_targets).half(),
            "future_visual_valid_mask": pad_candidate_axis(future_visual_valid),
            "future_spatial_targets": pad_candidate_axis(
                targets.future_state_targets[0]
            ),
            "future_spatial_valid_mask": pad_candidate_axis(
                targets.future_target_valid_mask[0]
            ),
            "ego_trajectory_target": targets.ego_trajectory_target[0],
        },
        "metadata": {
            "scene_token": str(current["scene_token"]),
            "current_frame_token": str(current["token"]),
            "current_front_valid_count": int(projected_masks[0].sum()),
            "current_candidate_count": len(online.current_track_tokens),
            "current_track_tokens": list(online.current_track_tokens),
            "ego_speed_meters_per_second": float(
                np.linalg.norm(current["ego_dynamic_state"][:2])
            ),
            "command_raw_index": int(np.argmax(current["driving_command"])),
            "posthoc_future_heading_radians": float(
                targets.ego_trajectory_target[0, -1, 2]
            ),
        },
    }
    return cached


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.physical_gpu)
    torch.set_num_threads(4)
    output_directory = arguments.output_directory.resolve()
    if not output_directory.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("cache must stay in project outputs")
    if output_directory.exists() and not arguments.resume:
        raise RuntimeError(
            "refusing to overwrite existing cache; explicit --resume required"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        PROJECT_ROOT
        / "results/data_surveys/navsim_recording_split_manifest_20261001.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = file_sha256(manifest_path)
    if len(manifest["groups"]) != 16:
        raise RuntimeError("unexpected split")
    window_records = []
    for group in manifest["groups"]:
        for start in group["pilot_window_starts"]:
            window_records.append(
                {
                    "recording_group": group["recording_group"],
                    "split": group["split"],
                    "segment_filename": group["segment_filename"],
                    "start_index": start,
                }
            )
    if len(window_records) != 373:
        raise RuntimeError("expected exactly 373 committed windows")
    configuration = {
        "manifest_sha256": manifest_sha,
        "schema_version": 1,
        "rectify_stored_image": True,
        "rectification_new_intrinsic": "unchanged stored K",
        "projection": "pinhole after full-resolution stored K/D rectification",
        "crop_resize": "28px top/bottom; cv2 linear 512x256; ROI pixel-center affine",
        "cache_dtype": "visual float16; training casts to float32",
        "candidate_filter_uses_future_validity": False,
        "target_teacher": "same official frozen encoder, two-frame clip ending at target time",
        "history_limit": "future teacher clip is training-only",
        "maximum_cache_bytes": 2 * 1024**3,
        "physical_gpu": arguments.physical_gpu,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in [
                Path(__file__),
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/adapters/navsim_visual_entities.py",
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/adapters/navsim_tracked_state.py",
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/models/frozen_driving_video_encoder.py",
            ]
        },
    }
    configuration_path = output_directory / "cache_configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise RuntimeError("resume source/config differs")
    else:
        configuration_path.write_text(json.dumps(configuration, indent=2) + "\n")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("approved GPU unavailable; no CPU encoder fallback")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    encoder = FrozenDrivingVideoEncoder(PROJECT_ROOT, device)
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    sensor_root = dataset_root / "sensor_blobs/mini"
    source_hash_path = output_directory / "source_image_sha256.json"
    source_hashes = (
        json.loads(source_hash_path.read_text()) if source_hash_path.exists() else {}
    )
    existing_index_path = output_directory / "cache_index.json"
    cached_records = (
        json.loads(existing_index_path.read_text())["records"]
        if existing_index_path.exists()
        else []
    )
    by_key = {
        (record["segment_filename"], record["start_index"]): record
        for record in cached_records
    }
    selected_records = (
        [record for record in window_records if record["split"] == "train"][:8]
        if arguments.profile_only
        else window_records
    )
    loaded_segment_name, frames = None, None
    new_window_seconds = []
    for record in selected_records:
        key = (record["segment_filename"], record["start_index"])
        if key in by_key:
            if (
                file_sha256(output_directory / by_key[key]["cache_file"])
                != by_key[key]["cache_sha256"]
            ):
                raise RuntimeError("existing cache hash mismatch")
            continue
        if record["segment_filename"] != loaded_segment_name:
            log_path = dataset_root / "navsim_logs/mini" / record["segment_filename"]
            matching_group = next(
                group
                for group in manifest["groups"]
                if group["segment_filename"] == record["segment_filename"]
            )
            if file_sha256(log_path) != matching_group["source_sha256"]:
                raise RuntimeError("shared source log changed since split commitment")
            with log_path.open("rb") as stream:
                frames = pickle.load(stream)
            loaded_segment_name = record["segment_filename"]
        window_start_time = time.perf_counter()
        cached = cache_window(
            frames, record["start_index"], sensor_root, encoder, device, source_hashes
        )
        cached["metadata"].update(record)
        cache_file = f"{Path(record['segment_filename']).stem}__window_{record['start_index']:05d}.pt"
        cache_path = output_directory / cache_file
        if cache_path.exists():
            raise RuntimeError(
                "unindexed cache file exists; inspect interrupted write before resuming"
            )
        torch.save(cached, cache_path)
        indexed_record = {
            **record,
            "cache_file": cache_file,
            "cache_sha256": file_sha256(cache_path),
            "cache_bytes": cache_path.stat().st_size,
            **cached["metadata"],
        }
        by_key[key] = indexed_record
        cached_records.append(indexed_record)
        seconds = time.perf_counter() - window_start_time
        new_window_seconds.append(seconds)
        if (
            sum(row["cache_bytes"] for row in cached_records)
            > configuration["maximum_cache_bytes"]
        ):
            raise RuntimeError("cache exceeded 2GiB cap")
        if (
            arguments.profile_only
            and indexed_record["cache_bytes"] * 373
            > configuration["maximum_cache_bytes"]
        ):
            raise RuntimeError("profile projects beyond cache cap")
        index = {
            "configuration": configuration,
            "encoder_provenance": encoder.provenance,
            "records": cached_records,
            "complete": len(cached_records) == len(window_records),
        }
        existing_index_path.write_text(json.dumps(index, indent=2) + "\n")
        source_hash_path.write_text(json.dumps(source_hashes, indent=2) + "\n")
        print(
            f"CACHE_WINDOW {len(cached_records)}/373 {record['split']} seconds={seconds:.3f}",
            flush=True,
        )
    report = {
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "configuration": configuration,
        "encoder_provenance": encoder.provenance,
        "cached_windows": len(cached_records),
        "complete": len(cached_records) == 373,
        "cached_window_bytes": sum(record["cache_bytes"] for record in cached_records),
        "total_directory_bytes": sum(
            path.stat().st_size for path in output_directory.iterdir() if path.is_file()
        ),
        "source_images_hashed": len(source_hashes),
        "new_windows_this_invocation": len(new_window_seconds),
        "encoder_clips_this_invocation": 9 * len(new_window_seconds),
        "invocation_wall_seconds": time.perf_counter() - start_time,
        "mean_new_window_seconds": float(np.mean(new_window_seconds))
        if new_window_seconds
        else None,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(device),
        "original_shared_files_modified": False,
        "local_image_export_history_independently_verified": False,
    }
    (
        output_directory
        / ("profile_report.json" if arguments.profile_only else "cache_report.json")
    ).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(
        "CACHE_PROFILE_DONE" if arguments.profile_only else "FEATURE_CACHE_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
