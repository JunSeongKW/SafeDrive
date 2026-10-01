"""Small CPU metadata/projection survey and recording-group split; no encoder/training.

Sampling is fixed by recording names and timestamp cadence, NOT future visibility
or model outcomes. Future turn bins are post-hoc descriptive labels only.
Original NAVSIM files are opened read-only; outputs stay inside this workspace.
"""

import argparse
import hashlib
import json
import math
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from PIL import Image, ImageDraw

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    annotation_track_lookup,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
)
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    project_lidar_box_to_front_roi,
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def deterministic_rank(name: str, seed: str, purpose: str) -> str:
    return hashlib.sha256(f"{seed}:{purpose}:{name}".encode()).hexdigest()


def recording_group_from_segment_name(segment_path: Path) -> str:
    # nuPlan capture timestamp + vehicle. Segment start/end suffixes are NOT split IDs.
    recording_group, segment_start, segment_end = segment_path.stem.rsplit("_", 2)
    if not segment_start.isdigit() or not segment_end.isdigit():
        raise ValueError(f"unrecognized NAVSIM segment filename: {segment_path.name}")
    return recording_group


def choose_recording_groups(segment_paths: list[Path], config: dict) -> list[dict]:
    grouped_paths = defaultdict(list)
    for segment_path in segment_paths:
        grouped_paths[recording_group_from_segment_name(segment_path)].append(
            segment_path
        )
    forced_development = set(
        config["previously_inspected_recording_groups_development_only"]
    )
    if not forced_development.issubset(grouped_paths):
        raise ValueError(
            "previously inspected development group absent from source subset"
        )
    num_extra = config["num_recording_groups"] - len(forced_development)
    candidates = sorted(
        set(grouped_paths) - forced_development,
        key=lambda name: deterministic_rank(
            name, config["sampling_seed"], "group_sample"
        ),
    )
    selected_names = sorted(forced_development | set(candidates[:num_extra]))
    development_names = forced_development | set(
        sorted(
            set(selected_names) - forced_development,
            key=lambda name: deterministic_rank(
                name, config["sampling_seed"], "development_split"
            ),
        )[: config["num_development_groups"] - len(forced_development)]
    )
    records = []
    for name in selected_names:
        selected_segment = min(
            grouped_paths[name],
            key=lambda path: deterministic_rank(
                path.name, config["sampling_seed"], "segment_sample"
            ),
        )
        records.append(
            {
                "recording_group": name,
                "split": "development" if name in development_names else "train",
                "segment_filename": selected_segment.name,
                "all_source_segment_aliases_same_split": [
                    path.name for path in sorted(grouped_paths[name])
                ],
            }
        )
    assert len(records) == config["num_recording_groups"]
    assert (
        sum(record["split"] == "development" for record in records)
        == config["num_development_groups"]
    )
    return records


def nonoverlapping_eligible_windows(frames: list, config: dict) -> list[int]:
    required_frames = config["num_history_frames"] + config["num_future_steps"]
    candidates, last_end = [], -1
    seen_current_tokens = set()
    for start in range(len(frames) - required_frames + 1):
        if start < last_end:
            continue
        window = frames[start : start + required_frames]
        intervals = np.diff([frame["timestamp"] for frame in window]) / 1e6
        current_token = str(window[config["num_history_frames"] - 1]["token"])
        if (
            np.all(np.abs(intervals - 0.5) <= 0.05)
            and len({str(frame["scene_token"]) for frame in window}) == 1
            and len({str(frame["log_token"]) for frame in window}) == 1
            and current_token not in seen_current_tokens
        ):
            candidates.append(start)
            last_end = start + required_frames
            seen_current_tokens.add(current_token)
    return candidates


def evenly_spaced_subset(values: list[int], cap: int) -> list[int]:
    if len(values) <= cap:
        return values.copy()
    return [values[index] for index in np.linspace(0, len(values) - 1, cap, dtype=int)]


def projected_track_rois(frame: dict, track_tokens: tuple, camera_channel: str):
    camera = frame["cams"].get(camera_channel)
    lookup = annotation_track_lookup(frame)
    roi_boxes = np.zeros((len(track_tokens), 4), dtype=np.float32)
    valid = np.zeros(len(track_tokens), dtype=bool)
    if camera is not None:
        for entity_index, track_token in enumerate(track_tokens):
            if track_token in lookup:
                roi_boxes[entity_index], valid[entity_index] = (
                    project_lidar_box_to_front_roi(
                        frame["anns"]["gt_boxes"][lookup[track_token]], camera
                    )
                )
    return roi_boxes, valid


def safe_camera_path(
    frame: dict, camera_channel: str, sensor_root: Path
) -> Path | None:
    camera = frame["cams"].get(camera_channel)
    if camera is None or not camera.get("data_path"):
        return None
    path = (sensor_root / camera["data_path"]).resolve()
    if not path.is_relative_to(sensor_root.resolve()):
        raise ValueError("camera path escapes read-only sensor root")
    return path


def aggregate_coverage(windows: list[dict], budget: int) -> dict:
    current_count = sum(window["current_front_valid"] for window in windows)
    candidate_count = sum(window["candidate_count"] for window in windows)
    visual_counts = (
        np.sum(
            [window["future_visual_valid_among_current_front"] for window in windows],
            axis=0,
        ).astype(int)
        if windows
        else np.zeros(8, dtype=int)
    )
    spatial_counts = (
        np.sum(
            [window["future_spatial_valid_among_current_front"] for window in windows],
            axis=0,
        ).astype(int)
        if windows
        else np.zeros(8, dtype=int)
    )
    side_only_count = sum(window["current_side_only_valid"] for window in windows)
    return {
        "window_count": len(windows),
        "recording_group_count": len({window["recording_group"] for window in windows}),
        "current_candidate_count": candidate_count,
        "current_front_valid_count": current_count,
        "current_front_fraction_of_candidates": current_count / candidate_count
        if candidate_count
        else None,
        "current_front_count_median": float(
            np.median([window["current_front_valid"] for window in windows])
        )
        if windows
        else None,
        "fraction_windows_current_front_count_gt_budget": sum(
            window["current_front_valid"] > budget for window in windows
        )
        / len(windows)
        if windows
        else None,
        "fraction_windows_current_front_count_zero": sum(
            window["current_front_valid"] == 0 for window in windows
        )
        / len(windows)
        if windows
        else None,
        "current_side_only_fraction_of_candidates": side_only_count / candidate_count
        if candidate_count
        else None,
        "future_visual_valid_counts": visual_counts.tolist(),
        "future_spatial_valid_counts": spatial_counts.tolist(),
        "future_visual_retention_given_current_front": (
            visual_counts / current_count
        ).tolist()
        if current_count
        else None,
        "future_spatial_retention_given_current_front": (
            spatial_counts / current_count
        ).tolist()
        if current_count
        else None,
        "spatial_but_not_visual_target_counts": (
            spatial_counts - visual_counts
        ).tolist(),
        "fraction_windows_current_two_frame_images_available": sum(
            window["current_images_available"] for window in windows
        )
        / len(windows)
        if windows
        else None,
        "nearest_current_rule_active_slots": sum(
            window["nearest_rule_active_slots"] for window in windows
        ),
        "nearest_current_rule_future_visual_valid_counts": [
            sum(window["nearest_rule_future_visual_valid"][step] for window in windows)
            for step in range(8)
        ],
        "nearest_current_rule_future_spatial_valid_counts": [
            sum(window["nearest_rule_future_spatial_valid"][step] for window in windows)
            for step in range(8)
        ],
    }


def write_projection_review(
    frames, window_record, track_tokens, sensor_root, output_directory, image_digests
):
    image_panels = []
    for label, frame, channel in (
        ("current front", frames[0], "CAM_F0"),
        ("current left", frames[0], "CAM_L0"),
        ("current right", frames[0], "CAM_R0"),
        ("future +4s front", frames[1], "CAM_F0"),
    ):
        path = safe_camera_path(frame, channel, sensor_root)
        if path is None or not path.is_file():
            continue
        image_digests[str(path)] = sha256_file(path)
        with Image.open(path) as image:
            rgb_image = np.asarray(image.convert("RGB"))
        if rgb_image.shape[:2] != (1080, 1920):
            raise ValueError("review image is not expected 1920x1080")
        import cv2

        panel = Image.fromarray(cv2.resize(rgb_image[28:-28], (512, 256)))
        painter = ImageDraw.Draw(panel)
        boxes, valid = projected_track_rois(frame, track_tokens, channel)
        for entity_index in np.flatnonzero(valid):
            painter.rectangle(boxes[entity_index].tolist(), outline="#26df91", width=1)
        image_panels.append((label, panel))
    sheet = Image.new("RGB", (1024, 560), "#161616")
    for index, (label, panel) in enumerate(image_panels):
        horizontal, vertical = (index % 2) * 512, (index // 2) * 280
        sheet.paste(panel, (horizontal, vertical + 24))
        ImageDraw.Draw(sheet).text(
            (horizontal + 8, vertical + 6),
            label + "; projection not occlusion visibility",
            fill="white",
        )
    output_path = (
        output_directory
        / f"{window_record['recording_group']}_{window_record['start_index']}_projection_review.png"
    )
    sheet.save(output_path)
    return str(output_path.relative_to(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/data_surveys/navsim_visual_target_coverage_20261001",
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if not output_directory.is_relative_to(PROJECT_ROOT / "outputs/data_surveys"):
        parser.error("output directory must stay within workspace outputs/data_surveys")
    if output_directory.exists():
        parser.error("refusing to replace previous survey outputs")
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    torch.set_num_threads(1)
    config = json.loads(
        (PROJECT_ROOT / "configs/exploration/data_survey.json").read_text()
    )
    if (config["num_history_frames"], config["num_future_steps"]) != (4, 8):
        raise ValueError("this v1 survey is explicitly pinned to history4/future8")
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    log_root = dataset_root / "navsim_logs" / config["source_subset"]
    sensor_root = dataset_root / "sensor_blobs" / config["source_subset"]
    source_paths = sorted(log_root.glob("*.pkl"))
    groups = choose_recording_groups(source_paths, config)
    # Commit the selection manifest locally BEFORE examining any coverage/outcome.
    (output_directory / "pre_survey_recording_split.json").write_text(
        json.dumps({"config": config, "groups": groups}, indent=2) + "\n"
    )
    state_config = TrackedStateAdapterConfig()
    results, failures, review_examples = [], [], {}
    shared_image_digests, source_log_digests = {}, {}
    seen_log_tokens_by_split = {"train": set(), "development": set()}
    for group in groups:
        path = log_root / group["segment_filename"]
        source_log_digests[str(path)] = sha256_file(path)
        with path.open("rb") as stream:
            frames = pickle.load(stream)  # trusted local lab NAVSIM pickle only
        eligible = nonoverlapping_eligible_windows(frames, config)
        group["log_token"] = str(frames[0]["log_token"])
        group["segment_frame_count"] = len(frames)
        group["eligible_nonoverlapping_window_count"] = len(eligible)
        pilot_starts = evenly_spaced_subset(
            eligible, config["pilot_windows_per_group_cap"]
        )
        group["pilot_window_starts"] = pilot_starts
        group["survey_window_starts"] = evenly_spaced_subset(
            pilot_starts, config["survey_windows_per_group"]
        )
        group["source_sha256"] = source_log_digests[str(path)]
        group["all_native_log_tokens"] = sorted(
            {str(frame["log_token"]) for frame in frames}
        )
        seen_log_tokens_by_split[group["split"]].update(group["all_native_log_tokens"])
        for start in group["survey_window_starts"]:
            history = frames[start : start + 4]
            current = history[-1]
            future = frames[start + 4 : start + 12]
            try:
                online = build_tracked_state_online_inputs(history, state_config)
                targets = build_tracked_state_training_targets(
                    current, future, online, state_config
                )
                _, front_valid = projected_track_rois(
                    current, online.current_track_tokens, "CAM_F0"
                )
                side_valid = np.zeros_like(front_valid)
                for channel in ("CAM_L0", "CAM_R0"):
                    _, side_camera_valid = projected_track_rois(
                        current, online.current_track_tokens, channel
                    )
                    side_valid |= side_camera_valid
                current_paths = [
                    safe_camera_path(frame, "CAM_F0", sensor_root)
                    for frame in history[-2:]
                ]
                current_images_available = all(
                    path is not None and path.is_file() for path in current_paths
                )
                if current_images_available:
                    for image_path in current_paths:
                        with Image.open(image_path) as image:
                            if image.size != (1920, 1080):
                                raise ValueError(
                                    "unsupported current camera image dimensions"
                                )
                (
                    visual_counts,
                    spatial_counts,
                    nearest_visual_counts,
                    nearest_spatial_counts,
                ) = [], [], [], []
                # Nearest current front K: fixed rule; NEVER use future availability to select.
                selected_indices = np.flatnonzero(front_valid)[
                    : config["selected_entity_budget"]
                ]
                for step, future_frame in enumerate(future):
                    _, projected_future_valid = projected_track_rois(
                        future_frame, online.current_track_tokens, "CAM_F0"
                    )
                    future_image_path = safe_camera_path(
                        future_frame, "CAM_F0", sensor_root
                    )
                    previous_image_path = safe_camera_path(
                        ([current] + future)[step], "CAM_F0", sensor_root
                    )
                    clip_available = all(
                        path is not None and path.is_file()
                        for path in (previous_image_path, future_image_path)
                    )
                    spatial_valid = (
                        front_valid
                        & targets.future_target_valid_mask[0, :, step].numpy()
                    )
                    visual_valid = (
                        spatial_valid & projected_future_valid & clip_available
                    )
                    visual_counts.append(int(visual_valid.sum()))
                    spatial_counts.append(int(spatial_valid.sum()))
                    nearest_visual_counts.append(
                        int(visual_valid[selected_indices].sum())
                    )
                    nearest_spatial_counts.append(
                        int(spatial_valid[selected_indices].sum())
                    )
                future_heading_degrees = math.degrees(
                    float(targets.ego_trajectory_target[0, -1, 2])
                )
                turn_threshold = config["turn_analysis_threshold_degrees"]
                turn_bin = (
                    "left_turn_future_heading_proxy"
                    if future_heading_degrees > turn_threshold
                    else "right_turn_future_heading_proxy"
                    if future_heading_degrees < -turn_threshold
                    else "low_turn_future_heading_proxy"
                )
                record = {
                    "recording_group": group["recording_group"],
                    "split": group["split"],
                    "segment_filename": group["segment_filename"],
                    "log_token": str(current["log_token"]),
                    "scene_token": str(current["scene_token"]),
                    "current_token": str(current["token"]),
                    "start_index": start,
                    "candidate_count": len(online.current_track_tokens),
                    "current_front_valid": int(front_valid.sum()),
                    "current_side_only_valid": int((side_valid & ~front_valid).sum()),
                    "current_front_and_side_union": int(
                        (front_valid | side_valid).sum()
                    ),
                    "current_images_available": current_images_available,
                    "current_camera_image_files_available": {
                        channel: (
                            safe_camera_path(current, channel, sensor_root) is not None
                            and safe_camera_path(
                                current, channel, sensor_root
                            ).is_file()
                        )
                        for channel in config["camera_channels_for_coverage"]
                    },
                    "future_visual_valid_among_current_front": visual_counts,
                    "future_spatial_valid_among_current_front": spatial_counts,
                    "nearest_rule_future_visual_valid": nearest_visual_counts,
                    "nearest_rule_future_spatial_valid": nearest_spatial_counts,
                    "nearest_rule_active_slots": len(selected_indices),
                    "posthoc_future_heading_degrees": future_heading_degrees,
                    "posthoc_turn_bin": turn_bin,
                    "current_command_index_uninterpreted": int(
                        np.argmax(current["driving_command"])
                    ),
                    "current_speed_meters_per_second": float(
                        np.linalg.norm(current["ego_dynamic_state"][:2])
                    ),
                    "map_location": str(current["map_location"]),
                    "front_distortion_coefficients": np.asarray(
                        current["cams"]["CAM_F0"]["distortion"]
                    ).tolist(),
                }
                results.append(record)
                # One first observed example for each descriptive turn bin and each split.
                review_key = f"{group['split']}:{turn_bin}"
                if review_key not in review_examples:
                    review_examples[review_key] = write_projection_review(
                        [current, future[-1]],
                        record,
                        online.current_track_tokens,
                        sensor_root,
                        output_directory,
                        shared_image_digests,
                    )
            except (ValueError, KeyError, FileNotFoundError) as error:
                failures.append(
                    {
                        "recording_group": group["recording_group"],
                        "start_index": start,
                        "error": str(error),
                    }
                )
        print(
            f"surveyed {group['recording_group']} {group['split']} windows={len(group['survey_window_starts'])}",
            flush=True,
        )
    assert (
        not seen_log_tokens_by_split["train"] & seen_log_tokens_by_split["development"]
    )
    assert all(
        sha256_file(Path(path)) == digest for path, digest in source_log_digests.items()
    )
    assert all(
        sha256_file(Path(path)) == digest
        for path, digest in shared_image_digests.items()
    )
    train_groups = [group for group in groups if group["split"] == "train"]
    development_groups = [group for group in groups if group["split"] == "development"]
    manifest = {
        "config": config,
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "groups": groups,
        "split_scope": "mini_exploration_NOT_official_train_test_or_final_holdout",
        "original_recording_groups_disjoint": True,
        "sampled_native_log_tokens_disjoint": True,
        "pilot_train_window_count": sum(
            len(group["pilot_window_starts"]) for group in train_groups
        ),
        "pilot_development_window_count": sum(
            len(group["pilot_window_starts"]) for group in development_groups
        ),
    }
    by_turn = {
        label: aggregate_coverage(
            [window for window in results if window["posthoc_turn_bin"] == label],
            config["selected_entity_budget"],
        )
        for label in (
            "left_turn_future_heading_proxy",
            "low_turn_future_heading_proxy",
            "right_turn_future_heading_proxy",
        )
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "small_multi_recording_projection_target_availability_NOT_visibility_or_model_performance",
        "config": config,
        "source_mini_segment_count": len(source_paths),
        "source_mini_recording_group_count": len(
            {recording_group_from_segment_name(path) for path in source_paths}
        ),
        "overall": aggregate_coverage(results, config["selected_entity_budget"]),
        "by_split": {
            split: aggregate_coverage(
                [window for window in results if window["split"] == split],
                config["selected_entity_budget"],
            )
            for split in ("train", "development")
        },
        "by_posthoc_turn": by_turn,
        "by_current_command_index_uninterpreted": {
            str(command): aggregate_coverage(
                [
                    window
                    for window in results
                    if window["current_command_index_uninterpreted"] == command
                ],
                config["selected_entity_budget"],
            )
            for command in sorted(
                {window["current_command_index_uninterpreted"] for window in results}
            )
        },
        "windows": results,
        "failures": failures,
        "projection_review_examples": review_examples,
        "shared_log_sha256": source_log_digests,
        "shared_review_image_sha256": shared_image_digests,
        "shared_read_files_hashes_unchanged": True,
        "source_script_sha256": sha256_file(Path(__file__)),
        "elapsed_cpu_seconds": time.perf_counter() - started,
        "limitations": [
            "mini convenience subset, not representative NAVSIM population",
            "GT current geometry/track",
            "fixed 40m and nearest32 prefilter",
            "projection valid not occlusion visibility",
            "side projection does not implement multiview encoder",
            "turn labels are future-derived analysis only, not merge/intersection labels",
            "command label semantics not independently verified",
            "distortion coefficients stored but pinhole image rectification unresolved",
            "one segment per recording sampled; all aliases same split",
            "future validity never filters current candidates or sampled windows",
        ],
    }
    (output_directory / "recording_split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (output_directory / "visual_target_coverage.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    summary = {key: value for key, value in report.items() if key != "windows"}
    summary["raw_window_report_path"] = str(
        (output_directory / "visual_target_coverage.json").relative_to(PROJECT_ROOT)
    )
    (output_directory / "visual_target_coverage_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "overall": report["overall"],
                "turn_bins": {
                    name: value["window_count"] for name, value in by_turn.items()
                },
                "failures": failures,
                "pilot_train": manifest["pilot_train_window_count"],
                "pilot_development": manifest["pilot_development_window_count"],
                "elapsed_seconds": report["elapsed_cpu_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
