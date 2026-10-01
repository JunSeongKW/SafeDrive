"""Preregistered native recording 3-way split; no model or future-target selection."""

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    front_image_path,
)


def native_recording_group(log_name):
    parts = str(log_name).rsplit("_", 2)
    return (
        parts[0]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit()
        else str(log_name)
    )


def assign_recording_splits(
    recording_groups, mini_groups, prior_development_groups, configuration
):
    seed = configuration["split_seed"]

    def rank(group, purpose):
        return hashlib.sha256(f"{seed}:{purpose}:{group}".encode()).hexdigest()

    clean_candidates = sorted(
        set(recording_groups) - set(mini_groups),
        key=lambda group: rank(group, "held_out"),
    )
    held_out = set(clean_candidates[: configuration["held_out_groups"]])
    if len(held_out) != configuration["held_out_groups"]:
        raise RuntimeError("not enough independent held-out recording groups")
    remaining = set(recording_groups) - held_out
    development = set(prior_development_groups) & remaining
    development.update(
        sorted(remaining - development, key=lambda group: rank(group, "development"))[
            : configuration["minimum_development_groups"] - len(development)
        ]
    )
    return {
        group: "held_out"
        if group in held_out
        else "development"
        if group in development
        else "train"
        for group in recording_groups
    }


def select_nonoverlapping_current_available_windows(
    segment_path, frames, sensor_root, occupied_intervals, seen_current_tokens
):
    """Structural future length/cadence only; NO future image/annotation validity test."""
    candidates, missing_current_files = [], 0
    for start in range(len(frames) - 11):
        window = frames[start : start + 12]
        timestamps = np.array([frame["timestamp"] for frame in window], dtype=np.int64)
        if np.any(np.abs(np.diff(timestamps) / 1e6 - 0.5) > 0.05):
            continue
        if (
            len({str(frame["scene_token"]) for frame in window}) != 1
            or len({str(frame["log_token"]) for frame in window}) != 1
        ):
            continue
        current = window[3]
        current_token = str(current["token"])
        begin, end = int(timestamps[0]), int(timestamps[-1])
        if current_token in seen_current_tokens or any(
            begin <= old_end and end >= old_begin
            for old_begin, old_end in occupied_intervals
        ):
            continue
        # Four HISTORY frames are required observations. Missing FUTURE images
        # do not discard the window; the cache builder creates loss-only masks.
        if not all(
            front_image_path(frame, sensor_root).is_file() for frame in window[:4]
        ):
            missing_current_files += 1
            continue
        occupied_intervals.append((begin, end))
        seen_current_tokens.add(current_token)
        candidates.append(
            {
                "segment_filename": segment_path.name,
                "start_index": start,
                "current_frame_token": current_token,
                "scene_token": str(current["scene_token"]),
                "native_log_name": str(current["log_name"]),
                "native_log_token": str(current["log_token"]),
                "history_begin_timestamp_us": begin,
                "future_end_timestamp_us": end,
                "command_raw_index": int(np.argmax(current["driving_command"])),
                "ego_speed_meters_per_second": float(
                    np.linalg.norm(current["ego_dynamic_state"][:2])
                ),
            }
        )
    return candidates, missing_current_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        type=Path,
        default=PROJECT_ROOT / "configs/exploration/pilot_foundation_decision_v1.json",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists() or not output_directory.is_relative_to(
        PROJECT_ROOT / "outputs"
    ):
        raise ValueError("require NEW project output directory")
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    configuration = json.loads(arguments.configuration.read_text())["expanded_data"]
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    log_root = dataset_root / "navsim_logs" / configuration["source_split"]
    sensor_root = dataset_root / "sensor_blobs" / configuration["source_split"]
    official_filter = PROJECT_ROOT / configuration["official_scene_filter"]
    official_names = yaml.safe_load(official_filter.read_text())["log_names"]
    available_paths = {path.stem: path for path in log_root.glob("*.pkl")}
    recording_segments = defaultdict(list)
    for name in official_names:
        if name in available_paths:
            recording_segments[native_recording_group(name)].append(
                available_paths[name]
            )
    mini_paths = list((dataset_root / "navsim_logs/mini").glob("*.pkl"))
    mini_groups = {native_recording_group(path.stem) for path in mini_paths}
    prior_manifest = json.loads(
        (
            PROJECT_ROOT
            / "results/data_surveys/navsim_recording_split_manifest_20261001.json"
        ).read_text()
    )
    prior_development = {
        row["recording_group"]
        for row in prior_manifest["groups"]
        if row["split"] == "development"
    }
    split_by_group = assign_recording_splits(
        recording_segments, mini_groups, prior_development, configuration
    )
    # Freeze assignments BEFORE reading window contents or measuring any model.
    assignments = {
        "source_split": "trainval",
        "seed": configuration["split_seed"],
        "official_filter_sha256": hashlib.sha256(
            official_filter.read_bytes()
        ).hexdigest(),
        "recording_grouping": "capture timestamp+vehicle from raw native log_name; segment suffixes grouped conservatively",
        "split_by_recording": split_by_group,
        "excluded_mini_held_out_groups": sorted(mini_groups),
        "held_out_model_evaluation_allowed": False,
    }
    (output_directory / "frozen_recording_assignments.json").write_text(
        json.dumps(assignments, indent=2) + "\n"
    )
    groups, windows, source_logs = [], [], {}
    native_log_splits, token_splits = {}, {}
    for group_index, group in enumerate(sorted(recording_segments)):
        occupied_intervals, seen_current_tokens, candidates = [], set(), []
        missing_current_count = 0
        for segment_path in sorted(recording_segments[group]):
            with segment_path.open("rb") as stream:
                frames = pickle.load(stream)
            if not frames:
                continue
            if any(
                native_recording_group(frame["log_name"]) != group for frame in frames
            ):
                raise RuntimeError(
                    "filename grouping differs from raw native recording identity"
                )
            selected, missing = select_nonoverlapping_current_available_windows(
                segment_path,
                frames,
                sensor_root,
                occupied_intervals,
                seen_current_tokens,
            )
            candidates.extend(selected)
            missing_current_count += missing
            with segment_path.open("rb") as stream:
                source_logs[segment_path.name] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        cap = configuration["maximum_windows_per_recording"]
        candidates.sort(
            key=lambda row: (row["history_begin_timestamp_us"], row["segment_filename"])
        )
        if len(candidates) > cap:
            candidates = [
                candidates[index]
                for index in np.linspace(0, len(candidates) - 1, cap, dtype=int)
            ]
        for row in candidates:
            row.update(recording_group=group, split=split_by_group[group])
            for identifier, registry in (
                (row["native_log_token"], native_log_splits),
                (row["current_frame_token"], token_splits),
            ):
                if identifier in registry and registry[identifier] != row["split"]:
                    raise RuntimeError(
                        "native log/current token crossed split boundary"
                    )
                registry[identifier] = row["split"]
        windows.extend(candidates)
        groups.append(
            {
                "recording_group": group,
                "split": split_by_group[group],
                "source_segments": [
                    path.name for path in sorted(recording_segments[group])
                ],
                "selected_windows": len(candidates),
                "missing_required_current_image_candidates": missing_current_count,
            }
        )
        if (group_index + 1) % 20 == 0:
            print(
                f"MANIFEST groups={group_index + 1}/{len(recording_segments)} windows={len(windows)}",
                flush=True,
            )
    counts = Counter(row["split"] for row in windows)
    group_counts = Counter(row["split"] for row in groups if row["selected_windows"])
    split_groups = {
        split: {row["recording_group"] for row in windows if row["split"] == split}
        for split in ("train", "development", "held_out")
    }
    if any(
        split_groups[first] & split_groups[second]
        for first, second in (
            ("train", "development"),
            ("train", "held_out"),
            ("development", "held_out"),
        )
    ):
        raise RuntimeError("recording leakage")
    if split_groups["held_out"] & mini_groups:
        raise RuntimeError("mini recording leaked into held-out")
    report = {
        "configuration": configuration,
        "assignments": assignments,
        "groups": groups,
        "records": windows,
        "source_log_sha256": source_logs,
        "counts": dict(counts),
        "recording_counts": dict(group_counts),
        "available_trainval_segment_count": len(available_paths),
        "official_navtrain_segment_count": len(official_names),
        "present_navtrain_segment_count": sum(map(len, recording_segments.values())),
        "mini_recording_group_count": len(mini_groups),
        "theoretical_windows_cap": len(groups) * cap,
        "nominal_target_scale_reached": all(
            counts[split] >= target
            for split, target in configuration["target_windows"].items()
        ),
        "context_counts": {
            split: dict(
                Counter(
                    str(row["command_raw_index"])
                    for row in windows
                    if row["split"] == split
                )
            )
            for split in split_groups
        },
        "future_validity_used_for_selection": False,
        "source_data_writes": False,
        "held_out_model_evaluation_performed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    (output_directory / "recording_split_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "counts",
                    "recording_counts",
                    "theoretical_windows_cap",
                    "nominal_target_scale_reached",
                    "wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
