"""Check cached track/time/spatial targets against read-only source annotations."""

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
)
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    front_track_rois,
)
from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
    gather_selected_training_targets,
    select_nearest_current_entities,
)


def main():
    cache_directory = (
        PROJECT_ROOT / "outputs/feature_caches/target_supervision_rectified_v1b"
    )
    cache_index = json.loads((cache_directory / "cache_index.json").read_text())
    grouped_records = defaultdict(list)
    for record in cache_index["records"]:
        grouped_records[record["segment_filename"]].append(record)
    report = {
        "checked_windows": 0,
        "track_order_mismatches": 0,
        "spatial_target_max_absolute_error": 0.0,
        "spatial_valid_mismatches": 0,
        "cached_visual_valid_but_no_track_or_projection": 0,
        "selected_gather_max_absolute_error": 0.0,
        "time_offset_max_error_seconds": 0.0,
        "future_loss_broadcast_shape": "[window,selected_slot,future_step,channel]",
        "visual_values": "Frozen teacher values not recomputed; source code maps clip index j+1 to future frame j with current track order.",
        "source_writes": False,
    }
    for filename, records in grouped_records.items():
        with (PROJECT_ROOT / "dataset/navsim_logs/mini" / filename).open(
            "rb"
        ) as stream:
            frames = pickle.load(stream)
        for record in records:
            cached = torch.load(
                cache_directory / record["cache_file"], weights_only=True
            )
            start = record["start_index"]
            current = frames[start + 3]
            online = build_tracked_state_online_inputs(
                frames[start : start + 4], TrackedStateAdapterConfig()
            )
            future = frames[start + 4 : start + 12]
            targets = build_tracked_state_training_targets(
                current, future, online, TrackedStateAdapterConfig()
            )
            candidate_count = len(online.current_track_tokens)
            report["track_order_mismatches"] += int(
                list(online.current_track_tokens)
                != cached["metadata"]["current_track_tokens"]
            )
            report["spatial_target_max_absolute_error"] = max(
                report["spatial_target_max_absolute_error"],
                float(
                    (
                        targets.future_state_targets[0]
                        - cached["training_targets"]["future_spatial_targets"][
                            :candidate_count
                        ]
                    )
                    .abs()
                    .max()
                )
                if candidate_count
                else 0.0,
            )
            report["spatial_valid_mismatches"] += int(
                (
                    targets.future_target_valid_mask[0]
                    != cached["training_targets"]["future_spatial_valid_mask"][
                        :candidate_count
                    ]
                ).sum()
            )
            report["time_offset_max_error_seconds"] = max(
                report["time_offset_max_error_seconds"],
                float(
                    (
                        targets.future_time_offsets_seconds
                        - torch.arange(1, 9).double() * 0.5
                    )
                    .abs()
                    .max()
                ),
            )
            for future_step, frame in enumerate(future):
                _, projected = front_track_rois(
                    frame, online.current_track_tokens, use_opencv_pixel_centers=True
                )
                cached_valid = cached["training_targets"]["future_visual_valid_mask"][
                    :candidate_count, future_step
                ]
                report["cached_visual_valid_but_no_track_or_projection"] += int(
                    (cached_valid & ~projected).sum()
                )
            current_inputs = cached["online_inputs"]
            selection = select_nearest_current_entities(
                current_inputs["current_entity_features"][None].float(),
                current_inputs["current_entity_valid_mask"][None],
                current_inputs["stable_entity_ids"][None],
            )
            batched_targets = {
                key: value[None].float() if value.is_floating_point() else value[None]
                for key, value in cached["training_targets"].items()
            }
            gathered = gather_selected_training_targets(selection, batched_targets)
            for slot, index in enumerate(selection.selected_entity_indices[0].tolist()):
                if index < 0:
                    continue
                for target_name in ("visual", "spatial"):
                    valid = gathered[f"{target_name}_valid"][0, slot]
                    if valid.any():
                        report["selected_gather_max_absolute_error"] = max(
                            report["selected_gather_max_absolute_error"],
                            float(
                                (
                                    gathered[target_name][0, slot, valid]
                                    - batched_targets[f"future_{target_name}_targets"][
                                        0, index, valid
                                    ]
                                )
                                .abs()
                                .max()
                            ),
                        )
            report["checked_windows"] += 1
    output_path = (
        PROJECT_ROOT
        / "results/future_prediction_diagnostics/cached_target_alignment_v1.json"
    )
    if output_path.exists():
        raise RuntimeError("refusing to overwrite prior diagnostic")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
