"""Same-image hypothesis audit. Neither round trips nor feature changes prove provenance."""

import argparse
import json
import pickle
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
    relative_ego_planar_pose,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists() or not output_directory.is_relative_to(
        PROJECT_ROOT / "outputs"
    ):
        raise ValueError("new workspace output required")
    output_directory.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    old_survey = json.loads(
        (
            PROJECT_ROOT
            / "outputs/data_surveys/navsim_visual_target_coverage_v1_20261001/visual_target_coverage.json"
        ).read_text()
    )
    chosen = {}
    for row in old_survey["windows"]:
        chosen.setdefault((row["split"], row["posthoc_turn_bin"]), row)
    examples = list(chosen.values())[:4]
    sensor_root = (PROJECT_ROOT / "dataset/sensor_blobs/mini").resolve()
    encoder = FrozenDrivingVideoEncoder(PROJECT_ROOT, torch.device("cpu"))
    report = {
        "resolved": False,
        "operating_assumption": "old cache assumes original distorted JPEG then stored K/D undistortion",
        "blocker": "Separate nuPlan original sensor JPEG / local export manifest or independent pixel alignment labels unavailable; metadata helper supplies raw path and K/D but does not document redistribution JPEG processing.",
        "roundtrip_is_provenance_proof": False,
        "feature_sensitivity_is_alignment_proof": False,
        "GPU_used": False,
        "encoder_provenance": encoder.provenance,
        "examples": [],
    }
    for example_index, row in enumerate(examples):
        if time.perf_counter() - started > 180:
            report["stopped_by_180_second_audit_cap"] = True
            break
        with (PROJECT_ROOT / "dataset/navsim_logs/mini" / row["segment_filename"]).open(
            "rb"
        ) as stream:
            frames = pickle.load(stream)
        start = row["start_index"]
        history = frames[start : start + 4]
        current = history[-1]
        online = build_tracked_state_online_inputs(history, TrackedStateAdapterConfig())
        images = {
            rectify: [
                load_front_image(frame, sensor_root, rectify_stored_image=rectify)
                for frame in history[-2:]
            ]
            for rectify in (False, True)
        }
        clips = torch.stack(
            [
                torch.stack(
                    [
                        torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
                        for image in images[rectify]
                    ],
                    dim=1,
                )
                for rectify in (False, True)
            ]
        )
        mean = clips.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
        std = clips.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
        grids = encoder((clips - mean) / std)
        pinhole_boxes, pinhole_valid = front_track_rois(
            current, online.current_track_tokens, use_opencv_pixel_centers=True
        )
        distorted_boxes, distorted_valid = front_track_rois(
            current,
            online.current_track_tokens,
            apply_stored_distortion=True,
            use_opencv_pixel_centers=True,
        )
        rectified_rois = pool_projected_entity_rois(
            grids[1:2], pinhole_boxes, pinhole_valid
        )[0]
        raw_pinhole_rois = pool_projected_entity_rois(
            grids[:1], pinhole_boxes, pinhole_valid
        )[0]
        raw_distorted_rois = pool_projected_entity_rois(
            grids[:1], distorted_boxes, distorted_valid
        )[0]
        common = pinhole_valid & distorted_valid
        sheet = Image.new("RGB", (1536, 284), "#101010")
        for column, (image, boxes, valid, label) in enumerate(
            (
                (
                    images[False][-1],
                    pinhole_boxes,
                    pinhole_valid,
                    "stored JPEG / pinhole",
                ),
                (
                    images[False][-1],
                    distorted_boxes,
                    distorted_valid,
                    "stored JPEG / stored-distortion",
                ),
                (
                    images[True][-1],
                    pinhole_boxes,
                    pinhole_valid,
                    "undistorted JPEG / pinhole",
                ),
            )
        ):
            panel = Image.fromarray(image.copy())
            drawing = ImageDraw.Draw(panel)
            for entity_index in valid.nonzero().flatten().tolist():
                drawing.rectangle(
                    boxes[entity_index].tolist(), outline="#20ef70", width=1
                )
            sheet.paste(panel, (512 * column, 24))
            ImageDraw.Draw(sheet).text((512 * column + 4, 4), label, fill="white")
        sheet.save(output_directory / f"hypotheses_{example_index}.png")
        current_seconds = (current["timestamp"] - history[-2]["timestamp"]) / 1e6
        past_delta = (
            -relative_ego_planar_pose(history[-2], current)[:2] / current_seconds
        )
        native_velocity = np.array(current["ego_dynamic_state"][:2])
        example = {
            "recording": row["recording_group"],
            "posthoc_turn_bin": row["posthoc_turn_bin"],
            "current_frame_token": str(current["token"]),
            "hypothesis_overlay": str(
                output_directory / f"hypotheses_{example_index}.png"
            ),
            "pixel_rgb_mean_absolute_change_0_255": float(
                np.abs(images[False][-1].astype(float) - images[True][-1]).mean()
            ),
            "mean_distortion_roi_coordinate_change_pixels": float(
                (pinhole_boxes[common] - distorted_boxes[common]).abs().mean()
            )
            if common.any()
            else None,
            "grid_feature_rms_change": float(
                (grids[0] - grids[1]).square().mean().sqrt()
            ),
            "roi_feature_rms_change_raw_pinhole_to_rectified": float(
                (raw_pinhole_rois[common] - rectified_rois[common])
                .square()
                .mean()
                .sqrt()
            )
            if common.any()
            else None,
            "roi_feature_rms_change_raw_distorted_to_rectified": float(
                (raw_distorted_rois[common] - rectified_rois[common])
                .square()
                .mean()
                .sqrt()
            )
            if common.any()
            else None,
            "current_native_velocity_meters_per_second": native_velocity.tolist(),
            "history_pose_difference_velocity_in_current_frame": past_delta.tolist(),
            "past_only_velocity_consistency_error_meters_per_second": float(
                np.linalg.norm(native_velocity - past_delta)
            ),
            "source_image": str(front_image_path(current, sensor_root)),
        }
        report["examples"].append(example)
        print(
            f"PREPROCESS_AUDIT example={example_index + 1} elapsed={time.perf_counter() - started:.2f}s",
            flush=True,
        )
    report["wall_seconds"] = time.perf_counter() - started
    report["peak_process_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    (output_directory / "preprocessing_evidence.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "resolved": report["resolved"],
                "examples": len(report["examples"]),
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
