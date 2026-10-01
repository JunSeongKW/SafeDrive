"""Compare stored-image projection hypotheses; write only workspace outputs."""

import argparse
import hashlib
import io
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
)
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    front_track_rois,
    load_front_image,
)


class CameraMetadataUnpickler(pickle.Unpickler):
    """Decode only the official CameraIntrinsic list wrapper, not the devkit."""

    def find_class(self, module, name):
        if (module, name) == ("nuplan.database.common.data_types", "CameraIntrinsic"):
            return list
        raise pickle.UnpicklingError(f"unexpected calibration type: {module}.{name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/projection_audit/front_camera_rectified_20261001",
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if not output_directory.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("audit output must be inside project outputs")
    if output_directory.exists():
        raise RuntimeError("refusing to overwrite projection audit")
    output_directory.mkdir(parents=True)
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    sensor_root = dataset_root / "sensor_blobs/mini"
    raw = json.loads(
        (
            PROJECT_ROOT
            / "outputs/data_surveys/navsim_visual_target_coverage_v1_20261001/visual_target_coverage.json"
        ).read_text()
    )
    chosen = {}
    for record in raw["windows"]:
        chosen.setdefault((record["split"], record["posthoc_turn_bin"]), record)
    report = {"examples": [], "projection_only_not_occlusion": True}
    for example_index, record in enumerate(chosen.values()):
        log_path = dataset_root / "navsim_logs/mini" / record["segment_filename"]
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)
        start = record["start_index"]
        history = frames[start : start + 4]
        online = build_tracked_state_online_inputs(history, TrackedStateAdapterConfig())
        sheet = Image.new("RGB", (1024, 840), "#101010")
        example = {"recording": record["recording_group"], "start": start, "panels": []}
        for row, offset in enumerate((0, 4, 8)):
            frame = frames[start + 3 + offset]
            roi_variants = []
            for column, rectify_image in enumerate((False, True)):
                rgb = load_front_image(
                    frame, sensor_root, rectify_stored_image=rectify_image
                )
                boxes, valid = front_track_rois(
                    frame, online.current_track_tokens, use_opencv_pixel_centers=True
                )
                panel = Image.fromarray(rgb.copy())
                painter = ImageDraw.Draw(panel)
                for entity_index in np.flatnonzero(valid.numpy()):
                    painter.rectangle(
                        boxes[entity_index].tolist(),
                        outline="#26ef80" if column == 0 else "#ffb43b",
                        width=1,
                    )
                    painter.text(
                        (float(boxes[entity_index, 0]), float(boxes[entity_index, 1])),
                        str(entity_index),
                        fill="white",
                    )
                sheet.paste(panel, (column * 512, row * 280 + 24))
                ImageDraw.Draw(sheet).text(
                    (column * 512 + 5, row * 280 + 6),
                    f"+{offset * 0.5:.1f}s {'ORIGINAL / PINHOLE' if column == 0 else 'RECTIFIED / PINHOLE'}",
                    fill="white",
                )
                roi_variants.append((boxes, valid))
            both = roi_variants[0][1] & roi_variants[1][1]
            example["panels"].append(
                {
                    "offset_seconds": offset * 0.5,
                    "pinhole_valid": int(roi_variants[0][1].sum()),
                    "rectified_pinhole_valid": int(roi_variants[1][1].sum()),
                    "mean_roi_coordinate_shift_pixels": float(
                        (roi_variants[0][0][both] - roi_variants[1][0][both])
                        .abs()
                        .mean()
                    )
                    if both.any()
                    else None,
                }
            )
        sheet_path = output_directory / f"projection_hypotheses_{example_index}.png"
        sheet.save(sheet_path)
        example["review_image"] = str(sheet_path.relative_to(PROJECT_ROOT))
        current = history[-1]
        example["camera_intrinsic"] = np.asarray(
            current["cams"]["CAM_F0"]["cam_intrinsic"]
        ).tolist()
        example["camera_distortion"] = np.asarray(
            current["cams"]["CAM_F0"]["distortion"]
        ).tolist()
        intrinsic = np.asarray(example["camera_intrinsic"], dtype=np.float64)
        distortion = np.asarray(example["camera_distortion"], dtype=np.float64)
        # Test in the actual calibrated front FOV, not near-plane extrapolation.
        camera_points = np.array(
            [
                [horizontal, vertical, 1.0]
                for horizontal in np.linspace(-0.5, 0.5, 11)
                for vertical in np.linspace(-0.25, 0.25, 7)
            ]
        )
        distorted_pixels = cv2.projectPoints(
            camera_points, np.zeros(3), np.zeros(3), intrinsic, distortion
        )[0]
        recovered_pixels = cv2.undistortPoints(
            distorted_pixels,
            intrinsic,
            distortion,
            R=None,
            P=intrinsic,
            criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
        ).reshape(-1, 2)
        expected_pixels = (camera_points @ intrinsic.T)[:, :2]
        example["distort_undistort_roundtrip_max_pixels"] = float(
            np.abs(recovered_pixels - expected_pixels).max()
        )
        database_path = Path(
            "/home/user/data/Dataset/nuplan/dataset/nuplan-v1.1/splits/mini"
        ) / (log_path.stem + ".db")
        if database_path.is_file():
            with sqlite3.connect(
                f"file:{database_path}?mode=ro&immutable=1", uri=True
            ) as connection:
                row = connection.execute(
                    "SELECT intrinsic, distortion, width, height FROM camera WHERE channel='CAM_F0'"
                ).fetchone()
                intrinsic = CameraMetadataUnpickler(io.BytesIO(row[0])).load()
                distortion = CameraMetadataUnpickler(io.BytesIO(row[1])).load()
                example["metadata_matches_original_camera_intrinsic"] = bool(
                    np.allclose(intrinsic, example["camera_intrinsic"])
                )
                example["metadata_matches_original_camera_distortion"] = bool(
                    np.allclose(distortion, example["camera_distortion"])
                )
                example["original_camera_image_size"] = list(row[2:])
        report["examples"].append(example)
    # Exact image-coordinate transform, including OpenCV half-pixel resize convention.
    report["crop_resize_transform"] = {
        "raw_size": [1920, 1080],
        "crop_top_bottom_pixels": 28,
        "cropped_size": [1920, 1024],
        "resized_size": [512, 256],
        "box_edge_coordinates": "x*512/1920, (y-28)*256/1024",
        "pixel_centers": "(x+.5)*512/1920-.5, (y-28+.5)*256/1024-.5",
        "feature_stride": 16,
        "roi_align_aligned": True,
        "resize_edge_vs_center_max_difference_pixels": 0.375,
    }
    report["source_script_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["chosen_operational_convention"] = (
        "Treat stored JPEG as original-distorted; remap with stored K/D and unchanged K, then crop/resize; pinhole ROI with OpenCV pixel-center resize coordinates."
    )
    report["unresolved_provenance"] = (
        "Separate original sensor JPEG is unavailable for byte comparison. Metadata/file-path agreement and primary OpenScene/nuPlan descriptions support this convention but do not prove local export history."
    )
    report["visibility_limit"] = (
        "Projection is geometric eligibility, not visibility; occlusion and image/lidar timing remain unverified."
    )
    report["primary_sources"] = [
        "https://github.com/OpenDriveLab/OpenScene/blob/main/docs/getting_started.md",
        "https://github.com/motional/nuplan-devkit/blob/master/docs/nuplan_schema.md",
        "https://arxiv.org/html/2503.12552v2#A1",
        "https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html",
    ]
    (output_directory / "projection_audit.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
