"""Read-only, front-camera GT-box/track pilot; NOT a learned detector.

Image appearance and explicit metric motion targets are separate. Future camera
visibility is a TRAINING loss mask, never an online selection input.
"""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.ops import roi_align

from .navsim_tracked_state import annotation_track_lookup


def project_lidar_box_to_front_roi(
    lidar_box: np.ndarray,
    camera: Mapping,
    minimum_depth: float = 0.1,
    apply_stored_distortion: bool = False,
    use_opencv_pixel_centers: bool = False,
) -> tuple[np.ndarray, bool]:
    """Full 3D cuboid -> clipped ROI in the 512x256 official cropped image.

    Defaults to the historical official pinhole convention. The explicit
    distortion option projects onto the stored Brown-Conrady camera image.
    Image export/rectification provenance must be audited before choosing it.
    Near-plane edge clipping avoids projecting behind-camera cuboid vertices.
    """
    lidar_box = np.asarray(lidar_box, dtype=np.float64)
    if (
        lidar_box.shape != (7,)
        or not np.isfinite(lidar_box).all()
        or np.any(lidar_box[3:6] <= 0)
        or minimum_depth <= 0
    ):
        return np.zeros(4, dtype=np.float32), False
    corner_signs = np.array(
        [
            [horizontal, lateral, vertical]
            for horizontal in (-0.5, 0.5)
            for lateral in (-0.5, 0.5)
            for vertical in (-0.5, 0.5)
        ]
    )
    cosine, sine = np.cos(lidar_box[6]), np.sin(lidar_box[6])
    yaw_rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    lidar_corners = (corner_signs * lidar_box[3:6]) @ yaw_rotation.T + lidar_box[:3]
    sensor_rotation = np.asarray(camera["sensor2lidar_rotation"])
    sensor_translation = np.asarray(camera["sensor2lidar_translation"])
    camera_corners = (lidar_corners - sensor_translation) @ np.linalg.inv(
        sensor_rotation
    ).T
    clipped_points = list(camera_corners[camera_corners[:, 2] >= minimum_depth])
    for first_corner in range(8):
        for second_corner in range(first_corner + 1, 8):
            if (
                np.count_nonzero(
                    corner_signs[first_corner] != corner_signs[second_corner]
                )
                != 1
            ):
                continue
            first_point, second_point = camera_corners[[first_corner, second_corner]]
            if (first_point[2] < minimum_depth) != (second_point[2] < minimum_depth):
                fraction = (minimum_depth - first_point[2]) / (
                    second_point[2] - first_point[2]
                )
                clipped_points.append(
                    first_point + fraction * (second_point - first_point)
                )
    if not clipped_points:
        return np.zeros(4, dtype=np.float32), False
    if apply_stored_distortion:
        pixels = cv2.projectPoints(
            np.asarray(clipped_points, dtype=np.float64),
            np.zeros(3),
            np.zeros(3),
            np.asarray(camera["cam_intrinsic"], dtype=np.float64),
            np.asarray(camera["distortion"], dtype=np.float64),
        )[0].reshape(-1, 2)
    else:
        projected = np.asarray(clipped_points) @ np.asarray(camera["cam_intrinsic"]).T
        pixels = projected[:, :2] / projected[:, 2:3]
    # Dataset image headers are checked by load_front_image: 1920x1080.
    pixels[:, 0] *= 512 / 1920
    pixels[:, 1] = (pixels[:, 1] - 28) * 256 / 1024
    if use_opencv_pixel_centers:
        pixels += np.array([0.5 * 512 / 1920 - 0.5, 0.5 * 256 / 1024 - 0.5])
    roi = np.concatenate((pixels.min(axis=0), pixels.max(axis=0)))
    roi[[0, 2]] = np.clip(roi[[0, 2]], 0, 512)
    roi[[1, 3]] = np.clip(roi[[1, 3]], 0, 256)
    valid = bool(np.isfinite(roi).all() and np.all(roi[2:] - roi[:2] >= 1.0))
    return roi.astype(np.float32) if valid else np.zeros(4, dtype=np.float32), valid


def front_track_rois(
    frame: Mapping,
    current_track_tokens: Sequence[str],
    apply_stored_distortion: bool = False,
    use_opencv_pixel_centers: bool = False,
) -> tuple[Tensor, Tensor]:
    """Associate ALL annotations by track before projecting; no future radius filter."""
    lookup = annotation_track_lookup(frame)
    roi_boxes, projected_valid = [], []
    for track_token in current_track_tokens:
        if track_token in lookup:
            roi, valid = project_lidar_box_to_front_roi(
                frame["anns"]["gt_boxes"][lookup[track_token]],
                frame["cams"]["CAM_F0"],
                apply_stored_distortion=apply_stored_distortion,
                use_opencv_pixel_centers=use_opencv_pixel_centers,
            )
        else:
            roi, valid = np.zeros(4, dtype=np.float32), False
        roi_boxes.append(roi)
        projected_valid.append(valid)
    return (
        torch.tensor(np.asarray(roi_boxes).reshape(-1, 4), dtype=torch.float32),
        torch.tensor(projected_valid, dtype=torch.bool),
    )


def front_image_path(frame: Mapping, sensor_blobs_root: Path) -> Path:
    root = sensor_blobs_root.resolve()
    path = (root / frame["cams"]["CAM_F0"]["data_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("camera path escapes sensor root")
    return path


@lru_cache(maxsize=16)
def front_rectification_maps(intrinsic_values: tuple, distortion_values: tuple):
    intrinsic = np.array(intrinsic_values, dtype=np.float64).reshape(3, 3)
    return cv2.initUndistortRectifyMap(
        intrinsic,
        np.array(distortion_values, dtype=np.float64),
        None,
        intrinsic,
        (1920, 1080),
        cv2.CV_32FC1,
    )


def load_front_image(
    frame: Mapping, sensor_blobs_root: Path, rectify_stored_image: bool = False
) -> np.ndarray:
    """Match official RGB PIL decode, 28px top/bottom crop, cv2 linear resize."""
    with Image.open(front_image_path(frame, sensor_blobs_root)) as image:
        if image.size != (1920, 1080):
            raise ValueError(f"unsupported image size: {image.size}")
        rgb_image = np.asarray(image.convert("RGB"))
    if rectify_stored_image:
        camera = frame["cams"]["CAM_F0"]
        map_horizontal, map_vertical = front_rectification_maps(
            tuple(np.asarray(camera["cam_intrinsic"]).flatten()),
            tuple(np.asarray(camera["distortion"]).flatten()),
        )
        rgb_image = cv2.remap(rgb_image, map_horizontal, map_vertical, cv2.INTER_LINEAR)
    return cv2.resize(rgb_image[28:-28], (512, 256), interpolation=cv2.INTER_LINEAR)


def build_front_video_clip(
    two_observed_frames: Sequence[Mapping],
    sensor_blobs_root: Path,
    rectify_stored_image: bool = False,
) -> Tensor:
    if len(two_observed_frames) != 2:
        raise ValueError("encoder clip needs two observed frames")
    image_tensors = [
        torch.from_numpy(
            load_front_image(frame, sensor_blobs_root, rectify_stored_image).copy()
        )
        .permute(2, 0, 1)
        .float()
        / 255
        for frame in two_observed_frames
    ]
    video_clip = torch.stack(image_tensors, dim=1)[None]
    image_mean = video_clip.new_tensor([0.485, 0.456, 0.406])[None, :, None, None, None]
    image_std = video_clip.new_tensor([0.229, 0.224, 0.225])[None, :, None, None, None]
    return (video_clip - image_mean) / image_std


def pool_projected_entity_rois(
    feature_grid: Tensor, roi_boxes: Tensor, projected_valid: Tensor
) -> Tensor:
    """ROIAlign raw teacher features; invalid regions are sanitized before pooling."""
    if feature_grid.shape[0] != 1 or tuple(feature_grid.shape[-2:]) != (16, 32):
        raise ValueError("expected single 16x32 encoder grid")
    roi_boxes = roi_boxes.to(device=feature_grid.device, dtype=feature_grid.dtype)
    projected_valid = projected_valid.to(feature_grid.device)
    if roi_boxes.shape != (projected_valid.numel(), 4):
        raise ValueError("ROI and validity shapes differ")
    if not torch.isfinite(roi_boxes[projected_valid]).all():
        raise ValueError("valid ROI contains non-finite coordinates")
    output = feature_grid.new_zeros(projected_valid.numel(), feature_grid.shape[1])
    if projected_valid.any():
        pooled = roi_align(
            feature_grid,
            [roi_boxes[projected_valid]],
            output_size=3,
            spatial_scale=1 / 16,
            sampling_ratio=2,
            aligned=True,
        ).mean(dim=(-2, -1))
        output[projected_valid] = pooled
    return output[None]
