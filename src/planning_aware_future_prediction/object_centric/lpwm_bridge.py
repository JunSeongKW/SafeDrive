"""Strict official LPWM loading; explicit coordinates and causal camera compensation."""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_ROOT = PROJECT_ROOT / "reference_repositories/LPWM"
ARTIFACT_ROOT = PROJECT_ROOT / "outputs/lpwm_navsim_adaptation_v1"


def load_official_lpwm(device="cpu", checkpoint_path=None):
    sys.path.insert(0, str(OFFICIAL_ROOT))
    import models as official_models
    from models import DLP
    if not getattr(official_models, "_navsim_contiguous_kl_adapter", False):
        original_dynamic_kl = official_models.calc_dynamic_kl
        def contiguous_dynamic_kl(*arguments, **keyword_arguments):
            # Official JIT uses .view on sliced (batch,time,particle,attribute) tensors.
            # Copy strides only; preserve tensor values, gradients and objective.
            arguments = tuple(value.contiguous() if torch.is_tensor(value) else value for value in arguments)
            keyword_arguments = {key: value.contiguous() if torch.is_tensor(value) else value for key, value in keyword_arguments.items()}
            return original_dynamic_kl(*arguments, **keyword_arguments)
        official_models.calc_dynamic_kl = contiguous_dynamic_kl
        official_models._navsim_contiguous_kl_adapter = True
    config = json.loads((ARTIFACT_ROOT / "pretrained/hparams.json").read_text())
    parameter_names = inspect.signature(DLP.__init__).parameters
    aliases = {"ch": "cdim", "num_static_frames": "n_static_frames", "context_dist": "ctx_dist",
               "image_goal_condition": "img_goal_condition"}
    arguments = {aliases.get(key, key): value for key, value in config.items() if aliases.get(key, key) in parameter_names}
    model = DLP(**arguments)
    if checkpoint_path is None:
        checkpoint_path = next((ARTIFACT_ROOT / "pretrained").glob("*best_lpips.pth"))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    # The official ELBO reshapes by this scalar, not by the incoming sequence length.
    # Retain every learned 20-step embedding; use its first 8 positions without resizing.
    model.timestep_horizon = 7
    return model.to(device), config


def image_tensor(images, device):
    return torch.as_tensor(np.ascontiguousarray(images), device=device).permute(0, 3, 1, 2).float().div(255)


def particle_geometry(encoded, frame_index, budget=16):
    """Official position and scale are ordered (height, width), despite an x/y comment."""
    centers = (encoded["z"][0, frame_index].detach().cpu().numpy()[:, ::-1] + 1) * 64 - .5
    sizes = torch.sigmoid(encoded["z_scale"][0, frame_index]).detach().cpu().numpy()[:, ::-1] * 128
    presence = encoded["obj_on"][0, frame_index].detach().cpu().numpy().reshape(-1)
    chosen = np.argsort(-presence, kind="stable")[:budget]
    return centers[chosen], sizes[chosen], presence[chosen], chosen


def transform_points(points, homography):
    return cv2.perspectiveTransform(np.asarray(points, dtype=np.float64).reshape(1, -1, 2), homography)[0]


def restore_particle_geometry(centers, sizes, homography):
    inverse_transform = np.linalg.inv(homography)
    corners = centers[:, None, :] + sizes[:, None, :] * np.array([[[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]])
    restored_corners = transform_points(corners.reshape(-1, 2), inverse_transform).reshape(-1, 4, 2)
    return transform_points(centers, inverse_transform), np.concatenate([restored_corners.min(1), restored_corners.max(1)], axis=-1)


def extrapolated_rotation_homographies(observed_camera_rotations, observed_camera_intrinsics, future_steps):
    """Accept observed arrays only. Future camera rotation is constant-rate extrapolation."""
    assert len(observed_camera_rotations) == len(observed_camera_intrinsics) and len(observed_camera_rotations) >= 2
    previous_rotation, current_rotation = observed_camera_rotations[-2:]
    rotation_increment = Rotation.from_matrix(previous_rotation.T @ current_rotation).as_rotvec()
    reference_rotation = observed_camera_rotations[0]
    reference_intrinsics = observed_camera_intrinsics[0]
    current_intrinsics = observed_camera_intrinsics[-1]
    return np.array([reference_intrinsics @ reference_rotation.T @ current_rotation @ Rotation.from_rotvec(rotation_increment * future_index).as_matrix() @ np.linalg.inv(current_intrinsics)
                     for future_index in range(1, future_steps + 1)])


def checkpoint_digest(checkpoint_path):
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as stream:
        while chunk := stream.read(1024**2):
            digest.update(chunk)
    return digest.hexdigest()
