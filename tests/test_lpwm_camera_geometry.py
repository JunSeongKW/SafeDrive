"""Geometry gates for real camera warps and the official LPWM coordinate convention."""
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from planning_aware_future_prediction.object_centric.lpwm_bridge import (
    OFFICIAL_ROOT, extrapolated_rotation_homographies, particle_geometry, transform_points,
)


def test_rotation_compensation_matches_projection_of_world_rays():
    intrinsics = np.array([[95., 0, 64], [0, 110., 61], [0, 0, 1.]])
    reference_rotation = Rotation.from_euler("xyz", [3, -6, 5], degrees=True).as_matrix()
    current_rotation = Rotation.from_euler("xyz", [7, 12, 15], degrees=True).as_matrix()
    world_rays = np.array([[.1, .2, 1], [-.2, .1, 1], [.3, -.1, 1]])
    reference_pixels = world_rays @ reference_rotation @ intrinsics.T
    current_pixels = world_rays @ current_rotation @ intrinsics.T
    reference_pixels = reference_pixels[:, :2] / reference_pixels[:, 2:]
    current_pixels = current_pixels[:, :2] / current_pixels[:, 2:]
    homography = intrinsics @ reference_rotation.T @ current_rotation @ np.linalg.inv(intrinsics)
    np.testing.assert_allclose(transform_points(current_pixels, homography), reference_pixels, atol=1e-8)


def test_forecast_rotation_matches_constant_observed_angular_velocity():
    angular_increment = np.array([.01, .03, -.02])
    observed_rotations = np.array([Rotation.from_rotvec(angular_increment * step).as_matrix() for step in range(4)])
    intrinsics = np.repeat(np.array([[[100., 0, 64], [0, 100., 64], [0, 0, 1.]]]), 4, axis=0)
    predicted = extrapolated_rotation_homographies(observed_rotations, intrinsics, 4)
    for future_step, transform in enumerate(predicted, 4):
        expected = intrinsics[0] @ Rotation.from_rotvec(angular_increment * future_step).as_matrix() @ np.linalg.inv(intrinsics[0])
        np.testing.assert_allclose(transform, expected, atol=1e-8)


def test_particle_centers_match_official_stn_rendering():
    sys.path.insert(0, str(OFFICIAL_ROOT))
    from utils.util_func import spatial_transform
    normalized_center = torch.tensor([[[[.4, -.3]]]])
    encoded = {"z": normalized_center, "z_scale": torch.full_like(normalized_center, -1.), "obj_on": torch.ones(1, 1, 1)}
    centers, _, _, _ = particle_geometry(encoded, 0, 1)
    glimpse = torch.zeros(1, 1, 9, 9)
    glimpse[0, 0, 4, 4] = 1
    rendered = spatial_transform(glimpse, normalized_center.reshape(1, 2), torch.ones(1, 2) * .25,
                                 (1, 1, 128, 128), inverse=True)[0, 0]
    row, column = np.unravel_index(int(rendered.argmax()), (128, 128))
    np.testing.assert_allclose(centers[0], [column, row], atol=.51)
