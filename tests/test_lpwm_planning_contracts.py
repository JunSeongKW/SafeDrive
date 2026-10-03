"""Coordinate and privileged-target masking contracts for LPWM planning."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

from planning_aware_future_prediction.object_centric.lpwm_planner import particle_image_boxes, compute_planning_objectives


def test_particle_boxes_use_height_width_and_pixel_centers():
    centers = torch.tensor([[[.5, -.5]]])
    scales = torch.zeros_like(centers)
    boxes = particle_image_boxes(centers, scales)
    expected_center = torch.tensor([[[.25 - .5 / 128, .75 - .5 / 128]]])
    torch.testing.assert_close((boxes[..., :2] + boxes[..., 2:]) / 2, expected_center)


def test_future_object_coordinates_use_current_ego_not_future_camera():
    script_path = Path(__file__).resolve().parents[1] / "scripts/prepare_lpwm_planning_data.py"
    specification = importlib.util.spec_from_file_location("prepare_lpwm_planning_data", script_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    current_pose = np.eye(4)
    current_pose[:2, :2] = [[0, -1], [1, 0]]
    current_pose[0, 3] = 10
    future_pose = np.eye(4)
    future_pose[:2, 3] = [10, 5]
    position = module.transform_box_center_to_current_ego(np.array([2, 0, 0, 1, 1, 1, 0]), {"ego2global": future_pose, "lidar2ego": np.eye(4)}, {"ego2global": current_pose})
    np.testing.assert_allclose(position, [5, -2])


def test_invalid_future_targets_do_not_change_objective_and_valid_futures_have_gradient():
    predictions = {"trajectory": torch.zeros(1, 8, 3, requires_grad=True), "current_boxes": torch.rand(1, 64, 4, requires_grad=True),
                   "current_xy": torch.zeros(1, 64, 2, requires_grad=True), "future_xy": torch.zeros(1, 64, 8, 2, requires_grad=True),
                   "object_class_logits": torch.zeros(1, 64, 4, requires_grad=True)}
    supervision = {"ego_trajectory_target": torch.zeros(1, 8, 3), "object_valid": torch.tensor([[True]]),
                   "object_boxes": torch.tensor([[[.4, .4, .6, .6]]]), "object_categories": torch.tensor([[1]]),
                   "object_current_xy": torch.ones(1, 1, 2), "object_future_xy": torch.ones(1, 1, 8, 2),
                   "object_future_valid": torch.tensor([[[True, True, False, False, False, False, False, False]]]),
                   "object_risk_weights": torch.ones(1, 1)}
    first = compute_planning_objectives(predictions, supervision, "object_future_risk")["total"]
    supervision["object_future_xy"][:, :, 2:] = 10000
    second = compute_planning_objectives(predictions, supervision, "object_future_risk")["total"]
    torch.testing.assert_close(first, second)
    second.backward()
    assert predictions["future_xy"].grad[:, :, :2].abs().sum() > 0
    assert predictions["future_xy"].grad[:, :, 2:].abs().sum() == 0
