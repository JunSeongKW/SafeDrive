import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    pool_projected_entity_rois,
    project_lidar_box_to_front_roi,
)
from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    compute_parameter_gradient_norm,
)
from planning_aware_future_prediction.models.visual_entity_future_planning_pilot import (
    VisualEntityFuturePlanningPilot,
)


class VisualPilotContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.model = VisualEntityFuturePlanningPilot(
            visual_feature_dim=8, num_future_steps=2, num_selected_entities=2
        ).eval()
        self.inputs = (
            torch.randn(1, 8, 16, 32),
            torch.randn(1, 4, 18),
            torch.tensor([[True, True, True, False]]),
            torch.tensor([[3, 1, 2, 4]]),
            torch.randn(1, 8),
        )

    def test_projection_front_behind_and_crop(self):
        camera = {
            "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.zeros(3),
            "cam_intrinsic": np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]]),
        }
        projected, valid = project_lidar_box_to_front_roi(
            np.array([0, 0, 10, 2, 2, 2, 0]), camera
        )
        self.assertTrue(valid)
        np.testing.assert_allclose(
            (projected[:2] + projected[2:]) / 2, [256, 128], atol=1e-5
        )
        self.assertFalse(
            project_lidar_box_to_front_roi(np.array([0, 0, -10, 2, 2, 2, 0]), camera)[1]
        )
        self.assertFalse(
            project_lidar_box_to_front_roi(np.array([0, 0, 10, -1, 2, 2, 0]), camera)[1]
        )
        self.assertTrue(
            project_lidar_box_to_front_roi(np.array([0, 0, 0.2, 2, 2, 2, 0]), camera)[1]
        )

    def test_roi_pool_invalid_nan_is_zero(self):
        features = torch.ones(1, 8, 16, 32)
        boxes = torch.tensor([[32.0, 32, 64, 64], [float("nan")] * 4])
        pooled = pool_projected_entity_rois(
            features, boxes, torch.tensor([True, False])
        )
        torch.testing.assert_close(pooled[0, 0], torch.ones(8))
        self.assertTrue(torch.equal(pooled[0, 1], torch.zeros(8)))

    def test_visual_and_spatial_auxiliary_do_not_train_selector_or_planner(self):
        output = self.model(*self.inputs)
        targets = self.model.detached_selection_auxiliary_losses(
            output,
            self.inputs[0],
            self.inputs[1],
            self.inputs[2],
            self.inputs[4],
            torch.randn(1, 4, 2, 8),
            torch.ones(1, 4, 2, dtype=torch.bool),
            torch.randn(1, 4, 2, 6),
            torch.ones(1, 4, 2, dtype=torch.bool),
        )
        for loss in targets.values():
            self.assertEqual(
                compute_parameter_gradient_norm(loss, self.model.entity_scorer), 0
            )
            self.assertEqual(
                compute_parameter_gradient_norm(loss, self.model.ego_planner), 0
            )
            self.assertGreater(
                compute_parameter_gradient_norm(loss, self.model.future_predictor), 0
            )
        self.assertGreater(
            compute_parameter_gradient_norm(
                output.ego_trajectory.square().mean(), self.model.entity_scorer
            ),
            0,
        )

    def test_future_detach_preserves_forward_no_future_branch_skips_prediction(self):
        reference = self.model(*self.inputs)
        detached = self.model(*self.inputs, detach_predicted_future=True)
        self.assertTrue(torch.equal(reference.ego_trajectory, detached.ego_trajectory))
        self.assertEqual(
            compute_parameter_gradient_norm(
                detached.ego_trajectory.square().mean(), self.model.future_predictor
            ),
            0,
        )
        without_future = self.model(*self.inputs, enable_future_branch=False)
        self.assertIsNone(without_future.entity_selection)
        self.assertIsNone(without_future.predicted_future_visual_latents)
        self.assertEqual(
            compute_parameter_gradient_norm(
                without_future.ego_trajectory.square().mean(), self.model.entity_scorer
            ),
            0,
        )

    def test_joint_entity_permutation_invariance(self):
        reference = self.model(*self.inputs)
        order = torch.tensor([2, 0, 3, 1])
        permuted = self.model(
            self.inputs[0],
            self.inputs[1][:, order],
            self.inputs[2][:, order],
            self.inputs[3][:, order],
            self.inputs[4],
        )
        torch.testing.assert_close(
            reference.ego_trajectory, permuted.ego_trajectory, atol=1e-6, rtol=1e-5
        )


if __name__ == "__main__":
    unittest.main()
