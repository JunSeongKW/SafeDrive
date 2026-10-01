"""Regression tests for the fixed-rule cache/training contracts, not research results."""

import unittest

import numpy as np
import torch

from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    project_lidar_box_to_front_roi,
)
from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
    FixedDistanceFutureSupervisionPilot,
    gather_selected_training_targets,
    masked_prediction_mse,
    select_nearest_current_entities,
    train_only_target_normalization,
)


class FixedDistanceTargetSupervisionTests(unittest.TestCase):
    def test_current_distance_identity_ties_and_padding(self):
        features = torch.zeros(1, 3, 12)
        features[0, :, -10] = torch.tensor([1.0, 1.0, 0.1])
        valid = torch.tensor([[True, True, False]])
        identities = torch.tensor([[8, 2, 1]])
        selection = select_nearest_current_entities(features, valid, identities)
        self.assertEqual(selection.selected_entity_indices.tolist(), [[1, 0, -1, -1]])
        order = torch.tensor([2, 0, 1])
        permuted = select_nearest_current_entities(
            features[:, order], valid[:, order], identities[:, order]
        )
        self.assertTrue(
            torch.equal(
                selection.selection_weights @ features,
                permuted.selection_weights @ features[:, order],
            )
        )
        empty = select_nearest_current_entities(
            features[:, :0], valid[:, :0], identities[:, :0]
        )
        self.assertFalse(empty.selected_entity_valid_mask.any())
        self.assertEqual(empty.selection_weights.shape, (1, 4, 0))

    def normalization_fixture(self, split="train"):
        return {
            "metadata": {"split": split},
            "online_inputs": {
                "current_entity_features": torch.zeros(2, 12),
                "current_entity_valid_mask": torch.tensor([True, True]),
                "stable_entity_ids": torch.tensor([1, 2]),
            },
            "training_targets": {
                "future_visual_targets": torch.tensor([[[2.0, 4.0]], [[100.0, 100.0]]]),
                "future_spatial_targets": torch.tensor(
                    [[[3.0, 6.0]], [[100.0, 100.0]]]
                ),
                "future_visual_valid_mask": torch.tensor([[True], [False]]),
                "future_spatial_valid_mask": torch.tensor([[True], [True]]),
            },
        }

    def test_normalization_is_train_only_selected_common_population(self):
        window = self.normalization_fixture()
        normalization = train_only_target_normalization([window])
        self.assertTrue(
            torch.equal(normalization["visual"]["mean"], torch.tensor([2.0, 4.0]))
        )
        self.assertEqual(normalization["visual"]["observations"], 1)
        self.assertTrue(
            torch.equal(normalization["visual"]["std"], torch.tensor([0.05, 0.05]))
        )
        with self.assertRaisesRegex(ValueError, "development"):
            train_only_target_normalization([self.normalization_fixture("development")])

    def test_common_mask_is_loss_only_and_zero_valid_loss_is_zero(self):
        window = self.normalization_fixture()
        online = window["online_inputs"]
        selection = select_nearest_current_entities(
            *(
                online[name][None]
                for name in (
                    "current_entity_features",
                    "current_entity_valid_mask",
                    "stable_entity_ids",
                )
            )
        )
        selected = gather_selected_training_targets(
            selection,
            {name: value[None] for name, value in window["training_targets"].items()},
        )
        self.assertEqual(int(selected["common_valid"].sum()), 1)
        self.assertEqual(int(selected["spatial_valid"].sum()), 2)
        prediction = torch.ones(1, 4, 1, 2, requires_grad=True)
        loss = masked_prediction_mse(
            prediction,
            torch.zeros_like(prediction),
            torch.zeros(1, 4, 1, dtype=torch.bool),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertEqual(float(prediction.grad.norm()), 0.0)
        self.assertEqual(selection.selected_entity_indices.tolist(), [[0, 1, -1, -1]])

    def test_inactive_modules_and_auxiliary_gradient_contract(self):
        torch.manual_seed(29)
        model = FixedDistanceFutureSupervisionPilot(visual_feature_dim=8)
        features = torch.randn(2, 5, 18)
        current = {
            "current_image_grid": torch.randn(2, 8, 16, 32),
            "current_entity_features": features,
            "current_entity_valid_mask": torch.ones(2, 5, dtype=torch.bool),
            "stable_entity_ids": torch.arange(5)[None].expand(2, -1),
            "current_ego_status": torch.randn(2, 8),
        }
        model.configure_active_modules(True)
        model.entity_scorer.forward = lambda *args: self.fail(
            "learned scorer must never be called"
        )
        output = model(**current)
        output.predicted_future_visual_latents.square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.future_predictor.parameters()
            )
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.ego_planner.parameters())
        )
        model.zero_grad(set_to_none=True)
        model.configure_active_modules(False)
        output = model(**current, enable_future_branch=False)
        output.ego_trajectory.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.future_predictor.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.ego_planner.predicted_future_adapter.parameters()
            )
        )

    def test_resize_pixel_center_transform(self):
        camera = {
            "sensor2lidar_rotation": np.eye(3),
            "sensor2lidar_translation": np.zeros(3),
            "cam_intrinsic": np.array(
                [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]
            ),
        }
        box = np.array([0.0, 0.0, 20.0, 2.0, 2.0, 2.0, 0.0])
        legacy, legacy_valid = project_lidar_box_to_front_roi(box, camera)
        corrected, corrected_valid = project_lidar_box_to_front_roi(
            box, camera, use_opencv_pixel_centers=True
        )
        self.assertTrue(legacy_valid and corrected_valid)
        np.testing.assert_allclose(
            corrected - legacy, [-0.3666667, -0.375, -0.3666667, -0.375], atol=2e-5
        )


if __name__ == "__main__":
    unittest.main()
