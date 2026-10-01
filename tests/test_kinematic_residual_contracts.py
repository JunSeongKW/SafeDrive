import sys
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_kinematic_and_ridge_references import (
    apply_fitted_ridge,
    fit_ridge_from_training,
)

from planning_aware_future_prediction.models.kinematic_future_priors import (
    predict_ego_kinematic_trajectory,
    predict_entity_constant_velocity_state,
)
from planning_aware_future_prediction.models.residual_future_supervision import (
    ResidualFutureSupervisionPilot,
)


class KinematicResidualContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def inputs(self, visual_dim=16):
        return {
            "current_image_grid": torch.randn(2, visual_dim, 16, 32),
            "current_entity_features": torch.randn(2, 6, visual_dim + 10),
            "current_entity_valid_mask": torch.tensor([[True] * 6, [False] * 6]),
            "stable_entity_ids": torch.arange(6).expand(2, -1),
            "current_ego_status": torch.randn(2, 8),
        }

    def model(self, **flags):
        model = ResidualFutureSupervisionPilot(visual_feature_dim=16, **flags)
        model.set_train_target_normalization(
            {
                name: {
                    "mean": torch.randn(dimension),
                    "std": torch.rand(dimension) + 0.2,
                }
                for name, dimension in (("visual", 16), ("spatial", 6))
            }
        )
        model.initialize_residual_output_heads()
        return model

    def test_ego_velocity_acceleration_units_and_zero_heading(self):
        status = torch.tensor([[0.0, 1.0, 0.0, 0.0, 2.0, -1.0, 0.5, 0.25]])
        prediction = predict_ego_kinematic_trajectory(status)
        torch.testing.assert_close(prediction[0, -1], torch.tensor([12.0, -2.0, 0.0]))
        torch.testing.assert_close(
            predict_ego_kinematic_trajectory(status, motion_model="constant_velocity")[
                0, -1
            ],
            torch.tensor([8.0, -4.0, 0.0]),
        )

    def test_entity_prior_fixed_frame_scaled_motion_and_heading(self):
        features = torch.zeros(1, 1, 26)
        features[..., -10:] = torch.tensor(
            [0.25, -0.1, 0.0, 1.0, 0.2, 0.4, 0.5, -0.2, 1.0, 0.0]
        )
        prior = predict_entity_constant_velocity_state(features)
        torch.testing.assert_close(
            prior[0, 0, -1], torch.tensor([0.75, -0.3, 0.0, 1.0, 0.5, -0.2])
        )

    def test_all_residuals_update_zero_equal_priors_and_invalid_zero(self):
        model = self.model(
            residual_visual=True, residual_spatial=True, residual_ego=True
        )
        inputs = self.inputs()
        output = model(**inputs)
        torch.testing.assert_close(
            output.ego_trajectory,
            predict_ego_kinematic_trajectory(inputs["current_ego_status"]),
        )
        selected = (
            output.entity_selection.hard_selection_weights
            @ inputs["current_entity_features"]
        )
        raw_visual = (
            output.predicted_future_visual_latents * model.visual_target_std
            + model.visual_target_mean
        )
        raw_spatial = (
            output.predicted_future_spatial_states * model.spatial_target_std
            + model.spatial_target_mean
        )
        torch.testing.assert_close(
            raw_visual[0], selected[0, :, None, :16].expand(-1, 8, -1)
        )
        torch.testing.assert_close(
            raw_spatial[0], predict_entity_constant_velocity_state(selected)[0]
        )
        self.assertEqual(
            float(output.predicted_future_visual_latents[1].detach().abs().sum()), 0
        )
        self.assertEqual(
            float(output.predicted_future_spatial_states[1].detach().abs().sum()), 0
        )

    def test_detach_keeps_values_auxiliary_remains_attached(self):
        model = self.model(residual_ego=False, residual_spatial=True)
        inputs = self.inputs()
        connected = model(**inputs)
        detached = model(**inputs, detach_future_for_planning=True)
        torch.testing.assert_close(
            connected.ego_trajectory, detached.ego_trajectory, rtol=0, atol=0
        )
        detached.ego_trajectory.square().mean().backward()
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.future_predictor.parameters()
            )
        )
        model.zero_grad(set_to_none=True)
        detached = model(**inputs, detach_future_for_planning=True)
        (
            detached.predicted_future_visual_latents.square().mean()
            + detached.predicted_future_spatial_states.square().mean()
        ).backward()
        self.assertGreater(float(model.future_predictor[-1].weight.grad.norm()), 0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.ego_planner.parameters())
        )

    def test_zero_final_head_early_zero_gradient_then_upstream_receives_gradient(self):
        model = self.model(residual_spatial=True)
        inputs = self.inputs()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        output = model(**inputs)
        loss = (
            output.predicted_future_visual_latents.square().mean()
            + output.predicted_future_spatial_states.square().mean()
        )
        loss.backward()
        self.assertEqual(float(model.future_predictor[0].weight.grad.norm()), 0)
        self.assertGreater(float(model.future_predictor[-1].weight.grad.norm()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        output = model(**inputs)
        (
            output.predicted_future_visual_latents.square().mean()
            + output.predicted_future_spatial_states.square().mean()
        ).backward()
        self.assertGreater(float(model.future_predictor[0].weight.grad.norm()), 0)

    def test_ridge_handles_constant_command_channels_and_train_only_scaling(self):
        generator = np.random.default_rng(29)
        current = generator.normal(size=(30, 8))
        current[:, 0] = 1
        target = np.repeat(current[:, None, 4:6], 8, axis=1)
        fitted = fit_ridge_from_training(current, target, 0.0001)
        self.assertLess(
            np.max(np.abs(apply_fitted_ridge(fitted, current) - target)), 0.0001
        )
        self.assertEqual(fitted["mean"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
