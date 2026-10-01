"""New detach and variance-axis contracts, separate from old recovery checks."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_visual_future_prediction_variance import variance_across_axis

from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
    FixedDistanceFutureSupervisionPilot,
)


class FuturePredictionDiagnosticTests(unittest.TestCase):
    def test_detach_preserves_values_but_not_planning_predictor_gradient(self):
        torch.manual_seed(29)
        model = FixedDistanceFutureSupervisionPilot(visual_feature_dim=8)
        model.configure_active_modules(True)
        current_inputs = {
            "current_image_grid": torch.randn(2, 8, 16, 32),
            "current_entity_features": torch.randn(2, 5, 18),
            "current_entity_valid_mask": torch.ones(2, 5, dtype=torch.bool),
            "stable_entity_ids": torch.arange(5)[None].expand(2, -1),
            "current_ego_status": torch.randn(2, 8),
        }
        attached = model(**current_inputs)
        detached = model(**current_inputs, detach_future_for_planning=True)
        self.assertTrue(torch.equal(attached.ego_trajectory, detached.ego_trajectory))
        detached.ego_trajectory.square().mean().backward(retain_graph=True)
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in model.future_predictor.parameters()
            )
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.ego_planner.parameters()
            )
        )
        model.zero_grad(set_to_none=True)
        (
            detached.predicted_future_visual_latents.square().mean()
            + detached.predicted_future_spatial_states.square().mean()
        ).backward()
        self.assertGreater(float(model.future_predictor[-1].weight.grad[:8].norm()), 0)
        self.assertGreater(
            float(model.future_predictor[-1].weight.grad[8:14].norm()), 0
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.ego_planner.parameters())
        )

    def test_axis_variances_do_not_conflate_time_with_samples(self):
        features = torch.tensor([0.0, 2.0])[None, None, :, None].expand(3, 2, -1, 4)
        valid = torch.ones(3, 2, 2, dtype=torch.bool)
        self.assertEqual(
            variance_across_axis(features, valid, 0)[
                "mean_channel_population_variance"
            ],
            0,
        )
        self.assertEqual(
            variance_across_axis(features, valid, 1)[
                "mean_channel_population_variance"
            ],
            0,
        )
        self.assertEqual(
            variance_across_axis(features, valid, 2)[
                "mean_channel_population_variance"
            ],
            1,
        )

    def test_variance_excludes_invalid_observations_and_singletons(self):
        features = torch.tensor([0.0, 2.0, 1000.0])[:, None, None, None]
        valid = torch.tensor([True, True, False])[:, None, None]
        self.assertEqual(
            variance_across_axis(features, valid, 0)[
                "mean_channel_population_variance"
            ],
            1,
        )
        self.assertIsNone(
            variance_across_axis(features, valid, 1)["mean_channel_population_variance"]
        )


if __name__ == "__main__":
    unittest.main()
