"""Numerical/selection contracts for real-checkpoint visualization."""

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    DriveJEPAAdaptiveFuture,
    EgoQueryPatchSelector,
)
from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
    PlanningConditionedPatchSelector,
)

module_spec = importlib.util.spec_from_file_location(
    "module_visualization",
    Path(__file__).resolve().parents[1]
    / "scripts/visualize_drive_jepa_learned_modules.py",
)
visualization = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(visualization)


class LearnedModuleVisualizationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(1)

    def test_patch_bounds_match_flattened_grid(self):
        self.assertEqual(visualization.patch_pixel_bounds(0), (0, 0, 16, 16))
        self.assertEqual(visualization.patch_pixel_bounds(511), (496, 240, 16, 16))
        with self.assertRaises(ValueError):
            visualization.patch_pixel_bounds(512)

    def test_scores_match_production_unique_selection(self):
        current = torch.randn(2, 512, 16)
        status = torch.randn(2, 8)
        coordinates = visualization.patch_coordinates()
        for selector_class in [EgoQueryPatchSelector, PlanningConditionedPatchSelector]:
            selector = selector_class(16, 8, 4).eval()
            scores = visualization.selector_scores(
                selector, current, status, coordinates
            )
            actual = selector(current, status, coordinates).selected_patch_indices
            self.assertTrue(torch.equal(scores.argsort(descending=True)[:, :4], actual))

    def test_direct_predictor_matches_production_wrapper(self):
        baseline = nn.Module()
        baseline.image_fc = nn.Linear(16, 16)
        current, status = torch.randn(1, 512, 16), torch.randn(1, 8)
        for contextual in [True, False]:
            model = DriveJEPAAdaptiveFuture(
                baseline,
                latent_dim=16,
                hidden_dim=8,
                contextual_predictor=contextual,
                ego_query_selector=contextual,
            ).eval()
            selection = model.patch_selector(current, status, model.patch_coordinates)
            expected = model._predict_selected(
                current, status, selection.hard_selection_weights
            )[0]
            actual = visualization.predict_selected(
                model.future_predictor,
                current,
                status,
                model.patch_coordinates,
                selection.selected_patch_indices[0],
            )
            torch.testing.assert_close(actual, expected)

    def test_scoped_loading_is_strict(self):
        linear = nn.Linear(2, 3)
        with self.assertRaises(RuntimeError):
            visualization.strict_restore_submodule(
                linear, {"module.weight": torch.ones(3, 2)}, "module."
            )

    def test_gallery_never_filters_future_validity_or_results(self):
        records = [
            {
                "command_raw_index": command,
                "current_frame_token": f"{command}-{index}",
                "recording_group": f"recording-{command}-{index}",
                "future_tubelet_valid": [False] * 4,
            }
            for command in [0, 1, 2]
            for index in range(3)
        ]
        selected = visualization.choose_gallery_records(records, "fixed-salt", 2)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len({row["recording_group"] for row in selected}), 6)
        self.assertEqual(
            selected,
            visualization.choose_gallery_records(records[::-1], "fixed-salt", 2),
        )

    def test_missing_targets_are_not_zero_error(self):
        prediction = torch.zeros(4, 4, 16)
        target = torch.ones_like(prediction)
        valid = torch.ones(4, 4, dtype=torch.bool)
        valid[:, 3] = False
        mse, _ = visualization.target_metrics(prediction, target, valid)
        exported = visualization.finite_nested(mse)
        self.assertEqual(exported[0], [1.0, 1.0, 1.0, None])


if __name__ == "__main__":
    unittest.main()
