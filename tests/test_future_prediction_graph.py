"""Exact graph contracts, separate from statistical toy learnability checks."""

import unittest

import torch

from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    SelectiveEntityFuturePredictionGraph,
    SequentialStraightThroughEntitySelection,
    compute_parameter_gradient_norm,
)


class FuturePredictionGraphContractTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = SelectiveEntityFuturePredictionGraph().double().eval()
        self.entity_features = torch.randn(3, 4, 6, dtype=torch.double)
        self.scene_context = torch.randn(3, 4, dtype=torch.double)
        self.ego_intent = torch.randn(3, 3, dtype=torch.double)
        self.entity_valid_mask = torch.ones(3, 4, dtype=torch.bool)
        self.stable_entity_ids = torch.tensor([[30, 10, 20, 40]]).expand(3, -1)
        self.future_latent_targets = torch.randn(
            3, 4, 2, 5, dtype=torch.double, requires_grad=True
        )
        self.future_target_valid_mask = torch.ones(3, 4, 2, dtype=torch.bool)
        self.ego_plan_target = torch.randn(3, 3, dtype=torch.double)

    def run_fixture_forward(self, **kwargs):
        return self.model(
            self.entity_features,
            self.scene_context,
            self.ego_intent,
            self.entity_valid_mask,
            self.stable_entity_ids,
            **kwargs
        )

    def compute_fixture_losses(self, prediction_output):
        return self.model.compute_training_losses(
            prediction_output,
            self.entity_features,
            self.scene_context,
            self.ego_intent,
            self.entity_valid_mask,
            self.future_latent_targets,
            self.future_target_valid_mask,
            self.ego_plan_target,
        )

    def test_planning_gradient_contract(self):
        training_loss = self.compute_fixture_losses(self.run_fixture_forward())[
            "planning_loss"
        ]
        for selection_operator in (
            self.model.entity_scorer,
            self.model.future_predictor,
            self.model.ego_planner,
        ):
            self.assertGreater(
                compute_parameter_gradient_norm(training_loss, selection_operator), 0
            )

    def test_hard_and_selection_detach_same_values_block_selector(self):
        reference_output = self.run_fixture_forward()
        for selection_gradient_mode in ("hard_indices", "detached_selection"):
            prediction_output = self.run_fixture_forward(
                selection_gradient_mode=selection_gradient_mode
            )
            self.assertTrue(
                torch.equal(reference_output.ego_plan, prediction_output.ego_plan)
            )
            training_loss = self.compute_fixture_losses(prediction_output)[
                "planning_loss"
            ]
            self.assertEqual(
                compute_parameter_gradient_norm(
                    training_loss, self.model.entity_scorer
                ),
                0,
            )
            self.assertGreater(
                compute_parameter_gradient_norm(
                    training_loss, self.model.future_predictor
                ),
                0,
            )

    def test_future_detach_preserves_forward_blocks_predictor_and_selector(self):
        reference_output = self.run_fixture_forward()
        prediction_output = self.run_fixture_forward(detach_future_latents=True)
        self.assertTrue(
            torch.equal(reference_output.ego_plan, prediction_output.ego_plan)
        )
        training_loss = self.compute_fixture_losses(prediction_output)["planning_loss"]
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.entity_scorer), 0
        )
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.future_predictor),
            0,
        )
        self.assertGreater(
            compute_parameter_gradient_norm(training_loss, self.model.ego_planner), 0
        )

    def test_auxiliary_blocks_selector_planner_target(self):
        training_loss = self.compute_fixture_losses(self.run_fixture_forward())[
            "future_latent_prediction_loss"
        ]
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.entity_scorer), 0
        )
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.ego_planner), 0
        )
        self.assertGreater(
            compute_parameter_gradient_norm(training_loss, self.model.future_predictor),
            0,
        )
        self.assertIsNone(
            torch.autograd.grad(
                training_loss, self.future_latent_targets, allow_unused=True
            )[0]
        )

    def test_targets_never_change_fixed_weight_forward(self):
        reference_output = self.run_fixture_forward()
        original_auxiliary_loss = self.compute_fixture_losses(reference_output)[
            "future_latent_prediction_loss"
        ]
        self.future_latent_targets = self.future_latent_targets.detach() + 100
        modified_auxiliary_loss = self.compute_fixture_losses(
            self.run_fixture_forward()
        )["future_latent_prediction_loss"]
        self.assertNotEqual(
            original_auxiliary_loss.item(), modified_auxiliary_loss.item()
        )
        prediction_output = self.run_fixture_forward()
        for expected_tensor, actual_tensor in (
            (
                reference_output.entity_selection.selected_entity_indices,
                prediction_output.entity_selection.selected_entity_indices,
            ),
            (
                reference_output.predicted_future_latents,
                prediction_output.predicted_future_latents,
            ),
            (reference_output.ego_plan, prediction_output.ego_plan),
        ):
            self.assertTrue(torch.equal(expected_tensor, actual_tensor))

    def test_removal_and_cross_sample_exchange_change_fixture_output(self):
        prediction_output = self.run_fixture_forward()
        plan_without_future_latents = self.model.decode_ego_plan(
            self.scene_context,
            self.ego_intent,
            torch.zeros_like(prediction_output.predicted_future_latents),
            prediction_output.entity_selection.selected_entity_valid_mask,
        )
        plan_with_cross_sample_futures = self.model.decode_ego_plan(
            self.scene_context,
            self.ego_intent,
            prediction_output.predicted_future_latents.flip(0),
            prediction_output.entity_selection.selected_entity_valid_mask,
        )
        self.assertGreater(
            (prediction_output.ego_plan - plan_without_future_latents)
            .abs()
            .max()
            .item(),
            1e-8,
        )
        self.assertGreater(
            (prediction_output.ego_plan - plan_with_cross_sample_futures)
            .abs()
            .max()
            .item(),
            1e-8,
        )

    def test_permutation_invariance_including_gradient(self):
        prediction_output = self.run_fixture_forward()
        training_loss = self.compute_fixture_losses(prediction_output)["planning_loss"]
        selector_gradients = torch.autograd.grad(
            training_loss, tuple(self.model.entity_scorer.parameters())
        )
        entity_permutation = torch.tensor([2, 0, 3, 1])
        permuted_output = self.model(
            self.entity_features[:, entity_permutation],
            self.scene_context,
            self.ego_intent,
            self.entity_valid_mask[:, entity_permutation],
            self.stable_entity_ids[:, entity_permutation],
        )
        selection_result = self.stable_entity_ids.gather(
            1, prediction_output.entity_selection.selected_entity_indices
        )
        permuted_selected_entity_ids = self.stable_entity_ids[
            :, entity_permutation
        ].gather(1, permuted_output.entity_selection.selected_entity_indices)
        self.assertTrue(torch.equal(selection_result, permuted_selected_entity_ids))
        torch.testing.assert_close(
            prediction_output.ego_plan, permuted_output.ego_plan, rtol=1e-12, atol=1e-12
        )
        other_loss = (permuted_output.ego_plan - self.ego_plan_target).square().mean()
        permuted_selector_gradients = torch.autograd.grad(
            other_loss, tuple(self.model.entity_scorer.parameters())
        )
        for expected_tensor, actual_tensor in zip(
            selector_gradients, permuted_selector_gradients
        ):
            torch.testing.assert_close(
                expected_tensor, actual_tensor, rtol=1e-12, atol=1e-12
            )

    def test_invalid_future_nan_zero_loss(self):
        self.future_latent_targets = torch.full_like(
            self.future_latent_targets, torch.nan
        )
        self.future_target_valid_mask.zero_()
        training_loss = self.compute_fixture_losses(self.run_fixture_forward())[
            "future_latent_prediction_loss"
        ]
        self.assertEqual(training_loss.item(), 0)
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.future_predictor),
            0,
        )
        self.assertEqual(
            compute_parameter_gradient_norm(training_loss, self.model.entity_scorer), 0
        )

    def test_empty_and_all_invalid_entities(self):
        self.entity_valid_mask.zero_()
        self.entity_features.fill_(torch.nan)
        prediction_output = self.run_fixture_forward()
        self.assertFalse(
            prediction_output.entity_selection.selected_entity_valid_mask.any()
        )
        self.assertTrue(torch.isfinite(prediction_output.ego_plan).all())
        self.assertEqual(
            self.compute_fixture_losses(prediction_output)[
                "future_latent_prediction_loss"
            ].item(),
            0,
        )
        self.entity_features, self.entity_valid_mask, self.stable_entity_ids = (
            self.entity_features[:, :0],
            self.entity_valid_mask[:, :0],
            self.stable_entity_ids[:, :0],
        )
        self.future_latent_targets, self.future_target_valid_mask = (
            self.future_latent_targets[:, :0],
            self.future_target_valid_mask[:, :0],
        )
        empty_entity_output = self.run_fixture_forward()
        self.assertEqual(
            empty_entity_output.entity_selection.selection_weights.shape, (3, 2, 0)
        )
        self.assertTrue(torch.isfinite(empty_entity_output.ego_plan).all())
        self.assertEqual(
            self.compute_fixture_losses(empty_entity_output)[
                "future_latent_prediction_loss"
            ].item(),
            0,
        )


class EntitySelectionContractTests(unittest.TestCase):
    def test_unique_padding_ties_and_invalid_gradients(self):
        entity_scores = torch.tensor(
            [[1.0, 1.0, 1.0, 999.0], [3.0, 1.0, 999.0, 999.0]], requires_grad=True
        )
        entity_valid_mask = torch.tensor(
            [[True, True, True, False], [True, True, False, False]]
        )
        stable_entity_ids = torch.tensor([[30, 10, 20, 0], [20, 10, 0, 0]])
        selection_result = SequentialStraightThroughEntitySelection(5)(
            entity_scores, entity_valid_mask, stable_entity_ids
        )
        self.assertEqual(
            selection_result.selected_entity_indices.tolist(),
            [[1, 2, 0, -1, -1], [0, 1, -1, -1, -1]],
        )
        self.assertTrue(
            torch.equal(
                selection_result.selection_weights,
                selection_result.hard_selection_weights,
            )
        )
        self.assertEqual(
            selection_result.selected_entity_valid_mask.sum(1).tolist(), [3, 2]
        )
        # Different slot costs also exercise the sequential surrogate gradient.
        (
            selection_result.selection_weights * torch.arange(20).reshape(1, 5, 4)
        ).sum().backward()
        self.assertTrue(torch.isfinite(entity_scores.grad).all())
        self.assertTrue(
            torch.equal(
                entity_scores.grad[~entity_valid_mask],
                torch.zeros_like(entity_scores.grad[~entity_valid_mask]),
            )
        )
        self.assertEqual(
            entity_scores.grad.abs().sum().item(), 0
        )  # K covers all valid entities

    def test_partial_budget_gradients_only_when_there_is_a_choice(self):
        entity_scores = torch.tensor(
            [[1.0, 2.0, 3.0], [1.0, 2.0, 999.0]], requires_grad=True
        )
        entity_valid_mask = torch.tensor([[True, True, True], [True, True, False]])
        stable_entity_ids = torch.tensor([[30, 10, 20], [30, 10, 0]])
        selection_result = SequentialStraightThroughEntitySelection(2)(
            entity_scores, entity_valid_mask, stable_entity_ids
        )
        (
            selection_result.selection_weights
            * torch.tensor([[[1.0, 2.0, 4.0], [4.0, 1.0, 2.0]]])
        ).sum().backward()
        self.assertGreater(entity_scores.grad[0].abs().sum().item(), 0)
        self.assertEqual(entity_scores.grad[1].abs().sum().item(), 0)

    def test_tie_permutation_preserves_identity(self):
        entity_scores = torch.zeros(1, 4)
        entity_valid_mask = torch.ones(1, 4, dtype=torch.bool)
        stable_entity_ids = torch.tensor([[80, 20, 40, 60]])
        selection_operator = SequentialStraightThroughEntitySelection(2)
        reference_output = selection_operator(
            entity_scores, entity_valid_mask, stable_entity_ids
        )
        entity_permutation = torch.tensor([3, 1, 0, 2])
        permuted_output = selection_operator(
            entity_scores[:, entity_permutation],
            entity_valid_mask[:, entity_permutation],
            stable_entity_ids[:, entity_permutation],
        )
        self.assertEqual(
            stable_entity_ids.gather(
                1, reference_output.selected_entity_indices
            ).tolist(),
            [[20, 40]],
        )
        self.assertTrue(
            torch.equal(
                stable_entity_ids.gather(1, reference_output.selected_entity_indices),
                stable_entity_ids[:, entity_permutation].gather(
                    1, permuted_output.selected_entity_indices
                ),
            )
        )

    def test_invalid_configuration_and_ambiguous_ids_rejected(self):
        with self.assertRaises(ValueError):
            SequentialStraightThroughEntitySelection(0)
        with self.assertRaises(ValueError):
            SequentialStraightThroughEntitySelection(1, temperature=0)
        selection_operator = SequentialStraightThroughEntitySelection(1)
        with self.assertRaises(ValueError):
            selection_operator(
                torch.zeros(1, 2),
                torch.ones(1, 2, dtype=torch.bool),
                torch.zeros(1, 2, dtype=torch.long),
            )
        with self.assertRaises(ValueError):
            selection_operator(
                torch.tensor([[float("nan")]]),
                torch.ones(1, 1, dtype=torch.bool),
                torch.zeros(1, 1, dtype=torch.long),
            )


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
