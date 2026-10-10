"""Numerical edge cases for the normalization-only intervention."""
import unittest

import torch

from planning_aware_future_prediction.models.soft_token_reweighting import ConditionalTokenImportance


class NormalizedTokenImportanceChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        torch.set_num_threads(2)
        self.module = ConditionalTokenImportance(8, 16, normalize_importance=True)
        self.features = torch.randn(2, 5, 8)
        self.valid = torch.tensor([[True, True, False, True, False], [True]*5])
        self.ego = torch.randn(2, 8)
        self.positions = torch.randn(5, 4)

    def forward_importance(self):
        return self.module(self.features, self.valid, self.ego, self.positions)

    def test_masked_population_standardization_and_gradient(self):
        importance = self.forward_importance()
        for index in range(2):
            valid_values = importance[index, self.valid[index]]
            self.assertLess(abs(float(valid_values.mean())), 1e-5)
            self.assertGreater(float(valid_values.std(unbiased=False)), .99)
        self.assertTrue((importance[~self.valid] == 0).all())
        weights = torch.randn_like(importance)
        (importance*weights).sum().backward()
        gradient = self.module.importance_mlp[-1].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()), 0)
        altered_features = self.features.clone()
        altered_features[~self.valid] += 1000
        altered = self.module(altered_features, self.valid, self.ego, self.positions)
        torch.testing.assert_close(importance, altered, atol=1e-5, rtol=1e-5)

    def test_single_valid_token_and_constant_output_have_finite_gradients(self):
        for single_valid in (False, True):
            self.module.zero_grad(set_to_none=True)
            if single_valid:
                self.valid[:] = False
                self.valid[:, 0] = True
            else:
                with torch.no_grad():
                    self.module.importance_mlp[-1].weight.zero_()
                    self.module.importance_mlp[-1].bias.zero_()
            importance = self.forward_importance()
            self.assertTrue((importance == 0).all())
            (importance*torch.randn_like(importance)).sum().backward()
            for parameter in self.module.parameters():
                self.assertTrue(parameter.grad is None or torch.isfinite(parameter.grad).all())

    def test_all_invalid_scene_is_rejected(self):
        self.valid[0] = False
        with self.assertRaises(ValueError):
            self.forward_importance()

    def test_normalization_adds_no_parameters_and_preserves_initial_weights(self):
        torch.manual_seed(0)
        original = ConditionalTokenImportance(8, 16)
        self.assertEqual(list(original.state_dict()), list(self.module.state_dict()))
        for name, tensor in original.state_dict().items():
            torch.testing.assert_close(tensor, self.module.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
