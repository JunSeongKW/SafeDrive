"""Exact graph contracts, separate from statistical toy learnability checks."""

import unittest

import torch

from selective_future_graph import SelectiveFutureGraph, SequentialSTTopK, gradient_norm


class GraphContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = SelectiveFutureGraph().double().eval()
        self.h = torch.randn(3, 4, 6, dtype=torch.double)
        self.c = torch.randn(3, 4, dtype=torch.double)
        self.u = torch.randn(3, 3, dtype=torch.double)
        self.valid = torch.ones(3, 4, dtype=torch.bool)
        self.ids = torch.tensor([[30, 10, 20, 40]]).expand(3, -1)
        self.target = torch.randn(3, 4, 2, 5, dtype=torch.double, requires_grad=True)
        self.future_valid = torch.ones(3, 4, 2, dtype=torch.bool)
        self.ego = torch.randn(3, 3, dtype=torch.double)

    def forward_graph(self, **kwargs):
        return self.model(self.h, self.c, self.u, self.valid, self.ids, **kwargs)

    def losses(self, output):
        return self.model.losses(output, self.h, self.c, self.u, self.valid,
                                 self.target, self.future_valid, self.ego)

    def test_planning_gradient_contract(self):
        loss = self.losses(self.forward_graph())["plan"]
        for module in (self.model.selector, self.model.predictor, self.model.planner):
            self.assertGreater(gradient_norm(loss, module), 0)

    def test_hard_and_selection_detach_same_values_block_selector(self):
        original = self.forward_graph()
        for mode in ("hard", "detach"):
            output = self.forward_graph(selection_mode=mode)
            self.assertTrue(torch.equal(original.trajectory, output.trajectory))
            loss = self.losses(output)["plan"]
            self.assertEqual(gradient_norm(loss, self.model.selector), 0)
            self.assertGreater(gradient_norm(loss, self.model.predictor), 0)

    def test_future_detach_preserves_forward_blocks_predictor_and_selector(self):
        original = self.forward_graph()
        output = self.forward_graph(detach_future=True)
        self.assertTrue(torch.equal(original.trajectory, output.trajectory))
        loss = self.losses(output)["plan"]
        self.assertEqual(gradient_norm(loss, self.model.selector), 0)
        self.assertEqual(gradient_norm(loss, self.model.predictor), 0)
        self.assertGreater(gradient_norm(loss, self.model.planner), 0)

    def test_auxiliary_blocks_selector_planner_target(self):
        loss = self.losses(self.forward_graph())["jepa"]
        self.assertEqual(gradient_norm(loss, self.model.selector), 0)
        self.assertEqual(gradient_norm(loss, self.model.planner), 0)
        self.assertGreater(gradient_norm(loss, self.model.predictor), 0)
        self.assertIsNone(torch.autograd.grad(loss, self.target, allow_unused=True)[0])

    def test_targets_never_change_fixed_weight_forward(self):
        original = self.forward_graph()
        first_loss = self.losses(original)["jepa"]
        self.target = self.target.detach() + 100
        second_loss = self.losses(self.forward_graph())["jepa"]
        self.assertNotEqual(first_loss.item(), second_loss.item())
        output = self.forward_graph()
        for a, b in ((original.selection.indices, output.selection.indices),
                     (original.future, output.future), (original.trajectory, output.trajectory)):
            self.assertTrue(torch.equal(a, b))

    def test_removal_and_cross_sample_exchange_change_fixture_output(self):
        output = self.forward_graph()
        removed = self.model.plan(self.c, self.u, torch.zeros_like(output.future), output.selection.valid)
        exchanged = self.model.plan(self.c, self.u, output.future.flip(0), output.selection.valid)
        self.assertGreater((output.trajectory - removed).abs().max().item(), 1e-8)
        self.assertGreater((output.trajectory - exchanged).abs().max().item(), 1e-8)

    def test_permutation_invariance_including_gradient(self):
        output = self.forward_graph()
        loss = self.losses(output)["plan"]
        grad = torch.autograd.grad(loss, tuple(self.model.selector.parameters()))
        permutation = torch.tensor([2, 0, 3, 1])
        other = self.model(self.h[:, permutation], self.c, self.u,
                           self.valid[:, permutation], self.ids[:, permutation])
        selected = self.ids.gather(1, output.selection.indices)
        perm_ids = self.ids[:, permutation].gather(1, other.selection.indices)
        self.assertTrue(torch.equal(selected, perm_ids))
        torch.testing.assert_close(output.trajectory, other.trajectory, rtol=1e-12, atol=1e-12)
        other_loss = (other.trajectory - self.ego).square().mean()
        other_grad = torch.autograd.grad(other_loss, tuple(self.model.selector.parameters()))
        for a, b in zip(grad, other_grad):
            torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)

    def test_invalid_future_nan_zero_loss(self):
        self.target = torch.full_like(self.target, torch.nan)
        self.future_valid.zero_()
        loss = self.losses(self.forward_graph())["jepa"]
        self.assertEqual(loss.item(), 0)
        self.assertEqual(gradient_norm(loss, self.model.predictor), 0)
        self.assertEqual(gradient_norm(loss, self.model.selector), 0)

    def test_empty_and_all_invalid_entities(self):
        self.valid.zero_()
        self.h.fill_(torch.nan)
        output = self.forward_graph()
        self.assertFalse(output.selection.valid.any())
        self.assertTrue(torch.isfinite(output.trajectory).all())
        self.assertEqual(self.losses(output)["jepa"].item(), 0)
        self.h, self.valid, self.ids = self.h[:, :0], self.valid[:, :0], self.ids[:, :0]
        self.target, self.future_valid = self.target[:, :0], self.future_valid[:, :0]
        empty = self.forward_graph()
        self.assertEqual(empty.selection.weights.shape, (3, 2, 0))
        self.assertTrue(torch.isfinite(empty.trajectory).all())
        self.assertEqual(self.losses(empty)["jepa"].item(), 0)


class SelectionContracts(unittest.TestCase):
    def test_unique_padding_ties_and_invalid_gradients(self):
        scores = torch.tensor([[1., 1., 1., 999.], [3., 1., 999., 999.]], requires_grad=True)
        valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
        ids = torch.tensor([[30, 10, 20, 0], [20, 10, 0, 0]])
        selected = SequentialSTTopK(5)(scores, valid, ids)
        self.assertEqual(selected.indices.tolist(), [[1, 2, 0, -1, -1], [0, 1, -1, -1, -1]])
        self.assertTrue(torch.equal(selected.weights, selected.hard_weights))
        self.assertEqual(selected.valid.sum(1).tolist(), [3, 2])
        # Different slot costs also exercise the sequential surrogate gradient.
        (selected.weights * torch.arange(20).reshape(1, 5, 4)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertTrue(torch.equal(scores.grad[~valid], torch.zeros_like(scores.grad[~valid])))
        self.assertEqual(scores.grad.abs().sum().item(), 0)  # K covers all valid entities

    def test_partial_budget_gradients_only_when_there_is_a_choice(self):
        scores = torch.tensor([[1., 2., 3.], [1., 2., 999.]], requires_grad=True)
        valid = torch.tensor([[True, True, True], [True, True, False]])
        ids = torch.tensor([[30, 10, 20], [30, 10, 0]])
        selected = SequentialSTTopK(2)(scores, valid, ids)
        (selected.weights * torch.tensor([[[1., 2., 4.], [4., 1., 2.]]])).sum().backward()
        self.assertGreater(scores.grad[0].abs().sum().item(), 0)
        self.assertEqual(scores.grad[1].abs().sum().item(), 0)

    def test_tie_permutation_preserves_identity(self):
        scores = torch.zeros(1, 4)
        valid = torch.ones(1, 4, dtype=torch.bool)
        ids = torch.tensor([[80, 20, 40, 60]])
        module = SequentialSTTopK(2)
        original = module(scores, valid, ids)
        perm = torch.tensor([3, 1, 0, 2])
        other = module(scores[:, perm], valid[:, perm], ids[:, perm])
        self.assertEqual(ids.gather(1, original.indices).tolist(), [[20, 40]])
        self.assertTrue(torch.equal(ids.gather(1, original.indices), ids[:, perm].gather(1, other.indices)))

    def test_invalid_configuration_and_ambiguous_ids_rejected(self):
        with self.assertRaises(ValueError):
            SequentialSTTopK(0)
        with self.assertRaises(ValueError):
            SequentialSTTopK(1, temperature=0)
        module = SequentialSTTopK(1)
        with self.assertRaises(ValueError):
            module(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            module(torch.tensor([[float("nan")]]), torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
