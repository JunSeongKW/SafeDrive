"""Contracts for encoder learning, masking identity, and teacher-only targets."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from planning_aware_future_prediction.models.intent_conditioned_encoder import (
    IntentConditionedEncoderTail, TrainingOnlyFutureHead, compute_selected_latent_loss,
    pool_spatial_regions, region_indices_to_patch_indices, select_training_regions,
    visible_patch_indices,
)


class SyntheticEncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_gain = nn.Parameter(torch.ones(1024))

    def forward(self, encoder_features, **kwargs):
        return encoder_features * self.feature_gain


class IntentConditionedEncoderTests(unittest.TestCase):
    def test_regions_keep_native_spatial_positions(self):
        selected_regions = torch.tensor([[0, 17, 127]])
        self.assertEqual(region_indices_to_patch_indices(selected_regions).tolist()[0],
                         [0, 1, 32, 33, 66, 67, 98, 99, 478, 479, 510, 511])
        visible = visible_patch_indices(selected_regions)
        self.assertEqual(visible.shape, (1, 500))
        self.assertFalse(torch.isin(visible, region_indices_to_patch_indices(selected_regions)).any())
        patches = torch.arange(512).float()[:, None]
        self.assertEqual(pool_spatial_regions(patches)[17].item(), 82.5)

    def test_guided_sampling_keeps_exploration_and_unique_budget(self):
        region_scores = torch.arange(128).float()[None]
        selected = select_training_regions(region_scores, "planning", 32, torch.Generator().manual_seed(4))
        self.assertEqual(selected.unique().numel(), 32)
        self.assertTrue(torch.isin(torch.arange(112, 128), selected).all())
        self.assertEqual(int((selected < 112).sum()), 16)

    def test_future_loss_updates_encoder_and_never_teacher(self):
        official = SimpleNamespace(blocks=nn.ModuleList([SyntheticEncoderBlock(), SyntheticEncoderBlock()]), norm=nn.LayerNorm(1024))
        encoder = IntentConditionedEncoderTail(official)
        head = TrainingOnlyFutureHead()
        prefix = torch.randn(2, 512, 1024)
        ego_status = torch.randn(2, 8)
        selected_regions = torch.tensor([[0, 2], [3, 5]])
        teacher = torch.randn(2, 4, 128, 1024, requires_grad=True)
        valid = torch.ones(2, 4, 128, dtype=torch.bool)
        loss = compute_selected_latent_loss(head(encoder(prefix, ego_status), selected_regions), teacher, valid, selected_regions)
        loss.backward()
        self.assertGreater(encoder.blocks[0].feature_gain.grad.abs().sum().item(), 0)
        self.assertGreater(encoder.intent_conditioning[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(official.blocks[0].feature_gain.grad)

    def test_no_intent_encoder_is_invariant_and_conditioned_encoder_can_change(self):
        official = SimpleNamespace(blocks=nn.ModuleList([SyntheticEncoderBlock(), SyntheticEncoderBlock()]), norm=nn.LayerNorm(1024))
        prefix = torch.randn(2, 8, 1024)
        original_status = torch.zeros(2, 8)
        changed_status = torch.ones(2, 8)
        plain = IntentConditionedEncoderTail(official, use_ego_intent=False)
        torch.testing.assert_close(plain(prefix, original_status), plain(prefix, changed_status), atol=0, rtol=0)
        conditioned = IntentConditionedEncoderTail(official)
        torch.testing.assert_close(conditioned(prefix, original_status), plain(prefix, original_status), atol=0, rtol=0)
        with torch.no_grad():
            conditioned.intent_conditioning[-1].weight.normal_(std=.01)
        self.assertGreater((conditioned(prefix, original_status) - conditioned(prefix, changed_status)).abs().max().item(), 0)

    def test_invalid_future_nan_values_do_not_supervise(self):
        predictions = torch.randn(1, 2, 4, 1024, requires_grad=True)
        targets = torch.full((1, 4, 128, 1024), float("nan"))
        valid = torch.zeros(1, 4, 128, dtype=torch.bool)
        loss = compute_selected_latent_loss(predictions, targets, valid, torch.tensor([[0, 1]]))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertEqual(predictions.grad.abs().sum().item(), 0.)


if __name__ == "__main__":
    unittest.main()
