"""LoRA preservation/gradient contracts and memory-safe execution selection."""
import os
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_lora_finetuning import LowRankLinearAdaptation
from queue_lpwm_adaptation_methods import (
    process_identity, identity_alive, select_execution_profile, projected_upsize_peak_gib)


class LoRAAdaptationTests(unittest.TestCase):
    def test_zero_initialization_and_update_preserve_original_map(self):
        torch.manual_seed(47)
        original = nn.Linear(12, 10)
        features = torch.randn(3, 5, 12, requires_grad=True)
        before_weight, before_bias = original.weight.detach().clone(), original.bias.detach().clone()
        adapted = LowRankLinearAdaptation(original, rank=4, alpha=8)
        torch.testing.assert_close(adapted(features), original(features), rtol=0, atol=0)
        optimizer = torch.optim.AdamW([parameter for parameter in adapted.parameters() if parameter.requires_grad], lr=.01)
        adapted(features).square().mean().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertIsNone(original.weight.grad)
        self.assertIsNone(original.bias.grad)
        self.assertGreater(float(adapted.lora_output_projection.grad.abs().sum()), 0)
        optimizer.step()
        torch.testing.assert_close(original.weight, before_weight, rtol=0, atol=0)
        torch.testing.assert_close(original.bias, before_bias, rtol=0, atol=0)
        self.assertGreater(float((adapted(features) - original(features)).detach().abs().sum()), 0)
        restored = LowRankLinearAdaptation(nn.Linear(12, 10), rank=4, alpha=8)
        restored.load_state_dict(adapted.state_dict(), strict=True)
        torch.testing.assert_close(restored(features), adapted(features), rtol=0, atol=0)

    def test_batch_selection_ignores_failed_profiles(self):
        records = [{"passed": False, "last_three_mean_update_seconds": .1},
                   {"passed": True, "last_three_mean_update_seconds": 4.},
                   {"passed": True, "last_three_mean_update_seconds": 3.}]
        self.assertIs(select_execution_profile(records), records[2])
        with self.assertRaises(RuntimeError):
            select_execution_profile(records[:1])

    def test_process_identity_rejects_pid_reuse(self):
        identity = process_identity(os.getpid())
        self.assertEqual(identity["parent_pid"], os.getppid())
        self.assertTrue(identity_alive(identity))
        self.assertFalse(identity_alive({**identity, "start_ticks": "different_process_start"}))

    def test_batch_upsize_is_rejected_before_likely_allocation_failure(self):
        self.assertGreater(projected_upsize_peak_gib(12.07, 4, 8, 1.), 21.5)
        self.assertLess(projected_upsize_peak_gib(9., 4, 8, 1.), 21.5)


if __name__ == "__main__":
    unittest.main()
