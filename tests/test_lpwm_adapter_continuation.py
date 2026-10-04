"""Protect frozen adapter weights, actual optimizer resume and queue dependencies."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import ParticleBlockWithAdapter
from lpwm_full_continuation import restore_preserved_full_checkpoint
from evaluate_lpwm_full_planning import digest
from queue_lpwm_followup_methods import predecessor_ready


class AdapterContinuationTests(unittest.TestCase):
    def test_adapter_changes_features_without_changing_native_weights(self):
        torch.manual_seed(47)
        native = nn.Linear(12, 12)
        before = copy.deepcopy(native.state_dict())
        adapted = ParticleBlockWithAdapter(native, 12, 4)
        features = torch.randn(2, 3, 12, requires_grad=True)
        torch.testing.assert_close(adapted(features), native(features), atol=0, rtol=0)
        optimizer = torch.optim.AdamW([parameter for parameter in adapted.parameters() if parameter.requires_grad], lr=.01)
        adapted(features).square().mean().backward()
        self.assertGreater(float(features.grad.abs().sum()), 0)
        self.assertIsNone(native.weight.grad)
        optimizer.step()
        for name, value in native.state_dict().items():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        self.assertGreater(float((adapted(features) - native(features)).detach().abs().sum()), 0)

    def test_resume_optimizer_reproduces_next_update(self):
        torch.manual_seed(47)
        model = nn.Linear(5, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        features = torch.randn(4, 5)
        model(features).square().mean().backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source_config.json"
            source.write_text('{}\n')
            checkpoint = root / "preserved_checkpoint.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_updates": 1,
                "elapsed_seconds": 3., "world_size": 2, "configuration_sha256": digest(source),
                "stage1_checkpoint_sha256": "stage1_reference"}, checkpoint)
            preserved_hash = digest(checkpoint)
            restored = nn.Linear(5, 3)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.9)
            report = restore_preserved_full_checkpoint(restored, restored_optimizer, {
                "world_size": 2, "resume_from_prior_full": {"checkpoint": checkpoint.name,
                    "source_configuration": source.name, "checkpoint_sha256": preserved_hash, "completed_updates": 1}}, root)
            self.assertEqual(report["next_update"], 2)
            for active_model, active_optimizer in ((model, optimizer), (restored, restored_optimizer)):
                active_optimizer.zero_grad(set_to_none=True)
                active_model(features).square().mean().backward()
                active_optimizer.step()
            for first, second in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(first, second, atol=0, rtol=0)
            self.assertEqual(digest(checkpoint), preserved_hash)

    def test_followup_requires_completed_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(predecessor_ready(root))
            (root / "queue_completion.json").write_text(json.dumps({"complete": True}))
            with self.assertRaises(FileNotFoundError):
                predecessor_ready(root)
            (root / "method_comparison_summary.json").write_text(json.dumps({"complete": True,
                "reports": {"partial": {"engineering_only": False, "checks": {"complete_registered_epoch": True}}}}))
            self.assertTrue(predecessor_ready(root))
            (root / "queue_failed.json").write_text('{}')
            with self.assertRaises(RuntimeError):
                predecessor_ready(root)


if __name__ == "__main__":
    unittest.main()
