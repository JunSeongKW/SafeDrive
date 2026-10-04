import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_object_supervision import (
    ParticleObjectStateHead, compute_object_auxiliary_losses, detached_roi_particle_weights)
from lpwm_stage2_admission import require_stage2_admission


class Stage2ObjectSupervisionTests(unittest.TestCase):
    def make_inputs(self):
        observed = torch.randn(2, 4, 64, 14, requires_grad=True)
        future = torch.randn(2, 8, 64, 14, requires_grad=True)
        attributes = torch.cat((observed[:, -1:], future), 1)
        head = ParticleObjectStateHead()
        predictions = {"observed_particle_attributes": observed, "predicted_particle_attributes": future,
            **head(attributes, torch.zeros(2, 8))}
        boxes = torch.tensor([30., 40., 60., 70.]).expand(2, 9, 3, 4).clone().requires_grad_()
        states = torch.randn(2, 9, 3, 4, requires_grad=True)
        valid = torch.ones(2, 9, 3, dtype=torch.bool)
        targets = {"projected_boxes": boxes, "current_ego_states": states, "state_valid": valid,
            "categories": torch.zeros(2, 3, dtype=torch.long)}
        configuration = {"state_scales": [40., 20., 10., 10.], "current_state_weight": .2,
            "future_state_weight": .2, "category_weight": .02, "gaussian_minimum_sigma_pixels": 2.}
        return predictions, targets, configuration

    def test_auxiliary_updates_features_but_not_gt_or_association(self):
        predictions, targets, configuration = self.make_inputs()
        losses = compute_object_auxiliary_losses(predictions, targets, configuration)
        losses["objective"].backward()
        self.assertIsNone(targets["projected_boxes"].grad)
        self.assertIsNone(targets["current_ego_states"].grad)
        self.assertGreater(float(predictions["observed_particle_attributes"].grad.abs().sum()), 0)
        self.assertGreater(float(predictions["predicted_particle_attributes"].grad.abs().sum()), 0)

    def test_unannotated_objects_have_no_negative_supervision(self):
        predictions, targets, configuration = self.make_inputs()
        targets["state_valid"].zero_()
        losses = compute_object_auxiliary_losses(predictions, targets, configuration)
        self.assertEqual(float(losses["objective"].detach()), 0.)
        losses["objective"].backward()
        self.assertEqual(float(predictions["predicted_particle_attributes"].grad.abs().sum()), 0.)

    def test_future_loss_does_not_use_observed_target_values(self):
        predictions, targets, configuration = self.make_inputs()
        before = compute_object_auxiliary_losses(predictions, targets, configuration)["future_state_loss"]
        altered = {**targets, "current_ego_states": targets["current_ego_states"].detach().clone()}
        altered["current_ego_states"][:, 0] += 1000
        after = compute_object_auxiliary_losses(predictions, altered, configuration)["future_state_loss"]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_association_keeps_all_particles_and_detaches_geometry(self):
        attributes = torch.randn(2, 9, 64, 14, requires_grad=True)
        boxes = torch.tensor([10., 10., 80., 90.]).expand(2, 9, 3, 4)
        weights, _ = detached_roi_particle_weights(attributes, boxes)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 9, 3))
        changed = attributes.detach().clone()
        changed[..., 4] = 0  # Presence is not a selection or negative label.
        alternate, _ = detached_roi_particle_weights(changed, boxes)
        torch.testing.assert_close(weights, alternate, rtol=0, atol=0)

    def test_original_failed_gate_still_blocks_without_amendment(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"preserved-checkpoint")
            (root / "adaptation_gate.json").write_text(json.dumps({"adaptation_gate_passed": False,
                "training": {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}}))
            with self.assertRaisesRegex(AssertionError, "Stage1 adaptation gate failed"):
                require_stage2_admission({}, root, root, checkpoint)


if __name__ == "__main__":
    unittest.main()
