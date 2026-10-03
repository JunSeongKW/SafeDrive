import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import compute_candidate_losses, compute_refinement_losses, compose_candidate_pdms
from queue_lpwm_validated_training import join_queue
import validate_lpwm_stage_transition as transition


class CandidateSupervisionTests(unittest.TestCase):
    def test_missing_teacher_does_not_nan_or_remove_imitation_gradient(self):
        predictions = {"imitation_logits": torch.randn(2, 5, requires_grad=True), "metric_logits": torch.randn(2, 5, 6, requires_grad=True)}
        labels = torch.rand(2, 5, 7)
        labels[0] = torch.nan
        losses = compute_candidate_losses(predictions, torch.randn(2, 8, 3), torch.randn(5, 8, 3), labels)
        losses["objective"].backward()
        self.assertTrue(torch.isfinite(losses["objective"]))
        self.assertGreater(float(predictions["imitation_logits"].grad[0].abs().sum()), 0)
        self.assertEqual(float(predictions["metric_logits"].grad[0].abs().sum()), 0)
        self.assertGreater(float(predictions["metric_logits"].grad[1].abs().sum()), 0)

    def test_imitation_control_does_not_train_metric_heads(self):
        predictions = {"imitation_logits": torch.randn(2, 5, requires_grad=True), "metric_logits": torch.randn(2, 5, 6, requires_grad=True)}
        loss = compute_candidate_losses(predictions, torch.randn(2, 8, 3), torch.randn(5, 8, 3), torch.rand(2, 5, 7), metric_weight=0.)
        loss["objective"].backward()
        self.assertEqual(float(predictions["metric_logits"].grad.abs().sum()), 0.)
        self.assertGreater(float(predictions["imitation_logits"].grad.abs().sum()), 0.)

    def test_navsim_v1_collision_and_drivability_zero_the_score(self):
        metrics = torch.ones(3, 6)
        metrics[0, 0] = 0
        metrics[1, 1] = 0
        metrics[2, 5] = 0  # Direction has weight zero in this official v1 setup.
        torch.testing.assert_close(compose_candidate_pdms(metrics), torch.tensor([0., 0., 1.]))

    def test_refinement_has_real_trajectory_and_temporal_supervision_gradients(self):
        predictions = {"refined_candidates": torch.randn(2, 3, 8, 3, requires_grad=True),
            "refined_metric_logits": torch.randn(2, 3, 6, requires_grad=True),
            "temporal_safety_logits": torch.randn(2, 3, 8, 2, requires_grad=True)}
        metrics = torch.rand(2, 3, 7)
        temporal = torch.rand(2, 3, 8, 2)
        metrics[0] = torch.nan
        temporal[0] = torch.nan
        losses = compute_refinement_losses(predictions, torch.zeros(2, 8, 3), torch.zeros(2, 8), metrics, temporal)
        losses["objective"].backward()
        for name, values in predictions.items():
            self.assertTrue(torch.isfinite(values.grad).all(), name)
            self.assertGreater(float(values.grad.abs().sum()), 0, name)
        self.assertEqual(float(predictions["temporal_safety_logits"].grad[0].abs().sum()), 0)

    def test_failed_queue_cannot_be_joined_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "queue_failed.json").write_text(json.dumps({"failed_stage": "stage1_gate"}))
            with self.assertRaisesRegex(RuntimeError, "queue failed"):
                join_queue(root)

    def test_one_failed_causal_scene_blocks_stage2_even_if_original_gate_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def write(relative, value):
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(value))
            write("stage1_config.json", {"output_directory": "stage1"})
            write("stage1/adaptation_gate.json", {"adaptation_gate_passed": True, "driving_risk_breakdown": {"small_object": {"recordings": 5}}})
            records = [{"scenario": scenario, "recording_group": str(index), "metrics": {"particle_feature_std": .1, "mean_visible_particles": 30}}
                for scenario in ("straight", "turn", "projected_overlap") for index in range(5)]
            write("stage1/evaluation/posttrained/metrics.json", {"checkpoint_sha256": "checkpoint", "records": records})
            scenes = [{"observed_particle_max_difference": 0., "forecast_max_difference": 0.} for _ in range(8)]
            scenes[-1]["forecast_max_difference"] = .1
            write("planner/stage1_causality_audit.json", {"checkpoint_sha256": "checkpoint", "scenes": scenes})
            specification = {"stage1_config": "stage1_config.json", "output_directory": "planner", "validation_gates": {"minimum_causal_audit_scenes": 8, "minimum_risk_recordings": 5}}
            with mock.patch.object(transition, "PROJECT_ROOT", root):
                report = transition.check_stage1(specification)
            self.assertFalse(report["passed"])
            self.assertIn("all_causal_interventions_pass", report["failed_checks"])


if __name__ == "__main__":
    unittest.main()
