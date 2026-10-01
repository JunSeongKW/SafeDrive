"""Input-only rule equivalence; no selector retraining or accuracy tuning."""

import importlib.util
import json
import unittest
from pathlib import Path

import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts/validate_future_prediction_graph.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "synthetic_graph_validation", SCRIPT_PATH
)
VALIDATION_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(VALIDATION_SCRIPT)


class SyntheticReferencePolicyTests(unittest.TestCase):
    def test_original_gradient_norms_unchanged_without_retraining(self):
        report_path = (
            SCRIPT_PATH.parents[1]
            / "results/synthetic_diagnostics/readability_refactor_validation_20261001.json"
        )
        reference = json.loads(report_path.read_text())["gradient_norms"]
        self.assertEqual(
            VALIDATION_SCRIPT.measure_training_signal_gradient_norms(), reference
        )

    def test_input_exact_match_equals_relevance_reference(self):
        entity_features, _, ego_intent, entity_ids, relevance, ego_target = (
            VALIDATION_SCRIPT.generate_synthetic_entity_batch(
                4096, torch.Generator().manual_seed(20000)
            )
        )
        scores = VALIDATION_SCRIPT.compute_input_exact_match_scores(
            entity_features, ego_intent
        )
        self.assertTrue(torch.equal(scores.bool(), relevance))
        selection = VALIDATION_SCRIPT.SequentialStraightThroughEntitySelection(2)(
            scores, torch.ones_like(relevance), entity_ids
        )
        metrics = VALIDATION_SCRIPT.compute_synthetic_selection_metrics(
            entity_features, selection.hard_selection_weights, relevance, ego_target
        )
        self.assertEqual(metrics["exact_set_accuracy"], 1.0)
        self.assertEqual(metrics["relevant_recall"], 1.0)
        self.assertLess(metrics["planning_mse"], 1e-12)
