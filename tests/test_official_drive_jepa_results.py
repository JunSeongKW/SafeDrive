"""CPU result-integrity checks; never import or train the pilot model."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_official_drive_jepa_evaluation.py"
MODULE_SPECIFICATION = importlib.util.spec_from_file_location("official_evaluation_audit", MODULE_PATH)
AUDIT_MODULE = importlib.util.module_from_spec(MODULE_SPECIFICATION)
MODULE_SPECIFICATION.loader.exec_module(AUDIT_MODULE)


class TestOfficialEvaluationAggregation(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary_directory.name)
        self.result_root = self.workspace / "results/official_drive_jepa_reproduction"
        self.result_root.mkdir(parents=True)
        self.metric_names = (
            "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
            "time_to_collision_within_bound", "comfort", "score",
        )
        self.specification = {"paper": {"reported_percent_scores": dict.fromkeys(self.metric_names, 50.0)}}
        (self.result_root / "expected_scene_tokens.json").write_text(json.dumps({"tokens": ["scene_a", "scene_b", "scene_c"]}))

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_csv(self, name, tokens, scores, valid=None):
        rows = [{"token": token, "valid": True if valid is None else valid[index], **dict.fromkeys(self.metric_names, score)} for index, (token, score) in enumerate(zip(tokens, scores))]
        csv_path = self.workspace / name
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        return csv_path

    def read_report(self):
        return json.loads((self.result_root / "full_navtest_results.json").read_text())

    def test_scene_mean_not_mean_of_unequal_shards_and_average_rows(self):
        first = self.write_csv("first.csv", ["scene_a", "average"], [0.0, 0.0])
        second = self.write_csv("second.csv", ["scene_b", "scene_c", "average"], [0.5, 1.0, 0.75])
        AUDIT_MODULE.summarize_official_evaluation(self.workspace, self.specification, [first, second])
        report = self.read_report()
        self.assertTrue(report["evaluation_complete"])
        self.assertEqual(report["successful_scenes"], 3)
        self.assertEqual(report["comparison"]["score"]["server_percent_score"], 50.0)

    def test_missing_scene_is_not_silently_ignored(self):
        csv_path = self.write_csv("missing.csv", ["scene_a", "scene_b"], [0.0, 0.5])
        with self.assertRaises(RuntimeError):
            AUDIT_MODULE.summarize_official_evaluation(self.workspace, self.specification, [csv_path])
        self.assertEqual(self.read_report()["missing_tokens"], ["scene_c"])

    def test_duplicate_scene_fails_completeness(self):
        csv_path = self.write_csv("duplicate.csv", ["scene_a", "scene_b", "scene_c", "scene_c"], [0.0, 0.5, 1.0, 1.0])
        with self.assertRaises(RuntimeError):
            AUDIT_MODULE.summarize_official_evaluation(self.workspace, self.specification, [csv_path])
        self.assertEqual(self.read_report()["duplicate_tokens"], ["scene_c"])

    def test_invalid_scene_fails_completeness(self):
        csv_path = self.write_csv("invalid.csv", ["scene_a", "scene_b", "scene_c"], [0.0, 0.5, 1.0], [True, False, True])
        with self.assertRaises(RuntimeError):
            AUDIT_MODULE.summarize_official_evaluation(self.workspace, self.specification, [csv_path])
        self.assertEqual(self.read_report()["failed_tokens"], ["scene_b"])

    def test_nonfinite_metric_is_a_failure_not_skipna_success(self):
        csv_path = self.write_csv("nonfinite.csv", ["scene_a", "scene_b", "scene_c"], [0.0, float("nan"), 1.0])
        with self.assertRaises(RuntimeError):
            AUDIT_MODULE.summarize_official_evaluation(self.workspace, self.specification, [csv_path])
        self.assertEqual(self.read_report()["nonfinite_metric_tokens"], ["scene_b"])


if __name__ == "__main__":
    unittest.main()
