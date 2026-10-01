"""CPU checks for descriptive CURRENT-input bins, not model performance tests."""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "scripts"))
MODULE_SPECIFICATION = importlib.util.spec_from_file_location(
    "official_current_context_summary", WORKSPACE / "scripts/summarize_official_navtest_current_context.py"
)
CONTEXT_MODULE = importlib.util.module_from_spec(MODULE_SPECIFICATION)
MODULE_SPECIFICATION.loader.exec_module(CONTEXT_MODULE)


class TestOfficialCurrentContextSummary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.configuration = json.loads(
            (WORKSPACE / "configs/analysis/official_navtest_current_context_v1.json").read_text()
        )

    def test_speed_boundaries_are_inclusive_lower_exclusive_upper(self):
        bins = self.configuration["speed_bins"]
        for bin_index, speed_bin in enumerate(bins):
            speed, name = CONTEXT_MODULE.classify_current_speed(
                [speed_bin["lower_inclusive_mps"], 0.0, 999.0, 999.0], bins
            )
            self.assertEqual(name, speed_bin["name"])
            self.assertEqual(speed, speed_bin["lower_inclusive_mps"])
            upper = speed_bin["upper_exclusive_mps"]
            if upper is not None:
                _, name = CONTEXT_MODULE.classify_current_speed([upper, 0.0], bins)
                self.assertEqual(name, bins[bin_index + 1]["name"])

    def test_speed_uses_velocity_norm_not_acceleration_or_future(self):
        speed, name = CONTEXT_MODULE.classify_current_speed([3.0, -4.0, 1000.0, -1000.0], self.configuration["speed_bins"])
        self.assertEqual(speed, 5.0)
        self.assertEqual(name, "medium_speed_5_to_10_mps")

    def test_missing_nonfinite_or_malformed_velocity_is_retained_as_invalid(self):
        for state in (None, [], [1.0], [float("nan"), 1.0], [float("inf"), 0.0], "not_velocity"):
            self.assertEqual(CONTEXT_MODULE.classify_current_speed(state, self.configuration["speed_bins"]), (None, "missing_or_invalid"))

    def test_command_preserves_all_four_raw_one_hot_indices(self):
        for command_index in range(4):
            command = [float(index == command_index) for index in range(4)]
            self.assertEqual(CONTEXT_MODULE.classify_stored_command(command), (command_index, f"command_index_{command_index}"))

    def test_invalid_command_is_not_silently_replaced_by_forward(self):
        for command in (None, [], [1, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0.2, 0.8, 0, 0], [float("nan"), 0, 1, 0]):
            self.assertEqual(CONTEXT_MODULE.classify_stored_command(command), (None, "missing_or_invalid"))

    def test_native_recording_counts_and_equal_scene_mean(self):
        rows = pd.DataFrame({
            "speed_bin": ["low_speed_2_to_5_mps"] * 3,
            "command_category": ["command_index_1"] * 3,
            "native_log_token": ["recording_a", "recording_a", ""],
            "native_log_name": ["export_a1", "export_a2", "export_unknown"],
            **{metric: [0.0, 0.5, 1.0] for metric in self.configuration["metric_names"]},
        })
        summary = CONTEXT_MODULE.context_group_summary(rows, "speed", "low_speed_2_to_5_mps", self.configuration)
        self.assertEqual(summary["scene_count"], 3)
        self.assertEqual(summary["native_recording_count"], 1)
        self.assertEqual(summary["exported_log_count"], 3)
        self.assertEqual(summary["missing_native_recording_id_scenes"], 1)
        self.assertEqual(summary["score_percent"], 50.0)
        self.assertFalse(summary["descriptive_interpretation_allowed"])
        joint = CONTEXT_MODULE.context_group_summary(rows, "speed_by_command", ("low_speed_2_to_5_mps", "command_index_1"), self.configuration)
        self.assertEqual(joint["scene_count"], summary["scene_count"])
        empty = CONTEXT_MODULE.context_group_summary(rows, "command", "command_index_3", self.configuration)
        self.assertEqual(empty["scene_count"], 0)
        self.assertIsNone(empty["score_percent"])

    def test_saved_analysis_is_complete_and_preserves_baseline(self):
        result_path = WORKSPACE / "results/foundation_selection/current_context_navtest_v1/context_summary.json"
        report = json.loads(result_path.read_text())
        self.assertEqual(report["expected_scenes"], 12146)
        self.assertEqual(report["missing_current_metadata_scenes"], 0)
        self.assertEqual(report["preserved_score_sha256_before"], report["preserved_score_sha256_after"])
        self.assertEqual(CONTEXT_MODULE.file_sha256(WORKSPACE / "results/official_drive_jepa_reproduction/official_scene_scores.csv"), report["preserved_score_sha256_after"])
        self.assertEqual(CONTEXT_MODULE.file_sha256(WORKSPACE / "configs/analysis/official_navtest_current_context_v1.json"), report["configuration_sha256"])
        for axis in ("speed", "command", "speed_by_command"):
            self.assertEqual(sum(group["scene_count"] for group in report["groups"] if group["axis"] == axis), 12146)
        self.assertFalse(report["gpu_or_inference_or_training_used"])
        self.assertAlmostEqual(report["overall_percent_metrics"]["score"], 89.22432019890219)


if __name__ == "__main__":
    unittest.main()
