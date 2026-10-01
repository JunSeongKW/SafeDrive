import json
import sys
import unittest
from collections import defaultdict
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from cache_registered_navtrain_features import approved_gpu_is_unoccupied
from prepare_navtrain_recording_split import (
    assign_recording_splits,
    native_recording_group,
    select_nonoverlapping_current_available_windows,
)
from summarize_pilot_foundation_decision import (
    paired_recording_bootstrap,
    scene_macro_from_rows,
)


class RecordingSplitAndPairedStatisticsTests(unittest.TestCase):
    def test_graphics_memory_blocks_even_without_compute_processes(self):
        with patch(
            "cache_registered_navtrain_features.subprocess.check_output",
            side_effect=["0, GPU-example, 4096\n"],
        ):
            self.assertFalse(approved_gpu_is_unoccupied(0))
        with patch(
            "cache_registered_navtrain_features.subprocess.check_output",
            side_effect=["0, GPU-example, 25\n", ""],
        ):
            self.assertTrue(approved_gpu_is_unoccupied(0))

    def test_segment_aliases_share_native_recording(self):
        self.assertEqual(
            native_recording_group("2021.05.12.22.28.35_veh-35_00620_01164"),
            "2021.05.12.22.28.35_veh-35",
        )
        self.assertEqual(native_recording_group("capture_vehicle"), "capture_vehicle")

    def test_deterministic_three_way_assignments_exclude_mini_held_out(self):
        configuration = {
            "split_seed": 29,
            "held_out_groups": 2,
            "minimum_development_groups": 2,
        }
        recordings = [f"recording_{index}" for index in range(10)]
        first = assign_recording_splits(
            recordings, recordings[:3], [recordings[0]], configuration
        )
        second = assign_recording_splits(
            list(reversed(recordings)), recordings[:3], [recordings[0]], configuration
        )
        self.assertEqual(first, second)
        self.assertEqual(first[recordings[0]], "development")
        self.assertFalse(
            any(first[recording] == "held_out" for recording in recordings[:3])
        )
        self.assertEqual(sum(split == "development" for split in first.values()), 2)

    def test_missing_future_images_do_not_filter_current_available_window(self):
        frames = [
            {
                "timestamp": index * 500000,
                "scene_token": "scene",
                "log_token": "log",
                "token": str(index),
                "log_name": "capture_vehicle",
                "driving_command": [1, 0, 0, 0],
                "ego_dynamic_state": [2, 0, 0, 0],
            }
            for index in range(12)
        ]
        checked_frames = []

        def current_image_path(frame, _sensor_root):
            checked_frames.append(frame["token"])
            return PROJECT_ROOT / "README.md"

        with patch(
            "prepare_navtrain_recording_split.front_image_path", current_image_path
        ):
            rows, missing = select_nonoverlapping_current_available_windows(
                Path("segment.pkl"), frames, PROJECT_ROOT, [], set()
            )
        self.assertEqual(len(rows), 1)
        self.assertEqual(missing, 0)
        self.assertEqual(checked_frames, ["0", "1", "2", "3"])

    def test_recording_cluster_keeps_scene_macro_estimand(self):
        rows = [
            {
                "recording_group": "recording_one",
                "scene_token": "scene_one",
                "mean_matched_seed_difference_meters": 3.0,
            },
            {
                "recording_group": "recording_two",
                "scene_token": "scene_two",
                "mean_matched_seed_difference_meters": 0.0,
            },
            {
                "recording_group": "recording_two",
                "scene_token": "scene_three",
                "mean_matched_seed_difference_meters": 0.0,
            },
        ]
        first = paired_recording_bootstrap(rows, 100, 29)
        self.assertEqual(first["scene_macro_difference_meters"], 1.0)
        self.assertEqual(first, paired_recording_bootstrap(rows, 100, 29))
        self.assertEqual(first["recordings"], 2)

    def test_scene_macro_not_window_weighted(self):
        rows = [
            {
                "recording_group": "recording",
                "scene_token": "scene_one",
                "ade_meters": 1.0,
            }
        ] * 3
        rows += [
            {
                "recording_group": "recording",
                "scene_token": "scene_two",
                "ade_meters": 3.0,
            }
        ]
        self.assertEqual(scene_macro_from_rows(rows), 2.0)

    def test_completed_manifest_caps_and_nonoverlap_and_holdout_protection(self):
        manifest_path = (
            PROJECT_ROOT
            / "outputs/pilot_foundation_decision/navtrain_three_way_split_v1/recording_split_manifest.json"
        )
        if not manifest_path.exists():
            self.skipTest("optional actual dataset audit absent on this machine")
        manifest = json.loads(manifest_path.read_text())
        recordings = defaultdict(list)
        for row in manifest["records"]:
            recordings[row["recording_group"]].append(row)
        excluded_mini = set(manifest["assignments"]["excluded_mini_held_out_groups"])
        for recording, rows in recordings.items():
            self.assertLessEqual(len(rows), 24)
            self.assertEqual(len({row["split"] for row in rows}), 1)
            if rows[0]["split"] == "held_out":
                self.assertNotIn(recording, excluded_mini)
            ordered = sorted(rows, key=lambda row: row["history_begin_timestamp_us"])
            self.assertTrue(
                all(
                    first["future_end_timestamp_us"]
                    < second["history_begin_timestamp_us"]
                    for first, second in pairwise(ordered)
                )
            )
        self.assertGreaterEqual(manifest["recording_counts"]["development"], 40)
        self.assertFalse(manifest["held_out_model_evaluation_performed"])
        self.assertFalse(manifest["future_validity_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
