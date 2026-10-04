"""Protect development isolation and paired-score validity in the quick study."""
from collections import Counter
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lpwm_partial_protocol import recording_balanced_indices
from evaluate_lpwm_partial_planning import paired_metrics, summarize_rows


class PartialPlanningProtocolTests(unittest.TestCase):
    def test_development_panel_is_balanced_unique_and_order_independent(self):
        records = [
            {"split": split, "recording_group": recording,
             "current_frame_token": f"{split}-{recording}-{offset}"}
            for split in ("train", "development")
            for recording in ("recording_a", "recording_b", "recording_c")
            for offset in range(5)
        ]
        selected = [records[index] for index in recording_balanced_indices(records, 8)]
        self.assertTrue(all(row["split"] == "development" for row in selected))
        self.assertEqual(len({row["current_frame_token"] for row in selected}), 8)
        self.assertEqual(sorted(Counter(row["recording_group"] for row in selected).values()), [2, 3, 3])
        reordered = list(reversed(records))
        other = [reordered[index] for index in recording_balanced_indices(reordered, 8)]
        self.assertEqual(selected, other)
        eligible = {row["current_frame_token"] for row in selected[:3]}
        filtered = [records[index]["current_frame_token"]
                    for index in recording_balanced_indices(records, 3, eligible)]
        self.assertEqual(set(filtered), eligible)
        with self.assertRaises(ValueError):
            recording_balanced_indices(records, 4, eligible)

    def test_missing_teacher_scores_do_not_become_zero_or_unpaired_gains(self):
        trained = [
            {"token": "shared", "recording_group": "recording_a", "metrics": {"pdms": .8}},
            {"token": "missing", "recording_group": "recording_b", "metrics": {"pdms": None}},
        ]
        reference = [
            {"token": "shared", "recording_group": "recording_a", "metrics": {"pdms": .6}},
            {"token": "missing", "recording_group": "recording_b", "metrics": {"pdms": .9}},
        ]
        comparison = paired_metrics(trained, reference, ("pdms",))["pdms"]
        self.assertEqual(comparison["clips"], 1)
        self.assertAlmostEqual(comparison["mean_difference"], .2)
        self.assertEqual(summarize_rows(trained), {"pdms": .8})
        self.assertEqual(summarize_rows(trained[1:]), {"pdms": None})


if __name__ == "__main__":
    unittest.main()
