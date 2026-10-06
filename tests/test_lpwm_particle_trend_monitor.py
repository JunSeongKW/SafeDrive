"""Check exact diagnostic milestones without loading a model or using a GPU."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import monitor_lpwm_particle_trends_every500 as monitor


class ExactCheckpointTests(unittest.TestCase):
    def test_snapshot_survives_atomic_latest_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "latest.pt"
            torch.save({"completed_updates": 500}, latest)
            update, preserved = monitor.preserve_exact_checkpoint(latest, root / "snapshots", 500)
            self.assertEqual(update, 500)
            self.assertEqual(preserved.stat().st_ino, latest.stat().st_ino)
            pending = root / "latest.pending.pt"
            torch.save({"completed_updates": 600}, pending)
            pending.replace(latest)
            self.assertEqual(torch.load(preserved, weights_only=False)["completed_updates"], 500)
            self.assertEqual(torch.load(latest, weights_only=False)["completed_updates"], 600)

    def test_older_checkpoint_does_not_satisfy_due_milestone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            latest = root / "latest.pt"
            torch.save({"completed_updates": 400}, latest)
            update, preserved = monitor.preserve_exact_checkpoint(latest, root / "snapshots", 500)
            self.assertEqual(update, 400)
            self.assertIsNone(preserved)
            self.assertEqual(list((root / "snapshots").glob("*.pt")), [])

    def test_later_checkpoint_is_rejected_instead_of_mislabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save({"completed_updates": 600}, root / "latest.pt")
            configuration = {"training_directory": root, "output_directory": root / "reports",
                             "monitor_directory": root / "diagnostics", "milestones": [500],
                             "updates_per_epoch": 1614}
            with self.assertRaisesRegex(RuntimeError, "checkpoint 500 was missed"):
                monitor.capture_due_checkpoint(configuration, 610, None)

    def test_new_intervals_preserve_epoch_and_final_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "diagnostics").mkdir()
            (root / "diagnostics/registration.json").write_text(json.dumps({"milestones": [100, 500, 1614, 3228, 40350]}))
            configuration_path = root / "configuration.json"
            configuration_path.write_text(json.dumps({"training_directory": "training", "monitor_directory": "diagnostics",
                "output_directory": "reports", "update_interval": 500, "total_updates": 40350,
                "preserve_existing_epoch_diagnostics": True}))
            with patch.object(monitor, "ROOT", root):
                configuration = monitor.load_configuration(configuration_path)
            self.assertTrue({500, 1000, 1500, 1614, 2500, 3228, 4000, 40350}.issubset(configuration["milestones"]))
            self.assertEqual(len(configuration["milestones"]), len(set(configuration["milestones"])))


if __name__ == "__main__":
    unittest.main()
