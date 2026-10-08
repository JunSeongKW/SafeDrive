from pathlib import Path
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from queue_small_corpus_guarded_jepa_overlap import defer_jepa, guarded_admission
import queue_small_corpus_guarded_jepa_overlap as queue


class GuardedJepaTests(unittest.TestCase):
    def test_triple_requires_observed_peak_headroom(self):
        active = ["lpwm_joint", "lpwm_sequential"]
        self.assertTrue(guarded_admission("jepa", active, 35_600_000_000, 3))
        self.assertFalse(guarded_admission("jepa", active, 37_000_000_000, 3))
        self.assertFalse(guarded_admission("jepa", active, 25_000_000_000, 2))

    def test_do_not_restart_into_same_memory_pressure(self):
        self.assertTrue(defer_jepa(2, 2))
        self.assertFalse(defer_jepa(1, 2))
        self.assertTrue(defer_jepa(1, 1))
        self.assertFalse(defer_jepa(0, 1))
        self.assertFalse(defer_jepa(2, None))

    def test_other_admissions_retain_conservative_policy(self):
        self.assertFalse(guarded_admission("lpwm_joint", ["jepa", "lpwm_sequential"], 20_000_000_000, 3))
        self.assertTrue(guarded_admission("jepa", [], 30_000_000, 3))

    def test_memory_pressure_saves_only_jepa_and_preserves_companions(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            jobs = []
            for kind in ("jepa", "lpwm_sequential", "lpwm_joint"):
                model_directory = study / kind
                model_directory.mkdir()
                jobs.append(SimpleNamespace(label="train_" + kind, directory=model_directory,
                    running=lambda: True, expected=model_directory / "complete.json", log=None,
                    returncode=lambda: 0))
            scheduler = queue.GuardedJepaOverlapScheduler({"maximum_concurrent_gpu_jobs": 3})
            scheduler.gpu_jobs = jobs.copy()
            scheduler.monitor_stop = Mock()
            scheduler.monitor_stop.is_set.side_effect = [False, True]
            with patch.object(queue, "DIRECTORY", study), patch.object(queue.base, "OUTPUT", study), \
                 patch.object(queue.recovery, "current_card_bytes", return_value=[45_100_000_000, 44_000_000_000]):
                scheduler.monitor_memory()
                self.assertIsNone(scheduler.monitor_error)
                marker = study / "jepa/pause.requested"
                self.assertEqual(json.loads(marker.read_text())["scheduler_pid"], os.getpid())
                self.assertFalse((study / "lpwm_joint/pause.requested").exists())
                self.assertFalse((study / "lpwm_sequential/pause.requested").exists())
                (study / "jepa/paused.json").write_text(json.dumps({"completed_updates": 1440}))
                (study / "jepa/latest.pt").write_bytes(b"saved state sentinel")
                jobs[0].running = lambda: False
                scheduler.reap_finished()
                self.assertEqual([job.label for job in scheduler.gpu_jobs], ["train_lpwm_sequential", "train_lpwm_joint"])
                self.assertEqual(scheduler.retry_below_count, 2)
                self.assertFalse(marker.exists())
                self.assertTrue((study / "memory_yields/1440/paused.json").exists())
                self.assertEqual(scheduler.configuration["maximum_concurrent_gpu_jobs"], 3)


if __name__ == "__main__":
    unittest.main()
