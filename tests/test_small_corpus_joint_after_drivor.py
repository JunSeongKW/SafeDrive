import json
from pathlib import Path
from types import SimpleNamespace
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import queue_small_corpus_joint_after_drivor as queue


class JointAfterDrivorTests(unittest.TestCase):
    def test_wait_for_complete_training_and_all_development_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            validation = study / "drivor/validation"
            validation.mkdir(parents=True)
            self.assertFalse(queue.joint_priority_active(study))
            (study / "drivor/complete.json").write_text(json.dumps({"completed_updates": 3200}))
            self.assertFalse(queue.joint_priority_active(study))
            score = validation / "pass5.pdms.json"
            score.write_text(json.dumps({"count": 1023, "failed": 0}))
            self.assertFalse(queue.joint_priority_active(study))
            score.write_text(json.dumps({"count": 1024, "failed": 1}))
            self.assertFalse(queue.joint_priority_active(study))
            score.write_text(json.dumps({"count": 1024, "failed": 0}))
            self.assertTrue(queue.joint_priority_active(study))
            (study / "lpwm_joint").mkdir()
            (study / "lpwm_joint/complete.json").write_text("{}")
            self.assertFalse(queue.joint_priority_active(study))

    def test_priority_yield_keeps_concurrency_and_resume_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            model_directory = study / "jepa"
            model_directory.mkdir()
            (model_directory / "pause.requested").write_text(json.dumps({
                "scheduler_pid": os.getpid(), "reason": "joint_after_drivor_priority"}))
            for name in ("paused.json", "progress.json"):
                (model_directory / name).write_text(json.dumps({"completed_updates": 1234}))
            (model_directory / "latest.pt").write_bytes(b"preserved checkpoint sentinel")
            job = SimpleNamespace(directory=model_directory, label="train_jepa", log=None,
                expected=model_directory / "complete.json", running=lambda: False, returncode=lambda: 0)
            scheduler = queue.JointAfterDrivorScheduler({"maximum_concurrent_gpu_jobs": 3})
            scheduler.gpu_jobs.append(job)
            with patch.object(queue.base, "OUTPUT", study), patch.object(queue, "DIRECTORY", study / "queue"):
                scheduler.reap_finished()
            self.assertEqual(scheduler.gpu_jobs, [])
            self.assertEqual(scheduler.configuration["maximum_concurrent_gpu_jobs"], 3)
            self.assertEqual((model_directory / "latest.pt").read_bytes(), b"preserved checkpoint sentinel")
            self.assertFalse((model_directory / "pause.requested").exists())
            self.assertTrue((study / "queue/priority_yields/train_jepa/1234/saved.json").exists())

    def test_joint_pair_fits_and_jepa_is_released_after_joint(self):
        self.assertTrue(queue.priority_defers("jepa", True, 3))
        self.assertFalse(queue.priority_defers("jepa", False, 3))
        self.assertFalse(queue.priority_defers("lpwm_sequential", True, 3))
        self.assertTrue(queue.priority_defers("lpwm_sequential", True, 1))
        self.assertTrue(queue.recovery.can_admit("lpwm_joint", ["lpwm_sequential"], 9_000_000_000, 3))
        self.assertFalse(queue.recovery.can_admit("lpwm_joint", ["jepa", "lpwm_sequential"], 16_000_000_000, 3))


if __name__ == "__main__":
    unittest.main()
