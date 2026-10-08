import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from queue_small_corpus_memory_recovery import archive_markers, can_admit


class MemoryRecoveryTests(unittest.TestCase):
    def test_joint_cannot_overlap_both_large_planners(self):
        self.assertFalse(can_admit("lpwm_joint", ["jepa", "lpwm_sequential"], 17_000_000_000, 3))
        self.assertFalse(can_admit("lpwm_joint", ["drivor", "jepa", "lpwm_sequential"], 17_000_000_000, 4))
        self.assertTrue(can_admit("lpwm_joint", ["lpwm_sequential"], 8_000_000_000, 3))
        self.assertFalse(can_admit("lpwm_joint", [], 20_000_000_000, 3))

    def test_initial_three_jobs_fit_reserved_budget(self):
        self.assertTrue(can_admit("drivor", [], 30_000_000, 3))
        self.assertTrue(can_admit("jepa", ["drivor"], 30_000_000, 3))
        self.assertTrue(can_admit("lpwm_sequential", ["drivor", "jepa"], 30_000_000, 3))
        self.assertFalse(can_admit("lpwm_sequential", ["drivor", "jepa"], 30_000_000, 2))

    def test_user_pause_is_never_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pause.requested").write_text(json.dumps({"owner": "user"}))
            with self.assertRaises(AssertionError):
                archive_markers(root, root / "history", {2114751})
            self.assertTrue((root / "pause.requested").exists())
            (root / "pause.requested").write_text(json.dumps({"scheduler_pid": 2114751}))
            (root / "paused.json").write_text("{}")
            archive_markers(root, root / "history", {2114751})
            self.assertTrue((root / "history/paused.json").exists())
            self.assertFalse((root / "pause.requested").exists())


if __name__ == "__main__":
    unittest.main()
