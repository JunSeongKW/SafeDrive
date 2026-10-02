"""CPU-only pause/resume guards. No GPU/model/worker is started."""
import json
import runpy
import tempfile
import unittest
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[1]


class WAJEPAUserPauseTests(unittest.TestCase):
    def setUp(self):
        self.acknowledge = runpy.run_path(str(WORKSPACE / "scripts/launch_official_wa_jepa_workers.py"))["acknowledge_user_pause_for_explicit_resume"]

    def test_user_pause_blocks_implicit_resume_and_keeps_marker(self):
        with tempfile.TemporaryDirectory() as fixture_directory:
            marker = Path(fixture_directory) / "evaluation_pause.json"
            marker.write_text(json.dumps({"requested_at_utc": "2026-10-02T10:12:51+00:00"}))
            with self.assertRaisesRegex(RuntimeError, "explicitly user-paused"):
                self.acknowledge(marker, False)
            self.assertTrue(marker.exists())

    def test_explicit_resume_archives_pause_without_deleting_history(self):
        with tempfile.TemporaryDirectory() as fixture_directory:
            marker = Path(fixture_directory) / "evaluation_pause.json"
            original_bytes = b'{"requested_at_utc": "2026-10-02T10:12:51+00:00"}'
            marker.write_bytes(original_bytes)
            self.acknowledge(marker, True)
            self.assertFalse(marker.exists())
            archived = list(Path(fixture_directory).glob("evaluation_pause_acknowledged_*.json"))
            self.assertEqual(len(archived), 1)
            self.assertEqual(archived[0].read_bytes(), original_bytes)

    def test_absent_pause_marker_does_not_create_files(self):
        with tempfile.TemporaryDirectory() as fixture_directory:
            self.acknowledge(Path(fixture_directory) / "evaluation_pause.json", False)
            self.assertEqual(list(Path(fixture_directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
