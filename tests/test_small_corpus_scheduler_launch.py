"""Exercise actual child startup so metadata failures cannot orphan jobs."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import queue_four_model_small_corpus_overlap_launch_fix as scheduler_module


class SchedulerLaunchTest(unittest.TestCase):
    def test_launch_registers_child_and_preserves_command_metadata(self):
        for cpu in (False, True):
            with self.subTest(cpu=cpu), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scheduling = root / "scheduling"
                scheduling.mkdir()
                expected = root / "job" / "complete.json"
                command = [sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('{}')", str(expected)]
                scheduler = scheduler_module.StudyScheduler(dict(maximum_concurrent_gpu_jobs=2))
                with patch.object(scheduler_module, "ROOT", root), patch.object(scheduler_module, "OUTPUT", root), \
                     patch.object(scheduler_module, "SCHEDULING", scheduling), \
                     patch.object(scheduler_module, "verify_original_sources", return_value={}):
                    job = scheduler.launch("example", command, expected.parent, expected, cpu=cpu)
                    try:
                        self.assertEqual(job.process.wait(timeout=10), 0)
                        metadata = json.loads((scheduling / "example.launch.json").read_text())
                        self.assertEqual(metadata["command"], command)
                        self.assertIsInstance(metadata["process_command"], str)
                        self.assertEqual(metadata["pid"], job.process.pid)
                        self.assertTrue(expected.is_file())
                        if cpu:
                            self.assertIs(scheduler.cpu_job, job)
                        else:
                            self.assertEqual(scheduler.gpu_jobs, [job])
                        self.assertIsNone(scheduler.launch("example", command, expected.parent, expected, cpu=cpu))
                    finally:
                        if job.running():
                            job.process.terminate()
                            job.process.wait(timeout=10)
                        job.log.close()


if __name__ == "__main__":
    unittest.main()
