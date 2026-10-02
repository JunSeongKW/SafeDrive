"""CPU-only checks for shared-GPU resume; never launch a model or signal a PID."""
import json
import runpy
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


class WAJEPABoundedSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        functions = runpy.run_path(str(WORKSPACE / "scripts/schedule_official_wa_jepa_workers.py"))
        cls.capacity = staticmethod(functions["has_launch_capacity"])
        cls.pending = staticmethod(functions["pending_shards_by_gpu"])
        cls.safety = json.loads((WORKSPACE / "configs/official_wa_jepa/shared_gpu_resume_v1.json").read_text())

    def test_worker_limit_even_when_device_is_empty(self):
        self.assertFalse(self.capacity(40 * 2**30, 100 * 2**30, [5 * 2**30, 5 * 2**30], self.safety))

    def test_memory_admission_with_headroom(self):
        self.assertTrue(self.capacity(18 * 2**30, 100 * 2**30, [], self.safety))
        self.assertFalse(self.capacity(11 * 2**30, 100 * 2**30, [], self.safety))

    def test_loading_worker_peak_budget_is_reserved(self):
        self.assertFalse(self.capacity(17 * 2**30, 100 * 2**30, [0], self.safety))
        self.assertTrue(self.capacity(17 * 2**30, 100 * 2**30, [5 * 2**30], self.safety))

    def test_low_host_memory_blocks_admission(self):
        self.assertFalse(self.capacity(40 * 2**30, 23 * 2**30, [], self.safety))

    def test_gpu_specific_increase_does_not_raise_gpu1_limit(self):
        safety = {**self.safety, "maximum_concurrent_workers_by_gpu": {"0": 5, "1": 2}}
        usages = [5 * 2**30, 5 * 2**30]
        self.assertTrue(self.capacity(30 * 2**30, 100 * 2**30, usages, safety, 0))
        self.assertFalse(self.capacity(30 * 2**30, 100 * 2**30, usages, safety, 1))

    def test_partition_is_unchanged_and_completed_shards_are_not_queued(self):
        tokens = [f"scene{index:03d}" for index in range(28)]
        assignments = [0, 1] * 7
        completed = set(tokens[0::14]) | set(tokens[1::14])
        queues = self.pending(tokens, completed, assignments)
        self.assertEqual(queues[0], [2, 4, 6, 8, 10, 12])
        self.assertEqual(queues[1], [3, 5, 7, 9, 11, 13])

    def test_unexpected_saved_token_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "unexpected scenes"):
            self.pending(["expected"], {"foreign"}, [0, 1])


if __name__ == "__main__":
    unittest.main()
