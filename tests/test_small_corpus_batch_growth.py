import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_small_corpus_scaled_batch import compile_execution_main
from queue_small_corpus_batch_growth import BatchGrowthScheduler, candidate_peak_bytes, profile_summary
import queue_small_corpus_batch_growth as scheduling

ROOT = Path(__file__).resolve().parents[1]


class BatchGrowthTests(unittest.TestCase):
    def test_compiles_real_trainer_without_modifying_registered_source(self):
        source = ROOT / "scripts/train_small_corpus_common_planner.py"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        for micro in (2, 4, 8):
            for stop in (0, 1008):
                function = compile_execution_main(SimpleNamespace(__file__=str(source)), micro, stop)
                self.assertIn(48_000_000_000, function.__code__.co_consts)
                self.assertNotIn(46_500_000_000, function.__code__.co_consts)
        self.assertEqual(before, hashlib.sha256(source.read_bytes()).hexdigest())
        with self.assertRaises(AssertionError):
            compile_execution_main(SimpleNamespace(__file__=str(source)), 3)

    def test_physical_batch_changes_keep_each_update_scene_ids_and_ssl_count(self):
        for completed in (0, 42, 639, 1280, 1920, 3199):
            expected_scenes = list(range(completed * 16, (completed + 1) * 16))
            ssl_batches = (completed + 1) * 3275 // 3200 - completed * 3275 // 3200
            for micro in (2, 4, 8):
                accumulation = 8 // micro
                self.assertEqual(micro * accumulation * 2, 16)
                observed = []
                ssl_ids = []
                for rank in (0, 1):
                    rank_scenes = expected_scenes[rank::2]
                    for index in range(accumulation):
                        observed.extend(rank_scenes[index * micro:(index + 1) * micro])
                    ssl_start = completed * 3275 // 3200 * 16
                    rank_ssl = list(range(ssl_start, ssl_start + ssl_batches * 16))[rank::2]
                    for index in range(accumulation):
                        ssl_ids.extend(rank_ssl[index * micro * ssl_batches:(index + 1) * micro * ssl_batches])
                self.assertEqual(sorted(observed), expected_scenes)
                self.assertEqual(len(ssl_ids), ssl_batches * 16)
                self.assertEqual(len(set(ssl_ids)), len(ssl_ids))

    def test_memory_admission_does_not_double_large_joint_batch_blindly(self):
        self.assertLess(candidate_peak_bytes(5_580_502_016, 2, 4) + 33_000_000_000, 48_000_000_000)
        self.assertGreater(candidate_peak_bytes(5_580_502_016, 2, 8) + 33_000_000_000, 48_000_000_000)
        self.assertGreater(candidate_peak_bytes(26_250_561_536, 2, 4) + 2_000_000_000, 48_000_000_000)

    def test_profile_requires_both_ranks_finite_and_tests_worst_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "complete.json").write_text(json.dumps(dict(completed_updates=1008, disposable_profile=True)))
            rows = [dict(completed_updates=1001 + index, loss=1., gradient_norm=2., total_objective=1.,
                         seconds=5. if index < 7 else 12., card_used_bytes=40_000_000_000,
                         peak_allocated_bytes=10_000_000_000) for index in range(8)]
            for rank in (0, 1):
                (folder / f"training_rank{rank}.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in rows))
            report = profile_summary(folder, 1008)
            self.assertTrue(report['passed'])
            self.assertEqual(report['seconds'], 5.)
            rows[-1]['card_used_bytes'] = 48_000_000_001
            (folder / 'training_rank1.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertFalse(profile_summary(folder, 1008)['passed'])
            rows[-1]['gradient_norm'] = float('nan')
            (folder / 'training_rank1.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertFalse(profile_summary(folder, 1008)['passed'])

    def test_resize_avoids_epoch_validation_boundary_and_near_finish(self):
        scheduler = BatchGrowthScheduler({'maximum_concurrent_gpu_jobs': 3})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            job = SimpleNamespace(directory=path)
            with patch.object(scheduling.base, 'prediction_ready', return_value=True):
                for update, allowed in [(639, False), (640, False), (660, True), (1919, False), (1950, True), (3100, False)]:
                    (path / 'progress.json').write_text(json.dumps(dict(completed_updates=update)))
                    self.assertEqual(scheduler.safe_resize_point(job), allowed)
            with patch.object(scheduling.base, 'prediction_ready', return_value=False):
                self.assertFalse(scheduler.safe_resize_point(job))

    def test_user_pause_is_not_reinterpreted_as_growth_authorization(self):
        scheduler = BatchGrowthScheduler({'maximum_concurrent_gpu_jobs': 3})
        scheduler.stop_requested = True
        scheduler.mark_training_pause = Mock()
        with self.assertRaises(scheduling.base.QueuePaused):
            scheduler.check_user_pause()
        scheduler.mark_training_pause.assert_called_once()


if __name__ == '__main__':
    unittest.main()
