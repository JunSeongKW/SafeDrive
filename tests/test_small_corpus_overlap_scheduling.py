"""Protect scientific completion and parallel-admission decisions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from queue_four_model_small_corpus_overlap import compare_pair_profiles, prediction_ready, process_identity, still_alive


class OverlapSchedulingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.configuration = dict(pair_profile_updates=8, minimum_parallel_speedup=1.05,
            minimum_profile_training_overlap=.6,
            parallel_loss_relative_tolerance=.0001, parallel_loss_absolute_tolerance=.00001,
            profile_card_ceiling_bytes=44000000000)

    def tearDown(self):
        self.temporary.cleanup()

    def profiles(self, name, seconds, loss_offset=0., memory=18000000000):
        path = self.root / name
        path.mkdir()
        rows = [dict(seconds=seconds if index else 100, loss=1/(index+1)+loss_offset,
                     elapsed_seconds=100+index*seconds,
                     card_used_bytes=memory) for index in range(8)]
        for rank in (0, 1):
            (path / f'training_rank{rank}.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        (path / 'progress.json').write_text(json.dumps(rows[-1]))
        return path

    def decision(self, parallel_seconds=3., loss_offset=0., memory=18000000000):
        serial = [self.profiles('serial_a', 2), self.profiles('serial_b', 3)]
        parallel = [self.profiles('parallel_a', parallel_seconds, loss_offset, memory),
                    self.profiles('parallel_b', parallel_seconds, loss_offset, memory)]
        return compare_pair_profiles(serial, parallel, self.configuration)

    def test_admits_faster_pair_with_matching_training_losses(self):
        decision = self.decision()
        self.assertTrue(decision['parallel_approved'])
        self.assertAlmostEqual(decision['projected_pair_speedup'], 5/3)

    def test_rejects_parallel_contention_despite_sufficient_memory(self):
        decision = self.decision(parallel_seconds=6)
        self.assertFalse(decision['parallel_approved'])
        self.assertFalse(decision['checks']['faster'])

    def test_rejects_jobs_that_did_not_actually_overlap(self):
        import os
        self.assertTrue(self.decision()['parallel_approved'])
        path = self.root / 'parallel_b/progress.json'
        shifted = path.stat().st_mtime+300
        os.utime(path, (shifted, shifted))
        decision = compare_pair_profiles([self.root/'serial_a', self.root/'serial_b'],
            [self.root/'parallel_a', self.root/'parallel_b'], self.configuration)
        self.assertFalse(decision['parallel_approved'])
        self.assertFalse(decision['checks']['actual_overlap'])

    def test_rejects_numerical_drift_even_when_parallel_is_fast(self):
        decision = self.decision(loss_offset=.01)
        self.assertFalse(decision['parallel_approved'])
        self.assertFalse(decision['checks']['loss_matches'])

    def test_rejects_unsafe_whole_card_memory(self):
        decision = self.decision(memory=45000000000)
        self.assertFalse(decision['parallel_approved'])
        self.assertFalse(decision['checks']['memory_safe'])

    def test_checks_other_rank_memory_as_well(self):
        self.assertTrue(self.decision()['parallel_approved'])
        path = self.root / 'parallel_b/training_rank1.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]['card_used_bytes'] = 45000000000
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        decision = compare_pair_profiles([self.root/'serial_a', self.root/'serial_b'],
            [self.root/'parallel_a', self.root/'parallel_b'], self.configuration)
        self.assertFalse(decision['parallel_approved'])

    def test_requires_metadata_after_prediction_file_and_full_dev_count(self):
        directory = self.root / 'condition'
        validation = directory / 'validation'
        validation.mkdir(parents=True)
        (validation / 'pass1.npz').write_bytes(b'file still being written')
        self.assertFalse(prediction_ready(directory, 1))
        (validation / 'pass1.json').write_text('{')
        self.assertFalse(prediction_ready(directory, 1))
        (validation / 'pass1.json').write_text(json.dumps({'count':3}))
        self.assertFalse(prediction_ready(directory, 1))
        (validation / 'pass1.json').write_text(json.dumps({'count':1024}))
        self.assertTrue(prediction_ready(directory, 1))

    def test_reused_pid_does_not_count_as_original_process(self):
        import os
        identity = process_identity(os.getpid())
        self.assertTrue(still_alive(identity))
        identity['start_ticks'] = 'wrong_start_time'
        self.assertFalse(still_alive(identity))


if __name__ == '__main__':
    unittest.main()
