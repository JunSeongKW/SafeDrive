from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from queue_small_corpus_batch_growth_now import measured_other_budget
from queue_small_corpus_batch_growth import candidate_peak_bytes


class MeasuredAdmissionTests(unittest.TestCase):
    def test_sequential_four_fits_measured_three_job_peak_but_eight_does_not(self):
        budget = measured_other_budget(41_555_066_880, 5_580_502_016,
                                       {'jepa':7_134_348_800, 'lpwm_joint':26_250_561_536})
        self.assertEqual(budget, 35_974_564_864)
        self.assertLess(candidate_peak_bytes(5_580_502_016, 2, 4) + budget, 48_000_000_000)
        self.assertGreater(candidate_peak_bytes(5_580_502_016, 2, 8) + budget, 48_000_000_000)

    def test_allocator_peak_floor_handles_staggered_samples(self):
        self.assertEqual(measured_other_budget(20_000_000_000, 5_000_000_000,
                                             {'one':10_000_000_000, 'two':12_000_000_000}), 24_000_000_000)


if __name__ == '__main__':
    unittest.main()
