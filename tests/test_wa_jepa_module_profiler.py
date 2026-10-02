"""CPU-only tests of the diagnostic packing wrapper; no model or GPU loading."""
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import profile_official_wa_jepa_modules as module_profiler


class CPUPackingFixture:
    @torch.no_grad()
    def predict_trajectory(self, features, *, future_token_ids=None):
        if future_token_ids is not None:
            features = features.index_select(0, future_token_ids)
        return features


@contextmanager
def fake_timing_region(records, region_name):
    records.append(region_name)
    yield


class WAJEPAModuleProfilerTests(unittest.TestCase):
    def test_no_grad_namespace_and_packing_values_are_preserved(self):
        model = CPUPackingFixture()
        sparse_module = SimpleNamespace(MultiViewCausalFutureMaskedJEPA=CPUPackingFixture)
        features = torch.tensor([1.0, 3.0, 7.0])
        selected_ids = torch.tensor([2, 0])
        expected = model.predict_trajectory(features, future_token_ids=selected_ids)
        records = []
        with patch.object(module_profiler, "timed_cuda_region", fake_timing_region), module_profiler.instrument_sparse_packing(model, sparse_module, records):
            observed = model.predict_trajectory(features, future_token_ids=selected_ids)
        self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(records, ["future_id_validation_and_packing"])
        self.assertNotIn("predict_trajectory", model.__dict__)

    def test_dense_path_does_not_execute_packing_region(self):
        model = CPUPackingFixture()
        sparse_module = SimpleNamespace(MultiViewCausalFutureMaskedJEPA=CPUPackingFixture)
        features = torch.tensor([1.0, 3.0, 7.0])
        records = []
        with patch.object(module_profiler, "timed_cuda_region", fake_timing_region), module_profiler.instrument_sparse_packing(model, sparse_module, records):
            observed = model.predict_trajectory(features)
        self.assertTrue(torch.equal(features, observed))
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
