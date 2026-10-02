"""Pure canonical-ID tests, not native model learning or performance evidence."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from planning_aware_future_prediction.wa_jepa_sparse_token_ids import (
    build_canonical_future_token_ids,
    decode_canonical_future_token_ids,
    derive_scene_selection_seed,
)


class WAJEPACanonicalTokenTests(unittest.TestCase):
    def create_ids(self, policy, quota=128, seed=29):
        return build_canonical_future_token_ids(
            num_cameras=4, num_future_tubelets=4, spatial_grid_rows=16,
            spatial_grid_columns=32, selection_policy=policy,
            selected_spatial_tubes_per_camera=quota, selection_seed=seed,
        )

    def test_all_ids_preserve_original_order(self):
        self.assertTrue(torch.equal(self.create_ids("packed_all", 512), torch.arange(8192).view(1, -1)))

    def test_fixed_lattice_quota_and_no_duplicates(self):
        selected_ids = self.create_ids("fixed_spatial_lattice")
        self.assertEqual(selected_ids.shape, (1, 2048))
        self.assertEqual(selected_ids.unique().numel(), 2048)
        coordinates = decode_canonical_future_token_ids(selected_ids, num_future_tubelets=4, spatial_grid_rows=16, spatial_grid_columns=32)
        self.assertTrue(torch.all(coordinates["spatial_row"] % 2 == 0))
        self.assertTrue(torch.all(coordinates["spatial_column"] % 2 == 0))
        for camera in range(4):
            self.assertEqual(int((coordinates["camera_index"] == camera).sum()), 512)

    def test_random_selection_reproducible_and_camera_independent(self):
        selected_ids = self.create_ids("seeded_random_spatial_tubes")
        self.assertTrue(torch.equal(selected_ids, self.create_ids("seeded_random_spatial_tubes")))
        self.assertFalse(torch.equal(selected_ids, self.create_ids("seeded_random_spatial_tubes", seed=30)))
        per_camera = selected_ids.reshape(4, 4, 128) % 512
        for camera in range(4):
            for tubelet in range(1, 4):
                self.assertTrue(torch.equal(per_camera[camera, 0], per_camera[camera, tubelet]))
        self.assertFalse(torch.equal(per_camera[0, 0], per_camera[1, 0]))

    def test_invalid_quota_and_unknown_policy_rejected(self):
        for policy, quota in (("packed_all", 128), ("fixed_spatial_lattice", 127), ("learned", 128), ("seeded_random_spatial_tubes", 0)):
            with self.assertRaises(ValueError):
                self.create_ids(policy, quota)

    def test_scene_seed_is_explicit_and_reproducible(self):
        self.assertEqual(derive_scene_selection_seed("014369205e025f0c", 29), derive_scene_selection_seed("014369205e025f0c", 29))
        self.assertNotEqual(derive_scene_selection_seed("014369205e025f0c", 29), derive_scene_selection_seed("other", 29))


if __name__ == "__main__":
    unittest.main()
