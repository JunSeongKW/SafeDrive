"""Require bounded numerical variation and fixed-scene output agreement."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import reassess_small_corpus_parallel_execution as audit


class RepeatabilityAdmissionTest(unittest.TestCase):
    def decision(self, repeated_loss=1.0003, parallel_loss=1.0004, displacement=0.):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            original = output / "scheduling_v2/profiles"
            revised = output / "scheduling_v3_calibrated"
            original.mkdir(parents=True)
            revised.mkdir()
            (original.parent / "jepa_ssl_and_drivor.admission.json").write_text(json.dumps(dict(
                checks=dict(faster=True, elapsed_training_faster=True, actual_overlap=True, memory_safe=True))))
            for kind in ("jepa_ssl", "drivor"):
                for condition, root, loss in (("serial", original / ("jepa_ssl_and_drivor_serial_" + kind), 1.),
                                             ("parallel", original / ("jepa_ssl_and_drivor_parallel1_" + kind), parallel_loss),
                                             ("repeated", revised / "profiles" / ("serial_repeat_" + kind), repeated_loss)):
                    root.mkdir(parents=True)
                    (root / "registration.json").write_text('{}')
                    rows = [dict(completed_updates=index+1, loss=loss, gradient_norm=1.) for index in range(8)]
                    for rank in (0, 1):
                        (root / f"training_rank{rank}.jsonl").write_text('\n'.join(json.dumps(row) for row in rows))
                    if kind == "jepa_ssl":
                        (root / "profile_after.json").write_text(json.dumps(dict(masked_latent_l1=1., spatial_feature_std=1.)))
                    else:
                        (root / "validation").mkdir()
                        trajectories = np.zeros((3, 8, 3))
                        if condition == "parallel":
                            trajectories[..., 0] = displacement
                        np.savez(root / "validation/profile_after.npz", tokens=np.arange(3), trajectories=trajectories)
            protocol = dict(loss_relative_cap=.001, gradient_relative_cap=.1, minimum_repeatability_envelope=.0001,
                            repeatability_envelope_multiplier=2., validation_loss_relative_cap=.005,
                            feature_std_relative_cap=.02, trajectory_mean_displacement_cap_m=.05,
                            trajectory_maximum_displacement_cap_m=.2, trajectory_maximum_heading_cap_rad=.02)
            with patch.object(audit, "OUTPUT", output), patch.object(audit, "DIRECTORY", revised), \
                 patch.object(audit, "ORIGINAL_PROFILES", original):
                return audit.assess(protocol)

    def test_allows_small_drift_within_measured_repeatability(self):
        self.assertTrue(self.decision()["parallel_approved"])

    def test_rejects_drift_larger_than_serial_repeatability(self):
        self.assertFalse(self.decision(repeated_loss=1.00001)["parallel_approved"])

    def test_rejects_large_drift_even_if_serial_is_also_unstable(self):
        self.assertFalse(self.decision(repeated_loss=1.1, parallel_loss=1.1)["parallel_approved"])

    def test_rejects_changed_planning_predictions_despite_similar_loss(self):
        self.assertFalse(self.decision(displacement=.1)["parallel_approved"])


if __name__ == "__main__":
    unittest.main()
