"""Reject third-job concurrency if memory, throughput or outputs degrade."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import queue_small_corpus_three_jobs as controller


class ThirdJobAdmissionTest(unittest.TestCase):
    def decision(self, memory=20_000_000_000, concurrent_seconds=11., displacement=0.):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("baseline", "concurrent", "jepa_ssl", "drivor", "scheduling_v2"):
                (root / name).mkdir()
            protocol = dict(loss_relative_cap=.001, gradient_relative_cap=.1, trajectory_mean_cap_m=.05,
                            trajectory_maximum_cap_m=.2, trajectory_heading_cap_rad=.02,
                            profile_card_ceiling_bytes=44_000_000_000, lpwm_slowdown_cap=1.25,
                            minimum_block_speedup=1.05)
            (root / "protocol.json").write_text(json.dumps(protocol))
            (root / "scheduling_v2/jepa_ssl_and_drivor.admission.json").write_text(json.dumps(dict(parallel_step_seconds=[2.,3.])))
            for name, seconds in (("baseline",10.),("concurrent",concurrent_seconds),("jepa_ssl",2.5),("drivor",3.5)):
                rows = [dict(completed_updates=index+1, seconds=seconds, loss=1.,gradient_norm=1.,card_used_bytes=memory) for index in range(8)]
                for rank in (0,1):
                    (root / name / f"training_rank{rank}.jsonl").write_text('\n'.join(json.dumps(row) for row in rows))
                if name in ("baseline","concurrent"):
                    (root / name / "validation").mkdir()
                    trajectory=np.zeros((3,8,3))
                    if name=="concurrent": trajectory[...,0]=displacement
                    np.savez(root/name/"validation/profile_after.npz",tokens=np.arange(3),trajectories=trajectory)
            scheduler=controller.ThreeJobScheduler({})
            with patch.object(controller,"DIRECTORY",root),patch.object(controller,"OUTPUT",root):
                return scheduler.assess_third_job(root/"baseline",root/"concurrent",True,dict(jepa_ssl=0,drivor=0))

    def test_accepts_bounded_faster_third_job(self):
        self.assertTrue(self.decision()["third_job_approved"])

    def test_rejects_card_memory_above_profile_headroom(self):
        self.assertFalse(self.decision(memory=45_000_000_000)["third_job_approved"])

    def test_rejects_slowdown(self):
        self.assertFalse(self.decision(concurrent_seconds=15.)["third_job_approved"])

    def test_rejects_changed_predictions(self):
        self.assertFalse(self.decision(displacement=.1)["third_job_approved"])


if __name__ == "__main__":
    unittest.main()
