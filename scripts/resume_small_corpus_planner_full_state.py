"""Resume the registered algorithm and restore RNG after loss/model setup."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import train_small_corpus_common_planner as original_training
import train_small_corpus_joint_adaptive_checkpointing as joint_execution


def digest_tensor(value):
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def main(arguments):
    output = arguments.output.resolve()
    rank = int(os.environ["LOCAL_RANK"])
    checkpoint = torch.load(output / "latest.pt", map_location="cpu", weights_only=False, mmap=True)
    assert checkpoint["registration"] == json.loads((output / "registration.json").read_text())
    completed = checkpoint["completed_updates"]
    assert checkpoint["scheduler"]["last_epoch"] == completed
    assert all(int(state["step"]) == completed for state in checkpoint["optimizer"]["state"].values())
    saved_rng = checkpoint["rank_rng_states"][rank]
    assert len(checkpoint["rank_rng_states"]) == 2
    del checkpoint
    original_model = original_training.CommonPlannerModel
    evidence = output / "resume_records"
    evidence.mkdir(exist_ok=True)
    proof = evidence / f"from_update{completed}_rank{rank}.json"
    assert not proof.exists(), "This restart already has a recorded RNG restoration"

    class RestoringPlanner(original_model):
        def train(self, mode=True):
            result = super().train(mode)
            if mode and not getattr(self, "_resume_rng_restored", False):
                # Native loading restores RNG before constructing the SSL LPIPS
                # network. Its initialization consumes RNG. Restore it again at
                # the actual first training iteration, after all setup/loaders.
                original_training.restore_random_state(saved_rng)
                current = original_training.random_state()
                assert torch.equal(current["torch_rng"], saved_rng["torch_rng"])
                assert torch.equal(current["cuda_rng"], saved_rng["cuda_rng"])
                self._resume_rng_restored = True
                original_training.write_json(proof, dict(completed_updates=completed,
                    next_update=completed + 1, planning_cursor=completed * 16,
                    ssl_cursor=(completed * 3275 // 3200) * 16 if arguments.kind == "lpwm_joint" else None,
                    restored_after_initialization=True,
                    torch_rng_sha256=digest_tensor(current["torch_rng"]),
                    cuda_rng_sha256=digest_tensor(current["cuda_rng"]),
                    wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
            return result

    original_training.CommonPlannerModel = RestoringPlanner
    if arguments.kind == "lpwm_joint":
        joint_execution.main(arguments)
    else:
        original_training.main(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["drivor", "jepa", "lpwm_sequential", "lpwm_joint"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--native-marker", type=Path)
    main(parser.parse_args())
