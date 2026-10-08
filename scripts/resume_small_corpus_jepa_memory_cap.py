"""Resume unchanged JEPA training with a bounded, expandable CUDA allocator."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import resume_small_corpus_planner_full_state as full_state

ALLOCATOR_CAP_BYTES = 8_500_000_000


def main(arguments):
    assert arguments.kind == "jepa"
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
    original_limit = torch.cuda.set_per_process_memory_fraction

    def bounded_limit(fraction, device=None):
        capacity = torch.cuda.get_device_properties(
            torch.cuda.current_device() if device is None else device).total_memory
        return original_limit(min(fraction, ALLOCATOR_CAP_BYTES / capacity), device)

    torch.cuda.set_per_process_memory_fraction = bounded_limit
    registration = dict(allocator_cap_bytes=ALLOCATOR_CAP_BYTES,
        allocator_environment=os.environ["PYTORCH_CUDA_ALLOC_CONF"],
        batch_loss_precision_optimizer_and_learning_rate_unchanged=True,
        resume_wrapper_sha256=hashlib.sha256(Path(full_state.__file__).read_bytes()).hexdigest(),
        execution_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if int(os.environ["LOCAL_RANK"]) == 0:
        path = arguments.output / "execution_allocator_registration.json"
        if path.exists():
            assert json.loads(path.read_text()) == registration
        else:
            full_state.original_training.write_json(path, registration)
    full_state.main(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["jepa"], default="jepa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    main(parser.parse_args())
