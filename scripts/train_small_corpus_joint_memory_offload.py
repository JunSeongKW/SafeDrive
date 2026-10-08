"""Run the unchanged joint trainer with lossless CPU storage of large saved tensors.

Only backward storage changes. Forward operators, precision, batch composition,
RNG, loss reduction and optimizer are inherited from the registered trainer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch

import train_small_corpus_common_planner as original_training


class SavedTensorOffload:
    def __init__(self, minimum_bytes=8 * 1024**2, host_limit_bytes=24_000_000_000):
        self.minimum_bytes = minimum_bytes
        self.host_limit_bytes = host_limit_bytes
        self.live_bytes = 0
        self.peak_bytes = 0
        self.total_bytes = 0

    def pack(self, tensor):
        size = tensor.numel() * tensor.element_size()
        if tensor.device.type != "cuda" or size < self.minimum_bytes or tensor.is_leaf:
            return tensor.detach()
        if self.live_bytes + size > self.host_limit_bytes:
            raise RuntimeError("Joint saved-tensor CPU storage exceeded its registered per-rank limit")
        destination = torch.empty(tensor.size(), dtype=tensor.dtype, device="cpu", pin_memory=True)
        destination.copy_(tensor)
        self.live_bytes += size
        self.peak_bytes = max(self.peak_bytes, self.live_bytes)
        self.total_bytes += size
        return StoredActivation(destination, tensor.device, self, size)

    @staticmethod
    def unpack(value):
        if isinstance(value, StoredActivation):
            return value.tensor.to(value.device, non_blocking=True)
        return value

    def context(self):
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


class StoredActivation:
    def __init__(self, tensor, device, accounting, size):
        self.tensor, self.device = tensor, device
        self.accounting, self.size = accounting, size

    def __del__(self):
        self.accounting.live_bytes -= self.size


def install_world_model_offload(world_model, accounting):
    original_forward = world_model.forward

    def forward_with_offload(module, *arguments, **keywords):
        if torch.is_grad_enabled() and module.training:
            with accounting.context():
                return original_forward(*arguments, **keywords)
        return original_forward(*arguments, **keywords)

    world_model.forward = MethodType(forward_with_offload, world_model)


def main(arguments):
    assert arguments.kind == "lpwm_joint"
    accounting = SavedTensorOffload()
    original_objective = original_training.PlanningObjective

    class OffloadedJointObjective(original_objective):
        def __init__(self, model, oracle, joint):
            assert joint
            super().__init__(model, oracle, joint)
            install_world_model_offload(model.backbone.world_model, accounting)

    original_training.PlanningObjective = OffloadedJointObjective
    original_memory_limit = torch.cuda.set_per_process_memory_fraction

    def limited_allocator(fraction, device=None):
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device).total_memory
        return original_memory_limit(min(fraction, 24_000_000_000 / capacity), device)

    torch.cuda.set_per_process_memory_fraction = limited_allocator
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        execution="lossless_large_saved_tensor_cpu_offload",
        minimum_offload_bytes=accounting.minimum_bytes,
        per_rank_host_saved_tensor_limit_bytes=accounting.host_limit_bytes,
        per_rank_gpu_allocator_limit_bytes=24_000_000_000,
        original_trainer_sha256=hashlib.sha256(Path(original_training.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forward_operators_precision_batch_loss_and_rng_unchanged=True,
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        destination = output / "execution_offload_registration.json"
        if destination.exists():
            assert json.loads(destination.read_text()) == metadata
        else:
            original_training.write_json(destination, metadata)
    try:
        original_training.main(arguments)
    finally:
        original_training.write_json(output / ("offload_rank" + os.environ["LOCAL_RANK"] + ".json"),
            dict(peak_cpu_saved_tensor_bytes=accounting.peak_bytes,
                 cumulative_copied_bytes=accounting.total_bytes,
                 remaining_live_bytes=accounting.live_bytes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lpwm_joint"], default="lpwm_joint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    main(parser.parse_args())
