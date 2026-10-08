"""Use recomputation during overlap and native activation storage when exclusive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

import train_small_corpus_common_planner as original_training


class ActivationStoragePolicy:
    def __init__(self, output, native_marker):
        self.output = output
        self.native_marker = native_marker
        self.recompute = True
        self.original_memory_limit = torch.cuda.set_per_process_memory_fraction

    def install(self, module):
        original_forward = module.forward

        def adaptive_forward(_module, *arguments, **keywords):
            if self.recompute and torch.is_grad_enabled():
                return checkpoint(original_forward, *arguments, use_reentrant=False,
                                  preserve_rng_state=True, **keywords)
            return original_forward(*arguments, **keywords)

        module.forward = MethodType(adaptive_forward, module)

    def update_before_forward(self):
        if not self.recompute or not self.native_marker.exists():
            return
        rank = int(os.environ["LOCAL_RANK"])
        # Includes our CUDA context and other processes. Require low overhead
        # beyond our own reserved allocator memory before restoring the 44GB cap.
        external_and_context = original_training.card_memory(rank) - torch.cuda.memory_reserved()
        if external_and_context > 2_000_000_000:
            return
        self.original_memory_limit(44_000_000_000 / torch.cuda.get_device_properties(rank).total_memory)
        self.recompute = False
        original_training.write_json(self.output / f"native_activation_storage_rank{rank}.json",
            dict(marker=str(self.native_marker), external_and_context_bytes=external_and_context,
                 no_restart_no_rng_reset=True, allocator_limit_bytes=44_000_000_000))


def main(arguments):
    assert arguments.kind == "lpwm_joint"
    assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True"
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = arguments.native_marker.resolve()
    policy = ActivationStoragePolicy(output, marker)
    original_objective = original_training.PlanningObjective

    class AdaptiveJointObjective(original_objective):
        def __init__(self, model, oracle, joint):
            assert joint
            super().__init__(model, oracle, joint)
            policy.install(model.backbone.world_model.decoder_module)
            policy.install(self.reconstruction.perceptual_loss)

        def forward(self, *inputs, **keywords):
            policy.update_before_forward()
            return super().forward(*inputs, **keywords)

    original_training.PlanningObjective = AdaptiveJointObjective

    def initial_allocator_limit(fraction, device=None):
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device).total_memory
        return policy.original_memory_limit(min(fraction, 27_000_000_000 / capacity), device)

    torch.cuda.set_per_process_memory_fraction = initial_allocator_limit
    metadata = dict(execution="adaptive_decoder_and_lpips_activation_storage",
        initial_allocator_limit_bytes=27_000_000_000, exclusive_allocator_limit_bytes=44_000_000_000,
        allocator_environment=os.environ["PYTORCH_CUDA_ALLOC_CONF"], native_marker=str(marker),
        checkpointed_modules=["world_model.decoder_module", "reconstruction.perceptual_loss"],
        preserve_rng_state=True, use_reentrant=False,
        original_trainer_sha256=hashlib.sha256(Path(original_training.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        microbatch_loss_dtype_optimizer_data_order_and_effective_batch_unchanged=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        destination = output / "execution_adaptive_checkpoint_registration.json"
        if destination.exists():
            assert json.loads(destination.read_text()) == metadata
        else:
            original_training.write_json(destination, metadata)
    original_training.main(arguments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lpwm_joint"], default="lpwm_joint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--profile-updates", type=int, default=0)
    parser.add_argument("--native-marker", type=Path, required=True)
    main(parser.parse_args())
