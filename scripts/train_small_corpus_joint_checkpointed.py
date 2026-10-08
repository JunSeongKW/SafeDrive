"""Keep the joint algorithm intact while recomputing RGB decoder/LPIPS activations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MethodType

import torch
from torch.utils.checkpoint import checkpoint

import train_small_corpus_common_planner as original_training


def checkpoint_module(module):
    original_forward = module.forward

    def recompute_forward(_module, *arguments, **keywords):
        if torch.is_grad_enabled():
            return checkpoint(original_forward, *arguments, use_reentrant=False, preserve_rng_state=True, **keywords)
        return original_forward(*arguments, **keywords)

    module.forward = MethodType(recompute_forward, module)


def install_joint_checkpointing(model, reconstruction):
    checkpoint_module(model.backbone.world_model.decoder_module)
    checkpoint_module(reconstruction.perceptual_loss)


def main(arguments):
    assert arguments.kind == "lpwm_joint"
    original_objective = original_training.PlanningObjective

    class CheckpointedJointObjective(original_objective):
        def __init__(self, model, oracle, joint):
            assert joint
            super().__init__(model, oracle, joint)
            install_joint_checkpointing(model, self.reconstruction)

    original_training.PlanningObjective = CheckpointedJointObjective
    original_memory_limit = torch.cuda.set_per_process_memory_fraction

    def limited_allocator(fraction, device=None):
        capacity = torch.cuda.get_device_properties(torch.cuda.current_device() if device is None else device).total_memory
        return original_memory_limit(min(fraction, 27_000_000_000 / capacity), device)

    torch.cuda.set_per_process_memory_fraction = limited_allocator
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(execution="decoder_and_lpips_activation_checkpointing",
        per_rank_gpu_allocator_limit_bytes=27_000_000_000,
        checkpointed_modules=["world_model.decoder_module", "reconstruction.perceptual_loss"],
        non_reentrant=True, preserve_rng_state=True,
        original_trainer_sha256=hashlib.sha256(Path(original_training.__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        forward_operators_precision_batch_loss_and_rng_unchanged=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        destination = output / "execution_checkpoint_registration.json"
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
    main(parser.parse_args())
