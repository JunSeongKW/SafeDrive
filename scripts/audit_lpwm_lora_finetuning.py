"""Verify LoRA initialization, causal gradients and frozen native weights."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import audit_lpwm_planning_finetuning as original_audit
from audit_lpwm_partial_finetuning import frozen_digest
from evaluate_lpwm_full_planning import digest, write_json
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
from planning_aware_future_prediction.object_centric.lpwm_lora_finetuning import (
    build_lora_planning_model, lora_parameter_inventory)
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import build_partial_planning_model
from train_lpwm_partial_planning import load_training_inputs, make_planning_inputs, module_gradient_norms


def run(arguments):
    device = torch.device(arguments.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(arguments.maximum_reserved_gib * 1024**3 /
            torch.cuda.get_device_properties(device).total_memory, device)
    specification, stage1, _, checkpoint, manifest, targets, frames, world_indices = load_training_inputs(
        arguments.config, arguments.condition)
    models, frozen_before, identity = [], [], {}
    def audited_builder(*builder_arguments):
        random_state = torch.get_rng_state()
        reference = build_partial_planning_model(*builder_arguments).eval()
        torch.set_rng_state(random_state)
        model = build_lora_planning_model(*builder_arguments).eval()
        # Neither adapter initialization nor insertion changes planner weights.
        world_ids = {id(parameter) for parameter in model.world_model.parameters()}
        reference_parameters = dict(reference.named_parameters())
        assert all(torch.equal(parameter, reference_parameters[name])
            for name, parameter in model.named_parameters() if id(parameter) not in world_ids)
        observed, status, _ = make_planning_inputs(manifest["records"], np.asarray(world_indices[:1]), frames, targets, torch.device("cpu"))
        with torch.inference_mode():
            initial = reference(observed, status)
            adapted = model(observed, status)
        for name in ("metric_logits", "observed_particle_attributes", "predicted_particle_attributes"):
            difference = float((initial[name] - adapted[name]).abs().max())
            assert difference == 0, (name, difference)
            identity[name] = difference
        del reference, initial, adapted
        frozen_before.append(frozen_digest(model))
        models.append(model)
        return model
    original_audit.build_planning_model = audited_builder
    original_audit.main(arguments)
    model = models[0]
    assert frozen_digest(model) == frozen_before[0]
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    groups = model.optimizer_parameter_groups(specification["lpwm_learning_rate"], specification["planner_learning_rate"])
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identities) == len(set(identities))
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    # Verify the original SSL branch also reaches the adapted regions. These
    # diagnostic gradients are never applied to training or saved as weights.
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
    video, status, _ = make_planning_inputs(manifest["records"], np.asarray(world_indices[:1]), frames, targets, device, include_future=True)
    # The official background encoder uses view on convolution outputs. CPU
    # channels-last propagation can make those outputs non-contiguous.
    video = video.contiguous()
    model.train().zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(video[:, :4], status, video, status, reconstruction_loss, stage1["loss"])
    assert torch.isfinite(prediction["world_objective"])
    prediction["world_objective"].backward()
    ssl_gradients = module_gradient_norms(model)
    assert all(ssl_gradients[name] > 0 and np.isfinite(ssl_gradients[name]) for name in ("image_encoder", "context", "dynamics"))
    assert ssl_gradients["rgb_decoder"] == 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    report = json.loads(arguments.output.read_text())
    report.update(configuration_sha256=digest(arguments.config), parameter_inventory=lora_parameter_inventory(model),
        zero_adapter_identity_max_differences=identity, frozen_native_weights_identical_after_optimizer_step=True,
        initial_frozen_sha256=frozen_before[0], ssl_gradient_norms=ssl_gradients,
        original_ssl_objective=float(prediction["world_objective"].detach()),
        lora_model_source_sha256=digest(PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_lora_finetuning.py"))
    write_json(arguments.output, report)
    print("LORA_INITIALIZATION_GRADIENT_FREEZE_SSL_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-profile", action="store_true")
    parser.add_argument("--maximum-reserved-gib", type=float, default=20)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    arguments.output = arguments.output.resolve()
    arguments.checkpoint = arguments.checkpoint.resolve()
    run(arguments)
