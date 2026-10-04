"""Check initial equivalence, causal planning gradients, SSL and update scopes."""
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
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import (
    build_adapter_or_full_planning_model, adaptation_parameter_inventory)
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import build_partial_planning_model
from train_lpwm_partial_planning import load_training_inputs, make_planning_inputs, module_gradient_norms
from run_lpwm_navsim_posttraining import unique_module_parameters
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import build_planning_model as build_original_full_model
from lpwm_full_continuation import restore_preserved_full_checkpoint


def run(arguments):
    device = torch.device(arguments.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(arguments.maximum_reserved_gib * 1024**3 /
            torch.cuda.get_device_properties(device).total_memory, device)
    specification, stage1, _, _, manifest, targets, frames, world_indices = load_training_inputs(
        arguments.config, arguments.condition)
    models, frozen_before, identity = [], [], {}
    adapter_method = specification["adaptation_method"] == "residual_adapter"

    def audited_builder(*builder_arguments):
        random_state = torch.get_rng_state()
        reference_builder = build_partial_planning_model if adapter_method else build_original_full_model
        reference = reference_builder(*builder_arguments).eval()
        torch.set_rng_state(random_state)
        model = build_adapter_or_full_planning_model(*builder_arguments).eval()
        world_ids = {id(parameter) for parameter in model.world_model.parameters()}
        reference_parameters = dict(reference.named_parameters())
        assert all(torch.equal(parameter, reference_parameters[name]) for name, parameter in model.named_parameters()
            if id(parameter) not in world_ids)
        observed, status, _ = make_planning_inputs(manifest["records"], np.asarray(world_indices[:1]), frames, targets,
            torch.device("cpu"))
        with torch.inference_mode():
            initial, adapted = reference(observed, status), model(observed, status)
        for name in ("metric_logits", "observed_particle_attributes", "predicted_particle_attributes"):
            difference = float((initial[name] - adapted[name]).abs().max())
            assert difference <= 1e-5, (name, difference)
            identity[name] = difference
        del reference, initial, adapted
        frozen_before.append(frozen_digest(model))
        models.append(model)
        return model

    original_audit.build_planning_model = audited_builder
    original_audit.main(arguments)
    model = models[0]
    assert frozen_digest(model) == frozen_before[0]
    groups = model.optimizer_parameter_groups(specification["lpwm_learning_rate"], specification["planner_learning_rate"])
    identities = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(identities) == len(set(identities))
    assert all(parameter.requires_grad for group in groups for parameter in group["params"])
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)

    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
    video, status, _ = make_planning_inputs(manifest["records"], np.asarray(world_indices[:1]), frames, targets,
        device, include_future=True)
    video = video.contiguous()
    model.train().zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        prediction = model(video[:, :4], status, video, status, reconstruction_loss, stage1["loss"])
    assert torch.isfinite(prediction["world_objective"])
    prediction["world_objective"].backward()
    gradients = module_gradient_norms(model)
    expected = ["image_encoder", "context", "dynamics"] + ([] if adapter_method else ["rgb_decoder"])
    assert all(gradients[name] > 0 and np.isfinite(gradients[name]) for name in expected)
    if adapter_method:
        assert gradients["rgb_decoder"] == 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    inventory = adaptation_parameter_inventory(model)
    inventory["disjoint_world_region_counts"] = {name: {
        "total": sum(parameter.numel() for parameter in parameters),
        "trainable": sum(parameter.numel() for parameter in parameters if parameter.requires_grad)}
        for name, parameters in unique_module_parameters(model.world_model).items()}
    report = json.loads(arguments.output.read_text())
    report.update(audit_passed=True, configuration_sha256=digest(arguments.config), parameter_inventory=inventory,
        initial_prediction_max_differences=identity, frozen_parameters_unchanged_after_diagnostic_update=True,
        ssl_gradient_norms=gradients, original_ssl_objective=float(prediction["world_objective"].detach()),
        model_source_sha256=digest(PROJECT_ROOT /
            "src/planning_aware_future_prediction/object_centric/lpwm_adapter_full_finetuning.py"))
    if not adapter_method:
        # Discard diagnostic updates. Check real saved weights and optimizer
        # against the original full architecture before authorizing continuation.
        model = model.cpu().eval()
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(
            specification["lpwm_learning_rate"], specification["planner_learning_rate"]))
        report["continuation_restore"] = restore_preserved_full_checkpoint(model, optimizer, specification, PROJECT_ROOT)
        reference = build_original_full_model(arguments.checkpoint, specification, arguments.condition, PROJECT_ROOT).eval()
        checkpoint = torch.load(PROJECT_ROOT / specification["resume_from_prior_full"]["checkpoint"],
            map_location="cpu", weights_only=False)
        reference.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        observed, status, _ = make_planning_inputs(manifest["records"], np.asarray(world_indices[:1]), frames, targets,
            torch.device("cpu"))
        with torch.inference_mode():
            legacy, resumed = reference(observed, status), model(observed, status)
        report["restored_checkpoint_prediction_max_differences"] = {}
        for name in ("metric_logits", "observed_particle_attributes", "predicted_particle_attributes"):
            difference = float((legacy[name] - resumed[name]).abs().max())
            assert difference == 0, (name, difference)
            report["restored_checkpoint_prediction_max_differences"][name] = difference
    write_json(arguments.output, report)
    print("ADAPTER_FULL_INITIALIZATION_GRADIENT_SSL_AUDIT_PASSED", specification["adaptation_method"], flush=True)


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
