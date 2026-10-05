"""Check real frozen LPWM state, matched initialization and planner gradients."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from lpwm_frozen_control_protocol import PROJECT_ROOT, verify_control_configuration
from evaluate_lpwm_full_planning import digest, write_json
from train_lpwm_frozen_control import load_training_inputs, make_planning_inputs, module_gradient_norms
from planning_aware_future_prediction.object_centric.lpwm_frozen_control import (
    build_frozen_control_model, frozen_control_inventory)
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import compute_candidate_losses
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT


def run(arguments):
    started = time.monotonic()
    specification = verify_control_configuration(arguments.config)
    device = torch.device(arguments.device)
    torch.set_num_threads(4)
    if device.type == "cuda":
        from lpwm_measured_card_budget_execution import install_allocator_budget, check_card_budget
        torch.cuda.set_device(device)
        check_card_budget(device.index)
        install_allocator_budget(specification, profile_only=True)
        torch.cuda.set_per_process_memory_fraction(.9, device)
    _, stage1, _, checkpoint, manifest, targets, frames, world_indices = load_training_inputs(arguments.config, "metric_plus_world")
    teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    indices = np.asarray([next(index for index in world_indices if np.isfinite(teacher[index]).all())])
    videos, status, trajectory = make_planning_inputs(manifest["records"], indices, frames, targets, device, True)
    observed = videos[:, :4].contiguous()
    reference_specification = json.loads((PROJECT_ROOT / specification["frozen_control"]["reference_configuration"]).read_text())
    torch.manual_seed(specification["seed"])
    reference = build_adapter_or_full_planning_model(checkpoint, reference_specification, "metric_plus_world", PROJECT_ROOT).to(device).eval()
    torch.manual_seed(specification["seed"])
    model = build_frozen_control_model(checkpoint, specification, "metric_plus_world", PROJECT_ROOT).to(device).eval()
    reference_parameters = dict(reference.named_parameters())
    planner_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert planner_names and all(torch.equal(parameter, reference_parameters[name])
        for name, parameter in model.named_parameters() if parameter.requires_grad)
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        expected = reference(observed, status)
        initial = model(observed, status)
    differences = {name: float((initial[name] - expected[name]).abs().max())
        for name in ("metric_logits", "imitation_logits", "observed_particle_attributes", "predicted_particle_attributes")}
    assert all(value <= 1e-5 for value in differences.values()), differences
    del reference, expected
    original_hash = model.frozen_representation_digest()
    original_planner = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(0., specification["planner_learning_rate"]),
        weight_decay=specification["weight_decay"])
    assert optimizer.param_groups[0]["params"] == []
    assert len({id(parameter) for group in optimizer.param_groups for parameter in group["params"]}) == len(planner_names)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    perceptual = LossLPIPS(normalized_rgb=False).to(device).eval().requires_grad_(False)
    gradients = None
    objective_values = []
    for step in range(2):
        model.train()
        assert all(not module.training for module in model.world_model.modules())
        optimizer.zero_grad(set_to_none=True)
        torch.manual_seed(specification["seed"] + step)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            predicted = model(observed, status, videos, status, perceptual, stage1["loss"])
            assert not predicted["world_objective"].requires_grad
            assert not predicted["observed_particle_attributes"].requires_grad
            assert not predicted["predicted_particle_attributes"].requires_grad
            losses = compute_candidate_losses(predicted, trajectory, model.trajectory_vocabulary,
                torch.from_numpy(np.array(teacher[indices])).to(device),
                specification["candidate_imitation_temperature_meters"], specification["metric_loss_weight"])
            objective = losses["objective"] + specification["world_objective_weight"] * predicted["world_objective"]
        assert torch.isfinite(objective)
        objective.backward()
        gradients = module_gradient_norms(model)
        assert gradients["planner_and_command"] > 0
        assert all(gradients[name] == 0 for name in ("image_encoder", "context", "dynamics", "rgb_decoder"))
        assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
        torch.nn.utils.clip_grad_norm_(model.parameters(), specification["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        objective_values.append(float(objective.detach()))
        del predicted, objective, losses
    assert model.frozen_representation_digest() == original_hash
    assert any(not torch.equal(parameter.detach().cpu(), original_planner[name])
        for name, parameter in model.named_parameters() if parameter.requires_grad)
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        trained = model(observed, status)
        for name in ("observed_particle_attributes", "predicted_particle_attributes"):
            assert torch.equal(initial[name], trained[name]), name
        altered_future = videos.clone()
        altered_future[:, 4:] = 1 - altered_future[:, 4:]
        altered = model(observed, status, altered_future, status, perceptual, stage1["loss"])
        assert torch.equal(trained["metric_logits"], altered["metric_logits"])
        changed_command = status.clone()
        changed_command[:, :4] = 0
        changed_command[:, (int(status[0, :4].argmax()) + 1) % 4] = 1
        conditioned = model(observed, changed_command)
        assert torch.equal(trained["observed_particle_attributes"], conditioned["observed_particle_attributes"])
        assert torch.equal(trained["predicted_particle_attributes"], conditioned["predicted_particle_attributes"])
        planner_command_difference = float((trained["metric_logits"] - conditioned["metric_logits"]).abs().max())
        assert planner_command_difference > 0
    assert model.frozen_representation_digest() == original_hash
    report = {"audit_passed": True, "engineering_only": True, "device": str(device),
        "configuration_sha256": digest(arguments.config), "stage1_checkpoint_sha256": digest(checkpoint),
        "initial_adapter_equivalence_max_differences": differences,
        "initial_trainable_planner_parameters_equal": True, "frozen_weights_and_buffers_unchanged": True,
        "frozen_encoder_command_film_unchanged": True, "particle_representations_unchanged_after_optimizer_steps": True,
        "future_auxiliary_cannot_change_planning_logits": True, "world_objective_has_no_gradient": True,
        "planner_changed_after_optimizer_steps": True, "planner_responds_to_ego_command_max_difference": planner_command_difference,
        "diagnostic_objectives": objective_values, "module_gradients": gradients,
        "inventory": frozen_control_inventory(model), "seconds": time.monotonic() - started,
        "source_sha256": {name: digest(PROJECT_ROOT / name) for name in (
            "src/planning_aware_future_prediction/object_centric/lpwm_frozen_control.py",
            "scripts/train_lpwm_frozen_control.py", "scripts/audit_lpwm_frozen_control.py",
            "scripts/lpwm_frozen_control_protocol.py")},
        "diagnostic_weights_discarded": True}
    write_json(arguments.output, report)
    print(json.dumps({key: report[key] for key in ("audit_passed", "device", "module_gradients", "seconds")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    arguments.output = arguments.output.resolve()
    run(arguments)
