"""Read-only throughput/equivalence checks; never update the active experiment."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_lpwm_full_planning import write_json
from lpwm_stage1_effect_protocol import PROJECT_ROOT, load_stage1_effect_inputs
from lpwm_measured_card_budget_execution import check_card_budget
from planning_aware_future_prediction.object_centric.lpwm_frozen_control import build_frozen_control_model
from planning_aware_future_prediction.object_centric.lpwm_planning_finetuning import particle_attributes
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import compute_candidate_losses
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
from train_lpwm_frozen_control import make_planning_inputs


@torch.no_grad()
def extract_attributes(model, observed, status):
    with model.encoder_command(status), torch.autocast("cuda", enabled=False):
        images = observed.float().contiguous()
        if model.world_model.normalize_rgb:
            images = images * 2 - 1
        encoded = model.world_model.encoder_module(images, deterministic=True)
    arguments = (encoded["z"], encoded["z_scale"], encoded["obj_on"], encoded["z_depth"],
        encoded["z_features"], encoded["z_bg_features"], encoded["z_context"][:, 1:].contiguous(), encoded["z_score"])
    future = model._sample_future(*arguments)
    return torch.cat((particle_attributes(encoded, "obj_on"), particle_attributes(future, "z_obj_on")[:, -8:]), 1)


def run(arguments):
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    check_card_budget(0)
    # The active training has its own budget. Keep this short diagnostic below
    # 10 GiB allocated; the shared-card guard remains48 decimal GB.
    torch.cuda.set_per_process_memory_fraction(10 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    specification, world_configuration, _, checkpoint, manifest, targets, frames, world_indices = load_stage1_effect_inputs(arguments.config, "metric_plus_world")
    torch.manual_seed(47)
    model = build_frozen_control_model(checkpoint, specification, "metric_plus_world", PROJECT_ROOT).to(device).eval()
    initial_hash = model.frozen_representation_digest()
    indices = np.asarray(world_indices[:32])
    videos, status, truth = make_planning_inputs(manifest["records"], indices, frames, targets, device, True)
    observed = videos[:, :4].contiguous()
    results = {}
    reference_attributes = None
    for batch_size in (8, 16, 32):
        measurements = []
        within_batch_reference = None
        within_batch_difference = 0.
        for repeat in range(4):
            check_card_budget(0)
            torch.manual_seed(1234)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                attributes = torch.cat([extract_attributes(model, observed[offset:offset + batch_size], status[offset:offset + batch_size])
                    for offset in range(0, len(indices), batch_size)], 0)
            torch.cuda.synchronize()
            if repeat:
                measurements.append(time.monotonic() - started)
            if within_batch_reference is None:
                within_batch_reference = attributes.detach().clone()
            within_batch_difference = max(within_batch_difference, float((attributes - within_batch_reference).abs().max()))
        if reference_attributes is None:
            reference_attributes = attributes.detach().clone()
        difference = (attributes - reference_attributes).abs()
        results[str(batch_size)] = {"median_seconds_for32scenes": statistics.median(measurements),
            "max_attribute_difference_vs_batch8": float(difference.max()),
            "mean_attribute_difference_vs_batch8": float(difference.mean()),
            "max_observed_difference_vs_batch8": float(difference[:, :4].max()),
            "max_future_difference_vs_batch8": float(difference[:, 4:].max()),
            "maximum_same_batch_repeat_difference": within_batch_difference,
            "bitwise_attributes_equal": torch.equal(attributes, reference_attributes),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}
        print("BATCH", batch_size, json.dumps(results[str(batch_size)]), flush=True)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    perceptual = LossLPIPS(normalized_rgb=False).to(device).eval().requires_grad_(False)
    teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    teacher_batch = torch.from_numpy(np.array(teacher[indices[:8]])).to(device)
    gradient_reference = None
    logit_reference = None
    monitor_results = {}
    for monitor in (True, False):
        model.train()
        measurements = []
        repeat_gradient_reference = None
        repeat_gradient_difference = 0.
        for repeat in range(4):
            check_card_budget(0)
            model.zero_grad(set_to_none=True)
            torch.manual_seed(1234)
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if monitor:
                    prediction = model(observed[:8], status[:8], videos[:4], status[:4], perceptual, world_configuration["loss"])
                else:
                    prediction = model(observed[:8], status[:8])
                loss = compute_candidate_losses(prediction, truth[:8], model.trajectory_vocabulary, teacher_batch,
                    specification["candidate_imitation_temperature_meters"], specification["metric_loss_weight"])["objective"]
                objective = loss + specification["world_objective_weight"] * prediction["world_objective"]
            objective.backward()
            torch.cuda.synchronize()
            if repeat:
                measurements.append(time.monotonic() - started)
            repeat_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
            if repeat_gradient_reference is None:
                repeat_gradient_reference = repeat_gradients
            repeat_gradient_difference = max(repeat_gradient_difference,
                max(float((gradient - repeat_gradient_reference[name]).abs().max()) for name, gradient in repeat_gradients.items()))
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
        if gradient_reference is None:
            gradient_reference = gradients
            logit_reference = prediction["metric_logits"].detach().clone()
        assert gradients.keys() == gradient_reference.keys()
        monitor_results[str(monitor)] = {"median_seconds_for8scenes_forward_backward": statistics.median(measurements),
            "all_planner_gradients_bitwise_equal": all(torch.equal(gradient, gradient_reference[name]) for name, gradient in gradients.items()),
            "max_logit_difference": float((prediction["metric_logits"] - logit_reference).abs().max()),
            "maximum_same_monitor_repeat_gradient_difference": repeat_gradient_difference,
            "max_planner_gradient_difference": max(float((gradient - gradient_reference[name]).abs().max()) for name, gradient in gradients.items())}
        print("DETACHED_MONITOR", monitor, json.dumps(monitor_results[str(monitor)]), flush=True)
    assert model.frozen_representation_digest() == initial_hash
    report = {"engineering_only": True, "optimizer_updates": 0, "active_training_changed": False,
        "physical_inference_batch_benchmark": results, "detached_world_monitor_benchmark": monitor_results,
        "limitations": ["Shared GPU has concurrent training; timings are diagnostic, not an isolated speed benchmark.",
            "Any feature difference prevents claiming bitwise-equivalent cached training.",
            "One batch gradient equivalence does not prove all future steps; an execution change requires a separate recorded amendment."]}
    write_json(arguments.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    arguments.config, arguments.output = arguments.config.resolve(), arguments.output.resolve()
    run(arguments)
