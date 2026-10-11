#!/usr/bin/env python
"""CPU-only checks of residual sampling and a real frozen DrivoR scorer.

No optimizer is constructed and no model weights are updated. An actual image
encoder forward is used ONLY once to create the tiny scorer feature cache.
Cached gradient checks install an encoder-call guard. Synthetic and real checks
are reported separately. Reference TOAD code is reused without modifying it.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.models.residual_control_sampler import (
    FrozenDrivorTrajectoryScorer, ResidualControlSampler, soft_best_candidate_loss)


def write_json(destination, value):
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(module):
    gradients = [parameter.grad.flatten() for parameter in module.parameters() if parameter.grad is not None]
    if not gradients:
        return 0.0
    combined = torch.cat(gradients)
    assert torch.isfinite(combined).all()
    return combined.norm().item()


def load_reference_helpers(configuration):
    # Use the same original DrivoR namespace as the existing checkpoint.
    from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
    import navsim.agents
    portable_agents = PROJECT_ROOT / configuration["toad_repository"] / "drivor_cem_portable/navsim/agents"
    if str(portable_agents) not in navsim.agents.__path__:
        navsim.agents.__path__.append(str(portable_agents))
    from navsim.agents.utils import cem_utils
    return CommonPlannerModel, cem_utils


def initialize_probe_heads(sampler, standard_deviation):
    for head in (sampler.mean_head, sampler.scale_head):
        torch.nn.init.normal_(head.weight, std=standard_deviation)
    return sampler


def distribution_seed(reference, proposals, initial_speed, base_trajectory, time_step):
    proposal_controls = reference.proposals_to_controls(proposals, initial_speed, time_step)
    base_controls = reference.proposals_to_controls(base_trajectory[:, None], initial_speed, time_step)[:, 0]
    floor = proposals.new_tensor([0.1, 0.025])
    standard_deviation = torch.maximum(proposal_controls.std(dim=1, unbiased=False) * 0.5, floor)
    return base_controls.detach(), standard_deviation.detach()


def check_identity(sampler, reference, features, mask, ego, proposals, initial_speed,
                   base_trajectory, candidates, time_step):
    base_controls, reference_std = distribution_seed(reference, proposals, initial_speed, base_trajectory, time_step)
    torch.manual_seed(917)
    normal_samples = torch.randn(len(features), candidates, sampler.num_control_steps * 2).reshape(
        len(features), candidates, sampler.num_control_steps, 2)
    sampled_controls, mean_correction, log_scale = sampler(
        features, mask, ego, base_controls, reference_std, normal_samples)
    original_controls = base_controls[:, None] + normal_samples * reference_std[:, None]
    assert torch.equal(sampled_controls, original_controls)
    assert mean_correction.count_nonzero() == 0 and log_scale.count_nonzero() == 0
    sampled_trajectory, _ = reference.bicycle_rollout(reference.clamp_controls(sampled_controls), initial_speed, time_step)
    trace = []
    torch.manual_seed(917)
    # Test the INITIAL samples against the unmodified official CEM implementation.
    # We do not backpropagate through top-k or the subsequent CEM update.
    reference.cem_optimize(lambda trajectory: -trajectory[..., 1].square().mean(dim=-1),
        proposals, initial_speed, torch.zeros_like(initial_speed), torch.zeros_like(initial_speed),
        init_traj=base_trajectory, dt=time_step, num_iterations=1,
        num_samples=candidates, num_elites=2, trace=trace)
    maximum_error = (sampled_trajectory - trace[1]["samples_traj"]).abs().max().item()
    assert maximum_error <= 2e-6
    return {"raw_control_maximum_error": (sampled_controls-original_controls).abs().max().item(),
            "original_TOAD_first_samples_trajectory_maximum_error": maximum_error}, normal_samples


def synthetic_checks(configuration, reference):
    started = time.perf_counter()
    torch.manual_seed(configuration["seed"])
    scene_count, candidates, control_steps = 2, configuration["sampled_candidates"], configuration["num_control_steps"]
    features = torch.randn(scene_count, 7, 1024)
    mask = torch.tensor([[True]*5+[False]*2, [True]*7])
    features[0, 5:] = float("nan")
    ego = torch.randn(scene_count, 8)
    initial_speed = torch.tensor([4.0, 7.0])
    controls = torch.randn(scene_count, 12, control_steps, 2) * torch.tensor([0.15, 0.01])
    proposals, _ = reference.bicycle_rollout(controls, initial_speed, 0.5)
    base_trajectory = proposals[:, 0]
    sampler = ResidualControlSampler(hidden_dimension=configuration["hidden_dimension"])
    identity, normal_samples = check_identity(sampler, reference, features, mask, ego,
        proposals, initial_speed, base_trajectory, candidates, 0.5)
    base_controls, reference_std = distribution_seed(reference, proposals, initial_speed, base_trajectory, 0.5)
    target = base_trajectory[:, None].clone()
    target[..., 1] += 0.7

    def objective(current_sampler):
        sampled_controls, mean_correction, log_scale = current_sampler(
            features, mask, ego, base_controls, reference_std, normal_samples)
        trajectory, _ = reference.bicycle_rollout(reference.clamp_controls(sampled_controls), initial_speed, 0.5)
        utility = -(trajectory-target).square().mean(dim=(-1, -2))
        return soft_best_candidate_loss(utility, mean_correction, log_scale)

    objective(sampler).backward()
    identity_gradients = {"feature_adapter": gradient_norm(sampler.feature_adapter),
        "mean_head": gradient_norm(sampler.mean_head), "scale_head": gradient_norm(sampler.scale_head)}
    assert identity_gradients["feature_adapter"] == 0
    assert identity_gradients["mean_head"] > 0 and identity_gradients["scale_head"] > 0
    probe = initialize_probe_heads(copy.deepcopy(sampler), configuration["nonidentity_head_standard_deviation"])
    probe.zero_grad(set_to_none=True)
    objective(probe).backward()
    gradients = {"feature_adapter": gradient_norm(probe.feature_adapter),
                 "mean_head": gradient_norm(probe.mean_head), "scale_head": gradient_norm(probe.scale_head)}
    assert min(gradients.values()) > 0
    disabled, _, _ = probe(features, mask, ego, base_controls, reference_std,
        normal_samples, sampler_enabled=False)
    assert torch.equal(disabled, base_controls[:, None]+reference_std[:, None]*normal_samples)
    return {"status": "passed", "identity": identity,
        "invalid_tokens_excluded_including_nan": True,
        "exact_identity_first_backward_gradient_norms": identity_gradients,
        "nonidentity_untrained_probe_gradient_norms": gradients,
        "sampler_OFF_restores_TOAD_controls_exactly": True,
        "trainable_parameters": sum(parameter.numel() for parameter in sampler.parameters()),
        "cpu_seconds": time.perf_counter()-started,
        "scope": "synthetic quadratic utility, not an actual planner scorer"}


def finite_difference_checks(scorer, trajectories, scene_features, ego_token):
    # FP64 prevents score cancellation in the small finite-difference test.
    double_scorer = copy.deepcopy(scorer).double().eval()
    candidate_trajectories = trajectories.detach().double().requires_grad_(True)
    scene_features, ego_token = scene_features.double(), ego_token.double()
    value = double_scorer(candidate_trajectories, scene_features, ego_token).mean()
    gradient, = torch.autograd.grad(value, candidate_trajectories)
    torch.manual_seed(182)
    rows = []
    for direction_index in range(3):
        direction = torch.randn_like(candidate_trajectories)
        direction = direction / direction.norm()
        analytical = (gradient * direction).sum().item()
        for step_size in (1e-4, 1e-5):
            with torch.no_grad():
                plus = double_scorer(candidate_trajectories+step_size*direction, scene_features, ego_token).mean()
                minus = double_scorer(candidate_trajectories-step_size*direction, scene_features, ego_token).mean()
            numerical = ((plus-minus)/(2*step_size)).item()
            absolute_error = abs(analytical-numerical)
            relative_error = absolute_error / max(abs(analytical), abs(numerical), 1e-10)
            passed = absolute_error <= 1e-7 + 1e-3*max(abs(analytical), abs(numerical))
            assert passed, (analytical, numerical, absolute_error)
            rows.append({"direction": direction_index, "step_size": step_size,
                "autograd_directional_derivative": analytical,
                "finite_difference_directional_derivative": numerical,
                "absolute_error": absolute_error, "relative_error": relative_error, "passed": passed})
    return {"dtype": "float64", "gradient_norm": gradient.norm().item(), "directions": rows}


def actual_checks(configuration, model_constructor, reference):
    started = time.perf_counter()
    visual_root = PROJECT_ROOT / configuration["visual_cache_root"]
    records = json.loads((visual_root / "subset_manifest.json").read_text())["records"]
    records = [record for record in records if record["pilot_split"] == "train"][:configuration["actual_samples"]]
    encoder_indices = [record["encoder_cache_row"] for record in records]
    source_indices = [record["source_cache_row"] for record in records]
    def read_selected(path, indices):
        return torch.from_numpy(np.array(np.load(path, mmap_mode="r")[indices]))
    visual_tokens = read_selected(visual_root / "cache/visual_tokens.npy", encoder_indices)
    valid_mask = read_selected(visual_root / "cache/valid_token_mask.npy", encoder_indices)
    ego = read_selected(visual_root / "cache/ego_status.npy", encoder_indices)
    original_ego = read_selected(PROJECT_ROOT / configuration["original_ego_cache"], source_indices)
    assert original_ego.ndim == 2 and original_ego.shape[1] == 11
    reordered_ego = torch.cat((original_ego[:, 7:11], original_ego[:, 3:5], original_ego[:, 5:7]), dim=-1)
    assert torch.equal(ego, reordered_ego)
    original_gt = read_selected(PROJECT_ROOT / configuration["original_gt_cache"], source_indices)
    cached_gt = read_selected(visual_root / "cache/trajectory.npy", encoder_indices)
    assert torch.equal(cached_gt, original_gt)
    assert all(max(record["observed_offsets_seconds"]) <= 0 for record in records)
    images = read_selected(PROJECT_ROOT / configuration["original_image_cache"], source_indices)
    if images.shape[-1] == 3:
        images = images.permute(0, 1, 4, 2, 3)
    images = images.float()/255
    model = model_constructor("drivor", None)
    saved = torch.load(PROJECT_ROOT / configuration["drivor_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    model.load_state_dict(saved["model"], strict=True)
    checkpoint_updates = saved.get("completed_updates", saved.get("update", saved.get("updates", saved.get("global_step"))))
    del saved
    model.requires_grad_(False).eval()
    before_digest = module_digest(model)
    extracted = []
    hook = model.planner.image_backbone.register_forward_hook(lambda _module, _inputs, output: extracted.append(output.detach().clone()))
    extraction_started = time.perf_counter()
    with torch.no_grad():
        original_output = model({"camera_feature": images, "ego_status": original_ego[:, None]})
        scene_features = extracted[0]
        ego_token = model.planner.hist_encoding(original_ego)[:, None].detach()
    extraction_seconds = time.perf_counter()-extraction_started
    hook.remove()
    assert len(extracted) == 1
    scorer = FrozenDrivorTrajectoryScorer(model.planner)
    candidate_proposals = original_output["proposals"].detach()
    original_scores = original_output["pdm_score"].detach()
    base_trajectory = original_output["trajectory"].detach()
    cached_calls = [0]
    def reject_backbone(_module, _arguments):
        cached_calls[0] += 1
        raise RuntimeError("Image encoder must not run during cached diagnostics")
    guard = model.planner.image_backbone.register_forward_pre_hook(reject_backbone)
    with torch.no_grad():
        scorer_values = scorer(candidate_proposals, scene_features, ego_token)
    maximum_score_error = (scorer_values-original_scores).abs().max().item()
    assert maximum_score_error <= 2e-6
    initial_speed = original_ego[:, 3].clamp_min(0)
    sampler = ResidualControlSampler(hidden_dimension=configuration["hidden_dimension"])
    identity, normal_samples = check_identity(sampler, reference, visual_tokens, valid_mask, ego,
        candidate_proposals, initial_speed, base_trajectory, configuration["sampled_candidates"], 0.5)
    base_controls, reference_std = distribution_seed(reference, candidate_proposals, initial_speed, base_trajectory, 0.5)
    probe = initialize_probe_heads(sampler, configuration["nonidentity_head_standard_deviation"])
    sampler_before = module_digest(probe)
    forward_started = time.perf_counter()
    sampled_controls, mean_correction, log_scale = probe(
        visual_tokens, valid_mask, ego, base_controls, reference_std, normal_samples)
    bounded_controls = reference.clamp_controls(sampled_controls)
    candidate_trajectories, speeds = reference.bicycle_rollout(bounded_controls, initial_speed, 0.5)
    candidate_trajectories.retain_grad()
    scores = scorer(candidate_trajectories, scene_features, ego_token)
    scorer_parameters = list(probe.feature_adapter.parameters()) + list(probe.mean_head.parameters()) + list(probe.scale_head.parameters())
    scorer_parameter_gradients = torch.autograd.grad(scores.mean(), scorer_parameters, retain_graph=True)
    adapter_count = len(list(probe.feature_adapter.parameters()))
    mean_count = len(list(probe.mean_head.parameters()))
    def gradient_list_norm(gradients):
        flattened = torch.cat([gradient.flatten() for gradient in gradients])
        assert torch.isfinite(flattened).all()
        return flattened.norm().item()
    scorer_only_module_gradients = {
        "feature_adapter": gradient_list_norm(scorer_parameter_gradients[:adapter_count]),
        "mean_head": gradient_list_norm(scorer_parameter_gradients[adapter_count:adapter_count+mean_count]),
        "scale_head": gradient_list_norm(scorer_parameter_gradients[adapter_count+mean_count:])}
    assert min(scorer_only_module_gradients.values()) > 0
    comfort_cost = reference.direct_comfort_cost(bounded_controls, speeds,
        original_ego[:, 5], torch.zeros_like(initial_speed), 0.5)
    anchor_cost = (bounded_controls-base_controls[:, None]).square().mean(dim=(-1, -2))
    utility = scores-0.05*comfort_cost-0.5*anchor_cost
    loss = soft_best_candidate_loss(utility, mean_correction, log_scale)
    backward_started = time.perf_counter()
    loss.backward()
    backward_seconds = time.perf_counter()-backward_started
    forward_backward_seconds = time.perf_counter()-forward_started
    for tensor in (sampled_controls, candidate_trajectories, scores, candidate_trajectories.grad):
        assert torch.isfinite(tensor).all()
    gradients = {"feature_adapter": gradient_norm(probe.feature_adapter),
                 "mean_head": gradient_norm(probe.mean_head), "scale_head": gradient_norm(probe.scale_head)}
    assert min(gradients.values()) > 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert module_digest(probe) == sampler_before and module_digest(model) == before_digest
    # Scorer-only gradient is checked separately from the full comfort/anchor loss.
    scorer_input = candidate_trajectories.detach().requires_grad_(True)
    scorer_gradient, = torch.autograd.grad(scorer(scorer_input, scene_features, ego_token).mean(), scorer_input)
    assert torch.isfinite(scorer_gradient).all() and scorer_gradient.norm() > 0
    finite_difference = finite_difference_checks(scorer, candidate_trajectories, scene_features, ego_token)
    cpu_profile = []
    for candidate_count in (8, 16, 32):
        cached_candidates = candidate_trajectories.detach().repeat(1, (candidate_count+7)//8, 1, 1)[:, :candidate_count].requires_grad_(True)
        saved_tensor_bytes = [0]
        def record_saved_tensor(tensor):
            # Sum of save-for-backward tensor sizes, not peak resident memory.
            saved_tensor_bytes[0] += tensor.numel()*tensor.element_size()
            return tensor
        profile_started = time.perf_counter()
        with torch.autograd.graph.saved_tensors_hooks(record_saved_tensor, lambda tensor: tensor):
            profile_score = scorer(cached_candidates, scene_features, ego_token).mean()
            torch.autograd.grad(profile_score, cached_candidates)
        cpu_profile.append({"scenes": len(records), "candidates_per_scene": candidate_count,
            "scorer_forward_backward_seconds": time.perf_counter()-profile_started,
            "saved_tensor_bytes_sum": saved_tensor_bytes[0],
            "measurement": "CPU one call; includes frozen weights saved for input backward; not GPU peak or training ETA"})
    with torch.no_grad():
        shifted_visual = visual_tokens.flip(0)
        shifted_controls, shifted_mean, shifted_scale = probe(
            shifted_visual, valid_mask, ego, base_controls, reference_std, normal_samples)
        shifted_trajectories, _ = reference.bicycle_rollout(reference.clamp_controls(shifted_controls), initial_speed, 0.5)
        first_alone = scorer(candidate_trajectories[:, :1], scene_features, ego_token)[:, 0]
        first_with_peers = scorer(candidate_trajectories, scene_features, ego_token)[:, 0]
        round_trip, _ = reference.bicycle_rollout(reference.clamp_controls(base_controls[:, None]), initial_speed, 0.5)
        mean_errors = (round_trip[:, 0, :, :2]-base_trajectory[:, :, :2]).norm(dim=-1).mean(dim=-1)
    assert cached_calls[0] == 0
    guard.remove()
    cache_root = PROJECT_ROOT / configuration["local_cache_directory"]
    cache_root.mkdir(parents=True, exist_ok=True)
    np.savez(cache_root / "actual_scorer_cache.npz", scene_features=scene_features.numpy(),
        ego_token=ego_token.numpy(), proposals=candidate_proposals.numpy(),
        base_trajectory=base_trajectory.numpy(), generated_trajectories=candidate_trajectories.detach().numpy(),
        ego_status=original_ego.numpy(), ground_truth_trajectory=original_gt.numpy(),
        reference_standard_deviation=reference_std.numpy(), initial_speed=initial_speed.numpy(),
        mean_correction=mean_correction.detach().numpy(), log_scale=log_scale.detach().numpy(),
        token=np.array([record["token"] for record in records]))
    records_path = cache_root / "sample_manifest.json"
    write_json(records_path, records)
    write_json(cache_root / "actual_scorer_cache_metadata.json", {
        "scorer_checkpoint": configuration["drivor_checkpoint"],
        "checkpoint_sha256": sha256_file(PROJECT_ROOT/configuration["drivor_checkpoint"]),
        "checkpoint_completed_updates": checkpoint_updates,
        "scorer_feature_location": "RegisterFrontEncoder output after temporal fusion, original planner image_backbone output",
        "scorer_feature_shape_per_scene": list(scene_features.shape[1:]),
        "scorer_feature_dtype": str(scene_features.dtype),
        "visual_sampler_cache": configuration["visual_cache_root"]+"/cache/visual_tokens.npy",
        "visual_sampler_metadata": configuration["visual_cache_root"]+"/cache/metadata.json",
        "sample_manifest": str(records_path.relative_to(PROJECT_ROOT)),
        "preprocessing": "front two past/current 512x256; original cropped INTER_LINEAR RGB cache; /255, ImageNet normalize, pad right6/bottom10 for DINO patch14",
        "encoder_compute_dtype": "CPU autocast bfloat16, final fusion float32",
        "encoder_frozen": True, "encoder_eval": True, "future_inputs": False,
        "observed_offsets_seconds": [record["observed_offsets_seconds"] for record in records],
        "registers_have_direct_image_coordinates": False,
        "cached_gradient_backbone_calls": cached_calls[0]})
    return {"status": "passed", "samples": len(records), "candidates_per_scene": normal_samples.shape[1],
        "tokens": [record["token"] for record in records], "checkpoint_updates": checkpoint_updates,
        "base_model": "local adapted DINOv2-S/front-history DrivoR, pass5; NOT TOAD published scorer checkpoint",
        "checkpoint_sha256": sha256_file(PROJECT_ROOT / configuration["drivor_checkpoint"]),
        "sample_matching": {"ego_exact_match": True, "gt_exact_match": True, "future_input_frames": False},
        "Drive_JEPA_visual_shape": list(visual_tokens.shape), "Drivor_scene_shape": list(scene_features.shape),
        "dtype": str(scene_features.dtype), "scorer_weights": scorer.score_weights,
        "original_vs_cached_scorer_maximum_error": maximum_score_error,
        "identity": identity, "backbone_calls_cache_creation": len(extracted),
        "backbone_calls_cached_gradient_checks": cached_calls[0],
        "all_frozen_parameter_gradients_absent": True, "all_model_weights_unchanged": True,
        "scorer_only_trajectory_gradient_norm": scorer_gradient.norm().item(),
        "surrogate_trajectory_gradient_norm": candidate_trajectories.grad.norm().item(),
        "sampling_module_gradient_norms": gradients, "finite_difference": finite_difference,
        "scorer_only_sampling_module_gradient_norms": scorer_only_module_gradients,
        "untrained_probe_visual_shuffle": {
            "mean_correction_mean_absolute_change": (shifted_mean-mean_correction).abs().mean().item(),
            "log_scale_mean_absolute_change": (shifted_scale-log_scale).abs().mean().item(),
            "raw_controls_mean_absolute_change": (shifted_controls-sampled_controls).abs().mean().item(),
            "candidate_xy_mean_displacement_m": (shifted_trajectories[..., :2]-candidate_trajectories[..., :2]).norm(dim=-1).mean().item(),
            "interpretation": "functional sensitivity of untrained nonidentity probe; NOT learned visual utility or causal performance"},
        "candidate_set_dependence_maximum_score_difference": (first_alone-first_with_peers).abs().max().item(),
        "base_trajectory_control_roundtrip_ADE_m": mean_errors.tolist(),
        "clamped_control_fraction": (sampled_controls!=bounded_controls).float().mean().item(),
        "zero_speed_fraction": (speeds==0).float().mean().item(),
        "control_ranges": {"acceleration_min": bounded_controls[..., 0].min().item(),
                           "acceleration_max": bounded_controls[..., 0].max().item(),
                           "yaw_rate_min": bounded_controls[..., 1].min().item(),
                           "yaw_rate_max": bounded_controls[..., 1].max().item()},
        "comfort_cost": {"mean": comfort_cost.mean().item(), "maximum": comfort_cost.max().item(),
                         "violating_candidate_fraction": (comfort_cost>0).float().mean().item(),
                         "past_yaw_rate_assumption": "zero; only current ego status cached"},
        "cache_creation_cpu_seconds": extraction_seconds, "surrogate_backward_cpu_seconds": backward_seconds,
        "surrogate_forward_backward_including_score_gradient_probe_cpu_seconds": forward_backward_seconds,
        "cached_scorer_cpu_profile": cpu_profile,
        "total_cpu_seconds": time.perf_counter()-started,
        "score_quality_validated": False, "official_PDMS_or_EPDMS_measured": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", default="configs/residual_control_sampler/feasibility_20261011.json")
    parser.add_argument("--synthetic-only", action="store_true")
    arguments = parser.parse_args()
    configuration = json.loads((PROJECT_ROOT / arguments.configuration).read_text())
    assert configuration["device"] == "cpu" and not configuration["training_authorized"]
    torch.set_num_threads(configuration["cpu_threads"])
    torch.manual_seed(configuration["seed"])
    torch.use_deterministic_algorithms(True)
    model_constructor, reference = load_reference_helpers(configuration)
    results_directory = PROJECT_ROOT / configuration["results_directory"]
    output = {"scope": configuration["scope"], "gpu_seconds": 0, "optimizer_steps": 0,
              "diagnostic_started_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                  "executable": sys.executable, "cpu_threads": torch.get_num_threads(),
                  "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")},
              "configuration": configuration,
              "TOAD_commit": subprocess.check_output(["git", "-C", str(PROJECT_ROOT/configuration["toad_repository"]),
                                                      "rev-parse", "HEAD"], text=True).strip(),
              "TOAD_helpers_sha256": sha256_file(Path(reference.__file__))}
    output["synthetic"] = synthetic_checks(configuration, reference)
    write_json(results_directory / "diagnostics.json", output)
    print("Synthetic checks passed", flush=True)
    if not arguments.synthetic_only:
        try:
            output["actual"] = actual_checks(configuration, model_constructor, reference)
        except Exception as error:
            output["actual"] = {"status": "failed", "exception_type": type(error).__name__, "message": str(error)}
            write_json(results_directory / "diagnostics.json", output)
            raise
    output["maximum_resident_memory_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(results_directory / "diagnostics.json", output)
    print(json.dumps({"synthetic": output["synthetic"]["status"],
                      "actual": output.get("actual", {}).get("status"), "gpu_seconds": 0}), flush=True)


if __name__ == "__main__":
    main()
