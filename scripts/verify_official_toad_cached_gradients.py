#!/usr/bin/env python
"""CPU double-precision finite differences on the frozen official scorer cache."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from calibrate_official_toad_navsim_v1_scorer import (
    PROJECT_ROOT, load_official_model, protected_files_audit, write_json, file_sha256,
)
from diagnose_residual_control_sampler import module_digest


def verify_cached_gradients(configuration):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    started = time.perf_counter()
    output = PROJECT_ROOT / configuration["output_directory"]
    cache = np.load(output / "scorer_cache_and_candidates.npz")
    model, loading = load_official_model(configuration, "cpu")
    model.double().eval().requires_grad_(False)
    initial_digest = module_digest(model)
    backbone_calls = []
    def reject_backbone(_module, _inputs):
        backbone_calls.append(1)
        raise RuntimeError("Backbone called in cached CPU validation")
    hook = model.image_backbone.register_forward_pre_hook(reject_backbone)
    registration = json.loads((output / "registration.json").read_text())
    results = []
    generator = torch.Generator().manual_seed(configuration["noise_seed"])
    for scene_index in range(configuration["finite_difference_scene_count"]):
        native = torch.tensor(cache["trajectories"][scene_index:scene_index+1, :128], dtype=torch.float64)
        nearby = torch.tensor(cache["trajectories"][scene_index:scene_index+1, 128:192], dtype=torch.float64)
        visual = torch.tensor(cache["scene_features"][scene_index:scene_index+1], dtype=torch.float64)
        ego_token = torch.tensor(cache["ego_tokens"][scene_index:scene_index+1], dtype=torch.float64)
        current_ego = registration["records"][scene_index]["ego_history"][-1]
        initial_speed = native.new_tensor([current_ego[3]])
        native_controls = model._proposals_to_controls(native, initial_speed, .5)
        base_controls = model._proposals_to_controls(nearby[:, :1], initial_speed, .5)[:, 0]
        minimum_std = native.new_tensor([configuration["acceleration_standard_deviation_floor"],
                                        configuration["yaw_rate_standard_deviation_floor"]])
        reference_std = torch.maximum(native_controls.std(dim=1, unbiased=False) *
                                      configuration["initial_standard_deviation_scale"], minimum_std)
        correction = torch.zeros_like(base_controls, requires_grad=True)
        controls = model._clamp_controls(base_controls + reference_std * correction)
        path, _ = model._bicycle_rollout(controls[:, None], initial_speed, .5)
        context = torch.cat((nearby[:, :1], path, nearby[:, 2:]), dim=1)
        objective = model._score_trajectories(context, visual, ego_token)[0, 1]
        gradient, = torch.autograd.grad(objective, correction)
        assert torch.isfinite(gradient).all() and gradient.norm() > 0
        directions = (gradient / gradient.norm(), torch.randn(gradient.shape, generator=generator, dtype=torch.float64))
        comparisons = []
        with torch.no_grad():
            replay = model._score_trajectories(nearby, visual, ego_token)
            replay_error = float((replay - torch.tensor(cache["nearby_rankings"][scene_index:scene_index+1], dtype=torch.float64)).abs().max())
            for direction in directions:
                direction = direction / direction.norm()
                epsilon = 1e-4
                shifted_scores = []
                for sign in (1, -1):
                    shifted_controls = model._clamp_controls(base_controls + sign * epsilon * reference_std * direction)
                    shifted_path, _ = model._bicycle_rollout(shifted_controls[:, None], initial_speed, .5)
                    shifted_context = torch.cat((nearby[:, :1], shifted_path, nearby[:, 2:]), dim=1)
                    shifted_scores.append(float(model._score_trajectories(shifted_context, visual, ego_token)[0, 1]))
                analytic = float((gradient * direction).sum())
                numerical = (shifted_scores[0] - shifted_scores[1]) / (2 * epsilon)
                absolute_error = abs(analytic - numerical)
                passed = absolute_error <= 1e-7 + 1e-3 * max(abs(analytic), abs(numerical))
                comparisons.append({"analytic": analytic, "finite_difference": numerical,
                    "epsilon": epsilon, "absolute_error": absolute_error,
                    "relative_error": absolute_error / max(abs(analytic), abs(numerical), 1e-12),
                    "passed": passed})
        results.append({"token": str(cache["tokens"][scene_index]), "gradient_norm": float(gradient.norm()),
            "fp64_vs_cached_fp32_score_maximum_error": replay_error, "directions": comparisons})
    hook.remove()
    assert not backbone_calls
    assert all(parameter.grad is None for parameter in model.parameters())
    assert initial_digest == module_digest(model)
    report = {"all_finite_difference_checks_passed": all(row["passed"] for scene in results for row in scene["directions"]),
        "scene_count": len(results), "direction_count": sum(len(scene["directions"]) for scene in results),
        "dtype": "float64", "device": "cpu", "gpu_seconds": 0, "optimizer_steps": 0,
        "encoder_calls": 0, "frozen_weights_unchanged": True, "frozen_parameter_gradients_absent": True,
        "finite_difference_tolerance": {"absolute": 1e-7, "relative": 1e-3},
        "scene_results": results, "cpu_wall_seconds": time.perf_counter() - started,
        "environment": {"torch": torch.__version__}, "strict_loading": loading["strict_loading"],
        "source_sha256": file_sha256(Path(__file__)), "protected_training": protected_files_audit(configuration),
        "note": "Double precision isolates derivatives from FP32 score quantization. Not a planning performance measurement."}
    write_json(PROJECT_ROOT / configuration["results_directory"] / "cpu_gradient_validation.json", report)
    print(json.dumps(report), flush=True)
    assert report["all_finite_difference_checks_passed"], "Actual scorer input derivative validation failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args()
    verify_cached_gradients(json.loads(arguments.configuration.read_text()))
