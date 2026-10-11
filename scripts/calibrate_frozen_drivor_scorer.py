#!/usr/bin/env python
"""CPU calibration of a frozen scorer against paired official NAVSIM v1 PDMS.

Uses separate existing environments for the model and official evaluator.
No optimizer, no model updates, and no differentiation through official PDMS.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
METRIC_NAMES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score")


def write_json(destination, content):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(destination)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_training_audit(configuration):
    registration = json.loads((PROJECT_ROOT / configuration["unchanged_running_training_registration"]).read_text())
    expected = {**registration["sources"], **registration["configurations"]}
    mismatches = [name for name, digest in expected.items()
                  if file_sha256(PROJECT_ROOT / name) != digest]
    if mismatches:
        raise RuntimeError("Existing registered training files changed: " + str(mismatches))
    progress_path = PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1/stage2/progress.json"
    return {"checked_files": len(expected), "mismatches": mismatches,
            "running_training_completed_updates": json.loads(progress_path.read_text())["completed_updates"]}


def prepare(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / configuration["corpus_directory"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    ego = np.load(manifest_path.parent / "ego.npy", mmap_mode="r")
    training_groups = {record["recording_group"] for record in manifest["records"]
                       if record["study_split"] == "train"}
    development_records = [record for record in manifest["records"] if record["study_split"] == "dev"]
    selected = []
    for command_string, count in configuration["command_counts"].items():
        command_index = int(command_string)
        eligible = [dict(record) for record in development_records
                    if int(ego[record["cache_row"], 7:11].argmax()) == command_index]
        random.Random(configuration["seed"] + command_index).shuffle(eligible)
        for record in eligible[:count]:
            record["command_index"] = command_index
            record["command_name"] = configuration["command_names"][command_string]
            assert record["split"] == "navval"
            assert record["recording_group"] not in training_groups
            assert Path(record["metric_cache_file"]).exists()
            selected.append(record)
    assert len(selected) == configuration["scene_count"]
    assert len({record["token"] for record in selected}) == len(selected)
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "records": selected,
        "manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(PROJECT_ROOT / configuration["checkpoint"]),
        "train_development_recording_overlap": 0,
        "used_previously_for_development": True, "independent_final_test": False,
        "protected_training": protected_training_audit(configuration)}
    destination = output / "registration.json"
    if destination.exists():
        previous = json.loads(destination.read_text())
        for name in ("configuration", "records", "manifest_sha256", "checkpoint_sha256"):
            assert previous[name] == registration[name], name
    else:
        write_json(destination, registration)
    write_json(PROJECT_ROOT / configuration["results_directory"] / "configuration.json", configuration)
    write_json(PROJECT_ROOT / configuration["results_directory"] / "subset_manifest.json",
               {name: registration[name] for name in ("records", "train_development_recording_overlap",
                 "used_previously_for_development", "independent_final_test", "manifest_sha256", "checkpoint_sha256")})
    print(json.dumps({"phase": "prepare", "scenes": len(selected),
        "registered_before_inference": True, "gpu_seconds": 0}), flush=True)


def ego_boundary(record, original_ego):
    """Use only previous/current raw ego frames to recover boundary yaw rate."""
    from pyquaternion import Quaternion
    path = PROJECT_ROOT / "dataset/navsim_logs/trainval" / (record["log_name"] + ".pkl")
    with path.open("rb") as stream:
        frames = pickle.load(stream)
    current_index = record["current_frame_index"]
    current, previous = frames[current_index], frames[current_index - 1]
    assert current["token"] == record["token"]
    assert str(current["cams"]["CAM_F0"]["data_path"]).endswith(Path(record["observed_front_paths"][-1]).name)
    assert np.allclose(current["ego_dynamic_state"], original_ego[3:7], atol=1e-6)
    elapsed = (current["timestamp"] - previous["timestamp"]) / 1e6
    assert 0.35 <= elapsed <= 0.65
    heading_current = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
    heading_previous = Quaternion(*previous["ego2global_rotation"]).yaw_pitch_roll[0]
    yaw_rate = np.arctan2(np.sin(heading_current-heading_previous),
                          np.cos(heading_current-heading_previous)) / elapsed
    return {"past_acceleration": float(original_ego[5]), "past_yaw_rate": float(yaw_rate),
        "previous_timestamp_offset_seconds": -elapsed, "latest_timestamp_offset_seconds": 0.0,
        "source": "raw previous/current ego poses; actual frame interval"}


def infer(configuration):
    import torch
    from diagnose_residual_control_sampler import distribution_seed, load_reference_helpers, module_digest
    from planning_aware_future_prediction.models.residual_control_sampler import FrozenDrivorTrajectoryScorer
    torch.set_num_threads(configuration["cpu_threads"])
    torch.manual_seed(configuration["seed"])
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Calibration is CPU-only"
    output = PROJECT_ROOT / configuration["output_directory"]
    registration = json.loads((output / "registration.json").read_text())
    records = registration["records"]
    model_constructor, reference = load_reference_helpers(configuration)
    model = model_constructor("drivor", None)
    saved = torch.load(PROJECT_ROOT / configuration["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    assert saved["completed_updates"] == configuration["expected_checkpoint_updates"]
    model.load_state_dict(saved["model"], strict=True)
    del saved
    model.eval().requires_grad_(False)
    initial_digest = module_digest(model)
    scorer = FrozenDrivorTrajectoryScorer(model.planner)
    cache_root = PROJECT_ROOT / configuration["corpus_directory"]
    images = np.load(cache_root / "images.npy", mmap_mode="r")
    original_egos = np.load(cache_root / "ego.npy", mmap_mode="r")
    cached_features, cached_ego_tokens, cached_controls, cached_standard_deviations = [], [], [], []
    trajectories, native_rankings, nearby_rankings, secondary_rankings, gradient_rankings = [], [], [], [], []
    scene_diagnostics = []
    encoder_cache_calls = cached_encoder_calls = 0
    native_count = configuration["native_candidate_count"]
    nearby_count = configuration["nearby_candidate_count"]
    started = time.perf_counter()
    extraction_seconds = scorer_diagnostic_seconds = 0.0
    for scene_index, record in enumerate(records):
        original_ego = torch.from_numpy(np.array(original_egos[record["cache_row"]]))[None]
        observed_images = torch.from_numpy(np.array(images[record["cache_row"]]))
        observed_images = observed_images.permute(0, 3, 1, 2)[None].contiguous().float() / 255
        boundary = ego_boundary(record, original_ego[0].numpy())
        extracted = []
        hook = model.backbone.register_forward_hook(lambda _module, _inputs, value: extracted.append(value.detach().clone()))
        extraction_started = time.perf_counter()
        with torch.no_grad():
            prediction = model({"image": observed_images, "ego_status": original_ego[:, None]})
            ego_token = model.planner.hist_encoding(original_ego)[:, None].detach()
        extraction_seconds += time.perf_counter() - extraction_started
        hook.remove()
        assert len(extracted) == 1
        encoder_cache_calls += 1
        scene_features = extracted[0]
        assert scene_features.shape == (1, 16, 256)
        def reject_encoder(_module, _inputs):
            nonlocal cached_encoder_calls
            cached_encoder_calls += 1
            raise RuntimeError("Unexpected encoder call during cached scorer diagnostic")
        guard = model.backbone.register_forward_pre_hook(reject_encoder)
        diagnostic_started = time.perf_counter()
        proposals, base = prediction["proposals"].detach(), prediction["trajectory"].detach()
        assert proposals.shape == (1, native_count, 8, 3)
        with torch.no_grad():
            native_scores = scorer(proposals, scene_features, ego_token)
        replay_error = (native_scores-prediction["pdm_score"]).abs().max().item()
        assert replay_error <= 2e-6
        initial_speed = original_ego[:, 3].clamp_min(0)
        base_controls, reference_std = distribution_seed(reference, proposals, initial_speed, base, 0.5)
        generator = torch.Generator().manual_seed(configuration["seed"] + 1000 + scene_index)
        noise = torch.randn(1, nearby_count-2, 8, 2, generator=generator)
        raw_controls = base_controls[:, None] + configuration["nearby_reference_std_multiplier"] * reference_std[:, None] * noise
        bounded_controls = reference.clamp_controls(torch.cat((base_controls[:, None], raw_controls), dim=1))
        rollouts, speeds = reference.bicycle_rollout(bounded_controls, initial_speed, 0.5)
        nearby = torch.cat((base[:, None], rollouts), dim=1).detach()
        assert nearby.shape == proposals.shape
        target_index = configuration["gradient_target_index"]
        assert target_index == 1  # Exact base fallback followed by BM roundtrip center.
        with torch.no_grad():
            nearby_scores = scorer(nearby, scene_features, ego_token)
            secondary_scores = scorer(nearby[:, :configuration["secondary_nearby_candidate_count"]], scene_features, ego_token)
        normalized_correction = torch.zeros_like(base_controls, requires_grad=True)
        variable_controls = reference.clamp_controls(base_controls + reference_std * normalized_correction)
        variable_trajectory, _ = reference.bicycle_rollout(variable_controls[:, None], initial_speed, 0.5)
        variable_context = torch.cat((nearby[:, :target_index], variable_trajectory,
                                     nearby[:, target_index+1:]), dim=1)
        neural_objective = scorer(variable_context, scene_features, ego_token)[0, target_index]
        gradient, = torch.autograd.grad(neural_objective, normalized_correction)
        assert torch.isfinite(gradient).all()
        gradient_norm = float(gradient.norm())
        gradient_direction = gradient / gradient.norm().clamp_min(1e-12)
        gradient_trajectories, gradient_scores, directional_rows = [], [], []
        with torch.no_grad():
            for radius in configuration["gradient_reference_std_step_sizes"]:
                for sign in (1, -1):
                    shifted_controls = reference.clamp_controls(base_controls + sign * radius * reference_std * gradient_direction)
                    shifted_trajectory, _ = reference.bicycle_rollout(shifted_controls[:, None], initial_speed, 0.5)
                    shifted_context = nearby.clone()
                    shifted_context[:, target_index] = shifted_trajectory[:, 0]
                    shifted_score = scorer(shifted_context, scene_features, ego_token)[0, target_index]
                    gradient_trajectories.append(shifted_trajectory[0, 0].numpy())
                    gradient_scores.append(float(shifted_score))
                    directional_rows.append({"reference_std_radius": radius, "gradient_sign": sign,
                        "neural_score_delta": float(shifted_score-nearby_scores[0, target_index]),
                        "control_displacement_rms": float((shifted_controls-base_controls).square().mean().sqrt())})
            comfort_cost = reference.direct_comfort_cost(bounded_controls, speeds,
                torch.tensor([boundary["past_acceleration"]]), torch.tensor([boundary["past_yaw_rate"]]), 0.5)
        all_trajectories = np.concatenate((proposals[0].numpy(), nearby[0].numpy(), np.stack(gradient_trajectories)), axis=0)
        assert np.isfinite(all_trajectories).all()
        assert torch.isfinite(native_scores).all() and torch.isfinite(nearby_scores).all()
        trajectories.append(all_trajectories)
        native_rankings.append(native_scores[0].numpy())
        nearby_rankings.append(nearby_scores[0].numpy())
        secondary_rankings.append(secondary_scores[0].numpy())
        gradient_rankings.append(np.array(gradient_scores))
        cached_features.append(scene_features[0].numpy())
        cached_ego_tokens.append(ego_token[0].numpy())
        cached_controls.append(base_controls[0].numpy())
        cached_standard_deviations.append(reference_std[0].numpy())
        scene_diagnostics.append({"token": record["token"], "boundary": boundary,
            "native_cached_score_maximum_error": replay_error,
            "gradient_norm_in_reference_std_coordinates": gradient_norm,
            "gradient_nonzero": gradient_norm > 1e-12, "gradient_variants": directional_rows,
            "roundtrip_xy_ADE_m": float((rollouts[:, 0, :, :2]-base[:, :, :2]).norm(dim=-1).mean()),
            "nearby_clipped_control_fraction": float((bounded_controls[:, 1:]-raw_controls).abs().gt(0).float().mean()),
            "nearby_helper_comfort_violation_fraction": float((comfort_cost > 1e-9).float().mean()),
            "same_path_candidate_set_score_maximum_delta": float((nearby_scores[:, :len(secondary_scores[0])]-secondary_scores).abs().max()),
            "reference_std_minimum": float(reference_std.min()), "reference_std_maximum": float(reference_std.max())})
        guard.remove()
        scorer_diagnostic_seconds += time.perf_counter() - diagnostic_started
        if (scene_index+1) % 8 == 0:
            print(json.dumps({"phase": "infer", "complete_scenes": scene_index+1,
                "total_scenes": len(records), "elapsed_seconds": time.perf_counter()-started}), flush=True)
    assert cached_encoder_calls == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert initial_digest == module_digest(model)
    np.savez_compressed(output / "scorer_cache_and_candidates.npz", tokens=np.array([record["token"] for record in records]),
        trajectories=np.stack(trajectories), native_rankings=np.stack(native_rankings),
        nearby_rankings=np.stack(nearby_rankings), secondary_rankings=np.stack(secondary_rankings),
        gradient_rankings=np.stack(gradient_rankings), scene_features=np.stack(cached_features),
        ego_tokens=np.stack(cached_ego_tokens), base_controls=np.stack(cached_controls),
        reference_standard_deviations=np.stack(cached_standard_deviations))
    write_json(output / "inference_complete.json", {"complete": True, "scenes": len(records),
        "checkpoint_updates": configuration["expected_checkpoint_updates"], "gpu_seconds": 0,
        "optimizer_steps": 0, "encoder_calls_for_cache_creation": encoder_cache_calls,
        "encoder_calls_for_cached_diagnostics": cached_encoder_calls, "frozen_weights_unchanged": True,
        "frozen_parameter_gradients_absent": True, "scene_feature_shape": [len(records), 16, 256],
        "scene_feature_dtype": "float32", "cache_creation_cpu_seconds": extraction_seconds,
        "scorer_diagnostics_cpu_seconds": scorer_diagnostic_seconds,
        "total_cpu_wall_seconds": time.perf_counter()-started,
        "protected_training": protected_training_audit(configuration), "scene_diagnostics": scene_diagnostics})


def official_scene_scores(job):
    """Reuse existing batched simulation, with reference-pair progress per path."""
    from dataclasses import asdict
    sys.path.insert(0, str(PROJECT_ROOT / "reference_repositories/DrivoR"))
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory, pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import MultiMetricIndex, WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    record, trajectories, native_rankings, nearby_rankings, secondary_rankings = job
    with lzma.open(record["metric_cache_file"], "rb") as stream:
        cache = pickle.load(stream)
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    initial = cache.ego_state
    reference = get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)
    states = np.stack([reference] + [get_trajectory_as_array(transform_trajectory(Trajectory(poses), initial),
        sampling, initial.time_point) for poses in trajectories])
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
    simulated = simulator.simulate_proposals(states, initial)
    scorer.score_proposals(simulated, cache.observation, cache.centerline, cache.route_lane_ids, cache.drivable_area_map)
    multiplicative = scorer._multi_metrics.prod(axis=0)
    raw_progress = scorer._progress_raw * multiplicative
    pair_maximum = np.maximum(raw_progress[0], raw_progress[1:])
    progress = np.where(pair_maximum > scorer._config.progress_distance_threshold,
        raw_progress[1:] / np.maximum(pair_maximum, 1e-12), (multiplicative[1:] != 0).astype(float))
    weighted = scorer._weighted_metrics[:, 1:].copy()
    weighted[WeightedMetricIndex.PROGRESS] = progress
    final_scores = multiplicative[1:] * (weighted * scorer._config.weighted_metrics_array[:, None]).sum(0) / scorer._config.weighted_metrics_array.sum()
    metrics = np.stack((scorer._multi_metrics[MultiMetricIndex.NO_COLLISION, 1:],
        scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA, 1:], progress,
        weighted[WeightedMetricIndex.TTC], weighted[WeightedMetricIndex.COMFORTABLE],
        weighted[WeightedMetricIndex.DRIVING_DIRECTION], final_scores), axis=-1)
    native_count, nearby_count = len(native_rankings), len(nearby_rankings)
    check_indices = {int(native_rankings.argmax()), int(final_scores[:native_count].argmax()),
        native_count+int(nearby_rankings.argmax()), native_count+int(final_scores[native_count:native_count+nearby_count].argmax()),
        native_count+int(secondary_rankings.argmax()), native_count+1, len(trajectories)-1}
    maximum_error = 0.0
    for candidate_index in sorted(check_indices):
        official = asdict(pdm_score(cache, Trajectory(trajectories[candidate_index]), sampling,
            PDMSimulator(sampling), PDMScorer(sampling)))
        errors = [abs(float(official[name])-metrics[candidate_index, metric_index])
                  for metric_index, name in enumerate(METRIC_NAMES)]
        maximum_error = max(maximum_error, max(errors))
        assert max(errors) <= 1e-8, (record["token"], candidate_index, errors)
    return {"token": record["token"], "official_metrics": metrics.tolist(),
        "pair_evaluator_maximum_error": maximum_error,
        "checked_candidate_count": len(check_indices)}


def score(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    records = json.loads((output / "registration.json").read_text())["records"]
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    assert arrays["tokens"].tolist() == [record["token"] for record in records]
    jobs = [(record, arrays["trajectories"][index], arrays["native_rankings"][index],
             arrays["nearby_rankings"][index], arrays["secondary_rankings"][index])
            for index, record in enumerate(records)]
    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=configuration["official_evaluation_workers"],
            mp_context=multiprocessing.get_context("spawn")) as pool:
        for index, row in enumerate(pool.map(official_scene_scores, jobs, chunksize=1)):
            rows.append(row)
            if (index+1) % 8 == 0:
                write_json(output / "official_score_progress.json", {"scored_scenes": index+1, "total_scenes": len(records)})
                print(json.dumps({"phase": "score", "complete_scenes": index+1,
                    "elapsed_seconds": time.perf_counter()-started}), flush=True)
    np.save(output / "official_metrics.npy", np.asarray([row["official_metrics"] for row in rows]))
    write_json(output / "official_evaluation_complete.json", {"complete": True,
        "scene_count": len(rows), "candidate_count_per_scene": arrays["trajectories"].shape[1],
        "official_evaluated_candidates": len(rows)*arrays["trajectories"].shape[1],
        "checked_against_unmodified_pair_evaluator": sum(row["checked_candidate_count"] for row in rows),
        "maximum_pair_evaluator_error": max(row["pair_evaluator_maximum_error"] for row in rows),
        "metric_names": METRIC_NAMES, "cpu_wall_seconds": time.perf_counter()-started,
        "workers": configuration["official_evaluation_workers"], "gpu_seconds": 0,
        "evaluation_definition": "Official NAVSIM v1 reference-plus-single-candidate progress normalization, reused batch simulation"})


def recording_bootstrap(values, groups, seed, repetitions):
    unique_groups = sorted(set(groups))
    sums = np.array([values[np.array(groups) == group].sum() for group in unique_groups])
    counts = np.array([sum(value == group for value in groups) for group in unique_groups])
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(unique_groups), (repetitions, len(unique_groups)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def report(configuration):
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    destination.mkdir(parents=True, exist_ok=True)
    registration = json.loads((output / "registration.json").read_text())
    records = registration["records"]
    inference = json.loads((output / "inference_complete.json").read_text())
    evaluation = json.loads((output / "official_evaluation_complete.json").read_text())
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    metrics = np.load(output / "official_metrics.npy")
    groups = [record["recording_group"] for record in records]
    scene_indices = np.arange(len(records))
    native_count, nearby_count = configuration["native_candidate_count"], configuration["nearby_candidate_count"]
    candidate_groups = {"native64": (0, arrays["native_rankings"]),
        "nearby64": (native_count, arrays["nearby_rankings"]),
        "nearby16": (native_count, arrays["secondary_rankings"])}
    group_results, scene_rows = {}, []
    base_selection = arrays["native_rankings"].argmax(1)
    base_metrics = metrics[scene_indices, base_selection]
    base_unsafe = (base_metrics[:, 0] < 1) | (base_metrics[:, 1] < 1)
    for name, (offset, rankings) in candidate_groups.items():
        current_metrics = metrics[:, offset:offset+rankings.shape[1]]
        scores = current_metrics[:, :, -1] * 100
        selected_index, best_index = rankings.argmax(1), scores.argmax(1)
        selected_metrics = current_metrics[scene_indices, selected_index]
        selected_pdms = selected_metrics[:, -1] * 100
        best_pdms = scores[scene_indices, best_index]
        regret = best_pdms-selected_pdms
        unsafe = (selected_metrics[:, 0] < 1) | (selected_metrics[:, 1] < 1)
        correlations = []
        for index in scene_indices:
            if np.ptp(scores[index]) > 1e-10 and np.ptp(rankings[index]) > 1e-10:
                correlations.append(float(spearmanr(rankings[index], scores[index]).statistic))
        group_results[name] = {"scene_count": len(records), "candidate_count": rankings.shape[1],
            "selected_pdms": float(selected_pdms.mean()), "oracle_best_pdms": float(best_pdms.mean()),
            "selection_regret_pdms_points": float(regret.mean()),
            "selection_regret_recording_bootstrap_95ci": recording_bootstrap(regret, groups,
                configuration["seed"], configuration["bootstrap_repetitions"]),
            "spearman_scene_mean": float(np.mean(correlations)) if correlations else None,
            "spearman_nonconstant_scene_count": len(correlations),
            "selected_collision_or_offroad_rate": float(unsafe.mean()),
            "unsafe_rate_change_from_native_percentage_points": float((unsafe.astype(float)-base_unsafe).mean()*100),
            "selected_mean_submetrics": {metric: float(selected_metrics[:, index].mean())
                for index, metric in enumerate(METRIC_NAMES[:-1])},
            "oracle_best_mean_submetrics": {metric: float(current_metrics[scene_indices, best_index, index].mean())
                for index, metric in enumerate(METRIC_NAMES[:-1])},
            "candidate_unsafe_fraction": float(((current_metrics[:, :, 0] < 1)|(current_metrics[:, :, 1] < 1)).mean()),
            "candidate_unsafe_but_ranked_top_quartile_fraction": float((((current_metrics[:, :, 0] < 1)|
                (current_metrics[:, :, 1] < 1)) & (rankings >= np.quantile(rankings, 0.75, axis=1)[:, None])).mean()),
            "by_command": {configuration["command_names"][command]: {
                "scenes": sum(record["command_index"] == int(command) for record in records),
                "selected_pdms": float(selected_pdms[[record["command_index"] == int(command) for record in records]].mean()),
                "oracle_best_pdms": float(best_pdms[[record["command_index"] == int(command) for record in records]].mean()),
                "selection_regret_pdms_points": float(regret[[record["command_index"] == int(command) for record in records]].mean())}
                for command in configuration["command_counts"]}}
        for index, record in enumerate(records):
            scene_rows.append({"token": record["token"], "recording_group": record["recording_group"],
                "command": record["command_name"], "candidate_group": name,
                "selected_pdms": selected_pdms[index], "oracle_best_pdms": best_pdms[index],
                "regret_pdms_points": regret[index], "selected_candidate_index": int(selected_index[index]),
                "selected_collision_or_offroad": bool(unsafe[index])})
    bm_center_metrics = metrics[:, native_count+configuration["gradient_target_index"]]
    gradient_metrics = metrics[:, native_count+nearby_count:]
    gradient_results = []
    for radius_index, radius in enumerate(configuration["gradient_reference_std_step_sizes"]):
        plus_index, minus_index = 2*radius_index, 2*radius_index+1
        plus_delta = (gradient_metrics[:, plus_index, -1]-bm_center_metrics[:, -1])*100
        minus_delta = (gradient_metrics[:, minus_index, -1]-bm_center_metrics[:, -1])*100
        plus_neural_delta = np.array([row["gradient_variants"][plus_index]["neural_score_delta"]
            for row in inference["scene_diagnostics"]])
        center_unsafe = (bm_center_metrics[:, 0] < 1)|(bm_center_metrics[:, 1] < 1)
        plus_unsafe = (gradient_metrics[:, plus_index, 0] < 1)|(gradient_metrics[:, plus_index, 1] < 1)
        gradient_results.append({"reference_std_radius": radius,
            "neural_score_plus_delta_mean": float(plus_neural_delta.mean()),
            "neural_score_plus_increase_fraction": float((plus_neural_delta > 1e-8).mean()),
            "actual_pdms_plus_delta_mean": float(plus_delta.mean()),
            "actual_pdms_plus_delta_recording_bootstrap_95ci": recording_bootstrap(plus_delta, groups,
                configuration["seed"], configuration["bootstrap_repetitions"]),
            "actual_pdms_minus_delta_mean": float(minus_delta.mean()),
            "actual_pdms_plus_delta_median": float(np.median(plus_delta)),
            "actual_pdms_plus_delta_quantiles": [float(value) for value in np.quantile(plus_delta, [0, 0.25, 0.5, 0.75, 1])],
            "actual_pdms_plus_drop_larger_than_point1_scene_count": int((plus_delta < -0.1).sum()),
            "actual_pdms_plus_drop_larger_than_1_scene_count": int((plus_delta < -1.0).sum()),
            "actual_pdms_plus_gain_larger_than_point1_scene_count": int((plus_delta > 0.1).sum()),
            "actual_pdms_plus_gain_larger_than_1_scene_count": int((plus_delta > 1.0).sum()),
            "actual_pdms_plus_improved_fraction": float((plus_delta > 1e-6).mean()),
            "actual_pdms_plus_worsened_fraction": float((plus_delta < -1e-6).mean()),
            "neural_score_up_actual_pdms_down_fraction": float(((plus_neural_delta > 1e-8)&(plus_delta < -1e-6)).mean()),
            "collision_or_offroad_plus_rate_delta_percentage_points": float((plus_unsafe.astype(float)-center_unsafe).mean()*100)})
    gates = {}
    for name in ("native64", "nearby64", "nearby16"):
        current = group_results[name]
        gates[name] = {"regret_within_5_points": current["selection_regret_pdms_points"] <= configuration[
                "maximum_native_and_nearby_selection_regret_pdms_points"],
            "unsafe_rate_degradation_within_1_percentage_point": current[
                "unsafe_rate_change_from_native_percentage_points"] <= configuration[
                "maximum_selected_collision_or_offroad_rate_degradation_percentage_points"]}
    quality_passed = all(all(checks.values()) for checks in gates.values())
    summary = {"complete": True, "benchmark": "NAVSIM v1 official navval development subset",
        "scene_count": len(records), "recording_count": len(set(groups)), "full_navtest": False,
        "independent_final_test": False, "checkpoint_updates": configuration["expected_checkpoint_updates"],
        "checkpoint_sha256": registration["checkpoint_sha256"], "groups": group_results,
        "gradient_direction_tests": gradient_results, "quality_gates": gates,
        "quality_gate_passed": quality_passed, "conditional_sampler_pilot_admitted": quality_passed,
        "gradient_nonzero_scenes": sum(row["gradient_nonzero"] for row in inference["scene_diagnostics"]),
        "mean_BM_roundtrip_xy_ADE_m": float(np.mean([row["roundtrip_xy_ADE_m"] for row in inference["scene_diagnostics"]])),
        "same_path_candidate_set_score_maximum_delta": max(row["same_path_candidate_set_score_maximum_delta"]
            for row in inference["scene_diagnostics"]), "cost": {"cache_creation_cpu_seconds": inference["cache_creation_cpu_seconds"],
            "scorer_diagnostics_cpu_seconds": inference["scorer_diagnostics_cpu_seconds"],
            "official_evaluation_cpu_wall_seconds": evaluation["cpu_wall_seconds"], "gpu_seconds": 0, "optimizer_steps": 0},
        "validation": {"original_vs_cached_score_maximum_error": max(row["native_cached_score_maximum_error"]
            for row in inference["scene_diagnostics"]), "official_pair_evaluator_maximum_error": evaluation["maximum_pair_evaluator_error"],
            "cached_encoder_calls": inference["encoder_calls_for_cached_diagnostics"],
            "frozen_weights_unchanged": inference["frozen_weights_unchanged"], "protected_training": protected_training_audit(configuration)},
        "historical_correction": "Previous 192-scene regret 25.0471 used pass3.pt/1920 updates, not this pass5.pt/3200-update checkpoint.",
        "next_action": "Proceed to pre-registered small pilot" if quality_passed else
            "Hold sampler training; inspect scorer ranking and off-support calibration before optimizing its reward"}
    write_json(destination / "results.json", summary)
    write_json(destination / "inference_diagnostics.json", inference)
    write_json(destination / "official_evaluation_validation.json", evaluation)
    with (destination / "scene_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scene_rows[0]))
        writer.writeheader()
        writer.writerows(scene_rows)
    np.savez_compressed(destination / "candidate_scores.npz", tokens=arrays["tokens"], official_metrics=metrics,
        native_rankings=arrays["native_rankings"], nearby_rankings=arrays["nearby_rankings"],
        secondary_rankings=arrays["secondary_rankings"], gradient_rankings=arrays["gradient_rankings"])
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    names = list(group_results)
    positions = np.arange(len(names))
    axes[0].bar(positions-0.18, [group_results[name]["selected_pdms"] for name in names], width=0.36, label="Scorer selected")
    axes[0].bar(positions+0.18, [group_results[name]["oracle_best_pdms"] for name in names], width=0.36, label="Official best in pool")
    axes[0].set_xticks(positions, names)
    axes[0].set_ylabel("Official PDMS (0-100)")
    axes[0].legend(fontsize=8)
    axes[1].bar(names, [group_results[name]["selection_regret_pdms_points"] for name in names])
    axes[1].axhline(5, color="red", linestyle="--", label="Pre-registered limit: 5")
    axes[1].set_ylabel("Selection regret (PDMS points)")
    axes[1].legend(fontsize=8)
    gradient_radii = [row["reference_std_radius"] for row in gradient_results]
    plus_means = np.array([row["actual_pdms_plus_delta_mean"] for row in gradient_results])
    confidence_bounds = np.array([row["actual_pdms_plus_delta_recording_bootstrap_95ci"] for row in gradient_results])
    axes[2].errorbar(gradient_radii, plus_means, yerr=np.stack((plus_means-confidence_bounds[:, 0],
        confidence_bounds[:, 1]-plus_means)), fmt="o-", capsize=4, label="Ascent mean + recording95%CI")
    axes[2].plot(gradient_radii, [row["actual_pdms_plus_delta_median"] for row in gradient_results],
        "s--", label="Ascent median")
    axes[2].plot([row["reference_std_radius"] for row in gradient_results],
        [row["actual_pdms_minus_delta_mean"] for row in gradient_results], "o-", label="Opposite direction")
    axes[2].axhline(0, color="gray", linewidth=0.8)
    axes[2].set_xlabel("Control displacement radius in reference-std units")
    axes[2].set_ylabel("Official PDMS change from BM center")
    axes[2].legend(fontsize=8)
    figure.suptitle("Frozen DrivoR scorer calibration | 64 development scenes | no training")
    figure.tight_layout()
    figure.savefig(destination / "scorer_calibration_summary.png", dpi=160)
    plt.close(figure)
    lines = ["# Frozen DrivoR scorer 검증", "", "원래 연구의 하위 질문: 시각 조건부 탐색을 학습하기 전에, 고정 scorer가 유용한 제어 후보를 판별하는가?", "",
        "64개 기존 공식 navval 개발 장면. 좌21/직진22/우21; 점수 확인 전 seed로 선택했다. 독립 최종 test·전체 NAVTEST가 아니다.", "",
        f"**결론: {'통과' if quality_passed else '사전 등록한 scorer 품질 기준 미충족 — sampler 학습 보류'}.**", "",
        "| 후보 묶음 | 후보 수 | 선택 PDMS | 후보 내 최고 PDMS | 선택 손실 | 충돌·도로이탈 선택률 |", "|---|---:|---:|---:|---:|---:|"]
    for name, current in group_results.items():
        lines.append(f"| {name} | {current['candidate_count']} | {current['selected_pdms']:.4f} | {current['oracle_best_pdms']:.4f} | {current['selection_regret_pdms_points']:.4f} | {current['selected_collision_or_offroad_rate']*100:.2f}% |")
    lines.extend(["", "## 통제와 해석", "",
        "- Native64는 기존 generator의 64개 경로다. Nearby64는 원래 선택 경로 fallback 1개, 제어→BM roundtrip center 1개, TOAD 초기 std의0.25배로 생성한62개 후보다. 학습한 sampler가 아니다.",
        "- Nearby16은 동일 nearby pool의 앞16개 경로를 다른 candidate context로 재채점한다. 공통경로의 점수 변화는 scorer의 candidate self-attention에 따른다. M16의 실제 TOAD std1.0 분포를 검증한 것은 아니다.",
        "- Native/nearby 모두 원래 neural score로 선택한다. 공식 PDMS로 얻은 best는 사후 oracle 진단이며 inference 선택에 사용하지 않는다.",
        "- 품질 기준: 모든 native64/nearby64/nearby16의 평균 selection regret≤5점, native 대비 collision-or-offroad 선택률 악화≤1pp. 결과를 보고 기준을 완화하지 않았다.",
        "- 공식 simulator는 전체후보를 일괄 처리하고 progress를 reference+개별후보 기준으로 복원한다. selected/best/center/gradient 후보 일부를 수정 없는 공식 evaluator와 모든 하위 지표까지 대조했다.",
        f"- 공식 evaluator 최대차 {evaluation['maximum_pair_evaluator_error']:.3g}. Cached/native neural score 최대차 {summary['validation']['original_vs_cached_score_maximum_error']:.3g}.",
        "", "## Gradient 방향의 실제 효과", "",
        "Scorer의 BM center score만 미분하고, 나머지63개 후보는 고정했다. reference std 좌표에서 unit gradient 방향과 반대 방향으로 동일 크기 보정한다. 이 검사는 optimizer 학습이 아니다.", "",
        "| 보정 반경 | Neural score 증가율 | +gradient 실제 PDMS 변화 | 반대 방향 실제 PDMS 변화 | Neural↑·PDMS↓ 비율 |", "|---|---:|---:|---:|---:|"])
    for current in gradient_results:
        lines.append(f"| {current['reference_std_radius']} | {current['neural_score_plus_increase_fraction']*100:.1f}% | {current['actual_pdms_plus_delta_mean']:+.4f} | {current['actual_pdms_minus_delta_mean']:+.4f} | {current['neural_score_up_actual_pdms_down_fraction']*100:.1f}% |")
    lines.extend(["", "**평균 상승은 소수 장면의 큰 개선에 의존한다.**",
        "세 보정 반경 모두 평균 변화의 recording-bootstrap95%구간이0을포함한다. 71.9%의하락은1e-6점 이상 변화 기준이며 대부분 작은 변화다. 모든 반경에서1점초과 하락은0장면이었다.", "",
        "| 보정 반경 | 평균변화95%구간 | 중앙값 변화 | 0.1점초과 하락 | 1점초과 개선 |", "|---|---|---:|---:|---:|"])
    for current in gradient_results:
        interval = current["actual_pdms_plus_delta_recording_bootstrap_95ci"]
        lines.append(f"| {current['reference_std_radius']} | [{interval[0]:+.4f}, {interval[1]:+.4f}] | {current['actual_pdms_plus_delta_median']:+.4f} | {current['actual_pdms_plus_drop_larger_than_point1_scene_count']}/64 | {current['actual_pdms_plus_gain_larger_than_1_scene_count']}/64 |")
    lines.extend(["", "Native 후보의 neural-vs-official Spearman 장면평균은 "
        f"{group_results['native64']['spearman_scene_mean']:+.4f}, nearby64는 {group_results['nearby64']['spearman_scene_mean']:+.4f}이다. "
        "Near pool에서는53개의비상수 장면만 포함한다. 좁은후보군에서progress의작은차이가상관에영향을줄수있으므로 음수상관을큰안전악화의증거로바꾸지않는다.",
        "", "## 선택 손실의 하위 지표", "",
        "| 후보 묶음 | 선택/최고 progress | 선택/최고 DAC | 선택/최고 TTC | 선택/최고 DDC |", "|---|---|---|---|---|"])
    for name, current in group_results.items():
        selected, best = current["selected_mean_submetrics"], current["oracle_best_mean_submetrics"]
        lines.append(f"| {name} | {selected['ego_progress']:.4f}/{best['ego_progress']:.4f} | {selected['drivable_area_compliance']:.4f}/{best['drivable_area_compliance']:.4f} | {selected['time_to_collision_within_bound']:.4f}/{best['time_to_collision_within_bound']:.4f} | {selected['driving_direction_compliance']:.4f}/{best['driving_direction_compliance']:.4f} |")
    lines.extend(["", "NAVSIMv1 공식 기본점수와 이neural scorer의 DDC 가중치는0이다. 따라서oracle최고PDMS를 모든주행조건을만족하는안전경로라고해석하지않는다. Native oracle은DDC평균0.8281로선택경로0.9531보다낮다. 원래metric을변경하지않고DDC도별도보고한다."])
    lines.extend(["", f"BM roundtrip 평균 ADE {summary['mean_BM_roundtrip_xy_ADE_m']:.4f}m. Gradient 성능은 원래 base가 아니라 BM center에 대한 대응 변화다.",
        "현재·이전 ego pose와 실제 timestamp 차이로 yaw rate를 복원했다. 미래 관측은 feature/comfort boundary에 입력하지 않았다.",
        "", "## 비용·보존", "",
        f"Cache 생성 CPU {inference['cache_creation_cpu_seconds']:.1f}s, cached scorer 진단 {inference['scorer_diagnostics_cpu_seconds']:.1f}s, 공식 평가 {evaluation['cpu_wall_seconds']:.1f}s /2worker. GPU0초·optimizer0step.",
        "Encoder는 cache 생성 때만 실행했고 cached gradient 단계 호출0회, 전체 frozen weight digest 불변이다. 기존LPWM 학습 source/config SHA를 전후검사했다.",
        "", "## 이전 검토의 체크포인트 정정", "",
        "이전192장면 selected72.76/oracle97.81/regret25.05는 pass3.pt(1920update) 결과다. 이전 타당성 검토에서 이를 pass5 scorer와 동일 checkpoint로 연결한 해석은 정정한다. 이번은 pass5.pt(3200update)의 별도 실제 측정이며 과거 로그·수치는 변경하지 않았다.",
        "", "## 다음 작업", "", summary["next_action"],
        "이번 검증은 scorer 품질과 국소 gradient를 검사하며 visual 정보의 추가 가치·sampler 성공·논문 novelty를 입증하지 않는다. Bootstrap은 recording 변동이며 학습 seed 변동이 아니다.",
        "", "![검증 요약](scorer_calibration_summary.png)", "", "## 재실행", "", "```bash",
        "cd /rhome/junseong/PlanningAwareFuturePrediction",
        "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src runtime/environments/kjs-lpwm-drivor-joint/bin/python -u scripts/calibrate_frozen_drivor_scorer.py --phase run --configuration configs/residual_control_sampler/scorer_calibration_20261011.json",
        "```", "", "완료 marker가 있는 phase는 재실행하지 않는다. 새로운 조건은 새 config/output 폴더로 등록한다."])
    (destination / "report.md").write_text("\n".join(lines)+"\n")
    write_json(output / "calibration_complete.json", summary)
    print(json.dumps({"phase": "report", "quality_gate_passed": quality_passed,
        "groups": group_results, "gradient_direction_tests": gradient_results}), flush=True)


def run(configuration, configuration_path):
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model_python = PROJECT_ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"
    official_environment = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation"
    stages = (("prepare", model_python, "registration.json"), ("infer", model_python, "inference_complete.json"),
              ("score", official_environment / "bin/python", "official_evaluation_complete.json"),
              ("report", model_python, "calibration_complete.json"))
    for phase, interpreter, completion_marker in stages:
        if (output / completion_marker).exists():
            continue
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(PROJECT_ROOT / "src")}
        if phase == "score":
            environment["LD_LIBRARY_PATH"] = str(official_environment / "lib")
            environment["OMP_NUM_THREADS"] = "1"
        command = [str(interpreter), "-u", str(Path(__file__).resolve()), "--phase", phase,
                   "--configuration", str(configuration_path)]
        with (output / (phase+".log")).open("a") as stream:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            write_json(output / "queue_state.json", {"phase": phase, "status": "running",
                "pid": os.getpid(), "child_pid": process.pid, "started_unix": started})
            code = process.wait()
        if code or not (output / completion_marker).exists():
            write_json(output / "queue_state.json", {"phase": phase, "status": "failed",
                "return_code": code, "log_file": phase+".log"})
            raise RuntimeError("Calibration phase failed: " + phase)
    summary = json.loads((output / "calibration_complete.json").read_text())
    write_json(output / "queue_state.json", {"status": "complete", "wall_seconds": time.time()-started,
        "quality_gate_passed": summary["quality_gate_passed"], "sampler_pilot_admitted": summary["conditional_sampler_pilot_admitted"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "infer", "score", "report", "run"), required=True)
    arguments = parser.parse_args()
    configuration_path = arguments.configuration.resolve()
    configuration = json.loads(configuration_path.read_text())
    if arguments.phase == "run":
        run(configuration, configuration_path)
    else:
        globals()[arguments.phase](configuration)
