#!/usr/bin/env python
"""Author-checkpoint scorer inspection; frozen weights and no optimizer.

Reuses the existing paired NAVSIM evaluator in a separate environment. Original
TOAD source is imported from an isolated nav1 checkout and never edited.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace, MethodType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import (
    METRIC_NAMES, file_sha256, recording_bootstrap, score as evaluate_candidates,
    write_json,
)


def protected_files_audit(configuration):
    registration = json.loads((PROJECT_ROOT / configuration["unchanged_running_training_registration"]).read_text())
    protected = {}
    for group in ("sources", "configurations", "immutable_inputs"):
        protected.update(registration[group])
    mismatches = [name for name, expected in protected.items()
                  if file_sha256(PROJECT_ROOT / name) != expected]
    if mismatches:
        raise RuntimeError("Registered LPWM files changed: " + str(mismatches))
    progress = json.loads((PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1/stage2/progress.json").read_text())
    return {"checked_files": len(protected), "mismatches": mismatches,
            "running_training_completed_updates": progress["completed_updates"]}


def prepare(configuration):
    from pyquaternion import Quaternion
    source_path = PROJECT_ROOT / configuration["subset_source"]
    source = json.loads(source_path.read_text())
    cached_ego = np.load(PROJECT_ROOT / configuration["subset_ego_cache"], mmap_mode="r")
    assert cached_ego.shape == (len(source["records"]), 11)
    selected = []
    raw_logs = {}
    for command, count in configuration["command_counts"].items():
        name = configuration["command_names"][command]
        eligible = [{**record, "source_panel_row": index} for index, record in enumerate(source["records"])
                    if int(cached_ego[index, 7:11].argmax()) == int(command)]
        eligible.sort(key=lambda record: hashlib.sha256(
            (configuration["subset_seed"] + record["token"]).encode()).hexdigest())
        assert len(eligible) >= count
        for record in eligible[:count]:
            if record["log_name"] not in raw_logs:
                with (PROJECT_ROOT / "dataset/navsim_logs/test" / (record["log_name"] + ".pkl")).open("rb") as stream:
                    frames = pickle.load(stream)
                raw_logs[record["log_name"]] = (frames, {frame["token"]: index for index, frame in enumerate(frames)})
            frames, token_to_index = raw_logs[record["log_name"]]
            current_index = token_to_index[record["token"]]
            current, previous = frames[current_index], frames[current_index - 1]
            current_command = np.asarray(current["driving_command"])
            assert int(current_command.argmax()) == int(command)
            np.testing.assert_array_equal(current_command, cached_ego[record["source_panel_row"], 7:11])
            np.testing.assert_allclose(current["ego_dynamic_state"], cached_ego[record["source_panel_row"], 3:7], atol=1e-6)
            camera_keys = ("CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0")
            camera_paths = []
            for camera_key in camera_keys:
                raw_path = Path(current["cams"][camera_key]["data_path"])
                resolved = raw_path if raw_path.is_absolute() else PROJECT_ROOT / "dataset/sensor_blobs/test" / raw_path
                assert resolved.is_file(), resolved
                camera_paths.append(str(resolved))
            current_heading = Quaternion(*current["ego2global_rotation"]).yaw_pitch_roll[0]
            previous_heading = Quaternion(*previous["ego2global_rotation"]).yaw_pitch_roll[0]
            interval = (current["timestamp"] - previous["timestamp"]) / 1e6
            assert 0.35 <= interval <= 0.65
            yaw_difference = np.arctan2(np.sin(current_heading - previous_heading),
                                        np.cos(current_heading - previous_heading))
            displacement = np.asarray(previous["ego2global_translation"][:2]) - np.asarray(current["ego2global_translation"][:2])
            rotation = np.array([[np.cos(current_heading), np.sin(current_heading)],
                                 [-np.sin(current_heading), np.cos(current_heading)]])
            previous_xy = rotation @ displacement
            previous_pose = np.r_[previous_xy, -yaw_difference]
            ego_history = [np.r_[previous_pose, previous["ego_dynamic_state"], previous["driving_command"]].tolist(),
                           np.r_[np.zeros(3), current["ego_dynamic_state"], current["driving_command"]].tolist()]
            assert np.asarray(ego_history).shape == (2, 11)
            assert Path(record["metric_cache_file"]).is_file()
            record.update(split="navtest", command_index=int(command), command_name=name,
                current_frame_index=current_index, current_camera_paths=camera_paths,
                camera_order=list(camera_keys), ego_history=ego_history,
                observed_timestamps=[previous["timestamp"], current["timestamp"]],
                boundary={"past_acceleration": float(current["ego_dynamic_state"][2]),
                          "past_yaw_rate": float(yaw_difference / interval),
                          "previous_timestamp_offset_seconds": -interval,
                          "latest_timestamp_offset_seconds": 0.0})
            selected.append(record)
    assert len(selected) == configuration["scene_count"]
    actual_sha = file_sha256(PROJECT_ROOT / configuration["checkpoint"])
    assert actual_sha == configuration["checkpoint_sha256"]
    repository = PROJECT_ROOT / configuration["official_repository"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    assert commit == configuration["official_repository_commit"]
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "records": selected, "checkpoint_sha256": actual_sha,
        "subset_source_sha256": file_sha256(source_path), "official_repository_commit": commit,
        "selection_before_predictions": True, "training_split_overlap": "NAVTEST is outside released trainval+Simscale training",
        "previous_development_exposure": True, "independent_final_test": False,
        "protected_training": protected_files_audit(configuration)}
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    if (output / "registration.json").exists():
        previous = json.loads((output / "registration.json").read_text())
        for name in ("configuration", "records", "checkpoint_sha256", "subset_source_sha256"):
            assert previous[name] == registration[name], name
    else:
        write_json(output / "registration.json", registration)
    destination = PROJECT_ROOT / configuration["results_directory"]
    write_json(destination / "configuration.json", configuration)
    write_json(destination / "subset_manifest.json", {key: value for key, value in registration.items() if key != "configuration"})
    print(json.dumps({"phase": "prepare", "scene_count": len(selected),
        "recording_count": len({record["recording_group"] for record in selected}),
        "checkpoint_sha256_verified": True}), flush=True)


def load_official_model(configuration, device):
    import torch
    from omegaconf import OmegaConf
    repository = PROJECT_ROOT / configuration["official_repository"]
    sys.path.insert(0, str(repository))
    from navsim.agents.drivoR.drivor_model import DrivoRModel
    torch.set_num_threads(configuration["cpu_threads"])
    torch.manual_seed(configuration["model_seed"])
    model_configuration = OmegaConf.load(repository / configuration["official_model_configuration"]).config
    model_configuration.image_backbone.model_weights = str(PROJECT_ROOT / configuration["backbone_initialization_weights"])
    model_configuration.use_cem = False  # Inspect scorer; no CEM optimization is executed.
    model_configuration.proposal_augment_random = configuration["official_random_bouquet_count"]
    model_configuration.long_trajectory_additional_poses = 2
    model = DrivoRModel(model_configuration)
    saved = torch.load(PROJECT_ROOT / configuration["checkpoint"], map_location="cpu", mmap=True, weights_only=False)
    assert saved["epoch"] == configuration["expected_checkpoint_epoch"]
    assert saved["global_step"] == configuration["expected_checkpoint_global_step"]
    checkpoint_state = {}
    for name, value in saved["state_dict"].items():
        prefix = "agent._drivor_model."
        assert name.startswith(prefix), name
        checkpoint_state[name[len(prefix):]] = value
    loading = str(model.load_state_dict(checkpoint_state, strict=True))
    model.eval().requires_grad_(False)
    report = {"strict_loading": loading, "checkpoint_tensor_count": len(checkpoint_state),
        "checkpoint_epoch": saved["epoch"], "checkpoint_global_step": saved["global_step"],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "configuration": OmegaConf.to_container(model_configuration, resolve=True),
        "source_path": str(repository / "navsim/agents/drivoR/drivor_model.py"),
        "source_sha256": file_sha256(repository / "navsim/agents/drivoR/drivor_model.py")}
    return model.to(device), report


def build_inputs(record, model_configuration, verify_preprocessing=False):
    import torch
    from PIL import Image
    # Execute the two unchanged official builder method bodies. Importing the
    # complete class also imports map/annotation dependencies belonging to the
    # separate NAVSIM evaluator environment; they are unused in RGB inference.
    builder_path = PROJECT_ROOT / "reference_repositories/TOADNav1OfficialCalibration/navsim/agents/drivoR/drivor_features.py"
    parsed_source = ast.parse(builder_path.read_text())
    builder_class = next(node for node in parsed_source.body if isinstance(node, ast.ClassDef) and node.name == "DrivoRFeatureBuilder")
    selected_methods = [node for node in builder_class.body if isinstance(node, ast.FunctionDef) and
                        node.name in ("compute_features", "_get_camera_feature")]
    assert len(selected_methods) == 2
    namespace = {"torch": torch, "np": np, "Image": Image, "Dict": dict, "AgentInput": object}
    exec(compile(ast.Module(body=selected_methods, type_ignores=[]), str(builder_path), "exec"), namespace)
    builder = SimpleNamespace(_config=model_configuration)
    builder._get_camera_feature = MethodType(namespace["_get_camera_feature"], builder)
    with (PROJECT_ROOT / "dataset/navsim_logs/test" / (record["log_name"] + ".pkl")).open("rb") as stream:
        current_frame = pickle.load(stream)[record["current_frame_index"]]
    assert current_frame["token"] == record["token"]
    camera_keys = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")
    cameras = {name: SimpleNamespace(image=None) for name in
               ("cam_f0", "cam_b0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2")}
    for camera_key, path in zip(camera_keys, record["current_camera_paths"]):
        with Image.open(path) as source:
            calibration = current_frame["cams"][camera_key.upper()]
            cameras[camera_key] = SimpleNamespace(image=np.asarray(source.convert("RGB")),
                intrinsics=calibration["cam_intrinsic"], sensor2lidar_rotation=calibration["sensor2lidar_rotation"],
                sensor2lidar_translation=calibration["sensor2lidar_translation"])
    ego_statuses = [SimpleNamespace(ego_pose=np.asarray(row[:3]), ego_velocity=np.asarray(row[3:5]),
        ego_acceleration=np.asarray(row[5:7]), driving_command=np.asarray(row[7:11])) for row in record["ego_history"]]
    observed_input = SimpleNamespace(cameras=[SimpleNamespace(**cameras)], ego_statuses=ego_statuses)
    official_features = namespace["compute_features"](builder, observed_input)
    image = official_features["image"][None].contiguous()
    ego = official_features["ego_status"][None]
    checks = {}
    assert image.shape == (1, 4, 3, 672, 1148)
    assert ego.shape == (1, 2, 11)
    if verify_preprocessing:
        mean = np.array([.485, .456, .406], dtype=np.float32)
        standard_deviation = np.array([.229, .224, .225], dtype=np.float32)
        independent_images = []
        for camera_key in camera_keys:
            resized = np.asarray(Image.fromarray(cameras[camera_key].image).resize((1148, 672)), dtype=np.float32)
            independent_images.append(torch.from_numpy((resized / 255 - mean) / standard_deviation).permute(2, 0, 1))
        difference = float((image[0] - torch.stack(independent_images)).abs().max())
        assert difference == 0
        assert np.array_equal(ego.numpy()[0], np.asarray(record["ego_history"], dtype=np.float32))
        checks = {"official_vs_independent_preprocessing_maximum_error": difference,
                  "ego_sample_matching_exact": True, "future_input_frames": 0}
    return {"image": image, "ego_status": ego}, checks


def gpu_memory_status(configuration):
    import torch
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    used_bytes = total_bytes - free_bytes
    if used_bytes >= configuration["maximum_total_card_bytes"]:
        raise RuntimeError("Whole GPU card memory limit reached")
    return {"card_used_bytes": used_bytes, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def infer(configuration, profile_only=False):
    import torch
    from diagnose_residual_control_sampler import module_digest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA inaccessible; rerun authorized inference outside the filesystem sandbox")
    torch.cuda.set_per_process_memory_fraction(configuration["maximum_process_bytes"] / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = PROJECT_ROOT / configuration["output_directory"]
    records = json.loads((output / "registration.json").read_text())["records"]
    previous_profile = output / "profile_complete.json"
    prior_gpu_seconds = 0.0 if profile_only or not previous_profile.exists() else json.loads(previous_profile.read_text())["gpu_phase_wall_seconds"]
    # Reserve two minutes for the preliminary failed process whose exact
    # duration was not captured; it failed before any image/scorer forward.
    preliminary_failure_reserve_seconds = 120.0 if (output / "profile_failed_before_inference.log").exists() else 0.0
    available_phase_seconds = configuration["maximum_new_gpu_phase_seconds"] - prior_gpu_seconds - preliminary_failure_reserve_seconds
    assert available_phase_seconds > 0
    write_json(output / ("profile_started.json" if profile_only else "inference_started.json"), {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_source_sha256": file_sha256(Path(__file__)),
        "available_phase_seconds": available_phase_seconds,
        "prior_profile_seconds": prior_gpu_seconds,
        "preliminary_failure_reserve_seconds": preliminary_failure_reserve_seconds})
    if profile_only:
        records = records[:1]
    model, loading_report = load_official_model(configuration, "cuda")
    initial_digest = module_digest(model)
    native_count = configuration["native_candidate_count"]
    nearby_count = configuration["nearby_candidate_count"]
    target_count = configuration["fixed_context_target_count"]
    trajectories, original_scores, proposer_scores, nearby_scores, secondary_scores = [], [], [], [], []
    gradient_scores, cached_features, cached_ego_tokens, all_context_scores, scene_diagnostics = [], [], [], [], []
    extraction_seconds = diagnostic_seconds = 0.0
    encoder_cache_calls = cached_encoder_calls = 0
    maximum_card = 0
    for scene_index, record in enumerate(records):
        if time.perf_counter() - started > available_phase_seconds:
            raise RuntimeError("GPU diagnostic time budget exceeded")
        gpu_memory_status(configuration)
        features, preprocessing_checks = build_inputs(record, model._config, scene_index < 3)
        features = {name: value.cuda() for name, value in features.items()}
        captured = []
        def capture_features(_module, _inputs, value):
            captured.append(value.detach())
        hook = model.image_backbone.register_forward_hook(capture_features)
        torch.cuda.synchronize()
        extraction_started = time.perf_counter()
        with torch.no_grad():
            prediction = model(features)
            ego_token = model.hist_encoding(features["ego_status"][:, -1])[:, None]
        torch.cuda.synchronize()
        scene_extraction_seconds = time.perf_counter() - extraction_started
        extraction_seconds += scene_extraction_seconds
        hook.remove()
        assert len(captured) == 1
        encoder_cache_calls += 1
        scene_features = captured[0]
        assert scene_features.shape == (1, 64, 256)
        proposals = prediction["proposals"].detach()
        assert proposals.shape == (1, native_count, 8, 3)
        assert prediction["proposal_list"][-1].shape[1] == configuration["learned_proposal_count"]
        base = prediction["trajectory"].detach()
        def reject_backbone(_module, _inputs):
            nonlocal cached_encoder_calls
            cached_encoder_calls += 1
            raise RuntimeError("Backbone executed during cached scorer diagnostic")
        guard = model.image_backbone.register_forward_pre_hook(reject_backbone)
        torch.cuda.synchronize()
        diagnostic_started = time.perf_counter()
        with torch.no_grad():
            native_ranking = model._score_trajectories(proposals, scene_features, ego_token)
            replay_error = float((native_ranking - prediction["pdm_score"]).abs().max())
            assert replay_error <= 2e-6, replay_error
            proposer_ranking = model._score_trajectories(proposals[:, :64], scene_features, ego_token)
            initial_speed = features["ego_status"][:, -1, 3]
            seed_controls = model._proposals_to_controls(proposals, initial_speed, .5)
            base_controls = model._proposals_to_controls(base[:, None], initial_speed, .5)[:, 0]
            floor = base_controls.new_tensor([configuration["acceleration_standard_deviation_floor"],
                                             configuration["yaw_rate_standard_deviation_floor"]])
            reference_std = torch.maximum(seed_controls.std(dim=1, unbiased=False) * configuration["initial_standard_deviation_scale"], floor)
            generator = torch.Generator(device="cuda").manual_seed(configuration["noise_seed"] + scene_index)
            noise = torch.randn(1, nearby_count - 2, 8, 2, device="cuda", generator=generator)
            raw_controls = base_controls[:, None] + reference_std[:, None] * noise
            controls = model._clamp_controls(torch.cat((base_controls[:, None], raw_controls), dim=1))
            rollouts, speeds = model._bicycle_rollout(controls, initial_speed, .5)
            nearby = torch.cat((base[:, None], rollouts), dim=1).detach()
            nearby_ranking = model._score_trajectories(nearby, scene_features, ego_token)
            secondary_ranking = model._score_trajectories(nearby[:, :target_count], scene_features, ego_token)
            fixed_targets = nearby[:, :target_count]
            native_context = torch.cat((fixed_targets, proposals[:, :nearby_count-target_count]), dim=1)
            alternate_controls = model._clamp_controls(base_controls[:, None] + reference_std[:, None] *
                torch.randn(1, nearby_count-target_count, 8, 2, device="cuda", generator=generator))
            alternate_paths, _ = model._bicycle_rollout(alternate_controls, initial_speed, .5)
            alternate_context = torch.cat((fixed_targets, alternate_paths), dim=1)
            context_rankings = torch.stack((nearby_ranking[:, :target_count],
                model._score_trajectories(native_context, scene_features, ego_token)[:, :target_count],
                model._score_trajectories(alternate_context, scene_features, ego_token)[:, :target_count]), dim=1)
            past_acceleration = base_controls.new_tensor([record["boundary"]["past_acceleration"]])
            past_yaw_rate = base_controls.new_tensor([record["boundary"]["past_yaw_rate"]])
            comfort = model._direct_comfort_cost(controls, speeds, past_acceleration, past_yaw_rate, .5)
        # Original scorer accepts arbitrary continuous paths with input gradients.
        normalized_correction = torch.zeros_like(base_controls, requires_grad=True)
        variable_controls = model._clamp_controls(base_controls + reference_std * normalized_correction)
        variable_path, _ = model._bicycle_rollout(variable_controls[:, None], initial_speed, .5)
        variable_context = torch.cat((nearby[:, :1], variable_path, nearby[:, 2:]), dim=1)
        neural_objective = model._score_trajectories(variable_context, scene_features, ego_token)[0, 1]
        gradient, = torch.autograd.grad(neural_objective, normalized_correction)
        assert torch.isfinite(gradient).all()
        gradient_norm = float(gradient.norm())
        direction = gradient / gradient.norm().clamp_min(1e-12)
        directional_paths, directional_rankings, directional_rows = [], [], []
        finite_differences = []
        with torch.no_grad():
            for radius in configuration["gradient_reference_std_step_sizes"]:
                for sign in (1, -1):
                    shifted_controls = model._clamp_controls(base_controls + sign * radius * reference_std * direction)
                    shifted_path, _ = model._bicycle_rollout(shifted_controls[:, None], initial_speed, .5)
                    shifted_context = nearby.clone()
                    shifted_context[:, 1] = shifted_path[:, 0]
                    shifted_score = model._score_trajectories(shifted_context, scene_features, ego_token)[0, 1]
                    directional_paths.append(shifted_path[0, 0].cpu().numpy())
                    directional_rankings.append(float(shifted_score))
                    directional_rows.append({"reference_std_radius": radius, "gradient_sign": sign,
                        "neural_score_delta": float(shifted_score - nearby_ranking[0, 1])})
            if scene_index < configuration["finite_difference_scene_count"]:
                for direction_index in range(2):
                    checked_direction = direction if direction_index == 0 else torch.randn(base_controls.shape, device="cuda", generator=generator)
                    checked_direction = checked_direction / checked_direction.norm().clamp_min(1e-12)
                    epsilon = configuration["finite_difference_epsilon"]
                    scores = []
                    for sign in (1, -1):
                        shifted = model._clamp_controls(base_controls + sign * epsilon * reference_std * checked_direction)
                        path, _ = model._bicycle_rollout(shifted[:, None], initial_speed, .5)
                        context = nearby.clone(); context[:, 1] = path[:, 0]
                        scores.append(float(model._score_trajectories(context, scene_features, ego_token)[0, 1]))
                    numerical = (scores[0] - scores[1]) / (2 * epsilon)
                    analytic = float((gradient * checked_direction).sum())
                    finite_differences.append({"analytic": analytic, "finite_difference": numerical,
                        "absolute_error": abs(analytic - numerical),
                        "relative_error": abs(analytic - numerical) / max(abs(analytic), abs(numerical), 1e-8)})
        guard.remove()
        torch.cuda.synchronize()
        scene_diagnostic_seconds = time.perf_counter() - diagnostic_started
        diagnostic_seconds += scene_diagnostic_seconds
        candidate_paths = np.concatenate((proposals[0].cpu().numpy(), nearby[0].cpu().numpy(), np.stack(directional_paths)), axis=0)
        assert np.isfinite(candidate_paths).all()
        assert all(torch.isfinite(value).all() for value in (native_ranking, nearby_ranking, context_rankings))
        memory = gpu_memory_status(configuration)
        maximum_card = max(maximum_card, memory["card_used_bytes"])
        scene_diagnostics.append({"token": record["token"], "preprocessing_checks": preprocessing_checks,
            "native_cached_score_maximum_error": replay_error, "gradient_norm": gradient_norm,
            "finite_difference_directions": finite_differences, "gradient_variants": directional_rows,
            "roundtrip_xy_ADE_m": float((rollouts[:, 0, :, :2] - base[:, :, :2]).norm(dim=-1).mean()),
            "nearby_clipped_control_fraction": float((controls[:, 1:] != raw_controls).float().mean()),
            "nearby_comfort_violation_fraction": float((comfort > 1e-9).float().mean()),
            "nearby64_vs16_same_path_score_maximum_delta": float((nearby_ranking[:, :target_count] - secondary_ranking).abs().max()),
            "fixed64_context_same_path_score_maximum_delta": float((context_rankings - context_rankings[:, :1]).abs().max()),
            "cache_extraction_seconds": scene_extraction_seconds, "cached_diagnostic_seconds": scene_diagnostic_seconds,
            "memory": memory})
        trajectories.append(candidate_paths)
        original_scores.append(native_ranking[0].cpu().numpy())
        proposer_scores.append(proposer_ranking[0].cpu().numpy())
        nearby_scores.append(nearby_ranking[0].cpu().numpy())
        secondary_scores.append(secondary_ranking[0].cpu().numpy())
        gradient_scores.append(np.asarray(directional_rankings))
        all_context_scores.append(context_rankings[0].cpu().numpy())
        cached_features.append(scene_features[0].cpu().numpy())
        cached_ego_tokens.append(ego_token[0].cpu().numpy())
        del prediction, captured, features, neural_objective, variable_context, variable_path
        if (scene_index + 1) % 8 == 0 or profile_only:
            write_json(output / "inference_progress.json", {"complete_scenes": scene_index + 1,
                "total_scenes": len(records), "elapsed_seconds": time.perf_counter() - started, "memory": memory})
            print(json.dumps({"phase": "profile" if profile_only else "infer", "complete_scenes": scene_index + 1,
                "elapsed_seconds": time.perf_counter() - started, "memory": memory}), flush=True)
    assert cached_encoder_calls == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    assert initial_digest == module_digest(model)
    summary = {"complete": True, "scenes": len(records), "loading": loading_report,
        "feature_shape": [len(records), 64, 256], "feature_dtype": "float32",
        "encoder_calls_for_cache_creation": encoder_cache_calls, "encoder_calls_for_cached_diagnostics": cached_encoder_calls,
        "frozen_weights_unchanged": True, "frozen_parameter_gradients_absent": True,
        "cache_creation_seconds": extraction_seconds, "cached_scorer_diagnostics_seconds": diagnostic_seconds,
        "gpu_phase_wall_seconds": time.perf_counter() - started, "maximum_card_used_bytes": maximum_card,
        "memory": gpu_memory_status(configuration), "physical_gpu": configuration["physical_gpu"],
        "optimizer_steps": 0, "protected_training": protected_files_audit(configuration),
        "diagnostic_source_sha256": file_sha256(Path(__file__)),
        "reused_evaluator_source_sha256": file_sha256(PROJECT_ROOT / "scripts/calibrate_frozen_drivor_scorer.py"),
        "official_feature_builder_source_sha256": file_sha256(PROJECT_ROOT / configuration["official_repository"] / "navsim/agents/drivoR/drivor_features.py"),
        "observed_inputs": {"camera_order": ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0"],
            "image_shape_per_scene": [4, 3, 672, 1148], "future_image_count": 0,
            "scorer_feature_location": "image_backbone output with learned register projection",
            "spatial_interpretation": "Register tokens do not directly correspond to image pixel locations",
            "all_visual_tokens_valid": True, "ego_shape_per_scene": [2, 11],
            "feature_builder_method_execution": "Unmodified compute_features and _get_camera_feature AST bodies; map imports omitted"},
        "scene_diagnostics": scene_diagnostics,
        "environment": {"python": sys.version, "torch": torch.__version__, "timm": __import__("timm").__version__}}
    if profile_only:
        write_json(output / "profile_complete.json", summary)
    else:
        np.savez_compressed(output / "scorer_cache_and_candidates.npz", tokens=np.array([record["token"] for record in records]),
            trajectories=np.stack(trajectories), native_rankings=np.stack(original_scores),
            proposer_rankings=np.stack(proposer_scores), nearby_rankings=np.stack(nearby_scores),
            secondary_rankings=np.stack(secondary_scores), gradient_rankings=np.stack(gradient_scores),
            context_rankings=np.stack(all_context_scores), scene_features=np.stack(cached_features), ego_tokens=np.stack(cached_ego_tokens))
        write_json(output / "inference_complete.json", summary)


def report(configuration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    destination.mkdir(parents=True, exist_ok=True)
    records = json.loads((output / "registration.json").read_text())["records"]
    inference = json.loads((output / "inference_complete.json").read_text())
    official = json.loads((output / "official_evaluation_complete.json").read_text())
    arrays = np.load(output / "scorer_cache_and_candidates.npz")
    metrics = np.load(output / "official_metrics.npy")
    assert arrays["tokens"].tolist() == [record["token"] for record in records]
    scene_indices = np.arange(len(records))
    recording_groups = [record["recording_group"] for record in records]
    native_count = configuration["native_candidate_count"]
    candidate_groups = {"learned64_alone": (0, arrays["proposer_rankings"]),
        "official128": (0, arrays["native_rankings"]), "nearby64": (native_count, arrays["nearby_rankings"]),
        "nearby16": (native_count, arrays["secondary_rankings"])}
    result_groups, rows = {}, []
    baseline_indices = arrays["native_rankings"].argmax(1)
    baseline_metrics = metrics[scene_indices, baseline_indices]
    baseline_unsafe = (baseline_metrics[:, 0] < 1) | (baseline_metrics[:, 1] < 1)
    for name, (offset, ranking) in candidate_groups.items():
        current_metrics = metrics[:, offset:offset + ranking.shape[1]]
        pdms = current_metrics[:, :, -1] * 100
        selected_indices, best_indices = ranking.argmax(1), pdms.argmax(1)
        selected, best = current_metrics[scene_indices, selected_indices], current_metrics[scene_indices, best_indices]
        regret = (best[:, -1] - selected[:, -1]) * 100
        unsafe = (selected[:, 0] < 1) | (selected[:, 1] < 1)
        correlations = [float(spearmanr(ranking[index], pdms[index]).statistic) for index in scene_indices
                        if np.ptp(pdms[index]) > 1e-10 and np.ptp(ranking[index]) > 1e-10]
        result_groups[name] = {"scene_count": len(records), "candidate_count": ranking.shape[1],
            "selected_pdms": float(selected[:, -1].mean() * 100), "oracle_best_pdms": float(best[:, -1].mean() * 100),
            "selection_regret_pdms_points": float(regret.mean()),
            "selection_regret_recording_bootstrap_95ci": recording_bootstrap(regret, recording_groups,
                configuration["noise_seed"], configuration["bootstrap_repetitions"]),
            "spearman_scene_mean": float(np.mean(correlations)) if correlations else None,
            "spearman_nonconstant_scene_count": len(correlations), "selected_collision_or_offroad_rate": float(unsafe.mean()),
            "unsafe_rate_change_from_official128_percentage_points": float((unsafe.astype(float) - baseline_unsafe).mean() * 100),
            "selected_mean_submetrics": {metric: float(selected[:, index].mean()) for index, metric in enumerate(METRIC_NAMES[:-1])},
            "oracle_mean_submetrics": {metric: float(best[:, index].mean()) for index, metric in enumerate(METRIC_NAMES[:-1])},
            "selected_augmented_bouquet_rate": float((selected_indices >= 64).mean()) if name == "official128" else None,
            "by_command": {configuration["command_names"][command]: {
                "scenes": sum(record["command_index"] == int(command) for record in records),
                "selected_pdms": float(selected[[record["command_index"] == int(command) for record in records], -1].mean() * 100),
                "oracle_best_pdms": float(best[[record["command_index"] == int(command) for record in records], -1].mean() * 100)}
                for command in configuration["command_counts"]}}
        for index, record in enumerate(records):
            rows.append({"condition": name, "token": record["token"], "recording_group": record["recording_group"],
                "command_name": record["command_name"], "selected_index": int(selected_indices[index]),
                "selected_pdms": float(selected[index, -1] * 100), "oracle_pdms": float(best[index, -1] * 100),
                "regret": float(regret[index])})
    directional_results = []
    center = metrics[:, native_count + 1, -1] * 100
    for radius_index, radius in enumerate(configuration["gradient_reference_std_step_sizes"]):
        changes = (metrics[:, native_count + configuration["nearby_candidate_count"] + 2 * radius_index, -1] * 100) - center
        score_changes = np.asarray([row["gradient_variants"][2 * radius_index]["neural_score_delta"] for row in inference["scene_diagnostics"]])
        directional_results.append({"reference_std_radius": radius, "neural_score_increase_rate": float((score_changes > 0).mean()),
            "actual_pdms_mean_change": float(changes.mean()), "actual_pdms_median_change": float(np.median(changes)),
            "actual_pdms_change_recording_bootstrap_95ci": recording_bootstrap(changes, recording_groups,
                configuration["noise_seed"], configuration["bootstrap_repetitions"]),
            "actual_decrease_above_1_point_count": int((changes < -configuration["large_actual_pdms_decrease_points"]).sum()),
            "actual_improvement_above_1_point_count": int((changes > 1).sum())})
    target_count = configuration["fixed_context_target_count"]
    context_rankings = arrays["context_rankings"]
    centered_rankings = context_rankings - context_rankings.mean(axis=2, keepdims=True)
    context_instability = centered_rankings.std(axis=1).mean(axis=1)
    chosen_target = context_rankings.argmax(axis=2)
    target_official = metrics[:, native_count:native_count + target_count, -1] * 100
    context_selected_pdms = target_official[scene_indices[:, None], chosen_target]
    target_regret = target_official.max(1) - context_selected_pdms[:, 0]
    association = None
    if np.ptp(context_instability) > 1e-12 and np.ptp(target_regret) > 1e-12:
        association = float(spearmanr(context_instability, target_regret).statistic)
    context_results = {"candidate_count_in_every_context": 64, "shared_target_count": target_count,
        "context_count": 3, "same_target_top1_change_scene_count": int((chosen_target != chosen_target[:, :1]).any(axis=1).sum()),
        "relative_score_instability_mean": float(context_instability.mean()),
        "instability_vs_actual_selection_regret_spearman": association,
        "mean_actual_selected_pdms_by_context": context_selected_pdms.mean(0).tolist(),
        "interpretation": "Functional context dependence only; correlation is exploratory and does not calibrate safety or prove causal scene uncertainty."}
    relevant_groups = ("official128", "nearby64", "nearby16")
    regret_gate = all(result_groups[name]["selection_regret_pdms_points"] <= configuration["maximum_selection_regret_pdms_points"] for name in relevant_groups)
    safety_gate = all(result_groups[name]["unsafe_rate_change_from_official128_percentage_points"] <= configuration["maximum_collision_or_offroad_degradation_percentage_points"] for name in relevant_groups)
    summary = {"complete": True, "benchmark": "NAVSIM v1 NAVTEST64 development diagnostic", "full_navtest": False,
        "independent_final_test": False, "scene_count": len(records), "recording_count": len(set(recording_groups)),
        "checkpoint_sha256": configuration["checkpoint_sha256"], "groups": result_groups,
        "gradient_direction_tests": directional_results, "fixed_size_context_diagnostic": context_results,
        "quality_gate_passed": regret_gate and safety_gate, "regret_gate_passed": regret_gate, "safety_gate_passed": safety_gate,
        "no_followup_training_automatically_started": True, "checks": {
            "strict_loading_passed": inference["loading"]["strict_loading"] == "<All keys matched successfully>",
            "encoder_calls_during_cached_diagnostic": inference["encoder_calls_for_cached_diagnostics"],
            "frozen_weights_unchanged": inference["frozen_weights_unchanged"],
            "frozen_parameter_gradients_absent": inference["frozen_parameter_gradients_absent"],
            "finite_nonzero_gradient_scenes": sum(row["gradient_norm"] > 1e-12 for row in inference["scene_diagnostics"]),
            "finite_difference_checks": [check for row in inference["scene_diagnostics"] for check in row["finite_difference_directions"]],
            "official_pair_evaluator_maximum_error": official["maximum_pair_evaluator_error"]},
        "roundtrip_xy_ADE_m": float(np.mean([row["roundtrip_xy_ADE_m"] for row in inference["scene_diagnostics"]])),
        "cost": {"inference_gpu_phase_wall_seconds": inference["gpu_phase_wall_seconds"],
            "cache_creation_seconds": inference["cache_creation_seconds"],
            "cached_scorer_diagnostics_seconds": inference["cached_scorer_diagnostics_seconds"],
            "maximum_card_used_bytes": inference["maximum_card_used_bytes"],
            "official_evaluation_seconds": official["cpu_wall_seconds"],
            "official_evaluated_candidates": official["official_evaluated_candidates"]},
        "protected_training": protected_files_audit(configuration)}
    write_json(destination / "results.json", summary)
    write_json(destination / "inference_diagnostics.json", inference)
    write_json(destination / "official_evaluation_validation.json", official)
    with (destination / "comparison.csv").open("w") as stream:
        columns = ("condition", "candidate_count", "selected_pdms", "oracle_best_pdms", "selection_regret_pdms_points", "selected_collision_or_offroad_rate", "spearman_scene_mean")
        writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
        for name, result in result_groups.items():
            writer.writerow({"condition": name, **{column: result[column] for column in columns[1:]}})
    with (destination / "scene_results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(result_groups)
    positions = np.arange(len(names))
    axes[0].bar(positions - .18, [result_groups[name]["selected_pdms"] for name in names], width=.36, label="Scorer selected")
    axes[0].bar(positions + .18, [result_groups[name]["oracle_best_pdms"] for name in names], width=.36, label="Official oracle")
    axes[0].set_xticks(positions, names, rotation=15); axes[0].set_ylim(0, 105); axes[0].set_ylabel("PDMS"); axes[0].legend()
    axes[1].scatter(arrays["nearby_rankings"].flatten(), metrics[:, native_count:native_count + configuration["nearby_candidate_count"], -1].flatten() * 100, s=3, alpha=.25)
    axes[1].set_xlabel("Frozen scorer log score"); axes[1].set_ylabel("Official PDMS"); axes[1].set_title("Off-proposal candidates")
    axes[2].scatter(context_instability, target_regret, s=14); axes[2].set_xlabel("Relative score context instability")
    axes[2].set_ylabel("Shared-target selection regret"); axes[2].set_title("Exploratory association")
    figure.suptitle("Official TOAD NAVSIM v1 checkpoint: NAVTEST64 diagnostic")
    figure.tight_layout(); figure.savefig(destination / "scorer_calibration_summary.png", dpi=160); plt.close(figure)
    lines = ["# 공식 TOAD NAVSIM v1 scorer 검사", "", "**추가 학습0. 공개 checkpoint를 완전히 고정한 NAVTEST64 진단이다.**", "",
        "공개 모델은 trainval+Simscale로 학습했으므로 NAVVAL을 독립 평가로 사용하지 않았다. 이전 비교에서 노출된 NAVTEST1024 패널에서 명령별 좌21/직22/우21을 점수 확인 전에 선택했다. 전체 NAVTEST·독립 최종 평가가 아니다.", "",
        "## 후보별 실제 결과", "", "| 조건 | 후보수 | 선택 PDMS | 실제 후보최고 | 선택손실 | 충돌·도로이탈 선택률 |", "|---|---:|---:|---:|---:|---:|"]
    for name, result in result_groups.items():
        lines.append(f"| {name} | {result['candidate_count']} | {result['selected_pdms']:.4f} | {result['oracle_best_pdms']:.4f} | {result['selection_regret_pdms_points']:.4f} | {result['selected_collision_or_offroad_rate']*100:.2f}% |")
    lines += ["", "## 통제와 공식 코드 확인", "",
        "- 4카메라(front/back/left/right), 현재1프레임,1148×672/PIL resize/ImageNet normalization. Ego 이전·현재2개는 comfort 경계에 사용하며 planner는 현재 ego11만 사용한다. 미래 영상 입력0.",
        "- 실제 cache는 image_backbone의64×256FP32 register scene feature와256차원 ego token이다. 기존 작은 모델16×256cache와 교환하지 않았다.",
        "- 공식 nav1 README는 proposal_augment_random=64를 설정한다. 코드 주석은 training-only라고 쓰지만 forward에는 training 조건문이 없어 eval에서도64bouquet를 추가한다. 이를 보존한 공식설정은128후보다. Learned64-alone은추가진단이지공식128설정의재현이아니다.",
        "- Nearby64는원본fallback1+BMroundtripcenter1+공식초기std(.5×128proposalcontrolstd/floor)62sample이다. CEM전체반복을실행한성능이아니다. Nearby16은같은앞16후보를별도context에서재채점한다.",
        "- 공식 source를수정하지않았고467tensor strictload/weight SHA를검증했다. Native cachedscore를원forward와대조했다. 가중치gradient누적0·cachedbackbone호출0·weightdigest불변.",
        "- 공식 NAVSIMv1 PDMS의reference+개별candidate progress 정규화를 사용했고선택/최고/center/gradient일부를수정없는pair evaluator와대조했다. Oracle점수는오프라인진단에만사용한다.",
        "- 이checkpoint와과거pass5/3200은학습데이터·입력·학습량·후보수·검증장면이달라점수차를scorer학습단독효과로해석하지않는다.", "",
        "## Gradient와 판단 안정성", "", "| 제어 보정 반경 | neural 증가율 | 실제 PDMS 평균 변화 | 중앙값 | 1점초과 하락 장면 |", "|---|---:|---:|---:|---:|"]
    for result in directional_results:
        lines.append(f"| {result['reference_std_radius']} | {result['neural_score_increase_rate']*100:.1f}% | {result['actual_pdms_mean_change']:+.4f} | {result['actual_pdms_median_change']:+.4f} | {result['actual_decrease_above_1_point_count']} |")
    lines += ["", f"BMroundtrip 평균ADE **{summary['roundtrip_xy_ADE_m']:.4f}m**. Gradient변화는원본경로가아니라roundtripcenter대비다. 평균변화recording-bootstrap구간과finite difference는results.json에기록했다.", "",
        f"동일64개context3개에서동일16target의최고선택이바뀐장면: **{context_results['same_target_top1_change_scene_count']}/64**. 상대점수불안정성과실제target선택손실Spearman: **{association}**. 이연관은탐색적진단이며안전신뢰도보장이나보정기효용증명이아니다.", "",
        "## 사전 기준과 다음 단계", "", f"기준: official128/nearby64/nearby16각평균선택손실≤5점,official128대비collision-or-offroad선택률악화≤1pp. 품질gate: **{'통과' if summary['quality_gate_passed'] else '미충족'}**. 기준을결과에맞춰완화하지않았다.", "",
        "검사통과도TOAD성능개선의보장은아니다. 통과하면원본/BMroundtrip/공식TOAD를같은고정weights·공통noise·총scorer예산으로비교하는후속검토가가능하다. 실패하면scorer오류·off-proposal하위지표를먼저분석한다. 후속학습과추가성능queue는자동시작하지않았다.", "",
        "## 비용·산출물", "", f"GPU추론/진단phase {inference['gpu_phase_wall_seconds']:.2f}초(모델load·I/O포함),cache추출 {inference['cache_creation_seconds']:.2f}초,cached진단 {inference['cached_scorer_diagnostics_seconds']:.2f}초. 전체GPU카드관측최대 {inference['maximum_card_used_bytes']/1e9:.3f}GB<48GB. 공식CPU평가 {official['cpu_wall_seconds']:.2f}초/{official['official_evaluated_candidates']}후보/2worker.",
        "GPUkernel시간으로해석하지않으며별도profile비용은profile_complete.json에기록한다. 기존LPWM등록293파일hash보존. 결과JSON/CSV/PNG·subsetmanifest·strictload/source/env/cache경로를보존했다.", "",
        "## 재실행", "", "```bash", "cd /rhome/junseong/PlanningAwareFuturePrediction", "bash scripts/run_official_toad_navsim_v1_scorer_calibration.sh", "```", "",
        "완료phase는재실행하지않는다. 새로운조건은별도config/output에등록한다.", "",
        "[공식NAV1코드·설정](https://github.com/valeoai/TOAD/tree/nav1) · [공개가중치](https://github.com/valeoai/TOAD/releases/tag/Nav1) · [TOAD원문](https://arxiv.org/abs/2606.07170)"]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "calibration_complete.json", summary)
    print(json.dumps({"phase": "report", "quality_gate_passed": summary["quality_gate_passed"], "groups": result_groups}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "profile", "infer", "score", "report"), required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text())
    if arguments.phase == "profile":
        infer(configuration, profile_only=True)
    elif arguments.phase == "score":
        evaluate_candidates(configuration)
    else:
        globals()[arguments.phase](configuration)
