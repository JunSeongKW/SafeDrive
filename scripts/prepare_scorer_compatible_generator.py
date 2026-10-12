#!/usr/bin/env python
"""Register NAVTRAIN adaptation groups and cache public DrivoR generator inputs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from calibrate_official_toad_navsim_v1_scorer import PROJECT_ROOT, protected_files_audit, write_json, file_sha256, gpu_memory_status
from compare_public_drivor_toad_scorers import model_configuration_for, load_official_model, score_metrics, original_drivor_forward


def load_configuration(path):
    return json.loads(path.read_text())


def register(configuration):
    manifest_path = PROJECT_ROOT / configuration["source_manifest"]
    manifest = load_configuration(manifest_path)
    comparison = load_configuration(PROJECT_ROOT / configuration["comparison_configuration"])
    reference = load_configuration(PROJECT_ROOT / comparison["reference_configuration"])
    observed_ego = np.load(manifest_path.parent / "ego.npy", mmap_mode="r")
    eligible = [record for record in manifest["records"] if record["split"] == "navtrain" and Path(record.get("metric_cache_file", "missing")).is_file() and all(Path(path).is_file() for path in record["current_camera_paths"])]
    random = np.random.default_rng(configuration["seed"])
    group_names = np.asarray(sorted({record["recording_group"] for record in eligible})); random.shuffle(group_names)
    tuning_groups = set(group_names[:20])
    records = []
    for subset, count in [("train", configuration["training_scenes"]), ("tuning", configuration["tuning_scenes"])]:
        available = [dict(record) for record in eligible if (record["recording_group"] in tuning_groups) == (subset == "tuning")]
        random.shuffle(available)
        # Fixed round-robin command balance, determined without score labels.
        command_lists = [[record for record in available if int(observed_ego[record["cache_row"], 7:11].argmax()) == command] for command in range(3)]
        chosen, command_cursors = [], [0, 0, 0]
        while len(chosen) < count:
            for command in range(3):
                if len(chosen) == count:
                    break
                assert command_cursors[command] < len(command_lists[command])
                record = command_lists[command][command_cursors[command]]; command_cursors[command] += 1
                record.update(pilot_split=subset, pilot_row=len(records) + len(chosen), command_index=command,
                    command_name=["left_turn", "straight", "right_turn"][command], raw_log_split="trainval")
                chosen.append(record)
        records.extend(chosen)
    development_records = load_configuration(PROJECT_ROOT / configuration["development_source"] / "registration.json")["records"]
    for record in development_records:
        records.append({**record, "pilot_split": "locked_development", "pilot_row": len(records), "raw_log_split": "test"})
    assert len(development_records) == configuration["development_scenes"]
    groups = {subset: {record["recording_group"] for record in records if record["pilot_split"] == subset} for subset in ["train", "tuning", "locked_development"]}
    assert not groups["train"] & groups["tuning"] and not groups["train"] & groups["locked_development"] and not groups["tuning"] & groups["locked_development"]
    assert len({record["token"] for record in records}) == len(records)
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "records": records, "subset_counts": dict(Counter(record["pilot_split"] for record in records)),
        "recording_counts": {subset: len(group) for subset, group in groups.items()},
        "recording_overlap": 0, "source_manifest_sha256": file_sha256(manifest_path),
        "checkpoint_sha256": file_sha256(PROJECT_ROOT / comparison["drivor_checkpoint"]),
        "checkpoint": comparison["drivor_checkpoint"], "pretrained_checkpoint_used_trainval": True,
        "independent_final_test": False, "already_exposed_development_panel": True,
        "protected_training": protected_files_audit(reference)}
    assert registration["checkpoint_sha256"] == comparison["drivor_checkpoint_sha256"]
    if (output / "registration.json").exists():
        prior = load_configuration(output / "registration.json")
        for key in ["configuration", "records", "checkpoint_sha256"]:
            assert prior[key] == registration[key], key
    else:
        write_json(output / "registration.json", registration)
    write_json(destination / "subset_manifest.json", registration); write_json(destination / "configuration.json", configuration)
    print(json.dumps({"registered": registration["subset_counts"], "recordings": registration["recording_counts"]}), flush=True)


def build_observed_inputs(record):
    import torch
    from PIL import Image
    from pyquaternion import Quaternion
    with (PROJECT_ROOT / "dataset/navsim_logs" / record["raw_log_split"] / (record["log_name"] + ".pkl")).open("rb") as stream:
        frames = pickle.load(stream)
    cursor = record["current_frame_index"]; current, previous = frames[cursor], frames[cursor - 1]
    assert current["token"] == record["token"]
    assert int(np.argmax(current["driving_command"])) == record["command_index"]
    current_heading = Quaternion(current["ego2global_rotation"]).yaw_pitch_roll[0]
    previous_heading = Quaternion(previous["ego2global_rotation"]).yaw_pitch_roll[0]
    translation = np.asarray(previous["ego2global_translation"][:2]) - current["ego2global_translation"][:2]
    cosine, sine = np.cos(current_heading), np.sin(current_heading)
    relative_xy = np.array([[cosine, sine], [-sine, cosine]]) @ translation
    delta_heading = np.arctan2(np.sin(previous_heading - current_heading), np.cos(previous_heading - current_heading))
    history = np.stack((np.r_[relative_xy, delta_heading, previous["ego_dynamic_state"], previous["driving_command"]],
        np.r_[np.zeros(3), current["ego_dynamic_state"], current["driving_command"]])).astype(np.float32)
    mean = np.array([.485, .456, .406], np.float32); deviation = np.array([.229, .224, .225], np.float32)
    images = []
    for camera_index, image_path in enumerate(record["current_camera_paths"]):
        with Image.open(image_path) as image:
            normalized = (np.asarray(image.convert("RGB").resize((1148, 672)), np.float32) / 255 - mean) / deviation
        images.append(torch.from_numpy(normalized).permute(2, 0, 1))
    if "ego_history" in record:
        np.testing.assert_allclose(history, record["ego_history"], atol=1e-5)
    return {"image": torch.stack(images)[None], "ego_status": torch.from_numpy(history)[None]}, history


def cache_inputs(configuration):
    import torch
    from diagnose_residual_control_sampler import module_digest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"])
    assert torch.cuda.is_available(), "CUDA requires the authorized host execution context"
    torch.cuda.set_per_process_memory_fraction(configuration["maximum_process_bytes"] / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    comparison = load_configuration(PROJECT_ROOT / configuration["comparison_configuration"])
    reference = load_configuration(PROJECT_ROOT / comparison["reference_configuration"])
    model, loading = load_official_model(model_configuration_for(comparison, "drivor_public"), "cuda")
    initial_digest = module_digest(model)
    output = PROJECT_ROOT / configuration["output_directory"]; registration = load_configuration(output / "registration.json")
    records = registration["records"]; selected = [record for record in records if record["pilot_split"] != "locked_development"]
    cache_path = output / "cache"; cache_path.mkdir(parents=True, exist_ok=True)
    shapes = {"scene_features": (64, 256), "ego_tokens": (1, 256), "ego_history": (2, 11),
        "trajectory_tokens": (64, 256), "native_trajectories": (64, 8, 3), "native_log_scores": (64,), "native_probabilities": (64, 6)}
    arrays = {name: np.lib.format.open_memmap(cache_path / (name + ".npy"), mode="w+", dtype=np.float32, shape=(len(records), *shape)) for name, shape in shapes.items()}
    native_head = model.traj_head[-1]
    torch.save(native_head.cpu().state_dict(), output / "native_head_initialization.pt"); native_head.cuda()
    visual_capture, token_capture = [], []
    visual_hook = model.image_backbone.register_forward_hook(lambda _module, _arguments, returned: visual_capture.append(returned))
    head_hook = native_head.register_forward_pre_hook(lambda _module, arguments: token_capture.append(arguments[0]))
    started = time.perf_counter(); maximum_card = 0; maximum_replay_error = 0.; step_seconds = []
    original_forward = original_drivor_forward(model, comparison)
    for scene_index, record in enumerate(selected):
        memory = gpu_memory_status(configuration)
        maximum_card = max(maximum_card, memory["card_used_bytes"])
        assert memory["card_used_bytes"] < configuration["maximum_total_card_bytes"]
        assert time.perf_counter() - started < configuration["maximum_new_gpu_phase_seconds"]
        inputs, history = build_observed_inputs(record)
        inputs = {key: value.cuda() for key, value in inputs.items()}
        visual_capture.clear(); token_capture.clear(); step_started = time.perf_counter()
        with torch.inference_mode():
            prediction = model(inputs)
            visual = visual_capture[-1]; tokens = token_capture[-1]; ego_token = model.hist_encoding(inputs["ego_status"][:, -1])[:, None]
            log_scores, probabilities, _ = score_metrics(model, prediction["proposals"], visual, ego_token)
            replay = native_head(tokens).reshape(1, 64, 8, 3)
            error = float((replay - prediction["proposals"]).abs().max()); maximum_replay_error = max(maximum_replay_error, error)
            assert error == 0
            if scene_index == 0:
                original = original_forward(inputs)
                assert torch.equal(original["proposals"], prediction["proposals"])
                assert torch.max(torch.abs(original["pdm_score"] - log_scores)) < 1e-5
        torch.cuda.synchronize(); step_seconds.append(time.perf_counter() - step_started)
        for name, value in [("scene_features", visual), ("trajectory_tokens", tokens), ("ego_tokens", ego_token),
            ("native_trajectories", prediction["proposals"]), ("native_log_scores", log_scores), ("native_probabilities", probabilities)]:
            arrays[name][record["pilot_row"]] = value[0].cpu().numpy()
        arrays["ego_history"][record["pilot_row"]] = history
        for array in arrays.values():
            array.flush()
        write_json(output / "cache_progress.json", {"completed_navtrain_scenes": scene_index + 1, "total_navtrain_scenes": len(selected), "elapsed_seconds": time.perf_counter() - started, "maximum_card_used_bytes": maximum_card})
        if (scene_index + 1) % 16 == 0:
            print(json.dumps({"phase": "cache", "scenes": scene_index + 1, "seconds": time.perf_counter() - started}), flush=True)
    visual_hook.remove(); head_hook.remove()
    # Development visual features/native paths are actual original-checkpoint outputs.
    development = np.load(PROJECT_ROOT / configuration["development_source"] / "scorer_cache_and_candidates.npz")
    development_scores = np.load(PROJECT_ROOT / configuration["development_score_source"] / "distribution_context_scores.npz")
    rows = [record["pilot_row"] for record in records if record["pilot_split"] == "locked_development"]
    assert development["tokens"].tolist() == [records[row]["token"] for row in rows]
    with torch.inference_mode():
        for source_index, row in enumerate(rows):
            visual = torch.from_numpy(development["drivor_scene_features"][source_index:source_index+1]).cuda()
            ego = torch.from_numpy(development["drivor_ego_tokens"][source_index:source_index+1]).cuda()
            tokens = model.trajectory_decoder(ego + model.init_feature.weight[None], visual)[-1]
            paths = native_head(tokens).reshape(1, 64, 8, 3)
            error = float(np.abs(paths[0].cpu().numpy() - development["trajectories"][source_index, :64]).max())
            maximum_replay_error = max(maximum_replay_error, error); assert error < 1e-4, error
            arrays["scene_features"][row] = visual[0].cpu().numpy(); arrays["ego_tokens"][row] = ego[0].cpu().numpy()
            arrays["trajectory_tokens"][row] = tokens[0].cpu().numpy(); arrays["native_trajectories"][row] = development["trajectories"][source_index, :64]
            arrays["native_log_scores"][row] = development_scores["separate_log_scores"][source_index, :64]
            arrays["native_probabilities"][row] = development_scores["separate_metric_probabilities"][source_index, :64]
            arrays["ego_history"][row] = records[row]["ego_history"]
    for array in arrays.values():
        array.flush()
    assert initial_digest == module_digest(model) and all(parameter.grad is None for parameter in model.parameters())
    write_json(output / "cache_complete.json", {"complete": True, "loading": loading, "cache_shapes": shapes, "dtype": "float32", "input_cameras": 4,
        "image_resolution": [1148, 672], "observed_image_frames": 1, "future_image_inputs": 0,
        "frozen_final_decoder_tokens_are_head_inputs": True, "maximum_original_vs_cached_head_error_m_or_rad": maximum_replay_error,
        "encoder_calls_during_cache": len(selected) + 1, "encoder_calls_during_cached_training": 0,
        "gpu_phase_wall_seconds": time.perf_counter() - started, "mean_gpu_forward_seconds_per_scene": float(np.mean(step_seconds)),
        "maximum_card_used_bytes": maximum_card, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "all_original_weights_frozen_and_unchanged": True,
        "cache_sha256": {name: file_sha256(cache_path / (name + ".npy")) for name in arrays},
        "native_head_initialization_sha256": file_sha256(output / "native_head_initialization.pt"),
        "protected_training": protected_files_audit(reference), "environment": {"torch": torch.__version__, "numpy": np.__version__}})
    print(json.dumps({"cache_complete": True, "gpu_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True); parser.add_argument("--phase", choices=["register", "cache"], required=True)
    arguments = parser.parse_args(); configuration = load_configuration(arguments.configuration)
    register(configuration) if arguments.phase == "register" else cache_inputs(configuration)
