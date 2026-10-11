#!/usr/bin/env python
"""Register train/development recordings and cache the actual frozen scorer.

Original corpus images are pixel caches, not features. This script executes the
encoder only during cache creation and stores the real 16x256 scene output.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import file_sha256, protected_training_audit, write_json


def prepare(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / configuration["corpus_directory"]
    manifest = json.loads((source / "manifest.json").read_text())
    rng = random.Random(configuration["seed"])
    train = [dict(record) for record in manifest["records"] if record["study_split"] == "train"
             and "metric_cache_file" in record and Path(record["metric_cache_file"]).exists()]
    development = [dict(record) for record in manifest["records"] if record["study_split"] == "dev"]
    rng.shuffle(train)
    rng.shuffle(development)
    train = train[:configuration["training_scenes"]]
    all_training_groups = {record["recording_group"] for record in manifest["records"] if record["study_split"] == "train"}
    development_groups = sorted({record["recording_group"] for record in development})
    rng.shuffle(development_groups)
    tuning_groups = set(development_groups[:max(1, len(development_groups)//4)])
    tuning = [record for record in development if record["recording_group"] in tuning_groups][:configuration["tuning_target_scenes"]]
    evaluation = [record for record in development if record["recording_group"] not in tuning_groups][:configuration["development_scenes"]-len(tuning)]
    ego = np.load(source / "ego.npy", mmap_mode="r")
    records = []
    for subset, chosen in (("train", train), ("tuning", tuning), ("locked_development", evaluation)):
        for index, record in enumerate(chosen):
            assert record["split"] == ("navtrain" if subset == "train" else "navval")
            if subset != "train":
                assert record["recording_group"] not in all_training_groups
            record["pilot_split"] = subset
            record["pilot_row"] = len(records)
            record["common_scout"] = subset == "train" and index < configuration["common_scout_scenes"]
            record["command_index"] = int(ego[record["cache_row"], 7:11].argmax())
            assert Path(record["metric_cache_file"]).exists()
            assert len(record["observed_front_paths"]) == 2
            records.append(record)
    assert len(records) == configuration["training_scenes"]+configuration["development_scenes"]
    assert len({record["token"] for record in records}) == len(records)
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration, "records": records,
        "checkpoint_sha256": file_sha256(PROJECT_ROOT / configuration["checkpoint"]),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "cache_builder_sha256": file_sha256(Path(__file__)),
        "train_dev_recording_overlap": 0, "tuning_locked_recording_overlap": 0,
        "subset_counts": dict(Counter(record["pilot_split"] for record in records)),
        "excluded_training_scenes_without_official_metric_cache": sum(
            record["study_split"] == "train" and "metric_cache_file" not in record for record in manifest["records"]),
        "independent_final_test": False, "previously_used_development_corpus": True,
        "protected_training": protected_training_audit(configuration)}
    path = output / "registration.json"
    if path.exists():
        previous = json.loads(path.read_text())
        for key in ("configuration", "records", "checkpoint_sha256", "source_manifest_sha256", "cache_builder_sha256"):
            assert previous[key] == registration[key], key
    else:
        write_json(path, registration)
    write_json(PROJECT_ROOT / configuration["results_directory"] / "subset_manifest.json", registration)
    print(json.dumps({"phase": "prepare", "counts": registration["subset_counts"]}), flush=True)


def scorer_arrays(model, trajectories, scene_features, ego_tokens):
    from planning_aware_future_prediction.models.scorer_selection_learning import SUBSCORE_NAMES
    scored_features = model.planner.scorer_attention(
        model.planner.pos_embed(trajectories.flatten(2)), scene_features) + ego_tokens
    logits = model.planner.scorer(trajectories, scored_features)[0]
    return scored_features, __import__("torch").stack([logits[name] for name in SUBSCORE_NAMES], -1)


def cache(configuration):
    import torch
    from diagnose_residual_control_sampler import distribution_seed, load_reference_helpers, module_digest
    from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score
    output = PROJECT_ROOT / configuration["output_directory"]
    registration = json.loads((output / "registration.json").read_text())
    records = registration["records"]
    torch.set_num_threads(configuration["cpu_threads"])
    torch.manual_seed(configuration["seed"])
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in ("0", "1", "")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(configuration["cache_allocator_limit_gib"] * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
    constructor, reference = load_reference_helpers(configuration)
    model = constructor("drivor", None)
    checkpoint = torch.load(PROJECT_ROOT / configuration["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    assert checkpoint["completed_updates"] == configuration["expected_checkpoint_updates"]
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval().requires_grad_(False)
    initial_digest = module_digest(model)
    model.to(device)
    score_weights = {name: float(getattr(model.planner._config, name)) for name in ("noc", "dac", "ddc", "ep", "ttc", "comfort")}
    specifications = {"scene_features": (16, 256), "ego_tokens": (1, 256), "ego_status": (11,),
        "native_trajectories": (64, 8, 3), "native_scored_features": (64, 256), "native_component_logits": (64, 6),
        "base_controls": (8, 2), "reference_standard_deviations": (8, 2), "ground_truth_trajectory": (8, 3),
        "residual_trajectories": (64, 8, 3), "residual_scored_features": (64, 256), "residual_component_logits": (64, 6)}
    destination = output / "cache"
    destination.mkdir(exist_ok=True)
    arrays = {name: np.lib.format.open_memmap(destination / (name+".npy"), mode="w+", dtype="float32", shape=(len(records), *shape))
              for name, shape in specifications.items()}
    source = PROJECT_ROOT / configuration["corpus_directory"]
    images = np.load(source / "images.npy", mmap_mode="r")
    egos = np.load(source / "ego.npy", mmap_mode="r")
    ground_truths = np.load(source / "trajectory.npy", mmap_mode="r")
    extracted = []
    hook = model.backbone.register_forward_hook(lambda _module, _inputs, value: extracted.append(value))
    started = time.perf_counter()
    maximum_replay_error = 0.
    cache_encoder_calls = 0
    profile_seconds = []
    for start in range(0, len(records), configuration["cache_batch_size"]):
        chosen = records[start:start+configuration["cache_batch_size"]]
        rows = [record["cache_row"] for record in chosen]
        observed = torch.from_numpy(np.array(images[rows])).permute(0, 1, 4, 2, 3).to(device).float()/255
        ego = torch.from_numpy(np.array(egos[rows])).to(device)
        extracted.clear()
        step_started = time.perf_counter()
        with torch.no_grad():
            predicted = model({"image": observed, "ego_status": ego[:, None]})
            cache_encoder_calls += 1
            assert len(extracted) == 1 and extracted[0].shape == (len(chosen), 16, 256)
            scene = extracted[0]
            ego_token = model.planner.hist_encoding(ego)[:, None]
            scored, logits = scorer_arrays(model, predicted["proposals"], scene, ego_token)
            replay = compose_drivor_log_score(logits, score_weights)
            error = float((replay-predicted["pdm_score"]).abs().max())
            assert error < 2e-5, error
            maximum_replay_error = max(maximum_replay_error, error)
            controls, deviations = distribution_seed(reference, predicted["proposals"], ego[:, 3].clamp_min(0), predicted["trajectory"], .5)
            noise = torch.stack([torch.randn(63, 8, 2, generator=torch.Generator().manual_seed(500000+record["pilot_row"])) for record in chosen]).to(device)
            residual_controls = reference.clamp_controls(controls[:, None]+.25*deviations[:, None]*noise)
            residuals = reference.bicycle_rollout(residual_controls, ego[:, 3].clamp_min(0), .5)[0]
            residuals = torch.cat((predicted["trajectory"][:, None], residuals), 1)
            residual_features, residual_logits = scorer_arrays(model, residuals, scene, ego_token)
            tensors = {"scene_features": scene, "ego_tokens": ego_token, "ego_status": ego,
                "native_trajectories": predicted["proposals"], "native_scored_features": scored,
                "native_component_logits": logits, "base_controls": controls, "reference_standard_deviations": deviations,
                "residual_trajectories": residuals, "residual_scored_features": residual_features,
                "residual_component_logits": residual_logits}
            for name, tensor in tensors.items():
                values = tensor.float().cpu().numpy()
                assert np.isfinite(values).all(), name
                arrays[name][start:start+len(chosen)] = values
            arrays["ground_truth_trajectory"][start:start+len(chosen)] = ground_truths[rows]
        if device.type == "cuda":
            torch.cuda.synchronize()
        profile_seconds.append(time.perf_counter()-step_started)
        if start == 0 or (start+len(chosen)) % 64 == 0:
            write_json(output / "cache_progress.json", {"completed_scenes": start+len(chosen), "total_scenes": len(records), "elapsed_seconds": time.perf_counter()-started})
            print(json.dumps({"phase": "cache", "scenes": start+len(chosen), "elapsed_seconds": time.perf_counter()-started}), flush=True)
    hook.remove()
    for array in arrays.values():
        array.flush()
    assert all(parameter.grad is None for parameter in model.parameters())
    model.cpu()
    assert module_digest(model) == initial_digest
    metadata = {"complete": True, "scene_count": len(records), "scene_feature_shape": [len(records),16,256],
        "dtype": "float32", "valid_token_mask": "All sixteen scene/register tokens valid; no padding",
        "spatial_interpretation": "Scene/register tokens have no direct image coordinate mapping",
        "encoder_checkpoint_sha256": registration["checkpoint_sha256"], "checkpoint_updates": 3200,
        "preprocessing": {"source": manifest_preprocessing(source), "camera": "CAM_F0", "observed_frames": 2,
                          "image_size": [512,256], "dinov2_boundary_padding": [6,10], "future_input": False},
        "feature_extraction": "RegisterFrontEncoder output, after temporal fusion, before DrivoR decoders",
        "score_weights": score_weights, "maximum_original_cached_score_error": maximum_replay_error,
        "encoder_calls_cache_creation": cache_encoder_calls, "encoder_calls_cached_training": 0,
        "frozen_weights_unchanged": True, "frozen_parameter_gradients_absent": True,
        "wall_seconds": time.perf_counter()-started, "physical_gpu": int(os.environ["CUDA_VISIBLE_DEVICES"]) if device.type == "cuda" else None,
        "gpu_occupied_seconds": time.perf_counter()-started if device.type == "cuda" else 0.,
        "maximum_allocated_gpu_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
        "maximum_reserved_gpu_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0,
        "median_forward_and_scorer_seconds": float(np.median(profile_seconds[2:])),
        "cache_files_sha256": {name: file_sha256(destination/(name+".npy")) for name in arrays},
        "protected_training": protected_training_audit(configuration)}
    write_json(output / "cache_complete.json", metadata)
    print(json.dumps({"phase":"cache", "complete":True, "wall_seconds": metadata["wall_seconds"]}), flush=True)


def manifest_preprocessing(source):
    return json.loads((source/"manifest.json").read_text())["preprocessing"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare","cache"), required=True)
    arguments = parser.parse_args()
    globals()[arguments.phase](json.loads(arguments.configuration.read_text()))
