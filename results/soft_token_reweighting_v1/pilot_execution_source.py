"""Bounded cache / verification / paired pilot / official evaluation stages."""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.models.soft_token_reweighting import (
    CachedTokenPlanner, copy_official_planner, cross_attention_bias,
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def scientific_source_hashes():
    relative_paths = ["scripts/run_soft_token_reweighting.py",
        "src/planning_aware_future_prediction/models/soft_token_reweighting.py",
        "configs/soft_token_reweighting/pilot_v1.json",
        "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py",
        "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_agent.py",
        "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_features.py"]
    return {path: sha256(PROJECT_ROOT / path) for path in relative_paths}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_setup(configuration):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(configuration["process_allocator_limit_bytes"] / total)
    return check_card(configuration)


def check_card(configuration):
    output = subprocess.check_output(["nvidia-smi", "--id=" + str(configuration["physical_gpu"]),
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
    used = int(output.strip()) * 1024**2
    if used > configuration["whole_card_limit_bytes"]:
        raise RuntimeError("Whole-card memory exceeds requested cap: " + str(used))
    return used


def setup_official():
    from audit_official_drive_jepa_evaluation import official_configuration
    return official_configuration(PROJECT_ROOT)


def official_loss(prediction, target):
    from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import l1_length_normalized_loss
    return l1_length_normalized_loss(prediction, target, alpha=5.0)


def prepare(configuration, output):
    started = time.time()
    assert not (output / "subset_manifest.json").exists(), "Use a fresh output directory for a replay"
    source = read_json(PROJECT_ROOT / configuration["source_image_manifest"])
    generator = np.random.default_rng(configuration["seed"])
    selected = []
    for official_split, pilot_split, count in (("navtrain", "train", configuration["train_scenes"]),
                                             ("navval", "validation", configuration["validation_scenes"])):
        candidates = [row for row in source["records"] if row["split"] == official_split]
        indices = generator.permutation(len(candidates))[:count]
        assert len(indices) == count
        for index in indices:
            row = dict(candidates[int(index)])
            row["source_cache_row"] = row["cache_row"]
            row["pilot_split"] = pilot_split
            row["encoder_cache_row"] = len(selected)
            selected.append(row)
    train_groups = {row["recording_group"] for row in selected if row["pilot_split"] == "train"}
    val_groups = {row["recording_group"] for row in selected if row["pilot_split"] == "validation"}
    assert not train_groups & val_groups
    assert len({row["token"] for row in selected}) == len(selected)
    # Verify original official log membership, observed paths, and actual timestamps.
    import yaml
    import cv2
    official_root = PROJECT_ROOT / "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1"
    official_assignment = yaml.safe_load((official_root / "navsim/planning/script/config/training/default_train_val_test_log_split.yaml").read_text())
    for split in ("navtrain", "navval"):
        official_logs = set(official_assignment["train_logs" if split == "navtrain" else "val_logs"])
        assert all(row["log_name"] in official_logs for row in selected if row["split"] == split)
    by_log = {}
    for row in selected:
        by_log.setdefault(row["log_name"], []).append(row)
    for log_name, records in by_log.items():
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / (log_name + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        for row in records:
            current_index = row["current_frame_index"]
            current = frames[current_index]
            assert current["token"] == row["token"]
            observed = frames[current_index - 1:current_index + 1]
            assert len(observed) == 2
            timestamps = [int(frame["timestamp"]) for frame in observed]
            offsets = [(stamp - timestamps[-1]) / 1e6 for stamp in timestamps]
            assert -0.65 < offsets[0] < -0.35 and offsets[-1] == 0
            for frame, path in zip(observed, row["observed_front_paths"]):
                raw_path = next(camera["data_path"] for name, camera in frame["cams"].items() if name.lower() == "cam_f0")
                assert str(path).endswith(raw_path)
            row["observed_timestamps_microseconds"] = timestamps
            row["observed_offsets_seconds"] = offsets
    source_images = np.load(PROJECT_ROOT / configuration["source_image_cache"], mmap_mode="r")
    source_directory = PROJECT_ROOT / "outputs/four_model_small_corpus_v1/corpus"
    source_ego = np.load(source_directory / "ego.npy", mmap_mode="r")
    source_trajectory = np.load(source_directory / "trajectory.npy", mmap_mode="r")
    source_indices = [row["source_cache_row"] for row in selected]
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ego = np.asarray(source_ego[source_indices])
    # Official PF ordering: command4, velocity2, acceleration2.
    np.save(cache / "ego_status.npy", np.concatenate((ego[:, 7:11], ego[:, 3:7]), axis=-1).astype(np.float32))
    np.save(cache / "trajectory.npy", source_trajectory[source_indices].astype(np.float32))
    np.save(cache / "valid_token_mask.npy", np.ones((len(selected), 128), dtype=bool))
    from planning_aware_future_prediction.models.soft_token_reweighting import token_coordinates
    np.save(cache / "token_coordinates.npy", token_coordinates().numpy())
    raw_checks = []
    for row in selected[:3] + selected[-3:]:
        raw_pair = []
        for path in row["observed_front_paths"]:
            image = cv2.imread(path)
            assert image is not None
            raw_pair.append(cv2.cvtColor(cv2.resize(image[28:-28], (512, 256)), cv2.COLOR_BGR2RGB))
        difference = np.abs(np.stack(raw_pair).astype(float) - source_images[row["source_cache_row"]]).max()
        assert difference == 0
        raw_checks.append({"token": row["token"], "raw_vs_image_cache_max_difference": float(difference)})
    manifest = {"records": selected, "counts": {"train": configuration["train_scenes"], "validation": configuration["validation_scenes"]},
        "seed": configuration["seed"], "train_recordings": len(train_groups), "validation_recordings": len(val_groups),
        "recording_overlap": 0, "official_split_log_membership_verified": True,
        "source_manifest_sha256": sha256(PROJECT_ROOT / configuration["source_image_manifest"]),
        "future_frames_in_features": False, "raw_preprocessing_checks": raw_checks,
        "scope": "official navtrain/navval subset; historical development reuse; pretrained checkpoint may have used validation for selection",
        "preparation_seconds": time.time() - started}
    write_json(output / "subset_manifest.json", manifest)
    write_json(output / "configuration.json", configuration)
    print(json.dumps({"prepared": manifest["counts"], "recording_overlap": 0}), flush=True)


def create_official_model(output):
    from hydra.utils import instantiate
    specification, assets, config, official_root = setup_official()
    with (output / "official_loading.log").open("w") as stream, contextlib.redirect_stdout(stream):
        agent = instantiate(config.agent)
        checkpoint = torch.load(config.agent.checkpoint_path, map_location="cpu", mmap=True)
        loading = agent.load_state_dict({key.replace("agent.", ""): value for key, value in checkpoint["state_dict"].items()}, strict=True)
    agent.eval().requires_grad_(False)
    agent._model.freeze_encoder = True
    assert sha256(assets["planning_checkpoint"]["path"]) == assets["planning_checkpoint"]["sha256"]
    return agent, specification, assets, str(loading)


@torch.no_grad()
def extract_tokens(official_model, rgb_images):
    # rgb_images: uint8 B,2,256,512,3; only the observed tubelet is passed.
    camera_clip = torch.as_tensor(np.array(rgb_images), device="cuda").permute(0, 4, 1, 2, 3).float().div_(255)
    flattened = camera_clip.permute(0, 2, 1, 3, 4).flatten(0, 1)
    normalized = official_model.transform(flattened).reshape(-1, 2, 3, 256, 512).permute(0, 2, 1, 3, 4)
    encoded = official_model.image_encoder(normalized)
    assert encoded.shape[1:] == (512, 1024)
    pooled = official_model.avg_pool(encoded.transpose(1, 2).reshape(-1, 1024, 16, 32)).flatten(2).transpose(1, 2)
    assert pooled.shape[1:] == (128, 1024)
    return pooled, camera_clip


def cache_features(configuration, output):
    started = time.time()
    gpu_setup(configuration)
    agent, specification, assets, loading = create_official_model(output)
    official = agent._model.cuda().eval()
    encoder_digest = state_digest(official.image_encoder)
    components = copy_official_planner(official)
    components.requires_grad_(True)
    torch.save(components, output / "initial_planner.pt")
    manifest = read_json(output / "subset_manifest.json")
    records = manifest["records"]
    image_cache = np.load(PROJECT_ROOT / configuration["source_image_cache"], mmap_mode="r")
    destination = output / "cache/visual_tokens.npy"
    progress_path = output / "cache/progress.json"
    completed = read_json(progress_path)["completed"] if progress_path.exists() else 0
    features = np.lib.format.open_memmap(destination, mode="r+" if completed else "w+",
        dtype=np.float32, shape=(len(records), 128, 1024))
    maximum_card = check_card(configuration)
    verification_rows = []
    decoder = CachedTokenPlanner(copy.deepcopy(components), "baseline").cuda().eval()
    ego = np.load(output / "cache/ego_status.npy", mmap_mode="r")
    for offset in range(completed, len(records), configuration["cache_batch_size"]):
        selected = records[offset:offset + configuration["cache_batch_size"]]
        images = image_cache[[row["source_cache_row"] for row in selected]]
        with torch.no_grad():
            tokens, camera_clip = extract_tokens(official, images)
            features[offset:offset + len(selected)] = tokens.cpu().numpy()
            if offset in {0, configuration["train_scenes"]}:
                status = torch.as_tensor(np.array(ego[offset:offset + len(selected)]), device="cuda")
                original = official(camera_clip, status)["trajectory"]
                restored = torch.as_tensor(np.array(features[offset:offset + len(selected)]), device="cuda")
                cached_prediction = decoder(restored, torch.ones(restored.shape[:2], device="cuda", dtype=torch.bool), status)
                difference = float((cached_prediction - original).abs().max())
                assert torch.allclose(original, cached_prediction, rtol=1e-5, atol=1e-5), difference
                verification_rows.append({"tokens": [row["token"] for row in selected],
                    "feature_max_difference": float((restored-tokens).abs().max()),
                    "prediction_max_difference": difference})
        if (offset // configuration["cache_batch_size"]) % 25 == 0 or offset + len(selected) == len(records):
            features.flush()
            maximum_card = max(maximum_card, check_card(configuration))
            progress = {"completed": offset + len(selected), "total": len(records),
                "seconds": time.time()-started, "maximum_card_used_bytes": maximum_card}
            write_json(progress_path, progress)
            print(json.dumps(progress), flush=True)
    assert all(parameter.grad is None for parameter in official.image_encoder.parameters())
    assert state_digest(official.image_encoder) == encoder_digest
    features.flush()
    metadata = {"complete": True, "feature_shape": list(features.shape), "dtype": str(features.dtype),
        "feature_location": "official image_encoder output (512x1024), then original AvgPool2d2 to128x1024 before learned image_fc",
        "preprocessing": "front only; past/current[-0.5,0]s; RGB crop28 top/bottom; cv2 INTER_LINEAR512x256; /255; ImageNet normalization",
        "token_grid": [8, 16], "joint_observed_tubelet": True, "future_inputs": False,
        "encoder_frozen": True, "encoder_eval": not official.image_encoder.training,
        "encoder_no_gradients": True, "encoder_state_sha256": encoder_digest,
        "full_planning_checkpoint": assets["planning_checkpoint"], "strict_loading": loading,
        "official_source_commit": specification["official_source_commit"],
        "planner_initial_sha256": sha256(output / "initial_planner.pt"),
        "subset_manifest_sha256": sha256(output / "subset_manifest.json"),
        "cache_visual_tokens_sha256": sha256(destination),
        "raw_vs_cache_checks": verification_rows, "cache_seconds": time.time()-started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "maximum_whole_card_bytes": maximum_card,
        "environment": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "gpu": torch.cuda.get_device_name(0)}}
    write_json(output / "cache/metadata.json", metadata)
    print("ENCODER_CACHE_COMPLETE", flush=True)


class EncoderFeatureCache:
    """Only encoder features/targets, memory mapped; never loads RGB or encoder."""
    def __init__(self, output):
        self.records = read_json(output / "subset_manifest.json")["records"]
        self.arrays = {name: np.load(output / "cache" / (name + ".npy"), mmap_mode="r")
                       for name in ("visual_tokens", "valid_token_mask", "ego_status", "trajectory")}

    def minibatch(self, indices, device="cuda"):
        return {name: torch.from_numpy(np.array(values[indices])).to(device)
                for name, values in self.arrays.items()}

    def indices(self, split):
        return np.array([index for index, row in enumerate(self.records) if row["pilot_split"] == split])


def model_from_initial(output, condition, device="cpu"):
    seed_everything(0)
    components = torch.load(output / "initial_planner.pt", map_location="cpu")
    components.requires_grad_(True)
    return CachedTokenPlanner(components, condition).to(device)


def prediction_from_minibatch(model, minibatch, **options):
    return model(minibatch["visual_tokens"], minibatch["valid_token_mask"], minibatch["ego_status"], **options)


def norm_of_gradients(parameters):
    terms = [parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def verify(configuration, output):
    setup_official()
    torch.set_num_threads(2)
    cache = EncoderFeatureCache(output)
    from types import SimpleNamespace
    from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
    from navsim.agents.drive_jepa_perception_free.drive_jepa_features import DriveJEPAFeatureDIBuilder
    builder = DriveJEPAFeatureDIBuilder(front_only=True)
    source_images = np.load(PROJECT_ROOT / configuration["source_image_cache"], mmap_mode="r")
    raw_target_checks = []
    for index in [0, 1, 2, configuration["train_scenes"], configuration["train_scenes"]+1, configuration["train_scenes"]+2]:
        record = cache.records[index]
        with (PROJECT_ROOT / "dataset/navsim_logs/trainval" / (record["log_name"] + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        cursor = record["current_frame_index"]
        history = frames[cursor-3:cursor+1]
        agent_input = AgentInput.from_scene_dict_list(history, PROJECT_ROOT / "dataset/sensor_blobs/trainval", 4,
            SensorConfig.build_front_only_sensors(include=[2, 3]))
        native_features = builder.compute_features(agent_input)
        image_tensor = torch.stack((native_features["camera_feature_2"], native_features["camera_feature_1"]))
        stored_image = torch.as_tensor(np.array(source_images[record["source_cache_row"]])).permute(0,3,1,2).float()/255
        torch.testing.assert_close(image_tensor, stored_image, rtol=0, atol=0)
        torch.testing.assert_close(native_features["status_feature"], torch.as_tensor(np.array(cache.arrays["ego_status"][index])), rtol=0, atol=0)
        # Use the official future-trajectory method without loading map/sensors for targets.
        native_scene = Scene.__new__(Scene)
        native_scene.scene_metadata = SimpleNamespace(num_history_frames=4)
        native_scene.frames = [SimpleNamespace(ego_status=Scene._build_ego_status(frame)) for frame in frames[cursor-3:cursor+9]]
        native_target = native_scene.get_future_trajectory(num_trajectory_frames=8).poses
        difference_target = float(np.abs(native_target-cache.arrays["trajectory"][index]).max())
        assert difference_target < 1e-5
        raw_target_checks.append({"token": record["token"], "raw_image_preprocessing_max_difference": 0,
            "ego_status_max_difference": 0, "gt_trajectory_max_difference": difference_target})
    observations = cache.minibatch([0, 1, 2], "cpu")
    original = model_from_initial(output, "baseline").eval()
    modified = model_from_initial(output, "conditioned").eval()
    assert state_digest(original.planner) == state_digest(modified.planner)
    with torch.no_grad():
        baseline_prediction = prediction_from_minibatch(original, observations)
        zero_prediction = prediction_from_minibatch(modified, observations, intervention="beta_zero")
        difference = float((baseline_prediction-zero_prediction).abs().max())
        assert torch.allclose(baseline_prediction, zero_prediction, rtol=2e-5, atol=2e-5)
        baseline_loss = official_loss(baseline_prediction, observations["trajectory"])
        zero_loss = official_loss(zero_prediction, observations["trajectory"])
        assert torch.allclose(baseline_loss, zero_loss, rtol=2e-5, atol=2e-5)
    modified.train()
    prediction = prediction_from_minibatch(modified, observations)
    official_loss(prediction, observations["trajectory"]).backward()
    gradient = {"importance": norm_of_gradients(modified.importance.parameters()),
                "planner": norm_of_gradients(modified.planner.parameters())}
    initial_bias_quantiles = torch.quantile((modified.importance.beta * modified.last_importance).detach().flatten(),
        torch.tensor([0., .01, .5, .99, 1.])).tolist()
    for name, value in gradient.items():
        assert np.isfinite(value) and value > 0, (name, value)
    for name, parameter in modified.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    # Cached model contains no encoder object. Poison any accidental encoder call.
    class ForbiddenEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.calls = 0
        def forward(self, *arguments):
            self.calls += 1
            raise AssertionError("Encoder forward inside cache training")
    forbidden = ForbiddenEncoder().eval()
    modified.image_encoder = forbidden
    modified.zero_grad(set_to_none=True)
    official_loss(prediction_from_minibatch(modified, observations), observations["trajectory"]).backward()
    assert forbidden.calls == 0 and forbidden.weight.grad is None
    del modified.image_encoder
    # Real padded sample: invalid contents cannot affect valid-token summary/output.
    padded = {name: tensor.clone() for name, tensor in observations.items()}
    padded["valid_token_mask"][0, 3:9] = False
    padded["valid_token_mask"][1, -11:] = False
    modified.eval()
    with torch.no_grad():
        padded_prediction = prediction_from_minibatch(modified, padded, capture_attention=True)
        attention = modified.last_attention
        complete_padding = torch.cat((~padded["valid_token_mask"], torch.zeros(3, 1, dtype=torch.bool)), dim=1)
        invalid_attention = attention.masked_select(complete_padding[:, None, None].expand_as(attention))
        assert float(invalid_attention.abs().max()) == 0
        padded["visual_tokens"][~padded["valid_token_mask"]] += 1000
        replacement_prediction = prediction_from_minibatch(modified, padded)
        assert torch.allclose(padded_prediction, replacement_prediction, rtol=2e-5, atol=2e-5)
    # Equal QK synthetic case, two batches, eight heads, eight waypoint queries.
    valid = torch.tensor([[True, True, False], [True, False, True]])
    importance = torch.zeros(2, 3, requires_grad=True)
    flat_mask = cross_attention_bias(importance, valid, 0.1, 8, 8)
    probabilities = flat_mask.softmax(-1).reshape(2, 8, 8, 4)
    raised = importance + torch.tensor([[2., 0., 0.], [0., 0., 3.]])
    shifted = cross_attention_bias(raised, valid, 0.1, 8, 8).softmax(-1).reshape(2, 8, 8, 4)
    assert torch.all(shifted[0, :, :, 0] > probabilities[0, :, :, 0])
    assert torch.all(shifted[1, :, :, 2] > probabilities[1, :, :, 2])
    assert (shifted[0, :, :, 2] == 0).all() and (shifted[1, :, :, 1] == 0).all()
    shifted[..., 0].sum().backward()
    assert importance.grad is not None and importance.grad.abs().sum() > 0
    synthetic_attention = torch.nn.MultiheadAttention(32, 8, dropout=0, batch_first=True)
    synthetic_queries = torch.zeros(2, 8, 32)
    synthetic_keys = torch.zeros(2, 4, 32)
    _, actual_probabilities = synthetic_attention(synthetic_queries, synthetic_keys, synthetic_keys,
        attn_mask=cross_attention_bias(raised, valid, 0.1, 8, 8), need_weights=True, average_attn_weights=False)
    torch.testing.assert_close(actual_probabilities, shifted)
    # Ego enters the native planner in A, B, C. B only zeros the importance input.
    unconditioned = model_from_initial(output, "unconditioned").eval()
    assert sum(p.numel() for p in unconditioned.importance.parameters()) == sum(p.numel() for p in modified.importance.parameters())
    with torch.no_grad():
        first_importance = unconditioned.importance(observations["visual_tokens"], observations["valid_token_mask"],
            torch.zeros_like(observations["ego_status"]), unconditioned.positions)
        changed_status = observations["ego_status"].clone()
        changed_status[:, :4] = changed_status[:, :4].roll(1, -1)
        unconditioned(observations["visual_tokens"], observations["valid_token_mask"], changed_status)
        assert torch.equal(first_importance, unconditioned.last_importance)
    metadata = read_json(output / "cache/metadata.json")
    assert metadata["complete"] and metadata["encoder_frozen"] and metadata["encoder_eval"]
    assert metadata["encoder_no_gradients"] and metadata["raw_vs_cache_checks"]
    result = {"passed": True, "cpu": True, "beta_zero_prediction_max_difference": difference,
        "beta_zero_loss_difference": float((baseline_loss-zero_loss).abs()), "gradient_norms": gradient,
        "initial_beta": float(modified.importance.beta), "initial_bias_quantiles": initial_bias_quantiles,
        "encoder_forward_calls_in_cached_training": forbidden.calls, "encoder_gradient": None,
        "invalid_attention_max": float(invalid_attention.abs().max()),
        "synthetic_bias_broadcast_shape": list(flat_mask.shape), "synthetic_bias_increases_attention": True,
        "padding_content_invariance": True, "same_native_initial_state": True,
        "same_unconditioned_conditioned_module_parameters": True,
        "unconditioned_importance_ego_invariance": True, "original_to_cache": metadata["raw_vs_cache_checks"],
        "raw_official_builder_target_checks": raw_target_checks,
        "atol": 2e-5, "rtol": 2e-5, "dropout": 0.0}
    write_json(output / "verification.json", result)
    print(json.dumps(result), flush=True)


def profile(configuration, output):
    started = time.perf_counter()
    assert read_json(output / "verification.json")["passed"]
    setup_official()
    gpu_setup(configuration)
    cache = EncoderFeatureCache(output)
    observations = cache.minibatch(cache.indices("train")[:configuration["candidate_batch_size"]])
    results = {}
    for condition in configuration["conditions"]:
        model = model_from_initial(output, condition, "cuda").train()
        optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"])
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(25):
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = official_loss(prediction_from_minibatch(model, observations), observations["trajectory"])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            if step >= 5:
                durations.append(time.perf_counter()-started)
        results[condition] = {"mean_step_seconds": float(np.mean(durations)),
            "p95_step_seconds": float(np.percentile(durations, 95)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "whole_card_used_bytes": check_card(configuration)}
        del optimizer, model
        torch.cuda.empty_cache()
    # Reserve 25% plus account for data/diagnostic overhead, one common budget.
    safe_seconds_per_three_steps = sum(max(row["p95_step_seconds"], row["mean_step_seconds"]) for row in results.values()) * 1.5
    steps = min(configuration["maximum_optimizer_steps"], int(configuration["maximum_training_gpu_seconds"] / safe_seconds_per_three_steps))
    assert steps > 0
    execution = {"common_steps": steps, "common_batch_size": configuration["candidate_batch_size"],
        "profile": results, "estimated_total_training_gpu_seconds": steps * safe_seconds_per_three_steps,
        "profile_weights_discarded": True, "training_restarts_from_initial_checkpoint": True,
        "maximum_training_gpu_seconds": configuration["maximum_training_gpu_seconds"]}
    generator = np.random.default_rng(configuration["seed"])
    training_indices = cache.indices("train")
    schedule = []
    while len(schedule) < steps:
        shuffled = generator.permutation(training_indices)
        for offset in range(0, len(shuffled)-execution["common_batch_size"]+1, execution["common_batch_size"]):
            schedule.append(shuffled[offset:offset+execution["common_batch_size"]])
            if len(schedule) == steps:
                break
    np.save(output / "training_schedule.npy", np.asarray(schedule))
    execution["schedule_sha256"] = sha256(output / "training_schedule.npy")
    execution["initial_planner_sha256"] = sha256(output / "initial_planner.pt")
    execution["source_sha256"] = scientific_source_hashes()
    execution["profiling_seconds"] = time.perf_counter()-started
    write_json(output / "execution.json", execution)
    print(json.dumps(execution), flush=True)


@torch.no_grad()
def evaluate_predictions(model, cache, indices, batch_size, intervention=None):
    model.eval()
    predictions = []
    for offset in range(0, len(indices), batch_size):
        observations = cache.minibatch(indices[offset:offset+batch_size])
        predictions.append(prediction_from_minibatch(model, observations, intervention=intervention).cpu())
    predicted = torch.cat(predictions)
    targets = torch.as_tensor(np.array(cache.arrays["trajectory"][indices]))
    distances = (predicted[..., :2]-targets[..., :2]).norm(dim=-1)
    return predicted.numpy(), {"loss": float(official_loss(predicted, targets)),
        "ade_meters": float(distances.mean()), "fde_meters": float(distances[:, -1].mean())}


def train(configuration, output, condition):
    assert read_json(output / "verification.json")["passed"]
    setup_official()
    gpu_setup(configuration)
    execution = read_json(output / "execution.json")
    assert execution["source_sha256"] == scientific_source_hashes(), "Scientific source changed after profiling"
    assert sha256(output / "initial_planner.pt") == execution["initial_planner_sha256"]
    assert sha256(output / "training_schedule.npy") == execution["schedule_sha256"]
    folder = output / condition
    folder.mkdir(exist_ok=True)
    assert not (folder / "training_complete.json").exists(), "Completed experiment must not be rerun in place"
    cache = EncoderFeatureCache(output)
    model = model_from_initial(output, condition, "cuda")
    initial_digest = state_digest(model.planner)
    assert not any("image_encoder" in name for name, _ in model.named_modules())
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"])
    schedule = np.load(output / "training_schedule.npy")
    validation_indices = cache.indices("validation")
    _, initial_validation = evaluate_predictions(model, cache, validation_indices, execution["common_batch_size"])
    model.train()
    seed_everything(configuration["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses, step_seconds, gradient_norms = [], [], []
    maximum_card = check_card(configuration)
    with (folder / "training.jsonl").open("w") as stream:
        for step, indices in enumerate(schedule, 1):
            begin_step = time.perf_counter()
            observations = cache.minibatch(indices)
            optimizer.zero_grad(set_to_none=True)
            prediction = prediction_from_minibatch(model, observations)
            loss = official_loss(prediction, observations["trajectory"])
            loss.backward()
            assert torch.isfinite(loss)
            gradient = norm_of_gradients(model.importance.parameters()) if model.importance is not None else None
            assert gradient is None or (np.isfinite(gradient) and gradient > 0)
            finite_gradients = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
            assert torch.stack(finite_gradients).all()
            optimizer.step()
            torch.cuda.synchronize()
            seconds = time.perf_counter()-begin_step
            losses.append(float(loss)); step_seconds.append(seconds)
            if gradient is not None:
                gradient_norms.append(gradient)
            row = {"step": step, "loss": float(loss), "seconds": seconds, "importance_gradient_norm": gradient,
                   "beta": float(model.importance.beta) if model.importance is not None else None}
            stream.write(json.dumps(row) + "\n")
            if step == 1 or step % 50 == 0 or step == len(schedule):
                stream.flush()
                maximum_card = max(maximum_card, check_card(configuration))
                row.update(condition=condition, total_steps=len(schedule),
                    elapsed_training_seconds=time.perf_counter()-started,
                    whole_card_used_bytes=maximum_card)
                write_json(folder / "progress.json", row)
                print(json.dumps(row), flush=True)
    training_seconds = time.perf_counter()-started
    torch.save({"model": model.cpu().state_dict(), "optimizer": optimizer.state_dict(),
                "condition": condition, "steps": len(schedule), "seed": configuration["seed"]}, folder / "checkpoint.pt")
    model.cuda()
    _, validation = evaluate_predictions(model, cache, validation_indices, execution["common_batch_size"])
    result = {"complete": True, "condition": condition, "steps": len(schedule),
        "batch_size": execution["common_batch_size"], "seed": configuration["seed"],
        "training_seconds": training_seconds, "gpu_hours": training_seconds / 3600,
        "mean_step_seconds": float(np.mean(step_seconds)), "p95_step_seconds": float(np.percentile(step_seconds, 95)),
        "train_loss_last100_mean": float(np.mean(losses[-100:])), "initial_validation": initial_validation,
        "validation": validation, "initial_native_planner_state_sha256": initial_digest,
        "initial_checkpoint_sha256": execution["initial_planner_sha256"], "schedule_sha256": execution["schedule_sha256"],
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "native_planner_trainable_parameters": sum(p.numel() for p in model.planner.parameters() if p.requires_grad),
        "importance_trainable_parameters": sum(p.numel() for p in model.importance.parameters()) if model.importance is not None else 0,
        "importance_gradient_mean": float(np.mean(gradient_norms)) if gradient_norms else None,
        "beta": float(model.importance.beta) if model.importance is not None else None,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "maximum_whole_card_bytes": maximum_card, "encoder_forward_calls": 0}
    assert execution["source_sha256"] == scientific_source_hashes()
    result["source_sha256"] = scientific_source_hashes()
    write_json(folder / "training_complete.json", result)
    print(json.dumps(result), flush=True)


def predict(configuration, output, condition):
    setup_official()
    gpu_setup(configuration)
    started = time.perf_counter()
    cache = EncoderFeatureCache(output)
    execution = read_json(output / "execution.json")
    folder = output / condition
    model = model_from_initial(output, condition, "cuda")
    checkpoint = torch.load(folder / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    indices = cache.indices("validation")
    modes = ["normal", "beta_zero", "shuffle"] if condition == "conditioned" else ["normal"]
    summaries = {}
    for mode in modes:
        predictions, summary = evaluate_predictions(model, cache, indices,
            execution["common_batch_size"], intervention=None if mode == "normal" else mode)
        np.savez(folder / ("predictions_" + mode + ".npz"), trajectories=predictions,
                 tokens=np.array([cache.records[index]["token"] for index in indices]))
        summaries[mode] = summary
    # Attention and importance use the same unchanged features and real ego input.
    attention_rows, importance_rows, sensitivity_rows = [], [], []
    with torch.no_grad():
        for offset in range(0, len(indices), execution["common_batch_size"]):
            observations = cache.minibatch(indices[offset:offset + execution["common_batch_size"]])
            prediction_from_minibatch(model, observations, capture_attention=True)
            attention_rows.append(model.last_attention.mean(dim=(1, 2)).cpu().numpy())
            importance_rows.append(model.last_importance.cpu().numpy())
            if model.importance is not None:
                real_status = observations["ego_status"] if condition == "conditioned" else torch.zeros_like(observations["ego_status"])
                alternatives = real_status.clone()
                alternatives[:, :4] = alternatives[:, :4].roll(1, -1)
                alternate = model.importance(observations["visual_tokens"], observations["valid_token_mask"], alternatives, model.positions)
                sensitivity_rows.append((alternate-model.last_importance).cpu().numpy())
        # Measure equal-batch decoder-only latency, excluding disk/cache transfer.
        observations = cache.minibatch(indices[:execution["common_batch_size"]])
        for _ in range(5):
            prediction_from_minibatch(model, observations)
        timings = []
        for _ in range(50):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            prediction_from_minibatch(model, observations)
            torch.cuda.synchronize()
            timings.append(time.perf_counter()-begin)
    attention_values, importance_values = np.concatenate(attention_rows), np.concatenate(importance_rows)
    beta = float(model.importance.beta) if model.importance is not None else 0.0
    logits = beta * importance_values
    weights = torch.tensor(logits).softmax(-1).numpy()
    normalized_entropy = -(weights * np.log(np.maximum(weights, 1e-30))).sum(-1) / np.log(weights.shape[-1])
    np.savez(folder / "token_diagnostics.npz", importance=importance_values, attention=attention_values,
             importance_probabilities=weights, command_importance_delta=np.concatenate(sensitivity_rows) if sensitivity_rows else np.zeros_like(importance_values))
    diagnostics = {"beta": beta, "importance_quantiles": np.quantile(importance_values, [0, .01, .5, .99, 1]).tolist(),
        "bias_quantiles": np.quantile(logits, [0, .01, .5, .99, 1]).tolist(),
        "mean_importance_std_within_scene": float(importance_values.std(-1).mean()),
        "mean_normalized_importance_entropy": float(normalized_entropy.mean()),
        "mean_attention_on_ego_token": float(attention_values[:, -1].mean()),
        "command_only_counterfactual_mean_absolute_importance_change": float(np.abs(np.concatenate(sensitivity_rows)).mean()) if sensitivity_rows else None,
        "command_only_counterfactual_relative_to_importance_std": float(np.abs(np.concatenate(sensitivity_rows)).mean() / max(1e-12, importance_values.std(-1).mean())) if sensitivity_rows else None,
        "counterfactual_scope": "importance module only, cyclic permutation of one-hot command; original decoder ego remains real",
        "decoder_batch_size": execution["common_batch_size"],
        "decoder_latency_mean_ms": float(np.mean(timings) * 1000),
        "decoder_latency_median_ms": float(np.median(timings) * 1000),
        "decoder_latency_p95_ms": float(np.percentile(timings, 95) * 1000),
        "latency_on_shared_gpu": True, "encoder_forward_calls": 0}
    write_json(folder / "prediction_complete.json", {"complete": True, "conditions": summaries,
        "diagnostics": diagnostics, "prediction_and_diagnosis_seconds": time.perf_counter()-started})
    print(json.dumps({"condition": condition, "prediction_complete": True, "diagnostics": diagnostics}), flush=True)


def score(configuration, output, condition):
    # Invoke this stage in the preserved official evaluation Python environment.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    from score_lpwm_drivor_navtest import score_scene
    started = time.perf_counter()
    manifest = read_json(output / "subset_manifest.json")
    selected = [row for row in manifest["records"] if row["pilot_split"] == "validation"]
    folder = output / condition
    prediction_metadata = read_json(folder / "prediction_complete.json")
    status = np.load(output / "cache/ego_status.npy", mmap_mode="r")
    commands = [int(status[row["encoder_cache_row"], :4].argmax()) for row in selected]
    speeds = [float(np.linalg.norm(status[row["encoder_cache_row"], 4:6])) for row in selected]
    results = {}
    for mode in prediction_metadata["conditions"]:
        predictions = np.load(folder / ("predictions_" + mode + ".npz"))
        assert predictions["tokens"].tolist() == [row["token"] for row in selected]
        jobs = [(row["token"], trajectory, row["metric_cache_file"]) for row, trajectory in zip(selected, predictions["trajectories"])]
        with ProcessPoolExecutor(max_workers=configuration["scoring_workers"], mp_context=multiprocessing.get_context("spawn")) as pool:
            rows = list(pool.map(score_scene, jobs, chunksize=8))
        assert len(rows) == len(selected) and all(row["valid"] for row in rows)
        with (folder / ("scores_" + mode + ".csv")).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in {"token", "log_name", "valid"}}
        scenarios = {}
        for command, label in enumerate(("left", "straight", "right", "other")):
            command_rows = [row for row, value in zip(rows, commands) if value == command]
            if command_rows:
                scenarios[label] = {"count": len(command_rows), "pdms": float(np.mean([row["score"] for row in command_rows])*100)}
        for label, selector in (("speed_below_0.5_mps", lambda speed: speed < .5),
                                ("speed_at_least_0.5_mps", lambda speed: speed >= .5)):
            scenario_rows = [row for row, speed in zip(rows, speeds) if selector(speed)]
            if scenario_rows:
                scenarios[label] = {"count": len(scenario_rows), "pdms": float(np.mean([row["score"] for row in scenario_rows])*100)}
        results[mode] = {"count": len(rows), "failed": 0, "pdms": metrics["score"]*100,
            "metrics": metrics, "scenarios": scenarios, **prediction_metadata["conditions"][mode]}
        print(json.dumps({"condition": condition, "mode": mode, "pdms": results[mode]["pdms"]}), flush=True)
    write_json(folder / "evaluation_complete.json", {"complete": True, "results": results,
        "benchmark": "NAVSIM v1 official PDMS on fixed navval subset", "full_benchmark": False,
        "epdms": "not applicable to this NAVSIM v1 scorer", "intersection_labels": "unavailable; not fabricated",
        "scoring_seconds": time.perf_counter()-started})


def latency(configuration, output):
    """Interleave all three cached decoders to reduce shared-GPU timing drift."""
    setup_official()
    gpu_setup(configuration)
    cache = EncoderFeatureCache(output)
    execution = read_json(output / "execution.json")
    observations = cache.minibatch(cache.indices("validation")[:execution["common_batch_size"]])
    models = {}
    for condition in configuration["conditions"]:
        model = model_from_initial(output, condition, "cuda")
        model.load_state_dict(torch.load(output / condition / "checkpoint.pt", map_location="cpu")["model"], strict=True)
        models[condition] = model.eval()
    times = {condition: [] for condition in models}
    with torch.no_grad():
        for model in models.values():
            for _ in range(5):
                prediction_from_minibatch(model, observations)
        for iteration in range(60):
            conditions = configuration["conditions"]
            order = conditions[iteration % 3:] + conditions[:iteration % 3]
            for condition in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction_from_minibatch(models[condition], observations)
                torch.cuda.synchronize()
                times[condition].append(1000*(time.perf_counter()-started))
    reference = np.array(times["baseline"])
    result = {"batch_size": execution["common_batch_size"], "rounds": 60,
        "order": "rotated A/B/C, shared GPU0, resident inputs, forward only, no encoder",
        "milliseconds": times, "summary": {}}
    for condition, values in times.items():
        result["summary"][condition] = {"mean_ms": float(np.mean(values)), "median_ms": float(np.median(values)),
            "mean_paired_extra_ms_vs_A": float(np.mean(np.array(values)-reference)),
            "ratio_of_means_vs_A": float(np.mean(values)/reference.mean())}
    write_json(output / "paired_latency.json", result)
    print(json.dumps(result["summary"]), flush=True)


def paired_recording_interval(differences, records):
    groups = sorted({row["recording_group"] for row in records})
    totals = np.array([np.sum([value for value, row in zip(differences, records) if row["recording_group"] == group]) for group in groups])
    counts = np.array([sum(row["recording_group"] == group for row in records) for group in groups])
    draws = np.random.default_rng(0).integers(0, len(groups), (10000, len(groups)))
    means = totals[draws].sum(-1) / counts[draws].sum(-1)
    return {"mean_difference_points": float(np.mean(differences)),
            "recording_bootstrap_95_interval": np.quantile(means, [.025, .975]).tolist(),
            "recording_count": len(groups), "bootstrap_seed": 0, "bootstrap_resamples": 10000}


def report(configuration, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import shutil
    results_directory = PROJECT_ROOT / configuration["results_directory"]
    results_directory.mkdir(parents=True, exist_ok=True)
    manifest = read_json(output / "subset_manifest.json")
    records = [row for row in manifest["records"] if row["pilot_split"] == "validation"]
    execution = read_json(output / "execution.json")
    result = {"configuration": configuration, "execution": execution,
              "verification": read_json(output / "verification.json"),
              "cache_metadata": read_json(output / "cache/metadata.json"), "models": {}}
    scores = {}
    for condition in configuration["conditions"]:
        folder = output / condition
        training = read_json(folder / "training_complete.json")
        evaluation = read_json(folder / "evaluation_complete.json")
        prediction = read_json(folder / "prediction_complete.json")
        result["models"][condition] = {"training": training, "evaluation": evaluation, "prediction": prediction,
            "checkpoint_sha256": sha256(folder / "checkpoint.pt")}
        for mode in evaluation["results"]:
            score_path = folder / ("scores_" + mode + ".csv")
            with score_path.open() as stream:
                score_rows = list(csv.DictReader(stream))
            scores[(condition, mode)] = np.array([float(row["score"])*100 for row in score_rows])
            assert [row["token"] for row in score_rows] == [row["token"] for row in records]
            shutil.copy2(score_path, results_directory / (condition + "_" + score_path.name))
    training_values = [row["training"] for row in result["models"].values()]
    for name in ("steps", "batch_size", "seed", "initial_native_planner_state_sha256", "schedule_sha256", "native_planner_trainable_parameters"):
        assert len({row[name] for row in training_values}) == 1, name
    assert result["models"]["unconditioned"]["training"]["importance_trainable_parameters"] == result["models"]["conditioned"]["training"]["importance_trainable_parameters"]
    training_total = sum(row["training_seconds"] for row in training_values)
    assert training_total <= configuration["maximum_training_gpu_seconds"], training_total
    comparisons = {}
    for label, variant, reference in (
        ("B_minus_A", ("unconditioned", "normal"), ("baseline", "normal")),
        ("C_minus_A", ("conditioned", "normal"), ("baseline", "normal")),
        ("C_minus_B", ("conditioned", "normal"), ("unconditioned", "normal")),
        ("C_beta_zero_minus_C", ("conditioned", "beta_zero"), ("conditioned", "normal")),
        ("C_shuffle_minus_C", ("conditioned", "shuffle"), ("conditioned", "normal"))):
        comparisons[label] = paired_recording_interval(scores[variant]-scores[reference], records)
    result["comparisons"] = comparisons
    result["training_total_gpu_hours"] = training_total / 3600
    result["paired_latency"] = read_json(output / "paired_latency.json")
    result["cost_definition"] = "One GPU per process; elapsed training residency hours on a shared GPU, including CPU minibatch/diagnostics; not isolated GPU kernel time. Cache/evaluation excluded."
    result["limits"] = ["Single seed0, small previously used navval development subset, no independent final test",
        "Full planning checkpoint initializes encoder AND original planner; this is continued adaptation, not training from scratch",
        "128 grid indices are spatial anchors; ViT and original planner encoder mix information across them",
        "No hard selection/compression and unchanged token count; no computational reduction claim",
        "NAVSIM v1 PDMS; EPDMS not available in this evaluator",
        "Shared-GPU latency has interference and is not an isolated deployment benchmark"]
    # Source provenance at execution completion; all relevant inputs included.
    sources = [PROJECT_ROOT / "scripts/run_soft_token_reweighting.py", PROJECT_ROOT / "scripts/run_soft_token_reweighting_suite.py",
        PROJECT_ROOT / "src/planning_aware_future_prediction/models/soft_token_reweighting.py",
        PROJECT_ROOT / "configs/soft_token_reweighting/pilot_v1.json"]
    result["source_sha256"] = {str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in sources}
    write_json(results_directory / "results.json", result)
    for filename in ("configuration.json", "execution.json", "verification.json"):
        shutil.copy2(output / filename, results_directory / filename)
    manifest_fields = ("token", "log_name", "recording_group", "split", "pilot_split",
                       "encoder_cache_row", "source_cache_row", "observed_offsets_seconds")
    write_json(results_directory / "subset_manifest.json", {
        "records": [{key: row[key] for key in manifest_fields} for row in manifest["records"]],
        "counts": manifest["counts"], "full_local_manifest": str(output / "subset_manifest.json"),
        "full_manifest_sha256": sha256(output / "subset_manifest.json"),
        "source_image_manifest": configuration["source_image_manifest"],
        "recording_overlap": 0})
    shutil.copy2(output / "cache/metadata.json", results_directory / "cache_metadata.json")
    summary_rows = []
    for condition, model in result["models"].items():
        for mode, values in model["evaluation"]["results"].items():
            training = model["training"]
            summary_rows.append({"condition": condition, "intervention": mode,
                "pdms": values["pdms"], "ade_meters": values["ade_meters"], "fde_meters": values["fde_meters"],
                "validation_loss": values["loss"], "trainable_parameters": training["trainable_parameters"],
                "steps": training["steps"], "batch_size": training["batch_size"],
                "training_gpu_hours": training["gpu_hours"], "peak_allocated_bytes": training["peak_allocated_bytes"],
                "decoder_batch_latency_ms": result["paired_latency"]["summary"][condition]["mean_ms"]})
    with (results_directory / "summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0])); writer.writeheader(); writer.writerows(summary_rows)
    # Curves and official paired scores, all same training schedule.
    colors = {"baseline": "#51657c", "unconditioned": "#d88824", "conditioned": "#257f68"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for condition, model in result["models"].items():
        logs = [json.loads(line) for line in (output / condition / "training.jsonl").read_text().splitlines()]
        losses = np.array([row["loss"] for row in logs]); window = min(25, len(losses))
        axes[0].plot(np.arange(window, len(losses)+1), np.convolve(losses, np.ones(window)/window, mode="valid"),
            label=condition, color=colors[condition])
        if condition != "baseline":
            axes[1].plot([row["step"] for row in logs], [row["beta"] for row in logs], label=condition, color=colors[condition])
    axes[0].set(title="Training loss (25-step moving mean)", xlabel="Optimizer update", ylabel="Official weighted L1")
    axes[0].legend(); axes[1].set(title="Learned bias scale", xlabel="Optimizer update", ylabel="beta"); axes[1].legend()
    values = [result["models"][condition]["evaluation"]["results"]["normal"]["pdms"] for condition in configuration["conditions"]]
    axes[2].bar(["A", "B", "C"], values, color=list(colors.values()))
    axes[2].set(title="Official PDMS / 1,000 navval scenes", ylabel="PDMS (0-100)", ylim=(0, 100))
    for index, value in enumerate(values):
        axes[2].text(index, value+1, f"{value:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(results_directory / "learning_and_pdms.png", dpi=160); plt.close(fig)
    # Deterministic first held-out example of each available ego command.
    status = np.load(output / "cache/ego_status.npy", mmap_mode="r")
    commands = [int(status[row["encoder_cache_row"], :4].argmax()) for row in records]
    example_indices = [commands.index(command) for command in (0, 1, 2) if command in commands]
    diagnostics = {condition: np.load(output / condition / "token_diagnostics.npz") for condition in configuration["conditions"]}
    fig, axes = plt.subplots(len(example_indices), 5, figsize=(20, 3.8*len(example_indices)), squeeze=False)
    attention_limit = max(float(np.quantile(diagnostics[condition]["attention"][example_indices, :128], .99)) for condition in configuration["conditions"])
    probability_limit = max(float(np.quantile(diagnostics[condition]["importance_probabilities"][example_indices], .99)) for condition in ("unconditioned", "conditioned"))
    for row_index, example_index in enumerate(example_indices):
        record = records[example_index]
        original_image = np.asarray(Image.open(record["observed_front_paths"][-1]).convert("RGB"))
        image_height, image_width = original_image.shape[:2]
        for column, condition, key, title in ((0, "baseline", "attention", "A: first cross-attention"),
            (1, "unconditioned", "importance_probabilities", "B: softmax(beta*r)"),
            (2, "unconditioned", "attention", "B: first cross-attention"),
            (3, "conditioned", "importance_probabilities", "C: softmax(beta*r)"),
            (4, "conditioned", "attention", "C: first cross-attention")):
            axis = axes[row_index, column]
            axis.imshow(original_image)
            grid = diagnostics[condition][key][example_index, :128].reshape(8,16)
            axis.imshow(grid, extent=(0, image_width, image_height-28, 28), origin="upper", alpha=.55,
                cmap="magma", vmin=0, vmax=attention_limit if key == "attention" else probability_limit,
                interpolation="nearest", aspect="auto")
            axis.set(xlim=(0,image_width), ylim=(image_height,0)); axis.axis("off")
            axis.set_title(title + "\n" + ["left", "straight", "right", "other"][commands[example_index]])
    fig.suptitle("Fixed navval examples: grid anchors, not detected objects\nAttention averaged over all 8 heads and 8 waypoint queries; ego memory omitted", fontsize=14)
    fig.tight_layout(rect=(0, .02, 1, .94)); fig.savefig(results_directory / "importance_and_attention.png", dpi=150); plt.close(fig)
    write_json(results_directory / "visualization_examples.json", {"selection": "first manifest entry for each command, independent of predictions",
        "tokens": [records[index]["token"] for index in example_indices], "attention_color_max": attention_limit,
        "importance_probability_color_max": probability_limit,
        "mapping": "Native pooled8x16 grid mapped to cropped original image. Global attention contextualizes these anchors; not object masks."})
    lines = ["# Frozen-token soft reweighting pilot", "", "단일 seed 0, 공식 navtrain 10,000장면 / navval 1,000장면(61개 recording)의 부분 개발셋 실험.",
        "이 실험은 고정된 시각 표현에 상황·주행 지시 조건부 중요도를 주는 것이 기존 decoder attention보다 planning에 도움이 되는지 묻는다.", "",
        "## 실제 결과", "", "| 모델 | PDMS | ADE m | FDE m | 검증 loss | 학습 GPU h | 추가 파라미터 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, model in result["models"].items():
        values = model["evaluation"]["results"]["normal"]; training = model["training"]
        lines.append(f"| {condition} | {values['pdms']:.4f} | {values['ade_meters']:.4f} | {values['fde_meters']:.4f} | {values['loss']:.5f} | {training['gpu_hours']:.4f} | {training['importance_trainable_parameters']:,} |")
    lines += ["", "| 대응 비교 | PDMS 차이 | recording bootstrap 95% 구간 |", "|---|---:|---|"]
    for label, comparison in comparisons.items():
        lines.append(f"| {label} | {comparison['mean_difference_points']:+.4f} | {comparison['recording_bootstrap_95_interval']} |")
    lines += ["", "## 입력·구현·통제", "",
        "공개 Drive-JEPA perception-free ViT-L의 full planning checkpoint(encoder와 planner)를 초기값으로 재사용했다. proposal-centric 모델이 아니다. Encoder는 eval/frozen이며, 관측 전방카메라 과거·현재(-0.5,0초) 2프레임만 쓴다. 상하28px crop, OpenCV linear512×256, ImageNet 정규화를 유지했다.",
        "Encoder의 B×512×1024 출력에 원래 AvgPool2d2를 적용한 B×128×1024 FP32 feature를 저장했다. 이번 방법이 추가로 token을 줄이지 않는다. 메모리는128개 시각 grid와1개 ego token이다. 원래 image_fc, 위치 embedding, planner Transformer encoder3층/decoder3층, trajectory head 전체를 A/B/C 공통으로 학습한다. 이 planner 내부 encoder는 동결한 시각 ViT와 다른 모듈이다.",
        "중요도는 LN(feature), valid-token 장면 평균, ego8→32 embedding, 좌표xy/camera/time을128hidden MLP에 넣어 계산한다. 첫 decoder cross-attention의 float mask(B×8,8,129)에 beta*r를 더한다. 모든 head/query에 공유하고 ego token bias는0, padding은-inf다. K/V, residual, 나머지 attention은 유지한다. 초기 beta=.1, 출력층 std=.001. Valid-token 평균을 빼서 상수 offset을 제거한다.",
        "B는 중요도 모듈에만 ego/명령0을 주며 native decoder에는 실제 ego를 준다. B/C 구조·파라미터 수와 초기 상태가 같다. A/B/C 모두 동일 pretrained planner와 seed0 batch schedule, Adam lr1e-4/기본 betas/weight_decay0, 공식 length-normalized L1(alpha5), dropout0, FP32를 쓴다. 추가 loss·regularizer·future module은 없다.",
        f"공통 batch{execution['common_batch_size']} × {execution['common_steps']}update. Profile 가중치는 버리고 원 초기값에서 재시작했다. 학습 총 {training_total/3600:.4f} GPU h, 3h 상한 이내.",
        "전체 split의 scene/log/timestamp를 확인했고 train101개 recording과 val61개 recording 중복0이다. 공식 log split을 유지한다. 데이터 규모 축소와 과거 navval 사용 이력, 공개 checkpoint의 validation 선택 가능성 때문에 독립 최종 test로 부르지 않는다.",
        "", "## 검증", "", "verification.json의 자동 gate가 통과했다. beta0 출력/loss 동등성, finite/nonzero 중요도·planner gradient, encoder frozen/no-gradient와 cached step의0 encoder call, padding확률0/내용불변, 실제MHA 합성 bias 단조증가, raw공식 feature builder/ego/GT와 캐시 일치가 포함된다.",
        "", "## 비용 및 중요도 진단", "", "| 모델 | step초 | decoder batch ms | peak allocated GB | beta | 중요도 entropy/log128 | command 변경 상대효과 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition, model in result["models"].items():
        training = model["training"]; diagnostic = model["prediction"]["diagnostics"]
        paired_ms = result["paired_latency"]["summary"][condition]["mean_ms"]
        lines.append(f"| {condition} | {training['mean_step_seconds']:.4f} | {paired_ms:.3f} | {training['peak_allocated_bytes']/1e9:.3f} | {diagnostic['beta']:.6f} | {diagnostic['mean_normalized_importance_entropy']:.6f} | {diagnostic['command_only_counterfactual_relative_to_importance_std']} |")
    lines += ["", f"캐시 생성 {result['cache_metadata']['cache_seconds']:.1f}초, 데이터/manifest 준비 {manifest['preparation_seconds']:.1f}초. 공식 채점 총 {sum(model['evaluation']['scoring_seconds'] for model in result['models'].values()):.1f}초, 예측/진단 총 {sum(model['prediction']['prediction_and_diagnosis_seconds'] for model in result['models'].values()):.1f}초. 학습 GPU 시간에서 분리했다. GPU0 공유 상태 실측으로 간섭이 있는 latency이며 속도 향상으로 해석하지 않는다. Token 수가 그대로여서 연산량 감소를 주장하지 않는다.",
        "", "## 상황별 결과", "", "| 상황 | 표본 | A | B | C |", "|---|---:|---:|---:|---:|"]
    scenarios = result["models"]["baseline"]["evaluation"]["results"]["normal"]["scenarios"]
    for scenario, value in scenarios.items():
        measurements = [result["models"][condition]["evaluation"]["results"]["normal"]["scenarios"][scenario]["pdms"] for condition in configuration["conditions"]]
        lines.append(f"| {scenario} | {value['count']} | " + " | ".join(f"{value:.4f}" for value in measurements) + " |")
    lines += ["", "좌/직진/우는 실제 ego command를 사용했다. 정지는 현재속도<0.5m/s라는 관측 기준으로만 표시했다. 신뢰 가능한 교차로 라벨은 없어 생성하지 않았다. 하위 NC/DAC/EP/TTC/comfort/DDC는 results.json 및 모델별 scores CSV에 보존했다.",
        "", "## PDMS 하위 지표 (0–100 환산)", "", "| 지표 | A | B | C |", "|---|---:|---:|---:|"]
    for metric in ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance"):
        measurements = [result["models"][condition]["evaluation"]["results"]["normal"]["metrics"][metric]*100 for condition in configuration["conditions"]]
        lines.append(f"| {metric} | " + " | ".join(f"{value:.4f}" for value in measurements) + " |")
    lines += [
        "", "## 해석과 다음 실험", ""]
    lower, upper = comparisons["C_minus_A"]["recording_bootstrap_95_interval"]
    if lower > 0:
        lines.append("이 부분 개발셋에서 C가 A보다 높은 PDMS를 보였고 기록단위 구간도0위다. 단일 seed와 기존 개발셋 결과이므로 일반적인 성능 향상이나 객체의 인과적 중요도 발견을 입증하지 않는다.")
    elif upper < 0:
        lines.append("이 조건에서는 C가 A보다 낮은 PDMS를 보였다. 현재 pilot은 조건부 중요도 추가가 유리하다는 가설을 지지하지 않는다.")
    else:
        lines.append("C−A의 기록단위 구간이0을 포함한다. 이 pilot으로 조건부 reweighting의 성능 향상을 확정할 수 없다. 평균차이와 seed간 불확실성을 구분해야 한다.")
    lines += ["C−B는 ego조건의 추가 이득을, C의 beta0/valid-token shuffle은 학습된 bias 사용 여부를 검사한다. 이 개입은 재학습 대조가 아니며 분포 변화 효과를 포함한다. 높은 entropy는 균일한 bias, 작은 command상대효과는 약한 명령 의존을 시사하며 객체 이해의 증거가 아니다.",
        "다음 실험 하나: 방법·데이터·batch·step·학습률을 고정한 A/B/C의 추가2seed 반복으로 효과의 방향과 분산을 확인한다. 이번 작업에서는 자동 실행하지 않았다.",
        "", "![Learning and PDMS](learning_and_pdms.png)", "", "![Importance and attention](importance_and_attention.png)",
        "", "시각화는 미리 고정된 명령별 첫 validation 장면이다. 위치는8×16 grid anchor이며 ViT와 planner encoder의 전역 정보 혼합 때문에 해당 픽셀만의 인과적 중요도로 해석하지 않는다.",
        "", "## 재실행", "", "완료 결과를 덮어쓰지 않도록 config의 output_directory/results_directory를 새 경로로 바꾼 후 실행한다.", "", "```bash",
        "cd /rhome/junseong/PlanningAwareFuturePrediction",
        "/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python scripts/run_soft_token_reweighting_suite.py --config configs/soft_token_reweighting/pilot_v1.json --replay-id rerun_001",
        "```", "", "캐시 및 checkpoint는 outputs/soft_token_reweighting_v1, 공유 수치/검증/manifest/그림은 results/soft_token_reweighting_v1에 있다."]
    (results_directory / "report.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"complete": True, "training_gpu_hours": training_total/3600, "results": str(results_directory)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "cache", "verify", "profile", "train", "predict", "score", "latency", "report"])
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/soft_token_reweighting/pilot_v1.json")
    parser.add_argument("--condition", choices=["baseline", "unconditioned", "conditioned"])
    arguments = parser.parse_args()
    configuration = read_json(arguments.config.resolve())
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    if arguments.stage == "prepare":
        prepare(configuration, output)
    elif arguments.stage == "cache":
        cache_features(configuration, output)
    elif arguments.stage == "verify":
        verify(configuration, output)
    elif arguments.stage == "profile":
        profile(configuration, output)
    elif arguments.stage == "train":
        train(configuration, output, arguments.condition)
    elif arguments.stage == "predict":
        predict(configuration, output, arguments.condition)
    elif arguments.stage == "score":
        score(configuration, output, arguments.condition)
    elif arguments.stage == "latency":
        latency(configuration, output)
    elif arguments.stage == "report":
        report(configuration, output)
    else:
        raise NotImplementedError(arguments.stage)


if __name__ == "__main__":
    main()
