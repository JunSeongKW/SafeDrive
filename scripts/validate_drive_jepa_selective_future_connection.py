"""Bounded real-image connection gate; no sweep or benchmark evaluation."""

import argparse
import contextlib
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def parameter_sha256(module):
    checksum = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        checksum.update(name.encode())
        checksum.update(str(tuple(tensor.shape)).encode())
        checksum.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return checksum.hexdigest()


def gradient_norm(module):
    return (
        sum(
            float(parameter.grad.detach().square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        ** 0.5
    )


def gradient_report(extension):
    return {
        "selector": gradient_norm(extension.patch_selector),
        "predictor": gradient_norm(extension.future_predictor),
        "bridge": gradient_norm(extension.future_bridge),
        "original_model": gradient_norm(extension.baseline_model),
    }


def eligible_train_segments(official_segments, split_by_recording):
    """Official navtrain membership alone does NOT exclude project dev/held-out."""
    return sorted(
        segment
        for segment in official_segments
        if split_by_recording.get(segment.rsplit("_", 2)[0]) == "train"
    )


def load_train_windows(workspace, official_root, agent, specification):
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader

    filter_path = (
        official_root
        / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
    )
    train_filter = yaml.safe_load(filter_path.read_text())
    split_manifest = json.loads(
        (workspace / specification["project_recording_split_manifest"]).read_text()
    )
    split_assignments = split_manifest["assignments"]["split_by_recording"]
    if specification["required_project_recording_split"] != "train":
        raise RuntimeError("Development/held-out diagnostic fitting forbidden")
    log_root = workspace / "dataset/navsim_logs/trainval"
    sensor_root = workspace / "dataset/sensor_blobs/trainval"
    chosen_recordings, windows, source_paths = set(), [], set()
    excluded_current_missing = []
    for segment in eligible_train_segments(
        train_filter["log_names"], split_assignments
    ):
        recording = segment.rsplit("_", 2)[0]
        log_path = log_root / f"{segment}.pkl"
        if recording in chosen_recordings or not log_path.is_file():
            continue
        loader = SceneLoader(
            log_root,
            sensor_root,
            SceneFilter(
                num_history_frames=4,
                num_future_frames=8,
                frame_interval=12,
                has_route=False,
                log_names=[segment],
            ),
            agent.get_sensor_config(),
        )
        for token in loader.tokens:
            frames = loader.scene_frames_dicts[token]
            camera_paths = [
                sensor_root
                / next(
                    value["data_path"]
                    for name, value in frame["cams"].items()
                    if name.lower() == "cam_f0"
                )
                for frame in frames
            ]
            if not all(camera_paths[index].is_file() for index in (2, 3)):
                excluded_current_missing.append(token)
                continue
            timestamps = [int(frame["timestamp"]) for frame in frames]
            intervals = np.diff(timestamps) / 1e6
            if np.any(
                np.abs(intervals - specification["frame_interval_seconds"])
                > specification["timestamp_tolerance_seconds"]
            ):
                raise RuntimeError("Unexpected cadence; do not silently resample")
            # Current input builder accesses history only; future images below are target-only.
            agent_input = loader.get_agent_input_from_token(token)
            features = agent.get_feature_builders()[0].compute_features(agent_input)
            clip = torch.stack(
                (features["camera_feature_2"], features["camera_feature_1"]), dim=1
            )
            scene = loader.get_scene_from_token(token)
            trajectory = agent.get_target_builders()[0].compute_targets(scene)[
                "trajectory"
            ]
            future_clips, future_valid = [], []
            for pair in specification["future_frame_pairs"]:
                valid = all(camera_paths[index].is_file() for index in pair)
                future_valid.append(valid)
                if valid:
                    cameras = [
                        SimpleNamespace(
                            cam_f0=SimpleNamespace(
                                image=np.array(
                                    Image.open(camera_paths[index]).convert("RGB")
                                )
                            )
                        )
                        for index in pair
                    ]
                    latest, earlier = agent.get_feature_builders()[
                        0
                    ]._get_camera_feature(SimpleNamespace(cameras=cameras))
                    future_clips.append(torch.stack((earlier, latest), dim=1))
                else:
                    future_clips.append(torch.zeros_like(clip))
            current_files = [camera_paths[index] for index in (2, 3)]
            source_paths.update([log_path, *current_files])
            source_paths.update(path for path in camera_paths[4:] if path.is_file())
            windows.append(
                {
                    "token": token,
                    "recording": recording,
                    "project_recording_split": split_assignments[recording],
                    "segment": segment,
                    "native_log_token": str(frames[3]["log_token"]),
                    "current_clip": clip,
                    "ego_status": features["status_feature"],
                    "ego_target": trajectory,
                    "future_clips": future_clips,
                    "future_valid": future_valid,
                    "actual_frame_intervals_seconds": intervals.tolist(),
                    "actual_future_tubelet_end_seconds": [
                        (timestamps[pair[-1]] - timestamps[3]) / 1e6
                        for pair in specification["future_frame_pairs"]
                    ],
                }
            )
            chosen_recordings.add(recording)
            break
        if len(windows) == specification["connection_recording_count"]:
            break
    if len(windows) != specification["connection_recording_count"]:
        raise RuntimeError("Insufficient current-input train recordings")
    heldout_path = (
        workspace
        / "results/official_drive_jepa_reproduction/expected_scene_tokens.json"
    )
    heldout = set(json.loads(heldout_path.read_text())["tokens"])
    if any(window["token"] in heldout for window in windows):
        raise RuntimeError("Train connection window overlaps official navtest")
    return windows, source_paths, excluded_current_missing


def measure_forward(label, operation, specification):
    for _ in range(specification["timing_warmups"]):
        with torch.no_grad():
            operation()
    durations = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(specification["timing_repeats"]):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            operation()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    return {
        "condition": label,
        "wall_seconds_per_two_window_batch": durations,
        "mean_seconds_per_batch": statistics.mean(durations),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "excludes": "sensor_disk_IO_future_teacher_encoding_official_scoring",
    }


def main():
    started = time.perf_counter()
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--specification",
        type=Path,
        default=workspace / "configs/drive_jepa_selective_future/connection_v1.json",
    )
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists():
        raise RuntimeError("Use a fresh output directory; previous runs are preserved")
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    if physical_gpu not in {"0", "1"}:
        raise RuntimeError("Use exactly one approved physical GPU0 or1")
    specification = json.loads(arguments.specification.read_text())
    free, _ = torch.cuda.mem_get_info()
    if free < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient free GPU memory; no allocation attempted")
    output_directory.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.manual_seed(specification["seed"])
    np.random.seed(specification["seed"])
    sys.path.insert(0, str(workspace / "src"))
    # Reuse configuration construction only; no writes/evaluation from the old audit utility.
    from audit_official_drive_jepa_evaluation import official_configuration

    from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
        DriveJEPASelectivePatchFuture,
    )

    baseline_spec, assets, configuration, official_root = official_configuration(
        workspace
    )
    source_root = official_root.parent
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source_root, text=True
    ).strip()
    if source_commit != baseline_spec["official_source_commit"]:
        raise RuntimeError("Official source commit differs")
    source_diff_before = subprocess.check_output(["git", "diff"], cwd=source_root)
    for field in ("planning_checkpoint", "initialization_encoder"):
        asset = assets[field]
        if (
            Path(asset["path"]).stat().st_size != asset["size_bytes"]
            or file_sha256(asset["path"]) != asset["sha256"]
        ):
            raise RuntimeError("Pinned checkpoint file differs")
    from hydra.utils import instantiate

    print("DRIVE_EXTENSION_LOAD official full checkpoint", flush=True)
    with (
        (output_directory / "official_model_loading.log").open("w") as loading_log,
        contextlib.redirect_stdout(loading_log),
    ):
        agent = instantiate(configuration.agent)
        checkpoint = torch.load(
            configuration.agent.checkpoint_path, map_location="cpu", mmap=True
        )
        full_state = {
            name.replace("agent.", ""): value
            for name, value in checkpoint["state_dict"].items()
        }
        loading = agent.load_state_dict(full_state, strict=True)
        checkpoint_metadata = {
            key: checkpoint.get(key) for key in ("epoch", "global_step")
        }
        del full_state, checkpoint
    agent.eval().cuda()
    baseline_before = parameter_sha256(agent._model)
    windows, source_paths, skipped_current = load_train_windows(
        workspace, official_root, agent, specification
    )
    input_hashes_before = {
        str(path): file_sha256(path) for path in sorted(source_paths)
    }
    metadata = [
        {
            key: window[key]
            for key in (
                "token",
                "recording",
                "project_recording_split",
                "segment",
                "native_log_token",
                "future_valid",
                "actual_frame_intervals_seconds",
                "actual_future_tubelet_end_seconds",
            )
        }
        for window in windows
    ]
    (output_directory / "pre_forward_train_windows.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    observed = torch.stack([window["current_clip"] for window in windows]).cuda()
    status = torch.stack([window["ego_status"] for window in windows]).cuda()
    ego_target = (
        torch.stack([window["ego_target"] for window in windows]).float().cuda()
    )
    extension = (
        DriveJEPASelectivePatchFuture(
            agent._model,
            latent_dim=specification["visual_latent_dim"],
            hidden_dim=specification["predictor_hidden_dim"],
            patch_budget=specification["selected_patch_budget"],
            future_tubelet_count=len(specification["future_frame_pairs"]),
            temperature=specification["selector_temperature"],
        )
        .cuda()
        .eval()
    )
    assert extension.baseline_model._transformer is agent._model._transformer
    assert extension.baseline_model._trajectory_head is agent._model._trajectory_head
    with torch.no_grad():
        original = agent._model(observed, status)["trajectory"]
        disabled = extension(observed, status, False)["trajectory"]
        initialized = extension(observed, status)["trajectory"]
    assert torch.equal(original, disabled), "Disabled branch changed baseline"
    assert torch.equal(original, initialized), "Zero residual changed inference"
    assert torch.isfinite(original).all()
    print("DRIVE_EXTENSION_PRESERVATION bitwise off/on-at-zero", flush=True)
    # Target clips are encoded one at a time, not a large teacher-video batch.
    encoded_targets = []
    with torch.no_grad():
        for window in windows:
            encoded_targets.append(
                torch.stack(
                    [
                        extension.encode_observed_clip(clip[None].cuda())[0].cpu()
                        if valid
                        else torch.zeros(512, specification["visual_latent_dim"])
                        for clip, valid in zip(
                            window["future_clips"], window["future_valid"]
                        )
                    ]
                )
            )
    future_target = torch.stack(encoded_targets).cuda()
    target_valid = torch.tensor(
        [window["future_valid"] for window in windows], device="cuda", dtype=torch.bool
    )[:, :, None].expand(-1, -1, 512)
    gradients = {}
    extension.zero_grad(set_to_none=True)
    initial_output = extension(observed, status)
    initial_loss = agent.compute_loss({}, {"trajectory": ego_target}, initial_output)
    initial_loss.backward()
    gradients["planning_at_zero_initialization"] = gradient_report(extension)
    assert gradients["planning_at_zero_initialization"]["bridge"] > 0
    assert gradients["planning_at_zero_initialization"]["selector"] == 0
    assert gradients["planning_at_zero_initialization"]["predictor"] == 0
    optimizer = torch.optim.SGD(
        [parameter for parameter in extension.parameters() if parameter.requires_grad],
        lr=specification["diagnostic_learning_rate"],
    )
    assert specification["diagnostic_optimizer_steps"] == 1
    optimizer.step()
    extension.zero_grad(set_to_none=True)
    del initial_output
    torch.cuda.reset_peak_memory_stats()
    current_output = extension(observed, status)
    planning_loss = agent.compute_loss({}, {"trajectory": ego_target}, current_output)
    planning_loss.backward()
    gradients["planning_after_one_diagnostic_step"] = gradient_report(extension)
    assert all(
        gradients["planning_after_one_diagnostic_step"][name] > 0
        for name in ("selector", "predictor", "bridge")
    )
    backward_peak = torch.cuda.max_memory_allocated()
    extension.zero_grad(set_to_none=True)
    current_output = extension(observed, status)
    auxiliary_loss = extension.compute_future_auxiliary_loss(
        current_output, status, future_target, target_valid
    )
    auxiliary_loss.backward()
    gradients["future_auxiliary"] = gradient_report(extension)
    assert gradients["future_auxiliary"]["predictor"] > 0
    assert (
        gradients["future_auxiliary"]["selector"]
        == gradients["future_auxiliary"]["bridge"]
        == 0
    )
    del current_output
    extension.zero_grad(set_to_none=True)
    normal = extension(observed, status)
    detached = extension(observed, status, detach_predicted_future=True)
    assert torch.equal(normal["trajectory"], detached["trajectory"])
    agent.compute_loss({}, {"trajectory": ego_target}, detached).backward()
    gradients["detach_predicted_future"] = gradient_report(extension)
    assert (
        gradients["detach_predicted_future"]["selector"]
        == gradients["detach_predicted_future"]["predictor"]
        == 0
    )
    assert gradients["detach_predicted_future"]["bridge"] > 0
    del normal, detached
    extension.zero_grad(set_to_none=True)
    selected_detached = extension(observed, status, detach_selection=True)
    agent.compute_loss({}, {"trajectory": ego_target}, selected_detached).backward()
    gradients["detach_selection"] = gradient_report(extension)
    assert gradients["detach_selection"]["selector"] == 0
    assert gradients["detach_selection"]["predictor"] > 0
    del selected_detached
    extension.zero_grad(set_to_none=True)
    with torch.no_grad():
        observed_output = extension(observed, status)
        selected_ids = observed_output["patch_selection"].selected_patch_indices
        assert all(
            row.unique().numel() == specification["selected_patch_budget"]
            for row in selected_ids
        )
        missing_loss = extension.compute_future_auxiliary_loss(
            observed_output,
            status,
            torch.full_like(future_target, float("nan")),
            torch.zeros_like(target_valid),
        )
        assert missing_loss.item() == 0
        assert torch.equal(
            selected_ids,
            extension(observed, status)["patch_selection"].selected_patch_indices,
        )
        assert torch.equal(original, extension(observed, status, False)["trajectory"])
    assert all(record["original_model"] == 0 for record in gradients.values())
    print("DRIVE_EXTENSION_GRADIENTS verified planning/aux/detach", flush=True)
    timing = [
        measure_forward(
            "official_original", lambda: agent._model(observed, status), specification
        ),
        measure_forward(
            "extension_disabled",
            lambda: extension(observed, status, False),
            specification,
        ),
        measure_forward(
            "extension_enabled", lambda: extension(observed, status), specification
        ),
    ]
    events = {}
    hooks = []
    for name, module in (
        ("encoder", agent._model.image_encoder),
        ("patch_selection", extension.patch_selector),
        ("future_predictor", extension.future_predictor),
        ("future_bridge", extension.future_bridge),
        ("official_transformer", agent._model._transformer),
    ):
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        events[name] = pair
        hooks.append(
            module.register_forward_pre_hook(lambda *_, event=pair[0]: event.record())
        )
        hooks.append(
            module.register_forward_hook(lambda *_, event=pair[1]: event.record())
        )
    with torch.no_grad():
        extension(observed, status)
    torch.cuda.synchronize()
    module_seconds = {
        name: start.elapsed_time(end) / 1000 for name, (start, end) in events.items()
    }
    for hook in hooks:
        hook.remove()
    baseline_after = parameter_sha256(agent._model)
    assert baseline_before == baseline_after
    assert input_hashes_before == {
        str(path): file_sha256(path) for path in sorted(source_paths)
    }
    assert source_diff_before == subprocess.check_output(
        ["git", "diff"], cwd=source_root
    )
    maximum_peak = max(
        backward_peak, *(condition["peak_allocated_bytes"] for condition in timing)
    )
    assert maximum_peak <= specification["maximum_peak_allocated_gib"] * 2**30
    elapsed = time.perf_counter() - started
    assert elapsed < specification["maximum_wall_seconds"]
    torch.save(
        {
            "specification": specification,
            "extension_state_without_original_weights": {
                name: value.cpu()
                for name, value in extension.state_dict().items()
                if not name.startswith("baseline_model.")
            },
            "optimizer_steps": 1,
            "purpose": "diagnostic_only_not_trained_model",
        },
        output_directory / "diagnostic_extension_step1.pt",
    )
    report = {
        "status": "official_real_image_connection_passed_NOT_performance_or_selection_learning",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": source_commit,
        "implementation_sha256": file_sha256(
            workspace
            / "src/planning_aware_future_prediction/models/drive_jepa_selective_patch_future.py"
        ),
        "validation_script_sha256": file_sha256(Path(__file__)),
        "environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "existing_conda_reused_read_only": True,
        },
        "existing_official_source_diff_sha256_unchanged": hashlib.sha256(
            source_diff_before
        ).hexdigest(),
        "planning_checkpoint": assets["planning_checkpoint"],
        "checkpoint_metadata": checkpoint_metadata,
        "strict_loading": {
            "missing_keys": loading.missing_keys,
            "unexpected_keys": loading.unexpected_keys,
        },
        "specification": specification,
        "specification_sha256": file_sha256(arguments.specification),
        "project_recording_split_manifest_sha256": file_sha256(
            workspace / specification["project_recording_split_manifest"]
        ),
        "previous_rejected_diagnostic": "results/drive_jepa_selective_future/project_split_exposure_audit_20261002.json",
        "train_windows": metadata,
        "skipped_for_missing_current_only": skipped_current,
        "input_shapes": {
            "observed_clip": list(observed.shape),
            "current_ego_status": list(status.shape),
            "future_target": list(future_target.shape),
            "ego_trajectory": list(original.shape),
        },
        "selected_patch_ids_after_bridge_step": selected_ids.tolist(),
        "actual_predictor_query_shape": list(
            extension.future_predictor.last_prediction_query_shape
        ),
        "original_transformer_and_head_reused": True,
        "branch_off_bitwise_equal_before_and_after_step": True,
        "branch_on_update0_bitwise_equal_under_no_grad_inference": True,
        "future_detach_forward_equal": True,
        "missing_target_loss_zero": True,
        "baseline_parameter_sha256_before": baseline_before,
        "baseline_parameter_sha256_after": baseline_after,
        "original_parameter_count": sum(
            parameter.numel() for parameter in agent._model.parameters()
        ),
        "new_trainable_parameter_count": sum(
            parameter.numel()
            for parameter in extension.parameters()
            if parameter.requires_grad
        ),
        "gradient_norms": gradients,
        "valid_future_patch_time_targets": int(target_valid.sum()),
        "valid_selected_patch_time_targets": int(
            torch.einsum(
                "bkn,btn->bkt",
                observed_output["patch_selection"].hard_selection_weights,
                target_valid.float(),
            ).sum()
        ),
        "losses_diagnostic_only": {
            "initial_planning": float(initial_loss),
            "planning_after_one_step": float(planning_loss),
            "future_auxiliary": float(auxiliary_loss),
        },
        "timing": timing,
        "enabled_module_cuda_seconds_one_shared_gpu_trial": module_seconds,
        "backward_peak_allocated_bytes": backward_peak,
        "physical_gpu": int(physical_gpu),
        "gpu_free_bytes_before_load": free,
        "total_wall_seconds": elapsed,
        "source_files_sha256_unchanged": input_hashes_before,
        "no_new_packages_or_shared_data_writes": True,
        "limitations": [
            "Only two train windows; not performance evaluation",
            "ST gradients are biased and do not prove useful discrete selection",
            "Fixed image-grid future target, not object tracking",
            "Front-only; shared-GPU timing",
            "No optimal choice, future prediction accuracy, safety gain or novelty claim",
        ],
    }
    (output_directory / "connection_results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "gradient_norms",
                    "new_trainable_parameter_count",
                    "total_wall_seconds",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
