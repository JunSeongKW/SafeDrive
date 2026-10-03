"""Post-train all official LPWM modules, then evaluate a predeclared adaptation gate."""
import argparse
from collections import defaultdict
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import (
    ARTIFACT_ROOT, checkpoint_digest, load_official_lpwm, particle_geometry,
)
from evaluate_lpwm_navsim_adaptation import object_metrics
from visualize_lpwm_posttraining_progress import capture_particle_snapshot


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def check_gpu_reserve(gpu, minimum_free_gib=6):
    available_mib = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if available_mib < minimum_free_gib * 1024:
        raise RuntimeError(f"GPU{gpu} free memory {available_mib} MiB is below reserve")
    return available_mib


def unique_module_parameters(model):
    context_parameters = list(model.encoder_module.ctx_enc.parameters())
    context_ids = {id(parameter) for parameter in context_parameters}
    groups = {
        "image_encoder": [parameter for parameter in model.encoder_module.parameters() if id(parameter) not in context_ids],
        "context": context_parameters,
        "dynamics": [parameter for parameter in model.dyn_module.parameters() if id(parameter) not in context_ids],
        "rgb_decoder": list(model.decoder_module.parameters()),
    }
    parameter_ids = [id(parameter) for parameters in groups.values() for parameter in parameters]
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == {id(parameter) for parameter in model.parameters()}
    return groups


def load_experiment(config_path, gpu):
    specification = json.loads(config_path.read_text())
    output_root = PROJECT_ROOT / specification["output_directory"]
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["configuration_sha256"] == checkpoint_digest(config_path)
    frames = np.load(output_root / "rgb_frames.npy", mmap_mode="r")
    assert gpu in specification["resource_limits"]["allowed_gpus"]
    torch.set_num_threads(4)
    torch.cuda.set_device(gpu)
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    # Official LPIPS loads its local .pth relative to the working directory.
    os.chdir(ARTIFACT_ROOT)
    return specification, output_root, manifest, frames, torch.device(f"cuda:{gpu}")


def make_video_tensor(records, frames, device):
    pixels = np.stack([frames[record["frame_cache_indices"]] for record in records])
    return torch.from_numpy(pixels).to(device).permute(0, 1, 4, 2, 3).float().div_(255)


def published_checkpoint():
    paths = list((ARTIFACT_ROOT / "pretrained").glob("*best_lpips.pth"))
    assert len(paths) == 1
    return paths[0]


def initialize_model(device, sequence_frames, checkpoint=None):
    model, official_configuration = load_official_lpwm(device, checkpoint or published_checkpoint())
    model.timestep_horizon = sequence_frames - 1
    return model, official_configuration


def official_loss(model, videos, reconstruction_loss, specification):
    objective = {key: value for key, value in specification["loss"].items() if key != "implementation"}
    return model(videos, warmup=False, with_loss=True,
        num_static=specification["training"]["num_static_frames"],
        recon_loss_func=reconstruction_loss, **objective)["loss_dict"]


def train(arguments):
    specification, output_root, manifest, frames, device = load_experiment(arguments.config, arguments.gpu)
    training = specification["training"]
    destination = output_root / ("profile" if arguments.profile else "stage1")
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "training_summary.json").exists():
        print("TRAINING_ALREADY_COMPLETE", flush=True)
        return
    torch.manual_seed(training["seed"])
    np.random.seed(training["seed"])
    random.seed(training["seed"])
    check_gpu_reserve(arguments.gpu)
    model, official_configuration = initialize_model(device, specification["data"]["sequence_frames"])
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=official_configuration["normalize_rgb"]).to(device).eval()
    groups = unique_module_parameters(model)
    initial_samples = {name: [parameter.detach().flatten()[:16].clone() for parameter in parameters] for name, parameters in groups.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], betas=training["betas"],
        eps=training["epsilon"], weight_decay=training["weight_decay"])
    records = [row for row in manifest["records"] if row["split"] == "train"]
    microbatch_size = training["microbatch_size"]
    accumulation = 1 if arguments.profile else training["gradient_accumulation"]
    effective_batch_size = microbatch_size * accumulation
    if len(records) % effective_batch_size:
        raise ValueError("Registered training set must be divisible by effective batch size")
    updates_per_epoch = len(records) // effective_batch_size
    total_updates = 3 if arguments.profile else updates_per_epoch * training["epochs"]
    epoch_orders = [np.random.default_rng(training["seed"] + epoch).permutation(len(records)) for epoch in range(training["epochs"])]
    completed_updates, prior_seconds, last_gradients = 0, 0., {}
    latest_path = destination / "latest.pt"
    if latest_path.exists():
        if not arguments.resume:
            raise FileExistsError("Incomplete run exists; use explicit --resume")
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert saved["configuration_sha256"] == checkpoint_digest(arguments.config)
        assert saved["manifest_sha256"] == checkpoint_digest(output_root / "manifest.json")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed_updates, prior_seconds = saved["completed_updates"], saved["elapsed_seconds"]
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
        last_gradients = saved["last_gradients"]
        del saved
    training_started = time.monotonic()
    stop_request = {"signal": None}
    def request_stop(signal_number, _frame):
        stop_request["signal"] = signal_number
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    def save_progress(reason):
        torch.cuda.synchronize(device)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "completed_updates": completed_updates, "elapsed_seconds": prior_seconds + time.monotonic() - training_started,
            "configuration_sha256": checkpoint_digest(arguments.config),
            "manifest_sha256": checkpoint_digest(output_root / "manifest.json"),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(device),
            "last_gradients": last_gradients, "reason": reason}
        pending = latest_path.with_suffix(".pending.pt")
        torch.save(state, pending)
        pending.replace(latest_path)
    model.train()
    if not arguments.profile and completed_updates == 0:
        capture_particle_snapshot(model, frames, manifest, output_root, specification, device, 0)
    with (destination / "training_log.jsonl").open("a") as log_stream:
        try:
            for update_index in range(completed_updates, total_updates):
                if update_index % 8 == 0:
                    check_gpu_reserve(arguments.gpu)
                    if prior_seconds + time.monotonic() - training_started > training["maximum_seconds"]:
                        raise RuntimeError("Registered stage1 time budget reached")
                epoch_index, within_epoch = divmod(update_index, updates_per_epoch)
                offset = within_epoch * effective_batch_size
                selected = epoch_orders[epoch_index][offset:offset + effective_batch_size]
                optimizer.zero_grad(set_to_none=True)
                mean_losses = defaultdict(float)
                update_started = time.monotonic()
                for micro_offset in range(0, effective_batch_size, microbatch_size):
                    videos = make_video_tensor([records[index] for index in selected[micro_offset:micro_offset + microbatch_size]], frames, device)
                    losses = official_loss(model, videos, reconstruction_loss, specification)
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError("Nonfinite official temporal ELBO")
                    (losses["loss"] / accumulation).backward()
                    for name, value in losses.items():
                        if value.numel() == 1:
                            mean_losses[name] += float(value.detach()) / accumulation
                    del videos, losses
                if update_index == 0 or arguments.profile or (update_index + 1) % 128 == 0:
                    last_gradients = {name: {"norm": float(torch.stack([parameter.grad.detach().square().sum()
                        for parameter in parameters if parameter.grad is not None]).sum().sqrt()),
                        "parameters_with_nonzero_gradient": sum(parameter.numel() for parameter in parameters
                            if parameter.grad is not None and bool(parameter.grad.detach().abs().max() > 0))}
                        for name, parameters in groups.items()}
                    assert all(value["norm"] > 0 and np.isfinite(value["norm"]) for value in last_gradients.values())
                gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True))
                optimizer.step()
                completed_updates = update_index + 1
                torch.cuda.synchronize(device)
                peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3
                if peak_memory > specification["resource_limits"]["maximum_allocated_gib"]:
                    raise RuntimeError(f"Allocated memory {peak_memory:.3f} GiB exceeds registered cap")
                if completed_updates % 16 == 0 or update_index == 0 or arguments.profile or completed_updates == total_updates:
                    record = {"update": completed_updates, "total_updates": total_updates,
                        "epoch_fraction": completed_updates / updates_per_epoch,
                        "sampled_clips": completed_updates * effective_batch_size,
                        "elapsed_seconds": prior_seconds + time.monotonic() - training_started,
                        "update_seconds": time.monotonic() - update_started,
                        "peak_allocated_gib": peak_memory, "gradient_norm_before_clip": gradient_norm,
                        "module_gradients": last_gradients, **mean_losses}
                    log_stream.write(json.dumps(record) + "\n")
                    log_stream.flush()
                    write_json(destination / "progress.json", record)
                    print("TRAINING", json.dumps(record), flush=True)
                if not arguments.profile and completed_updates % training["checkpoint_interval_updates"] == 0:
                    save_progress("periodic")
                if not arguments.profile and completed_updates % updates_per_epoch == 0:
                    torch.save(model.state_dict(), destination / f"epoch{epoch_index + 1}.pt")
                if not arguments.profile and completed_updates in specification["visualization"]["optimizer_updates"]:
                    optimizer.zero_grad(set_to_none=True)
                    capture_particle_snapshot(model, frames, manifest, output_root, specification, device, completed_updates)
                if stop_request["signal"] is not None:
                    save_progress("signal")
                    write_json(destination / "stopped.json", {"signal": stop_request["signal"], "completed_updates": completed_updates})
                    raise SystemExit(130)
        except Exception as error:
            with contextlib.suppress(Exception):
                save_progress("error")
            write_json(destination / "stopped.json", {"error": repr(error), "completed_updates": completed_updates})
            raise
    changed_samples = {name: sum(bool((parameter.detach().flatten()[:16] - initial).abs().max() > 0)
        for parameter, initial in zip(parameters, initial_samples[name])) for name, parameters in groups.items()}
    assert all(value > 0 for value in changed_samples.values())
    checkpoint_path = destination / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    if not arguments.profile:
        save_progress("complete")
    summary = {"completed_updates": completed_updates, "effective_batch_size": effective_batch_size,
        "sampled_clips": completed_updates * effective_batch_size,
        "seen_train_clips": min(completed_updates * effective_batch_size, len(records)),
        "epochs": completed_updates / updates_per_epoch,
        "seconds": prior_seconds + time.monotonic() - training_started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "unique_parameter_counts": {name: sum(parameter.numel() for parameter in parameters) for name, parameters in groups.items()},
        "parameter_tensors_changed_in_sample": changed_samples, "module_gradients": last_gradients,
        "initial_checkpoint_sha256": checkpoint_digest(published_checkpoint()),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path), "checkpoint": str(checkpoint_path),
        "configuration_sha256": checkpoint_digest(arguments.config),
        "official_elbo": True, "planning_loss": False, "object_label_supervision": False,
        "profile_only": arguments.profile, "resumed": arguments.resume}
    write_json(destination / "training_summary.json", summary)
    print("TRAINING_DONE", json.dumps(summary), flush=True)


def object_region_mse(predicted, target, objects_by_frame, background=False):
    squared_error_sum, scalar_count = 0., 0
    for frame_index, objects in enumerate(objects_by_frame):
        pixel_mask = np.zeros((128, 128), dtype=bool)
        for annotated in objects:
            left, top = np.floor(annotated["box"][:2]).astype(int).clip(0, 127)
            right, bottom = np.ceil(annotated["box"][2:]).astype(int).clip(0, 128)
            pixel_mask[top:bottom, left:right] = True
        if background:
            pixel_mask = ~pixel_mask
        differences = (predicted[frame_index] - target[frame_index]) ** 2
        squared_error_sum += float(differences[pixel_mask].sum())
        scalar_count += int(pixel_mask.sum()) * 3
    return squared_error_sum / scalar_count if scalar_count else None


def evaluate(arguments):
    specification, output_root, manifest, frames, device = load_experiment(arguments.config, arguments.gpu)
    evaluation_root = output_root / "evaluation" / arguments.name
    evaluation_root.mkdir(parents=True, exist_ok=True)
    if (evaluation_root / "metrics.json").exists():
        print("EVALUATION_ALREADY_COMPLETE", arguments.name, flush=True)
        return
    check_gpu_reserve(arguments.gpu)
    model, _ = initialize_model(device, specification["data"]["sequence_frames"], arguments.checkpoint)
    model.eval()
    from utils.loss_functions import LossLPIPS
    perceptual_loss = LossLPIPS(normalized_rgb=False).to(device).eval().perceptual_loss
    records = [row for row in manifest["records"] if row["split"] == "development"]
    risk_metadata = json.loads((output_root / "driving_risk_metadata.json").read_text())["records_by_token"]
    if arguments.profile:
        records = records[:2]
    diagnostic_tokens = set()
    for scenario in ("straight", "turn", "projected_overlap", "other"):
        chosen = sorted((row for row in records if row["scenario"] == scenario),
            key=lambda row: hashlib.sha256(("lpwm-posttraining-visual:" + row["current_frame_token"]).encode()).hexdigest())[:2]
        diagnostic_tokens.update(row["current_frame_token"] for row in chosen)
    evaluated_records = []
    incremental_path = evaluation_root / "records.jsonl"
    if incremental_path.exists():
        evaluated_records = [json.loads(line) for line in incremental_path.read_text().splitlines() if line.strip()]
    completed_tokens = {row["token"] for row in evaluated_records}
    observed_frames = specification["data"]["observed_frames"]
    future_frames = specification["data"]["future_frames"]
    with torch.inference_mode(), incremental_path.open("a") as output_stream:
        for record_index, record in enumerate(records):
            if record["current_frame_token"] in completed_tokens:
                continue
            if record_index % 8 == 0:
                check_gpu_reserve(arguments.gpu)
            video = make_video_tensor([record], frames, device)
            encoded = model(video, deterministic=True)
            reconstruction = encoded["rec_rgb"].reshape_as(video)[0]
            observed_video = video[:, :observed_frames].contiguous()
            generated, future_particles = model.sample_from_x(observed_video,
                num_steps=future_frames, cond_steps=observed_frames, deterministic=True,
                use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            prediction = generated[0, -future_frames:]
            target = video[0, observed_frames:]
            persistence = video[0, observed_frames - 1:observed_frames].expand_as(target)
            def lpips_by_frame(first, second):
                return perceptual_loss(first * 2 - 1, second * 2 - 1).flatten(1).mean(1)
            forecast_lpips = lpips_by_frame(prediction, target)
            persistence_lpips = lpips_by_frame(persistence, target)
            raw_images = video[0].permute(0, 2, 3, 1).cpu().numpy()
            predicted_images = prediction.permute(0, 2, 3, 1).cpu().numpy()
            persistence_images = persistence.permute(0, 2, 3, 1).cpu().numpy()
            metrics = {
                "reconstruction_mse": float((reconstruction - video[0]).square().mean()),
                "reconstruction_lpips": float(lpips_by_frame(reconstruction, video[0]).mean()),
                "forecast_mse": float((prediction - target).square().mean()),
                "persistence_mse": float((persistence - target).square().mean()),
                "forecast_lpips": float(forecast_lpips.mean()), "persistence_lpips": float(persistence_lpips.mean()),
                "forecast_mse_by_horizon": (prediction - target).square().flatten(1).mean(1).cpu().tolist(),
                "persistence_mse_by_horizon": (persistence - target).square().flatten(1).mean(1).cpu().tolist(),
                "forecast_lpips_by_horizon": forecast_lpips.cpu().tolist(),
                "persistence_lpips_by_horizon": persistence_lpips.cpu().tolist(),
                "forecast_object_region_mse": object_region_mse(predicted_images, raw_images[observed_frames:], record["objects_by_frame"][observed_frames:]),
                "persistence_object_region_mse": object_region_mse(persistence_images, raw_images[observed_frames:], record["objects_by_frame"][observed_frames:]),
                "mean_particle_presence": float(encoded["obj_on"].mean()),
                "mean_visible_particles": float(encoded["obj_on"].sum(2).mean()),
                "particle_feature_std": float(encoded["z_features"].std()),
            }
            centers, sizes, presence, particle_ids = particle_geometry(encoded, observed_frames - 1, 16)
            particle_boxes = np.concatenate([centers - sizes / 2, centers + sizes / 2], -1)
            geometry_metrics, _ = object_metrics(centers, particle_boxes, record["objects_by_frame"][observed_frames - 1])
            metrics.update({"top16_" + key: value for key, value in geometry_metrics.items()})
            risk_record = risk_metadata[record["current_frame_token"]]
            risk_objects = risk_record["objects_by_frame"]
            for label in ("small", "far"):
                selected_future_objects = [[obj for obj in frame_objects if obj[label]]
                    for frame_objects in risk_objects[observed_frames:]]
                metrics[f"forecast_{label}_object_region_mse"] = object_region_mse(predicted_images, raw_images[observed_frames:], selected_future_objects)
                metrics[f"persistence_{label}_object_region_mse"] = object_region_mse(persistence_images, raw_images[observed_frames:], selected_future_objects)
                selected_current_objects = [obj for obj in risk_objects[observed_frames - 1] if obj[label]]
                subset_geometry, _ = object_metrics(centers, particle_boxes, selected_current_objects)
                metrics.update({f"top16_{label}_" + key: value for key, value in subset_geometry.items()})
            metrics["forecast_background_region_mse"] = object_region_mse(predicted_images, raw_images[observed_frames:], risk_objects[observed_frames:], background=True)
            metrics["persistence_background_region_mse"] = object_region_mse(persistence_images, raw_images[observed_frames:], risk_objects[observed_frames:], background=True)
            forecast_geometry = {"z": future_particles["z_pos"], "z_scale": future_particles["z_scale"], "obj_on": future_particles["z_obj_on"]}
            future_centers, future_sizes, _, _ = particle_geometry(forecast_geometry, -1, 16)
            future_boxes = np.concatenate([future_centers - future_sizes / 2, future_centers + future_sizes / 2], -1)
            future_geometry_metrics, _ = object_metrics(future_centers, future_boxes, record["objects_by_frame"][-1])
            metrics.update({"forecast4s_top16_" + key: value for key, value in future_geometry_metrics.items()})
            correspondences = []
            for frame_index in range(specification["data"]["sequence_frames"]):
                frame_centers, frame_sizes, _, frame_particle_ids = particle_geometry(encoded, frame_index, 16)
                frame_boxes = np.concatenate([frame_centers - frame_sizes / 2, frame_centers + frame_sizes / 2], -1)
                _, matched = object_metrics(frame_centers, frame_boxes, record["objects_by_frame"][frame_index])
                correspondences.append({track: int(frame_particle_ids[index]) for track, index in matched.items()})
            matched_pairs = retained_pairs = 0
            for previous, current in zip(correspondences[:-1], correspondences[1:]):
                for track in previous.keys() & current.keys():
                    matched_pairs += 1
                    retained_pairs += int(previous[track] == current[track])
            metrics["same_patch_particle_id_fraction"] = retained_pairs / matched_pairs if matched_pairs else None
            metrics["matched_track_pairs"] = matched_pairs
            if record_index == 0:
                altered_video = video.clone()
                altered_video[:, observed_frames:] = 1 - altered_video[:, observed_frames:]
                altered = model.encode_all(altered_video, deterministic=True)
                difference = float((altered["z"][:, :observed_frames] - encoded["z"][:, :observed_frames]).abs().max())
                assert difference < 1e-5, difference
                repeated, _ = model.sample_from_x(altered_video[:, :observed_frames].contiguous(),
                    num_steps=future_frames, cond_steps=observed_frames, deterministic=True,
                    use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
                forecast_difference = float((repeated[:, -future_frames:] - generated[:, -future_frames:]).abs().max())
                assert forecast_difference < 1e-5, forecast_difference
                metrics["future_intervention_observed_particle_difference"] = difference
                metrics["future_intervention_forecast_difference"] = forecast_difference
            if record["current_frame_token"] in diagnostic_tokens:
                np.savez_compressed(evaluation_root / (record["current_frame_token"] + ".npz"),
                    raw_images=raw_images, prediction=predicted_images,
                    reconstruction=reconstruction.permute(0, 2, 3, 1).cpu().numpy(),
                    current_centers=centers, current_particle_boxes=particle_boxes,
                    current_presence=presence, current_particle_ids=particle_ids)
            result = {"token": record["current_frame_token"], "recording_group": record["recording_group"],
                "scenario": record["scenario"], "yaw_range_degrees": record["yaw_range_degrees"],
                "projected_overlap_fraction": record["projected_overlap_fraction"],
                "risk_flags": risk_record["risk_flags"], "mean_median_flow_pixels": risk_record["mean_median_flow_pixels"],
                "ego_speed_meters_per_second": record["ego_speed_meters_per_second"], "metrics": metrics}
            output_stream.write(json.dumps(result, allow_nan=False) + "\n")
            output_stream.flush()
            evaluated_records.append(result)
            if record_index % 16 == 0:
                print("EVALUATION", arguments.name, record_index + 1, len(records), flush=True)
            del encoded, future_particles, video, generated, reconstruction, prediction, target
    assert len(evaluated_records) == len(records)
    report = {"name": arguments.name, "records": evaluated_records,
        "checkpoint_sha256": checkpoint_digest(arguments.checkpoint or published_checkpoint()),
        "configuration_sha256": checkpoint_digest(arguments.config),
        "forecast_protocol": "4 observed RGB frames only -> 8 future frames, deterministic policy-prior rollout; no best-of-N",
        "diagnostic_tokens": sorted(diagnostic_tokens), "profile_only": arguments.profile}
    write_json(evaluation_root / "metrics.json", report)
    print("EVALUATION_DONE", arguments.name, flush=True)


def visualize(arguments):
    specification, output_root, manifest, frames, device = load_experiment(arguments.config, arguments.gpu)
    assert arguments.checkpoint is None, "Standalone update0 visualization must use the published checkpoint"
    model, _ = initialize_model(device, specification["data"]["sequence_frames"])
    capture_particle_snapshot(model, frames, manifest, output_root, specification, device, 0)
    print("VISUALIZATION_READY", str(output_root / "visualization/index.html"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/posttraining_v1.json")
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate", "visualize"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--name", default="published")
    parser.add_argument("--checkpoint", type=Path)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    if arguments.checkpoint is not None:
        arguments.checkpoint = arguments.checkpoint.resolve()
    {"train": train, "evaluate": evaluate, "visualize": visualize}[arguments.mode](arguments)
