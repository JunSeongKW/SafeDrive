"""Bounded matched LPWM encoder/planner learning and causal inference."""
import argparse
import gc
import hashlib
import json
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_bridge import checkpoint_digest
from planning_aware_future_prediction.object_centric.lpwm_planner import IntentConditionedParticlePlanner, compute_planning_objectives, match_current_objects

OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"
CONFIG_PATH = PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".partial")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def resource_guard(gpu, started, maximum_seconds=3600):
    free_mib = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
    if free_mib < 6144 or torch.cuda.max_memory_allocated(gpu) > 20 * 1024**3 or time.monotonic() - started > maximum_seconds:
        raise RuntimeError("LPWM planning resource bound reached; preserve other processes")


def input_batch(cache, indices, device):
    images = cache["observed_images"][indices].to(device).permute(0, 1, 4, 2, 3).float().div(255)
    status = cache["ego_status"][indices].to(device)
    return images, status


def supervision_batch(cache, indices, device):
    return {key: value[indices].to(device) for key, value in cache.items() if key not in ("observed_images", "ego_status")}


def gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    values = [gradient.detach().square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.


@torch.no_grad()
def evaluate_model(model, cache, records, device, interventions=False):
    model.eval()
    evaluations, feature_rows = {}, []
    for split in ("train", "development"):
        selected_indices = [index for index, record in enumerate(records) if record["split"] == split]
        rows, future_intervention_rows, scene_intervention_rows, features = [], [], [], []
        for start_index in range(0, len(selected_indices), 8):
            indices = selected_indices[start_index:start_index + 8]
            observed_images, ego_status = input_batch(cache, indices, device)
            predictions = model(observed_images, ego_status)
            trajectories = predictions["trajectory"].cpu()
            targets = cache["ego_trajectory_target"][indices]
            if model.condition != "ego_only":
                features.append(predictions["particle_attributes"].flatten(1).cpu())
            else:
                features.append(cache["ego_status"][indices])
            changed_future = model(observed_images, ego_status, "zero_predicted_futures")["trajectory"].cpu() if interventions and split == "development" else None
            changed_scene = model(observed_images, ego_status, "shuffle_scene_particles")["trajectory"].cpu() if interventions and split == "development" else None
            for position, index in enumerate(indices):
                record = records[index]
                row = {"token": record["current_frame_token"], "recording": record["recording_group"], "scenario": record["scenario"],
                       "trajectory": trajectories[position].tolist(), "xy_ade_m": float((trajectories[position, :, :2] - targets[position, :, :2]).norm(dim=-1).mean())}
                if split == "development" and model.condition != "ego_only":
                    valid = cache["object_valid"][index]
                    particle_indices, object_indices = match_current_objects(predictions["current_boxes"][position], cache["object_boxes"][index, valid].to(device))
                    if len(object_indices):
                        future_valid = cache["object_future_valid"][index, valid][object_indices]
                        future_errors = (predictions["future_xy"][position, particle_indices].cpu() - cache["object_future_xy"][index, valid][object_indices]).norm(dim=-1)
                        weights = cache["object_risk_weights"][index, valid][object_indices, None] * future_valid
                        row["object_future_ade_m"] = float(future_errors[future_valid].mean()) if future_valid.any() else None
                        row["risk_weighted_object_future_ade_m"] = float((future_errors * weights).sum() / weights.sum().clamp_min(1))
                        predicted_boxes = predictions["current_boxes"][position, particle_indices].cpu()
                        target_boxes = cache["object_boxes"][index, valid][object_indices]
                        intersection = (torch.minimum(predicted_boxes[:, 2:], target_boxes[:, 2:]) - torch.maximum(predicted_boxes[:, :2], target_boxes[:, :2])).clamp_min(0).prod(-1)
                        union = (predicted_boxes[:, 2:] - predicted_boxes[:, :2]).prod(-1) + (target_boxes[:, 2:] - target_boxes[:, :2]).prod(-1) - intersection
                        row["matched_box_iou"] = float((intersection / union.clamp_min(1e-8)).mean())
                        row["matched_box_recall_at_03"] = float((intersection / union.clamp_min(1e-8) >= .3).float().mean())
                rows.append(row)
                for changed, destination in ((changed_future, future_intervention_rows), (changed_scene, scene_intervention_rows)):
                    if changed is not None:
                        destination.append({"token": row["token"], "recording": row["recording"], "scenario": row["scenario"], "trajectory": changed[position].tolist(),
                                            "xy_ade_m": float((changed[position, :, :2] - targets[position, :, :2]).norm(dim=-1).mean()),
                                            "trajectory_change_m": float((changed[position, :, :2] - trajectories[position, :, :2]).norm(dim=-1).mean())})
            del predictions
        evaluations[split] = {"windows": rows, "mean_ade_m": float(np.mean([row["xy_ade_m"] for row in rows])),
                              "zero_predicted_futures": future_intervention_rows, "shuffle_scene_particles": scene_intervention_rows}
        feature_rows.append(torch.cat(features))
    return evaluations, {"train": feature_rows[0], "development": feature_rows[1]}


def train_run(condition, seed, gpu, specification, cache, records, smoke=False):
    run_directory = OUTPUT_ROOT / ("smoke" if smoke else "runs") / f"{condition}_seed{seed}"
    if (run_directory / "results.json").exists():
        return
    run_directory.mkdir(parents=True, exist_ok=True)
    device = f"cuda:{gpu}"
    started = time.monotonic()
    resource_guard(gpu, started)
    torch.cuda.reset_peak_memory_stats(gpu)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = IntentConditionedParticlePlanner(PROJECT_ROOT / specification["initial_checkpoint"], condition).to(device)
    encoder_parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and name.startswith("particle_encoder.")]
    other_parameters = [parameter for name, parameter in model.named_parameters() if parameter.requires_grad and not name.startswith("particle_encoder.")]
    optimizer = torch.optim.AdamW([{"params": encoder_parameters, "lr": specification["encoder_learning_rate"]}, {"params": other_parameters, "lr": specification["head_learning_rate"]}], weight_decay=specification["weight_decay"])
    initial_encoder = next(model.particle_encoder.parameters()).detach().clone()
    training_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    updates = 3 if smoke else specification["updates"]
    schedule = np.random.default_rng(seed).choice(training_indices, (updates, specification["batch_size"]), replace=True)
    schedule_digest = hashlib.sha256(schedule.tobytes()).hexdigest()
    gradient_contract = {}
    model.train()
    with (run_directory / "training_log.jsonl").open("w") as stream:
        for update, indices in enumerate(schedule):
            if update % 50 == 0:
                resource_guard(gpu, started)
            observed_images, ego_status = input_batch(cache, indices, device)
            supervision = supervision_batch(cache, indices, device)
            predictions = model(observed_images, ego_status)
            losses = compute_planning_objectives(predictions, supervision, condition, specification["object_loss_weight"])
            if not torch.isfinite(losses["total"]):
                raise FloatingPointError("Nonfinite task objective")
            if update == 2 and encoder_parameters:
                gradient_contract["planning_to_encoder_gradient_norm"] = gradient_norm(losses["planning"], encoder_parameters)
                if "object_supervision" in losses:
                    gradient_contract["object_future_to_encoder_gradient_norm"] = gradient_norm(losses["object_supervision"], encoder_parameters)
                assert gradient_contract["planning_to_encoder_gradient_norm"] > 0
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            if update % 50 == 0 or update == updates - 1:
                row = {"update": update + 1, "seconds": time.monotonic() - started, **{key: float(value.detach()) for key, value in losses.items()}}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(condition, seed, json.dumps(row), flush=True)
            del predictions, losses, observed_images, ego_status, supervision
    training_seconds = time.monotonic() - started
    encoder_change = float((next(model.particle_encoder.parameters()).detach() - initial_encoder).abs().max())
    if encoder_parameters:
        assert encoder_change > 0
    else:
        assert encoder_change == 0
    validation = {**gradient_contract, "encoder_parameter_max_change": encoder_change}
    if condition != "ego_only":
        model.eval()
        images, status = input_batch(cache, training_indices[:2], device)
        with torch.no_grad():
            initial_attributes, _ = model.encode_observations(images, status)
            changed_status = status.clone()
            changed_status[:, :4] = changed_status[:, :4].roll(1, -1)
            changed_attributes, _ = model.encode_observations(images, changed_status)
            validation["encoder_intent_attribute_mean_change"] = float((changed_attributes - initial_attributes).abs().mean())
            # Mutating every privileged target cannot alter the forward result: API never receives them.
            repeated_attributes, _ = model.encode_observations(images.clone(), status.clone())
            validation["repeated_observation_attribute_max_difference"] = float((repeated_attributes - initial_attributes).abs().max())
            assert validation["repeated_observation_attribute_max_difference"] == 0
    if smoke:
        evaluations, features = {}, {}
    else:
        evaluations, features = evaluate_model(model, cache, records, device, interventions=condition == "object_future_risk")
        torch.save(features, run_directory / "encoder_probe_features.pt")
        checkpoint_path = run_directory / "model.pt"
        torch.save(model.cpu().state_dict(), checkpoint_path)
    report = {"condition": condition, "seed": seed, "completed_updates": updates, "batch_schedule_sha256": schedule_digest,
              "configuration_sha256": checkpoint_digest(CONFIG_PATH), "training_seconds": training_seconds,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(gpu) / 1024**3,
              "trainable_encoder_parameters": sum(parameter.numel() for parameter in encoder_parameters),
              "trainable_head_parameters": sum(parameter.numel() for parameter in other_parameters),
              "validation": validation, "evaluations": evaluations}
    write_json(run_directory / "results.json", report)
    print("LPWM_PLANNING_RUN_DONE", condition, seed, training_seconds, validation, flush=True)
    del model, optimizer, encoder_parameters, other_parameters
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    parser.add_argument("--worker", type=int, choices=(0, 1))
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(arguments.gpu)
    specification = json.loads(CONFIG_PATH.read_text())
    records = json.loads((OUTPUT_ROOT / "manifest.json").read_text())["records"]
    cache = torch.load(OUTPUT_ROOT / "supervised_cache.pt", map_location="cpu", weights_only=True)
    if arguments.condition:
        jobs = [(arguments.condition, arguments.seed)]
    else:
        jobs = [(condition, seed) for condition in specification["conditions"] for seed in specification["seeds"]]
        jobs = [job for index, job in enumerate(jobs) if index % 2 == arguments.worker]
    started = time.monotonic()
    initial_digest = checkpoint_digest(PROJECT_ROOT / specification["initial_checkpoint"])
    for condition, seed in jobs:
        if time.monotonic() - started > specification["resources"]["worker_wall_seconds"]:
            raise RuntimeError("Worker wall-clock cap")
        train_run(condition, seed, arguments.gpu, specification, cache, records, arguments.smoke)
    assert checkpoint_digest(PROJECT_ROOT / specification["initial_checkpoint"]) == initial_digest
    if arguments.worker is not None and not arguments.smoke:
        write_json(OUTPUT_ROOT / f"worker{arguments.worker}_complete.json", {"jobs": jobs, "initial_checkpoint_preserved": True, "wall_seconds": time.monotonic() - started})
    print("LPWM_PLANNING_WORKER_DONE", flush=True)


if __name__ == "__main__":
    main()
