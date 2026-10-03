"""Distributed full-dataset LPWM/planner fine-tuning after the stage1 gate."""
import argparse
from collections import defaultdict
import contextlib
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_planning_finetuning import PlanningFineTunedLPWM
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT, checkpoint_digest
from run_lpwm_navsim_posttraining import write_json, unique_module_parameters, check_gpu_reserve
from train_lpwm_navtrain_distributed import distributed_epoch_order
from visualize_lpwm_planning_progress import capture_planning_snapshot


def planning_loss(predicted_trajectory, target_trajectory):
    return F.smooth_l1_loss(predicted_trajectory[..., :2].float(), target_trajectory[..., :2]) + .5 * (
        1 - (predicted_trajectory[..., 2].float() - target_trajectory[..., 2]).cos()).mean()


def load_training_inputs(config_path, condition):
    specification = json.loads(config_path.read_text())
    assert condition in specification["conditions"]
    stage1_configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_configuration["output_directory"]
    gate = json.loads((stage1_root / "adaptation_gate.json").read_text())
    assert gate["adaptation_gate_passed"], "Merged stage2 requires successful stage1 adaptation"
    checkpoint_path = stage1_root / "stage1/checkpoint.pt"
    assert checkpoint_digest(checkpoint_path) == gate["training"]["checkpoint_sha256"]
    manifest = json.loads((stage1_root / "planning_manifest.json").read_text())
    records = manifest["records"]
    with np.load(stage1_root / "planning_targets.npz") as stored_targets:
        target_arrays = {name: stored_targets[name] for name in stored_targets.files}
    frame_cache = np.load(stage1_root / "rgb_frames.npy", mmap_mode="r")
    stage1_records = json.loads((stage1_root / "manifest.json").read_text())["records"]
    index_by_token = {record["current_frame_token"]: index for index, record in enumerate(records)}
    world_indices = [index_by_token[record["current_frame_token"]] for record in stage1_records if record["split"] == "train"]
    return specification, stage1_configuration, stage1_root, checkpoint_path, manifest, target_arrays, frame_cache, world_indices


def make_planning_inputs(records, indices, frame_cache, targets, device, include_future=False):
    frame_count = 12 if include_future else 4
    image_indices = np.array([records[index]["frame_cache_indices"][:frame_count] for index in indices])
    assert np.all(image_indices >= 0)
    images = torch.from_numpy(np.array(frame_cache[image_indices])).to(device).permute(0, 1, 4, 2, 3).float().div_(255)
    ego_status = torch.from_numpy(targets["ego_status"][indices].copy()).to(device)
    trajectories = torch.from_numpy(targets["ego_trajectory_target"][indices].copy()).to(device)
    return images, ego_status, trajectories


def module_gradient_norms(model):
    groups = unique_module_parameters(model.world_model)
    world_ids = {id(parameter) for parameter in model.world_model.parameters()}
    groups["planner_and_command"] = [parameter for parameter in model.parameters() if id(parameter) not in world_ids]
    return {name: float(torch.stack([parameter.grad.detach().float().square().sum() for parameter in parameters if parameter.grad is not None]).sum().sqrt())
        if any(parameter.grad is not None for parameter in parameters) else 0. for name, parameters in groups.items()}


def run(arguments):
    rank, world_size, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert world_size == 2 and local_rank in (0, 1)
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    distributed.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    specification, stage1_configuration, stage1_root, start_checkpoint, manifest, target_arrays, frame_cache, world_indices = load_training_inputs(arguments.config, arguments.condition)
    output_root = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    if arguments.profile:
        output_root = output_root / "profile"
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "training_summary.json").exists():
        distributed.destroy_process_group()
        return
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    torch.manual_seed(specification["seed"])
    model = PlanningFineTunedLPWM(start_checkpoint).to(device)
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
    if arguments.condition == "planning_only":
        model.world_model.decoder_module.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(specification["lpwm_learning_rate"], specification["planner_learning_rate"]),
        weight_decay=specification["weight_decay"])
    records = manifest["records"]
    train_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    dev_indices = sorted([index for index, record in enumerate(records) if record["split"] == "development"],
        key=lambda index: hashlib.sha256(("lpwm-planning-monitor:" + records[index]["current_frame_token"]).encode()).hexdigest())[:512]
    microbatch_size = specification["microbatch_size_per_gpu"]
    accumulation = specification["gradient_accumulation"]
    rank_batch = microbatch_size * accumulation
    global_batch = rank_batch * world_size
    updates_per_epoch = math.ceil(len(train_indices) / global_batch)
    total_updates = 3 if arguments.profile else updates_per_epoch * specification["epochs"]
    completed_updates, previous_seconds = 0, 0.
    latest_path = output_root / "latest.pt"
    if latest_path.exists():
        if not arguments.resume:
            raise FileExistsError("Existing stage2 run requires --resume")
        saved = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert saved["configuration_sha256"] == checkpoint_digest(arguments.config)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        completed_updates, previous_seconds = saved["completed_updates"], saved["elapsed_seconds"]
        del saved
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
        find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=True)
    training_start = time.monotonic()
    stopping = {"signal": None}
    def stop_requested(signal_number, _frame):
        stopping["signal"] = signal_number
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    def save(reason):
        if rank == 0:
            pending = latest_path.with_suffix(".pending.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "completed_updates": completed_updates, "elapsed_seconds": previous_seconds + time.monotonic() - training_start,
                "configuration_sha256": checkpoint_digest(arguments.config), "stage1_checkpoint_sha256": checkpoint_digest(start_checkpoint),
                "world_size": world_size, "reason": reason}, pending)
            pending.replace(latest_path)
    @torch.inference_mode()
    def validate(epoch):
        model.eval()
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        selected = dev_indices[rank::world_size]
        for offset in range(0, len(selected), microbatch_size):
            chosen = np.asarray(selected[offset:offset + microbatch_size])
            observed, status, target = make_planning_inputs(records, chosen, frame_cache, target_arrays, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(observed, status)["trajectory"].float()
            errors = (prediction[..., :2] - target[..., :2]).norm(dim=-1)
            totals += torch.stack((errors.mean(1).sum(), errors[:, -1].sum(), errors.new_tensor(len(chosen)))).double()
        distributed.all_reduce(totals)
        if rank == 0:
            row = {"epoch": epoch, "update": completed_updates, "clips": int(totals[2]),
                "ade_meters": float(totals[0] / totals[2]), "fde_meters": float(totals[1] / totals[2])}
            with (output_root / "validation_log.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("VALIDATION", json.dumps(row), flush=True)
        model.train()
    gradients, profile_planning_gradients = {}, {}
    stage1_manifest = None
    if not arguments.profile and rank == 0:
        stage1_manifest = json.loads((stage1_root / "manifest.json").read_text())
    def snapshot(update):
        if rank == 0:
            capture_planning_snapshot(model, frame_cache, stage1_manifest, records, target_arrays,
                output_root, stage1_configuration, device, update)
        distributed.barrier()
    try:
        model.train()
        if not arguments.profile and completed_updates == 0:
            snapshot(0)
        cached_epoch, epoch_order = None, None
        for update_index in range(completed_updates, total_updates):
            if update_index % 8 == 0:
                check_gpu_reserve(local_rank)
                if previous_seconds + time.monotonic() - training_start > specification["maximum_seconds_per_condition"]:
                    raise RuntimeError("Registered stage2 time limit reached")
            torch.manual_seed(specification["seed"] + update_index * world_size + rank)
            epoch, epoch_offset = divmod(update_index, updates_per_epoch)
            if epoch != cached_epoch:
                epoch_order = train_indices[distributed_epoch_order(len(train_indices), global_batch, specification["seed"], epoch)]
                cached_epoch = epoch
            global_offset = epoch_offset * global_batch
            selected = epoch_order[global_offset + rank * rank_batch:global_offset + (rank + 1) * rank_batch]
            optimizer.zero_grad(set_to_none=True)
            if not arguments.profile:
                warmup_updates = max(1, int(total_updates * .01))
                learning_rate_factor = min(1., (update_index + 1) / warmup_updates) if update_index < warmup_updates else .1 + .9 * .5 * (1 + math.cos(math.pi * (update_index - warmup_updates) / (total_updates - warmup_updates)))
                optimizer.param_groups[0]["lr"] = specification["lpwm_learning_rate"] * learning_rate_factor
                optimizer.param_groups[1]["lr"] = specification["planner_learning_rate"] * learning_rate_factor
            losses_to_log = torch.zeros(3, device=device, dtype=torch.float64)
            update_start = time.monotonic()
            for micro_index in range(accumulation):
                chosen = selected[micro_index * microbatch_size:(micro_index + 1) * microbatch_size]
                observed, status, target = make_planning_inputs(records, chosen, frame_cache, target_arrays, device)
                world_video = world_status = None
                if arguments.condition == "planning_plus_world":
                    generator = np.random.default_rng(specification["seed"] + update_index * world_size * accumulation + rank * accumulation + micro_index)
                    selected_world = generator.choice(world_indices, size=specification["world_auxiliary_clips_per_gpu_microbatch"], replace=False)
                    world_video, world_status, _ = make_planning_inputs(records, selected_world, frame_cache, target_arrays, device, include_future=True)
                synchronization = contextlib.nullcontext() if micro_index == accumulation - 1 else distributed_model.no_sync()
                with synchronization, torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = distributed_model(observed, status, world_video, world_status,
                        reconstruction_loss, stage1_configuration["loss"])
                    trajectory_loss = planning_loss(predicted["trajectory"], target)
                    objective = trajectory_loss + specification["world_objective_weight"] * predicted["world_objective"]
                    if not torch.isfinite(objective):
                        raise FloatingPointError("Nonfinite stage2 objective")
                    (objective / accumulation).backward()
                losses_to_log += torch.stack((trajectory_loss.detach(), predicted["world_objective"].detach(), objective.detach())).double() / accumulation
                del observed, status, target, predicted, objective, trajectory_loss, world_video, world_status
            if update_index == 0 or arguments.profile or (update_index + 1) % 128 == 0:
                gradients = module_gradient_norms(model)
                assert all(gradients[name] > 0 and np.isfinite(gradients[name]) for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
            nn.utils.clip_grad_norm_(model.parameters(), specification["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            completed_updates = update_index + 1
            torch.cuda.synchronize(device)
            if torch.cuda.max_memory_allocated(device) / 1024**3 > specification["maximum_allocated_gib"]:
                raise RuntimeError("Stage2 GPU allocated-memory cap exceeded")
            if update_index == 0 or arguments.profile or completed_updates % 16 == 0 or completed_updates == total_updates:
                distributed.all_reduce(losses_to_log)
                losses_to_log /= world_size
                if rank == 0:
                    row = {"update": completed_updates, "total_updates": total_updates, "epoch_fraction": completed_updates / updates_per_epoch,
                        "train_clips": len(train_indices), "global_batch_size": global_batch,
                        "sampled_clips_including_padding": completed_updates * global_batch,
                        "planning_loss": float(losses_to_log[0]), "world_objective": float(losses_to_log[1]), "loss": float(losses_to_log[2]),
                        "module_gradients": gradients, "lpwm_lr": optimizer.param_groups[0]["lr"], "planner_lr": optimizer.param_groups[1]["lr"],
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "update_seconds": time.monotonic() - update_start, "elapsed_seconds": previous_seconds + time.monotonic() - training_start}
                    write_json(output_root / "progress.json", row)
                    with (output_root / "training_log.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print("TRAINING", json.dumps(row), flush=True)
            if not arguments.profile and completed_updates % specification["checkpoint_interval_updates"] == 0:
                save("periodic")
                distributed.barrier()
            if not arguments.profile and completed_updates % updates_per_epoch == 0:
                optimizer.zero_grad(set_to_none=True)
                validate(epoch + 1)
                save("epoch_complete")
                if rank == 0:
                    torch.save(model.state_dict(), output_root / f"epoch{epoch + 1:02d}.pt")
                if epoch + 1 in (1, 5, 10, 15, 20):
                    snapshot(completed_updates)
                distributed.barrier()
            stop_value = torch.tensor(int(stopping["signal"] is not None), device=device)
            distributed.all_reduce(stop_value, op=distributed.ReduceOp.MAX)
            if int(stop_value):
                save("signal")
                distributed.barrier()
                return
        if rank == 0:
            checkpoint_path = output_root / "checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_path)
            save("complete")
            summary = {"condition": arguments.condition, "completed_updates": completed_updates, "epochs": completed_updates / updates_per_epoch,
                "train_clips": len(train_indices), "effective_batch_size": global_batch, "world_size": world_size,
                "seconds": previous_seconds + time.monotonic() - training_start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3, "module_gradients": gradients,
                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "checkpoint_sha256": checkpoint_digest(checkpoint_path), "stage1_checkpoint_sha256": checkpoint_digest(start_checkpoint),
                "configuration_sha256": checkpoint_digest(arguments.config), "profile_only": arguments.profile}
            write_json(output_root / "training_summary.json", summary)
            print("TRAINING_DONE", json.dumps(summary), flush=True)
        distributed.barrier()
    except Exception as error:
        if rank == 0:
            with contextlib.suppress(Exception):
                save("error")
            write_json(output_root / "stopped.json", {"error": repr(error), "completed_updates": completed_updates})
        raise
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_planning/full_joint_training_v1.json")
    parser.add_argument("--condition", choices=("planning_plus_world", "planning_only"), required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
