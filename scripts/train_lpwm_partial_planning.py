"""One full-navtrain epoch of selective LPWM fine-tuning with early monitoring."""
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
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import compute_candidate_losses, compute_refinement_losses, uses_world_objective
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import build_partial_planning_model as build_planning_model, parameter_inventory
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT, checkpoint_digest
from run_lpwm_navsim_posttraining import write_json, unique_module_parameters, check_gpu_reserve
from train_lpwm_navtrain_distributed import distributed_epoch_order
from visualize_lpwm_planning_progress import capture_planning_snapshot
from lpwm_stage2_admission import require_stage2_admission
from planning_aware_future_prediction.object_centric.lpwm_object_supervision import ObjectTargetCache, compute_object_auxiliary_losses
from lpwm_partial_protocol import prepare_protocol


def planning_loss(predicted_trajectory, target_trajectory):
    return F.smooth_l1_loss(predicted_trajectory[..., :2].float(), target_trajectory[..., :2]) + .5 * (
        1 - (predicted_trajectory[..., 2].float() - target_trajectory[..., 2]).cos()).mean()


def load_training_inputs(config_path, condition):
    specification = json.loads(config_path.read_text())
    assert condition in specification["conditions"]
    stage1_configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_configuration["output_directory"]
    gate = json.loads((stage1_root / "adaptation_gate.json").read_text())
    checkpoint_path = stage1_root / "stage1/checkpoint.pt"
    require_stage2_admission(specification, PROJECT_ROOT, stage1_root, checkpoint_path)
    assert checkpoint_digest(checkpoint_path) == gate["training"]["checkpoint_sha256"]
    if specification.get("planner_architecture") == "particle_candidate_metrics":
        readiness = json.loads((PROJECT_ROOT / specification["output_directory"] / "stage1_validation_gate.json").read_text())
        assert readiness["passed"] and readiness["checkpoint_sha256"] == checkpoint_digest(checkpoint_path)
        teacher = json.loads((PROJECT_ROOT / specification["teacher_directory"] / "completion.json").read_text())
        assert teacher["teacher_gate_passed"]
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
    import ctypes
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-stage2", 0, 0, 0)
    rank, world_size, local_rank = (int(os.environ[name]) for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    assert world_size == 2 and local_rank in (0, 1)
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    distributed.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    specification, stage1_configuration, stage1_root, start_checkpoint, manifest, target_arrays, frame_cache, world_indices = load_training_inputs(arguments.config, arguments.condition)
    if "maximum_reserved_gib" in specification:
        torch.cuda.set_per_process_memory_fraction(specification["maximum_reserved_gib"] * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    output_root = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    if arguments.profile:
        output_root = output_root / "profile"
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "training_summary.json").exists():
        previous = json.loads((output_root / "training_summary.json").read_text())
        assert previous["configuration_sha256"] == checkpoint_digest(arguments.config), "Completed run belongs to another configuration"
        distributed.destroy_process_group()
        return
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    torch.manual_seed(specification["seed"])
    model = build_planning_model(start_checkpoint, specification, arguments.condition, PROJECT_ROOT).to(device)
    if rank == 0:
        write_json(output_root / "parameter_inventory.json", parameter_inventory(model))
    candidate_teacher = None
    refinement_oracle = None
    object_targets = ObjectTargetCache(PROJECT_ROOT / specification["object_auxiliary"]["target_directory"]) if "object_future" in arguments.condition else None
    if specification.get("planner_architecture") == "particle_candidate_metrics":
        candidate_teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    if "refinement" in arguments.condition:
        from lpwm_refinement_oracle import RefinementOracleClient
        refinement_oracle = RefinementOracleClient(PROJECT_ROOT / specification["teacher_directory"] / "metric_cache_manifest.json",
            output_root, rank, specification["refinement_oracle_workers_per_rank"])
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
    if not uses_world_objective(arguments.condition):
        model.world_model.decoder_module.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(specification["lpwm_learning_rate"], specification["planner_learning_rate"]),
        weight_decay=specification["weight_decay"])
    records = manifest["records"]
    train_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    if not arguments.profile:
        # The queue fixes this panel before training. Reading avoids concurrent writes by ranks.
        protocol = json.loads((PROJECT_ROOT / specification["output_directory"] / "evaluation_protocol.json").read_text())
        assert protocol["configuration_sha256"] == checkpoint_digest(arguments.config)
        dev_indices = protocol["planning_indices"][:specification["trend_evaluation"]["monitor_scenes"]]
    else:
        dev_indices = []
    microbatch_size = specification["microbatch_size_per_gpu"]
    accumulation = specification["gradient_accumulation"]
    rank_batch = microbatch_size * accumulation
    global_batch = rank_batch * world_size
    updates_per_epoch = math.ceil(len(train_indices) / global_batch)
    total_updates = specification.get("profile_updates", 3) if arguments.profile else updates_per_epoch * specification["epochs"]
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
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
        model.eval()
        totals = torch.zeros(5, device=device, dtype=torch.float64)
        selected = dev_indices[rank::world_size]
        for offset in range(0, len(selected), microbatch_size):
            chosen = np.asarray(selected[offset:offset + microbatch_size])
            observed, status, target = make_planning_inputs(records, chosen, frame_cache, target_arrays, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(observed, status)
            errors = (prediction["trajectory"].float()[..., :2] - target[..., :2]).norm(dim=-1)
            selected_candidates = prediction["candidate_indices"].cpu().numpy()
            scores = np.asarray(candidate_teacher[chosen, selected_candidates, -1])
            finite_scores = scores[np.isfinite(scores)]
            totals += torch.stack((errors.mean(1).sum(), errors[:, -1].sum(), errors.new_tensor(len(chosen)),
                errors.new_tensor(float(finite_scores.sum())), errors.new_tensor(len(finite_scores)))).double()
        distributed.all_reduce(totals)
        if rank == 0:
            row = {"epoch": epoch, "update": completed_updates, "clips": int(totals[2]),
                "ade_meters": float(totals[0] / totals[2]), "fde_meters": float(totals[1] / totals[2]),
                "cached_official_pdms_percent": float(100 * totals[3] / totals[4]), "scored_clips": int(totals[4]),
                "scope": "fixed development monitor; no checkpoint or hyperparameter selection"}
            with (output_root / "validation_log.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print("VALIDATION", json.dumps(row), flush=True)
        model.train()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
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
            validate(0)
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
            losses_to_log = torch.zeros(16, device=device, dtype=torch.float64)
            update_start = time.monotonic()
            input_preparation_seconds = 0.
            for micro_index in range(accumulation):
                input_started = time.monotonic()
                chosen = selected[micro_index * microbatch_size:(micro_index + 1) * microbatch_size]
                observed, status, target = make_planning_inputs(records, chosen, frame_cache, target_arrays, device)
                world_video = world_status = None
                if uses_world_objective(arguments.condition):
                    generator = np.random.default_rng(specification["seed"] + update_index * world_size * accumulation + rank * accumulation + micro_index)
                    selected_world = generator.choice(world_indices, size=specification["world_auxiliary_clips_per_gpu_microbatch"], replace=False)
                    world_video, world_status, _ = make_planning_inputs(records, selected_world, frame_cache, target_arrays, device, include_future=True)
                input_preparation_seconds += time.monotonic() - input_started
                synchronization = contextlib.nullcontext() if micro_index == accumulation - 1 else distributed_model.no_sync()
                with synchronization, torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = distributed_model(observed, status, world_video, world_status,
                        reconstruction_loss, stage1_configuration["loss"])
                    imitation_value = metric_value = coverage_value = target.new_zeros(())
                    if candidate_teacher is not None:
                        teacher_metrics = torch.from_numpy(np.array(candidate_teacher[chosen])).to(device)
                        candidate_losses = compute_candidate_losses(predicted, target, model.trajectory_vocabulary, teacher_metrics,
                            specification["candidate_imitation_temperature_meters"],
                            0. if arguments.condition.startswith("imitation") else specification["metric_loss_weight"])
                        trajectory_loss = candidate_losses["objective"]
                        imitation_value, metric_value, coverage_value = (candidate_losses[name] for name in ("imitation_loss", "metric_loss", "teacher_coverage"))
                    else:
                        trajectory_loss = planning_loss(predicted["trajectory"], target)
                    refinement_values = [target.new_zeros(()) for _ in range(4)]
                    if refinement_oracle is not None:
                        metric_labels, temporal_labels = refinement_oracle.score(chosen, predicted["refined_candidates"].detach().float().cpu().numpy())
                        refinement_losses = compute_refinement_losses(predicted, target, status,
                            torch.from_numpy(metric_labels).to(device), torch.from_numpy(temporal_labels).to(device))
                        trajectory_loss = trajectory_loss + refinement_losses["objective"]
                        refinement_values = [refinement_losses[name] for name in ("refinement_imitation", "refined_metric_loss", "temporal_safety_loss", "comfort_proxy")]
                    objective = trajectory_loss + specification["world_objective_weight"] * predicted["world_objective"]
                    object_values = [target.new_zeros(()) for _ in range(6)]
                    if object_targets is not None:
                        auxiliary_targets = object_targets.select_batch(chosen, device)
                        object_losses = compute_object_auxiliary_losses(predicted, auxiliary_targets, specification["object_auxiliary"])
                        objective = objective + object_losses["objective"]
                        object_values = [object_losses[name] for name in ("current_state_loss", "future_state_loss", "category_loss",
                            "association_fallback_fraction", "valid_current_objects", "valid_future_objects")]
                    if not torch.isfinite(objective):
                        raise FloatingPointError("Nonfinite stage2 objective")
                    (objective / accumulation).backward()
                losses_to_log += torch.stack((trajectory_loss.detach(), predicted["world_objective"].detach(), objective.detach(),
                    imitation_value.detach(), metric_value.detach(), coverage_value.detach(), *[value.detach() for value in refinement_values],
                    *[value.detach() for value in object_values])).double() / accumulation
                del observed, status, target, predicted, objective, trajectory_loss, world_video, world_status
            if update_index == 0 or arguments.profile or (update_index + 1) % 128 == 0:
                gradients = module_gradient_norms(model)
                assert all(gradients[name] > 0 and np.isfinite(gradients[name]) for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
                assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
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
                        "imitation_loss": float(losses_to_log[3]), "metric_distillation_loss": float(losses_to_log[4]), "teacher_coverage": float(losses_to_log[5]),
                        "refinement_imitation": float(losses_to_log[6]), "refined_metric_loss": float(losses_to_log[7]),
                        "temporal_safety_loss": float(losses_to_log[8]), "refinement_comfort_proxy": float(losses_to_log[9]),
                        "object_current_state_loss": float(losses_to_log[10]), "object_future_state_loss": float(losses_to_log[11]),
                        "object_category_loss": float(losses_to_log[12]), "object_association_fallback_fraction": float(losses_to_log[13]),
                        "valid_current_object_targets": float(losses_to_log[14]), "valid_future_object_targets": float(losses_to_log[15]),
                        "module_gradients": gradients, "lpwm_lr": optimizer.param_groups[0]["lr"], "planner_lr": optimizer.param_groups[1]["lr"],
                        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                        "data_loader_workers_per_rank": 0, "input_preparation_seconds": input_preparation_seconds,
                        "update_seconds": time.monotonic() - update_start, "elapsed_seconds": previous_seconds + time.monotonic() - training_start}
                    write_json(output_root / "progress.json", row)
                    with (output_root / "training_log.jsonl").open("a") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print("TRAINING", json.dumps(row), flush=True)
            if not arguments.profile and completed_updates % specification["checkpoint_interval_updates"] == 0:
                save("periodic")
                distributed.barrier()
            if not arguments.profile and completed_updates % specification["trend_evaluation"]["monitor_interval_updates"] == 0:
                optimizer.zero_grad(set_to_none=True)
                validate(completed_updates / updates_per_epoch)
            if not arguments.profile and completed_updates == total_updates // 2:
                optimizer.zero_grad(set_to_none=True)
                snapshot(completed_updates)
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
        if refinement_oracle is not None:
            refinement_oracle.close()
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_planning/partial_output_layers_v1.json")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
