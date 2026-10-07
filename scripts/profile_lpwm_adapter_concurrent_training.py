"""Measure resumed Adapter training alongside the active joint experiment.

Profile updates are discarded. The historical checkpoint/configuration is read only.
"""
import argparse
import contextlib
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

import train_lpwm_partial_planning as original
from lpwm_adapter_memory_execution import CheckpointedFrameReconstructionLoss
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import (
    build_adapter_or_full_planning_model,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIGURATION = PROJECT_ROOT / "configs/lpwm_planning/card_budget_measured_v4/residual_adapter_batch8.json"
SOURCE_DIRECTORY = PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/metric_plus_world"


def card_used_bytes(gpu_index):
    value = subprocess.check_output([
        "nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ], text=True)
    return int(value.strip()) * 1024**2


def native_digest(model):
    digest = hashlib.sha256()
    for name, value in model.world_model.state_dict().items():
        if ".residual_adapter." not in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(arguments):
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    assert int(os.environ["WORLD_SIZE"]) == 2
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    report = {"rank": rank, "profile_only": True, "updates_discarded": True, "passed": False,
              "source_sha256": original.checkpoint_digest(Path(__file__)),
              "lpips_frames_per_chunk": arguments.lpips_frame_chunk}
    started = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.cuda.set_device(device)
        distributed.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
        distributed.barrier()
        used_before_model = card_used_bytes(device.index)
        allowance = min(int(arguments.allocator_gib * 1024**3),
                        47_400_000_000 - used_before_model - 256 * 1024**2)
        assert allowance > 0
        torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(device).total_memory, device)
        report.update(card_used_before_model=used_before_model, allocator_allowance_bytes=allowance)
        inputs = original.load_training_inputs(SOURCE_CONFIGURATION, "metric_plus_world")
        specification, stage1_configuration, _, stage1_checkpoint, manifest, targets, frames, world_indices = inputs
        os.environ["TORCH_HOME"] = str(original.ARTIFACT_ROOT / "torch")
        os.chdir(original.ARTIFACT_ROOT)
        torch.manual_seed(specification["seed"])
        model = build_adapter_or_full_planning_model(stage1_checkpoint, specification, "metric_plus_world", PROJECT_ROOT).to(device)
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(
            specification["lpwm_learning_rate"], specification["planner_learning_rate"]),
            weight_decay=specification["weight_decay"])
        saved = torch.load(SOURCE_DIRECTORY / "latest.pt", map_location="cpu", weights_only=False)
        assert saved["completed_updates"] == 4707
        assert saved["configuration_sha256"] == original.checkpoint_digest(SOURCE_CONFIGURATION)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        report["restored_optimizer_states"] = len(optimizer.state)
        report["source_completed_updates"] = saved["completed_updates"]
        del saved
        initial_native_digest = native_digest(model)
        from utils.loss_functions import LossLPIPS
        reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
        if arguments.lpips_frame_chunk:
            reconstruction_loss = CheckpointedFrameReconstructionLoss(reconstruction_loss, arguments.lpips_frame_chunk)
        teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
        records = manifest["records"]
        training_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
        assert len(training_indices) == 75297
        accumulation = 8 // arguments.microbatch
        assert arguments.microbatch in (1, 2, 4, 8)
        world_microbatch = max(1, arguments.microbatch // 2)
        report.update(microbatch_per_gpu=arguments.microbatch, gradient_accumulation=accumulation,
                      effective_planning_batch=16, effective_world_batch=8)
        wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                         broadcast_buffers=False, gradient_as_bucket_view=True)
        order = training_indices[original.distributed_epoch_order(len(training_indices), 16, specification["seed"], 1)]
        model.train()
        report["updates"] = []
        for offset in range(arguments.updates):
            update_index = 4707 + offset
            torch.manual_seed(specification["seed"] + update_index * 2 + rank)
            optimizer.zero_grad(set_to_none=True)
            chosen_rank = order[offset * 16 + rank * 8:offset * 16 + (rank + 1) * 8]
            update_started = time.monotonic()
            loss_values = []
            for micro_index in range(accumulation):
                chosen = chosen_rank[micro_index * arguments.microbatch:(micro_index + 1) * arguments.microbatch]
                observed, status, target = original.make_planning_inputs(records, chosen, frames, targets, device)
                world_video = world_status = None
                world_weight = 2. if arguments.microbatch == 1 else 1.
                if arguments.microbatch != 1 or micro_index % 2 == 1:
                    generator = np.random.default_rng(specification["seed"] + update_index * 2 * accumulation + rank * accumulation + micro_index)
                    chosen_world = generator.choice(world_indices, size=world_microbatch, replace=False)
                    world_video, world_status, _ = original.make_planning_inputs(records, chosen_world, frames, targets, device, include_future=True)
                synchronize = contextlib.nullcontext() if micro_index == accumulation - 1 else wrapped.no_sync()
                with synchronize, torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = wrapped(observed, status, world_video, world_status, reconstruction_loss, stage1_configuration["loss"])
                    metrics = torch.from_numpy(np.array(teacher[chosen])).to(device)
                    losses = original.compute_candidate_losses(predicted, target, model.trajectory_vocabulary,
                        metrics, specification["candidate_imitation_temperature_meters"], specification["metric_loss_weight"])
                    objective = losses["objective"] + world_weight * specification["world_objective_weight"] * predicted["world_objective"]
                    assert torch.isfinite(objective)
                    (objective / accumulation).backward()
                loss_values.append(float(objective.detach()))
                del observed, status, target, world_video, world_status, predicted, losses, objective, metrics
            gradients = original.module_gradient_norms(model)
            assert all(np.isfinite(gradients[name]) and gradients[name] > 0
                       for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(), specification["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            used = card_used_bytes(device.index)
            assert used < 47_600_000_000, used
            row = {"update": update_index + 1, "seconds": time.monotonic() - update_started,
                   "loss": float(np.mean(loss_values)), "gradient_groups": gradients,
                   "card_used_bytes": used, "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
            report["updates"].append(row)
            original.write_json(destination / f"rank{rank}.json", report)
            print(json.dumps({"rank": rank, **row}), flush=True)
        assert native_digest(model) == initial_native_digest
        report.update(passed=True, native_weights_and_buffers_unchanged=True,
                      optimizer_steps=sorted({int(state["step"]) for state in optimizer.state.values()}))
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        original.write_json(destination / f"rank{rank}.json", report)
        if distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8), default=2)
    parser.add_argument("--allocator-gib", type=float, default=5.5)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--lpips-frame-chunk", type=int, default=0)
    run(parser.parse_args())
