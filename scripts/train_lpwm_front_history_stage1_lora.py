"""Stage 1 adapted front-history LPWM with planning-path LoRA and DrivoR loss.

Explicit fork: new input semantics, initialization, split and learning rates.
Preserved older registered trainers and checkpoints are never overwritten.
"""
import argparse
import contextlib
import ctypes
from datetime import timedelta
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import (
    make_official_loss, oracle_loss_callback)
from planning_aware_future_prediction.object_centric.lpwm_front_history_stage1_lora import FrontHistoryStage1LoRAPlanner
from lpwm_front_history_data import FrontHistorySceneDataset
from visualize_lpwm_front_history import snapshot as front_history_snapshot
from lpwm_drivor_oracle import DrivoROracleClient


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


OfficialSceneDataset = FrontHistorySceneDataset


def device_inputs(examples, first, last, device, benchmark):
    features = {"image": examples["images"][first:last].to(device, non_blocking=True).permute(0,1,4,2,3).float()/255,
                "ego_status": examples["ego"][first:last].to(device, non_blocking=True)[:, None]}
    targets = {"trajectory": examples["trajectory"][first:last].to(device, non_blocking=True)}
    if benchmark == "navsim_v1":
        targets["trajectory_long"] = examples["trajectory_long"][first:last].to(device, non_blocking=True)
    return features, targets


def card_used_bytes(local_rank):
    return int(subprocess.check_output(["nvidia-smi", "--id=" + str(local_rank),
        "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2


def configure_allocator(local_rank):
    # Leave non-PyTorch workspace and fluctuating foreign allocations headroom.
    used = card_used_bytes(local_rank)
    allowance = 48_000_000_000 - used - 1024**3
    assert allowance > 8 * 1024**3, f"Insufficient GPU {local_rank} admission budget"
    torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(local_rank).total_memory, local_rank)


def gradient_norms(model):
    return {name: sum(parameter.grad.detach().float().square().sum().item()
        for parameter in parameters if parameter.grad is not None)**.5
        for name, parameters in model.gradient_groups().items()}


def rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(), "python": random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(path, model, optimizer, scheduler, completed, epoch, next_update, metadata):
    all_rng = [None] * distributed.get_world_size()
    distributed.all_gather_object(all_rng, rng_state())
    if distributed.get_rank() == 0:
        pending = path.with_suffix(".pending.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "completed_updates": completed, "epoch": epoch, "next_update_in_epoch": next_update,
            "rng_by_rank": all_rng, **metadata}, pending)
        pending.replace(path)
    distributed.barrier()


snapshot = front_history_snapshot


def run(arguments):
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-s2lora", 0, 0, 0)
    configuration = json.loads(arguments.config.read_text())
    execution = json.loads(arguments.execution.read_text())
    if not arguments.benchmark_updates:
        execution_registration = json.loads((PROJECT_ROOT / execution["registration"]).read_text())
        for name, expected in execution_registration["sources"].items():
            assert digest(PROJECT_ROOT / name) == expected, name
    assert set(execution["overrides"]) <= {"microbatch_per_gpu", "loader_workers_per_rank", "oracle_workers_per_rank"}
    configuration.update(execution["overrides"])
    assert configuration["effective_batch"] == 64
    rank, local_rank, world_size = (int(os.environ[name]) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    assert world_size == 2 and local_rank in (0,1)
    torch.set_num_threads(2)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    configure_allocator(local_rank)
    distributed.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=30))
    seed = configuration["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    benchmark = configuration["benchmark"]
    output = PROJECT_ROOT / configuration["output_directory"]
    if arguments.benchmark_updates:
        assert arguments.benchmark_output is not None and arguments.resume_checkpoint is not None
        output = arguments.benchmark_output.resolve()
        assert not (output / "latest.pt").exists(), "Preserve completed benchmark evidence"
    if arguments.engineering_updates:
        output = output / "ddp_engineering" / arguments.execution.stem
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = PROJECT_ROOT / configuration["manifest"]
    manifest = json.loads(manifest_path.read_text())
    if not arguments.engineering_updates:
        assert manifest["complete"] and manifest["counts"] == {"train": 75297, "development": 1024}
        registration = json.loads((PROJECT_ROOT / configuration["registration"]).read_text())
        assert registration["configurations"][str(arguments.config.relative_to(PROJECT_ROOT))] == digest(arguments.config)
        for name, expected in registration["sources"].items():
            assert digest(PROJECT_ROOT / name) == expected, f"Registered source changed: {name}"
        for name, expected in registration["immutable_inputs"].items():
            assert digest(PROJECT_ROOT / name) == expected, f"Registered input changed: {name}"
    records = [record for record in manifest["records"] if record["split"] in configuration["training_splits"]]
    if arguments.engineering_updates:
        records = (records[:64] * 64)[:128]
    assert len({record["token"] for record in records}) == len(records) or arguments.engineering_updates
    dataset = OfficialSceneDataset(records)
    microbatch = configuration["microbatch_per_gpu"]
    rank_batch = configuration["effective_batch"] // world_size
    assert 0 < microbatch <= rank_batch
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed)
    loader = DataLoader(dataset, batch_size=rank_batch, sampler=sampler, num_workers=configuration["loader_workers_per_rank"],
        pin_memory=True, persistent_workers=configuration["loader_workers_per_rank"] > 0)
    updates_per_epoch = len(loader)
    total_updates = updates_per_epoch * configuration["epochs"]
    optimizations = execution.get("optimizations", {})
    assert set(optimizations) <= {"sdpa_training", "batched_gradient_finite_check"}
    model = FrontHistoryStage1LoRAPlanner(PROJECT_ROOT / configuration["stage1_checkpoint"], benchmark).to(device)
    assert not optimizations.get("sdpa_training", False)
    initial_native_digest = model.frozen_native_digest()
    diagnostic_records = [record for record in manifest["records"] if record["split"] == "development"]
    if rank == 0:
        write_json(output / "parameter_inventory.json", model.parameter_inventory())
    optimizer = torch.optim.AdamW(model.optimizer_groups(configuration["lpwm_lr"], configuration["planner_lr"]),
        weight_decay=.01)
    # DrivoR warmup/cosine convention, using this experiment's actual train count.
    schedule_steps = math.ceil(configuration["scheduler_dataset_size"] / configuration["effective_batch"]) * configuration["epochs"]
    warmup_steps = int(schedule_steps * .1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
        torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=warmup_steps),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=schedule_steps-warmup_steps)], milestones=[warmup_steps])
    criterion, official_config = make_official_loss(benchmark)
    metadata = {"configuration_sha256": digest(arguments.config), "benchmark": benchmark,
        "execution_sha256": digest(arguments.execution), "execution_overrides": execution["overrides"], "runtime_optimizations": optimizations,
        "frozen_native_sha256": initial_native_digest, "stage1_checkpoint_sha256": digest(PROJECT_ROOT / configuration["stage1_checkpoint"]), "training_scene_count": len(records), "updates_per_epoch": updates_per_epoch, "total_updates": total_updates}
    completed, start_epoch, next_update = 0, 0, 0
    latest = output / "latest.pt"
    resume_rng = None
    resume_path = arguments.resume_checkpoint or latest
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        assert state["configuration_sha256"] == metadata["configuration_sha256"]
        model.load_state_dict(state["model"], strict=True)
        assert model.frozen_native_digest() == initial_native_digest, "Resume altered pretrained backbone"
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        completed, start_epoch, next_update = state["completed_updates"], state["epoch"], state["next_update_in_epoch"]
        resume_rng = state["rng_by_rank"][rank]
        if rank == 0:
            optimizer_steps = [int(value["step"]) for value in state["optimizer"]["state"].values() if "step" in value]
            write_json(output / ("resume_" + arguments.execution.stem + ".json"), {
                "checkpoint_sha256": digest(resume_path), "completed_updates": completed,
                "next_update_in_epoch": next_update, "epoch": start_epoch,
                "optimizer_steps_min": min(optimizer_steps), "optimizer_steps_max": max(optimizer_steps),
                "optimizer_state_count": len(optimizer_steps), "scheduler_state_restored": True,
                "rank_rng_restored_before_training": True, "effective_batch": configuration["effective_batch"],
                "runtime_optimizations": optimizations, "benchmark_updates_discarded": bool(arguments.benchmark_updates),
                "execution_sha256": digest(arguments.execution)})
        del state
    elif rank == 0:
        torch.save({name: value.cpu() for name, value in model.planner.state_dict().items()
            if not name.startswith("image_backbone.")}, output / "initial_official_planner.pt")
        write_json(output / "protocol.json", metadata | configuration)
        snapshot(model, diagnostic_records, device, output, "before", benchmark)
    if rank == 0:
        write_json(output / "active_execution.json", {
            "execution_configuration": str(arguments.execution), "execution_sha256": digest(arguments.execution),
            "overrides": execution["overrides"], "runtime_optimizations": optimizations, "effective_batch": configuration["effective_batch"],
            "completed_updates_at_start": completed})
    distributed.barrier()
    wrapper = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    if resume_rng is not None:
        restore_rng(resume_rng)
    else:
        # Independent dropout streams, identical model/optimizer initialization.
        torch.manual_seed(seed + rank)
    oracle = DrivoROracleClient(manifest_path, output, rank, configuration["oracle_workers_per_rank"])
    stop_requested = [False]
    def request_stop(_signal, _frame):
        stop_requested[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.time()
    run_initial_updates = completed
    log = (output / f"rank{rank}_training.jsonl").open("a", buffering=1)
    previous_update_finished = time.time()
    try:
        for epoch in range(start_epoch, configuration["epochs"]):
            sampler.set_epoch(epoch)
            model.train()
            for update_index, examples in enumerate(loader):
                if epoch == start_epoch and update_index < next_update:
                    continue
                update_started = time.time()
                between_update_seconds = update_started - previous_update_finished
                forward_seconds = oracle_seconds = backward_seconds = 0.
                local_count = len(examples["token"])
                optimizer.zero_grad(set_to_none=True)
                totals = {}
                for first in range(0, local_count, microbatch):
                    last = min(first+microbatch, local_count)
                    synchronization = contextlib.nullcontext() if last == local_count else wrapper.no_sync()
                    with synchronization:
                        features, targets = device_inputs(examples, first, last, device, benchmark)
                        phase_started = time.time()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            prediction = wrapper(features)
                        torch.cuda.synchronize()
                        forward_seconds += time.time() - phase_started
                        phase_started = time.time()
                        scores = torch.from_numpy(oracle.score(examples["token"][first:last],
                            prediction["proposals"].detach().float().cpu().numpy())).to(device)
                        oracle_seconds += time.time() - phase_started
                        phase_started = time.time()
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            losses = criterion(targets, prediction, official_config, scoring_function=oracle_loss_callback(scores))
                        assert torch.isfinite(losses["loss"]), "Non-finite planning loss"
                        weight = (last-first)/local_count
                        (losses["loss"] * weight).backward()
                        torch.cuda.synchronize()
                        backward_seconds += time.time() - phase_started
                        for name in ("loss", "trajectory_loss", "final_score_loss", "score", "best_score"):
                            totals[name] = totals.get(name,0.) + float(losses[name].detach())*weight
                    del prediction, losses, features, targets, scores
                if completed < 3 or (completed+1) % 100 == 0 or completed == run_initial_updates:
                    norms = gradient_norms(model)
                    assert all(np.isfinite(value) and value > 0 for value in norms.values()), norms
                else:
                    norms = None
                if optimizations.get("batched_gradient_finite_check", False):
                    finite_checks = [torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None]
                    assert torch.stack(finite_checks).all(), "Non-finite planning gradients"
                else:
                    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
                model.assert_native_frozen()
                optimizer.step()
                scheduler.step()
                completed += 1
                torch.cuda.synchronize()
                used = card_used_bytes(local_rank)
                memory_exceeded = used > 48_000_000_000
                status = metadata | {"completed_updates": completed, "epoch": epoch+1,
                    "rank_scene_tokens_sha256": hashlib.sha256("\n".join(examples["token"]).encode()).hexdigest(),
                    "update_in_epoch": update_index+1, "seconds_this_update": time.time()-update_started,
                    "elapsed_seconds": time.time()-started, "card_used_bytes": used,
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "gradient_norms": norms,
                    "timing_seconds": {"between_updates_including_sync_and_data": between_update_seconds, "forward": forward_seconds, "oracle": oracle_seconds, "backward": backward_seconds},
                    "learning_rates": [group["lr"] for group in optimizer.param_groups], **totals}
                status["estimated_remaining_seconds"] = (time.time()-started)/(completed-run_initial_updates)*(total_updates-completed)
                log.write(json.dumps(status, allow_nan=False)+"\n")
                if rank == 0:
                    write_json(output / "progress.json", status)
                    print(json.dumps(status, allow_nan=False), flush=True)
                previous_update_finished = time.time()
                pause_file = (output.parent / "pause.requested").exists() or (output / "pause.requested").exists()
                stopping = torch.tensor(int(stop_requested[0] or pause_file or memory_exceeded), device=device)
                distributed.all_reduce(stopping, op=distributed.ReduceOp.MAX)
                engineering_done = arguments.engineering_updates and completed >= arguments.engineering_updates
                benchmark_done = arguments.benchmark_updates and completed - run_initial_updates >= arguments.benchmark_updates
                epoch_end = update_index+1 == updates_per_epoch
                epoch_boundary_requested = arguments.stop_after_epoch and epoch_end
                diagnostic_boundary = completed % configuration["visualization_every"] == 0
                if completed % configuration["checkpoint_every"] == 0 or stopping.item() or epoch_end or engineering_done or benchmark_done or diagnostic_boundary:
                    assert model.frozen_native_digest() == initial_native_digest, "Frozen native weights/buffers changed"
                    save_checkpoint(latest, model, optimizer, scheduler, completed,
                        epoch+int(epoch_end), 0 if epoch_end else update_index+1, metadata)
                if diagnostic_boundary:
                    if rank == 0:
                        snapshot(model, diagnostic_records, device, output, f"update_{completed:06d}", benchmark)
                    distributed.barrier()
                if benchmark_done:
                    if rank == 0:
                        write_json(output / "passed.json", {"passed": True, "benchmark_updates": arguments.benchmark_updates,
                            "resume_updates": run_initial_updates, "completed_updates": completed,
                            "benchmark_weights_discarded": True, "last_status": status})
                    return
                if engineering_done:
                    if rank == 0:
                        write_json(output / "passed.json", {"passed": True, "updates": completed, "last_status": status})
                    return
                if stopping.item():
                    if rank == 0:
                        write_json(output / "paused.json", {"completed_updates": completed, "memory_exceeded": memory_exceeded})
                    return
            if rank == 0:
                snapshot(model, diagnostic_records, device, output, f"epoch_{epoch+1:02d}", benchmark)
                # Epoch checkpoints preserve model evidence; optimizer resume is latest.pt.
                torch.save({"model": model.state_dict(), **metadata, "completed_updates": completed},
                    output / f"epoch_{epoch+1:02d}.pt")
            distributed.barrier()
            if arguments.stop_after_epoch:
                if rank == 0:
                    write_json(output / "epoch_validation_pending.json", {"epoch": epoch+1, "completed_updates": completed})
                return
        assert model.frozen_native_digest() == initial_native_digest
        if rank == 0:
            write_json(output / "training_complete.json", metadata | {"completed_updates": completed,
                "checkpoint": str(latest), "checkpoint_sha256": digest(latest)})
    finally:
        log.close()
        oracle.close()
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stop-after-epoch", action="store_true")
    parser.add_argument("--engineering-updates", type=int, default=0)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--benchmark-updates", type=int, default=0)
    parser.add_argument("--benchmark-output", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    options = parser.parse_args()
    options.config = options.config.resolve()
    run(options)
