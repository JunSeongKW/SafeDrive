"""Epoch-based full LPWM post-training on all available training clips, GPUs 0 and 1."""
import argparse
from collections import defaultdict
import contextlib
import ctypes
import faulthandler
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
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_lpwm_navsim_posttraining import (
    load_experiment, initialize_model, unique_module_parameters, official_loss,
    make_video_tensor, check_gpu_reserve, published_checkpoint, checkpoint_digest, write_json,
)
from visualize_lpwm_posttraining_progress import capture_particle_snapshot
from lpwm_prefetched_video_batches import create_rank_video_loader


def distributed_epoch_order(num_clips, global_batch_size, seed, epoch):
    order = np.random.default_rng(seed + epoch).permutation(num_clips)
    padded_count = math.ceil(num_clips / global_batch_size) * global_batch_size
    if padded_count > num_clips:
        order = np.concatenate([order, order[:padded_count - num_clips]])
    assert set(order) == set(range(num_clips))
    return order


def run(arguments):
    # Linux task name plus the matching interpreter alias make the job identifiable.
    assert ctypes.CDLL(None).prctl(15, b"kjs-lpwm-stage1", 0, 0, 0) == 0
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2 and local_rank in (0, 1)
    specification, output_root, manifest, frames, device = load_experiment(arguments.config, local_rank)
    distributed.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    training = specification["training"]
    execution = json.loads(arguments.execution_config.read_text()) if arguments.execution_config else {}
    profile_name = arguments.execution_config.stem if arguments.execution_config else "distributed_profile"
    run_root = output_root / (("throughput_profiles/" + profile_name) if arguments.profile and execution else "distributed_profile" if arguments.profile else "stage1")
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "training_summary.json").exists():
        distributed.destroy_process_group()
        return
    torch.manual_seed(training["seed"])
    model, official_configuration = initialize_model(device, specification["data"]["sequence_frames"])
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = LossLPIPS(normalized_rgb=official_configuration["normalize_rgb"]).to(device).eval()
    parameters_by_module = unique_module_parameters(model)
    initial_parameter_samples = {name: [parameter.detach().flatten()[:16].clone() for parameter in parameters]
        for name, parameters in parameters_by_module.items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"],
        betas=training["betas"], eps=training["epsilon"], weight_decay=training["weight_decay"])
    configuration_digest = checkpoint_digest(arguments.config)
    manifest_digest = checkpoint_digest(output_root / "manifest.json")
    records = [row for row in manifest["records"] if row["split"] == "train"]
    development_records = sorted([row for row in manifest["records"] if row["split"] == "development"],
        key=lambda row: hashlib.sha256(("lpwm-epoch-monitor:" + row["current_frame_token"]).encode()).hexdigest())[:512]
    microbatch_size = execution.get("microbatch_size_per_gpu", training["microbatch_size"])
    accumulation = execution.get("gradient_accumulation", 1 if arguments.profile else training["gradient_accumulation"])
    loader_workers = execution.get("data_loader_workers_per_rank", 0)
    rank_batch_size = microbatch_size * accumulation
    global_batch_size = rank_batch_size * world_size
    updates_per_epoch = math.ceil(len(records) / global_batch_size)
    if execution:
        assert global_batch_size == training["microbatch_size"] * training["gradient_accumulation"] * world_size
    total_updates = arguments.profile_updates if arguments.profile else updates_per_epoch * training["epochs"]
    clip_frame_indices = np.array([record["frame_cache_indices"] for record in records])
    completed_updates, previous_seconds = 0, 0.
    latest_path = run_root / "latest.pt"
    if latest_path.exists():
        if not arguments.resume:
            raise FileExistsError("Use --resume for an existing incomplete training run")
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        assert checkpoint["configuration_sha256"] == configuration_digest
        assert checkpoint["manifest_sha256"] == manifest_digest
        assert checkpoint["world_size"] == world_size
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        completed_updates, previous_seconds = checkpoint["completed_updates"], checkpoint["elapsed_seconds"]
        del checkpoint
    distributed_model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
        broadcast_buffers=False, find_unused_parameters=False, gradient_as_bucket_view=True)
    start_time = time.monotonic()
    stop_requested = {"signal": None}
    def request_stop(signal_number, _frame):
        stop_requested["signal"] = signal_number
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last_module_gradients = {}
    def save_checkpoint(reason):
        if rank != 0:
            return
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "completed_updates": completed_updates, "elapsed_seconds": previous_seconds + time.monotonic() - start_time,
            "world_size": world_size, "configuration_sha256": configuration_digest, "manifest_sha256": manifest_digest,
            "last_module_gradients": last_module_gradients, "reason": reason,
            "execution": execution,
            "rng_contract": "training torch seed = base + optimizer_update * world_size + rank; epoch order seed = base + epoch"}
        pending = latest_path.with_suffix(".pending.pt")
        torch.save(payload, pending)
        pending.replace(latest_path)
    def validate_epoch(epoch_number):
        model.eval()
        torch.manual_seed(20261003 + rank)
        sums = defaultdict(float)
        validation_count = 0
        with torch.inference_mode():
            rank_records = development_records[rank::world_size]
            for offset in range(0, len(rank_records), microbatch_size):
                chosen = rank_records[offset:offset + microbatch_size]
                video = make_video_tensor(chosen, frames, device)
                losses = official_loss(model, video, reconstruction_loss, specification)
                for name in ("loss", "loss_rec", "kl_dyn", "loss_kl_context", "obj_on_l1", "psnr"):
                    sums[name] += float(losses[name]) * len(chosen)
                validation_count += len(chosen)
                del video, losses
        names = sorted(sums)
        totals = torch.tensor([validation_count] + [sums[name] for name in names], device=device, dtype=torch.float64)
        distributed.all_reduce(totals)
        if rank == 0:
            result = {"epoch": epoch_number, "optimizer_update": completed_updates, "validation_clips": int(totals[0]),
                **{name: float(totals[index + 1] / totals[0]) for index, name in enumerate(names)}}
            with (run_root / "validation_log.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print("VALIDATION", json.dumps(result), flush=True)
        model.train()
    def snapshot(update):
        if rank == 0:
            capture_particle_snapshot(model, frames, manifest, output_root, specification, device, update)
        distributed.barrier()
    if not arguments.profile and completed_updates == 0:
        snapshot(0)
        validate_epoch(0)
    model.train()
    log_stream = (run_root / f"rank{rank}_training_log.jsonl").open("a")
    try:
        cached_epoch, epoch_order, video_loader, video_iterator = None, None, None, None
        for update_index in range(completed_updates, total_updates):
            if update_index % 8 == 0:
                check_gpu_reserve(local_rank)
                if previous_seconds + time.monotonic() - start_time > training["maximum_seconds"]:
                    raise RuntimeError("Registered wall-clock limit reached; checkpoint retained")
            torch.manual_seed(training["seed"] + update_index * world_size + rank)
            epoch_index, within_epoch = divmod(update_index, updates_per_epoch)
            if cached_epoch != epoch_index:
                epoch_order = distributed_epoch_order(len(records), global_batch_size, training["seed"], epoch_index)
                cached_epoch = epoch_index
                if loader_workers:
                    video_loader = create_rank_video_loader(output_root / "rgb_frames.npy", clip_frame_indices,
                        epoch_order, rank, world_size, microbatch_size, accumulation, within_epoch, loader_workers)
                    video_iterator = iter(video_loader)
            global_start = within_epoch * global_batch_size
            selected = epoch_order[global_start + rank * rank_batch_size:global_start + (rank + 1) * rank_batch_size]
            optimizer.zero_grad(set_to_none=True)
            averaged_losses = defaultdict(float)
            data_wait_seconds = 0.
            update_start = time.monotonic()
            for micro_index in range(accumulation):
                if update_index == 0:
                    print(f"STARTUP rank={rank} microbatch={micro_index} prepare", flush=True)
                chosen = selected[micro_index * microbatch_size:(micro_index + 1) * microbatch_size]
                synchronization = contextlib.nullcontext() if micro_index == accumulation - 1 else distributed_model.no_sync()
                with synchronization:
                    data_started = time.monotonic()
                    if video_iterator is None:
                        video = make_video_tensor([records[index] for index in chosen], frames, device)
                    else:
                        video = next(video_iterator).to(device, non_blocking=True).permute(0, 1, 4, 2, 3).float().div_(255)
                    data_wait_seconds += time.monotonic() - data_started
                    if update_index == 0:
                        print(f"STARTUP rank={rank} microbatch={micro_index} forward", flush=True)
                    losses = official_loss(distributed_model, video, reconstruction_loss, specification)
                    if not torch.isfinite(losses["loss"]):
                        raise FloatingPointError("Nonfinite official LPWM ELBO")
                    if update_index == 0:
                        print(f"STARTUP rank={rank} microbatch={micro_index} backward", flush=True)
                    (losses["loss"] / accumulation).backward()
                    if update_index == 0:
                        print(f"STARTUP rank={rank} microbatch={micro_index} backward_complete", flush=True)
                for name, value in losses.items():
                    if value.numel() == 1:
                        averaged_losses[name] += float(value.detach()) / accumulation
                del video, losses
            if update_index == 0 or arguments.profile or (update_index + 1) % 128 == 0:
                last_module_gradients = {name: {"norm": float(torch.stack([parameter.grad.detach().square().sum()
                    for parameter in parameters if parameter.grad is not None]).sum().sqrt()),
                    "parameters_with_nonzero_gradient": sum(parameter.numel() for parameter in parameters
                        if parameter.grad is not None and bool(parameter.grad.abs().max() > 0))}
                    for name, parameters in parameters_by_module.items()}
                assert all(value["norm"] > 0 and np.isfinite(value["norm"]) for value in last_module_gradients.values())
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True))
            optimizer.step()
            completed_updates = update_index + 1
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(120, repeat=True)
            torch.cuda.synchronize(device)
            peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
            if peak_gib > execution.get("maximum_allocated_gib", specification["resource_limits"]["maximum_allocated_gib"]):
                raise RuntimeError(f"GPU{local_rank} peak {peak_gib:.3f} exceeds memory cap")
            if update_index == 0 or arguments.profile or completed_updates % 16 == 0 or completed_updates == total_updates:
                names = sorted(averaged_losses)
                synchronized_losses = torch.tensor([averaged_losses[name] for name in names], dtype=torch.float64, device=device)
                distributed.all_reduce(synchronized_losses)
                metrics = {name: float(synchronized_losses[index] / world_size) for index, name in enumerate(names)}
                progress = {"update": completed_updates, "total_updates": total_updates,
                    "epoch_fraction": completed_updates / updates_per_epoch, "updates_per_epoch": updates_per_epoch,
                    "sampled_clips_including_epoch_padding": completed_updates * global_batch_size,
                    "global_batch_size": global_batch_size, "microbatch_size_per_gpu": microbatch_size,
                    "gradient_accumulation": accumulation, "world_size": world_size,
                    "data_loader_workers_per_rank": loader_workers, "data_wait_seconds": data_wait_seconds,
                    "elapsed_seconds": previous_seconds + time.monotonic() - start_time,
                    "update_seconds": time.monotonic() - update_start, "peak_allocated_gib": peak_gib,
                    "gradient_norm_before_clip": gradient_norm, "module_gradients": last_module_gradients, **metrics}
                log_stream.write(json.dumps(progress) + "\n")
                log_stream.flush()
                if rank == 0:
                    write_json(run_root / "progress.json", progress)
                    with (run_root / "training_log.jsonl").open("a") as stream:
                        stream.write(json.dumps(progress) + "\n")
                    print("TRAINING", json.dumps(progress), flush=True)
            if not arguments.profile and completed_updates % training["checkpoint_interval_updates"] == 0:
                save_checkpoint("periodic")
                distributed.barrier()
            if not arguments.profile and completed_updates in specification["visualization"]["optimizer_updates"]:
                optimizer.zero_grad(set_to_none=True)
                snapshot(completed_updates)
            if not arguments.profile and completed_updates % updates_per_epoch == 0:
                optimizer.zero_grad(set_to_none=True)
                validate_epoch(epoch_index + 1)
                save_checkpoint("epoch_complete")
                if rank == 0:
                    torch.save(model.state_dict(), run_root / f"epoch{epoch_index + 1:02d}.pt")
                if epoch_index + 1 in specification["visualization"].get("epochs", []):
                    snapshot(completed_updates)
                distributed.barrier()
            stop_tensor = torch.tensor(int(stop_requested["signal"] is not None), device=device)
            distributed.all_reduce(stop_tensor, op=distributed.ReduceOp.MAX)
            if int(stop_tensor):
                save_checkpoint("signal")
                if rank == 0:
                    write_json(run_root / "stopped.json", {"reason": "signal", "completed_updates": completed_updates})
                distributed.barrier()
                return
        if rank == 0:
            changed = {name: sum(bool((parameter.detach().flatten()[:16] - initial).abs().max() > 0)
                for parameter, initial in zip(parameters, initial_parameter_samples[name])) for name, parameters in parameters_by_module.items()}
            assert all(value > 0 for value in changed.values())
            checkpoint_path = run_root / "checkpoint.pt"
            torch.save(model.state_dict(), checkpoint_path)
            if not arguments.profile:
                save_checkpoint("complete")
            summary = {"completed_updates": completed_updates, "effective_batch_size": global_batch_size,
                "sampled_clips": completed_updates * global_batch_size, "seen_train_clips": min(len(records), completed_updates * global_batch_size),
                "epochs": completed_updates / updates_per_epoch, "world_size": world_size,
                "seconds": previous_seconds + time.monotonic() - start_time,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "unique_parameter_counts": {name: sum(parameter.numel() for parameter in parameters) for name, parameters in parameters_by_module.items()},
                "parameter_tensors_changed_in_sample": changed, "module_gradients": last_module_gradients,
                "initial_checkpoint_sha256": checkpoint_digest(published_checkpoint()),
                "checkpoint_sha256": checkpoint_digest(checkpoint_path), "checkpoint": str(checkpoint_path),
                "configuration_sha256": configuration_digest, "manifest_sha256": manifest_digest,
                "execution": execution,
                "official_elbo": True, "planning_loss": False, "profile_only": arguments.profile, "resumed": arguments.resume}
            write_json(run_root / "training_summary.json", summary)
            print("TRAINING_DONE", json.dumps(summary), flush=True)
        distributed.barrier()
    except Exception as error:
        if rank == 0:
            with contextlib.suppress(Exception):
                save_checkpoint("error")
            write_json(run_root / "stopped.json", {"error": repr(error), "completed_updates": completed_updates})
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        log_stream.close()
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-config", type=Path)
    parser.add_argument("--profile-updates", type=int, default=3)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    if arguments.execution_config:
        arguments.execution_config = arguments.execution_config.resolve()
    run(arguments)
