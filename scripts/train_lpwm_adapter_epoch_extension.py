"""Continue the completed Adapter optimizer with bounded memory and epoch stops."""
import argparse
import contextlib
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel

import train_lpwm_partial_planning as original
from lpwm_adapter_memory_execution import CheckpointedFrameReconstructionLoss
from profile_lpwm_adapter_concurrent_training import native_digest, card_used_bytes
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import (
    build_adapter_or_full_planning_model, adaptation_parameter_inventory,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_registration(configuration_path, specification):
    registration = json.loads((PROJECT_ROOT / specification["output_directory"] / "registration.json").read_text())
    assert registration["configuration_sha256"] == original.checkpoint_digest(configuration_path)
    for relative, expected in registration["sources"].items():
        assert original.checkpoint_digest(PROJECT_ROOT / relative) == expected, relative
    return registration


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    registration = verify_registration(arguments.config, specification)
    output = PROJECT_ROOT / specification["output_directory"]
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    assert int(os.environ["WORLD_SIZE"]) == 2
    import ctypes
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-adapt", 0, 0, 0)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    distributed.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=10))
    distributed.barrier()
    used_before_model = card_used_bytes(device.index)
    allowance = min(int(specification["maximum_allocator_gib_per_gpu"] * 1024**3),
                    47_400_000_000 - used_before_model - 256 * 1024**2)
    assert allowance >= 5 * 1024**3, "Defer until sufficient memory is available"
    torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(device).total_memory, device)
    inputs = original.load_training_inputs(PROJECT_ROOT / specification["base_configuration"], "metric_plus_world")
    base, stage1, _, stage1_checkpoint, manifest, targets, frames, world_indices = inputs
    os.environ["TORCH_HOME"] = str(original.ARTIFACT_ROOT / "torch")
    os.chdir(original.ARTIFACT_ROOT)
    torch.manual_seed(base["seed"])
    model = build_adapter_or_full_planning_model(stage1_checkpoint, base, "metric_plus_world", PROJECT_ROOT).to(device)
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(base["lpwm_learning_rate"], base["planner_learning_rate"]),
                                 weight_decay=base["weight_decay"])
    latest = output / "latest.pt"
    resume_path = latest if latest.exists() else output / "source_epoch01_resume.pt"
    saved = torch.load(resume_path, map_location="cpu", weights_only=False)
    expected = registration["base_configuration_sha256"] if resume_path.name == "source_epoch01_resume.pt" else registration["configuration_sha256"]
    assert saved["configuration_sha256"] == expected
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    completed = int(saved["completed_updates"])
    prior_seconds = float(saved.get("extension_elapsed_seconds", 0.))
    assert {int(state["step"]) for state in optimizer.state.values()} == {completed}
    rates = [group["lr"] for group in optimizer.param_groups]
    assert np.allclose(rates, registration["continued_learning_rates"], rtol=0, atol=0)
    del saved
    initial_native = native_digest(model)
    assert initial_native == registration["frozen_native_sha256"]
    from utils.loss_functions import LossLPIPS
    reconstruction_loss = CheckpointedFrameReconstructionLoss(
        LossLPIPS(normalized_rgb=False).to(device).eval(), specification["lpips_frames_per_chunk"])
    teacher = np.load(PROJECT_ROOT / base["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    records = manifest["records"]
    train_indices = np.array([index for index, record in enumerate(records) if record["split"] == "train"])
    assert len(train_indices) == 75297 and math.ceil(len(train_indices) / 16) == specification["updates_per_epoch"]
    wrapped = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True,
                                     broadcast_buffers=False, gradient_as_bucket_view=True)
    model.train()
    started = time.monotonic()
    stopping = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopping.append(True))
    if rank == 0:
        original.write_json(output / "parameter_inventory.json", adaptation_parameter_inventory(model))
        original.write_json(output / "resume_check.json", {"completed_updates": completed,
            "optimizer_states": len(optimizer.state), "learning_rates": rates,
            "native_digest": initial_native, "resume_path": str(resume_path)})

    def save(reason):
        current_native = native_digest(model)
        assert current_native == initial_native
        assert {int(state["step"]) for state in optimizer.state.values()} == {completed}
        if rank == 0:
            pending = latest.with_suffix(".pending.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "completed_updates": completed, "configuration_sha256": registration["configuration_sha256"],
                "source_checkpoint_sha256": registration["source_checkpoint_sha256"],
                "stage1_checkpoint_sha256": registration["stage1_checkpoint_sha256"],
                "frozen_native_sha256": current_native, "world_size": 2, "reason": reason,
                "extension_elapsed_seconds": prior_seconds + time.monotonic() - started}, pending)
            pending.replace(latest)
            original.write_json(output / "saved_state.json", {"reason": reason, "completed_updates": completed,
                "checkpoint_sha256": original.checkpoint_digest(latest), "updated_unix": time.time()})
        distributed.barrier()

    cached_epoch, epoch_order = None, None
    target = arguments.target_epoch * specification["updates_per_epoch"]
    try:
        while completed < target:
            pause = bool(stopping) or (output / "yield_for_primary.requested").exists() or (output / "pause.requested").exists()
            used = card_used_bytes(device.index)
            pause = pause or used >= specification["secondary_stop_card_bytes"]
            flag = torch.tensor(int(pause), device=device)
            distributed.all_reduce(flag, op=distributed.ReduceOp.MAX)
            if int(flag):
                save("resource_or_user_pause")
                return
            update_index = completed
            epoch, epoch_offset = divmod(update_index, specification["updates_per_epoch"])
            if epoch != cached_epoch:
                epoch_order = train_indices[original.distributed_epoch_order(len(train_indices), 16, base["seed"], epoch)]
                cached_epoch = epoch
            selected = epoch_order[epoch_offset * 16 + rank * 8:epoch_offset * 16 + (rank + 1) * 8]
            torch.manual_seed(base["seed"] + update_index * 2 + rank)
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(3, device=device, dtype=torch.float64)
            step_started = time.monotonic()
            for micro_index in range(8):
                chosen = selected[micro_index:micro_index + 1]
                observed, status, target_trajectory = original.make_planning_inputs(records, chosen, frames, targets, device)
                world_video = world_status = None
                if micro_index % 2 == 1:
                    generator = np.random.default_rng(base["seed"] + update_index * 16 + rank * 8 + micro_index)
                    chosen_world = generator.choice(world_indices, size=1, replace=False)
                    world_video, world_status, _ = original.make_planning_inputs(records, chosen_world, frames, targets, device, include_future=True)
                synchronize = contextlib.nullcontext() if micro_index == 7 else wrapped.no_sync()
                with synchronize, torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = wrapped(observed, status, world_video, world_status, reconstruction_loss, stage1["loss"])
                    metrics = torch.from_numpy(np.array(teacher[chosen])).to(device)
                    losses = original.compute_candidate_losses(predicted, target_trajectory, model.trajectory_vocabulary,
                        metrics, base["candidate_imitation_temperature_meters"], base["metric_loss_weight"])
                    world_value = 2 * predicted["world_objective"]
                    objective = losses["objective"] + base["world_objective_weight"] * world_value
                    assert torch.isfinite(objective)
                    (objective / 8).backward()
                totals += torch.stack((losses["objective"].detach(), world_value.detach(), objective.detach())).double() / 8
                del observed, status, target_trajectory, world_video, world_status, predicted, losses, world_value, objective, metrics
            gradients = original.module_gradient_norms(model)
            assert all(np.isfinite(gradients[name]) and gradients[name] > 0
                       for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
            assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
            pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), base["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            completed += 1
            distributed.all_reduce(totals)
            totals /= 2
            torch.cuda.synchronize(device)
            row = {"completed_updates": completed, "epoch_fraction": completed / specification["updates_per_epoch"],
                "target_updates": 3 * specification["updates_per_epoch"], "planning_loss": float(totals[0]),
                "world_objective": float(totals[1]), "loss": float(totals[2]), "gradient_groups": gradients,
                "gradient_norm_before_clip": float(pre_clip), "gradient_clip": base["gradient_clip"],
                "learning_rates": rates, "seconds_this_update": time.monotonic() - step_started,
                "extension_elapsed_seconds": prior_seconds + time.monotonic() - started,
                "card_used_bytes": card_used_bytes(device.index), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "updated_unix": time.time()}
            assert row["card_used_bytes"] < specification["maximum_card_used_bytes"]
            original.write_json(output / f"rank{rank}_progress.json", row)
            with (output / f"rank{rank}_training.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            if rank == 0:
                original.write_json(output / "progress.json", row)
                print(json.dumps(row), flush=True)
            if completed % specification["checkpoint_interval_updates"] == 0:
                save("periodic")
        save("epoch_complete")
        if rank == 0:
            epoch_checkpoint = output / f"epoch{arguments.target_epoch:02d}_resume.pt"
            assert not epoch_checkpoint.exists()
            os.link(latest, epoch_checkpoint)
            original.write_json(output / f"epoch{arguments.target_epoch:02d}_training_complete.json", {
                "completed_updates": completed, "checkpoint": str(epoch_checkpoint),
                "checkpoint_sha256": original.checkpoint_digest(epoch_checkpoint), "learning_rates": rates,
                "frozen_native_sha256": initial_native, "complete": True})
        distributed.barrier()
    except Exception as error:
        # Periodic/epoch checkpoints remain valid. Never overwrite a good state with an OOM partial update.
        original.write_json(output / f"rank{rank}_failure.json", {"completed_updates": completed, "error": repr(error)})
        raise
    finally:
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-epoch", type=int, choices=(2, 3), required=True)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
