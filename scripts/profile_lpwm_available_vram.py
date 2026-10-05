"""Measure LPWM execution alternatives without keeping diagnostic weight updates.

Run on the idle second GPU while the existing queue evaluates on GPU0. This
single-GPU profile measures throughput and capacity, not distributed training
speed or model quality. Registered training sources and settings stay intact.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def card_memory_bytes(physical_gpu):
    output = subprocess.check_output([
        "nvidia-smi", f"--id={physical_gpu}", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True)
    return int(output.strip()) * 1024**2


def run(arguments):
    import numpy as np
    import torch
    from train_lpwm_resumed_full_planning import load_training_inputs, make_planning_inputs
    from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
    from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import compute_candidate_losses
    from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
    import planning_aware_future_prediction.object_centric.lpwm_planning_finetuning as full_model
    from evaluate_lpwm_full_planning import digest, write_json

    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-prof", 0, 0, 0)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(arguments.physical_gpu)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    # Keep one decimal GB below the user cap for context/workspace fluctuations.
    torch.empty(1, device=device)
    initial_usage = card_memory_bytes(arguments.physical_gpu)
    allowance = 47_000_000_000 - initial_usage + torch.cuda.memory_reserved(device)
    if allowance <= 0:
        raise RuntimeError("No safe profile memory allowance")
    torch.cuda.set_per_process_memory_fraction(
        min(1., allowance / torch.cuda.get_device_properties(device).total_memory), device)
    report = {"profile_only": True, "weight_updates_discarded": True,
        "physical_gpu": arguments.physical_gpu, "planning_batch_size": arguments.batch_size,
        "world_batch_size": arguments.batch_size // 2, "recompute_policy": arguments.recompute,
        "configuration_sha256": digest(arguments.config), "initial_card_used_bytes": initial_usage,
        "allocator_allowance_bytes": allowance, "measurements": [],
        "limitation": "Single GPU without DDP; not a distributed speed or quality measurement."}
    original_checkpoint = full_model.checkpoint

    def execution_checkpoint(function, *inputs, **keywords):
        skip = arguments.recompute == "none" or (
            arguments.recompute == "future_only" and function.__name__ == "calculate_world_objective")
        if skip:
            assert keywords == {"use_reentrant": False}
            return function(*inputs)
        return original_checkpoint(function, *inputs, **keywords)

    full_model.checkpoint = execution_checkpoint
    try:
        specification, stage1, _, stage1_checkpoint, manifest, targets, frames, world_indices = load_training_inputs(
            arguments.config, "metric_plus_world")
        assert specification["adaptation_method"] == "full_low_learning_rate"
        torch.manual_seed(specification["seed"])
        model = build_adapter_or_full_planning_model(stage1_checkpoint, specification, "metric_plus_world", PROJECT_ROOT)
        source = PROJECT_ROOT / specification["resume_from_prior_full"]["checkpoint"]
        assert digest(source) == specification["resume_from_prior_full"]["checkpoint_sha256"]
        saved = torch.load(source, map_location="cpu", weights_only=False)
        assert saved["completed_updates"] == 2095
        model.load_state_dict(saved["model"], strict=True)
        model.to(device).train()
        optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(
            specification["lpwm_learning_rate"], specification["planner_learning_rate"]),
            weight_decay=specification["weight_decay"])
        optimizer.load_state_dict(saved["optimizer"])
        report["optimizer_states_restored"] = len(saved["optimizer"]["state"])
        del saved
        os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
        os.chdir(ARTIFACT_ROOT)
        from utils.loss_functions import LossLPIPS
        reconstruction_loss = LossLPIPS(normalized_rgb=False).to(device).eval()
        teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
        train_indices = np.array([index for index, record in enumerate(manifest["records"]) if record["split"] == "train"])
        for update in range(arguments.updates):
            if card_memory_bytes(arguments.physical_gpu) > 48_000_000_000:
                raise RuntimeError("48GB total-card guard")
            torch.manual_seed(specification["seed"] + update * 2 + 1)
            generator = np.random.default_rng(specification["seed"] + update * 2 + 1)
            selected = generator.choice(train_indices, arguments.batch_size, replace=False)
            selected_world = generator.choice(world_indices, arguments.batch_size // 2, replace=False)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.monotonic()
            observed, status, target = make_planning_inputs(manifest["records"], selected, frames, targets, device)
            world_video, world_status, _ = make_planning_inputs(manifest["records"], selected_world, frames, targets, device, True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(observed, status, world_video, world_status, reconstruction_loss, stage1["loss"])
                metrics = torch.from_numpy(np.array(teacher[selected])).to(device)
                losses = compute_candidate_losses(prediction, target, model.trajectory_vocabulary, metrics,
                    specification["candidate_imitation_temperature_meters"], specification["metric_loss_weight"])
                objective = losses["objective"] + specification["world_objective_weight"] * prediction["world_objective"]
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite profile objective")
                objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), specification["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            row = {"update": update + 1, "update_seconds": time.monotonic() - started,
                "loss": float(objective.detach()), "gradient_norm": float(gradient_norm),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "card_used_bytes": card_memory_bytes(arguments.physical_gpu)}
            if update == 0:
                state_digest = hashlib.sha256()
                for parameter_name, tensor in model.state_dict().items():
                    state_digest.update(parameter_name.encode())
                    state_digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
                report["model_state_sha256_after_first_update"] = state_digest.hexdigest()
            report["measurements"].append(row)
            write_json(arguments.output, report)
            print(json.dumps(row), flush=True)
            if row["card_used_bytes"] > 48_000_000_000:
                raise RuntimeError("48GB total-card guard")
            del prediction, losses, objective, metrics, observed, status, target, world_video, world_status
        steady = [row["update_seconds"] for row in report["measurements"][2:]]
        report.update(passed=True, mean_update_seconds=statistics.mean(steady),
            median_update_seconds=statistics.median(steady),
            planning_samples_per_second=arguments.batch_size / statistics.mean(steady))
    except BaseException as error:
        report.update(passed=False, error=repr(error))
        raise
    finally:
        full_model.checkpoint = original_checkpoint
        write_json(arguments.output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--batch-size", type=int, choices=(8, 12, 16), default=8)
    parser.add_argument("--recompute", choices=("original", "future_only", "none"), default="original")
    parser.add_argument("--updates", type=int, default=8)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    arguments.output = arguments.output.resolve()
    assert arguments.updates >= 4
    run(arguments)
