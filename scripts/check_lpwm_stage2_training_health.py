"""Read-only live-training audit and bounded, fixed-subset checkpoint evaluation.

Never modifies the training process, its checkpoint, configuration or queue.
An atomic hard link pins the checkpoint inode while the trainer keeps running.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate_lpwm_full_planning import digest, write_json
from train_lpwm_full_planning import load_training_inputs, make_planning_inputs
from run_lpwm_navsim_posttraining import unique_module_parameters
from summarize_lpwm_posttraining import paired_recording_interval
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import build_planning_model, compute_candidate_losses


def main(arguments):
    started = time.monotonic()
    torch.set_num_threads(2)
    root = arguments.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "summary.json").exists():
        raise FileExistsError("A completed audit must not be overwritten")
    configuration, _, _, stage1_checkpoint, manifest, targets, frames, _ = load_training_inputs(arguments.config, arguments.condition)
    training_root = PROJECT_ROOT / configuration["output_directory"]
    condition_root = training_root / arguments.condition
    registration = json.loads((training_root / "queue_registration.json").read_text())
    source_checks = {name: digest(PROJECT_ROOT / name) == checksum for name, checksum in registration["source_sha256"].items()}
    assert all(source_checks.values()) and digest(arguments.config) == registration["configuration_sha256"]
    snapshot = root / "checkpoint_snapshot.pt"
    if not snapshot.exists():
        snapshot.hardlink_to(condition_root / "latest.pt")
    saved = torch.load(snapshot, map_location="cpu", mmap=True, weights_only=False)
    assert saved["configuration_sha256"] == registration["configuration_sha256"]
    records = manifest["records"]
    selected = sorted([index for index, record in enumerate(records) if record["split"] == "development"],
        key=lambda index: hashlib.sha256(("lpwm-planning-monitor:" + records[index]["current_frame_token"]).encode()).hexdigest())[:arguments.scenes]
    protocol = {"registered_unix": time.time(), "checkpoint_update": saved["completed_updates"],
        "checkpoint_sha256": digest(snapshot), "configuration_sha256": digest(arguments.config),
        "diagnostic_source_sha256": digest(__file__), "condition": arguments.condition,
        "scene_count": len(selected), "recordings": len({records[index]["recording_group"] for index in selected}),
        "tokens": [records[index]["current_frame_token"] for index in selected],
        "selection": "First128 of the already defined epoch-monitor hash ordering, fixed before checkpoint predictions",
        "maximum_diagnostic_reserved_gib": 4., "stop_if_gpu_total_used_bytes_exceeds": 44000000000,
        "inference_batch_size": 1, "maximum_seconds": 600,
        "interpretation": "Small development snapshot vs identically initialized untrained planner; not full development, independent test, GT auxiliary comparison or checkpoint selection"}
    write_json(root / "protocol.json", protocol)
    torch.manual_seed(configuration["seed"])
    model = build_planning_model(stage1_checkpoint, configuration, arguments.condition, PROJECT_ROOT).eval()
    grouped_parameters = unique_module_parameters(model.world_model)
    world_ids = {id(parameter) for parameter in model.world_model.parameters()}
    grouped_parameters["planner_and_command"] = [parameter for parameter in model.parameters() if id(parameter) not in world_ids]
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    weight_changes = {}
    for group, parameters in grouped_parameters.items():
        total, changed_scalars, changed_tensors, maximum, square_delta, square_before = 0, 0, 0, 0., 0., 0.
        for parameter in parameters:
            before = parameter.detach()
            after = saved["model"][names_by_id[id(parameter)]]
            assert torch.isfinite(after).all()
            difference = after.float() - before.float()
            count = int(torch.count_nonzero(difference))
            total += difference.numel()
            changed_scalars += count
            changed_tensors += int(count > 0)
            maximum = max(maximum, float(difference.abs().max()))
            square_delta += float(difference.double().square().sum())
            square_before += float(before.double().square().sum())
        weight_changes[group] = {"parameter_scalars": total, "changed_scalars": changed_scalars,
            "changed_tensors": changed_tensors, "parameter_tensors": len(parameters), "maximum_absolute_change": maximum,
            "relative_l2_change": (square_delta / max(square_before, 1e-30))**.5}
        assert changed_scalars > 0, group
    optimizer_steps = [int(value["step"]) for value in saved["optimizer"]["state"].values()]
    assert min(optimizer_steps) == max(optimizer_steps) == saved["completed_updates"]
    write_json(root / "weight_audit.json", {"checkpoint_update": saved["completed_updates"],
        "weight_changes": weight_changes, "optimizer_state_count": len(optimizer_steps), "optimizer_step": optimizer_steps[0],
        "source_hashes_unchanged": source_checks})
    print("WEIGHT_AUDIT_DONE", saved["completed_updates"], flush=True)
    device = torch.device(f"cuda:{arguments.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(4 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    gpu_usage_samples = []
    def check_resources():
        used_mib, free_mib = [int(value.strip()) for value in subprocess.check_output(["nvidia-smi", f"--id={arguments.gpu}",
            "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip().split(",")]
        gpu_usage_samples.append(used_mib * 1024**2)
        if used_mib * 1024**2 > 44000000000 or free_mib < 8 * 1024:
            raise RuntimeError("Diagnostic stopped to preserve training VRAM reserve")
        if time.monotonic() - started > 600:
            raise RuntimeError("Bounded diagnostic exceeded ten minutes")
    check_resources()
    teacher = np.load(PROJECT_ROOT / configuration["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    evaluations = {}
    for label in ("initial", "checkpoint"):
        if label == "checkpoint":
            model.load_state_dict(saved["model"], strict=True)
        model.to(device).eval()
        rows = []
        with torch.inference_mode():
            for offset, index in enumerate(selected):
                check_resources()
                indices = np.array([index])
                observed, status, target = make_planning_inputs(records, indices, frames, targets, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(observed, status)
                distances = (output["trajectory"].float()[..., :2] - target[..., :2]).norm(dim=-1)
                labels = torch.from_numpy(np.array(teacher[indices])).to(device)
                losses = compute_candidate_losses(output, target, model.trajectory_vocabulary, labels)
                candidate_index = int(output["candidate_indices"][0])
                score = float(teacher[index, candidate_index, -1])
                rows.append({"token": records[index]["current_frame_token"], "recording_group": records[index]["recording_group"],
                    "candidate_index": candidate_index, "metrics": {"ade_meters": float(distances.mean()),
                        "fde_meters": float(distances[:, -1].mean()), "pdms": score if np.isfinite(score) else None,
                        "imitation_loss": float(losses["imitation_loss"]),
                        "metric_bce": float(losses["metric_loss"]) if bool(losses["teacher_coverage"]) else None}})
                if (offset + 1) % 32 == 0:
                    print("MIDTRAINING_EVALUATION", label, offset + 1, len(selected), flush=True)
        evaluations[label] = rows
        write_json(root / (label + "_predictions.json"), rows)
        model.to("cpu")
        del observed, status, target, output, labels, losses
        gc.collect()
        torch.cuda.empty_cache()
    metrics = ("ade_meters", "fde_meters", "pdms", "imitation_loss", "metric_bce")
    means = {label: {metric: float(np.mean([row["metrics"][metric] for row in rows if row["metrics"][metric] is not None]))
        for metric in metrics} for label, rows in evaluations.items()}
    comparisons = {metric: paired_recording_interval(evaluations["checkpoint"], evaluations["initial"], metric) for metric in metrics}
    report = {"protocol": protocol, "weight_changes": weight_changes, "optimizer_step": optimizer_steps[0],
        "metrics": means, "checkpoint_minus_initial": comparisons, "registered_training_sources_unchanged": source_checks,
        "diagnostic_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "maximum_sampled_gpu_total_used_bytes": max(gpu_usage_samples), "seconds": time.monotonic() - started,
        "full_validation_completed": False, "world_retention_evaluated": False, "training_modified": False}
    write_json(root / "summary.json", report)
    print("HEALTH_AUDIT_DONE", json.dumps(means), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--scenes", type=int, default=128)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    main(arguments)
