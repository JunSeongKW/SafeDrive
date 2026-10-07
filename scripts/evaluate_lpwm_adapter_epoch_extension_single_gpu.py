"""Evaluate each continued Adapter epoch on the unchanged development panels."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

import train_lpwm_partial_planning as original
from evaluate_lpwm_partial_planning import summarize_rows, paired_metrics
from profile_lpwm_adapter_concurrent_training import native_digest, card_used_bytes
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
from train_lpwm_adapter_epoch_extension import verify_registration

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    registration = verify_registration(arguments.config, specification)
    root = PROJECT_ROOT / specification["output_directory"]
    destination = root / f"epoch{arguments.epoch:02d}_evaluation"
    destination.mkdir(exist_ok=True)
    if (destination / "summary.json").exists():
        return
    device = torch.device("cuda", 0)
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    allowance = min(3 * 1024**3, 47_000_000_000 - card_used_bytes(0) - 256 * 1024**2)
    assert allowance > 2 * 1024**3
    torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(device).total_memory, device)
    inputs = original.load_training_inputs(PROJECT_ROOT / specification["base_configuration"], "metric_plus_world")
    base, stage1_configuration, stage1_root, stage1_checkpoint, manifest, targets, frames, _ = inputs
    protocol = json.loads((PROJECT_ROOT / base["output_directory"] / "evaluation_protocol.json").read_text())
    assert original.checkpoint_digest(PROJECT_ROOT / base["output_directory"] / "evaluation_protocol.json") == registration["evaluation_protocol_sha256"]
    records = manifest["records"]
    teacher = np.load(PROJECT_ROOT / base["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    torch.manual_seed(base["seed"])
    model = build_adapter_or_full_planning_model(stage1_checkpoint, base, "metric_plus_world", PROJECT_ROOT).to(device).eval()
    checkpoint = root / ("source_epoch01_resume.pt" if arguments.verify_baseline else f"epoch{arguments.epoch:02d}_resume.pt")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["completed_updates"] == arguments.epoch * 4707
    model.load_state_dict(saved["model"], strict=True)
    del saved
    assert native_digest(model) == registration["frozen_native_sha256"]
    baseline = PROJECT_ROOT / specification["source_evaluation"]
    initial_rows = json.loads((baseline / "initial.json").read_text())
    epoch1_rows = json.loads((baseline / "trained.json").read_text())
    epoch1_world = json.loads((baseline / "world.json").read_text())
    if arguments.verify_baseline:
        assert arguments.epoch == 1
        baseline_by_token = {row["token"]: row for row in epoch1_rows}
        checks = []
        with torch.inference_mode():
            for offset in range(0, 16, 4):
                selected = np.asarray(protocol["planning_indices"][offset:offset + 4])
                observed, status, truth = original.make_planning_inputs(records, selected, frames, targets, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(observed, status)
                errors = (prediction["trajectory"].float()[..., :2] - truth[..., :2]).norm(dim=-1)
                for batch_index, index in enumerate(selected):
                    reference = baseline_by_token[records[index]["current_frame_token"]]
                    candidate = int(prediction["candidate_indices"][batch_index])
                    difference = abs(float(errors[batch_index].mean()) - reference["metrics"]["ade_meters"])
                    assert candidate == reference["candidate_index"] and difference < 1e-6
                    checks.append({"token": reference["token"], "candidate_matches": True, "ade_difference": difference})
        original.write_json(root / "baseline_replay_check.json", {"passed": True, "checks": checks,
            "maximum_ade_difference": max(row["ade_difference"] for row in checks),
            "checkpoint_sha256": original.checkpoint_digest(checkpoint), "card_used_bytes": card_used_bytes(0)})
        print("Baseline replay passed:16 identical candidate choices and ADE.", flush=True)
        return
    started = time.monotonic()
    max_card_bytes = 0
    outputs = {}

    def resource_check():
        nonlocal max_card_bytes
        max_card_bytes = max(max_card_bytes, card_used_bytes(0))
        assert max_card_bytes < 47_500_000_000
        return (root / "pause.requested").exists() or (root / "yield_for_primary.requested").exists()

    with torch.inference_mode():
        for label in ("trained", "persistent_future"):
            path = destination / f"{label}.json"
            rows = json.loads(path.read_text()) if path.exists() else []
            batch = specification["planning_evaluation_microbatch"]
            for offset in range(len(rows), len(protocol["planning_indices"]), batch):
                if resource_check():
                    original.write_json(path, rows)
                    return
                selected = np.asarray(protocol["planning_indices"][offset:offset + batch])
                observed, status, truth = original.make_planning_inputs(records, selected, frames, targets, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(observed, status, intervention="persistent_future" if label == "persistent_future" else None)
                errors = (prediction["trajectory"].float()[..., :2] - truth[..., :2]).norm(dim=-1)
                metric_targets = torch.from_numpy(np.array(teacher[selected])).to(device)
                valid = torch.isfinite(metric_targets[..., :6]).all((-1, -2))
                calibration = F.binary_cross_entropy_with_logits(prediction["metric_logits"].float(),
                    metric_targets[..., :6].nan_to_num(0), reduction="none").mean(1).sum(-1)
                for batch_index, index in enumerate(selected):
                    candidate = int(prediction["candidate_indices"][batch_index])
                    rows.append({"token": records[index]["current_frame_token"], "recording_group": records[index]["recording_group"],
                        "candidate_index": candidate, "metrics": {"ade_meters": float(errors[batch_index].mean()),
                        "fde_meters": float(errors[batch_index, -1]),
                        "pdms": float(teacher[index, candidate, -1]) if valid[batch_index] else None,
                        "metric_bce": float(calibration[batch_index]) if valid[batch_index] else None}})
                if len(rows) % 32 == 0 or len(rows) == 1024:
                    original.write_json(path, rows)
                    original.write_json(destination / "progress.json", {"stage": label, "completed": len(rows), "epoch": arguments.epoch})
            assert [row["token"] for row in rows] == protocol["planning_tokens"]
            outputs[label] = rows
        os.environ["TORCH_HOME"] = str(original.ARTIFACT_ROOT / "torch")
        os.chdir(original.ARTIFACT_ROOT)
        from utils.loss_functions import LossLPIPS
        perceptual = LossLPIPS(normalized_rgb=False).to(device).eval().perceptual_loss
        references = {row["token"]: row for row in json.loads((stage1_root / "evaluation/posttrained/metrics.json").read_text())["records"]}
        world_path = destination / "world.json"
        world_rows = json.loads(world_path.read_text()) if world_path.exists() else []
        for index in protocol["world_indices"][len(world_rows):]:
            if resource_check():
                original.write_json(world_path, world_rows)
                return
            video, status, _ = original.make_planning_inputs(records, np.array([index]), frames, targets, device, include_future=True)
            with model.encoder_command(status):
                reconstructed = model.world_model(video, deterministic=True)["rec_rgb"].reshape_as(video)[0]
                forecast, _ = model.world_model.sample_from_x(video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                    deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            token = records[index]["current_frame_token"]
            world_rows.append({"token": token, "recording_group": records[index]["recording_group"],
                "risk_flags": references[token]["risk_flags"], "metrics": {
                    "reconstruction_lpips": float(perceptual(reconstructed * 2 - 1, video[0] * 2 - 1).mean()),
                    "forecast_lpips": float(perceptual(forecast[0, -8:] * 2 - 1, video[0, 4:] * 2 - 1).mean())}})
            if len(world_rows) % 16 == 0 or len(world_rows) == 256:
                original.write_json(world_path, world_rows)
                original.write_json(destination / "progress.json", {"stage": "world_retention", "completed": len(world_rows), "epoch": arguments.epoch})
    assert native_digest(model) == registration["frozen_native_sha256"]
    assert len(world_rows) == 256 and len(outputs["trained"]) == 1024
    assert sum(row["metrics"]["pdms"] is not None for row in outputs["trained"]) == 1021
    world_references = [references[row["token"]] for row in world_rows]
    by_recording = {}
    for row in outputs["trained"]:
        by_recording.setdefault(row["recording_group"], []).append(row)
    report = {"complete": True, "epoch": arguments.epoch, "completed_updates": arguments.epoch * 4707,
        "checkpoint_sha256": original.checkpoint_digest(checkpoint), "configuration_sha256": registration["configuration_sha256"],
        "planning_scenes": 1024, "valid_pdms_scenes": 1021, "world_clips": 256,
        "means": {name: summarize_rows(rows) for name, rows in outputs.items()},
        "world_means": summarize_rows(world_rows), "epoch1_pdms": summarize_rows(epoch1_rows)["pdms"],
        "trained_minus_epoch1": paired_metrics(outputs["trained"], epoch1_rows, ("pdms", "ade_meters", "fde_meters", "metric_bce")),
        "trained_minus_initial": paired_metrics(outputs["trained"], initial_rows, ("pdms", "ade_meters", "metric_bce")),
        "predicted_minus_persistent_future": paired_metrics(outputs["trained"], outputs["persistent_future"], ("pdms", "ade_meters")),
        "world_vs_stage1": paired_metrics(world_rows, world_references, ("reconstruction_lpips", "forecast_lpips")),
        "world_vs_epoch1": paired_metrics(world_rows, epoch1_world, ("reconstruction_lpips", "forecast_lpips")),
        "by_recording": {name: summarize_rows(rows) for name, rows in by_recording.items()},
        "world_risk_breakdown": {risk: paired_metrics([row for row in world_rows if row["risk_flags"].get(risk)],
            world_references, ("reconstruction_lpips", "forecast_lpips"))
            for risk in sorted({name for row in world_rows for name, value in row["risk_flags"].items() if value})},
        "seconds_this_execution": time.monotonic() - started, "maximum_card_used_bytes": max_card_bytes,
        "scope": protocol["scope"], "frozen_native_sha256": registration["frozen_native_sha256"],
        "limitations": ["One training seed, exposed internal development panel, not navtest",
                         "Additional epochs hold terminal learning rates constant and use regrouped microbatches",
                         "No independent frozen-planner continuation; Adapter-only benefit not isolated"]}
    original.write_json(destination / "summary.json", report)
    original.write_json(PROJECT_ROOT / specification["shared_results_directory"] / f"epoch{arguments.epoch:02d}_summary.json", report)
    print(json.dumps({"epoch": arguments.epoch, "means": report["means"], "delta": report["trained_minus_epoch1"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epoch", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--verify-baseline", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
