"""Preregistered small development panels for the full-navtrain partial study."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_partial_finetuning import build_partial_planning_model
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
from train_lpwm_partial_planning import load_training_inputs, make_planning_inputs
from evaluate_lpwm_full_planning import digest, write_json
from lpwm_partial_protocol import prepare_protocol
from run_lpwm_navsim_posttraining import check_gpu_reserve
from summarize_lpwm_posttraining import paired_recording_interval


def summarize_rows(rows):
    result = {}
    for name in rows[0]["metrics"]:
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def paired_metrics(first, second, names):
    result = {}
    for name in names:
        valid = {row["token"] for row in first if row["metrics"].get(name) is not None}
        valid &= {row["token"] for row in second if row["metrics"].get(name) is not None}
        result[name] = paired_recording_interval([row for row in first if row["token"] in valid],
            [row for row in second if row["token"] in valid], name)
    return result


def evaluate(arguments):
    started = time.monotonic()
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{arguments.gpu}")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(12 * 1024**3 / torch.cuda.get_device_properties(device).total_memory, device)
    specification, stage1_configuration, stage1_root, initial_checkpoint, manifest, targets, frames, _ = load_training_inputs(arguments.config, arguments.condition)
    protocol = prepare_protocol(arguments.config)
    root = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    engineering = arguments.engineering_checkpoint is not None
    destination = root / ("engineering_evaluation" if engineering else "trend_evaluation")
    if (destination / "summary.json").exists():
        assert json.loads((destination / "summary.json").read_text())["configuration_sha256"] == digest(arguments.config)
        return
    checkpoint_path = arguments.engineering_checkpoint or (root / "checkpoint.pt")
    if engineering:
        protocol = {**protocol, "planning_indices": protocol["planning_indices"][:4],
            "world_indices": protocol["world_indices"][:2], "planning_recordings": 4,
            "scope": "Engineering execution check only; profile weights, four planning scenes and two world clips"}
    else:
        training = json.loads((root / "training_summary.json").read_text())
        assert not training["profile_only"] and training["epochs"] == specification["epochs"]
    teacher = np.load(PROJECT_ROOT / specification["teacher_directory"] / "candidate_metrics.npy", mmap_mode="r")
    records = manifest["records"]
    torch.manual_seed(specification["seed"])
    model = build_partial_planning_model(initial_checkpoint, specification, arguments.condition, PROJECT_ROOT).to(device).eval()
    outputs = {}
    with torch.inference_mode():
        for label in ("initial", "trained", "persistent_future"):
            if label == "trained":
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
            rows = []
            for offset in range(0, len(protocol["planning_indices"]), 4):
                check_gpu_reserve(arguments.gpu)
                selected = np.asarray(protocol["planning_indices"][offset:offset + 4])
                observed, status, truth = make_planning_inputs(records, selected, frames, targets, device)
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
                if offset % 128 == 0:
                    write_json(destination / "progress.json", {"stage": label, "completed": len(rows), "total": len(protocol["planning_indices"])})
            outputs[label] = rows
            write_json(destination / (label + ".json"), rows)
        os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
        os.chdir(ARTIFACT_ROOT)
        from utils.loss_functions import LossLPIPS
        perceptual = LossLPIPS(normalized_rgb=False).to(device).eval().perceptual_loss
        references = {row["token"]: row for row in json.loads((stage1_root / "evaluation/posttrained/metrics.json").read_text())["records"]}
        world_rows = []
        for offset, index in enumerate(protocol["world_indices"]):
            check_gpu_reserve(arguments.gpu)
            video, status, _ = make_planning_inputs(records, np.array([index]), frames, targets, device, include_future=True)
            with model.encoder_command(status):
                reconstructed = model.world_model(video, deterministic=True)["rec_rgb"].reshape_as(video)[0]
                forecast, _ = model.world_model.sample_from_x(video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                    deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            token = records[index]["current_frame_token"]
            world_rows.append({"token": token, "recording_group": records[index]["recording_group"],
                "risk_flags": references[token]["risk_flags"], "metrics": {
                    "reconstruction_lpips": float(perceptual(reconstructed * 2 - 1, video[0] * 2 - 1).mean()),
                    "forecast_lpips": float(perceptual(forecast[0, -8:] * 2 - 1, video[0, 4:] * 2 - 1).mean())}})
            if offset % 32 == 0:
                write_json(destination / "progress.json", {"stage": "world_retention", "completed": len(world_rows), "total": len(protocol["world_indices"])})
    write_json(destination / "world.json", world_rows)
    world_references = [references[row["token"]] for row in world_rows]
    world_comparisons = paired_metrics(world_rows, world_references, ("reconstruction_lpips", "forecast_lpips"))
    initial_comparisons = paired_metrics(outputs["trained"], outputs["initial"], ("pdms", "ade_meters", "metric_bce"))
    retained = {name: comparison["ci95"][1] <= .1 * float(np.mean([row["metrics"][name] for row in world_references]))
        for name, comparison in world_comparisons.items()}
    metric_upper = initial_comparisons["metric_bce"]["ci95"][1]
    pdms_lower = initial_comparisons["pdms"]["ci95"][0]
    checks = {"complete_registered_epoch": not engineering,
        "planning_loss_learned": metric_upper is not None and metric_upper < 0,
        "pdms_improved_over_random_planner": pdms_lower is not None and pdms_lower > 0, **retained}
    report = {"condition": arguments.condition, "configuration_sha256": digest(arguments.config),
        "checkpoint_sha256": digest(checkpoint_path), "engineering_only": engineering, "planning_scenes": len(outputs["trained"]),
        "world_clips": len(world_rows), "planning_recordings": protocol["planning_recordings"],
        "means": {name: summarize_rows(rows) for name, rows in outputs.items()},
        "trained_minus_initial": initial_comparisons,
        "predicted_minus_persistent_future": paired_metrics(outputs["trained"], outputs["persistent_future"], ("pdms", "ade_meters")),
        "world_vs_stage1": world_comparisons, "checks": checks, "trend_checks_passed": all(checks.values()),
        "world_risk_breakdown": {risk: paired_metrics([row for row in world_rows if row["risk_flags"].get(risk)],
            world_references, ("reconstruction_lpips", "forecast_lpips"))
            for risk in sorted({name for row in world_rows for name, value in row["risk_flags"].items() if value})},
        "seconds": time.monotonic() - started, "scope": protocol["scope"],
        "limitations": ["One epoch and one seed, not convergence", "Initial reference is an untrained planner",
            "Partial tuning benefit is not isolated from planner learning without a separately trained frozen-LPWM control",
            "Small development panels; whole-dev and independent test are not completed"]}
    write_json(destination / "summary.json", report)
    write_json(PROJECT_ROOT / specification["shared_results_directory"] / (("engineering_" if engineering else "") + arguments.condition + "_trend_summary.json"), report)
    print("PARTIAL_TREND_EVALUATION_DONE", json.dumps(report["means"]), flush=True)


def summarize(config_path):
    specification = json.loads(config_path.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    no_objects, with_objects = specification["conditions"]
    panels = {condition: json.loads((root / condition / "trend_evaluation/trained.json").read_text()) for condition in specification["conditions"]}
    reports = {condition: json.loads((root / condition / "trend_evaluation/summary.json").read_text()) for condition in specification["conditions"]}
    comparison = paired_metrics(panels[with_objects], panels[no_objects], ("pdms", "ade_meters", "metric_bce"))
    lower = comparison["pdms"]["ci95"][0]
    supports = lower is not None and lower > 0 and reports[with_objects]["trend_checks_passed"]
    report = {"complete": True, "conditions": reports, "object_auxiliary_minus_no_auxiliary": comparison,
        "object_supervision_decision": "promising_trend_requires_full_validation" if supports else "benefit_not_established_keep_optional",
        "scope": "One full navtrain epoch per condition; fixed small development panels, no independent test or seed replication"}
    write_json(PROJECT_ROOT / specification["shared_results_directory"] / "trend_ablation_summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--engineering-checkpoint", type=Path)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    summarize(arguments.config) if arguments.summarize else evaluate(arguments)
