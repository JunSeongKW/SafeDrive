"""Independent causal/gradient audit; never used as a trained planning result."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from planning_aware_future_prediction.object_centric.lpwm_planning_finetuning import PlanningFineTunedLPWM
from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import build_planning_model, compute_candidate_losses, compute_refinement_losses
from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
from run_lpwm_navsim_posttraining import published_checkpoint, write_json
from train_lpwm_full_planning import make_planning_inputs, planning_loss, module_gradient_norms


def main(arguments):
    torch.set_num_threads(4)
    device = torch.device(arguments.device)
    torch.manual_seed(47)
    start = time.monotonic()
    root = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2"
    manifest = json.loads((root / "planning_manifest.json").read_text())
    records = manifest["records"]
    frames = np.load(root / "rgb_frames.npy", mmap_mode="r")
    with np.load(root / "planning_targets.npz") as stored:
        targets = {name: stored[name] for name in stored.files}
    indices = np.array([next(index for index, record in enumerate(records)
        if record["split"] == "train" and min(record["frame_cache_indices"]) >= 0)])
    full_video, status, target = make_planning_inputs(records, indices, frames, targets, device, include_future=True)
    checkpoint_path = arguments.checkpoint or published_checkpoint()
    specification = json.loads(arguments.config.read_text()) if arguments.config else None
    model = (build_planning_model(checkpoint_path, specification, arguments.condition, PROJECT_ROOT)
        if specification else PlanningFineTunedLPWM(checkpoint_path)).to(device).train()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(full_video[:, :4], status)
        if specification:
            teacher_root = PROJECT_ROOT / specification["teacher_directory"]
            if arguments.teacher_profile:
                segment_path = teacher_root / "profile/segments" / (Path(records[int(indices[0])]["segment_filename"]).stem + ".npz")
                with np.load(segment_path) as segment:
                    matches = np.flatnonzero(segment["indices"] == indices[0])
                    assert len(matches) == 1
                    teacher_metrics = torch.from_numpy(segment["labels"][matches].copy()).to(device)
            else:
                teacher = np.load(teacher_root / "candidate_metrics.npy", mmap_mode="r")
                teacher_metrics = torch.from_numpy(np.array(teacher[indices])).to(device)
            assert torch.isfinite(teacher_metrics).all()
            objective = compute_candidate_losses(output, target, model.trajectory_vocabulary, teacher_metrics)["objective"]
            if "refinement" in arguments.condition:
                from lpwm_refinement_oracle import RefinementOracleClient
                oracle_manifest = teacher_root / "metric_cache_manifest.json"
                if arguments.teacher_profile:
                    profile_entries = []
                    for path in (teacher_root / "profile/segments").glob("*.json"):
                        profile_entries.extend(json.loads(path.read_text())["entries"])
                    oracle_manifest = teacher_root / "profile/engineering_oracle_manifest.json"
                    write_json(oracle_manifest, profile_entries)
                oracle = RefinementOracleClient(oracle_manifest, arguments.output.parent / "engineering_refinement", 0, workers=2)
                try:
                    metric_targets, temporal_targets = oracle.score(indices, output["refined_candidates"].detach().float().cpu().numpy())
                finally:
                    oracle.close()
                assert np.isfinite(metric_targets).all() and np.isfinite(temporal_targets).all()
                objective = objective + compute_refinement_losses(output, target, status,
                    torch.from_numpy(metric_targets).to(device), torch.from_numpy(temporal_targets).to(device))["objective"]
        else:
            objective = planning_loss(output["trajectory"], target)
    auxiliary_gradient_audit = {}
    if specification and "object_future" in arguments.condition:
        from planning_aware_future_prediction.object_centric.lpwm_object_supervision import ObjectTargetCache, compute_object_auxiliary_losses
        cache = ObjectTargetCache(PROJECT_ROOT / specification["object_auxiliary"]["target_directory"])
        object_targets = cache.select_batch(indices, device)
        auxiliary = compute_object_auxiliary_losses(output, object_targets, specification["object_auxiliary"])
        assert auxiliary["valid_future_objects"] > 0
        auxiliary["future_state_loss"].backward(retain_graph=True)
        auxiliary_gradient_audit = module_gradient_norms(model)
        assert all(auxiliary_gradient_audit[name] > 0 and np.isfinite(auxiliary_gradient_audit[name])
            for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
        assert auxiliary_gradient_audit["rgb_decoder"] == 0
        model.zero_grad(set_to_none=True)
        objective = objective + auxiliary["objective"]
    objective.backward()
    gradients = module_gradient_norms(model)
    refinement_gradients = {}
    if "refinement" in arguments.condition:
        for name in ("future_refinement_decoder", "refinement_offset_head", "refined_metric_head", "temporal_safety_head"):
            parameters = getattr(model, name).parameters()
            refinement_gradients[name] = float(torch.stack([parameter.grad.float().square().sum() for parameter in parameters if parameter.grad is not None]).sum().sqrt())
        assert all(value > 0 and np.isfinite(value) for value in refinement_gradients.values())
    assert all(gradients[name] > 0 for name in ("image_encoder", "context", "dynamics", "planner_and_command"))
    assert gradients["rgb_decoder"] == 0
    optimizer = torch.optim.AdamW(model.optimizer_parameter_groups(1e-6, 3e-4), weight_decay=1e-4)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        first = model(full_video[:, :4], status)
        altered = full_video.clone()
        altered[:, 4:] = 1 - altered[:, 4:]
        second = model(altered[:, :4], status)
        future_difference = float((first["trajectory"] - second["trajectory"]).abs().max())
        assert future_difference < 1e-5
        logit_difference = float((first["metric_logits"] - second["metric_logits"]).abs().max()) if specification else future_difference
        if "refinement" in arguments.condition:
            logit_difference = max(logit_difference, float((first["refined_metric_logits"] - second["refined_metric_logits"]).abs().max()))
        assert logit_difference < 1e-5
        changed_command = status.clone()
        changed_command[:, :4] = status[:, :4].roll(1, -1)
        conditioned = model(full_video[:, :4], changed_command)
        intent_difference = float((first["observed_particle_attributes"] - conditioned["observed_particle_attributes"]).abs().mean())
        assert intent_difference > 0
    report = {"engineering_audit_only": True, "checkpoint": str(checkpoint_path), "device": str(device),
        "condition": arguments.condition,
        "source_sha256": {name: __import__("hashlib").sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in (
            "src/planning_aware_future_prediction/object_centric/lpwm_candidate_planner.py", "scripts/lpwm_refinement_oracle.py")},
        "planning_gradient_norms": gradients, "isolated_future_object_loss_gradient_norms": auxiliary_gradient_audit,
        "future_target_intervention_max_prediction_difference": future_difference,
        "refinement_gradient_norms": refinement_gradients,
        "future_target_intervention_max_metric_logit_difference": logit_difference,
        "intent_particle_mean_absolute_difference_after_one_diagnostic_update": intent_difference,
        "diagnostic_updates": 1, "diagnostic_weights_saved": False, "seconds": time.monotonic() - start}
    write_json(arguments.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--teacher-profile", action="store_true")
    parser.add_argument("--condition", default="metric_plus_world")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/lpwm_navtrain_planning_v1/engineering_cpu_audit.json")
    arguments = parser.parse_args()
    main(arguments)
