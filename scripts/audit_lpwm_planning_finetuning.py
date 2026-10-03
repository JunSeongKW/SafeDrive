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
    model = PlanningFineTunedLPWM(checkpoint_path).to(device).train()
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        output = model(full_video[:, :4], status)
        objective = planning_loss(output["trajectory"], target)
    objective.backward()
    gradients = module_gradient_norms(model)
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
        changed_command = status.clone()
        changed_command[:, :4] = status[:, :4].roll(1, -1)
        conditioned = model(full_video[:, :4], changed_command)
        intent_difference = float((first["observed_particle_attributes"] - conditioned["observed_particle_attributes"]).abs().mean())
    report = {"engineering_audit_only": True, "checkpoint": str(checkpoint_path), "device": str(device),
        "planning_gradient_norms": gradients, "future_target_intervention_max_prediction_difference": future_difference,
        "intent_particle_mean_absolute_difference_after_one_diagnostic_update": intent_difference,
        "diagnostic_updates": 1, "diagnostic_weights_saved": False, "seconds": time.monotonic() - start}
    write_json(arguments.output, report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/lpwm_navtrain_planning_v1/engineering_cpu_audit.json")
    arguments = parser.parse_args()
    main(arguments)
