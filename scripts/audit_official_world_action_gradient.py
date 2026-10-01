"""Official WA-JEPA attention CPU boundary check, NOT full checkpoint/model reproduction."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = PROJECT_ROOT / "reference_repositories/WA-JEPA"
sys.path.insert(0, str(REFERENCE_ROOT))
from models.multiview_causal_future_jepa import ModalitySpecificJointAttention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output_file.exists():
        raise ValueError("do not replace previous audit")
    arguments.output_file.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REFERENCE_ROOT, text=True
        ).strip(),
        "scope": "actual official ModalitySpecificJointAttention, tiny CPU tensors; no flow sampler or visual encoder/checkpoint",
        "checks": [],
    }
    for coupled in (False, True):
        torch.manual_seed(29)
        attention = ModalitySpecificJointAttention(
            16, 4, traj_loss_grad_to_scene_flow=coupled
        )
        context = torch.randn(2, 3, 16, requires_grad=True)
        scene = torch.randn(2, 5, 16, requires_grad=True)
        trajectory = torch.randn(2, 2, 16, requires_grad=True)
        _, _, predicted_trajectory = attention(context, scene, trajectory)
        predicted_trajectory.square().mean().backward()
        scene_gradient = 0.0 if scene.grad is None else float(scene.grad.norm())
        report["checks"].append(
            {
                "traj_loss_grad_to_scene_flow": coupled,
                "trajectory_loss_to_scene_input_gradient_norm": scene_gradient,
                "trajectory_loss_to_scene_qkv_gradient_norm": 0.0
                if attention.scene_proj.qkv.weight.grad is None
                else float(attention.scene_proj.qkv.weight.grad.norm()),
            }
        )
        if (scene_gradient > 0) != coupled:
            raise RuntimeError("upstream gradient flag violated")
    report["wall_seconds"] = time.perf_counter() - started
    arguments.output_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
