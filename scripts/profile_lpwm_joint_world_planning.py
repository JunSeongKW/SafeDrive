"""Real-batch gradient, causality and memory gates for joint LPWM world planning."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from planning_aware_future_prediction.object_centric.lpwm_joint_world_planner import JointLPWMWorldPlanner
from planning_aware_future_prediction.object_centric.lpwm_planner import compute_planning_objectives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    arguments = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_device(arguments.gpu)
    device = f"cuda:{arguments.gpu}"
    torch.manual_seed(29)
    started = time.monotonic()
    model = JointLPWMWorldPlanner(PROJECT_ROOT / "outputs/lpwm_navsim_adaptation_v1/runs/raw_seed29/checkpoint.pt").to(device).train()
    parameter_groups = model.world_parameter_groups()
    optimizer = torch.optim.AdamW([{"params": parameters, "lr": 3e-4 if name == "planner_and_intent" else 2e-6} for name, parameters in parameter_groups.items()])
    cache = torch.load(PROJECT_ROOT / "outputs/lpwm_planning_v1/supervised_cache.pt", map_location="cpu", weights_only=True)
    indices = list(range(arguments.batch_size))
    images = cache["observed_images"][indices].to(device).permute(0, 1, 4, 2, 3).float() / 255
    status = cache["ego_status"][indices].to(device)
    supervision = {key: value[indices].to(device) for key, value in cache.items() if key not in ("observed_images", "ego_status")}
    records = []
    for update in range(3):
        step_started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=arguments.amp):
            prediction = model(images, status)
            losses = compute_planning_objectives(prediction, supervision, "object_future_risk", .5)
        gradients = {}
        if update == 2:
            for name in ("image_encoder", "context", "dynamics"):
                selected = parameter_groups[name]
                for objective in ("planning", "future_state"):
                    values = torch.autograd.grad(losses[objective], selected, retain_graph=True, allow_unused=True)
                    norm = sum(float(value.detach().square().sum()) for value in values if value is not None)**.5
                    gradients[name + "/" + objective] = norm
                    assert norm > 0, (name, objective, norm)
                    del values
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(arguments.gpu)
        row = {"update": update + 1, "seconds": time.monotonic() - step_started, "total_loss": float(losses["total"].detach()),
               "peak_allocated_gib": torch.cuda.max_memory_allocated(arguments.gpu) / 1024**3, "gradients": gradients}
        records.append(row)
        print(json.dumps(row), flush=True)
        del prediction, losses
    model.eval()
    with torch.no_grad():
        original = model(images, status)
        for value in supervision.values():
            if value.is_floating_point():
                value.fill_(10000)
        repeated = model(images, status)
        assert torch.equal(original["trajectory"], repeated["trajectory"])
        assert torch.equal(original["predicted_future_attributes"], repeated["predicted_future_attributes"])
        changed = model(images, status, "future_particle_persistence")
    result = {"batch_size": arguments.batch_size, "bf16_autocast": arguments.amp, "future_particle_steps": 8, "observed_image_steps": 2,
              "unique_parameters_by_module": {name: sum(parameter.numel() for parameter in parameters) for name, parameters in parameter_groups.items()},
              "steps": records, "future_label_intervention_max_difference": 0.,
              "future_particle_persistence_trajectory_difference_m": float((changed["trajectory"][..., :2] - original["trajectory"][..., :2]).norm(dim=-1).mean()),
              "total_seconds": time.monotonic() - started}
    output = PROJECT_ROOT / "outputs/lpwm_joint_world_planning_v1"
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_bf16" if arguments.amp else ""
    (output / f"profile_batch{arguments.batch_size}{suffix}.json").write_text(json.dumps(result, indent=2) + "\n")
    print("JOINT_WORLD_PROFILE_PASSED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
