"""CPU probes of pinned public safety-loss kernels, without loading full planners.

Only framework registration/reduction decorators are removed from the selected
AST nodes. Loss mathematics and tensor construction remain unchanged. This is
an autograd diagnostic, not a driving benchmark or a reproduction of training.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIRECTORY = PROJECT_ROOT / "results/e2e_planner_safety_audit_20261004"


def load_selected_definitions(source_path, definition_names):
    parsed_source = ast.parse(source_path.read_text())
    definitions = []
    for definition in parsed_source.body:
        if isinstance(definition, (ast.FunctionDef, ast.ClassDef)) and definition.name in definition_names:
            definition.decorator_list = []
            definitions.append(definition)
    assert {definition.name for definition in definitions} == set(definition_names)
    module = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def summarize_gradient_step(loss_function, initial_offsets):
    predicted_offsets = torch.tensor(initial_offsets, dtype=torch.float32, requires_grad=True)
    loss = loss_function(predicted_offsets).mean()
    gradient = torch.autograd.grad(loss, predicted_offsets)[0]
    adjusted_offsets = predicted_offsets.detach() - 0.05 * gradient
    adjusted_loss = loss_function(adjusted_offsets).mean()
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    assert adjusted_loss < loss
    return {
        "loss_before": float(loss.detach()),
        "loss_after_one_coordinate_step": float(adjusted_loss),
        "coordinate_gradient": gradient.tolist(),
        "gradient_norm": float(gradient.norm()),
        "coordinate_step_size": 0.05,
        "scope": "synthetic two-step geometry; official undecorated loss kernel only",
    }


def main():
    torch.set_num_threads(1)
    manifest = json.loads((RESULT_DIRECTORY / "source_manifest.json").read_text())
    for source in manifest["sources"].values():
        for entry in source["files"]:
            assert hashlib.sha256((PROJECT_ROOT / entry["local_path"]).read_bytes()).hexdigest() == entry["sha256"]

    source_root = PROJECT_ROOT / "reference_repositories/E2ESafetyGradientAudit"
    vad_source = source_root / "VAD/projects/mmdet3d_plugin/VAD/utils/plan_loss.py"
    vad = load_selected_definitions(vad_source, ("plan_col_loss", "plan_map_bound_loss", "segments_intersect"))

    def collision_objective(predicted_offsets):
        return vad["plan_col_loss"](predicted_offsets, torch.tensor([[[1.0, 1.5]]]), torch.zeros(1, 1, 2, 2))

    def boundary_objective(predicted_offsets):
        boundary = torch.tensor([[[[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]]])
        return vad["plan_map_bound_loss"](predicted_offsets, boundary)

    collision = summarize_gradient_step(collision_objective, [[[0.2, 0.2], [0.3, 0.1]]])
    boundary = summarize_gradient_step(boundary_objective, [[[0.4, 0.2], [0.1, 0.2]]])

    predicted_ego = torch.tensor([[[0.2, 0.2], [0.3, 0.1]]], requires_grad=True)
    predicted_agent = torch.tensor([[[1.0, 1.5]]], requires_grad=True)
    predicted_agent_motion = torch.zeros(1, 1, 2, 2, requires_grad=True)
    coupled_loss = vad["plan_col_loss"](predicted_ego, predicted_agent, predicted_agent_motion).mean()
    coupled_gradients = torch.autograd.grad(coupled_loss, (predicted_ego, predicted_agent, predicted_agent_motion))

    uniad_source = source_root / "UniAD/projects/mmdet3d_plugin/losses/planning_loss.py"
    collision_module = load_selected_definitions(uniad_source, ("CollisionLoss",))["CollisionLoss"](delta=0.0)
    object_corners = torch.zeros(1, 8, 3)
    object_corners[0, [0, 3, 4, 7], :2] = torch.tensor([[0.5, 0.5], [0.5, 2.5], [2.5, 2.5], [2.5, 0.5]])
    object_box = SimpleNamespace(tensor=torch.zeros(1, 7), corners=object_corners)
    predicted_trajectory = torch.tensor([[[1.0, 1.0]]], requires_grad=True)

    def uniad_objective(trajectory):
        return collision_module(trajectory, torch.zeros(1, 1, 3), torch.ones(1, 1), [object_box])

    uniad_loss = uniad_objective(predicted_trajectory)
    epsilon = 0.001
    positive_shift = predicted_trajectory.detach().clone()
    negative_shift = predicted_trajectory.detach().clone()
    positive_shift[..., 0] += epsilon
    negative_shift[..., 0] -= epsilon
    numerical_gradient = (uniad_objective(positive_shift) - uniad_objective(negative_shift)) / (2 * epsilon)
    assert uniad_loss > 0 and not uniad_loss.requires_grad
    assert numerical_gradient.abs() > 0

    result = {
        "torch_version": torch.__version__,
        "device": "cpu",
        "full_model_training_or_evaluation": False,
        "source_manifest": str((RESULT_DIRECTORY / "source_manifest.json").relative_to(PROJECT_ROOT)),
        "vad_collision": collision,
        "vad_boundary": boundary,
        "vad_collision_coupled_gradient_norms": dict(zip(
            ("ego_offsets", "predicted_agent_position", "predicted_agent_future_offsets"),
            [float(gradient.norm()) for gradient in coupled_gradients],
        )),
        "uniad_collision": {
            "commit": manifest["sources"]["UniAD"]["commit"],
            "loss": float(uniad_loss),
            "loss_requires_grad": uniad_loss.requires_grad,
            "numerical_x_derivative": float(numerical_gradient),
            "cause": "CollisionLoss.to_corners reconstructs predicted center via torch.tensor(bbox[:2])",
            "scope": "downloaded public implementation only; not evidence about authors' historical training",
        },
    }
    (RESULT_DIRECTORY / "cpu_gradient_probes.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
