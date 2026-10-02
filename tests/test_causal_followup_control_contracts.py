"""Check new runner routing on CPU; real-model gradients are logged per run."""

import sys
from pathlib import Path

import torch
from test_drive_jepa_adaptive_future import make_graph
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_drive_jepa_causal_followup import (
    gradient_contract,
    resolve_joint_future_auxiliary_weight,
)


class DiagnosticAgent:
    def compute_loss(self, features, targets, predictions):
        return (predictions["trajectory"] - targets["trajectory"]).square().mean()


def test_joint_auxiliary_control_weight_does_not_claim_absent_warmup():
    specification = {"future_auxiliary_weight": 0.01}
    assert resolve_joint_future_auxiliary_weight(specification, {}, False) == 0.01
    assert (
        resolve_joint_future_auxiliary_weight(
            specification, {"joint_auxiliary_weight_scale": 0}, False
        )
        == 0
    )
    assert resolve_joint_future_auxiliary_weight(specification, {}, True) == 0


def make_observed_batch(graph, current, status):
    result = graph.forward_cached_observations(current, status)
    return {
        "current_patch_latents": current,
        "current_ego_status": status,
        "future_target_latents": torch.randn(2, 2, 8, 16),
        "future_target_valid_mask": torch.ones(2, 2, 8, dtype=torch.bool),
        "ego_trajectory_target": torch.randn_like(result["trajectory"]),
    }


def test_runner_current_feature_control_has_no_predictor_parameter_gradient():
    graph, current, status = make_graph()
    nn.init.normal_(graph.future_bridge.output_projection.weight, std=0.01)
    graph.future_predictor.requires_grad_(False)
    report = gradient_contract(
        graph,
        DiagnosticAgent(),
        make_observed_batch(graph, current, status),
        "current_feature_control",
        {},
    )
    assert report["planning"]["patch_selector"] > 0
    assert report["planning"]["future_predictor"] == 0
    assert all(value == 0 for value in report["auxiliary"].values())


def test_runner_explicit_selection_control_keeps_predictor_but_not_policy_gradients():
    graph, current, status = make_graph()
    nn.init.normal_(graph.future_bridge.output_projection.weight, std=0.01)
    nn.init.normal_(graph.future_predictor.delta_head.weight, std=0.01)
    graph.patch_selector.requires_grad_(False)
    ids = torch.tensor([[1, 3], [0, 4]], dtype=torch.long)
    report = gradient_contract(
        graph,
        DiagnosticAgent(),
        make_observed_batch(graph, current, status),
        "fixed_lattice",
        {},
        ids,
    )
    assert report["planning"]["patch_selector"] == 0
    assert report["planning"]["future_predictor"] > 0
    assert report["auxiliary"]["future_predictor"] > 0
