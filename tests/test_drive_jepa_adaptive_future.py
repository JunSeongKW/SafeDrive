"""Architecture contracts; actual official tail/batch checks are in the runner."""

import inspect

import pytest
import torch
from torch import nn
from test_drive_jepa_selective_patch_future import FixturePlanner, gradient_norm

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    ContextualResidualFuturePredictor,
    DriveJEPAAdaptiveFuture,
    EgoQueryPatchSelector,
    LowRankAdaptedLinear,
)


def make_graph():
    torch.manual_seed(29)
    graph = DriveJEPAAdaptiveFuture(
        FixturePlanner(), latent_dim=16, hidden_dim=24, patch_budget=2,
        future_tubelet_count=2, grid_height=2, grid_width=4,
    ).eval()
    return graph, torch.randn(2, 8, 16), torch.randn(2, 8)


def test_residual_predictor_initializes_to_exact_current_persistence():
    graph, current, status = make_graph()
    with torch.no_grad():
        result = graph.forward_cached_observations(current, status)
        disabled = graph.forward_cached_observations(current, status, enable_future_branch=False)
    expected = result["patch_selection"].hard_selection_weights @ current
    assert torch.equal(result["predicted_future_latents"], expected[:, :, None].expand(-1, -1, 2, -1))
    assert torch.equal(result["trajectory"], disabled["trajectory"])


def test_lora_initial_identity_and_nonzero_output_factor_gradient():
    torch.manual_seed(29)
    linear = nn.Linear(16, 48)
    lora = LowRankAdaptedLinear(linear)
    observed = torch.randn(2, 8, 16)
    assert torch.equal(linear(observed), lora(observed))
    lora(observed).square().mean().backward()
    assert gradient_norm(linear) == 0
    assert lora.output_factor.grad.norm() > 0
    assert lora.input_factor.grad.norm() == 0  # zero B is not a disconnected graph


def test_contextual_predictor_uses_unselected_current_context_after_delta_opens():
    predictor = ContextualResidualFuturePredictor(16, 24, 2).eval()
    nn.init.normal_(predictor.delta_head.weight, std=0.01)
    current = torch.randn(2, 8, 16, requires_grad=True)
    selected = current[:, :2]
    coordinates = torch.randn(8, 2)
    predictor(selected, coordinates[:2].expand(2, -1, -1), current,
              torch.randn(2, 8), coordinates).square().mean().backward()
    assert current.grad[:, 2:].norm() > 0


def test_ego_query_scores_are_order_equivariant_and_unique():
    torch.manual_seed(29)
    selector = EgoQueryPatchSelector(16, 24, 3)
    current, status, coordinates = torch.randn(2, 8, 16), torch.randn(2, 8), torch.randn(8, 2)
    permutation = torch.randperm(8)
    original = selector(current, status, coordinates)
    permuted = selector(current[:, permutation], status, coordinates[permutation])
    assert torch.equal(original.selected_patch_indices, permutation[permuted.selected_patch_indices])
    assert all(row.unique().numel() == 3 for row in original.selected_patch_indices)
    original.selected_coordinates.square().mean().backward()
    assert gradient_norm(selector.intent_query) > 0


def test_adapted_context_aux_gradient_not_policy_or_teacher_gradient():
    graph, current, status = make_graph()
    adapted = current.clone().requires_grad_(True)
    target = torch.randn(2, 2, 8, 16, requires_grad=True)
    output = graph.forward_from_current_patch_latents(current, status, current_prediction_context_latents=adapted)
    graph.compute_future_auxiliary_loss(output, status, target, torch.ones(2, 2, 8, dtype=torch.bool)).backward()
    assert adapted.grad.norm() > 0
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_bridge) == 0
    assert gradient_norm(graph.future_predictor) > 0
    assert gradient_norm(graph.baseline_model) == 0
    assert target.grad is None


def test_planning_and_detach_gradient_contracts():
    graph, current, status = make_graph()
    nn.init.normal_(graph.future_bridge.output_projection.weight, std=0.01)
    nn.init.normal_(graph.future_predictor.delta_head.weight, std=0.01)
    output = graph.forward_cached_observations(current, status)
    output["trajectory"].square().mean().backward()
    assert gradient_norm(graph.patch_selector) > 0
    assert gradient_norm(graph.future_predictor) > 0
    assert gradient_norm(graph.baseline_model) == 0
    graph.zero_grad(set_to_none=True)
    detached = graph.forward_cached_observations(current, status, detach_predicted_future=True)
    assert torch.equal(output["trajectory"], detached["trajectory"])
    detached["trajectory"].square().mean().backward()
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) == 0
    assert gradient_norm(graph.future_bridge) > 0


def test_no_future_gt_forward_and_invalid_mask_safety():
    graph, current, status = make_graph()
    assert "future_target_latents" not in inspect.signature(graph.forward_cached_observations).parameters
    output = graph.forward_cached_observations(current, status)
    target = torch.full((2, 2, 8, 16), float("nan"))
    mask = torch.zeros(2, 2, 8, dtype=torch.bool)
    assert graph.compute_future_auxiliary_loss(output, status, target, mask).item() == 0
    with pytest.raises(ValueError):
        graph.compute_future_auxiliary_loss(output, status, target, mask.float())
