"""CPU contract tests; official real-image execution is a separate gate."""

import inspect

import pytest
import torch
from torch import nn

from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
    DriveJEPASelectivePatchFuture,
)


class FixturePatchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Conv2d(3, 16, 16, stride=16)

    def forward(self, camera_clip):
        return self.projection(camera_clip.mean(dim=2)).flatten(2).transpose(1, 2)


class FixtureTrajectoryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(16, 3)

    def forward(self, queries):
        return {"trajectory": self.projection(queries)}


class FixturePlanner(nn.Module):
    def __init__(self):
        super().__init__()
        self.transform = nn.Identity()
        self.image_encoder = FixturePatchEncoder()
        self.avg_pool = nn.AvgPool2d(2, 2)
        self.image_fc = nn.Linear(16, 16)
        self._status_encoding = nn.Linear(8, 16)
        self._keyval_embedding = nn.Embedding(3, 16)
        self._query_embedding = nn.Embedding(2, 16)
        self._transformer = nn.Transformer(16, 8, 1, 1, 32, dropout=0, batch_first=True)
        self._trajectory_head = FixtureTrajectoryHead()

    def forward(self, camera_clip, ego_status):
        latents = self.image_encoder(camera_clip)
        image_grid = latents.transpose(1, 2).reshape(-1, 16, 2, 4)
        image_memory = self.image_fc(
            self.avg_pool(image_grid).flatten(2).transpose(1, 2).clone()
        )
        status_memory = self._status_encoding(ego_status)[:, None]
        memory = torch.cat((image_memory, status_memory), dim=1)
        memory = memory.clone() + self._keyval_embedding.weight[None]
        queries = self._query_embedding.weight[None].repeat(camera_clip.shape[0], 1, 1)
        return self._trajectory_head(self._transformer(src=memory, tgt=queries))


@pytest.fixture
def graph_inputs():
    torch.manual_seed(29)
    graph = DriveJEPASelectivePatchFuture(
        FixturePlanner(), 16, 24, 2, 2, grid_height=2, grid_width=4
    ).eval()
    return graph, torch.randn(2, 3, 2, 32, 64), torch.randn(2, 8)


def gradient_norm(module):
    return (
        sum(
            float(parameter.grad.square().sum())
            for parameter in module.parameters()
            if parameter.grad is not None
        )
        ** 0.5
    )


def open_residual(graph, images, status):
    optimizer = torch.optim.SGD(
        [parameter for parameter in graph.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    graph(images, status)["trajectory"].square().mean().backward()
    optimizer.step()
    graph.zero_grad(set_to_none=True)


def test_disabled_and_zero_initialized_branch_preserve_output(graph_inputs):
    graph, images, status = graph_inputs
    with torch.no_grad():
        original = graph.baseline_model(images, status)["trajectory"]
        disabled = graph(images, status, False)["trajectory"]
        enabled = graph(images, status)["trajectory"]
    assert torch.equal(original, disabled)
    assert torch.equal(original, enabled)


def test_cached_current_interface_preserves_original_and_online_outputs(graph_inputs):
    graph, images, status = graph_inputs
    with torch.no_grad():
        latents = graph.encode_observed_clip(images)
        original = graph.baseline_model(images, status)["trajectory"]
        for enabled in (False, True):
            cached = graph.forward_from_current_patch_latents(latents, status, enabled)
            assert torch.equal(original, cached["trajectory"])
    open_residual(graph, images, status)
    assert torch.equal(
        graph(images, status)["trajectory"],
        graph.forward_from_current_patch_latents(latents, status)["trajectory"],
    )


def test_explicit_policy_ids_bypass_selector_but_train_predictor(graph_inputs):
    graph, images, status = graph_inputs
    open_residual(graph, images, status)
    latents = graph.encode_observed_clip(images)
    selected_ids = torch.tensor([[1, 5], [0, 7]])
    output = graph.forward_from_current_patch_latents(
        latents, status, selected_patch_indices=selected_ids
    )
    assert torch.equal(output["patch_selection"].selected_patch_indices, selected_ids)
    output["trajectory"].square().mean().backward()
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) > 0
    for invalid in (
        torch.tensor([[1, 1], [0, 7]]),
        selected_ids + 8,
        selected_ids.float(),
    ):
        with pytest.raises(ValueError):
            graph.forward_from_current_patch_latents(
                latents, status, selected_patch_indices=invalid
            )


def test_singleton_batch_grid_stride_matches_official_einops(graph_inputs):
    graph, images, status = graph_inputs
    # Official ViT output is contiguous B,N,D; the convolution fixture is not.
    latents = graph.encode_observed_clip(images[:1]).contiguous()
    observed_strides = []
    handle = graph.baseline_model.avg_pool.register_forward_pre_hook(
        lambda module, inputs: observed_strides.append(inputs[0].stride())
    )
    with torch.no_grad():
        graph.forward_from_current_patch_latents(latents, status[:1])
    handle.remove()
    expected_grid = latents.reshape(1, 2, 4, 16).permute(0, 3, 1, 2)
    assert observed_strides == [expected_grid.stride()]
    assert expected_grid.stride(0) == latents.shape[1] * latents.shape[2]


def test_unique_hard_ids_and_sparse_prediction_queries(graph_inputs):
    graph, images, status = graph_inputs
    result = graph(images, status)
    selected = result["patch_selection"]
    assert torch.equal(selected.selection_weights, selected.hard_selection_weights)
    assert all(row.unique().numel() == 2 for row in selected.selected_patch_indices)
    assert graph.future_predictor.last_prediction_query_shape[:3] == (2, 2, 2)
    assert result["predicted_future_latents"].shape == (2, 2, 2, 16)


def test_initial_zero_gradient_is_not_disconnection(graph_inputs):
    graph, images, status = graph_inputs
    graph(images, status)["trajectory"].square().mean().backward()
    assert gradient_norm(graph.future_bridge.output_projection) > 0
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) == 0


def test_planning_reaches_selector_predictor_after_one_step(graph_inputs):
    graph, images, status = graph_inputs
    open_residual(graph, images, status)
    graph(images, status)["trajectory"].square().mean().backward()
    assert gradient_norm(graph.patch_selector) > 0
    assert gradient_norm(graph.future_predictor) > 0
    assert gradient_norm(graph.future_bridge) > 0
    assert gradient_norm(graph.baseline_model) == 0


def test_auxiliary_blocks_selector_bridge_and_target_gradients(graph_inputs):
    graph, images, status = graph_inputs
    result = graph(images, status)
    target = torch.randn(2, 2, 8, 16, requires_grad=True)
    valid = torch.ones(2, 2, 8, dtype=torch.bool)
    graph.compute_future_auxiliary_loss(result, status, target, valid).backward()
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) > 0
    assert gradient_norm(graph.future_bridge) == 0
    assert target.grad is None


def test_future_detach_preserves_forward_and_blocks_both_routes(graph_inputs):
    graph, images, status = graph_inputs
    open_residual(graph, images, status)
    original = graph(images, status)["trajectory"]
    detached = graph(images, status, detach_predicted_future=True)["trajectory"]
    assert torch.equal(original, detached)
    detached.square().mean().backward()
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) == 0
    assert gradient_norm(graph.future_bridge) > 0


def test_selection_detach_blocks_only_selection(graph_inputs):
    graph, images, status = graph_inputs
    open_residual(graph, images, status)
    detached = graph(images, status, detach_selection=True)
    detached["trajectory"].square().mean().backward()
    assert gradient_norm(graph.patch_selector) == 0
    assert gradient_norm(graph.future_predictor) > 0


def test_missing_nan_targets_zero_loss_and_do_not_change_selection(graph_inputs):
    graph, images, status = graph_inputs
    result = graph(images, status)
    selected_before = result["patch_selection"].selected_patch_indices.clone()
    loss = graph.compute_future_auxiliary_loss(
        result,
        status,
        torch.full((2, 2, 8, 16), float("nan")),
        torch.zeros(2, 2, 8, dtype=torch.bool),
    )
    assert torch.isfinite(loss) and loss.item() == 0
    loss.backward()
    assert gradient_norm(graph.future_predictor) == 0
    assert torch.equal(
        selected_before, graph(images, status)["patch_selection"].selected_patch_indices
    )


def test_forward_has_no_future_gt_interface_and_planner_stays_frozen(graph_inputs):
    graph, _, _ = graph_inputs
    parameters = inspect.signature(graph.forward).parameters
    assert "future_target_latents" not in parameters
    assert "future_target_valid_mask" not in parameters
    graph.train()
    assert not graph.baseline_model.training
    assert not any(
        parameter.requires_grad for parameter in graph.baseline_model.parameters()
    )


def test_invalid_budget_and_invalid_target_mask_rejected(graph_inputs):
    graph, images, status = graph_inputs
    graph.patch_selector.patch_budget = 9
    with pytest.raises(ValueError, match="budget"):
        graph(images, status)
    graph.patch_selector.patch_budget = 2
    with pytest.raises(ValueError, match="mask"):
        graph.compute_future_auxiliary_loss(
            graph(images, status),
            status,
            torch.randn(2, 2, 8, 16),
            torch.ones(2, 2, 8),
        )


def test_target_broadcasting_and_nonfinite_valid_targets_rejected(graph_inputs):
    graph, images, status = graph_inputs
    result = graph(images, status)
    with pytest.raises(ValueError, match="shape"):
        graph.compute_future_auxiliary_loss(
            result,
            status,
            torch.randn(2, 1, 8, 16),
            torch.ones(2, 1, 8, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="finite"):
        graph.compute_future_auxiliary_loss(
            result,
            status,
            torch.full((2, 2, 8, 16), float("nan")),
            torch.ones(2, 2, 8, dtype=torch.bool),
        )
