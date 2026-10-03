import torch
from test_drive_jepa_selective_patch_future import gradient_norm
from test_spatial_region_future import region_fixture

from planning_aware_future_prediction.models.region_future_research import (
    LocatedRegionFutureBridge,
    compute_history_reconstruction_losses,
    observed_history_motion_scores,
    replace_selected_current_regions,
)


def located_fixture(sparse=True):
    model, current, ego = region_fixture()
    model.patch_selector.region_budget = model.patch_selector.patch_budget = 2
    model.patch_selector.scorer.patch_budget = 2
    model.future_bridge = LocatedRegionFutureBridge(
        model.patch_selector.region_pooling_weights, 16, 16, 2, sparse)
    return model, current, ego


def test_located_bridge_zero_and_disabled_preserve_baseline():
    model, current, ego = located_fixture()
    with torch.no_grad():
        original = model.forward_cached_observations(current, ego, enable_future_branch=False)["trajectory"]
        assert torch.equal(original, model.forward_cached_observations(current, ego)["trajectory"])


def test_located_residual_only_updates_selected_memory_cell():
    model, current, ego = region_fixture()
    model.future_bridge = LocatedRegionFutureBridge(model.patch_selector.region_pooling_weights, 16, 16, 2)
    model.future_bridge.output_projection.weight.data.copy_(torch.eye(16))
    model.patch_selector.explicit_region_indices = torch.zeros(2, 1, dtype=torch.long)
    residual = model.forward_cached_observations(current, ego)["future_memory_residual"]
    assert residual[:, 0].abs().sum() > 0
    assert torch.count_nonzero(residual[:, 1]) == 0


def test_joint_slot_permutation_preserves_output_but_wrong_location_changes_it():
    from dataclasses import replace
    model, current, ego = located_fixture()
    bridge = model.future_bridge
    bridge.output_projection.weight.data.copy_(torch.eye(16))
    selection = model.patch_selector(current, ego, model.patch_coordinates)
    prediction = torch.randn(2, 2, 2, 16)
    current_memory = torch.randn(2, 2, 16)
    original = bridge(current_memory, prediction, selection.selected_coordinates, selection)
    reordered = replace(selection, selection_weights=selection.selection_weights.flip(1))
    aligned = bridge(current_memory, prediction.flip(1), selection.selected_coordinates.flip(1), reordered)
    misaligned = bridge(current_memory, prediction.flip(1), selection.selected_coordinates, selection)
    torch.testing.assert_close(original, aligned)
    assert not torch.allclose(original, misaligned)


def test_hard_sparse_edges_exclude_messages_and_gate_gets_gradient():
    model, current, ego = located_fixture()
    bridge = model.future_bridge
    bridge.output_projection.weight.data.copy_(torch.eye(16))
    bridge.relation_query.weight.data.zero_()
    bridge.relation_key.weight.data.zero_()
    bridge.relation_query.bias.data.fill_(1)
    bridge.relation_key.bias.data.fill_(-1)
    selection = model.patch_selector(current, ego, model.patch_coordinates)
    prediction = torch.randn(2, 2, 2, 16)
    current_memory = torch.randn(2, 2, 16)
    residual = bridge(current_memory, prediction, selection.selected_coordinates, selection)
    assert torch.equal(bridge.last_hard_edges, torch.eye(2, dtype=torch.bool)[None].expand(2, -1, -1))
    altered = prediction.clone()
    altered[:, 1] += torch.randn_like(altered[:, 1]) * 10
    changed = bridge(current_memory, altered, selection.selected_coordinates, selection)
    output_slot = selection.selected_region_indices[:, 0]
    torch.testing.assert_close(residual[torch.arange(2), output_slot], changed[torch.arange(2), output_slot])
    bridge.connection_sparsity_loss().backward()
    assert gradient_norm(bridge.relation_key) > 0


def test_history_mask_replaces_context_and_selected_shortcut():
    model, current, ego = located_fixture()
    selection = model.patch_selector(current, ego, model.patch_coordinates)
    earlier = torch.randn_like(current)
    masked_slots = torch.zeros(2, 1, dtype=torch.long)
    replaced, mask = replace_selected_current_regions(current, earlier, selection, masked_slots)
    assert torch.equal(replaced[mask], earlier[mask])
    assert torch.equal(replaced[~mask], current[~mask])
    torch.testing.assert_close((selection.hard_selection_weights @ replaced)[:, 0],
                               (selection.hard_selection_weights @ earlier)[:, 0])


def test_history_targets_reach_predictor_only_and_invalid_future_is_safe():
    model, current, ego = located_fixture()
    predictor = model.future_predictor
    predictor.current_time_embedding = torch.nn.Parameter(torch.zeros(1, 16))
    outputs = model.forward_cached_observations(current, ego)
    future_loss, reconstruction_loss = compute_history_reconstruction_losses(
        model, outputs, ego, torch.randn_like(current), torch.ones(2, dtype=torch.bool),
        torch.full((2, 2, 8, 16), torch.nan), torch.zeros(2, 2, 8, dtype=torch.bool),
        torch.zeros(2, 1, dtype=torch.long), True)
    assert float(future_loss) == 0
    (future_loss + reconstruction_loss).backward()
    assert gradient_norm(predictor) > 0
    assert gradient_norm(model.patch_selector) == gradient_norm(model.future_bridge) == 0
    assert gradient_norm(model.baseline_model) == 0


def test_motion_uses_observed_second_difference_and_localizes_change():
    frames = torch.zeros(4, 3, 8, 16)
    assert torch.count_nonzero(observed_history_motion_scores(frames, (2, 4))) == 0
    frames[3, :, :4, :4] = 1
    scores = observed_history_motion_scores(frames, (2, 4))
    assert scores.argmax() == 0
    assert torch.count_nonzero(scores) == 1


def test_detached_selection_blocks_located_scatter_gradient():
    model, current, ego = located_fixture()
    model.future_bridge.output_projection.weight.data.copy_(torch.eye(16))
    model.forward_cached_observations(current, ego, detach_selection=True)["trajectory"].square().mean().backward()
    assert gradient_norm(model.patch_selector) == 0
    assert gradient_norm(model.future_bridge) > 0
