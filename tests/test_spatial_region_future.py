import pytest
import torch
from test_drive_jepa_selective_patch_future import FixturePlanner, gradient_norm

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    DriveJEPAAdaptiveFuture,
)
from planning_aware_future_prediction.models.spatial_region_future import (
    SpatialRegionSelector,
    compute_planner_retention_loss,
    compute_region_future_auxiliary_loss,
    pooled_future_targets,
)


def region_fixture(region_side=2):
    torch.manual_seed(29)
    model = DriveJEPAAdaptiveFuture(
        FixturePlanner(),
        latent_dim=16,
        hidden_dim=16,
        patch_budget=1,
        future_tubelet_count=2,
        grid_height=2,
        grid_width=4,
    ).eval()
    model.patch_selector = SpatialRegionSelector(
        model.patch_selector, 2, 4, region_side, 1
    )
    return model, torch.randn(2, 8, 16), torch.randn(2, 8)


def test_region_pooling_is_nonoverlapping_average():
    model, current, ego = region_fixture()
    pooling = model.patch_selector.region_pooling_weights
    assert pooling.shape == (2, 8)
    assert torch.equal(pooling.sum(1), torch.ones(2))
    assert torch.equal((pooling > 0).sum(0), torch.ones(8, dtype=torch.long))
    selection = model.patch_selector(current, ego, model.patch_coordinates)
    assert torch.equal(selection.selection_weights, selection.hard_selection_weights)
    assert (selection.hard_selection_weights > 0).sum() == 8


def test_initial_and_disabled_region_path_preserve_planner():
    model, current, ego = region_fixture()
    with torch.no_grad():
        original = model.forward_cached_observations(
            current, ego, enable_future_branch=False
        )["trajectory"]
        assert torch.equal(
            original, model.forward_cached_observations(current, ego)["trajectory"]
        )


def test_retention_reaches_selector_only():
    model, current, ego = region_fixture()
    outputs = model.forward_cached_observations(current, ego)
    compute_planner_retention_loss(
        model, current, ego, outputs["patch_selection"]
    ).backward()
    assert gradient_norm(model.patch_selector) > 0
    assert gradient_norm(model.future_predictor) == 0
    assert gradient_norm(model.future_bridge) == 0
    assert gradient_norm(model.baseline_model) == 0


def test_region_auxiliary_blocks_selector_and_requires_every_valid_member():
    model, current, ego = region_fixture()
    model.patch_selector.explicit_region_indices = torch.zeros(2, 1, dtype=torch.long)
    outputs = model.forward_cached_observations(current, ego)
    targets = torch.randn(2, 2, 8, 16)
    valid = torch.ones(2, 2, 8, dtype=torch.bool)
    valid[0, 0, 0] = False
    targets[0, 0, 0] = torch.nan
    pooled, selected_valid = pooled_future_targets(
        outputs["patch_selection"], targets, valid
    )
    assert torch.isfinite(pooled).all()
    assert selected_valid.sum() == 3
    compute_region_future_auxiliary_loss(model, outputs, ego, targets, valid).backward()
    assert gradient_norm(model.future_predictor) > 0
    assert (
        gradient_norm(model.patch_selector) == gradient_norm(model.future_bridge) == 0
    )


def test_empty_future_mask_zero_loss():
    model, current, ego = region_fixture()
    outputs = model.forward_cached_observations(current, ego)
    loss = compute_region_future_auxiliary_loss(
        model,
        outputs,
        ego,
        torch.full((2, 2, 8, 16), torch.nan),
        torch.zeros(2, 2, 8, dtype=torch.bool),
    )
    assert float(loss.detach()) == 0
    loss.backward()


def test_explicit_region_ids_reject_invalid():
    model, current, ego = region_fixture()
    model.patch_selector.explicit_region_indices = torch.full((2, 1), 2)
    with pytest.raises(ValueError, match="outside"):
        model.forward_cached_observations(current, ego)


def test_side_one_matches_original_selector():
    model, current, ego = region_fixture(region_side=1)
    region = model.patch_selector(current, ego, model.patch_coordinates)
    native = model.patch_selector.scorer(current, ego, model.patch_coordinates)
    assert torch.equal(region.selection_weights, native.selection_weights)
    assert torch.equal(region.selected_region_indices, native.selected_patch_indices)
