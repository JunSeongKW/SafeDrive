"""Current-only region selection on an unchanged, frozen Drive-JEPA planner.

Regions pool neighbouring native latent tokens, not independent pixel crops or
object instances. Teacher retention is a current-planner reliance proxy; it is
not a label of causal importance or future prediction utility.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SpatialRegionSelection:
    selection_weights: torch.Tensor  # ST region weights expanded to native grid
    hard_selection_weights: torch.Tensor  # region averaging weights on native grid
    selected_region_indices: torch.Tensor
    selected_coordinates: torch.Tensor


class SpatialRegionSelector(nn.Module):
    """Non-overlapping square regions; preserve all current planner inputs.

    explicit_region_indices is a transient inference control, never a parameter
    or a future-validity mask. Region IDs must not be passed as native patch IDs.
    """

    def __init__(self, selector, grid_height, grid_width, region_side, region_budget):
        super().__init__()
        if region_side < 1 or grid_height % region_side or grid_width % region_side:
            raise ValueError("Region side must divide both native grid dimensions")
        region_count = (grid_height // region_side) * (grid_width // region_side)
        if not 1 <= region_budget <= region_count:
            raise ValueError("Region budget outside candidate count")
        self.scorer = selector
        self.scorer.patch_budget = region_budget
        self.patch_budget = region_budget  # inherited predictor API compatibility
        self.region_budget = region_budget
        self.region_side = region_side
        self.region_count = region_count
        self.explicit_region_indices = None
        pooling = torch.zeros(region_count, grid_height * grid_width)
        native_grid = torch.arange(grid_height * grid_width).reshape(
            grid_height, grid_width
        )
        region_index = 0
        for row in range(0, grid_height, region_side):
            for column in range(0, grid_width, region_side):
                native_ids = native_grid[
                    row : row + region_side, column : column + region_side
                ].flatten()
                pooling[region_index, native_ids] = 1 / region_side**2
                region_index += 1
        self.register_buffer("region_pooling_weights", pooling)

    def forward(self, current_patch_latents, current_ego_status, patch_coordinates):
        region_features = self.region_pooling_weights @ current_patch_latents
        region_coordinates = self.region_pooling_weights @ patch_coordinates
        if self.explicit_region_indices is None:
            region_selection = self.scorer(
                region_features, current_ego_status, region_coordinates
            )
            region_weights = region_selection.selection_weights
            hard_region_weights = region_selection.hard_selection_weights
            region_indices = region_selection.selected_patch_indices
        else:
            region_indices = self.explicit_region_indices
            if region_indices.dtype != torch.long or region_indices.shape != (
                current_patch_latents.shape[0],
                self.region_budget,
            ):
                raise ValueError(
                    "Explicit region IDs must be long [batch, region budget]"
                )
            if (region_indices < 0).any() or (
                region_indices >= self.region_count
            ).any():
                raise ValueError("Region ID outside candidate grid")
            if any(
                row.unique().numel() != self.region_budget for row in region_indices
            ):
                raise ValueError("Explicit region IDs must be unique")
            hard_region_weights = F.one_hot(region_indices, self.region_count).to(
                current_patch_latents.dtype
            )
            region_weights = hard_region_weights
        native_weights = region_weights @ self.region_pooling_weights
        native_hard = hard_region_weights @ self.region_pooling_weights
        return SpatialRegionSelection(
            native_weights,
            native_hard,
            region_indices,
            native_weights @ patch_coordinates,
        )


def pooled_future_targets(selection, future_target_latents, future_valid_mask):
    """Require every member of a region to have a valid teacher target."""
    if (
        future_valid_mask.dtype != torch.bool
        or future_valid_mask.shape != future_target_latents.shape[:-1]
    ):
        raise ValueError("Expected bool [batch,time,native_patch] validity")
    if not torch.isfinite(future_target_latents[future_valid_mask]).all():
        raise ValueError("Valid future teacher latents must be finite")
    hard_weights = selection.hard_selection_weights.detach()
    safe_target = torch.where(
        future_valid_mask[..., None], future_target_latents.detach(), 0
    )
    targets = torch.einsum("bkn,btnd->bktd", hard_weights, safe_target)
    invalid_members = torch.einsum(
        "bkn,btn->bkt", (hard_weights > 0).float(), (~future_valid_mask).float()
    )
    return targets, invalid_members == 0


def compute_region_future_auxiliary_loss(
    model, online_outputs, current_ego_status, future_target_latents, future_valid_mask
):
    selection = online_outputs["patch_selection"]
    targets, valid = pooled_future_targets(
        selection, future_target_latents, future_valid_mask
    )
    prediction = model._predict_selected(
        online_outputs["current_patch_latents"].detach(),
        current_ego_status.detach(),
        selection.hard_selection_weights.detach(),
    )
    error = (prediction - targets).square().mean(-1)
    return torch.where(valid, error, 0).sum() / valid.sum().clamp_min(1)


def planner_retention_outputs(
    model, current_patch_latents, current_ego_status, selection
):
    """Selector-only gradient from masked frozen-planner readout.

    Training-only intervention. The actual planning forward always receives the
    full original current grid. Globally contextual latents and replacement by
    their current mean limit the interpretation of this proxy.
    """
    coverage = selection.selection_weights.sum(1) * model.patch_selector.region_side**2
    current = current_patch_latents.detach()
    retained_features = coverage[..., None] * current + (
        1 - coverage[..., None]
    ) * current.mean(1, keepdim=True)
    retained = model.forward_from_current_patch_latents(
        retained_features, current_ego_status.detach(), enable_future_branch=False
    )["trajectory"]
    with torch.no_grad():
        original = model.forward_from_current_patch_latents(
            current, current_ego_status.detach(), enable_future_branch=False
        )["trajectory"]
    return retained, original


def compute_planner_retention_loss(
    model, current_patch_latents, current_ego_status, selection
):
    retained, original = planner_retention_outputs(
        model, current_patch_latents, current_ego_status, selection
    )
    return F.smooth_l1_loss(retained[..., :2], original[..., :2])
