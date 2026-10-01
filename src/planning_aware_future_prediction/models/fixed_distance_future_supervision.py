"""Fixed current-distance selection: target-supervision study, NOT selector learning."""

import torch
from torch import Tensor

from .selective_entity_future_prediction import EntitySelectionResult
from .visual_entity_future_planning_pilot import (
    VisualEntityFuturePlanningPilot,
    VisualFuturePilotOutput,
)


def select_nearest_current_entities(
    current_entity_features: Tensor,
    current_entity_valid_mask: Tensor,
    stable_entity_ids: Tensor,
    num_selected_entities: int = 4,
) -> EntitySelectionResult:
    """Distance from current GT state; stable identity resolves ties, not tensor order."""
    current_xy = current_entity_features[..., -10:-8]
    squared_distance = current_xy.square().sum(dim=-1)
    identity_order = stable_entity_ids.argsort(dim=-1, stable=True)
    identity_ordered_distances = squared_distance.gather(1, identity_order)
    identity_ordered_valid = current_entity_valid_mask.gather(1, identity_order)
    distance_order = identity_ordered_distances.masked_fill(
        ~identity_ordered_valid, torch.inf
    ).argsort(dim=-1, stable=True)
    selected_indices = identity_order.gather(1, distance_order)[
        :, :num_selected_entities
    ]
    selected_valid = current_entity_valid_mask.gather(1, selected_indices)
    if selected_indices.shape[1] < num_selected_entities:
        missing_slots = num_selected_entities - selected_indices.shape[1]
        selected_indices = torch.nn.functional.pad(selected_indices, (0, missing_slots))
        selected_valid = torch.nn.functional.pad(selected_valid, (0, missing_slots))
    if current_entity_features.shape[1] == 0:
        assignments = current_entity_features.new_zeros(
            current_entity_features.shape[0], num_selected_entities, 0
        )
    else:
        assignments = (
            torch.nn.functional.one_hot(
                selected_indices, current_entity_features.shape[1]
            ).to(current_entity_features.dtype)
            * selected_valid[..., None]
        )
    return EntitySelectionResult(
        assignments,
        assignments,
        selected_indices.masked_fill(~selected_valid, -1),
        selected_valid,
    )


class FixedDistanceFutureSupervisionPilot(VisualEntityFuturePlanningPilot):
    """Reuses scaffold weights/heads; scorer is frozen and never called."""

    def configure_active_modules(self, enable_future_branch: bool):
        self.requires_grad_(True)
        self.entity_scorer.requires_grad_(False)
        self.future_predictor.requires_grad_(enable_future_branch)
        self.ego_planner.predicted_future_adapter.requires_grad_(enable_future_branch)
        self.ego_planner.future_time_embedding.requires_grad_(enable_future_branch)
        self.ego_planner.future_type_embedding.requires_grad_(enable_future_branch)

    def forward(
        self,
        current_image_grid: Tensor,
        current_entity_features: Tensor,
        current_entity_valid_mask: Tensor,
        stable_entity_ids: Tensor,
        current_ego_status: Tensor,
        enable_future_branch: bool = True,
        detach_future_for_planning: bool = False,
    ) -> VisualFuturePilotOutput:
        safe_features = torch.where(
            current_entity_valid_mask[..., None], current_entity_features, 0.0
        )
        if not torch.isfinite(safe_features).all():
            raise ValueError("non-finite current entity features")
        selection = select_nearest_current_entities(
            safe_features,
            current_entity_valid_mask,
            stable_entity_ids,
            self.entity_selection_operator.num_selected_entities,
        )
        predicted_visual, predicted_spatial = None, None
        if enable_future_branch:
            predicted_visual, predicted_spatial = self.predict_selected_future(
                selection.selection_weights @ safe_features,
                current_image_grid.mean(dim=(-2, -1)),
                current_ego_status,
                selection.selected_entity_valid_mask,
            )
        # F keeps attached outputs for BOTH auxiliary heads. Detach only the
        # planner inputs, not the predictor call or returned training outputs.
        planner_visual = predicted_visual
        planner_spatial = predicted_spatial
        if detach_future_for_planning and enable_future_branch:
            planner_visual = predicted_visual.detach()
            planner_spatial = predicted_spatial.detach()
        trajectory = self.ego_planner(
            current_image_grid,
            safe_features,
            current_entity_valid_mask,
            current_ego_status,
            planner_visual,
            planner_spatial,
            selection.selected_entity_valid_mask,
        )
        return VisualFuturePilotOutput(
            trajectory, selection, predicted_visual, predicted_spatial
        )


def planning_imitation_loss(predicted_trajectory: Tensor, target_trajectory: Tensor):
    position_loss = torch.nn.functional.smooth_l1_loss(
        (predicted_trajectory[..., :2] - target_trajectory[..., :2]) / 10.0,
        torch.zeros_like(predicted_trajectory[..., :2]),
    )
    heading_loss = (
        1.0 - torch.cos(predicted_trajectory[..., 2] - target_trajectory[..., 2])
    ).mean()
    return position_loss + 0.1 * heading_loss


def gather_selected_training_targets(selection, training_targets):
    assignments = selection.hard_selection_weights.detach()
    selected_targets = {}
    for target_name in ("visual", "spatial"):
        native_valid = training_targets[f"future_{target_name}_valid_mask"]
        safe_target = torch.where(
            native_valid[..., None],
            training_targets[f"future_{target_name}_targets"],
            0.0,
        )
        selected_targets[target_name] = torch.einsum(
            "bsn,bntd->bstd", assignments, safe_target
        )
        selected_targets[f"{target_name}_valid"] = (
            torch.einsum(
                "bsn,bnt->bst", assignments, native_valid.to(assignments.dtype)
            )
            > 0
        ) & selection.selected_entity_valid_mask[..., None]
    selected_targets["common_valid"] = (
        selected_targets["visual_valid"] & selected_targets["spatial_valid"]
    )
    return selected_targets


def masked_prediction_mse(predicted, normalized_target, valid_mask):
    error = torch.where(
        valid_mask[..., None], (predicted - normalized_target).square(), 0.0
    )
    return error.sum() / (valid_mask.sum() * predicted.shape[-1]).clamp_min(1)


def train_only_target_normalization(cached_training_windows):
    """Train selected-common-valid channel population moments; no dev argument."""
    channel_values = {"visual": [], "spatial": []}
    for window in cached_training_windows:
        if window["metadata"]["split"] != "train":
            raise ValueError("development window cannot fit normalization")
        online = {
            key: value[None].float() if value.is_floating_point() else value[None]
            for key, value in window["online_inputs"].items()
        }
        training_targets = {
            key: value[None].float() if value.is_floating_point() else value[None]
            for key, value in window["training_targets"].items()
        }
        selection = select_nearest_current_entities(
            online["current_entity_features"],
            online["current_entity_valid_mask"],
            online["stable_entity_ids"],
        )
        selected = gather_selected_training_targets(selection, training_targets)
        for target_name, observations in channel_values.items():
            observations.append(
                selected[target_name][selected["common_valid"]].double()
            )
    normalization = {}
    for target_name, observations in channel_values.items():
        values = torch.cat(observations)
        if len(values) == 0:
            raise RuntimeError("no common training targets")
        normalization[target_name] = {
            "mean": values.mean(dim=0).float(),
            "std": values.std(dim=0, correction=0).clamp_min(0.05).float(),
            "observations": len(values),
            "channels_at_std_floor": int(
                (values.std(dim=0, correction=0) < 0.05).sum()
            ),
        }
    return normalization
