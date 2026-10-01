"""Same pilot modules, explicit physics/persistence priors + zero-initialized deltas."""

import torch

from .fixed_distance_future_supervision import FixedDistanceFutureSupervisionPilot
from .kinematic_future_priors import (
    predict_ego_kinematic_trajectory,
    predict_entity_constant_velocity_state,
)


class ResidualFutureSupervisionPilot(FixedDistanceFutureSupervisionPilot):
    """Flags allow the small VISUAL-ONLY control without changing ego/spatial heads.

    Outputs keep the old contract: ego XY meters, future heads normalized by
    train-fitted channel moments. Priors are constructed from current inputs;
    future target validity is NEVER an input or padding policy.
    """

    def __init__(
        self,
        *,
        residual_visual=True,
        residual_spatial=False,
        residual_ego=False,
        visual_feature_dim=1024,
        num_future_steps=8,
        num_selected_entities=4,
    ):
        super().__init__(visual_feature_dim, num_future_steps, num_selected_entities)
        self.residual_visual = residual_visual
        self.residual_spatial = residual_spatial
        self.residual_ego = residual_ego
        for name, dimension in (("visual", visual_feature_dim), ("spatial", 6)):
            self.register_buffer(
                f"{name}_target_mean", torch.zeros(dimension), persistent=False
            )
            self.register_buffer(
                f"{name}_target_std", torch.ones(dimension), persistent=False
            )

    def set_train_target_normalization(self, normalization):
        for name in ("visual", "spatial"):
            getattr(self, f"{name}_target_mean").copy_(normalization[name]["mean"])
            getattr(self, f"{name}_target_std").copy_(normalization[name]["std"])

    def initialize_residual_output_heads(self):
        """Call AFTER copying common initial weights, never on resumed checkpoints."""
        final_layer = self.future_predictor[-1]
        with torch.no_grad():
            shaped_weight = final_layer.weight.view(
                self.num_future_steps, self.visual_feature_dim + 6, -1
            )
            shaped_bias = final_layer.bias.view(
                self.num_future_steps, self.visual_feature_dim + 6
            )
            if self.residual_visual:
                shaped_weight[:, : self.visual_feature_dim].zero_()
                shaped_bias[:, : self.visual_feature_dim].zero_()
            if self.residual_spatial:
                shaped_weight[:, self.visual_feature_dim :].zero_()
                shaped_bias[:, self.visual_feature_dim :].zero_()
            if self.residual_ego:
                self.ego_planner.trajectory_head[-1].weight.zero_()
                self.ego_planner.trajectory_head[-1].bias.zero_()

    def predict_selected_future(
        self,
        selected_entity_features,
        current_scene_context,
        current_ego_status,
        selected_entity_valid_mask,
    ):
        visual_delta, spatial_delta = super().predict_selected_future(
            selected_entity_features,
            current_scene_context,
            current_ego_status,
            selected_entity_valid_mask,
        )
        selected_mask = selected_entity_valid_mask[:, :, None, None]
        if self.residual_visual:
            visual_prior = (
                selected_entity_features[..., : self.visual_feature_dim]
                - self.visual_target_mean
            ) / self.visual_target_std
            visual_delta = (visual_delta + visual_prior[:, :, None]) * selected_mask
        if self.residual_spatial:
            spatial_prior = predict_entity_constant_velocity_state(
                selected_entity_features, self.num_future_steps
            )
            spatial_prior = (
                spatial_prior - self.spatial_target_mean
            ) / self.spatial_target_std
            spatial_delta = (spatial_delta + spatial_prior) * selected_mask
        return visual_delta, spatial_delta

    def forward(
        self,
        current_image_grid,
        current_entity_features,
        current_entity_valid_mask,
        stable_entity_ids,
        current_ego_status,
        enable_future_branch=True,
        detach_future_for_planning=False,
    ):
        output = super().forward(
            current_image_grid,
            current_entity_features,
            current_entity_valid_mask,
            stable_entity_ids,
            current_ego_status,
            enable_future_branch=enable_future_branch,
            detach_future_for_planning=detach_future_for_planning,
        )
        if self.residual_ego:
            output.ego_trajectory = (
                output.ego_trajectory
                + predict_ego_kinematic_trajectory(
                    current_ego_status, self.num_future_steps, "constant_acceleration"
                )
            )
        return output
