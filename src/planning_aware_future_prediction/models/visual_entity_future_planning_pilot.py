"""New GT-ROI visual/spatial pilot, NOT official Drive-JEPA reproduction.

Only its official frozen encoder is reused. Planner uses the upstream pattern of
image/status memory + trajectory queries, with newly initialized small layers.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from .selective_entity_future_prediction import (
    ContextConditionedEntityScorer,
    EntitySelectionResult,
    SequentialStraightThroughEntitySelection,
)


@dataclass
class VisualFuturePilotOutput:
    ego_trajectory: Tensor
    entity_selection: EntitySelectionResult | None
    predicted_future_visual_latents: Tensor | None
    predicted_future_spatial_states: Tensor | None


class CurrentAndPredictedFutureTrajectoryDecoder(nn.Module):
    def __init__(
        self, visual_feature_dim: int, num_future_steps: int, hidden_dim: int = 128
    ):
        super().__init__()
        self.current_image_adapter = nn.Linear(visual_feature_dim, hidden_dim)
        self.current_entity_adapter = nn.Linear(visual_feature_dim + 10, hidden_dim)
        self.ego_status_adapter = nn.Linear(8, hidden_dim)
        self.predicted_future_adapter = nn.Linear(visual_feature_dim + 6, hidden_dim)
        self.image_position_embedding = nn.Parameter(
            torch.randn(1, 128, hidden_dim) * 0.02
        )
        self.future_time_embedding = nn.Parameter(
            torch.randn(1, 1, num_future_steps, hidden_dim) * 0.02
        )
        self.entity_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.future_type_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.trajectory_queries = nn.Parameter(
            torch.randn(1, num_future_steps, hidden_dim) * 0.02
        )
        self.trajectory_transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=hidden_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)
        )

    def forward(
        self,
        current_image_grid: Tensor,
        current_entity_features: Tensor,
        current_entity_valid_mask: Tensor,
        current_ego_status: Tensor,
        predicted_future_visual_latents: Tensor | None = None,
        predicted_future_spatial_states: Tensor | None = None,
        selected_entity_valid_mask: Tensor | None = None,
    ) -> Tensor:
        image_tokens = (
            functional.avg_pool2d(current_image_grid, 2).flatten(2).transpose(1, 2)
        )
        image_memory = (
            self.current_image_adapter(image_tokens) + self.image_position_embedding
        )
        entity_memory = (
            self.current_entity_adapter(current_entity_features)
            + self.entity_type_embedding
        )
        status_memory = self.ego_status_adapter(current_ego_status)[:, None]
        memory_parts = [image_memory, status_memory, entity_memory]
        valid_parts = [
            torch.ones(
                image_memory.shape[:2], dtype=torch.bool, device=image_memory.device
            ),
            torch.ones(
                status_memory.shape[:2], dtype=torch.bool, device=image_memory.device
            ),
            current_entity_valid_mask,
        ]
        if predicted_future_visual_latents is not None:
            future_memory = (
                self.predicted_future_adapter(
                    torch.cat(
                        (
                            predicted_future_visual_latents,
                            predicted_future_spatial_states,
                        ),
                        dim=-1,
                    )
                )
                + self.future_time_embedding
            )
            memory_parts.append(
                future_memory.flatten(1, 2) + self.future_type_embedding
            )
            valid_parts.append(
                selected_entity_valid_mask[:, :, None]
                .expand(-1, -1, predicted_future_visual_latents.shape[2])
                .flatten(1, 2)
            )
        memory = torch.cat(memory_parts, dim=1)
        padding_mask = ~torch.cat(valid_parts, dim=1)
        decoded = self.trajectory_transformer(
            memory,
            self.trajectory_queries.expand(memory.shape[0], -1, -1),
            src_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )
        trajectory = self.trajectory_head(decoded)
        return torch.cat(
            (trajectory[..., :2], trajectory[..., 2:].tanh() * torch.pi), dim=-1
        )


class VisualEntityFuturePlanningPilot(nn.Module):
    def __init__(
        self,
        visual_feature_dim: int = 1024,
        num_future_steps: int = 8,
        num_selected_entities: int = 4,
    ):
        super().__init__()
        self.visual_feature_dim = visual_feature_dim
        self.num_future_steps = num_future_steps
        self.entity_scorer = ContextConditionedEntityScorer(
            visual_feature_dim + 10, visual_feature_dim, 8, 128
        )
        self.entity_selection_operator = SequentialStraightThroughEntitySelection(
            num_selected_entities
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(visual_feature_dim * 2 + 18, 128),
            nn.GELU(),
            nn.Linear(128, num_future_steps * (visual_feature_dim + 6)),
        )
        self.ego_planner = CurrentAndPredictedFutureTrajectoryDecoder(
            visual_feature_dim, num_future_steps
        )

    def predict_selected_future(
        self,
        selected_entity_features: Tensor,
        current_scene_context: Tensor,
        current_ego_status: Tensor,
        selected_entity_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        common_context = torch.cat((current_scene_context, current_ego_status), dim=-1)
        predictor_inputs = torch.cat(
            (
                selected_entity_features,
                common_context[:, None].expand(
                    -1, selected_entity_features.shape[1], -1
                ),
            ),
            dim=-1,
        )
        predicted = (
            self.future_predictor(predictor_inputs).reshape(
                predictor_inputs.shape[0],
                predictor_inputs.shape[1],
                self.num_future_steps,
                self.visual_feature_dim + 6,
            )
            * selected_entity_valid_mask[:, :, None, None]
        )
        return predicted[..., : self.visual_feature_dim], predicted[
            ..., self.visual_feature_dim :
        ]

    def forward(
        self,
        current_image_grid: Tensor,
        current_entity_features: Tensor,
        current_entity_valid_mask: Tensor,
        stable_entity_ids: Tensor,
        current_ego_status: Tensor,
        enable_future_branch: bool = True,
        detach_predicted_future: bool = False,
    ) -> VisualFuturePilotOutput:
        # No training target, future validity, future image, or future box argument.
        safe_entity_features = torch.where(
            current_entity_valid_mask[..., None], current_entity_features, 0.0
        )
        if not torch.isfinite(safe_entity_features).all():
            raise ValueError("valid current entity contains non-finite features")
        selection, visual_prediction, spatial_prediction = None, None, None
        if enable_future_branch:
            scene_context = current_image_grid.mean(dim=(-2, -1))
            scores = self.entity_scorer(
                safe_entity_features, scene_context, current_ego_status
            )
            selection = self.entity_selection_operator(
                scores, current_entity_valid_mask, stable_entity_ids
            )
            selected_features = selection.selection_weights @ safe_entity_features
            visual_prediction, spatial_prediction = self.predict_selected_future(
                selected_features,
                scene_context,
                current_ego_status,
                selection.selected_entity_valid_mask,
            )
            if detach_predicted_future:
                visual_prediction, spatial_prediction = (
                    visual_prediction.detach(),
                    spatial_prediction.detach(),
                )
        ego_trajectory = self.ego_planner(
            current_image_grid,
            safe_entity_features,
            current_entity_valid_mask,
            current_ego_status,
            visual_prediction,
            spatial_prediction,
            selection.selected_entity_valid_mask if selection is not None else None,
        )
        return VisualFuturePilotOutput(
            ego_trajectory, selection, visual_prediction, spatial_prediction
        )

    def detached_selection_auxiliary_losses(
        self,
        output: VisualFuturePilotOutput,
        current_image_grid: Tensor,
        current_entity_features: Tensor,
        current_entity_valid_mask: Tensor,
        current_ego_status: Tensor,
        future_visual_targets: Tensor,
        future_visual_valid_mask: Tensor,
        future_spatial_targets: Tensor,
        future_spatial_valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        selection = output.entity_selection
        if selection is None:
            raise ValueError("no future branch has no auxiliary prediction loss")
        safe_features = torch.where(
            current_entity_valid_mask[..., None], current_entity_features, 0.0
        )
        detached_hard_selection = selection.hard_selection_weights.detach()
        predicted_visual, predicted_spatial = self.predict_selected_future(
            detached_hard_selection @ safe_features,
            current_image_grid.mean(dim=(-2, -1)),
            current_ego_status,
            selection.selected_entity_valid_mask,
        )

        def selected_masked_loss(
            predicted: Tensor, targets: Tensor, valid: Tensor
        ) -> Tensor:
            safe_targets = torch.where(valid[..., None], targets.detach(), 0.0)
            selected_targets = torch.einsum(
                "bsn,bntd->bstd", detached_hard_selection, safe_targets
            )
            selected_valid = (
                torch.einsum(
                    "bsn,bnt->bst", detached_hard_selection, valid.to(predicted.dtype)
                )
                > 0
            )
            selected_valid = (
                selected_valid & selection.selected_entity_valid_mask[:, :, None]
            )
            squared_error = (predicted - selected_targets).square()
            masked_error = torch.where(selected_valid[..., None], squared_error, 0.0)
            return masked_error.sum() / (
                selected_valid.sum() * predicted.shape[-1]
            ).clamp_min(1)

        return {
            "visual_prediction": selected_masked_loss(
                predicted_visual, future_visual_targets, future_visual_valid_mask
            ),
            "spatial_prediction": selected_masked_loss(
                predicted_spatial, future_spatial_targets, future_spatial_valid_mask
            ),
        }
