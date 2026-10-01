"""CPU graph diagnostic, not a NAVSIM agent or a trained visual JEPA model."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class EntitySelectionResult:
    selection_weights: Tensor  # [B, K, N], hard forward, optional ST backward
    hard_selection_weights: Tensor
    selected_entity_indices: Tensor  # -1 for inactive slots
    selected_entity_valid_mask: Tensor  # [B, K]


class SequentialStraightThroughEntitySelection(nn.Module):
    """Unique hard selection with a conditional, biased soft-backward surrogate.

    Each slot softmaxes over currently remaining valid entities. Previously
    chosen HARD indices are excluded; no gradient through that exclusion.
    Ties use ascending stable entity ID, never incidental tensor position.
    """

    def __init__(self, num_selected_entities: int, temperature: float = 0.7):
        super().__init__()
        if num_selected_entities < 1 or not temperature > 0:
            raise ValueError("k and temperature must be positive")
        self.num_selected_entities, self.temperature = (
            num_selected_entities,
            temperature,
        )

    def forward(
        self,
        entity_scores: Tensor,
        entity_valid_mask: Tensor,
        stable_entity_ids: Tensor,
    ) -> EntitySelectionResult:
        if (
            entity_scores.ndim != 2
            or entity_valid_mask.shape != entity_scores.shape
            or stable_entity_ids.shape != entity_scores.shape
        ):
            raise ValueError("scores, valid and entity_ids must have shape [B, N]")
        if (
            entity_valid_mask.dtype != torch.bool
            or stable_entity_ids.dtype != torch.long
        ):
            raise ValueError("valid must be bool; entity_ids must be int64")
        if not torch.isfinite(entity_scores[entity_valid_mask]).all():
            raise ValueError("valid scores must be finite")
        for sample_entity_ids, sample_valid_mask in zip(
            stable_entity_ids, entity_valid_mask
        ):
            valid_entity_ids = sample_entity_ids[sample_valid_mask]
            if valid_entity_ids.unique().numel() != valid_entity_ids.numel():
                raise ValueError("valid entity IDs must be unique within each sample")
        batch_size, num_candidate_entities = entity_scores.shape
        if num_candidate_entities == 0:
            empty_assignments = entity_scores.new_zeros(
                batch_size, self.num_selected_entities, 0
            )
            return EntitySelectionResult(
                empty_assignments,
                empty_assignments,
                stable_entity_ids.new_full(
                    (batch_size, self.num_selected_entities), -1
                ),
                entity_valid_mask.new_zeros(batch_size, self.num_selected_entities),
            )

        stable_entity_order = torch.argsort(stable_entity_ids, dim=-1, stable=True)
        ordered_entity_scores = entity_scores.gather(1, stable_entity_order)
        remaining_entity_mask = entity_valid_mask.gather(1, stable_entity_order)
        (
            soft_assignment_rows,
            hard_assignment_rows,
            selected_entity_indices,
            slot_valid_masks,
        ) = ([], [], [], [])
        for _ in range(self.num_selected_entities):
            active_slot_mask = remaining_entity_mask.any(dim=-1)
            masked_entity_scores = ordered_entity_scores.masked_fill(
                ~remaining_entity_mask, -torch.inf
            )
            # All-invalid samples must not softmax an all-minus-inf vector.
            safe_entity_scores = torch.where(
                active_slot_mask[:, None],
                masked_entity_scores,
                torch.zeros_like(masked_entity_scores),
            )
            soft_assignment = (safe_entity_scores / self.temperature).softmax(
                dim=-1
            ) * remaining_entity_mask
            selected_ordered_index = safe_entity_scores.argmax(dim=-1)
            hard_assignment = (
                F.one_hot(selected_ordered_index, num_candidate_entities).to(
                    entity_scores.dtype
                )
                * active_slot_mask[:, None]
            )
            hard_assignment_rows.append(
                torch.zeros_like(hard_assignment).scatter(
                    1, stable_entity_order, hard_assignment
                )
            )
            soft_assignment_rows.append(
                torch.zeros_like(soft_assignment).scatter(
                    1, stable_entity_order, soft_assignment
                )
            )
            selected_original_index = stable_entity_order.gather(
                1, selected_ordered_index[:, None]
            ).squeeze(1)
            selected_entity_indices.append(
                torch.where(active_slot_mask, selected_original_index, -1)
            )
            slot_valid_masks.append(active_slot_mask)
            remaining_entity_mask = remaining_entity_mask & ~hard_assignment.bool()
        hard_assignment = torch.stack(hard_assignment_rows, dim=1)
        soft_assignment_tensor = torch.stack(soft_assignment_rows, dim=1)
        # No subset decision when the budget covers every valid entity. Our
        # planner is order invariant: rank gradients here would be spurious.
        soft_assignment_tensor = (
            soft_assignment_tensor
            * (entity_valid_mask.sum(dim=-1) > self.num_selected_entities)[
                :, None, None
            ]
        )
        # Parentheses ensure exactly hard forward values, not 1+p-p roundoff.
        selection_weights = hard_assignment + (
            soft_assignment_tensor - soft_assignment_tensor.detach()
        )
        return EntitySelectionResult(
            selection_weights,
            hard_assignment,
            torch.stack(selected_entity_indices, dim=1),
            torch.stack(slot_valid_masks, dim=1),
        )


class ContextConditionedEntityScorer(nn.Module):
    """Shared scoring by entity content and intent; no slot/index embeddings."""

    def __init__(
        self,
        entity_feature_dim: int,
        scene_context_dim: int,
        ego_intent_dim: int,
        hidden_feature_dim: int = 32,
    ):
        super().__init__()
        self.entity_score_network = nn.Sequential(
            nn.Linear(
                entity_feature_dim + scene_context_dim + ego_intent_dim,
                hidden_feature_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, 1),
        )

    def forward(
        self, entity_features: Tensor, scene_context: Tensor, ego_intent: Tensor
    ) -> Tensor:
        num_candidate_entities = entity_features.shape[1]
        scene_intent_features = torch.cat((scene_context, ego_intent), dim=-1)[
            :, None
        ].expand(-1, num_candidate_entities, -1)
        return self.entity_score_network(
            torch.cat((entity_features, scene_intent_features), dim=-1)
        ).squeeze(-1)


@dataclass
class SelectiveFuturePredictionOutput:
    entity_selection: EntitySelectionResult
    predicted_future_latents: Tensor
    ego_plan: Tensor


class SelectiveEntityFuturePredictionGraph(nn.Module):
    """Minimal differentiable fixture with externally supplied detached targets.

    C is a pooled context vector in this fixture, not an implemented BEV stack.
    Future targets are never arguments of forward(). No target encoder here.
    """

    def __init__(
        self,
        entity_feature_dim=6,
        scene_context_dim=4,
        ego_intent_dim=3,
        future_latent_dim=5,
        num_future_steps=2,
        num_selected_entities=2,
        hidden_feature_dim=24,
        num_ego_plan_steps=1,
    ):
        super().__init__()
        if num_ego_plan_steps < 1:
            raise ValueError("num_ego_plan_steps must be positive")
        self.num_ego_plan_steps = num_ego_plan_steps
        self.num_future_steps, self.future_latent_dim = (
            num_future_steps,
            future_latent_dim,
        )
        self.entity_scorer = ContextConditionedEntityScorer(
            entity_feature_dim, scene_context_dim, ego_intent_dim, hidden_feature_dim
        )
        self.entity_selection_operator = SequentialStraightThroughEntitySelection(
            num_selected_entities
        )
        self.future_predictor = nn.Sequential(
            nn.Linear(
                entity_feature_dim + scene_context_dim + ego_intent_dim,
                hidden_feature_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, num_future_steps * future_latent_dim),
        )
        self.ego_planner = nn.Sequential(
            nn.Linear(
                scene_context_dim
                + ego_intent_dim
                + num_future_steps * future_latent_dim,
                hidden_feature_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, 3 * num_ego_plan_steps),
        )

    def predict_selected_entity_future_latents(
        self,
        selected_entity_queries: Tensor,
        scene_context: Tensor,
        ego_intent: Tensor,
        selected_entity_valid_mask: Tensor,
    ) -> Tensor:
        scene_intent_features = torch.cat((scene_context, ego_intent), dim=-1)[
            :, None
        ].expand(-1, selected_entity_queries.shape[1], -1)
        future_latent_values = self.future_predictor(
            torch.cat((selected_entity_queries, scene_intent_features), dim=-1)
        )
        future_latent_values = future_latent_values.reshape(
            selected_entity_queries.shape[0],
            selected_entity_queries.shape[1],
            self.num_future_steps,
            self.future_latent_dim,
        )
        return future_latent_values * selected_entity_valid_mask[..., None, None]

    def decode_ego_plan(
        self,
        scene_context: Tensor,
        ego_intent: Tensor,
        predicted_future_latents: Tensor,
        selected_entity_valid_mask: Tensor,
    ) -> Tensor:
        num_valid_selected_entities = (
            selected_entity_valid_mask.sum(dim=-1)
            .clamp_min(1)
            .to(predicted_future_latents.dtype)
        )
        pooled_future_latents = (
            predicted_future_latents * selected_entity_valid_mask[..., None, None]
        ).sum(dim=1) / num_valid_selected_entities[:, None, None]
        ego_plan = self.ego_planner(
            torch.cat(
                (scene_context, ego_intent, pooled_future_latents.flatten(1)), dim=-1
            )
        )
        if self.num_ego_plan_steps == 1:
            return ego_plan  # Preserve the original synthetic fixture interface.
        return ego_plan.reshape(ego_plan.shape[0], self.num_ego_plan_steps, 3)

    def forward(
        self,
        entity_features: Tensor,
        scene_context: Tensor,
        ego_intent: Tensor,
        entity_valid_mask: Tensor,
        stable_entity_ids: Tensor,
        selection_gradient_mode="straight_through",
        detach_future_latents=False,
    ) -> SelectiveFuturePredictionOutput:
        if selection_gradient_mode not in (
            "straight_through",
            "hard_indices",
            "detached_selection",
        ):
            raise ValueError("unknown selection mode")
        safe_entity_features = entity_features.masked_fill(
            ~entity_valid_mask[..., None], 0
        )
        entity_selection = self.entity_selection_operator(
            self.entity_scorer(safe_entity_features, scene_context, ego_intent),
            entity_valid_mask,
            stable_entity_ids,
        )
        if selection_gradient_mode == "hard_indices":
            entity_selection.selection_weights = entity_selection.hard_selection_weights
        elif selection_gradient_mode == "detached_selection":
            entity_selection.selection_weights = (
                entity_selection.selection_weights.detach()
            )
        selected_entity_queries = (
            entity_selection.selection_weights @ safe_entity_features
        )
        predicted_future_latents = self.predict_selected_entity_future_latents(
            selected_entity_queries,
            scene_context,
            ego_intent,
            entity_selection.selected_entity_valid_mask,
        )
        planning_future_latents = (
            predicted_future_latents.detach()
            if detach_future_latents
            else predicted_future_latents
        )
        ego_plan = self.decode_ego_plan(
            scene_context,
            ego_intent,
            planning_future_latents,
            entity_selection.selected_entity_valid_mask,
        )
        return SelectiveFuturePredictionOutput(
            entity_selection, predicted_future_latents, ego_plan
        )

    def compute_training_losses(
        self,
        prediction_output: SelectiveFuturePredictionOutput,
        entity_features: Tensor,
        scene_context: Tensor,
        ego_intent: Tensor,
        entity_valid_mask: Tensor,
        future_latent_targets: Tensor,
        future_target_valid_mask: Tensor,
        ego_plan_target: Tensor,
    ):
        planning_loss = F.mse_loss(prediction_output.ego_plan, ego_plan_target)
        # Separate predictor call: auxiliary cannot backprop directly to selector.
        safe_entity_features = entity_features.masked_fill(
            ~entity_valid_mask[..., None], 0
        )
        selected_entity_queries = (
            prediction_output.entity_selection.hard_selection_weights.detach()
            @ safe_entity_features
        )
        auxiliary_future_latents = self.predict_selected_entity_future_latents(
            selected_entity_queries,
            scene_context,
            ego_intent,
            prediction_output.entity_selection.selected_entity_valid_mask,
        )
        batch_size, num_selected_slots = (
            prediction_output.entity_selection.selected_entity_indices.shape
        )
        if entity_features.shape[1] == 0:
            selected_future_targets = torch.zeros_like(auxiliary_future_latents)
            selected_future_target_mask = future_target_valid_mask.new_zeros(
                batch_size, num_selected_slots, self.num_future_steps
            )
        else:
            selected_entity_indices = (
                prediction_output.entity_selection.selected_entity_indices.clamp_min(0)
            )
            selected_future_targets = future_latent_targets.detach().gather(
                1,
                selected_entity_indices[..., None, None].expand(
                    -1, -1, self.num_future_steps, self.future_latent_dim
                ),
            )
            selected_future_target_mask = future_target_valid_mask.gather(
                1,
                selected_entity_indices[..., None].expand(
                    -1, -1, self.num_future_steps
                ),
            )
        selected_future_target_mask = (
            selected_future_target_mask
            & prediction_output.entity_selection.selected_entity_valid_mask[..., None]
        )
        # Invalid targets can contain NaN; mask BEFORE forming squared errors.
        selected_future_targets = torch.where(
            selected_future_target_mask[..., None],
            selected_future_targets,
            torch.zeros_like(selected_future_targets),
        )
        future_prediction_error = torch.where(
            selected_future_target_mask[..., None],
            auxiliary_future_latents - selected_future_targets,
            torch.zeros_like(auxiliary_future_latents),
        )
        num_valid_target_elements = (
            selected_future_target_mask.sum() * self.future_latent_dim
        ).clamp_min(1)
        future_latent_prediction_loss = (
            future_prediction_error.square().sum() / num_valid_target_elements
        )
        return {
            "planning_loss": planning_loss,
            "future_latent_prediction_loss": future_latent_prediction_loss,
        }


def compute_parameter_gradient_norm(
    training_loss: Tensor, model_component: nn.Module
) -> float:
    gradients = torch.autograd.grad(
        training_loss,
        tuple(model_component.parameters()),
        allow_unused=True,
        retain_graph=True,
    )
    if any(
        parameter_gradient is not None and not torch.isfinite(parameter_gradient).all()
        for parameter_gradient in gradients
    ):
        raise AssertionError("non-finite gradient")
    return (
        sum(
            parameter_gradient.square().sum().item()
            for parameter_gradient in gradients
            if parameter_gradient is not None
        )
        ** 0.5
    )
