"""Controlled adaptations of sparse routing, history masking and motion selection.

These operate on globally contextual camera regions, not tracked object slots.
They are inspired by SPARTAN, C-JEPA and IA-JEPA, not paper reproductions.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .spatial_region_future import pooled_future_targets


class LocatedRegionFutureBridge(nn.Module):
    """Route future messages back to the selected CURRENT memory positions.

    A single directed relation layer uses current-state-dependent edges. Sparse
    mode uses hard sigmoid-threshold edges with straight-through gradients and
    mandatory self edges. It is dense tensor computation, not a sparse kernel.
    """

    def __init__(self, region_pooling_weights, latent_dim=1024, planner_dim=256,
                 future_tubelet_count=4, sparse_connections=False):
        super().__init__()
        self.register_buffer("region_pooling_weights", region_pooling_weights.clone())
        self.sparse_connections = sparse_connections
        self.future_projection = nn.Linear(latent_dim, planner_dim)
        self.coordinate_projection = nn.Linear(2, planner_dim)
        self.time_weights = nn.Parameter(torch.zeros(future_tubelet_count))
        self.current_normalization = nn.LayerNorm(planner_dim)
        self.relation_query = nn.Linear(planner_dim, planner_dim)
        self.relation_key = nn.Linear(planner_dim, planner_dim)
        self.message_normalization = nn.LayerNorm(planner_dim)
        self.output_projection = nn.Linear(planner_dim, planner_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        self.last_edge_probabilities = None
        self.last_hard_edges = None

    def forward(self, current_image_memory, predicted_future_latents, coordinates,
                selection=None):
        if selection is None:
            raise ValueError("Located bridge requires explicit spatial selection")
        # Native region weights average member cells. Membership (not averaging)
        # recovers a one-hot selection on the planner's pooled spatial grid.
        membership = (self.region_pooling_weights > 0).to(current_image_memory.dtype)
        region_assignment = selection.selection_weights @ membership.T
        if region_assignment.shape[-1] != current_image_memory.shape[1]:
            raise ValueError("Selected regions must align with planner memory grid")
        selected_current = region_assignment @ current_image_memory
        normalized_current = self.current_normalization(selected_current)
        edge_logits = (
            self.relation_query(normalized_current)
            @ self.relation_key(normalized_current).transpose(1, 2)
        ) / math.sqrt(current_image_memory.shape[-1])
        edge_probability = edge_logits.sigmoid()
        self_edges = torch.eye(edge_logits.shape[-1], device=edge_logits.device,
                               dtype=torch.bool)[None]
        hard_edges = (edge_probability >= 0.5) | self_edges
        if self.sparse_connections:
            edge_gate = hard_edges.to(edge_logits.dtype) + (
                edge_probability - edge_probability.detach()
            ) * (~self_edges)
        else:
            edge_gate = torch.ones_like(edge_logits)
            hard_edges = torch.ones_like(hard_edges)
        attention = edge_logits.softmax(-1) * edge_gate
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        future_message = (
            self.future_projection(predicted_future_latents)
            * self.time_weights.softmax(0)[None, None, :, None]
        ).sum(2) + self.coordinate_projection(coordinates)
        region_residual = self.output_projection(
            self.message_normalization(future_message + attention @ future_message)
        )
        self.last_edge_probabilities = edge_probability
        self.last_hard_edges = hard_edges.detach()
        return region_assignment.transpose(1, 2) @ region_residual

    def connection_sparsity_loss(self):
        probability = self.last_edge_probabilities
        if probability is None:
            raise RuntimeError("Call the bridge before its connection regularizer")
        off_diagonal = ~torch.eye(probability.shape[-1], device=probability.device,
                                 dtype=torch.bool)
        return probability[:, off_diagonal].mean()


def replace_selected_current_regions(current_patch_latents, earlier_patch_latents,
                                     selection, masked_slot_indices):
    """Training-only current-history intervention; the earlier anchor is retained.

    Replace both the selected residual shortcut and the decoder context through
    the shared native grid. Unmasked contextual latents can still encode global
    information: this intervention does not establish causal identifiability.
    """
    hard_weights = selection.hard_selection_weights.detach()
    masked_weights = hard_weights.gather(
        1, masked_slot_indices[..., None].expand(-1, -1, hard_weights.shape[-1])
    )
    native_mask = masked_weights.sum(1) > 0
    return torch.where(native_mask[..., None], earlier_patch_latents.detach(),
                       current_patch_latents.detach()), native_mask


def compute_history_reconstruction_losses(model, outputs, current_ego_status,
                                          earlier_patch_latents,
                                          earlier_history_valid,
                                          future_target_latents, future_valid_mask,
                                          masked_slot_indices, mask_current_history):
    """Detached selection, observed earlier anchors; targets enter loss only.

    Both control and masked conditions reconstruct the same sampled current
    regions plus all selected future times. Missing earlier observations are
    omitted from this auxiliary objective only, never from planning/evaluation.
    """
    selection = outputs["patch_selection"]
    current = outputs["current_patch_latents"].detach()
    predictor_input, native_mask = replace_selected_current_regions(
        current, earlier_patch_latents, selection, masked_slot_indices
    )
    if not mask_current_history:
        predictor_input = current
    hard_weights = selection.hard_selection_weights.detach()
    predictor = model.future_predictor
    prediction = predictor(
        hard_weights @ predictor_input,
        hard_weights @ model.patch_coordinates,
        predictor_input,
        current_ego_status.detach(),
        model.patch_coordinates,
        query_time_embeddings=torch.cat((predictor.current_time_embedding,
                                         predictor.time_embeddings.weight), 0),
    )
    future_targets, future_valid = pooled_future_targets(
        selection, future_target_latents, future_valid_mask
    )
    future_valid = future_valid & earlier_history_valid[:, None, None]
    future_error = (prediction[:, :, 1:] - future_targets).square().mean(-1)
    future_loss = torch.where(future_valid, future_error, 0).sum() / future_valid.sum().clamp_min(1)
    current_targets = hard_weights @ current
    current_error = (prediction[:, :, 0] - current_targets).square().mean(-1)
    current_valid = torch.zeros_like(current_error, dtype=torch.bool).scatter_(
        1, masked_slot_indices, True
    ) & earlier_history_valid[:, None]
    current_loss = torch.where(current_valid, current_error, 0).sum() / current_valid.sum().clamp_min(1)
    return future_loss, current_loss


def observed_history_motion_scores(observed_images, region_grid=(8, 16)):
    """IA-JEPA-inspired second difference of absolute observed frame changes.

    Input: exactly four observed RGB images [4,3,height,width] in [0,1].
    No future frames, collision annotations, targets or target validity inputs.
    Ego motion/texture remain confounders; this is deliberately a heuristic control.
    """
    if observed_images.ndim != 4 or observed_images.shape[:2] != (4, 3):
        raise ValueError("Expected exactly four observed RGB frames")
    changes = (observed_images[1:] - observed_images[:-1]).abs()
    motion_map = (changes[1:] - changes[:-1]).abs().mean((0, 1))
    return F.adaptive_avg_pool2d(motion_map[None, None], region_grid).flatten()
