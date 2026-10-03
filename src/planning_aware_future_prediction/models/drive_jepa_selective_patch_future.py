"""Learned patch-future residual for an existing official Drive-JEPA planner.

No NAVSIM imports, new planner or future-GT argument in online forward.
"""

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class PatchSelection:
    selection_weights: Tensor
    hard_selection_weights: Tensor
    selected_patch_indices: Tensor
    selected_coordinates: Tensor


class PlanningConditionedPatchSelector(nn.Module):
    """Unique hard camera-grid IDs, biased content/position soft-backward ST."""

    def __init__(self, latent_dim, hidden_dim, patch_budget, temperature=0.7):
        super().__init__()
        if patch_budget < 1 or temperature <= 0:
            raise ValueError("Positive patch budget and temperature required")
        self.patch_budget = patch_budget
        self.temperature = temperature
        self.current_projection = nn.Linear(latent_dim, hidden_dim)
        self.context_projection = nn.Linear(latent_dim, hidden_dim)
        self.ego_projection = nn.Linear(8, hidden_dim)
        self.coordinate_projection = nn.Linear(2, hidden_dim)
        self.score_head = nn.Sequential(nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, current_patch_latents, current_ego_status, patch_coordinates):
        batch_size, candidate_count, _ = current_patch_latents.shape
        if self.patch_budget > candidate_count:
            raise ValueError("Patch budget exceeds current candidate count")
        if current_ego_status.shape != (batch_size, 8):
            raise ValueError("Expected official eight-channel ego status")
        scores = self.score_head(
            self.current_projection(current_patch_latents)
            + self.context_projection(current_patch_latents.mean(dim=1))[:, None]
            + self.ego_projection(current_ego_status)[:, None]
            + self.coordinate_projection(patch_coordinates)[None]
        ).squeeze(-1)
        remaining = torch.ones_like(scores, dtype=torch.bool)
        hard_rows, soft_rows, patch_indices = [], [], []
        for _ in range(self.patch_budget):
            masked_scores = scores.masked_fill(~remaining, -torch.inf)
            soft_weights = (masked_scores / self.temperature).softmax(dim=-1)
            selected_index = masked_scores.argmax(dim=-1)
            hard_weights = F.one_hot(selected_index, candidate_count).to(scores.dtype)
            hard_rows.append(hard_weights)
            soft_rows.append(soft_weights)
            patch_indices.append(selected_index)
            remaining = remaining & ~hard_weights.bool()
        hard_weights = torch.stack(hard_rows, dim=1)
        soft_weights = torch.stack(soft_rows, dim=1)
        selection_weights = hard_weights + (soft_weights - soft_weights.detach())
        return PatchSelection(
            selection_weights,
            hard_weights,
            torch.stack(patch_indices, dim=1),
            selection_weights @ patch_coordinates,
        )


class LightweightPatchFuturePredictor(nn.Module):
    def __init__(self, latent_dim, hidden_dim, future_tubelet_count):
        super().__init__()
        self.future_tubelet_count = future_tubelet_count
        self.time_embeddings = nn.Embedding(future_tubelet_count, 16)
        self.prediction_network = nn.Sequential(
            nn.Linear(2 * latent_dim + 8 + 2 + 16, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.last_prediction_query_shape = None

    def forward(self, selected_current_latents, coordinates, scene_context, ego_status):
        batch_size, patch_budget, _ = selected_current_latents.shape
        selected_context = torch.cat(
            (
                selected_current_latents,
                coordinates,
                scene_context[:, None].expand(-1, patch_budget, -1),
                ego_status[:, None].expand(-1, patch_budget, -1),
            ),
            dim=-1,
        )
        query_features = torch.cat(
            (
                selected_context[:, :, None].expand(
                    -1, -1, self.future_tubelet_count, -1
                ),
                self.time_embeddings.weight[None, None].expand(
                    batch_size, patch_budget, -1, -1
                ),
            ),
            dim=-1,
        )
        self.last_prediction_query_shape = tuple(query_features.shape)
        return self.prediction_network(query_features)


class FutureMemoryResidualBridge(nn.Module):
    """Residual information adapter; NOT a replacement trajectory planner."""

    def __init__(self, latent_dim, planner_dim, future_tubelet_count):
        super().__init__()
        self.future_projection = nn.Linear(latent_dim, planner_dim)
        self.coordinate_projection = nn.Linear(2, planner_dim)
        self.time_embeddings = nn.Embedding(future_tubelet_count, planner_dim)
        self.attention = nn.MultiheadAttention(
            planner_dim, 8, dropout=0.0, batch_first=True
        )
        self.output_projection = nn.Linear(planner_dim, planner_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, current_image_memory, predicted_future_latents, coordinates, selection=None):
        memory = (
            self.future_projection(predicted_future_latents)
            + self.coordinate_projection(coordinates)[:, :, None]
            + self.time_embeddings.weight[None, None]
        ).flatten(1, 2)
        attended, _ = self.attention(
            current_image_memory, memory, memory, need_weights=False
        )
        return self.output_projection(attended)


class DriveJEPASelectivePatchFuture(nn.Module):
    """Wrap frozen official model; enabled planner remains differentiable in input."""

    def __init__(
        self,
        baseline_model,
        latent_dim=1024,
        hidden_dim=128,
        patch_budget=4,
        future_tubelet_count=4,
        grid_height=16,
        grid_width=32,
        temperature=0.7,
    ):
        super().__init__()
        self.baseline_model = baseline_model.requires_grad_(False).eval()
        self.grid_height, self.grid_width = grid_height, grid_width
        self.patch_selector = PlanningConditionedPatchSelector(
            latent_dim, hidden_dim, patch_budget, temperature
        )
        self.future_predictor = LightweightPatchFuturePredictor(
            latent_dim, hidden_dim, future_tubelet_count
        )
        planner_dim = baseline_model.image_fc.out_features
        self.future_bridge = FutureMemoryResidualBridge(
            latent_dim, planner_dim, future_tubelet_count
        )
        rows, columns = torch.meshgrid(
            torch.linspace(-1, 1, grid_height),
            torch.linspace(-1, 1, grid_width),
            indexing="ij",
        )
        self.register_buffer(
            "patch_coordinates", torch.stack((columns, rows), dim=-1).flatten(0, 1)
        )

    def train(self, mode=True):
        super().train(mode)
        self.baseline_model.eval()
        return self

    def encode_observed_clip(self, observed_camera_clip: Tensor) -> Tensor:
        """Match original preprocessing exactly; encoder is frozen for this gate."""
        batch_size, channels, frame_count, height, width = observed_camera_clip.shape
        if frame_count != 2 or (height // 16, width // 16) != (
            self.grid_height,
            self.grid_width,
        ):
            raise ValueError("Official two-frame patch grid required")
        flattened = observed_camera_clip.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frame_count, channels, height, width
        )
        normalized = self.baseline_model.transform(flattened)
        normalized = normalized.reshape(
            batch_size, frame_count, channels, height, width
        ).permute(0, 2, 1, 3, 4)
        with torch.no_grad():
            latents = self.baseline_model.image_encoder(normalized)
        if latents.shape[1] != self.grid_height * self.grid_width:
            raise RuntimeError("Unexpected official encoder grid")
        return latents.detach()

    def _predict_selected(self, current_latents, ego_status, selection_weights):
        return self.future_predictor(
            selection_weights @ current_latents,
            selection_weights @ self.patch_coordinates,
            current_latents.mean(dim=1),
            ego_status,
        )

    def forward(
        self,
        observed_camera_clip: Tensor,
        current_ego_status: Tensor,
        enable_future_branch: bool = True,
        detach_selection: bool = False,
        detach_predicted_future: bool = False,
    ):
        if not enable_future_branch:
            return self.baseline_model(observed_camera_clip, current_ego_status)
        current_latents = self.encode_observed_clip(observed_camera_clip)
        return self.forward_from_current_patch_latents(
            current_latents,
            current_ego_status,
            detach_selection=detach_selection,
            detach_predicted_future=detach_predicted_future,
        )

    def forward_from_current_patch_latents(
        self,
        current_latents: Tensor,
        current_ego_status: Tensor,
        enable_future_branch: bool = True,
        selected_patch_indices: Tensor = None,
        detach_selection: bool = False,
        detach_predicted_future: bool = False,
        current_prediction_context_latents: Tensor = None,
    ):
        """Frozen-encoder cache interface; explicit IDs are nonlearned controls.

        Only current observations enter this path. Future targets belong exclusively
        to compute_future_auxiliary_loss. Original planner weights remain frozen.
        """
        if (
            current_latents.ndim != 3
            or current_latents.shape[1] != self.patch_coordinates.shape[0]
        ):
            raise ValueError("Expected current [batch, camera-grid patch, latent]")
        prediction_context = (
            current_latents
            if current_prediction_context_latents is None
            else current_prediction_context_latents
        )
        if prediction_context.shape != current_latents.shape:
            raise ValueError("Adapted current context must match the frozen patch grid")
        selected = predicted_future = coordinates = None
        if enable_future_branch:
            if selected_patch_indices is None:
                selected = self.patch_selector(
                    prediction_context, current_ego_status, self.patch_coordinates
                )
            else:
                expected = (current_latents.shape[0], self.patch_selector.patch_budget)
                if (
                    selected_patch_indices.shape != expected
                    or selected_patch_indices.dtype != torch.long
                ):
                    raise ValueError("Explicit patch IDs must be long [batch, budget]")
                if (selected_patch_indices < 0).any() or (
                    selected_patch_indices >= current_latents.shape[1]
                ).any():
                    raise ValueError("Explicit patch ID outside current grid")
                if any(
                    row.unique().numel() != expected[1]
                    for row in selected_patch_indices
                ):
                    raise ValueError("Explicit patch IDs must be unique")
                hard_weights = F.one_hot(
                    selected_patch_indices, current_latents.shape[1]
                ).to(current_latents.dtype)
                selected = PatchSelection(
                    hard_weights,
                    hard_weights,
                    selected_patch_indices,
                    hard_weights @ self.patch_coordinates,
                )
            selection_weights = selected.selection_weights
            if detach_selection:
                selection_weights = selection_weights.detach()
                selected = replace(selected, selection_weights=selection_weights)
            predicted_future = self._predict_selected(
                prediction_context, current_ego_status, selection_weights
            )
            coordinates = selection_weights @ self.patch_coordinates
            if detach_predicted_future:
                predicted_future = predicted_future.detach()
                coordinates = coordinates.detach()
        # Match official einops B(HW)D -> BDHW, including the batch=1 stride.
        # transpose-then-reshape can give a different singleton-batch stride and
        # select a different CUDA pooling kernel despite identical feature values.
        image_grid = current_latents.reshape(
            current_latents.shape[0], self.grid_height, self.grid_width, -1
        ).permute(0, 3, 1, 2)
        pooled = (
            self.baseline_model.avg_pool(image_grid).flatten(-2, -1).permute(0, 2, 1)
        )
        image_memory = self.baseline_model.image_fc(pooled.clone())
        residual = (
            self.future_bridge(image_memory, predicted_future, coordinates, selection=selected)
            if enable_future_branch
            else torch.zeros_like(image_memory)
        )
        image_memory = image_memory + residual
        status_memory = self.baseline_model._status_encoding(current_ego_status)
        key_values = torch.cat((image_memory, status_memory[:, None]), dim=1)
        key_values = (
            key_values.clone() + self.baseline_model._keyval_embedding.weight[None]
        )
        queries = self.baseline_model._query_embedding.weight[None].repeat(
            current_latents.shape[0], 1, 1
        )
        query_output = self.baseline_model._transformer(src=key_values, tgt=queries)
        result = self.baseline_model._trajectory_head(query_output)
        return {
            **result,
            "patch_selection": selected,
            "predicted_future_latents": predicted_future,
            "future_memory_residual": residual,
            "current_patch_latents": current_latents,
            "current_prediction_context_latents": prediction_context,
        }

    def compute_future_auxiliary_loss(
        self,
        online_outputs,
        current_ego_status: Tensor,
        future_target_latents: Tensor,
        future_target_valid_mask: Tensor,
    ) -> Tensor:
        current_latents = online_outputs["current_patch_latents"].detach()
        hard_selection = online_outputs[
            "patch_selection"
        ].hard_selection_weights.detach()
        prediction = self._predict_selected(
            current_latents, current_ego_status.detach(), hard_selection
        )
        target = future_target_latents.detach()
        valid = future_target_valid_mask.detach()
        expected_target_shape = (
            current_latents.shape[0],
            self.future_predictor.future_tubelet_count,
            current_latents.shape[1],
            current_latents.shape[2],
        )
        if target.shape != expected_target_shape:
            raise ValueError(
                "Future target shape must match batch/time/grid/latent exactly"
            )
        if valid.dtype != torch.bool or valid.shape != target.shape[:-1]:
            raise ValueError("Target mask must be bool [batch,time,patch]")
        if not torch.isfinite(target[valid]).all():
            raise ValueError("Valid target values must be finite")
        safe_target = torch.where(valid[..., None], target, torch.zeros_like(target))
        selected_target = torch.einsum("bkn,btnd->bktd", hard_selection, safe_target)
        selected_valid = torch.einsum(
            "bkn,btn->bkt", hard_selection, valid.to(prediction.dtype)
        ).bool()
        error = (prediction - selected_target).square().mean(dim=-1)
        return torch.where(selected_valid, error, torch.zeros_like(error)).sum() / (
            selected_valid.sum().clamp_min(1)
        )
