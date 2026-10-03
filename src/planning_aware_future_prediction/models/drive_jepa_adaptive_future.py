"""Bounded architecture follow-up; original Drive-JEPA planner stays frozen.

LoRA adapts a copied encoder tail for the future branch only. Frozen baseline
features still feed the original planner. Teacher targets never enter forward.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from .drive_jepa_selective_patch_future import (
    DriveJEPASelectivePatchFuture,
    PatchSelection,
)


class LowRankAdaptedLinear(nn.Module):
    """Frozen linear map plus alpha/rank * B(A(x)); initially identical."""

    def __init__(self, frozen_linear, rank=4, alpha=4):
        super().__init__()
        if rank < 1:
            raise ValueError("Positive LoRA rank required")
        self.frozen_linear = frozen_linear.requires_grad_(False)
        self.input_factor = nn.Parameter(
            frozen_linear.weight.new_empty(rank, frozen_linear.in_features)
        )
        self.output_factor = nn.Parameter(
            frozen_linear.weight.new_zeros(frozen_linear.out_features, rank)
        )
        nn.init.kaiming_uniform_(self.input_factor, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, features):
        return self.frozen_linear(features) + self.scale * F.linear(
            F.linear(features, self.input_factor), self.output_factor
        )


def configure_future_projection(model, projection_mode, seed):
    """Controlled frozen/random/official/LoRA future-memory projection.

    Called after loading the common warmup state. Original image_fc is copied,
    never mutated. This changes only the future bridge's projection, not encoder,
    current memory, trajectory decoder, target definition or auxiliary weights.
    """
    if projection_mode == "learned_random":
        return
    if projection_mode == "frozen_random":
        model.future_bridge.future_projection.requires_grad_(False)
        return
    if projection_mode not in ("frozen_official", "lora_official"):
        raise ValueError("Unregistered future projection mode")
    projection = copy.deepcopy(model.baseline_model.image_fc).requires_grad_(False)
    if projection_mode == "lora_official":
        # Keep other module initialization and training RNG untouched.
        devices = [projection.weight.device.index] if projection.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 50000)
            projection = LowRankAdaptedLinear(projection, rank=4, alpha=4)
    model.future_bridge.future_projection = projection


class FutureBranchLoRAEncoderTail(nn.Module):
    """Last official ViT blocks copied, with only QKV low-rank factors trainable."""

    def __init__(self, official_encoder, block_count=4, rank=4, alpha=4):
        super().__init__()
        self.blocks = copy.deepcopy(official_encoder.blocks[-block_count:])
        self.normalization = copy.deepcopy(official_encoder.norm)
        self.requires_grad_(False)
        for block in self.blocks:
            block.attn.qkv = LowRankAdaptedLinear(block.attn.qkv, rank, alpha)
        self.eval()

    def train(self, mode=True):
        # Official stochastic layers stay disabled; LoRA has no dropout.
        return super().train(False)

    def forward(self, frozen_prefix_latents, grid_height=16, grid_width=32):
        adapted = frozen_prefix_latents
        for block in self.blocks:
            adapted = block(
                adapted,
                mask=None,
                attn_mask=None,
                T=1,
                H_patches=grid_height,
                W_patches=grid_width,
            )
        return self.normalization(adapted)


class EgoQueryPatchSelector(nn.Module):
    """Ego-conditioned query scores content/position keys; unique hard K + ST."""

    def __init__(self, latent_dim, hidden_dim, patch_budget, temperature=0.7):
        super().__init__()
        self.patch_budget, self.temperature = patch_budget, temperature
        self.patch_keys = nn.Linear(latent_dim, hidden_dim)
        self.coordinate_keys = nn.Linear(2, hidden_dim)
        self.intent_query = nn.Sequential(
            nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.context_query = nn.Linear(latent_dim, hidden_dim)
        self.key_normalization = nn.LayerNorm(hidden_dim)
        self.query_normalization = nn.LayerNorm(hidden_dim)

    def forward(self, current_patch_latents, current_ego_status, patch_coordinates):
        if current_ego_status.shape != (current_patch_latents.shape[0], 8):
            raise ValueError("Expected eight-channel observed ego status")
        if self.patch_budget > current_patch_latents.shape[1]:
            raise ValueError("Insufficient candidate patches")
        keys = self.key_normalization(
            self.patch_keys(current_patch_latents)
            + self.coordinate_keys(patch_coordinates)[None]
        )
        query = self.query_normalization(
            self.intent_query(current_ego_status)
            + self.context_query(current_patch_latents.mean(dim=1))
        )
        scores = torch.einsum("bnd,bd->bn", keys, query) / math.sqrt(keys.shape[-1])
        remaining = torch.ones_like(scores, dtype=torch.bool)
        hard_rows, soft_rows, indices = [], [], []
        for _ in range(self.patch_budget):
            masked = scores.masked_fill(~remaining, -torch.inf)
            soft = (masked / self.temperature).softmax(dim=-1)
            index = masked.argmax(dim=-1)
            hard = F.one_hot(index, scores.shape[1]).to(scores.dtype)
            hard_rows.append(hard)
            soft_rows.append(soft)
            indices.append(index)
            remaining = remaining & ~hard.bool()
        hard = torch.stack(hard_rows, dim=1)
        soft = torch.stack(soft_rows, dim=1)
        weights = hard + (soft - soft.detach())
        return PatchSelection(
            weights, hard, torch.stack(indices, dim=1), weights @ patch_coordinates
        )


class ContextualResidualFuturePredictor(nn.Module):
    """Selected queries cross-attend ALL current patches; predict feature change.

    Only K*time output queries, but context projection/attention still uses the full
    current grid. This is not a claim of end-to-end sparse encoder computation.
    """

    def __init__(self, latent_dim, hidden_dim, future_tubelet_count):
        super().__init__()
        self.future_tubelet_count = future_tubelet_count
        self.current_projection = nn.Linear(latent_dim, hidden_dim)
        self.coordinate_projection = nn.Linear(2, hidden_dim)
        self.ego_projection = nn.Linear(8, hidden_dim)
        self.time_embeddings = nn.Embedding(future_tubelet_count, hidden_dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                hidden_dim,
                4,
                4 * hidden_dim,
                dropout=0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            2,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.delta_head = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.last_prediction_query_shape = None

    def forward(
        self,
        selected_current_latents,
        coordinates,
        current_patch_latents,
        ego_status,
        all_patch_coordinates,
        query_time_embeddings=None,
    ):
        batch_size, patch_budget, _ = selected_current_latents.shape
        memory = (
            self.current_projection(current_patch_latents)
            + self.coordinate_projection(all_patch_coordinates)[None]
        )
        ego_query = self.ego_projection(ego_status)
        memory = torch.cat((memory, ego_query[:, None]), dim=1)
        time_embeddings = (
            self.time_embeddings.weight
            if query_time_embeddings is None
            else query_time_embeddings
        )
        queries = (
            self.current_projection(selected_current_latents)[:, :, None]
            + self.coordinate_projection(coordinates)[:, :, None]
            + time_embeddings[None, None]
            + ego_query[:, None, None]
        )
        self.last_prediction_query_shape = tuple(queries.shape)
        decoded = self.decoder(queries.flatten(1, 2), memory).reshape(
            batch_size, patch_budget, time_embeddings.shape[0], -1
        )
        return selected_current_latents[:, :, None] + self.delta_head(decoded)


class DriveJEPAAdaptiveFuture(DriveJEPASelectivePatchFuture):
    def __init__(
        self,
        baseline_model,
        contextual_predictor=True,
        ego_query_selector=True,
        use_encoder_lora=False,
        **kwargs,
    ):
        super().__init__(baseline_model, **kwargs)
        latent_dim, hidden_dim = (
            kwargs.get("latent_dim", 1024),
            kwargs.get("hidden_dim", 128),
        )
        if contextual_predictor:
            self.future_predictor = ContextualResidualFuturePredictor(
                latent_dim, hidden_dim, kwargs.get("future_tubelet_count", 4)
            )
        if ego_query_selector:
            self.patch_selector = EgoQueryPatchSelector(
                latent_dim,
                hidden_dim,
                kwargs.get("patch_budget", 4),
                kwargs.get("temperature", 0.7),
            )
        self.encoder_lora_tail = (
            FutureBranchLoRAEncoderTail(baseline_model.image_encoder)
            if use_encoder_lora
            else None
        )

    def _predict_selected(self, current_latents, ego_status, selection_weights):
        if not isinstance(self.future_predictor, ContextualResidualFuturePredictor):
            return super()._predict_selected(
                current_latents, ego_status, selection_weights
            )
        return self.future_predictor(
            selection_weights @ current_latents,
            selection_weights @ self.patch_coordinates,
            current_latents,
            ego_status,
            self.patch_coordinates,
        )

    def forward_cached_observations(
        self,
        current_patch_latents,
        current_ego_status,
        current_prefix_latents=None,
        **kwargs,
    ):
        prediction_context = current_patch_latents
        if self.encoder_lora_tail is not None and kwargs.get(
            "enable_future_branch", True
        ):
            if current_prefix_latents is None:
                raise ValueError(
                    "LoRA requires frozen current encoder-prefix activations"
                )
            prediction_context = self.encoder_lora_tail(
                current_prefix_latents, self.grid_height, self.grid_width
            )
        return self.forward_from_current_patch_latents(
            current_patch_latents,
            current_ego_status,
            current_prediction_context_latents=prediction_context,
            **kwargs,
        )

    def forward(
        self,
        observed_camera_clip,
        current_ego_status,
        enable_future_branch=True,
        **kwargs,
    ):
        if not enable_future_branch:
            return self.baseline_model(observed_camera_clip, current_ego_status)
        captured_prefix = []
        handle = None
        if self.encoder_lora_tail is not None:
            handle = self.baseline_model.image_encoder.blocks[
                -4
            ].register_forward_pre_hook(
                lambda module, inputs: captured_prefix.append(inputs[0].detach())
            )
        try:
            current_latents = self.encode_observed_clip(observed_camera_clip)
        finally:
            if handle is not None:
                handle.remove()
        return self.forward_cached_observations(
            current_latents,
            current_ego_status,
            captured_prefix[0] if captured_prefix else None,
            **kwargs,
        )

    def compute_future_auxiliary_loss(
        self,
        online_outputs,
        current_ego_status,
        future_target_latents,
        future_target_valid_mask,
    ):
        # Deliberately retain current-context -> LoRA gradient, never selector or GT.
        context = online_outputs["current_prediction_context_latents"]
        hard_selection = online_outputs[
            "patch_selection"
        ].hard_selection_weights.detach()
        prediction = self._predict_selected(
            context, current_ego_status.detach(), hard_selection
        )
        target, valid = (
            future_target_latents.detach(),
            future_target_valid_mask.detach(),
        )
        if target.shape != (
            context.shape[0],
            self.future_predictor.future_tubelet_count,
            context.shape[1],
            context.shape[2],
        ):
            raise ValueError("Future target must match batch/time/grid/latent")
        if valid.dtype != torch.bool or valid.shape != target.shape[:-1]:
            raise ValueError("Future target mask must be boolean and shape-matched")
        if not torch.isfinite(target[valid]).all():
            raise ValueError("Valid targets must be finite")
        safe_target = torch.where(valid[..., None], target, torch.zeros_like(target))
        selected_target = torch.einsum("bkn,btnd->bktd", hard_selection, safe_target)
        selected_valid = torch.einsum(
            "bkn,btn->bkt", hard_selection, valid.to(prediction.dtype)
        ).bool()
        error = (prediction - selected_target).square().mean(dim=-1)
        return torch.where(
            selected_valid, error, torch.zeros_like(error)
        ).sum() / selected_valid.sum().clamp_min(1)
