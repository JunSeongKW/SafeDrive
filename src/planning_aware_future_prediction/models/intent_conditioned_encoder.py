"""Planning-facing encoder adaptation with training-only latent prediction.

The frozen prefix may be cached exactly. The trainable tail IS the encoder used
by the planner, not a separate future branch. Masks are applied before the first
prefix block; cached masked prefixes retain their original spatial token IDs.
"""

import copy

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@torch.no_grad()
def encode_frozen_prefix(baseline_model, observed_camera_clip, num_trainable_blocks=2,
                         observed_patch_indices=None):
    """Exact original preprocessing and frozen prefix; remove tokens before attention."""
    batch_size, channels, frame_count, image_height, image_width = observed_camera_clip.shape
    if frame_count != 2 or (image_height, image_width) != (256, 512):
        raise ValueError("Expected the original two-frame 256x512 front-camera input")
    normalized_clip = observed_camera_clip.permute(0, 2, 1, 3, 4).reshape(
        batch_size * frame_count, channels, image_height, image_width)
    normalized_clip = baseline_model.transform(normalized_clip).reshape(
        batch_size, frame_count, channels, image_height, image_width).permute(0, 2, 1, 3, 4)
    encoder = baseline_model.image_encoder
    prefix_features = encoder.patch_embed(normalized_clip)
    if not encoder.use_rope:
        prefix_features = prefix_features + encoder.interpolate_pos_encoding(normalized_clip, encoder.pos_embed)
    if observed_patch_indices is not None:
        prefix_features = prefix_features.gather(
            1, observed_patch_indices[..., None].expand(-1, -1, prefix_features.shape[-1]))
    for block in encoder.blocks[:-num_trainable_blocks]:
        prefix_features = block(prefix_features, mask=observed_patch_indices, attn_mask=None,
                                T=1, H_patches=16, W_patches=32)
    return prefix_features.detach()


def plan_from_encoder_features(baseline_model, encoder_features, ego_status):
    """Official PF planning computation; frozen weights still pass input gradients."""
    image_grid = encoder_features.reshape(-1, 16, 32, 1024).permute(0, 3, 1, 2)
    pooled_features = baseline_model.avg_pool(image_grid).flatten(-2).transpose(1, 2)
    image_memory = baseline_model.image_fc(pooled_features.clone())
    status_memory = baseline_model._status_encoding(ego_status)
    memory = torch.cat((image_memory, status_memory[:, None]), dim=1)
    memory = memory.clone() + baseline_model._keyval_embedding.weight[None]
    queries = baseline_model._query_embedding.weight[None].repeat(encoder_features.shape[0], 1, 1)
    return baseline_model._trajectory_head(baseline_model._transformer(src=memory, tgt=queries))


def pool_spatial_regions(patch_features: Tensor) -> Tensor:
    """Average 2x2 patches, preserving preceding batch/time dimensions."""
    leading_shape = patch_features.shape[:-2]
    latent_width = patch_features.shape[-1]
    region_features = patch_features.reshape(-1, 8, 2, 16, 2, latent_width).mean((2, 4))
    return region_features.reshape(*leading_shape, 128, latent_width)


def region_indices_to_patch_indices(region_indices: Tensor) -> Tensor:
    region_rows, region_columns = region_indices // 16, region_indices % 16
    top_left = region_rows * 64 + region_columns * 2
    offsets = torch.tensor([0, 1, 32, 33], device=region_indices.device)
    return (top_left[..., None] + offsets).flatten(-2)


def visible_patch_indices(selected_regions: Tensor) -> Tensor:
    removed_patches = region_indices_to_patch_indices(selected_regions)
    observed_mask = torch.ones(selected_regions.shape[0], 512, dtype=torch.bool,
                               device=selected_regions.device)
    observed_mask.scatter_(1, removed_patches, False)
    all_indices = torch.arange(512, device=selected_regions.device).expand_as(observed_mask)
    return all_indices[observed_mask].reshape(selected_regions.shape[0], -1)


def select_training_regions(region_scores, selection_policy, selected_region_budget, generator):
    """Sample without replacement, with 50% uniform exploration for guided targets."""
    selected_rows = []
    for scores in region_scores.cpu():
        priority_count = 0 if selection_policy == "uniform" else selected_region_budget // 2
        priority = torch.argsort(scores, descending=True, stable=True)[:priority_count]
        remaining_mask = torch.ones(scores.numel(), dtype=torch.bool)
        remaining_mask[priority] = False
        remaining_indices = torch.arange(scores.numel())[remaining_mask]
        random_indices = remaining_indices[torch.randperm(len(remaining_indices), generator=generator)
                                             [:selected_region_budget - priority_count]]
        selected_rows.append(torch.cat((priority, random_indices)).sort().values)
    return torch.stack(selected_rows).to(region_scores.device)


class IntentConditionedEncoderTail(nn.Module):
    """Full updates of the final original ViT blocks with optional internal FiLM."""

    def __init__(self, official_encoder, num_trainable_blocks=2, use_ego_intent=True):
        super().__init__()
        self.blocks = copy.deepcopy(official_encoder.blocks[-num_trainable_blocks:])
        self.normalization = copy.deepcopy(official_encoder.norm)
        self.blocks.requires_grad_(True)
        self.normalization.requires_grad_(True)
        self.use_ego_intent = use_ego_intent
        self.intent_conditioning = nn.Sequential(nn.Linear(8, 128), nn.GELU(),
                                                nn.Linear(128, num_trainable_blocks * 2048))
        nn.init.zeros_(self.intent_conditioning[-1].weight)
        nn.init.zeros_(self.intent_conditioning[-1].bias)
        self.intent_conditioning.requires_grad_(use_ego_intent)
        self.register_buffer("ego_status_scale", torch.tensor([1., 1., 1., 1., 20., 20., 5., 5.]))

    def forward(self, observed_prefix_features, ego_status, observed_patch_indices=None):
        encoder_features = observed_prefix_features
        modulation = None
        if self.use_ego_intent:
            modulation = self.intent_conditioning(ego_status / self.ego_status_scale)
            modulation = modulation.reshape(ego_status.shape[0], len(self.blocks), 2, 1024)
        for block_index, block in enumerate(self.blocks):
            if modulation is not None:
                feature_scale = modulation[:, block_index, 0, None]
                feature_shift = modulation[:, block_index, 1, None]
                encoder_features = encoder_features * (1 + feature_scale) + feature_shift
            encoder_features = block(encoder_features, mask=observed_patch_indices,
                                     attn_mask=None, T=1, H_patches=16, W_patches=32)
        return self.normalization(encoder_features)


class TrainingOnlyFutureHead(nn.Module):
    """Position/time queries predict fixed-teacher latents from online encoder output.

    Ego intent reaches this head only through encoder features. No direct ego
    shortcut, cached current-feature residual, or path into planning is present.
    """

    def __init__(self, latent_width=1024, hidden_width=128, num_future_steps=4):
        super().__init__()
        self.feature_projection = nn.Linear(latent_width, hidden_width)
        self.position_projection = nn.Linear(2, hidden_width)
        self.future_step_embedding = nn.Embedding(num_future_steps, hidden_width)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            hidden_width, 4, 4 * hidden_width, dropout=0, activation="gelu",
            batch_first=True, norm_first=True), 2, norm=nn.LayerNorm(hidden_width))
        self.output_projection = nn.Linear(hidden_width, latent_width)
        patch_rows, patch_columns = torch.meshgrid(torch.linspace(-1, 1, 16),
                                                   torch.linspace(-1, 1, 32), indexing="ij")
        self.register_buffer("patch_coordinates", torch.stack((patch_columns, patch_rows), -1).flatten(0, 1))
        self.register_buffer("region_coordinates", pool_spatial_regions(self.patch_coordinates))

    def forward(self, encoder_features, selected_regions, observed_patch_indices=None):
        coordinates = self.patch_coordinates[None].expand(encoder_features.shape[0], -1, -1)
        if observed_patch_indices is not None:
            coordinates = coordinates.gather(1, observed_patch_indices[..., None].expand(-1, -1, 2))
        memory = self.feature_projection(encoder_features) + self.position_projection(coordinates)
        selected_coordinates = self.region_coordinates[selected_regions]
        queries = (self.position_projection(selected_coordinates)[:, :, None]
                   + self.future_step_embedding.weight[None, None])
        predicted_features = self.output_projection(self.decoder(queries.flatten(1, 2), memory))
        return predicted_features.reshape(encoder_features.shape[0], selected_regions.shape[1], 4, -1)


def compute_selected_latent_loss(predicted_features, teacher_region_features,
                                future_valid_mask, selected_regions):
    """Only valid fixed-teacher targets supervise the encoder and predictor."""
    selected_targets = teacher_region_features.gather(
        2, selected_regions[:, None, :, None].expand(-1, 4, -1, teacher_region_features.shape[-1]))
    selected_targets = selected_targets.transpose(1, 2).detach()
    selected_valid = future_valid_mask.gather(2, selected_regions[:, None].expand(-1, 4, -1)).transpose(1, 2)
    safe_targets = torch.where(selected_valid[..., None], selected_targets, torch.zeros_like(selected_targets))
    normalized_targets = F.layer_norm(safe_targets, (safe_targets.shape[-1],))
    errors = (predicted_features - normalized_targets).square().mean(-1)
    return torch.where(selected_valid, errors, torch.zeros_like(errors)).sum() / selected_valid.sum().clamp_min(1)
