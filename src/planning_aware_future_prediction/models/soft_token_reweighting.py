"""Cached perception-free planning with bias only in the first cross-attention.

The components are copied from the actual official Drive-JEPA instance. There is
no image encoder in this training module, and no change to its key/value vectors.
"""
import copy

import torch
from torch import nn


PLANNER_COMPONENTS = (
    "image_fc", "_status_encoding", "_keyval_embedding", "_query_embedding",
    "_transformer", "_trajectory_head",
)


def copy_official_planner(official_model):
    return nn.ModuleDict({name: copy.deepcopy(getattr(official_model, name)).cpu()
                          for name in PLANNER_COMPONENTS})


def token_coordinates():
    rows, columns = torch.meshgrid(torch.linspace(-1, 1, 8),
                                  torch.linspace(-1, 1, 16), indexing="ij")
    # One front camera; a tubelet jointly describes [-0.5, 0] seconds.
    return torch.stack((columns, rows, torch.zeros_like(rows),
                        torch.full_like(rows, -0.25)), dim=-1).flatten(0, 1)


class ConditionalTokenImportance(nn.Module):
    def __init__(self, feature_dimension=1024, hidden_dimension=128,
                 normalize_importance=False, normalization_epsilon=1e-6):
        super().__init__()
        self.normalize_importance = normalize_importance
        self.normalization_epsilon = normalization_epsilon
        if normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        self.feature_normalization = nn.LayerNorm(feature_dimension)
        self.ego_embedding = nn.Sequential(nn.Linear(8, 32), nn.SiLU())
        self.importance_mlp = nn.Sequential(
            nn.Linear(2 * feature_dimension + 32 + 4, hidden_dimension),
            nn.SiLU(), nn.Linear(hidden_dimension, 1),
        )
        nn.init.normal_(self.importance_mlp[-1].weight, std=0.001)
        nn.init.zeros_(self.importance_mlp[-1].bias)
        self.beta = nn.Parameter(torch.tensor(0.1))

    def forward(self, visual_tokens, valid_token_mask, ego_status, positions):
        if not valid_token_mask.any(dim=1).all():
            raise ValueError("Every scene needs at least one valid visual token")
        normalized = self.feature_normalization(visual_tokens)
        valid_weights = valid_token_mask.unsqueeze(-1).to(normalized.dtype)
        scene_context = (normalized * valid_weights).sum(1) / valid_weights.sum(1)
        ego_embedding = self.ego_embedding(ego_status)
        token_count = visual_tokens.shape[1]
        module_input = torch.cat((normalized,
            scene_context[:, None].expand(-1, token_count, -1),
            ego_embedding[:, None].expand(-1, token_count, -1),
            positions[None].expand(visual_tokens.shape[0], -1, -1)), dim=-1)
        importance = self.importance_mlp(module_input).squeeze(-1)
        # Remove the unidentifiable scene-wide offset without detaching gradients.
        valid_mean = (importance * valid_token_mask).sum(1, keepdim=True) / valid_token_mask.sum(1, keepdim=True)
        centered_importance = (importance - valid_mean).masked_fill(~valid_token_mask, 0)
        if self.normalize_importance:
            # Population std across valid tokens; vector_norm has a finite
            # derivative at a constant/one-valid-token input. No detach or
            # additional parameters. Preserve the original centering above.
            valid_count = valid_token_mask.sum(1, keepdim=True).to(importance.dtype)
            standard_deviation = torch.linalg.vector_norm(
                centered_importance, dim=1, keepdim=True) / valid_count.sqrt()
            centered_importance = centered_importance / (
                standard_deviation + self.normalization_epsilon)
        return centered_importance


def cross_attention_bias(visual_importance, valid_token_mask, beta, head_count, query_count):
    """MHA float mask [batch*heads, queries, visual+ego]; -inf means forbidden."""
    visual_bias = beta * visual_importance
    visual_bias = visual_bias.masked_fill(~valid_token_mask, float("-inf"))
    # The original ego memory remains visible, with no learned importance bias.
    complete_bias = torch.cat((visual_bias, visual_bias.new_zeros(visual_bias.shape[0], 1)), dim=1)
    return complete_bias[:, None, None, :].expand(-1, head_count, query_count, -1).reshape(
        -1, query_count, complete_bias.shape[-1])


class CachedTokenPlanner(nn.Module):
    def __init__(self, planner_components, condition="baseline",
                 normalize_importance=False, normalization_epsilon=1e-6):
        super().__init__()
        if condition not in {"baseline", "unconditioned", "conditioned"}:
            raise ValueError(condition)
        self.planner = planner_components
        self.condition = condition
        self.importance = None if condition == "baseline" else ConditionalTokenImportance(
            normalize_importance=normalize_importance,
            normalization_epsilon=normalization_epsilon)
        self.register_buffer("positions", token_coordinates())
        self.last_importance = None
        self.last_attention = None

    def forward(self, visual_tokens, valid_token_mask, ego_status,
                intervention=None, capture_attention=False):
        transformer = self.planner["_transformer"]
        visual_memory = self.planner["image_fc"](visual_tokens)
        ego_memory = self.planner["_status_encoding"](ego_status)
        memory_input = torch.cat((visual_memory, ego_memory[:, None]), dim=1)
        memory_input = memory_input + self.planner["_keyval_embedding"].weight[None]
        queries = self.planner["_query_embedding"].weight[None].expand(visual_tokens.shape[0], -1, -1)
        padding_mask = torch.cat((~valid_token_mask,
                                  torch.zeros_like(valid_token_mask[:, :1])), dim=1)
        padding = padding_mask if padding_mask.any() else None
        if self.importance is None and not capture_attention:
            return self.planner["_trajectory_head"](transformer(
                src=memory_input, tgt=queries, src_key_padding_mask=padding,
                memory_key_padding_mask=padding))["trajectory"]
        memory = transformer.encoder(memory_input, src_key_padding_mask=padding)
        if self.importance is None:
            importance = visual_tokens.new_zeros(valid_token_mask.shape)
            beta = 0.0
        else:
            conditioning = ego_status if self.condition == "conditioned" else torch.zeros_like(ego_status)
            importance = self.importance(visual_tokens, valid_token_mask, conditioning, self.positions)
            if intervention == "shuffle":
                # Deterministic for the fixed evaluation batch, valid positions only.
                generator = torch.Generator(device=importance.device).manual_seed(1907)
                permutation = torch.arange(importance.shape[1], device=importance.device).repeat(importance.shape[0], 1)
                for scene_index in range(importance.shape[0]):
                    valid_indices = torch.where(valid_token_mask[scene_index])[0]
                    permutation[scene_index, valid_indices] = valid_indices[torch.randperm(
                        len(valid_indices), generator=generator, device=importance.device)]
                importance = importance.gather(1, permutation)
            beta = self.importance.beta if intervention != "beta_zero" else 0.0
        self.last_importance = importance
        mask = cross_attention_bias(importance, valid_token_mask, beta,
                                    transformer.nhead, queries.shape[1])
        if capture_attention:
            # Capture the exact cross-attention query after first self-attention.
            first_layer = transformer.decoder.layers[0]
            if first_layer.norm_first:
                raise ValueError("Official PF uses post-norm decoder layers")
            self_output = first_layer.self_attn(queries, queries, queries, need_weights=False)[0]
            cross_query = first_layer.norm1(queries + first_layer.dropout1(self_output))
            _, self.last_attention = first_layer.multihead_attn(
                cross_query, memory, memory, attn_mask=mask,
                need_weights=True, average_attn_weights=False)
        for layer_index, layer in enumerate(transformer.decoder.layers):
            queries = layer(queries, memory,
                            memory_mask=mask if layer_index == 0 else None,
                            memory_key_padding_mask=padding if layer_index != 0 else None)
        if transformer.decoder.norm is not None:
            queries = transformer.decoder.norm(queries)
        return self.planner["_trajectory_head"](queries)["trajectory"]
