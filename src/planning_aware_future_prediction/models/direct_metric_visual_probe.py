"""Predict candidate NC/DAC from frozen spatial tokens without scorer shortcuts.

Ego, trajectory and metric are joined nonlinearly in the query. The prediction
head sees only the visual readout. Position enters keys, never values. A shared
reader and a metric-specific reader contain exactly the same parameters.
"""
import math

import torch
from torch import nn


class DirectMetricVisualProbe(nn.Module):
    def __init__(self, visual_dimension=384, reader_dimension=64,
                 hidden_dimension=128, metric_dimension=16,
                 condition="metric_specific"):
        super().__init__()
        if condition not in {"shared", "metric_specific"}:
            raise ValueError(condition)
        self.condition = condition
        self.reader_dimension = reader_dimension
        self.visual_normalization = nn.LayerNorm(visual_dimension)
        self.key_projection = nn.Linear(visual_dimension, reader_dimension)
        self.value_projection = nn.Linear(visual_dimension, reader_dimension)
        self.key_position_projection = nn.Linear(4, reader_dimension)
        self.metric_embedding = nn.Parameter(torch.randn(2, metric_dimension) * 0.2)
        self.candidate_metric_query = nn.Sequential(
            nn.Linear(11 + 24 + metric_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, reader_dimension),
        )
        self.metric_prediction_head = nn.Sequential(
            nn.Linear(reader_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, 2),
        )

    def forward(self, visual_tokens, valid_token_mask, positions, ego_status,
                trajectories, *, intervention=None, capture_attention=False):
        if not valid_token_mask.any(-1).all():
            raise ValueError("Each scene must contain a valid visual token")
        safe_tokens = visual_tokens.float().masked_fill(~valid_token_mask[..., None], 0.)
        normalized_features = self.visual_normalization(safe_tokens)
        position_values = positions[None] if positions.ndim == 2 else positions
        safe_positions = position_values.expand(len(visual_tokens), -1, -1).masked_fill(
            ~valid_token_mask[..., None], 0.)
        keys = self.key_projection(normalized_features) + self.key_position_projection(safe_positions)
        values = self.value_projection(normalized_features)
        ego_scale = ego_status.new_tensor([1., 1., 1., 10., 10., 4., 4., 1., 1., 1., 1.])
        trajectory_scale = trajectories.new_tensor([20., 20., math.pi])
        candidate_count = trajectories.shape[1]
        candidate_condition = torch.cat((
            (ego_status / ego_scale)[:, None].expand(-1, candidate_count, -1),
            (trajectories / trajectory_scale).flatten(2),
        ), -1)
        joint_condition = torch.cat((
            candidate_condition[:, :, None].expand(-1, -1, 2, -1),
            self.metric_embedding[None, None].expand(len(visual_tokens), candidate_count, -1, -1),
        ), -1)
        queries = self.candidate_metric_query(joint_condition)
        attention_logits = torch.einsum("bmkd,bnd->bmkn", queries, keys) / math.sqrt(self.reader_dimension)
        if self.condition == "shared":
            attention_logits = attention_logits.mean(2, keepdim=True).expand(-1, -1, 2, -1)
        if intervention == "uniform_attention":
            attention_logits = torch.zeros_like(attention_logits)
        attention = attention_logits.masked_fill(
            ~valid_token_mask[:, None, None], float("-inf")).softmax(-1)
        if intervention == "swap_metric_attention":
            attention = attention.flip(2)
        visual_readout = torch.einsum("bmkn,bnd->bmkd", attention, values)
        predicted_logits = self.metric_prediction_head(visual_readout).diagonal(dim1=-2, dim2=-1)
        return predicted_logits, attention if capture_attention else None


def replace_nc_dac_logits(original_component_logits, predicted_metric_logits, *, probe_enabled=True):
    """Preserve native score composition; OFF exactly restores native logits."""
    if not probe_enabled:
        return original_component_logits
    return torch.cat((predicted_metric_logits, original_component_logits[..., 2:]), -1)
