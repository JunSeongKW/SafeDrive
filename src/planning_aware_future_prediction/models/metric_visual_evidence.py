"""NC/DAC residual scoring from frozen spatial image tokens.

Both conditions have the same parameters. The shared condition averages the
two query logits BEFORE softmax, giving both metrics exactly the same visual
reading. The metric-specific condition retains each metric's own reading.
Native scene/candidate representations and the other four scores are retained.
"""
import math

import torch
from torch import nn
from torch.nn import functional as functional

from .scorer_selection_learning import compose_drivor_log_score


class MetricVisualScorerCorrection(nn.Module):
    def __init__(self, visual_dimension=384, reader_dimension=64,
                 hidden_dimension=128, condition="metric_specific",
                 maximum_logit_correction=3.0):
        super().__init__()
        if condition not in {"shared", "metric_specific"}:
            raise ValueError(condition)
        self.condition = condition
        self.reader_dimension = reader_dimension
        self.maximum_logit_correction = maximum_logit_correction
        self.visual_normalization = nn.LayerNorm(visual_dimension)
        self.key_projection = nn.Linear(visual_dimension, reader_dimension)
        self.value_projection = nn.Linear(visual_dimension, reader_dimension)
        self.position_projection = nn.Linear(4, reader_dimension)
        self.candidate_query = nn.Sequential(nn.Linear(11+24, hidden_dimension), nn.SiLU(),
                                             nn.Linear(hidden_dimension, reader_dimension))
        self.metric_embedding = nn.Parameter(torch.randn(2, reader_dimension)*0.02)
        self.scored_normalization = nn.LayerNorm(256)
        self.residual_adapter = nn.Sequential(
            nn.Linear(reader_dimension+256+11+24+6, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU())
        self.correction_head = nn.Linear(hidden_dimension, 2)
        # First step trains this head; subsequent steps train the reader too.
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)

    def forward(self, visual_tokens, valid_token_mask, positions, ego_status,
                trajectories, scored_features, component_logits,
                intervention=None, capture_attention=False):
        if not valid_token_mask.any(-1).all():
            raise ValueError("Each scene must contain a valid visual token")
        safe_tokens = visual_tokens.float().masked_fill(~valid_token_mask[..., None], 0.)
        normalized = self.visual_normalization(safe_tokens)
        position_values = positions[None] if positions.ndim == 2 else positions
        position_features = self.position_projection(position_values)
        keys = self.key_projection(normalized)+position_features
        values = self.value_projection(normalized)+position_features
        ego_scale = ego_status.new_tensor([1., 1., 1., 10., 10., 4., 4., 1., 1., 1., 1.])
        trajectory_scale = trajectories.new_tensor([20., 20., math.pi])
        candidate_count = trajectories.shape[1]
        candidate_ego = (ego_status/ego_scale)[:, None].expand(-1, candidate_count, -1)
        candidate_paths = (trajectories/trajectory_scale).flatten(2)
        queries = self.candidate_query(torch.cat((candidate_ego, candidate_paths), -1))
        queries = queries[:, :, None]+self.metric_embedding[None, None]
        attention_logits = torch.einsum("bmkd,bnd->bmkn", queries, keys)/math.sqrt(self.reader_dimension)
        if self.condition == "shared":
            attention_logits = attention_logits.mean(2, keepdim=True).expand(-1, -1, 2, -1)
        if intervention == "uniform_attention":
            attention_logits = torch.zeros_like(attention_logits)
        attention_logits = attention_logits.masked_fill(~valid_token_mask[:, None, None], float("-inf"))
        attention = attention_logits.softmax(-1)
        if intervention == "swap_metric_attention":
            attention = attention.flip(2)
        visual_evidence = torch.einsum("bmkn,bnd->bmkd", attention, values)
        original_inputs = torch.cat((self.scored_normalization(scored_features), candidate_ego,
                                     candidate_paths, component_logits/5.), -1)
        inputs = torch.cat((visual_evidence, original_inputs[:, :, None].expand(-1, -1, 2, -1)), -1)
        raw_residual = self.correction_head(self.residual_adapter(inputs)).diagonal(dim1=-2, dim2=-1)
        residual = self.maximum_logit_correction*raw_residual.tanh()
        if intervention == "correction_off":
            residual = torch.zeros_like(residual)
        corrected_logits = torch.cat((component_logits[..., :2]+residual, component_logits[..., 2:]), -1)
        return corrected_logits, attention if capture_attention else None


def compute_metric_evidence_losses(corrected_logits, official_metrics, score_weights,
                                   pair_indices, ranking_temperature=1.):
    component_loss = functional.binary_cross_entropy_with_logits(
        corrected_logits[..., :2], official_metrics[..., :2])
    predicted_scores = compose_drivor_log_score(corrected_logits, score_weights)
    first_indices, second_indices = pair_indices.unbind(-1)
    target_scores = official_metrics[..., 6]
    target_gap = target_scores.gather(1, first_indices)-target_scores.gather(1, second_indices)
    predicted_gap = predicted_scores.gather(1, first_indices)-predicted_scores.gather(1, second_indices)
    gap_weights = target_gap.abs()
    ranking_loss = (gap_weights*functional.softplus(-target_gap.sign()*predicted_gap/ranking_temperature)).sum()/gap_weights.sum().clamp_min(1e-8)
    return component_loss, ranking_loss
