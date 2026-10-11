"""Diagnostic residual Gaussian sampler; no encoder, optimizer, or CEM unrolling.

Bounds are provisional trust-region choices, not experimentally selected values.
Exact identity initialization blocks adapter gradients on the FIRST backward:
the two zero-weight output heads receive gradients first. The diagnostic also
checks a small nonzero-head initialization without taking an optimizer step.
"""
import math

import torch
from torch import nn


class ResidualControlSampler(nn.Module):
    def __init__(self, visual_dimension=1024, ego_dimension=8,
                 num_control_steps=8, hidden_dimension=128,
                 maximum_mean_correction=1.0, maximum_log_scale=math.log(2.0)):
        super().__init__()
        self.num_control_steps = num_control_steps
        self.maximum_mean_correction = maximum_mean_correction
        self.maximum_log_scale = maximum_log_scale
        self.visual_normalization = nn.LayerNorm(visual_dimension)
        # Both visual/no-visual controls have exactly the same module/parameter count.
        input_dimension = visual_dimension + ego_dimension + 4 * num_control_steps
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU())
        self.mean_head = nn.Linear(hidden_dimension, 2 * num_control_steps)
        self.scale_head = nn.Linear(hidden_dimension, 2 * num_control_steps)
        for output_head in (self.mean_head, self.scale_head):
            nn.init.zeros_(output_head.weight)
            nn.init.zeros_(output_head.bias)

    def distribution_parameters(self, visual_tokens, valid_token_mask, ego_status,
                                base_controls, reference_standard_deviation,
                                use_visual_features=True, sampler_enabled=True):
        if not valid_token_mask.any(dim=1).all():
            raise ValueError("Each scene requires at least one valid visual token")
        valid_weights = valid_token_mask[..., None].to(visual_tokens.dtype)
        # Invalid values, including NaNs, are excluded rather than multiplied by zero.
        valid_features = visual_tokens.masked_fill(~valid_token_mask[..., None], 0.0)
        scene_summary = valid_features.sum(dim=1) / valid_weights.sum(dim=1)
        scene_summary = self.visual_normalization(scene_summary)
        if not use_visual_features:
            scene_summary = torch.zeros_like(scene_summary)
        adapter_input = torch.cat((scene_summary, ego_status,
                                   base_controls.flatten(1),
                                   reference_standard_deviation.flatten(1)), dim=-1)
        adapted_features = self.feature_adapter(adapter_input)
        mean_correction = self.maximum_mean_correction * self.mean_head(adapted_features).tanh()
        log_scale = self.maximum_log_scale * self.scale_head(adapted_features).tanh()
        shape = (visual_tokens.shape[0], self.num_control_steps, 2)
        mean_correction, log_scale = mean_correction.reshape(shape), log_scale.reshape(shape)
        if not sampler_enabled:
            mean_correction, log_scale = torch.zeros_like(mean_correction), torch.zeros_like(log_scale)
        return mean_correction, log_scale

    def forward(self, visual_tokens, valid_token_mask, ego_status, base_controls,
                reference_standard_deviation, standard_normal_samples, **options):
        mean_correction, log_scale = self.distribution_parameters(
            visual_tokens, valid_token_mask, ego_status, base_controls,
            reference_standard_deviation, **options)
        sampled_controls = base_controls[:, None] + reference_standard_deviation[:, None] * (
            mean_correction[:, None] + log_scale[:, None].exp() * standard_normal_samples)
        return sampled_controls, mean_correction, log_scale


def gaussian_reference_kl(mean_correction, log_scale):
    """KL BEFORE clipping/rollout, summed over all time/control dimensions."""
    return 0.5 * (log_scale.mul(2).exp() + mean_correction.square()
                  - 1.0 - 2.0 * log_scale).flatten(1).sum(dim=1)


def soft_best_candidate_loss(candidate_utility, mean_correction, log_scale,
                             temperature=0.1, reference_kl_weight=0.01):
    """Diagnostic surrogate only: NOT the final CEM or official PDMS objective."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    soft_best = temperature * (torch.logsumexp(candidate_utility / temperature, dim=1)
                               - math.log(candidate_utility.shape[1]))
    return (-soft_best + reference_kl_weight * gaussian_reference_kl(
        mean_correction, log_scale)).mean()


class FrozenDrivorTrajectoryScorer(nn.Module):
    """Original DrivoR scoring computation, with input detach omitted.

    Native DrivoR decoder has candidate-to-candidate self-attention: scores are
    functions of the whole candidate SET, not independent trajectory costs.
    All weights are frozen; no no_grad context is used in forward.
    """
    def __init__(self, planner):
        super().__init__()
        self.position_embedding = planner.pos_embed
        self.scorer_attention = planner.scorer_attention
        self.score_heads = planner.scorer
        self.score_weights = {name: float(getattr(planner._config, name))
                              for name in ("noc", "dac", "ddc", "ttc", "ep", "comfort")}
        self.requires_grad_(False)
        self.eval()

    def forward(self, trajectories, scene_features, ego_token):
        embedded_trajectories = self.position_embedding(trajectories.flatten(2))
        scored_features = self.scorer_attention(embedded_trajectories, scene_features) + ego_token
        logits = self.score_heads(trajectories, scored_features)[0]
        weights = self.score_weights
        return (weights["noc"] * logits["no_at_fault_collisions"].sigmoid().log()
                + weights["dac"] * logits["drivable_area_compliance"].sigmoid().log()
                + weights["ddc"] * logits["driving_direction_compliance"].sigmoid().log()
                + (weights["ttc"] * logits["time_to_collision_within_bound"].sigmoid()
                   + weights["ep"] * logits["ego_progress"].sigmoid()
                   + weights["comfort"] * logits["comfort"].sigmoid()).log())
