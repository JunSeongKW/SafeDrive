"""Small trainable modules over immutable DrivoR scorer caches.

The six frozen sub-score logits are corrected; the original score composition
and candidate count are retained. Acquisition learning is supervised and does
not pretend that the official evaluator or argmax is differentiable.
"""
import torch
from torch import nn
from torch.nn import functional as functional

SUBSCORE_NAMES = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
                  "time_to_collision_within_bound", "comfort", "driving_direction_compliance")


def compose_drivor_log_score(component_logits, score_weights):
    log_probability = functional.logsigmoid(component_logits)
    probability = component_logits.sigmoid()
    safety = (score_weights["noc"] * log_probability[..., 0]
              + score_weights["dac"] * log_probability[..., 1]
              + score_weights["ddc"] * log_probability[..., 5])
    quality = (score_weights["ep"] * probability[..., 2]
               + score_weights["ttc"] * probability[..., 3]
               + score_weights["comfort"] * probability[..., 4])
    return safety + quality.clamp_min(torch.finfo(quality.dtype).tiny).log()


def build_sampler_condition(scene_features, ego_status, base_controls, reference_standard_deviations):
    ego_scale = ego_status.new_tensor([1., 1., 1., 10., 10., 4., 4., 1., 1., 1., 1.])
    control_scale = base_controls.new_tensor([4., 1.])
    return torch.cat((scene_features.mean(1), ego_status / ego_scale,
                      (base_controls / control_scale).flatten(1),
                      (reference_standard_deviations / control_scale).flatten(1)), -1)


class SelectionErrorFamilySampler(nn.Module):
    """Regress observed acquisition utility for eight control families."""
    def __init__(self, condition_dimension=299, family_count=8, hidden_dimension=128):
        super().__init__()
        self.network = nn.Sequential(nn.LayerNorm(condition_dimension),
            nn.Linear(condition_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, family_count))

    def forward(self, scene_condition):
        return self.network(scene_condition)

    def probabilities(self, scene_condition, temperature=0.3, exploration_probability=0.25):
        probabilities = (self(scene_condition) / temperature).softmax(-1)
        return ((1-exploration_probability) * probabilities
                + exploration_probability / probabilities.shape[-1])


class CachedScorerCorrection(nn.Module):
    """Candidate-dependent six-logit residual; zero initialization is exact OFF."""
    def __init__(self, hidden_dimension=128, maximum_logit_correction=3.0):
        super().__init__()
        self.scored_feature_normalization = nn.LayerNorm(256)
        self.scene_feature_normalization = nn.LayerNorm(256)
        self.feature_adapter = nn.Sequential(nn.Linear(256+256+11+24+6, hidden_dimension),
            nn.SiLU(), nn.Linear(hidden_dimension, hidden_dimension), nn.SiLU())
        self.correction_head = nn.Linear(hidden_dimension, 6)
        nn.init.zeros_(self.correction_head.weight)
        nn.init.zeros_(self.correction_head.bias)
        self.maximum_logit_correction = maximum_logit_correction

    def forward(self, scored_features, scene_features, ego_status, trajectories,
                component_logits, *, correction_enabled=True, visual_condition_enabled=True):
        scene_summary = self.scene_feature_normalization(scene_features.mean(1))
        if not visual_condition_enabled:
            scene_summary = torch.zeros_like(scene_summary)
        ego_scale = ego_status.new_tensor([1., 1., 1., 10., 10., 4., 4., 1., 1., 1., 1.])
        trajectory_scale = trajectories.new_tensor([20., 20., 3.141592653589793])
        candidate_count = trajectories.shape[1]
        inputs = torch.cat((self.scored_feature_normalization(scored_features),
            scene_summary[:, None].expand(-1, candidate_count, -1),
            (ego_status / ego_scale)[:, None].expand(-1, candidate_count, -1),
            (trajectories / trajectory_scale).flatten(2), component_logits / 5.), -1)
        residual = self.maximum_logit_correction * self.correction_head(self.feature_adapter(inputs)).tanh()
        if not correction_enabled:
            residual = torch.zeros_like(residual)
        return component_logits + residual


def compute_scorer_correction_losses(corrected_logits, official_metrics, score_weights,
                                    pair_indices, ranking_temperature=1.0):
    component_loss = functional.binary_cross_entropy_with_logits(corrected_logits, official_metrics[..., :6])
    neural_scores = compose_drivor_log_score(corrected_logits, score_weights)
    official_scores = official_metrics[..., 6]
    first_index, second_index = pair_indices.unbind(-1)
    first_neural, second_neural = neural_scores.gather(1, first_index), neural_scores.gather(1, second_index)
    official_gap = official_scores.gather(1, first_index) - official_scores.gather(1, second_index)
    signed_neural_gap = official_gap.sign() * (first_neural-second_neural)
    weights = official_gap.abs()
    ranking_loss = (weights * functional.softplus(-signed_neural_gap / ranking_temperature)).sum() / weights.sum().clamp_min(1e-8)
    return component_loss, ranking_loss
