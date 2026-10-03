"""LPWM particle memory with train-only candidate and rule-metric distillation.

An adaptation inspired by DrivoR's compact-memory scorer and Hydra-MDP's
vocabulary/teacher supervision, not a reproduction of either visual backbone.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lpwm_planning_finetuning import PlanningFineTunedLPWM
from .lpwm_planner import normalized_ego_status


def compose_candidate_pdms(metric_probabilities):
    """NAVSIM-v1 weights; driving direction is supervised but has weight zero."""
    collision, drivable, progress, ttc, comfort, _direction = metric_probabilities.unbind(-1)
    return collision * drivable * (5 * progress + 5 * ttc + 2 * comfort) / 12


class ParticleCandidatePlanner(PlanningFineTunedLPWM):
    def __init__(self, pretrained_checkpoint, trajectory_vocabulary, metric_selection=True,
                 imitation_exponent=.1, hidden_dimension=256, refine_candidates=False):
        super().__init__(pretrained_checkpoint, hidden_dimension)
        del self.waypoint_queries, self.trajectory_decoder, self.trajectory_head
        candidates = torch.as_tensor(trajectory_vocabulary, dtype=torch.float32)
        assert candidates.ndim == 3 and candidates.shape[1:] == (8, 3) and torch.isfinite(candidates).all()
        self.register_buffer("trajectory_vocabulary", candidates)
        self.candidate_projection = nn.Sequential(nn.Linear(24, hidden_dimension), nn.GELU(), nn.LayerNorm(hidden_dimension))
        decoder_layer = nn.TransformerDecoderLayer(hidden_dimension, 8, hidden_dimension * 4,
            dropout=.1, batch_first=True, norm_first=True)
        self.candidate_decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.imitation_head = nn.Linear(hidden_dimension, 1)
        self.metric_head = nn.Linear(hidden_dimension, 6)
        self.metric_selection = metric_selection
        self.imitation_exponent = imitation_exponent
        self.refine_candidates = refine_candidates
        if refine_candidates:
            self.future_refinement_decoder = nn.TransformerDecoder(decoder_layer, 2)
            self.refinement_offset_head = nn.Linear(hidden_dimension, 24)
            nn.init.normal_(self.refinement_offset_head.weight, std=.0001)
            nn.init.zeros_(self.refinement_offset_head.bias)
            self.refined_score_decoder = nn.TransformerDecoder(decoder_layer, 2)
            self.refined_metric_head = nn.Linear(hidden_dimension, 6)
            self.temporal_safety_head = nn.Linear(hidden_dimension, 16)

    def plan_from_memory(self, memory, ego_status):
        scaled_candidates = self.trajectory_vocabulary / self.trajectory_vocabulary.new_tensor([20., 20., 3.141592653589793])
        queries = self.candidate_projection(scaled_candidates.flatten(1))[None]
        queries = queries + self.ego_projection(normalized_ego_status(ego_status))[:, None]
        decoded = self.candidate_decoder(queries, memory)
        imitation_logits = self.imitation_head(decoded).squeeze(-1)
        metric_logits = self.metric_head(decoded)
        estimated_pdms = compose_candidate_pdms(metric_logits.float().sigmoid())
        if self.metric_selection:
            selection_logits = estimated_pdms.clamp_min(1e-8).log() + self.imitation_exponent * imitation_logits.float().log_softmax(-1)
        else:
            selection_logits = imitation_logits.float()
        selected_indices = selection_logits.argmax(-1)
        output = {"trajectory": self.trajectory_vocabulary[selected_indices], "candidate_indices": selected_indices,
            "imitation_logits": imitation_logits, "metric_logits": metric_logits, "estimated_candidate_pdms": estimated_pdms}
        if self.refine_candidates:
            # Selection uses observed-model predictions only, never expert labels.
            imitation_shortlist = imitation_logits.topk(16, dim=-1).indices
            remaining_selection = selection_logits.detach().scatter(1, imitation_shortlist, float("-inf"))
            shortlist = torch.cat((remaining_selection.topk(16, dim=-1).indices, imitation_shortlist), -1)
            candidate_features = decoded.gather(1, shortlist[..., None].expand(-1, -1, decoded.shape[-1]))
            future_memory = memory[:, 4 * 64:]
            refined_features = self.future_refinement_decoder(candidate_features, future_memory)
            offsets = self.refinement_offset_head(refined_features).reshape(len(ego_status), 32, 8, 3).tanh()
            offsets = offsets * offsets.new_tensor([3., 3., .3])
            refined_trajectories = self.trajectory_vocabulary[shortlist] + offsets
            # Label generation is nondifferentiable. Keep the score regression
            # from moving trajectories merely to make scores easier to predict.
            score_queries = self.candidate_projection((refined_trajectories.detach() / offsets.new_tensor([20., 20., 3.141592653589793])).flatten(2))
            score_features = self.refined_score_decoder(score_queries + candidate_features, memory)
            refined_metric_logits = self.refined_metric_head(score_features)
            temporal_safety_logits = self.temporal_safety_head(score_features).reshape(len(ego_status), 32, 8, 2)
            refined_selection = compose_candidate_pdms(refined_metric_logits.float().sigmoid()).clamp_min(1e-8).log()
            refined_selection += self.imitation_exponent * imitation_logits.float().log_softmax(-1).gather(1, shortlist)
            refined_indices = refined_selection.argmax(-1)
            batch_indices = torch.arange(len(ego_status), device=ego_status.device)
            output.update(trajectory=refined_trajectories[batch_indices, refined_indices],
                candidate_indices=shortlist[batch_indices, refined_indices], refined_candidates=refined_trajectories,
                refinement_shortlist=shortlist, refined_metric_logits=refined_metric_logits,
                temporal_safety_logits=temporal_safety_logits,
                selected_refined_metric_logits=refined_metric_logits[batch_indices, refined_indices])
        return output


def compute_candidate_losses(predictions, target_trajectory, candidate_trajectories, teacher_metrics,
                             imitation_temperature=.5, metric_weight=1.):
    candidate_distance = (candidate_trajectories[None, ..., :2] - target_trajectory[:, None, :, :2]).abs().mean((-1, -2))
    imitation_targets = (-candidate_distance / imitation_temperature).softmax(-1)
    imitation_loss = -(imitation_targets * predictions["imitation_logits"].float().log_softmax(-1)).sum(-1).mean()
    valid_teacher = torch.isfinite(teacher_metrics[..., :6]).all((-1, -2))
    # Finite placeholder values are masked before reduction. Invalid scenes still
    # contribute imitation supervision and retain their normal training frequency.
    finite_targets = teacher_metrics[..., :6].nan_to_num(0.).float()
    metric_per_scene = F.binary_cross_entropy_with_logits(predictions["metric_logits"].float(), finite_targets, reduction="none").mean(1).sum(-1)
    metric_loss = (metric_per_scene * valid_teacher).sum() / len(target_trajectory)
    objective = imitation_loss + metric_weight * metric_loss
    return {"objective": objective, "imitation_loss": imitation_loss, "metric_loss": metric_loss,
            "teacher_coverage": valid_teacher.float().mean()}


def build_planning_model(checkpoint, specification, condition, project_root):
    if specification.get("planner_architecture") != "particle_candidate_metrics":
        return PlanningFineTunedLPWM(checkpoint)
    candidates = np.load(Path(project_root) / specification["teacher_directory"] / "trajectory_vocabulary.npy")
    return ParticleCandidatePlanner(checkpoint, candidates, metric_selection=not condition.startswith("imitation"),
        imitation_exponent=specification["inference_imitation_exponent"], refine_candidates="refinement" in condition)


def compute_refinement_losses(predictions, target_trajectory, ego_status, metric_targets, temporal_targets):
    candidates = predictions["refined_candidates"].float()
    distance = (candidates[..., :2] - target_trajectory[:, None, :, :2]).norm(dim=-1).mean(-1)
    closest = distance.detach().argmin(-1)
    chosen = candidates[torch.arange(len(candidates), device=candidates.device), closest]
    imitation = F.smooth_l1_loss(chosen[..., :2], target_trajectory[..., :2]) + .5 * (1 - (chosen[..., 2] - target_trajectory[..., 2]).cos()).mean()
    valid = torch.isfinite(metric_targets).all((-1, -2))
    metric_per_scene = F.binary_cross_entropy_with_logits(predictions["refined_metric_logits"].float(), metric_targets[..., :6].nan_to_num(0.), reduction="none").mean(1).sum(-1)
    temporal_per_scene = F.binary_cross_entropy_with_logits(predictions["temporal_safety_logits"].float(), temporal_targets.nan_to_num(0.), reduction="none").mean((1, 2)).sum(-1)
    metric_loss = (metric_per_scene * valid).mean()
    temporal_loss = (temporal_per_scene * valid).mean()
    positions = torch.cat((chosen.new_zeros((len(chosen), 1, 2)), chosen[..., :2]), 1)
    velocities = positions.diff(dim=1) / .5
    accelerations = torch.cat((ego_status[:, None, 4:6].float(), velocities), 1).diff(dim=1) / .5
    jerks = torch.cat((ego_status[:, None, 6:8].float(), accelerations), 1).diff(dim=1) / .5
    comfort = F.relu(accelerations.norm(dim=-1) / 4 - 1).square().mean() + F.relu(jerks.norm(dim=-1) / 8 - 1).square().mean()
    return {"objective": imitation + metric_loss + .5 * temporal_loss + .01 * comfort,
        "refinement_imitation": imitation, "refined_metric_loss": metric_loss, "temporal_safety_loss": temporal_loss, "comfort_proxy": comfort}


def uses_world_objective(condition):
    return condition.endswith("plus_world")
