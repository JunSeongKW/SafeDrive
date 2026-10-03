"""LPWM particle encoder with supervised metric futures and a trajectory planner.

No GT boxes, tracks, future images, or future ego poses are model inputs.
Official LPWM's RGB decoder/context/dynamics are not part of this task model.
"""
import contextlib
import io

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F

from .lpwm_bridge import load_official_lpwm


def normalized_ego_status(ego_status):
    return ego_status / ego_status.new_tensor([1, 1, 1, 1, 10, 10, 3, 3])


def particle_image_boxes(particle_centers, particle_scales):
    centers_xy = (particle_centers.flip(-1) + 1) / 2 - .5 / 128
    sizes_xy = particle_scales.sigmoid().flip(-1)
    return torch.cat((centers_xy - sizes_xy / 2, centers_xy + sizes_xy / 2), -1)


def match_current_objects(predicted_boxes, target_boxes):
    if not len(target_boxes):
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    predicted_centers = (predicted_boxes[:, :2] + predicted_boxes[:, 2:]) / 2
    target_centers = (target_boxes[:, :2] + target_boxes[:, 2:]) / 2
    cost = torch.cdist(predicted_centers.detach(), target_centers, p=1)
    cost = cost + .5 * torch.cdist((predicted_boxes[:, 2:] - predicted_boxes[:, :2]).detach(), target_boxes[:, 2:] - target_boxes[:, :2], p=1)
    return linear_sum_assignment(cost.cpu().numpy())


class IntentConditionedParticlePlanner(nn.Module):
    def __init__(self, checkpoint_path, condition, hidden_dimension=128):
        super().__init__()
        self.condition = condition
        self.encoder_intent_enabled = condition not in ("frozen_particles", "object_future_risk_no_intent", "ego_only")
        self.encoder_frozen = condition in ("frozen_particles", "ego_only")
        with contextlib.redirect_stdout(io.StringIO()):
            official_model, _ = load_official_lpwm("cpu", checkpoint_path)
        self.particle_encoder = official_model.encoder_module
        self.particle_encoder.use_ctx_enc = False
        self.particle_encoder.ctx_enc = None
        self.normalize_rgb = official_model.normalize_rgb
        del official_model
        self.encoder_intent_film = nn.Sequential(nn.Linear(8, 64), nn.SiLU(), nn.Linear(64, 64))
        nn.init.zeros_(self.encoder_intent_film[-1].weight)
        nn.init.zeros_(self.encoder_intent_film[-1].bias)
        self._active_ego_status = None
        self.particle_encoder.particle_enc.particle_attribute_enc.cnn.conv_in.register_forward_hook(self._condition_attribute_features)
        if self.encoder_frozen:
            self.particle_encoder.requires_grad_(False)
        if not self.encoder_intent_enabled:
            self.encoder_intent_film.requires_grad_(False)
        self.particle_projection = nn.Sequential(nn.Linear(14, hidden_dimension), nn.LayerNorm(hidden_dimension), nn.GELU())
        self.observation_time_embedding = nn.Parameter(torch.randn(2, hidden_dimension) * .02)
        self.particle_index_embedding = nn.Parameter(torch.randn(64, hidden_dimension) * .02)
        observation_layer = nn.TransformerEncoderLayer(hidden_dimension, 4, 256, dropout=0., batch_first=True, norm_first=True)
        self.observed_particle_attention = nn.TransformerEncoder(observation_layer, 2, enable_nested_tensor=False)
        self.object_state_head = nn.Sequential(nn.Linear(hidden_dimension, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 18))
        self.object_class_head = nn.Linear(hidden_dimension, 4)
        self.future_state_projection = nn.Sequential(nn.Linear(18, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, hidden_dimension))
        self.ego_projection = nn.Sequential(nn.Linear(8, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, hidden_dimension))
        self.trajectory_queries = nn.Parameter(torch.randn(8, hidden_dimension) * .02)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dimension, 4, 256, dropout=0., batch_first=True, norm_first=True)
        self.trajectory_decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.trajectory_head = nn.Linear(hidden_dimension, 3)
        nn.init.zeros_(self.trajectory_head.weight)
        nn.init.zeros_(self.trajectory_head.bias)
        self.ego_only_head = nn.Sequential(nn.Linear(8, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, hidden_dimension), nn.GELU(), nn.Linear(hidden_dimension, 24))
        nn.init.zeros_(self.ego_only_head[-1].weight)
        nn.init.zeros_(self.ego_only_head[-1].bias)
        if condition == "ego_only":
            for name, parameter in self.named_parameters():
                parameter.requires_grad_(name.startswith("ego_only_head."))
        else:
            self.ego_only_head.requires_grad_(False)

    def _condition_attribute_features(self, module, arguments, output):
        if not self.encoder_intent_enabled:
            return output
        conditioning = self.encoder_intent_film(normalized_ego_status(self._active_ego_status))
        scale, shift = conditioning.chunk(2, -1)
        repeats = output.shape[0] // conditioning.shape[0]
        assert repeats * conditioning.shape[0] == output.shape[0]
        scale = .1 * scale.tanh().repeat_interleave(repeats, 0)[..., None, None]
        shift = .1 * shift.tanh().repeat_interleave(repeats, 0)[..., None, None]
        return output * (1 + scale) + shift

    def encode_observations(self, observed_images, ego_status):
        assert observed_images.ndim == 5 and observed_images.shape[1:] == (2, 3, 128, 128)
        self._active_ego_status = ego_status
        try:
            encoder_input = observed_images * 2 - 1 if self.normalize_rgb else observed_images
            if self.encoder_frozen:
                self.particle_encoder.eval()
                with torch.no_grad():
                    encoded = self.particle_encoder(encoder_input, deterministic=True)
            else:
                encoded = self.particle_encoder(encoder_input, deterministic=True)
        finally:
            self._active_ego_status = None
        attributes = torch.cat((encoded["z"], encoded["z_scale"].sigmoid(), encoded["obj_on"],
                                encoded["z_depth"].tanh(), encoded["z_features"].tanh(),
                                encoded["z_bg_features"][:, :, None].expand(-1, -1, 64, -1).tanh()), -1)
        return attributes, particle_image_boxes(encoded["z"][:, -1], encoded["z_scale"][:, -1])

    def forward(self, observed_images, ego_status, intervention=None):
        normalized_status = normalized_ego_status(ego_status)
        future_times = torch.arange(1, 9, device=ego_status.device, dtype=ego_status.dtype) * .5
        kinematic_trajectory = ego_status.new_zeros((len(ego_status), 8, 3))
        kinematic_trajectory[..., :2] = ego_status[:, None, 4:6] * future_times[None, :, None]
        if self.condition == "ego_only":
            residual = self.ego_only_head(normalized_status).reshape(-1, 8, 3)
            return {"trajectory": kinematic_trajectory + residual * residual.new_tensor([10, 10, 1])}
        attributes, current_boxes = self.encode_observations(observed_images, ego_status)
        if intervention == "shuffle_scene_particles":
            attributes = attributes.roll(1, 0)
        particles = self.particle_projection(attributes)
        particles = particles + self.observation_time_embedding[None, :, None] + self.particle_index_embedding[None, None]
        observed_memory = self.observed_particle_attention(particles.flatten(1, 2))
        current_memory = observed_memory[:, -64:]
        object_states = self.object_state_head(current_memory)
        current_xy = object_states[..., :2] * 20
        future_xy = current_xy[:, :, None] + object_states[..., 2:].reshape(-1, 64, 8, 2) * 10
        if intervention == "zero_predicted_futures":
            state_for_planner = torch.cat((object_states[..., :2], torch.zeros_like(object_states[..., 2:])), -1)
        else:
            state_for_planner = object_states
        future_memory = current_memory + self.future_state_projection(state_for_planner)
        planner_memory = torch.cat((observed_memory, future_memory), 1)
        queries = self.trajectory_queries[None] + self.ego_projection(normalized_status)[:, None]
        decoded = self.trajectory_decoder(queries, planner_memory)
        residual = self.trajectory_head(decoded) * decoded.new_tensor([10, 10, 1])
        return {"trajectory": kinematic_trajectory + residual, "particle_attributes": attributes,
                "current_boxes": current_boxes, "current_xy": current_xy, "future_xy": future_xy,
                "object_states": object_states, "object_class_logits": self.object_class_head(current_memory)}


def compute_planning_objectives(predictions, supervision, condition, object_loss_weight=.5):
    target_trajectory = supervision["ego_trajectory_target"]
    planning_xy = F.smooth_l1_loss(predictions["trajectory"][..., :2], target_trajectory[..., :2])
    heading_error = predictions["trajectory"][..., 2] - target_trajectory[..., 2]
    planning_loss = planning_xy + .5 * (1 - heading_error.cos()).mean()
    result = {"planning": planning_loss, "total": planning_loss}
    if not condition.startswith("object_future"):
        return result
    geometry_losses, current_losses, future_losses, class_losses = [], [], [], []
    for batch_index in range(len(target_trajectory)):
        valid = supervision["object_valid"][batch_index]
        target_boxes = supervision["object_boxes"][batch_index, valid]
        particle_indices, object_indices = match_current_objects(predictions["current_boxes"][batch_index], target_boxes)
        category_labels = torch.zeros(64, dtype=torch.long, device=target_trajectory.device)
        if len(object_indices):
            categories = supervision["object_categories"][batch_index, valid][object_indices]
            category_labels[particle_indices] = categories
            weights = supervision["object_risk_weights"][batch_index, valid][object_indices] if "risk" in condition else torch.ones(len(object_indices), device=target_trajectory.device)
            weights = weights / weights.mean()
            matched_boxes = predictions["current_boxes"][batch_index, particle_indices]
            geometry_losses.append(((matched_boxes - target_boxes[object_indices]).abs().mean(-1) * weights).mean())
            current_target = supervision["object_current_xy"][batch_index, valid][object_indices]
            current_losses.append((F.smooth_l1_loss(predictions["current_xy"][batch_index, particle_indices] / 20, current_target / 20, reduction="none").mean(-1) * weights).mean())
            future_target = supervision["object_future_xy"][batch_index, valid][object_indices]
            future_valid = supervision["object_future_valid"][batch_index, valid][object_indices]
            future_errors = F.smooth_l1_loss(predictions["future_xy"][batch_index, particle_indices] / 20, future_target / 20, reduction="none").mean(-1)
            displacement_error = F.smooth_l1_loss((predictions["future_xy"][batch_index, particle_indices] - predictions["current_xy"][batch_index, particle_indices, None]) / 10,
                                                 (future_target - current_target[:, None]) / 10, reduction="none").mean(-1)
            future_weights = weights[:, None] * future_valid
            future_losses.append(((future_errors + displacement_error) * future_weights).sum() / future_weights.sum().clamp_min(1))
        class_losses.append(F.cross_entropy(predictions["object_class_logits"][batch_index], category_labels, weight=target_trajectory.new_tensor([.1, 1, 1, 1])))
    zero = planning_loss * 0
    result.update({"geometry": torch.stack(geometry_losses).mean() if geometry_losses else zero,
                   "current_state": torch.stack(current_losses).mean() if current_losses else zero,
                   "future_state": torch.stack(future_losses).mean() if future_losses else zero,
                   "object_class": torch.stack(class_losses).mean()})
    result["object_supervision"] = 2 * result["geometry"] + result["current_state"] + result["future_state"] + .2 * result["object_class"]
    result["total"] = planning_loss + object_loss_weight * result["object_supervision"]
    return result
