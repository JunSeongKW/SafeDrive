"""Optional loss-side supervision, with no GT inputs to LPWM or the planner."""
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lpwm_planner import normalized_ego_status


class ParticleObjectStateHead(nn.Module):
    """Small shared head applied before any GT association; removed at inference."""
    def __init__(self, hidden_dimension=64, num_categories=3):
        super().__init__()
        self.readout = nn.Sequential(nn.Linear(19, hidden_dimension), nn.SiLU(),
            nn.Linear(hidden_dimension, 4 + num_categories))

    def forward(self, attributes, ego_status):
        # Foreground attributes only: position, scale, presence, depth, appearance.
        status = normalized_ego_status(ego_status)[:, None, None].expand(-1, 9, 64, -1)
        times = torch.arange(9, device=attributes.device, dtype=attributes.dtype) / 8
        times = times[None, :, None, None].expand(len(attributes), -1, 64, -1)
        outputs = self.readout(torch.cat((attributes[..., :10], status, times), -1))
        return {"object_state_predictions": outputs[..., :4], "object_category_logits": outputs[..., 4:]}


def detached_roi_particle_weights(attributes, boxes, minimum_sigma_pixels=2.):
    """Many-to-many Gaussian mass inside GT ROIs; positions use LPWM (y,x).

    Association cannot move particles or collapse their scale/presence. All
    particles participate, independent of presence; GT coordinates never enter
    the feature head. Degenerate support uses the nearest particle as a reported
    fallback, without silently excluding that object from the training loss.
    """
    with torch.no_grad():
        centers = (attributes[..., :2].float().detach().flip(-1) + 1) * 64 - .5
        sigma = (attributes[..., 2:4].float().detach().flip(-1) * 64).clamp_min(minimum_sigma_pixels)
        boxes = boxes.detach().float()
        lower = boxes[..., :2][..., None, :]
        upper = boxes[..., 2:][..., None, :]
        standardized_lower = (lower - centers[:, :, None]) / (sigma[:, :, None] * 2**.5)
        standardized_upper = (upper - centers[:, :, None]) / (sigma[:, :, None] * 2**.5)
        mass = (.5 * (standardized_upper.erf() - standardized_lower.erf())).clamp_min(0).prod(-1)
        support = mass.sum(-1, keepdim=True)
        nearest = ((centers[:, :, None] - ((lower + upper) * .5)) / sigma[:, :, None]).square().sum(-1).argmin(-1)
        fallback = F.one_hot(nearest, attributes.shape[-2]).float()
        weights = torch.where(support > 1e-10, mass / support.clamp_min(1e-10), fallback)
    return weights, (support.squeeze(-1) <= 1e-10)


def compute_object_auxiliary_losses(predictions, targets, configuration):
    attributes = torch.cat((predictions["observed_particle_attributes"][:, -1:],
        predictions["predicted_particle_attributes"]), 1)
    weights, fallback = detached_roi_particle_weights(attributes, targets["projected_boxes"],
        configuration["gaussian_minimum_sigma_pixels"])
    states = torch.einsum("btok,btkd->btod", weights, predictions["object_state_predictions"].float())
    logits = torch.einsum("btok,btkc->btoc", weights, predictions["object_category_logits"].float())
    valid = targets["state_valid"].detach()
    state_targets = targets["current_ego_states"].detach() / states.new_tensor(configuration["state_scales"])
    errors = F.smooth_l1_loss(states, state_targets, reduction="none").mean(-1)
    # Average objects within each scene and time, then average valid times/scenes.
    # Dense scenes and eight future steps do not overwhelm the current loss.
    per_time = (errors * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    time_valid = valid.any(-1)
    current_loss = (per_time[:, 0] * time_valid[:, 0]).sum() / time_valid[:, 0].sum().clamp_min(1)
    future_per_scene = (per_time[:, 1:] * time_valid[:, 1:]).sum(-1) / time_valid[:, 1:].sum(-1).clamp_min(1)
    future_valid = time_valid[:, 1:].any(-1)
    future_loss = (future_per_scene * future_valid).sum() / future_valid.sum().clamp_min(1)
    category_targets = targets["categories"].detach()[:, None].expand(-1, 9, -1)
    category_error = F.cross_entropy(logits.flatten(0, 2), category_targets.flatten(), reduction="none").reshape_as(valid)
    category_per_time = (category_error * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    category_loss = (category_per_time * time_valid).sum() / time_valid.sum().clamp_min(1)
    objective = (configuration["current_state_weight"] * current_loss
        + configuration["future_state_weight"] * future_loss + configuration["category_weight"] * category_loss)
    return {"objective": objective, "current_state_loss": current_loss, "future_state_loss": future_loss,
        "category_loss": category_loss, "association_fallback_fraction": (fallback * valid).sum() / valid.sum().clamp_min(1),
        "valid_current_objects": valid[:, 0].sum(), "valid_future_objects": valid[:, 1:].sum()}


class ObjectTargetCache:
    def __init__(self, root):
        root = Path(root)
        if not (root / "completion.json").exists():
            raise FileNotFoundError("Object target cache has not completed")
        self.offsets = np.load(root / "object_offsets.npy", mmap_mode="r")
        self.arrays = {name: np.load(root / (name + ".npy"), mmap_mode="r") for name in (
            "projected_boxes", "current_ego_states", "state_valid", "categories")}

    def select_batch(self, record_indices, device):
        maximum_objects = max(1, max(int(self.offsets[index + 1] - self.offsets[index]) for index in record_indices))
        values = {"projected_boxes": np.zeros((len(record_indices), 9, maximum_objects, 4), np.float32),
            "current_ego_states": np.zeros((len(record_indices), 9, maximum_objects, 4), np.float32),
            "state_valid": np.zeros((len(record_indices), 9, maximum_objects), bool),
            "categories": np.zeros((len(record_indices), maximum_objects), np.int64)}
        for batch_index, record_index in enumerate(record_indices):
            start, end = self.offsets[record_index:record_index + 2]
            count = end - start
            for name in ("projected_boxes", "current_ego_states"):
                values[name][batch_index, :, :count] = self.arrays[name][start:end].transpose(1, 0, 2)
            values["state_valid"][batch_index, :, :count] = self.arrays["state_valid"][start:end].T
            values["categories"][batch_index, :count] = self.arrays["categories"][start:end]
        return {name: torch.from_numpy(value).to(device) for name, value in values.items()}
