"""Offline target distributions and differentiable generator set matching.

Official quality labels are TRAINING-ONLY teacher information. No scorer reward
gradient or differentiability of NAVSIM is assumed. The original trajectory
head remains the generator: its copied native weights are the learned module.
"""
import math

import torch


def construct_teacher_distributions(actual_quality, predicted_quality, quality_temperature=0.1,
        reliability_temperature=0.15, maximum_reliability_mixture=0.8,
        maximum_target_quality_loss=0.01):
    quality_weights = (actual_quality / quality_temperature).softmax(dim=-1)
    scoring_error = (predicted_quality - actual_quality).abs()
    accurate_weights = quality_weights * (-scoring_error / reliability_temperature).exp()
    accurate_weights = accurate_weights / accurate_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    quality_mean = (quality_weights * actual_quality).sum(dim=-1)
    accurate_mean = (accurate_weights * actual_quality).sum(dim=-1)
    quality_drop = (quality_mean - accurate_mean).clamp_min(0)
    mixing = (maximum_target_quality_loss / quality_drop.clamp_min(1e-12)).clamp_max(maximum_reliability_mixture)
    compatible_weights = (1 - mixing[:, None]) * quality_weights + mixing[:, None] * accurate_weights
    return quality_weights, compatible_weights, mixing


def trajectory_distribution_coordinates(trajectories, heading_distance_scale_m=2.0):
    waypoint_count = trajectories.shape[-2]
    positions = trajectories[..., :2].flatten(-2)
    heading = trajectories[..., 2]
    angle_coordinates = torch.stack((heading.sin(), heading.cos()), dim=-1).flatten(-2)
    return torch.cat((positions, heading_distance_scale_m * angle_coordinates), dim=-1) / math.sqrt(waypoint_count)


def distribution_matching_losses(generated_trajectories, native_trajectories, target_weights, quality_weights,
        kernel_bandwidths_m=(1., 2., 4.), heading_distance_scale_m=2.0,
        coverage_temperature_m2=1.0):
    generated_coordinates = trajectory_distribution_coordinates(generated_trajectories, heading_distance_scale_m)
    native_coordinates = trajectory_distribution_coordinates(native_trajectories, heading_distance_scale_m)
    generated_pair_distance = torch.cdist(generated_coordinates, generated_coordinates).square()
    native_pair_distance = torch.cdist(native_coordinates, native_coordinates).square()
    cross_distance = torch.cdist(generated_coordinates, native_coordinates).square()
    generated_kernel = torch.stack([(-generated_pair_distance / (2 * bandwidth**2)).exp() for bandwidth in kernel_bandwidths_m]).mean(dim=0)
    native_kernel = torch.stack([(-native_pair_distance / (2 * bandwidth**2)).exp() for bandwidth in kernel_bandwidths_m]).mean(dim=0)
    cross_kernel = torch.stack([(-cross_distance / (2 * bandwidth**2)).exp() for bandwidth in kernel_bandwidths_m]).mean(dim=0)
    # Biased empirical squared MMD: uniform generated set vs weighted teacher.
    matching = generated_kernel.mean(dim=(-2, -1)) + torch.einsum("bi,bij,bj->b", target_weights, native_kernel, target_weights)
    matching = matching - 2 * (cross_kernel.mean(dim=1) * target_weights).sum(dim=-1)
    anchor = torch.nn.functional.smooth_l1_loss(generated_coordinates, native_coordinates, reduction="none").mean(dim=(-2, -1))
    # Coverage uses the SAME quality-only teacher for B/C. No oracle test labels.
    position_distances = torch.cdist(generated_trajectories[..., :2].flatten(-2) / math.sqrt(generated_trajectories.shape[-2]),
        native_trajectories[..., :2].flatten(-2) / math.sqrt(native_trajectories.shape[-2])).square()
    soft_nearest_distance = -coverage_temperature_m2 * (torch.logsumexp(-position_distances / coverage_temperature_m2, dim=1) - math.log(generated_trajectories.shape[1]))
    coverage = (quality_weights * soft_nearest_distance).sum(dim=-1)
    return {"distribution_mmd": matching.mean(), "native_anchor": anchor.mean(), "quality_coverage": coverage.mean()}
