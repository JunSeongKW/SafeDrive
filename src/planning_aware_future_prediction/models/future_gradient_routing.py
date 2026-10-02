"""Controlled optimization interventions, not a replacement planning model."""

from contextlib import contextmanager

import torch


@contextmanager
def use_current_feature_prediction(predictor):
    """Planning-side persistence control; no future target is read.

    The predictor is still executed and its output replaced, so this diagnostic
    implementation does NOT claim predictor-compute savings.
    """
    handle = predictor.register_forward_hook(
        lambda module, inputs, output: inputs[0][:, :, None].expand_as(output)
    )
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def freeze_parameter_gradients(module):
    """Treat parameters as constants while retaining gradients to module inputs.

    This is deliberately NOT no_grad/output.detach. Auxiliary prediction is
    evaluated after the context exits, with the original parameter flags restored.
    """
    parameters = list(module.parameters())
    original_flags = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, original_flags):
            parameter.requires_grad_(enabled)


def project_auxiliary_gradient(planning_gradients, auxiliary_gradients, parameters):
    """One-sided PCGrad-inspired projection; never rotate the planning gradient.

    Projection is global across predictor parameters, not independently per layer.
    The auxiliary list must already include its configured loss weight. This is
    not the symmetric, randomized multi-task PCGrad algorithm from the paper.
    """
    planning = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(planning_gradients, parameters)
    ]
    auxiliary = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for gradient, parameter in zip(auxiliary_gradients, parameters)
    ]
    inner_product = sum(
        (left * right).sum() for left, right in zip(planning, auxiliary)
    )
    planning_squared_norm = sum(gradient.square().sum() for gradient in planning)
    coefficient = torch.minimum(inner_product, torch.zeros_like(inner_product)) / (
        planning_squared_norm.clamp_min(1e-20)
    )
    combined = [
        plan + auxiliary_part - coefficient * plan
        for plan, auxiliary_part in zip(planning, auxiliary)
    ]
    return combined, {
        "conflicting": bool(inner_product < 0),
        "projection_coefficient": float(coefficient),
    }


def uniform_xy_ade_with_heading(prediction, target, heading_weight=0.1):
    """Diagnostic objective: meters of XY ADE + weighted circular radians.

    Not official Drive-JEPA loss or a differentiable PDMS. Evaluation still
    records official IL and scene-macro ADE for every condition.
    """
    xy_error = (prediction[..., :2] - target[..., :2]).norm(dim=-1).mean()
    heading_difference = prediction[..., 2] - target[..., 2]
    heading_error = (
        torch.atan2(heading_difference.sin(), heading_difference.cos()).abs().mean()
    )
    return xy_error + heading_weight * heading_error
