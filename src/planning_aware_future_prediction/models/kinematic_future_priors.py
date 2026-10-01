"""Unit-explicit priors; use only CURRENT inputs in a fixed rear-axle frame."""

import torch
from torch import Tensor


def future_time_offsets(num_future_steps: int, reference: Tensor) -> Tensor:
    return (
        torch.arange(
            1, num_future_steps + 1, device=reference.device, dtype=reference.dtype
        )
        * 0.5
    )


def predict_ego_kinematic_trajectory(
    current_ego_status: Tensor, num_future_steps=8, motion_model="constant_acceleration"
) -> Tensor:
    """Status=[command4,vx,vy,ax,ay], m/s and m/s²; origin is current ego.

    Fixed vector velocity/acceleration, no future GT or inferred turning rate.
    Heading prior zero is a declared limitation, not a curvature prediction.
    """
    if current_ego_status.shape[-1] != 8:
        raise ValueError("expected command4 + velocity2 + acceleration2")
    offsets = future_time_offsets(num_future_steps, current_ego_status)
    xy_positions = current_ego_status[..., None, 4:6] * offsets[:, None]
    if motion_model == "constant_acceleration":
        xy_positions = (
            xy_positions
            + 0.5 * current_ego_status[..., None, 6:8] * offsets[:, None].square()
        )
    elif motion_model == "stationary":
        xy_positions = torch.zeros_like(xy_positions)
    elif motion_model != "constant_velocity":
        raise ValueError("unknown ego motion model")
    return torch.cat((xy_positions, torch.zeros_like(xy_positions[..., :1])), dim=-1)


def predict_entity_constant_velocity_state(
    selected_current_entity_features: Tensor,
    num_future_steps=8,
    extrapolate_position=True,
) -> Tensor:
    """Return scaled [x/40,y/40,sin(yaw),cos(yaw),vx/10,vy/10].

    Current geometry tail: [x/40,y/40,sin,cos,width/10,length/10,vx/10,vy/10,class2].
    Positions and velocities remain in the CURRENT ego frame at every horizon;
    constant heading/velocity, no rotation into a future ego frame.
    """
    current_state = selected_current_entity_features[..., -10:][..., [0, 1, 2, 3, 6, 7]]
    predicted_state = (
        current_state[..., None, :]
        .expand(*current_state.shape[:-1], num_future_steps, 6)
        .clone()
    )
    if extrapolate_position:
        offsets = future_time_offsets(num_future_steps, current_state)
        predicted_state[..., :2] += (
            current_state[..., None, 4:6] * (10.0 / 40.0) * offsets[:, None]
        )
    return predicted_state
