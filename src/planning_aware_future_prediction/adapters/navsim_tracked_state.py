"""Privileged GT-state diagnostic, NOT a perception adapter or visual JEPA.

Raw NAVSIM annotations are associated by track token before any future filtering.
Forward inputs are built exclusively from history/current frames. Future frames
are handled by a separate target builder. Shared dataset files are never written.
"""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class TrackedStateAdapterConfig:
    num_history_frames: int = 4
    num_future_steps: int = 8
    max_candidate_entities: int = 32
    entity_radius_meters: float = 40.0
    position_scale_meters: float = 40.0
    velocity_scale_meters_per_second: float = 10.0
    box_size_scale_meters: float = 10.0
    frame_interval_seconds: float = 0.5
    timestamp_tolerance_seconds: float = 0.05

    def __post_init__(self):
        for field_name in (
            "num_history_frames",
            "num_future_steps",
            "max_candidate_entities",
        ):
            if getattr(self, field_name) < 1:
                raise ValueError(f"{field_name} must be positive")
        for field_name in (
            "entity_radius_meters",
            "position_scale_meters",
            "velocity_scale_meters_per_second",
            "box_size_scale_meters",
            "frame_interval_seconds",
            "timestamp_tolerance_seconds",
        ):
            if (
                not math.isfinite(getattr(self, field_name))
                or getattr(self, field_name) <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")


@dataclass
class TrackedStateOnlineInputs:
    entity_features: Tensor  # [1, candidate_count, 10]; current GT geometry + class
    scene_context: Tensor  # [1, history_count*3+4]; ego history + dynamics only
    ego_intent: Tensor  # [1, 4]; current driving command
    entity_valid_mask: Tensor  # [1, candidate_count]
    stable_entity_ids: Tensor  # [1, candidate_count]; tie resolution, not scorer input
    current_track_tokens: tuple[str, ...]  # diagnostic/target association only
    current_frame_token: str


@dataclass
class TrackedStateTrainingTargets:
    future_state_targets: (
        Tensor  # [1, candidate_count, future_steps, 6], NOT visual latent
    )
    future_target_valid_mask: Tensor  # [1, candidate_count, future_steps]
    ego_trajectory_target: Tensor  # [1, future_steps, 3], current rear-axle frame
    future_time_offsets_seconds: Tensor  # [future_steps]; actual timestamps


def quaternion_wxyz_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quaternion_values = np.asarray(quaternion, dtype=np.float64)
    if quaternion_values.shape != (4,) or not np.isfinite(quaternion_values).all():
        raise ValueError("expected finite wxyz quaternion")
    quaternion_norm = np.linalg.norm(quaternion_values)
    if quaternion_norm <= 0:
        raise ValueError("zero quaternion")
    scalar, axis_x, axis_y, axis_z = quaternion_values / quaternion_norm
    return np.array(
        [
            [
                1 - 2 * (axis_y**2 + axis_z**2),
                2 * (axis_x * axis_y - axis_z * scalar),
                2 * (axis_x * axis_z + axis_y * scalar),
            ],
            [
                2 * (axis_x * axis_y + axis_z * scalar),
                1 - 2 * (axis_x**2 + axis_z**2),
                2 * (axis_y * axis_z - axis_x * scalar),
            ],
            [
                2 * (axis_x * axis_z - axis_y * scalar),
                2 * (axis_y * axis_z + axis_x * scalar),
                1 - 2 * (axis_x**2 + axis_y**2),
            ],
        ]
    )


def ego_global_planar_pose(frame: Mapping) -> np.ndarray:
    rotation = quaternion_wxyz_rotation_matrix(frame["ego2global_rotation"])
    translation = np.asarray(frame["ego2global_translation"], dtype=np.float64)
    if translation.shape != (3,) or not np.isfinite(translation).all():
        raise ValueError("expected finite ego translation")
    # Match Scene._build_ego_status / pyquaternion.yaw_pitch_roll exactly.
    # atan2(R[1,0], R[0,0]) uses another Euler convention when roll/pitch != 0.
    return np.array(
        [translation[0], translation[1], math.atan2(-rotation[0, 1], rotation[0, 0])]
    )


def relative_ego_planar_pose(frame: Mapping, current_frame: Mapping) -> np.ndarray:
    frame_pose = ego_global_planar_pose(frame)
    origin_pose = ego_global_planar_pose(current_frame)
    delta_position = frame_pose[:2] - origin_pose[:2]
    cosine, sine = math.cos(origin_pose[2]), math.sin(origin_pose[2])
    relative_position = np.array([[cosine, sine], [-sine, cosine]]) @ delta_position
    delta_heading = math.atan2(
        math.sin(frame_pose[2] - origin_pose[2]),
        math.cos(frame_pose[2] - origin_pose[2]),
    )
    return np.array([*relative_position, delta_heading])


def annotation_states_in_current_ego_frame(
    frame: Mapping, current_frame: Mapping
) -> np.ndarray:
    """Lidar box/velocity -> global -> current planar ego frame; no ego-motion drift."""
    boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64)
    velocity = np.asarray(frame["anns"]["gt_velocity_3d"], dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7 or velocity.shape != (len(boxes), 3):
        raise ValueError("unsupported NAVSIM annotation schema")
    lidar_to_ego = quaternion_wxyz_rotation_matrix(frame["lidar2ego_rotation"])
    ego_to_global = quaternion_wxyz_rotation_matrix(frame["ego2global_rotation"])
    current_pose = ego_global_planar_pose(current_frame)
    cosine, sine = math.cos(current_pose[2]), math.sin(current_pose[2])
    global_to_current = np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    lidar_to_current = global_to_current @ ego_to_global @ lidar_to_ego
    global_translation = ego_to_global @ np.asarray(
        frame["lidar2ego_translation"]
    ) + np.asarray(frame["ego2global_translation"])
    translation_in_current = global_to_current @ (
        global_translation
        - np.array([current_pose[0], current_pose[1], global_translation[2]])
    )
    position_in_current = boxes[:, :3] @ lidar_to_current.T + translation_in_current
    velocity_in_current = velocity @ lidar_to_current.T
    heading_vectors = np.stack(
        (np.cos(boxes[:, 6]), np.sin(boxes[:, 6]), np.zeros(len(boxes))), axis=-1
    )
    heading_in_current = heading_vectors @ lidar_to_current.T
    headings = np.arctan2(heading_in_current[:, 1], heading_in_current[:, 0])
    return np.column_stack(
        (
            position_in_current[:, :2],
            np.sin(headings),
            np.cos(headings),
            boxes[:, 3:5],
            velocity_in_current[:, :2],
        )
    )


def annotation_track_lookup(frame: Mapping) -> dict[str, int]:
    tokens = [str(token) for token in frame["anns"]["track_tokens"]]
    if any(not token for token in tokens) or len(tokens) != len(set(tokens)):
        raise ValueError("empty/duplicate track token in raw annotations")
    if len(tokens) != len(frame["anns"]["gt_boxes"]) or len(tokens) != len(
        frame["anns"]["gt_names"]
    ):
        raise ValueError("annotation lengths differ")
    return {token: index for index, token in enumerate(tokens)}


def stable_track_identity(track_token: str) -> int:
    # Deterministic across runs; do not use Python's process-randomized hash().
    return int.from_bytes(hashlib.sha256(track_token.encode()).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def build_tracked_state_online_inputs(
    history_frames: Sequence[Mapping],
    config: TrackedStateAdapterConfig,
) -> TrackedStateOnlineInputs:
    if len(history_frames) != config.num_history_frames:
        raise ValueError(
            "online builder accepts exactly the configured history, not future frames"
        )
    history_timestamps = np.asarray(
        [frame["timestamp"] for frame in history_frames], dtype=np.float64
    )
    if np.any(
        np.abs(np.diff(history_timestamps) / 1e6 - config.frame_interval_seconds)
        > config.timestamp_tolerance_seconds
    ):
        raise ValueError("history timestamps are not the expected contiguous cadence")
    history_log_names = {
        str(frame["log_name"]) for frame in history_frames if "log_name" in frame
    }
    if len(history_log_names) > 1:
        raise ValueError("history cannot span different logs")
    current_frame = history_frames[-1]
    lookup = annotation_track_lookup(current_frame)
    states = annotation_states_in_current_ego_frame(current_frame, current_frame)
    names = current_frame["anns"]["gt_names"]
    candidates = [
        (token, index)
        for token, index in lookup.items()
        if names[index] in ("vehicle", "pedestrian")
        and np.isfinite(states[index]).all()
        and np.linalg.norm(states[index, :2]) <= config.entity_radius_meters
    ]
    candidates.sort(
        key=lambda candidate: (
            np.linalg.norm(states[candidate[1], :2]),
            stable_track_identity(candidate[0]),
        )
    )
    candidates = candidates[: config.max_candidate_entities]
    candidate_count = len(candidates)
    features = np.zeros((candidate_count, 10), dtype=np.float32)
    for entity_index, (_, annotation_index) in enumerate(candidates):
        features[entity_index, :8] = states[annotation_index]
        features[entity_index, 8:] = (
            names[annotation_index] == "vehicle",
            names[annotation_index] == "pedestrian",
        )
    features[:, :2] /= config.position_scale_meters
    features[:, 4:6] /= config.box_size_scale_meters
    features[:, 6:8] /= config.velocity_scale_meters_per_second
    history_poses = np.stack(
        [relative_ego_planar_pose(frame, current_frame) for frame in history_frames]
    )
    history_poses[:, :2] /= config.position_scale_meters
    dynamics = np.asarray(current_frame["ego_dynamic_state"], dtype=np.float32)
    command = np.asarray(current_frame["driving_command"], dtype=np.float32)
    if dynamics.shape != (4,) or command.shape != (4,):
        raise ValueError("expected ego dynamics[4] and command[4]")
    context = np.concatenate((history_poses.flatten(), dynamics))
    if not np.isfinite(context).all() or not np.isfinite(command).all():
        raise ValueError("non-finite online ego inputs")
    identities = [stable_track_identity(token) for token, _ in candidates]
    if len(set(identities)) != candidate_count:
        raise ValueError("stable track identity hash collision")
    return TrackedStateOnlineInputs(
        torch.from_numpy(features)[None],
        torch.tensor(context, dtype=torch.float32)[None],
        torch.from_numpy(command.copy())[None],
        torch.ones(1, candidate_count, dtype=torch.bool),
        torch.tensor(identities, dtype=torch.int64)[None],
        tuple(token for token, _ in candidates),
        str(current_frame["token"]),
    )


def build_tracked_state_training_targets(
    current_frame: Mapping,
    future_frames: Sequence[Mapping],
    online_inputs: TrackedStateOnlineInputs,
    config: TrackedStateAdapterConfig,
) -> TrackedStateTrainingTargets:
    if str(current_frame["token"]) != online_inputs.current_frame_token:
        raise ValueError("online inputs and current frame differ")
    if len(future_frames) != config.num_future_steps:
        raise ValueError("wrong future frame count")
    if "log_name" in current_frame and any(
        frame.get("log_name") != current_frame["log_name"] for frame in future_frames
    ):
        raise ValueError("future targets cannot span different logs")
    offsets = np.array(
        [
            (frame["timestamp"] - current_frame["timestamp"]) / 1e6
            for frame in future_frames
        ]
    )
    intervals = np.diff(np.concatenate(([0.0], offsets)))
    if not np.isfinite(offsets).all() or np.any(
        np.abs(intervals - config.frame_interval_seconds)
        > config.timestamp_tolerance_seconds
    ):
        raise ValueError("future timestamps are not the expected contiguous cadence")
    candidate_count = len(online_inputs.current_track_tokens)
    targets = np.zeros((candidate_count, config.num_future_steps, 6), dtype=np.float32)
    valid = np.zeros((candidate_count, config.num_future_steps), dtype=bool)
    for future_step, future_frame in enumerate(future_frames):
        lookup = annotation_track_lookup(future_frame)
        states = annotation_states_in_current_ego_frame(future_frame, current_frame)
        for entity_index, track_token in enumerate(online_inputs.current_track_tokens):
            if track_token not in lookup:
                continue
            state = states[lookup[track_token]]
            if not np.isfinite(state).all():
                continue
            targets[entity_index, future_step] = state[[0, 1, 2, 3, 6, 7]]
            valid[entity_index, future_step] = True
    targets[..., :2] /= config.position_scale_meters
    targets[..., 4:6] /= config.velocity_scale_meters_per_second
    ego_targets = np.stack(
        [relative_ego_planar_pose(frame, current_frame) for frame in future_frames]
    )
    return TrackedStateTrainingTargets(
        torch.from_numpy(targets)[None],
        torch.from_numpy(valid)[None],
        torch.tensor(ego_targets, dtype=torch.float32)[None],
        torch.tensor(offsets, dtype=torch.float64),
    )
