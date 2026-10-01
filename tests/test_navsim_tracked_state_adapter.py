"""Geometry/association contracts; these fixtures are NOT perception validation."""

import math
import unittest

import numpy as np
import torch

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    annotation_states_in_current_ego_frame,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
    ego_global_planar_pose,
)
from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    SelectiveEntityFuturePredictionGraph,
)


def make_annotation_frame(
    timestamp, token, tracks=("first", "second"), ego_x=0.0, ego_heading=0.0
):
    boxes = np.array(
        [[3.0 + index, 0, 0, 4, 2, 1, 0] for index in range(len(tracks))]
    ).reshape(-1, 7)
    return {
        "timestamp": timestamp,
        "token": token,
        "ego2global_translation": [ego_x, 0, 0],
        "ego2global_rotation": [
            math.cos(ego_heading / 2),
            0,
            0,
            math.sin(ego_heading / 2),
        ],
        "lidar2ego_translation": [0, 0, 0],
        "lidar2ego_rotation": [1, 0, 0, 0],
        "ego_dynamic_state": [1, 0, 0, 0],
        "driving_command": [0, 1, 0, 0],
        "anns": {
            "gt_boxes": boxes,
            "gt_names": np.array(["vehicle"] * len(tracks)),
            "gt_velocity_3d": np.tile([1.0, 0, 0], (len(tracks), 1)),
            "track_tokens": list(tracks),
        },
    }


class NavsimTrackedStateAdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = TrackedStateAdapterConfig(
            num_history_frames=2, num_future_steps=2
        )
        self.history = [
            make_annotation_frame(0, "past"),
            make_annotation_frame(500000, "current"),
        ]
        self.future = [
            make_annotation_frame(1000000, "future1", ego_x=1),
            make_annotation_frame(1500000, "future2", ego_x=2),
        ]

    def build_inputs(self):
        return build_tracked_state_online_inputs(self.history, self.config)

    def build_targets(self, online_inputs=None):
        return build_tracked_state_training_targets(
            self.history[-1],
            self.future,
            online_inputs or self.build_inputs(),
            self.config,
        )

    def test_ego_motion_transform_includes_translation_and_velocity_rotation(self):
        current = self.history[-1]
        future = make_annotation_frame(
            1000000, "future", ego_x=10, ego_heading=math.pi / 2
        )
        states = annotation_states_in_current_ego_frame(future, current)
        np.testing.assert_allclose(states[0], [10, 3, 1, 0, 4, 2, 0, 1], atol=1e-12)

    def test_official_yaw_convention_with_nonzero_roll_pitch(self):
        frame = self.history[-1]
        frame["ego2global_rotation"] = [0.5, 0.1, -0.2, -0.8]
        quaternion_values = np.array(frame["ego2global_rotation"])
        scalar, axis_x, axis_y, axis_z = quaternion_values / np.linalg.norm(
            quaternion_values
        )
        expected_yaw = math.atan2(
            2 * (scalar * axis_z - axis_x * axis_y), 1 - 2 * (axis_y**2 + axis_z**2)
        )
        self.assertAlmostEqual(ego_global_planar_pose(frame)[2], expected_yaw)

    def test_future_association_survives_reordering(self):
        reference = self.build_targets()
        for future in self.future:
            for name in ("gt_boxes", "gt_names", "gt_velocity_3d"):
                future["anns"][name] = future["anns"][name][::-1].copy()
            future["anns"]["track_tokens"].reverse()
        torch.testing.assert_close(
            self.build_targets().future_state_targets, reference.future_state_targets
        )

    def test_disappearing_and_new_tracks_do_not_redefine_candidates(self):
        inputs = self.build_inputs()
        self.future[0] = make_annotation_frame(
            1000000, "future1", tracks=("first", "new")
        )
        targets = self.build_targets(inputs)
        self.assertEqual(inputs.current_track_tokens, ("first", "second"))
        self.assertTrue(
            torch.equal(
                targets.future_target_valid_mask,
                torch.tensor([[[True, True], [False, True]]]),
            )
        )

    def test_match_before_range_filter_retains_far_future_track(self):
        self.future[0]["anns"]["gt_boxes"][0, 0] = 200
        self.assertTrue(self.build_targets().future_target_valid_mask[0, 0, 0])

    def test_invalid_future_geometry_masked(self):
        self.future[0]["anns"]["gt_boxes"][0, 0] = np.nan
        targets = self.build_targets()
        self.assertFalse(targets.future_target_valid_mask[0, 0, 0])
        self.assertTrue(torch.isfinite(targets.future_state_targets).all())

    def test_future_mutation_cannot_change_online_inputs(self):
        reference = self.build_inputs()
        self.future[0]["anns"]["gt_boxes"][:] = np.nan
        actual = self.build_inputs()
        for name in (
            "entity_features",
            "scene_context",
            "ego_intent",
            "stable_entity_ids",
        ):
            self.assertTrue(
                torch.equal(getattr(reference, name), getattr(actual, name))
            )
        with self.assertRaises(ValueError):
            build_tracked_state_online_inputs(self.history + self.future, self.config)

    def test_current_annotation_permutation_retains_identity_and_features(self):
        reference = self.build_inputs()
        current = self.history[-1]
        for name in ("gt_boxes", "gt_names", "gt_velocity_3d"):
            current["anns"][name] = current["anns"][name][::-1].copy()
        current["anns"]["track_tokens"].reverse()
        actual = self.build_inputs()
        self.assertTrue(torch.equal(actual.entity_features, reference.entity_features))
        self.assertTrue(
            torch.equal(actual.stable_entity_ids, reference.stable_entity_ids)
        )

    def test_empty_current_candidate_set(self):
        self.history[-1] = make_annotation_frame(500000, "current", tracks=())
        self.assertEqual(self.build_inputs().entity_features.shape, (1, 0, 10))
        self.assertEqual(self.build_targets().future_state_targets.shape, (1, 0, 2, 6))

    def test_duplicate_track_and_timestamp_gap_rejected(self):
        self.history[-1]["anns"]["track_tokens"] = ["first", "first"]
        with self.assertRaises(ValueError):
            self.build_inputs()
        self.history[-1]["anns"]["track_tokens"] = ["first", "second"]
        self.future[-1]["timestamp"] += 500000
        with self.assertRaises(ValueError):
            self.build_targets()

    def test_lidar_extrinsic_translation_is_respected(self):
        frame = self.history[-1]
        frame["lidar2ego_translation"] = [2, 1, 0]
        states = annotation_states_in_current_ego_frame(frame, frame)
        np.testing.assert_allclose(states[0, :2], [5, 1])

    def test_history_gap_and_cross_log_future_rejected(self):
        self.history[0]["timestamp"] -= 500000
        with self.assertRaises(ValueError):
            self.build_inputs()
        self.history[0]["timestamp"] += 500000
        self.history[-1]["log_name"] = "current_log"
        for frame in self.future:
            frame["log_name"] = "other_log"
        with self.assertRaises(ValueError):
            self.build_targets()

    def test_multiple_ego_waypoints_forward_and_backward(self):
        inputs, targets = self.build_inputs(), self.build_targets()
        model = SelectiveEntityFuturePredictionGraph(
            entity_feature_dim=10,
            scene_context_dim=10,
            ego_intent_dim=4,
            future_latent_dim=6,
            num_future_steps=2,
            num_selected_entities=1,
            num_ego_plan_steps=2,
        )
        output = model(
            inputs.entity_features,
            inputs.scene_context,
            inputs.ego_intent,
            inputs.entity_valid_mask,
            inputs.stable_entity_ids,
        )
        self.assertEqual(output.ego_plan.shape, (1, 2, 3))
        losses = model.compute_training_losses(
            output,
            inputs.entity_features,
            inputs.scene_context,
            inputs.ego_intent,
            inputs.entity_valid_mask,
            targets.future_state_targets,
            targets.future_target_valid_mask,
            targets.ego_trajectory_target,
        )
        sum(losses.values()).backward()
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in model.entity_scorer.parameters()
            ),
            0,
        )
