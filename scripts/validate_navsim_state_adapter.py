"""Real NAVSIM mini GT-state plumbing diagnostic; no perception/JEPA claims.

Read only a trusted local NAVSIM pickle, two small windows, and their front-image
headers. No weight download, simulator metric cache, GPU, or dataset writes.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
)
from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    SelectiveEntityFuturePredictionGraph,
    compute_parameter_gradient_norm,
)


def file_sha256(file_path):
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure_state_gradient_contracts(model, online_inputs, training_targets):
    report = {}
    reference_output = model(
        online_inputs.entity_features,
        online_inputs.scene_context,
        online_inputs.ego_intent,
        online_inputs.entity_valid_mask,
        online_inputs.stable_entity_ids,
    )
    for contract_name, selection_mode, detach_future, loss_key in (
        ("planning", "straight_through", False, "planning_loss"),
        ("state_auxiliary", "straight_through", False, "future_latent_prediction_loss"),
        ("selection_detach", "detached_selection", False, "planning_loss"),
        ("future_detach", "straight_through", True, "planning_loss"),
    ):
        prediction = model(
            online_inputs.entity_features,
            online_inputs.scene_context,
            online_inputs.ego_intent,
            online_inputs.entity_valid_mask,
            online_inputs.stable_entity_ids,
            selection_mode,
            detach_future,
        )
        losses = model.compute_training_losses(
            prediction,
            online_inputs.entity_features,
            online_inputs.scene_context,
            online_inputs.ego_intent,
            online_inputs.entity_valid_mask,
            training_targets.future_state_targets,
            training_targets.future_target_valid_mask,
            training_targets.ego_trajectory_target,
        )
        norms = {
            "selector": compute_parameter_gradient_norm(
                losses[loss_key], model.entity_scorer
            ),
            "predictor": compute_parameter_gradient_norm(
                losses[loss_key], model.future_predictor
            ),
            "planner": compute_parameter_gradient_norm(
                losses[loss_key], model.ego_planner
            ),
        }
        if contract_name == "planning":
            assert all(value > 0 for value in norms.values())
        elif contract_name == "state_auxiliary":
            assert norms["selector"] == norms["planner"] == 0 and norms["predictor"] > 0
        elif contract_name == "selection_detach":
            assert (
                norms["selector"] == 0
                and norms["predictor"] > 0
                and norms["planner"] > 0
            )
            assert torch.equal(prediction.ego_plan, reference_output.ego_plan)
        else:
            assert norms["selector"] == norms["predictor"] == 0 and norms["planner"] > 0
            assert torch.equal(prediction.ego_plan, reference_output.ego_plan)
        report[contract_name] = norms

    predicted_without_state = model.decode_ego_plan(
        online_inputs.scene_context,
        online_inputs.ego_intent,
        torch.zeros_like(reference_output.predicted_future_latents),
        reference_output.entity_selection.selected_entity_valid_mask,
    )
    perturbation = (
        (reference_output.ego_plan - predicted_without_state).abs().max().item()
    )
    assert perturbation > 0

    # Replacing only labels must not change fixed-weight forward outputs.
    model.compute_training_losses(
        reference_output,
        online_inputs.entity_features,
        online_inputs.scene_context,
        online_inputs.ego_intent,
        online_inputs.entity_valid_mask,
        training_targets.future_state_targets + 100,
        training_targets.future_target_valid_mask,
        training_targets.ego_trajectory_target,
    )
    repeated_output = model(
        online_inputs.entity_features,
        online_inputs.scene_context,
        online_inputs.ego_intent,
        online_inputs.entity_valid_mask,
        online_inputs.stable_entity_ids,
    )
    assert torch.equal(reference_output.ego_plan, repeated_output.ego_plan)
    assert torch.equal(
        reference_output.entity_selection.selected_entity_indices,
        repeated_output.entity_selection.selected_entity_indices,
    )
    return report, perturbation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-log-path", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/adapter_diagnostics/navsim_state_adapter.json",
    )
    arguments = parser.parse_args()
    output_path = arguments.output.resolve()
    if not output_path.is_relative_to(PROJECT_ROOT / "outputs"):
        parser.error(
            "diagnostic output must be inside the project's outputs/ directory"
        )
    dataset_root = arguments.dataset_root.resolve()
    log_path = arguments.trusted_log_path or next(
        iter(sorted((dataset_root / "navsim_logs/mini").glob("*.pkl")))
    )
    log_path = log_path.resolve()
    if log_path.parent != (dataset_root / "navsim_logs/mini").resolve():
        parser.error("this limited diagnostic accepts only trusted NAVSIM mini logs")
    started_seconds = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(19)
    log_digest_before = file_sha256(log_path)
    with log_path.open("rb") as stream:
        frames = pickle.load(stream)  # trusted research-lab NAVSIM data ONLY
    config = TrackedStateAdapterConfig()
    scene_results = []
    for window_start_index in (0, 12):
        history_frames = frames[
            window_start_index : window_start_index + config.num_history_frames
        ]
        current_index = window_start_index + config.num_history_frames - 1
        future_frames = frames[
            current_index + 1 : current_index + 1 + config.num_future_steps
        ]
        online_inputs = build_tracked_state_online_inputs(history_frames, config)
        targets = build_tracked_state_training_targets(
            history_frames[-1], future_frames, online_inputs, config
        )
        # Check our SE(2) ego label conversion against the existing quaternion package.
        import numpy as np
        from pyquaternion import Quaternion

        current = history_frames[-1]
        reference_yaw = Quaternion(current["ego2global_rotation"]).yaw_pitch_roll[0]
        reference_rotation = np.array(
            [
                [np.cos(reference_yaw), np.sin(reference_yaw)],
                [-np.sin(reference_yaw), np.cos(reference_yaw)],
            ]
        )
        reference_positions = np.stack(
            [
                reference_rotation
                @ (
                    np.asarray(frame["ego2global_translation"][:2])
                    - np.asarray(current["ego2global_translation"][:2])
                )
                for frame in future_frames
            ]
        )
        np.testing.assert_allclose(
            targets.ego_trajectory_target[0, :, :2].numpy(),
            reference_positions,
            rtol=1e-6,
            atol=1e-6,
        )

        candidate_count = online_inputs.entity_features.shape[1]
        selected_budget = 4
        assert candidate_count > selected_budget  # subset choice must be nontrivial
        model = SelectiveEntityFuturePredictionGraph(
            entity_feature_dim=10,
            scene_context_dim=online_inputs.scene_context.shape[-1],
            ego_intent_dim=4,
            future_latent_dim=6,
            num_future_steps=config.num_future_steps,
            num_selected_entities=selected_budget,
            hidden_feature_dim=32,
            num_ego_plan_steps=config.num_future_steps,
        ).eval()
        prediction = model(
            online_inputs.entity_features,
            online_inputs.scene_context,
            online_inputs.ego_intent,
            online_inputs.entity_valid_mask,
            online_inputs.stable_entity_ids,
        )
        contracts, perturbation = measure_state_gradient_contracts(
            model, online_inputs, targets
        )
        losses = model.compute_training_losses(
            prediction,
            online_inputs.entity_features,
            online_inputs.scene_context,
            online_inputs.ego_intent,
            online_inputs.entity_valid_mask,
            targets.future_state_targets,
            targets.future_target_valid_mask,
            targets.ego_trajectory_target,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        sum(losses.values()).backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        optimizer.step()  # One engineering update, NOT joint-learning stability evidence.
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        image_path = (
            dataset_root / "sensor_blobs/mini" / current["cams"]["CAM_F0"]["data_path"]
        )
        image_digest_before = file_sha256(image_path)
        with Image.open(image_path) as image:
            image_size = list(image.size)
            image.verify()
        assert file_sha256(image_path) == image_digest_before
        scene_results.append(
            {
                "scene_token": str(current["scene_token"]),
                "current_frame_token": str(current["token"]),
                "window_start_index": window_start_index,
                "candidate_count": candidate_count,
                "selected_entity_budget": selected_budget,
                "online_input_shapes": {
                    name: list(getattr(online_inputs, name).shape)
                    for name in (
                        "entity_features",
                        "scene_context",
                        "ego_intent",
                        "entity_valid_mask",
                        "stable_entity_ids",
                    )
                },
                "future_state_target_shape": list(targets.future_state_targets.shape),
                "predicted_future_state_shape": list(
                    prediction.predicted_future_latents.shape
                ),
                "ego_plan_shape": list(prediction.ego_plan.shape),
                "valid_future_target_count": targets.future_target_valid_mask.sum().item(),
                "invalid_future_target_count": (~targets.future_target_valid_mask)
                .sum()
                .item(),
                "future_time_offsets_seconds": targets.future_time_offsets_seconds.tolist(),
                "gradient_norms": contracts,
                "state_removal_max_output_change": perturbation,
                "joint_backward_and_single_optimizer_step_finite": True,
                "losses_before_update": {
                    "planning_mse": losses["planning_loss"].item(),
                    "future_state_prediction_mse": losses[
                        "future_latent_prediction_loss"
                    ].item(),
                },
                "front_image_size": image_size,
                "front_image_read_scope": "header_and_integrity_only_not_encoder_forward",
                "front_image_sha256": image_digest_before,
                "model_parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
            }
        )
    assert file_sha256(log_path) == log_digest_before
    source_paths = [
        Path(__file__),
        PROJECT_ROOT
        / "src/planning_aware_future_prediction/adapters/navsim_tracked_state.py",
        PROJECT_ROOT
        / "src/planning_aware_future_prediction/models/selective_entity_future_prediction.py",
    ]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in source_paths
        },
        "diagnostic_scope": "real_NAVSIM_GT_state_adapter_not_visual_JEPA_or_official_baseline",
        "privileged_current_annotations": True,
        "future_inputs_to_selector": False,
        "official_baseline_reproduced": False,
        "visual_target_encoder_tested": False,
        "shared_input_hashes_unchanged": True,
        "adapter_config": asdict(config),
        "model_initialization_seed": 19,
        "window_start_indices": [0, 12],
        "num_optimizer_updates_per_window": 1,
        "optimizer_learning_rate": 1e-3,
        "device": "cpu",
        "torch_version": torch.__version__,
        "python_version": sys.version,
        "python_executable": sys.executable,
        "log_path": str(log_path),
        "log_sha256": log_digest_before,
        "windows_from_same_log_not_independent_scenes": True,
        "elapsed_seconds": time.perf_counter() - started_seconds,
        "scenes": scene_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
