"""One real-video window: frozen official encoder -> GT ROI -> gradient diagnostic.

No official benchmark metric, learned detector, training sweep, or cache rebuild.
Only project outputs/results are written; original NAVSIM files are read-only.
"""

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from PIL import Image, ImageDraw
from torch.nn import functional

from planning_aware_future_prediction.adapters.navsim_tracked_state import (
    TrackedStateAdapterConfig,
    build_tracked_state_online_inputs,
    build_tracked_state_training_targets,
)
from planning_aware_future_prediction.adapters.navsim_visual_entities import (
    build_front_video_clip,
    front_image_path,
    front_track_rois,
    load_front_image,
    pool_projected_entity_rois,
)
from planning_aware_future_prediction.models.frozen_driving_video_encoder import (
    FrozenDrivingVideoEncoder,
)
from planning_aware_future_prediction.models.selective_entity_future_prediction import (
    compute_parameter_gradient_norm,
)
from planning_aware_future_prediction.models.visual_entity_future_planning_pilot import (
    VisualEntityFuturePlanningPilot,
)


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gradient_norms(loss, model):
    return {
        "selector": compute_parameter_gradient_norm(loss, model.entity_scorer),
        "predictor": compute_parameter_gradient_norm(loss, model.future_predictor),
        "planner": compute_parameter_gradient_norm(loss, model.ego_planner),
    }


def write_roi_contact_sheet(
    frames, sensor_root, track_tokens, selected_indices, destination
):
    displayed_steps = [0, 2, 4, 8]
    contact_sheet = Image.new("RGB", (1024, 560), "#111111")
    for panel_index, step in enumerate(displayed_steps):
        panel = Image.fromarray(load_front_image(frames[step], sensor_root))
        painter = ImageDraw.Draw(panel)
        boxes, valid = front_track_rois(frames[step], track_tokens)
        for entity_index, box in enumerate(boxes):
            if not valid[entity_index]:
                continue
            is_selected = entity_index in selected_indices
            color = "#ffdc00" if is_selected else "#20dc8b"
            painter.rectangle(
                box.tolist(), outline=color, width=2 if is_selected else 1
            )
            if is_selected:
                painter.text(
                    (float(box[0]), max(0, float(box[1]) - 11)),
                    f"{entity_index}:{track_tokens[entity_index][:4]}",
                    fill=color,
                )
        horizontal = (panel_index % 2) * 512
        vertical = (panel_index // 2) * 280
        contact_sheet.paste(panel, (horizontal, vertical + 24))
        ImageDraw.Draw(contact_sheet).text(
            (horizontal + 8, vertical + 6),
            f"step={step} GT projected ROI; yellow=UNTRAINED selection",
            fill="white",
        )
    contact_sheet.save(destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/visual_pilot/visual_future_pilot_20261001.json",
    )
    arguments = parser.parse_args()
    output_path = arguments.output.resolve()
    if not output_path.is_relative_to(PROJECT_ROOT / "outputs/visual_pilot"):
        parser.error("output must be within outputs/visual_pilot")
    if output_path.exists():
        parser.error("refusing to overwrite an existing diagnostic result")
    torch.set_num_threads(4)
    torch.manual_seed(29)
    started = time.perf_counter()
    device = torch.device(arguments.device)
    if device.type == "cuda":
        import os

        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
            parser.error("this bounded diagnostic requires CUDA_VISIBLE_DEVICES=0")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.reset_peak_memory_stats()
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    log_path = (
        dataset_root / "navsim_logs/mini/2021.05.12.22.00.38_veh-35_01008_01518.pkl"
    )
    sensor_root = dataset_root / "sensor_blobs/mini"
    log_sha_before = file_sha256(log_path)
    with log_path.open("rb") as log_file:
        frames = pickle.load(log_file)  # trusted lab NAVSIM local pickle ONLY
    config = TrackedStateAdapterConfig()
    start = arguments.window_start
    if start < 0 or start + config.num_history_frames + config.num_future_steps > len(
        frames
    ):
        parser.error("window outside log")
    history = frames[start : start + config.num_history_frames]
    current = history[-1]
    future = frames[
        start + config.num_history_frames : start
        + config.num_history_frames
        + config.num_future_steps
    ]
    state_online = build_tracked_state_online_inputs(history, config)
    state_targets = build_tracked_state_training_targets(
        current, future, state_online, config
    )
    image_frames = [history[-2], current, *future]
    image_digests_before = {
        str(front_image_path(frame, sensor_root)): file_sha256(
            front_image_path(frame, sensor_root)
        )
        for frame in image_frames
    }
    encoder = FrozenDrivingVideoEncoder(PROJECT_ROOT, device)
    print("Verified official frozen encoder loaded strictly", flush=True)
    feature_grids = []
    for clip_step in range(config.num_future_steps + 1):
        observed_clip = build_front_video_clip(
            image_frames[clip_step : clip_step + 2], sensor_root
        ).to(device)
        feature_grids.append(encoder(observed_clip))
        print(f"encoded clip {clip_step}: {tuple(feature_grids[-1].shape)}", flush=True)
    projected_boxes, projected_masks, roi_features = [], [], []
    for frame, grid in zip([current, *future], feature_grids):
        boxes, valid = front_track_rois(frame, state_online.current_track_tokens)
        projected_boxes.append(boxes)
        projected_masks.append(valid.to(device))
        roi_features.append(pool_projected_entity_rois(grid, boxes, valid))
    current_image_grid = feature_grids[0].detach()
    current_entity_valid = projected_masks[0][None] & state_online.entity_valid_mask.to(
        device
    )
    current_entity_features = torch.cat(
        (roi_features[0], state_online.entity_features.to(device)), dim=-1
    )
    current_ego_status = torch.cat(
        (
            state_online.ego_intent,
            torch.tensor(current["ego_dynamic_state"], dtype=torch.float32)[None],
        ),
        dim=-1,
    ).to(device)
    model_inputs = (
        current_image_grid,
        current_entity_features,
        current_entity_valid,
        state_online.stable_entity_ids.to(device),
        current_ego_status,
    )
    future_visual_targets = torch.stack(roi_features[1:], dim=2).detach()
    future_visual_valid = torch.stack(projected_masks[1:], dim=1)[
        None
    ] & state_targets.future_target_valid_mask.to(device)
    future_spatial_targets = state_targets.future_state_targets.to(device).detach()
    future_spatial_valid = state_targets.future_target_valid_mask.to(device)
    ego_target = state_targets.ego_trajectory_target.to(device)
    if current_entity_valid.sum() <= 4:
        raise RuntimeError(
            "window has <=4 front candidates: no subset decision; choose another window and report why"
        )
    model = VisualEntityFuturePlanningPilot().to(device).eval()
    output = model(*model_inputs)
    planning_loss = functional.mse_loss(output.ego_trajectory, ego_target)
    auxiliary = model.detached_selection_auxiliary_losses(
        output,
        current_image_grid,
        current_entity_features,
        current_entity_valid,
        current_ego_status,
        future_visual_targets,
        future_visual_valid,
        future_spatial_targets,
        future_spatial_valid,
    )
    gradient_contracts = {"planning": gradient_norms(planning_loss, model)}
    for name, loss in auxiliary.items():
        gradient_contracts[name] = gradient_norms(loss, model)
        assert (
            gradient_contracts[name]["selector"]
            == gradient_contracts[name]["planner"]
            == 0
        )
        assert gradient_contracts[name]["predictor"] > 0
    assert all(norm > 0 for norm in gradient_contracts["planning"].values())
    detached = model(*model_inputs, detach_predicted_future=True)
    assert torch.equal(detached.ego_trajectory, output.ego_trajectory)
    gradient_contracts["future_detach"] = gradient_norms(
        functional.mse_loss(detached.ego_trajectory, ego_target), model
    )
    assert (
        gradient_contracts["future_detach"]["selector"]
        == gradient_contracts["future_detach"]["predictor"]
        == 0
    )
    no_future = model(*model_inputs, enable_future_branch=False)
    gradient_contracts["no_future_branch"] = gradient_norms(
        functional.mse_loss(no_future.ego_trajectory, ego_target), model
    )
    assert no_future.predicted_future_visual_latents is None
    assert (
        gradient_contracts["no_future_branch"]["selector"]
        == gradient_contracts["no_future_branch"]["predictor"]
        == 0
    )
    # Same-capacity current-feature target control: target only changes the loss.
    current_visual_targets = roi_features[0][:, :, None].expand_as(
        future_visual_targets
    )
    current_spatial_targets = (
        state_online.entity_features[:, :, [0, 1, 2, 3, 6, 7]]
        .to(device)[:, :, None]
        .expand_as(future_spatial_targets)
    )
    current_targets_valid = current_entity_valid[:, :, None].expand_as(
        future_visual_valid
    )
    current_target_losses = model.detached_selection_auxiliary_losses(
        output,
        current_image_grid,
        current_entity_features,
        current_entity_valid,
        current_ego_status,
        current_visual_targets,
        current_targets_valid,
        current_spatial_targets,
        current_targets_valid,
    )
    repeat_output = model(*model_inputs)
    assert torch.equal(repeat_output.ego_trajectory, output.ego_trajectory)
    assert not future_visual_targets.requires_grad and not any(
        parameter.requires_grad for parameter in encoder.parameters()
    )
    selected_indices = output.entity_selection.selected_entity_indices[0].tolist()
    selected_valid_visual_count = int(
        future_visual_valid[
            0, [index for index in selected_indices if index >= 0]
        ].sum()
    )
    zero_future_plan = model.ego_planner(
        current_image_grid,
        current_entity_features,
        current_entity_valid,
        current_ego_status,
        torch.zeros_like(output.predicted_future_visual_latents),
        torch.zeros_like(output.predicted_future_spatial_states),
        output.entity_selection.selected_entity_valid_mask,
    )
    future_zero_difference = (
        (zero_future_plan - output.ego_trajectory).abs().max().item()
    )
    before_update = model.entity_scorer.entity_score_network[0].weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    joint_loss = (
        planning_loss + auxiliary["visual_prediction"] + auxiliary["spatial_prediction"]
    )
    joint_loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    assert not torch.equal(
        before_update, model.entity_scorer.entity_score_network[0].weight.detach()
    )
    assert file_sha256(log_path) == log_sha_before
    assert all(
        file_sha256(Path(path)) == digest
        for path, digest in image_digests_before.items()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization_path = output_path.parent / (
        output_path.stem + "_roi_contact_sheet.png"
    )
    write_roi_contact_sheet(
        [current, *future],
        sensor_root,
        state_online.current_track_tokens,
        selected_indices,
        visualization_path,
    )
    valid_appearance_pairs = future_visual_valid[0] & current_entity_valid[0, :, None]
    appearance_similarity = (
        functional.cosine_similarity(
            current_visual_targets[0], future_visual_targets[0], dim=-1
        )[valid_appearance_pairs]
        .detach()
        .cpu()
    )
    metric_motion = (
        (
            (future_spatial_targets[0, :, :, :2] - current_spatial_targets[0, :, :, :2])
            * 40
        )
        .norm(dim=-1)[valid_appearance_pairs]
        .cpu()
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "one_same_scene_GT_ROI_front_video_plumbing_not_benchmark_or_H1_H2",
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "official_encoder": encoder.provenance,
        "environment": {
            "torch": torch.__version__,
            "python": sys.version,
            "executable": sys.executable,
            "device": str(device),
        },
        "window": {
            "log_path": str(log_path),
            "start_index": start,
            "current_token": str(current["token"]),
            "scene_token": str(current.get("scene_token")),
            "future_time_offsets_seconds": state_targets.future_time_offsets_seconds.tolist(),
        },
        "tensor_shapes": {
            "current_grid": list(current_image_grid.shape),
            "current_entity_features": list(current_entity_features.shape),
            "future_visual_targets": list(future_visual_targets.shape),
            "future_spatial_targets": list(future_spatial_targets.shape),
            "selected_predicted_visual": list(
                output.predicted_future_visual_latents.shape
            ),
            "ego_trajectory": list(output.ego_trajectory.shape),
        },
        "coverage": {
            "current_radius_candidates_after_cap32": len(
                state_online.current_track_tokens
            ),
            "current_front_projected_candidates": int(current_entity_valid.sum()),
            "selected_budget": 4,
            "future_front_projected_per_step": [
                int(mask.sum()) for mask in projected_masks[1:]
            ],
            "selected_valid_visual_targets": selected_valid_visual_count,
            "selected_future_target_slots": 32,
            "occlusion_visibility_verified": False,
        },
        "selected_current_entity_indices_UNTRAINED": selected_indices,
        "operational_prediction_budget": {
            "entity_slots": 4,
            "time_steps": 8,
            "visual_dim": 1024,
            "spatial_dim": 6,
            "predicted_visual_values": 32768,
            "predicted_spatial_values": 192,
            "encoder_processes_whole_front_image": True,
            "flops_or_latency_savings_verified": False,
        },
        "gradient_contracts": gradient_contracts,
        "diagnostic_losses_NOT_performance": {
            "planning_mse": planning_loss.item(),
            **{name: loss.item() for name, loss in auxiliary.items()},
            "current_feature_target_losses": {
                name: loss.item() for name, loss in current_target_losses.items()
            },
        },
        "contracts": {
            "future_detach_forward_equal": True,
            "training_target_change_forward_equal": True,
            "frozen_teacher_no_gradient": True,
            "finite_joint_backward": True,
            "single_selector_optimizer_update": True,
            "no_future_branch_skips_selector_predictor": True,
            "shared_input_hashes_unchanged": True,
        },
        "untrained_future_zeroing_max_plan_difference_NOT_usefulness": future_zero_difference,
        "target_content_diagnostic": {
            "current_to_future_cosine_mean": appearance_similarity.mean().item(),
            "current_to_future_cosine_min": appearance_similarity.min().item(),
            "paired_metric_displacement_mean_meters": metric_motion.mean().item(),
            "paired_metric_displacement_max_meters": metric_motion.max().item(),
            "location_is_explicit_separate_GT_state_target_NOT_assumed_in_ROI": True,
            "clip_time_span_seconds": 0.5,
        },
        "new_scaffold_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "new_scaffold_parameters_by_module": {
            "selector": sum(
                parameter.numel() for parameter in model.entity_scorer.parameters()
            ),
            "predictor": sum(
                parameter.numel() for parameter in model.future_predictor.parameters()
            ),
            "planner": sum(
                parameter.numel() for parameter in model.ego_planner.parameters()
            ),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if device.type == "cuda"
        else None,
        "elapsed_seconds_including_hash_load_encode": time.perf_counter() - started,
        "visualization_path": str(visualization_path),
        "shared_file_sha256": {str(log_path): log_sha_before, **image_digests_before},
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in [
                Path(__file__),
                *sorted(
                    (
                        PROJECT_ROOT / "src/planning_aware_future_prediction/adapters"
                    ).glob("*.py")
                ),
                *sorted(
                    (PROJECT_ROOT / "src/planning_aware_future_prediction/models").glob(
                        "*.py"
                    )
                ),
            ]
        },
        "limitations": [
            "GT current boxes/association are privileged",
            "frozen GT ROI visual+spatial mixed supervision not pure JEPA",
            "front-only candidate/40m/max32 prefilter",
            "pinhole distortion/rectification unverified",
            "two-frame tubelet representation not instantaneous object state",
            "random new planner not official baseline",
            "one update no stability/importance/efficiency result",
            "no official planning metrics",
            "future visibility changes target availability may bias later training",
        ],
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "report": str(output_path),
                "coverage": report["coverage"],
                "gradient_contracts": gradient_contracts,
                "elapsed_seconds": report["elapsed_seconds_including_hash_load_encode"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
