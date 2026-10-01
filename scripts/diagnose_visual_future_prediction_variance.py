"""Read-only cache/checkpoint diagnostics; no encoder training or cache generation."""

import argparse
import json
import os
from pathlib import Path

import torch
import train_target_supervision_ablation as exploration

PROJECT_ROOT = exploration.PROJECT_ROOT
Model = exploration.FixedDistanceFutureSupervisionPilot


def load_existing_experiment(device):
    configuration = json.loads(
        (
            PROJECT_ROOT / "configs/exploration/target_supervision_run_v1.json"
        ).read_text()
    )
    cache_directory = PROJECT_ROOT / configuration["cache_directory"]
    cache_index = json.loads((cache_directory / "cache_index.json").read_text())
    windows = [
        torch.load(cache_directory / record["cache_file"], weights_only=True)
        for record in cache_index["records"]
    ]
    original_directory = (
        PROJECT_ROOT / "outputs/target_supervision_exploration/seed29_updates200_v1"
    )
    normalization_cpu = torch.load(
        original_directory / "train_only_normalization.pt", weights_only=True
    )
    normalization = {
        name: {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
        for name, values in normalization_cpu.items()
    }
    return configuration, windows, normalization, original_directory


@torch.no_grad()
def collect_predictions(model, windows, device, detach_future_for_planning=False):
    observations = {
        name: []
        for name in (
            "visual",
            "spatial",
            "current_visual",
            "common_valid",
            "visual_target",
            "spatial_target",
            "selected_valid",
        )
    }
    for start in range(0, len(windows), 8):
        online, targets = exploration.collate_cached_windows(
            windows[start : start + 8], device
        )
        arguments = (
            {"detach_future_for_planning": True} if detach_future_for_planning else {}
        )
        output = model(**online, **arguments)
        selected = exploration.gather_selected_training_targets(
            output.entity_selection, targets
        )
        selected_current = (
            output.entity_selection.hard_selection_weights
            @ online["current_entity_features"]
        )
        values = {
            "visual": output.predicted_future_visual_latents,
            "spatial": output.predicted_future_spatial_states,
            "current_visual": selected_current[..., :1024],
            "common_valid": selected["common_valid"],
            "visual_target": selected["visual"],
            "spatial_target": selected["spatial"],
            "selected_valid": output.entity_selection.selected_entity_valid_mask,
        }
        for name, value in values.items():
            observations[name].append(value.cpu())
    return {name: torch.cat(values) for name, values in observations.items()}


def variance_across_axis(values, valid_mask, axis):
    """Hold other axes fixed; population variance over valid entries of ONE axis."""
    weights = valid_mask.double()[..., None]
    values = values.double()
    count = weights.sum(dim=axis, keepdim=True)
    mean = (values * weights).sum(dim=axis, keepdim=True) / count.clamp_min(1)
    variance = ((values - mean).square() * weights).sum(dim=axis) / count.squeeze(
        axis
    ).clamp_min(1)
    eligible = count.squeeze(axis).squeeze(-1) >= 2
    return {
        "mean_channel_population_variance": float(variance[eligible].mean())
        if eligible.any()
        else None,
        "eligible_groups": int(eligible.sum()),
    }


def visual_prediction_diagnostics(observations, normalization):
    mean = normalization["visual"]["mean"].cpu()
    std = normalization["visual"]["std"].cpu()
    target = (observations["visual_target"] - mean) / std
    current = (observations["current_visual"] - mean) / std
    mask = observations["common_valid"]
    reference_predictions = {
        "predictor": observations["visual"],
        "persistence": current[:, :, None].expand_as(target),
        "train_mean": torch.zeros_like(target),
    }
    report = {
        "common_valid_slot_times": int(mask.sum()),
        "per_horizon": [],
        "baselines_fit_on": "train only",
    }
    for name, prediction in reference_predictions.items():
        error = prediction - target
        report[name] = {
            "normalized_mse": float(error[mask].square().mean()),
            "raw_mse": float((error * std)[mask].square().mean()),
            "pooled_sample_entity_time_channel_variance": exploration.channel_variance(
                prediction[mask]
            ),
            "across_windows_fixed_slot_and_horizon": variance_across_axis(
                prediction, mask, 0
            ),
            "across_entities_within_window_horizon": variance_across_axis(
                prediction, mask, 1
            ),
            "across_time_within_window_entity": variance_across_axis(
                prediction, mask, 2
            ),
            "future_minus_current_normalized_rms": float(
                (prediction - current[:, :, None])[mask].square().mean().sqrt()
            ),
        }
    report["target"] = {
        "pooled_sample_entity_time_channel_variance": exploration.channel_variance(
            target[mask]
        ),
        "across_windows_fixed_slot_and_horizon": variance_across_axis(target, mask, 0),
        "across_entities_within_window_horizon": variance_across_axis(target, mask, 1),
        "across_time_within_window_entity": variance_across_axis(target, mask, 2),
        "future_minus_current_normalized_rms": float(
            (target - current[:, :, None])[mask].square().mean().sqrt()
        ),
    }
    for future_step in range(target.shape[2]):
        valid = mask[:, :, future_step]
        row = {
            "future_seconds": (future_step + 1) * 0.5,
            "valid_count": int(valid.sum()),
        }
        for name, prediction in reference_predictions.items():
            row[name] = {
                "normalized_mse": float(
                    (prediction[:, :, future_step] - target[:, :, future_step])[valid]
                    .square()
                    .mean()
                ),
                "channel_variance": exploration.channel_variance(
                    prediction[:, :, future_step][valid]
                ),
                "future_minus_current_rms": float(
                    (prediction[:, :, future_step] - current)[valid]
                    .square()
                    .mean()
                    .sqrt()
                ),
            }
        row["target_variance"] = exploration.channel_variance(
            target[:, :, future_step][valid]
        )
        row["actual_future_minus_current_rms"] = float(
            (target[:, :, future_step] - current)[valid].square().mean().sqrt()
        )
        report["per_horizon"].append(row)
    recovered = target * std + mean
    report["normalization_inverse_max_error"] = float(
        (recovered[mask] - observations["visual_target"][mask]).abs().max()
    )
    manual = (observations["visual"][mask] - target[mask]).square().mean()
    implementation = exploration.masked_prediction_mse(
        observations["visual"], target, mask
    )
    report["mask_denominator_equivalence_absolute_error"] = float(
        (manual - implementation).abs()
    )
    return report


@torch.no_grad()
def perturbation_amplitudes(model, windows, device):
    """Original donor policy, with actual input changes and availability confounds."""
    recipient_indices = list(range(len(windows)))
    donor_indices = [
        next(
            (index + offset) % len(windows)
            for offset in range(1, len(windows))
            if windows[(index + offset) % len(windows)]["metadata"]["recording_group"]
            != windows[index]["metadata"]["recording_group"]
        )
        for index in recipient_indices
    ]
    sums = {
        name: {
            "visual_squared_change": 0.0,
            "spatial_squared_change": 0.0,
            "visual_squared_original": 0.0,
            "spatial_squared_original": 0.0,
            "active_slot_times": 0,
            "xy_change_sum": 0.0,
            "window_ade_delta_sum": 0.0,
            "donor_inactive_recipient_active_slots": 0,
        }
        for name in ("zero", "swap")
    }
    for start in range(0, len(windows), 8):
        online, targets = exploration.collate_cached_windows(
            windows[start : start + 8], device
        )
        donor_online, _ = exploration.collate_cached_windows(
            [windows[index] for index in donor_indices[start : start + 8]], device
        )
        original = model(**online)
        donor = model(**donor_online)
        valid = original.entity_selection.selected_entity_valid_mask[:, :, None].expand(
            -1, -1, 8
        )
        original_ade = (
            (
                original.ego_trajectory[..., :2]
                - targets["ego_trajectory_target"][..., :2]
            )
            .norm(dim=-1)
            .mean(dim=-1)
        )
        for name, sums_row in sums.items():
            visual = (
                torch.zeros_like(original.predicted_future_visual_latents)
                if name == "zero"
                else donor.predicted_future_visual_latents
            )
            spatial = (
                torch.zeros_like(original.predicted_future_spatial_states)
                if name == "zero"
                else donor.predicted_future_spatial_states
            )
            perturbed = model.ego_planner(
                online["current_image_grid"],
                online["current_entity_features"],
                online["current_entity_valid_mask"],
                online["current_ego_status"],
                visual,
                spatial,
                original.entity_selection.selected_entity_valid_mask,
            )
            for target_name, changed, unchanged in (
                ("visual", visual, original.predicted_future_visual_latents),
                ("spatial", spatial, original.predicted_future_spatial_states),
            ):
                sums_row[f"{target_name}_squared_change"] += float(
                    (changed - unchanged)[valid].double().square().sum()
                )
                sums_row[f"{target_name}_squared_original"] += float(
                    unchanged[valid].double().square().sum()
                )
            sums_row["active_slot_times"] += int(valid.sum())
            sums_row["xy_change_sum"] += float(
                (perturbed[..., :2] - original.ego_trajectory[..., :2])
                .norm(dim=-1)
                .mean(dim=-1)
                .sum()
            )
            sums_row["window_ade_delta_sum"] += float(
                (
                    (perturbed[..., :2] - targets["ego_trajectory_target"][..., :2])
                    .norm(dim=-1)
                    .mean(dim=-1)
                    - original_ade
                ).sum()
            )
            if name == "swap":
                sums_row["donor_inactive_recipient_active_slots"] += int(
                    (
                        original.entity_selection.selected_entity_valid_mask
                        & ~donor.entity_selection.selected_entity_valid_mask
                    ).sum()
                )
    report = {}
    for name, row in sums.items():
        report[name] = {
            "feature_space": "train-normalized coordinates supplied to planner; current-selected active slots only",
            "visual_feature_change_rms": (
                row["visual_squared_change"] / max(row["active_slot_times"] * 1024, 1)
            )
            ** 0.5,
            "spatial_feature_change_rms": (
                row["spatial_squared_change"] / max(row["active_slot_times"] * 6, 1)
            )
            ** 0.5,
            "visual_change_to_original_rms_ratio": (
                row["visual_squared_change"]
                / max(row["visual_squared_original"], 1e-30)
            )
            ** 0.5,
            "mean_output_xy_change_meters": row["xy_change_sum"] / len(windows),
            "mean_window_ade_delta_meters": row["window_ade_delta_sum"] / len(windows),
            "donor_inactive_recipient_active_slots": row[
                "donor_inactive_recipient_active_slots"
            ],
            "interpretation": "Dependency only. Zero may be OOD. Swap also changes slot availability; small feature changes constrain interpretation.",
        }
    return report


def predictor_current_roi_sensitivity(model, windows, device):
    online, _ = exploration.collate_cached_windows(windows[:8], device)
    selected = (
        exploration.select_nearest_current_entities(
            online["current_entity_features"],
            online["current_entity_valid_mask"],
            online["stable_entity_ids"],
        )
        if hasattr(exploration, "select_nearest_current_entities")
        else None
    )
    if selected is None:
        from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
            select_nearest_current_entities,
        )

        selected = select_nearest_current_entities(
            online["current_entity_features"],
            online["current_entity_valid_mask"],
            online["stable_entity_ids"],
        )
    features = (
        (selected.hard_selection_weights @ online["current_entity_features"])
        .detach()
        .requires_grad_()
    )
    prediction, _ = model.predict_selected_future(
        features,
        online["current_image_grid"].mean(dim=(-2, -1)),
        online["current_ego_status"],
        selected.selected_entity_valid_mask,
    )
    gradient = torch.autograd.grad(prediction.square().sum(), features)[0]
    changed_features = features.detach().clone()
    changed_features[..., :1024] = 0
    changed_prediction, _ = model.predict_selected_future(
        changed_features,
        online["current_image_grid"].mean(dim=(-2, -1)),
        online["current_ego_status"],
        selected.selected_entity_valid_mask,
    )
    valid = selected.selected_entity_valid_mask
    return {
        "selected_current_roi_gradient_norm": float(gradient[..., :1024].norm()),
        "zero_current_roi_prediction_change_rms": float(
            (changed_prediction - prediction)[valid].square().mean().sqrt().detach()
        ),
        "zero_is_ood": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.physical_gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    output_directory = arguments.output_directory.resolve()
    if (
        not output_directory.is_relative_to(PROJECT_ROOT / "outputs")
        or output_directory.exists()
    ):
        raise ValueError("require new project outputs directory")
    output_directory.mkdir(parents=True)
    _configuration, windows, normalization, original_directory = (
        load_existing_experiment(device)
    )
    report = {
        "baseline_commit": "9353acf",
        "new_training_updates": 0,
        "conditions": {},
        "torch_version": torch.__version__,
    }
    for label in ("C", "E"):
        checkpoint_path = original_directory / label / "last_update_200.pt"
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model = Model().to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        condition_report = {
            "checkpoint_sha256": exploration.file_sha256(checkpoint_path),
            "splits": {},
        }
        for split_name in ("train", "development"):
            selected_windows = [
                window
                for window in windows
                if window["metadata"]["split"] == split_name
            ]
            observations = collect_predictions(model, selected_windows, device)
            condition_report["splits"][split_name] = visual_prediction_diagnostics(
                observations, normalization
            )
        development = [
            window for window in windows if window["metadata"]["split"] == "development"
        ]
        condition_report["perturbations"] = perturbation_amplitudes(
            model, development, device
        )
        condition_report["current_roi_input_check"] = predictor_current_roi_sensitivity(
            model, development, device
        )
        report["conditions"][label] = condition_report
        print(
            f"DIAGNOSED {label} {json.dumps(condition_report['splits']['development']['predictor'])}",
            flush=True,
        )
    (output_directory / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print("VARIANCE_DIAGNOSTICS_DONE", flush=True)


if __name__ == "__main__":
    main()
