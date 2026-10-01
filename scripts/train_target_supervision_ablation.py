"""Matched 200-update target-supervision exploration; NOT official planning evaluation."""

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch

from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
    FixedDistanceFutureSupervisionPilot,
    gather_selected_training_targets,
    masked_prediction_mse,
    planning_imitation_loss,
    train_only_target_normalization,
)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state_sha256(state_dict):
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode())
        digest.update(tensor.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def own_process_gpu_memory_megabytes():
    try:
        rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).splitlines()
        return max(
            (
                int(row.split(",")[1])
                for row in rows
                if row.split(",")[0].strip().isdigit()
                and int(row.split(",")[0]) == os.getpid()
            ),
            default=None,
        )
    except (ValueError, subprocess.SubprocessError, OSError):
        return None


def collate_cached_windows(windows, device):
    def tensor_section(section):
        result = {}
        for name in windows[0][section]:
            stacked = torch.stack([window[section][name] for window in windows]).to(
                device
            )
            result[name] = stacked.float() if stacked.is_floating_point() else stacked
        return result

    return tensor_section("online_inputs"), tensor_section("training_targets")


def normalized_selected_targets(selected_targets, normalization):
    return {
        name: (selected_targets[name] - normalization[name]["mean"])
        / normalization[name]["std"]
        for name in ("visual", "spatial")
    }


def compute_losses(model_output, training_targets, normalization, condition):
    planning_loss = planning_imitation_loss(
        model_output.ego_trajectory, training_targets["ego_trajectory_target"]
    )
    selected = gather_selected_training_targets(
        model_output.entity_selection, training_targets
    )
    normalized_targets = normalized_selected_targets(selected, normalization)
    components = {"planning": planning_loss}
    total_loss = planning_loss
    # With a hard fixed rule and frozen encoder, this output already has the
    # detached selection contract. Aux uses predictor only, never the planner.
    for name, predicted in (
        ("visual", model_output.predicted_future_visual_latents),
        ("spatial", model_output.predicted_future_spatial_states),
    ):
        weight = condition[f"{name}_weight"]
        if weight:
            components[name] = masked_prediction_mse(
                predicted, normalized_targets[name], selected["common_valid"]
            )
            total_loss = total_loss + weight * components[name]
    return total_loss, components, selected


def parameter_gradient_norm(parameters):
    norms = [
        parameter.grad.square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return float(torch.stack(norms).sum().sqrt()) if norms else 0.0


def tensor_gradient_norm(loss, parameters):
    gradients = torch.autograd.grad(
        loss, list(parameters), retain_graph=True, allow_unused=True
    )
    terms = [gradient.square().sum() for gradient in gradients if gradient is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def channel_variance(observations):
    if observations.shape[0] < 2:
        return None
    return float(observations.double().var(dim=0, correction=0).mean())


def prediction_metrics(
    predictions, selected_targets, normalization, current_visual, current_spatial
):
    report = {}
    for target_name in ("visual", "spatial"):
        predicted_normalized = predictions[target_name]
        mean, std = (
            normalization[target_name]["mean"].cpu(),
            normalization[target_name]["std"].cpu(),
        )
        raw_target = selected_targets[target_name]
        normalized_target = (raw_target - mean) / std
        raw_prediction = predicted_normalized * std + mean
        persistence = current_visual if target_name == "visual" else current_spatial
        for mask_name in ("common_valid", f"{target_name}_valid"):
            valid = selected_targets[mask_name]
            count = int(valid.sum())
            row = {"valid_slot_time_count": count}
            if count:
                predicted_values, target_values = (
                    predicted_normalized[valid],
                    normalized_target[valid],
                )
                row["normalized_mse"] = float(
                    (predicted_values - target_values).square().mean()
                )
                row["raw_mse"] = float(
                    (raw_prediction[valid] - raw_target[valid]).square().mean()
                )
                row["persistence_normalized_mse"] = float(
                    ((persistence - raw_target) / std)[valid].square().mean()
                )
                row["prediction_channel_variance"] = channel_variance(predicted_values)
                row["target_channel_variance"] = channel_variance(target_values)
                row["prediction_to_target_variance_ratio"] = (
                    row["prediction_channel_variance"]
                    / max(row["target_channel_variance"], 1e-12)
                    if row["prediction_channel_variance"] is not None
                    else None
                )
                if target_name == "visual":
                    row["raw_cosine_similarity"] = float(
                        torch.nn.functional.cosine_similarity(
                            raw_prediction[valid], raw_target[valid], dim=-1
                        ).mean()
                    )
                else:
                    error = raw_prediction[valid] - raw_target[valid]
                    row["position_vector_rmse_meters"] = float(
                        (error[:, :2].square().sum(dim=-1).mean()).sqrt() * 40.0
                    )
                    row["velocity_vector_rmse_meters_per_second"] = float(
                        (error[:, 4:6].square().sum(dim=-1).mean()).sqrt() * 10.0
                    )
                    predicted_yaw = torch.atan2(
                        raw_prediction[valid][:, 2], raw_prediction[valid][:, 3]
                    )
                    target_yaw = torch.atan2(
                        raw_target[valid][:, 2], raw_target[valid][:, 3]
                    )
                    row["heading_mae_radians"] = float(
                        torch.atan2(
                            torch.sin(predicted_yaw - target_yaw),
                            torch.cos(predicted_yaw - target_yaw),
                        )
                        .abs()
                        .mean()
                    )
            report[f"{target_name}_{mask_name}"] = row
    report["visual_per_horizon_common_valid_variance"] = []
    for future_step in range(8):
        valid = selected_targets["common_valid"][:, :, future_step]
        report["visual_per_horizon_common_valid_variance"].append(
            {
                "future_seconds": (future_step + 1) * 0.5,
                "valid_count": int(valid.sum()),
                "prediction_channel_variance": channel_variance(
                    predictions["visual"][:, :, future_step][valid]
                ),
                "target_channel_variance": channel_variance(
                    (
                        (
                            selected_targets["visual"]
                            - normalization["visual"]["mean"].cpu()
                        )
                        / normalization["visual"]["std"].cpu()
                    )[:, :, future_step][valid]
                ),
            }
        )
    return report


def aggregate_planning_metrics(per_window):
    by_scene, by_recording = defaultdict(list), defaultdict(list)
    buckets = defaultdict(list)
    for row in per_window:
        by_scene[(row["recording_group"], row["scene_token"])].append(row["ade_meters"])
        by_recording[row["recording_group"]].append(row["ade_meters"])
        count = row["current_front_valid_count"]
        buckets["current_count_gt4" if count > 4 else "current_count_le4"].append(row)
        if count == 0:
            buckets["current_count_zero"].append(row)
        speed = row["ego_speed_meters_per_second"]
        buckets[
            "speed_lt2" if speed < 2 else "speed_2_to_8" if speed < 8 else "speed_ge8"
        ].append(row)
        buckets[f"command_raw_{row['command_raw_index']}"].append(row)
        angle = row["posthoc_future_heading_radians"]
        buckets[
            "posthoc_heading_left"
            if angle > np.deg2rad(15)
            else "posthoc_heading_right"
            if angle < -np.deg2rad(15)
            else "posthoc_heading_low"
        ].append(row)
    summary = {
        "windows": len(per_window),
        "scenes": len(by_scene),
        "recordings": len(by_recording),
        "planning_loss": float(np.mean([row["planning_loss"] for row in per_window])),
        "window_mean_ade_meters": float(
            np.mean([row["ade_meters"] for row in per_window])
        ),
        "scene_macro_ade_meters": float(
            np.mean([np.mean(values) for values in by_scene.values()])
        ),
        "recording_macro_ade_meters": float(
            np.mean([np.mean(values) for values in by_recording.values()])
        ),
        "fde_meters": float(np.mean([row["fde_meters"] for row in per_window])),
        "heading_mae_radians": float(
            np.mean([row["heading_mae_radians"] for row in per_window])
        ),
        "common_valid_slot_time_count": sum(
            row["common_valid_slot_time_count"] for row in per_window
        ),
        "native_visual_valid_slot_time_count": sum(
            row["native_visual_valid_slot_time_count"] for row in per_window
        ),
        "native_spatial_valid_slot_time_count": sum(
            row["native_spatial_valid_slot_time_count"] for row in per_window
        ),
        "zero_common_target_windows": sum(
            row["common_valid_slot_time_count"] == 0 for row in per_window
        ),
        "nominal_selected_entity_count": 4 * len(per_window),
        "active_selected_entity_count": sum(
            row["active_selected_entity_count"] for row in per_window
        ),
        "per_recording_ade_meters": {
            key: float(np.mean(values)) for key, values in sorted(by_recording.items())
        },
        "context_buckets": {},
    }
    for name, rows in sorted(buckets.items()):
        recording_count = len({row["recording_group"] for row in rows})
        summary["context_buckets"][name] = {
            "windows": len(rows),
            "recordings": recording_count,
            "window_mean_ade_meters": float(
                np.mean([row["ade_meters"] for row in rows])
            ),
            "insufficient_coverage": len(rows) < 8 or recording_count < 2,
        }
    return summary


@torch.no_grad()
def evaluate_model(model, windows, normalization, condition, device, batch_size):
    model.eval()
    evaluation_start = time.perf_counter()
    outputs, selected_batches, current_visual, current_spatial, per_window = (
        [],
        [],
        [],
        [],
        [],
    )
    for start in range(0, len(windows), batch_size):
        window_batch = windows[start : start + batch_size]
        online, targets = collate_cached_windows(window_batch, device)
        output = model(**online, enable_future_branch=condition["enable_future_branch"])
        selected = gather_selected_training_targets(output.entity_selection, targets)
        for sample, window in enumerate(window_batch):
            predicted, target = (
                output.ego_trajectory[sample],
                targets["ego_trajectory_target"][sample],
            )
            xy_error = (predicted[:, :2] - target[:, :2]).norm(dim=-1)
            heading_error = torch.atan2(
                torch.sin(predicted[:, 2] - target[:, 2]),
                torch.cos(predicted[:, 2] - target[:, 2]),
            ).abs()
            per_window.append(
                {
                    **window["metadata"],
                    "ade_meters": float(xy_error.mean()),
                    "fde_meters": float(xy_error[-1]),
                    "heading_mae_radians": float(heading_error.mean()),
                    "planning_loss": float(planning_imitation_loss(predicted, target)),
                    "common_valid_slot_time_count": int(
                        selected["common_valid"][sample].sum()
                    ),
                    "native_visual_valid_slot_time_count": int(
                        selected["visual_valid"][sample].sum()
                    ),
                    "native_spatial_valid_slot_time_count": int(
                        selected["spatial_valid"][sample].sum()
                    ),
                    "active_selected_entity_count": int(
                        output.entity_selection.selected_entity_valid_mask[sample].sum()
                    ),
                }
            )
        if condition["enable_future_branch"]:
            outputs.append(
                {
                    "visual": output.predicted_future_visual_latents.cpu(),
                    "spatial": output.predicted_future_spatial_states.cpu(),
                }
            )
            selected_batches.append(
                {name: values.cpu() for name, values in selected.items()}
            )
            selected_current = (
                output.entity_selection.hard_selection_weights
                @ online["current_entity_features"]
            )
            current_visual.append(
                selected_current[..., :1024, None]
                .transpose(-1, -2)
                .expand(-1, -1, 8, -1)
                .cpu()
            )
            current_spatial.append(
                selected_current[..., -10:][..., [0, 1, 2, 3, 6, 7]][:, :, None]
                .expand(-1, -1, 8, -1)
                .cpu()
            )
    summary = aggregate_planning_metrics(per_window)
    if outputs:
        summary["prediction"] = prediction_metrics(
            {
                name: torch.cat([output[name] for output in outputs])
                for name in ("visual", "spatial")
            },
            {
                name: torch.cat([selected[name] for selected in selected_batches])
                for name in selected_batches[0]
            },
            normalization,
            torch.cat(current_visual),
            torch.cat(current_spatial),
        )
    else:
        summary["prediction"] = None
    summary["evaluation_wall_seconds"] = time.perf_counter() - evaluation_start
    return summary, per_window


@torch.no_grad()
def future_branch_dependence(model, windows, condition, device, batch_size):
    if not condition["enable_future_branch"]:
        return {"status": "branch_absent"}
    model.eval()
    # Swap to a deterministic different-scene AND different-recording donor;
    # donor prediction is generated from donor ONLINE inputs, never GT future.
    donor_indices = []
    for index, window in enumerate(windows):
        donor_indices.append(
            next(
                (index + offset) % len(windows)
                for offset in range(1, len(windows))
                if windows[(index + offset) % len(windows)]["metadata"][
                    "recording_group"
                ]
                != window["metadata"]["recording_group"]
            )
        )
    changes = {name: [] for name in ("zero", "swap", "remove_branch")}
    for start in range(0, len(windows), batch_size):
        recipient_windows = windows[start : start + batch_size]
        donor_windows = [
            windows[index] for index in donor_indices[start : start + batch_size]
        ]
        online, targets = collate_cached_windows(recipient_windows, device)
        donor_online, _ = collate_cached_windows(donor_windows, device)
        output = model(**online)
        donor_output = model(**donor_online)
        for perturbation_name, perturbation_rows in changes.items():
            visual = (
                torch.zeros_like(output.predicted_future_visual_latents)
                if perturbation_name == "zero"
                else donor_output.predicted_future_visual_latents
            )
            spatial = (
                torch.zeros_like(output.predicted_future_spatial_states)
                if perturbation_name == "zero"
                else donor_output.predicted_future_spatial_states
            )
            perturbed_trajectory = model.ego_planner(
                online["current_image_grid"],
                online["current_entity_features"],
                online["current_entity_valid_mask"],
                online["current_ego_status"],
                visual,
                spatial,
                output.entity_selection.selected_entity_valid_mask,
            )
            if perturbation_name == "remove_branch":
                perturbed_trajectory = model(
                    **online, enable_future_branch=False
                ).ego_trajectory
            original_error = (
                (
                    output.ego_trajectory[..., :2]
                    - targets["ego_trajectory_target"][..., :2]
                )
                .norm(dim=-1)
                .mean(dim=-1)
            )
            perturbed_error = (
                (
                    perturbed_trajectory[..., :2]
                    - targets["ego_trajectory_target"][..., :2]
                )
                .norm(dim=-1)
                .mean(dim=-1)
            )
            output_change = (
                (perturbed_trajectory[..., :2] - output.ego_trajectory[..., :2])
                .norm(dim=-1)
                .mean(dim=-1)
            )
            perturbation_rows.extend(
                [
                    {
                        **window["metadata"],
                        "ade_delta_meters": float(
                            perturbed_error[index] - original_error[index]
                        ),
                        "output_xy_change_meters": float(output_change[index]),
                    }
                    for index, window in enumerate(recipient_windows)
                ]
            )
    return {
        name: {
            "mean_window_ade_delta_meters": float(
                np.mean([row["ade_delta_meters"] for row in rows])
            ),
            "mean_output_xy_change_meters": float(
                np.mean([row["output_xy_change_meters"] for row in rows])
            ),
            "donor_uses_future_gt": False,
            "recipient_current_padding_preserved": True,
            "interpretation": "Dependency diagnostic, not evidence that the branch is beneficial; zero leaves type/time tokens present, remove_branch omits future memory entirely.",
        }
        for name, rows in changes.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        type=Path,
        default=PROJECT_ROOT / "configs/exploration/target_supervision_run_v1.json",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    parser.add_argument("--profile-only", action="store_true")
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.physical_gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("approved GPU unavailable")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    configuration = json.loads(arguments.configuration.read_text())
    if (
        configuration["optimizer_updates"] != 200
        or configuration["selection"]["learned_selector"]
        or configuration["selection"]["num_selected_entities"] != 4
    ):
        raise RuntimeError(
            "this runner is limited to the approved fixed-rule 200-update pilot"
        )
    output_directory = arguments.output_directory.resolve()
    if not output_directory.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("output must stay in project outputs")
    if output_directory.exists():
        raise RuntimeError("refusing to overwrite an experiment")
    output_directory.mkdir(parents=True)
    cache_directory = PROJECT_ROOT / configuration["cache_directory"]
    cache_index = json.loads((cache_directory / "cache_index.json").read_text())
    if not arguments.profile_only and not cache_index["complete"]:
        raise RuntimeError("full committed small cache required")
    if cache_index["configuration"]["manifest_sha256"] != file_sha256(
        PROJECT_ROOT / configuration["split_manifest"]
    ):
        raise RuntimeError("split differs from cache")
    windows = []
    for record in cache_index["records"]:
        path = cache_directory / record["cache_file"]
        if file_sha256(path) != record["cache_sha256"]:
            raise RuntimeError("feature cache hash mismatch")
        windows.append(torch.load(path, map_location="cpu", weights_only=True))
    training_windows = [
        window for window in windows if window["metadata"]["split"] == "train"
    ]
    development_windows = [
        window for window in windows if window["metadata"]["split"] == "development"
    ]
    if not arguments.profile_only and (
        len(training_windows),
        len(development_windows),
    ) != (277, 96):
        raise RuntimeError("unexpected window split")
    if {window["metadata"]["recording_group"] for window in training_windows} & {
        window["metadata"]["recording_group"] for window in development_windows
    }:
        raise RuntimeError("train/development recording leakage")
    normalization_cpu = train_only_target_normalization(training_windows)
    normalization = {
        name: {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in values.items()
        }
        for name, values in normalization_cpu.items()
    }
    torch.save(normalization_cpu, output_directory / "train_only_normalization.pt")
    random.seed(configuration["seed"])
    np.random.seed(configuration["seed"])
    torch.manual_seed(configuration["seed"])
    initial_model = FixedDistanceFutureSupervisionPilot()
    shared_initial_state = copy.deepcopy(initial_model.state_dict())
    initial_hash = state_sha256(shared_initial_state)
    torch.save(shared_initial_state, output_directory / "shared_initial_state.pt")
    recording_indices = defaultdict(list)
    for index, window in enumerate(training_windows):
        recording_indices[window["metadata"]["recording_group"]].append(index)
    generator = random.Random(configuration["seed"])
    recordings = sorted(recording_indices)
    batch_sequence = []
    for _ in range(configuration["optimizer_updates"]):
        indices = []
        for _ in range(configuration["batch_size"]):
            recording = generator.choice(recordings)
            indices.append(generator.choice(recording_indices[recording]))
        batch_sequence.append(indices)
    batch_sequence_hash = hashlib.sha256(
        json.dumps(batch_sequence).encode()
    ).hexdigest()
    (output_directory / "shared_batch_sequence.json").write_text(
        json.dumps(
            {
                "training_window_order": [
                    window["metadata"]["current_frame_token"]
                    for window in training_windows
                ],
                "batch_indices": batch_sequence,
                "sha256": batch_sequence_hash,
            },
            indent=2,
        )
        + "\n"
    )
    provenance = {
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip(),
        "configuration": configuration,
        "configuration_sha256": file_sha256(arguments.configuration),
        "cache_index_sha256": file_sha256(cache_directory / "cache_index.json"),
        "cache_configuration": cache_index["configuration"],
        "encoder_provenance": cache_index["encoder_provenance"],
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): file_sha256(path)
            for path in [
                Path(__file__),
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/models/fixed_distance_future_supervision.py",
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/models/visual_entity_future_planning_pilot.py",
            ]
        },
        "shared_initial_state_sha256": initial_hash,
        "shared_batch_sequence_sha256": batch_sequence_hash,
        "physical_gpu": arguments.physical_gpu,
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "training_windows": len(training_windows),
        "development_windows": len(development_windows),
        "normalization_training_selected_common_observations": normalization_cpu[
            "visual"
        ]["observations"],
        "normalization_channels_at_floor": {
            name: value["channels_at_std_floor"]
            for name, value in normalization_cpu.items()
        },
        "normalization_output_convention": "Predictor outputs whitened target coordinates, fed to planner identically B-E; inverse normalization used for raw/physical forecast metrics.",
        "condition_results": [],
    }
    (output_directory / "run_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    whole_start_time = time.perf_counter()
    training_total_seconds = 0.0
    supervision_count_hashes = []
    for condition in configuration["conditions"]:
        condition_directory = output_directory / condition["label"]
        condition_directory.mkdir()
        model = FixedDistanceFutureSupervisionPilot().to(device)
        model.load_state_dict(shared_initial_state, strict=True)
        if state_sha256(model.state_dict()) != initial_hash:
            raise RuntimeError("condition initial weights differ")
        model.configure_active_modules(condition["enable_future_branch"])
        active_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer_configuration = configuration["optimizer"]
        optimizer = torch.optim.AdamW(
            active_parameters,
            lr=optimizer_configuration["learning_rate"],
            weight_decay=optimizer_configuration["weight_decay"],
        )
        torch.cuda.reset_peak_memory_stats(device)
        condition_result = {
            "condition": condition,
            "shared_initial_state_sha256": initial_hash,
            "batch_sequence_sha256": batch_sequence_hash,
            "stored_scaffold_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "active_trainable_parameters": sum(
                parameter.numel() for parameter in active_parameters
            ),
            "active_predictor_parameters": sum(
                parameter.numel()
                for parameter in model.future_predictor.parameters()
                if parameter.requires_grad
            ),
            "active_planner_parameters": sum(
                parameter.numel()
                for parameter in model.ego_planner.parameters()
                if parameter.requires_grad
            ),
            "active_selector_parameters": 0,
            "process_gpu_memory_sample_megabytes": [],
            "evaluations": [],
        }
        if not arguments.profile_only:
            for split_name, split_windows in (
                ("train", training_windows),
                ("development", development_windows),
            ):
                summary, per_window = evaluate_model(
                    model,
                    split_windows,
                    normalization,
                    condition,
                    device,
                    configuration["batch_size"],
                )
                condition_result["evaluations"].append(
                    {"update": 0, "split": split_name, **summary}
                )
                condition_result["process_gpu_memory_sample_megabytes"].append(
                    own_process_gpu_memory_megabytes()
                )
                (condition_directory / f"{split_name}_update_000.json").write_text(
                    json.dumps(per_window, indent=2) + "\n"
                )
        condition_training_seconds, update_rows = 0.0, []
        max_updates = (
            1 if arguments.profile_only else configuration["optimizer_updates"]
        )
        for update, indices in enumerate(batch_sequence[:max_updates], start=1):
            model.train()
            online, targets = collate_cached_windows(
                [training_windows[index] for index in indices], device
            )
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            update_start_time = time.perf_counter()
            output = model(
                **online, enable_future_branch=condition["enable_future_branch"]
            )
            total_loss, components, selected = compute_losses(
                output, targets, normalization, condition
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("non-finite loss")
            if update == 1:
                gradient_contract = {}
                for loss_name, loss in components.items():
                    weight = (
                        1.0
                        if loss_name == "planning"
                        else condition[f"{loss_name}_weight"]
                    )
                    gradient_contract[loss_name] = {
                        "weighted_predictor_gradient_norm": tensor_gradient_norm(
                            weight * loss, model.future_predictor.parameters()
                        )
                        if condition["enable_future_branch"]
                        else 0.0,
                        "weighted_planner_gradient_norm": tensor_gradient_norm(
                            weight * loss,
                            [
                                parameter
                                for parameter in model.ego_planner.parameters()
                                if parameter.requires_grad
                            ],
                        ),
                    }
                condition_result["first_update_gradient_contract"] = gradient_contract
            total_loss.backward()
            if not all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in active_parameters
            ):
                raise RuntimeError("non-finite gradient")
            row = {
                "update": update,
                "total_loss": float(total_loss.detach()),
                **{
                    f"{name}_loss": float(loss.detach())
                    for name, loss in components.items()
                },
                "common_valid_slot_time_count": int(selected["common_valid"].sum()),
                "visual_native_valid_slot_time_count": int(
                    selected["visual_valid"].sum()
                ),
                "spatial_native_valid_slot_time_count": int(
                    selected["spatial_valid"].sum()
                ),
                "zero_common_target_samples": int(
                    (selected["common_valid"].sum(dim=(1, 2)) == 0).sum()
                ),
                "predictor_gradient_norm_before_clip": parameter_gradient_norm(
                    model.future_predictor.parameters()
                ),
                "planner_gradient_norm_before_clip": parameter_gradient_norm(
                    model.ego_planner.parameters()
                ),
            }
            row["global_gradient_norm_before_clip"] = float(
                torch.nn.utils.clip_grad_norm_(
                    active_parameters, optimizer_configuration["gradient_clip_norm"]
                )
            )
            optimizer.step()
            torch.cuda.synchronize(device)
            row["update_wall_seconds"] = time.perf_counter() - update_start_time
            condition_training_seconds += row["update_wall_seconds"]
            update_rows.append(row)
            if (
                condition_training_seconds
                > configuration["time_limits_seconds"]["per_condition_training"]
                or training_total_seconds + condition_training_seconds
                > configuration["time_limits_seconds"]["all_training"]
            ):
                raise RuntimeError("training protection time cap exceeded")
            if update % 25 == 0 or arguments.profile_only:
                print(
                    f"TRAIN {condition['label']} update={update} loss={row['total_loss']:.6f} seconds={condition_training_seconds:.2f}",
                    flush=True,
                )
            if (
                not arguments.profile_only
                and update in configuration["evaluation_updates"]
            ):
                for split_name, split_windows in (
                    ("train", training_windows),
                    ("development", development_windows),
                ):
                    summary, per_window = evaluate_model(
                        model,
                        split_windows,
                        normalization,
                        condition,
                        device,
                        configuration["batch_size"],
                    )
                    condition_result["evaluations"].append(
                        {"update": update, "split": split_name, **summary}
                    )
                    condition_result["process_gpu_memory_sample_megabytes"].append(
                        own_process_gpu_memory_megabytes()
                    )
                    (
                        condition_directory / f"{split_name}_update_{update:03d}.json"
                    ).write_text(json.dumps(per_window, indent=2) + "\n")
                (condition_directory / "learning_curve.json").write_text(
                    json.dumps(condition_result["evaluations"], indent=2) + "\n"
                )
        training_total_seconds += condition_training_seconds
        condition_result["process_gpu_memory_sample_megabytes"].append(
            own_process_gpu_memory_megabytes()
        )
        condition_result.update(
            {
                "optimizer_updates": len(update_rows),
                "sample_draws": len(update_rows) * configuration["batch_size"],
                "training_update_wall_seconds": condition_training_seconds,
                "mean_training_update_wall_seconds": condition_training_seconds
                / len(update_rows),
                "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(device),
                "supervision_count_sequence_sha256": hashlib.sha256(
                    json.dumps(
                        [row["common_valid_slot_time_count"] for row in update_rows]
                    ).encode()
                ).hexdigest(),
            }
        )
        for module_name in ("future_predictor", "ego_planner", "entity_scorer"):
            difference = sum(
                (tensor.cpu() - shared_initial_state[name]).double().square().sum()
                for name, tensor in model.state_dict().items()
                if name.startswith(module_name + ".")
            )
            condition_result[f"{module_name}_parameter_update_l2"] = float(
                difference.sqrt()
            )
        supervision_count_hashes.append(
            condition_result["supervision_count_sequence_sha256"]
        )
        if not arguments.profile_only:
            condition_result["development_future_branch_dependence"] = (
                future_branch_dependence(
                    model,
                    development_windows,
                    condition,
                    device,
                    configuration["batch_size"],
                )
            )
            torch.save(
                {
                    "model": model.cpu().state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "normalization": normalization_cpu,
                    "configuration": configuration,
                    "condition": condition,
                    "optimizer_updates": len(update_rows),
                    "shared_initial_state_sha256": initial_hash,
                    "cache_index_sha256": provenance["cache_index_sha256"],
                },
                condition_directory / "last_update_200.pt",
            )
        (condition_directory / "training_updates.json").write_text(
            json.dumps(update_rows, indent=2) + "\n"
        )
        (condition_directory / "condition_result.json").write_text(
            json.dumps(condition_result, indent=2) + "\n"
        )
        provenance["condition_results"].append(condition_result)
        (
            output_directory
            / (
                "profile_report.json"
                if arguments.profile_only
                else "comparison_report.json"
            )
        ).write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"CONDITION_DONE {condition['label']}", flush=True)
        del model, optimizer, output, total_loss, components, active_parameters
        torch.cuda.empty_cache()
    if len(set(supervision_count_hashes)) != 1:
        raise RuntimeError(
            "conditions used different selected common supervision counts"
        )
    provenance["same_common_supervision_count_sequence_verified"] = True
    provenance["whole_invocation_wall_seconds"] = time.perf_counter() - whole_start_time
    provenance["total_training_update_wall_seconds"] = training_total_seconds
    provenance["status"] = (
        "profile_complete"
        if arguments.profile_only
        else "all_five_conditions_completed_200_updates"
    )
    (
        output_directory
        / (
            "profile_report.json"
            if arguments.profile_only
            else "comparison_report.json"
        )
    ).write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        "TRAIN_PROFILE_DONE" if arguments.profile_only else "TARGET_COMPARISON_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
