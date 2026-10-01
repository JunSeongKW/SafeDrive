"""CPU references on the preserved cache; train-only ridge and metric-unit audit."""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import train_target_supervision_ablation as exploration
from diagnose_visual_future_prediction_variance import load_existing_experiment

sys.path.insert(0, str(exploration.PROJECT_ROOT / "src"))
from planning_aware_future_prediction.models.fixed_distance_future_supervision import (
    select_nearest_current_entities,
)
from planning_aware_future_prediction.models.kinematic_future_priors import (
    predict_ego_kinematic_trajectory,
    predict_entity_constant_velocity_state,
)


def scene_macro_xy_ade(predicted_xy, target_xy, metadata):
    errors = np.linalg.norm(predicted_xy - target_xy, axis=-1).mean(axis=-1)
    scene_values = defaultdict(list)
    for value, row in zip(errors, metadata):
        scene_values[(row["recording_group"], row["scene_token"])].append(float(value))
    return float(np.mean([np.mean(values) for values in scene_values.values()]))


def fit_ridge_from_training(current_status, target_xy, ridge_lambda):
    feature_mean = current_status.mean(axis=0)
    feature_std = np.maximum(current_status.std(axis=0), 1e-8)
    design = np.column_stack(
        ((current_status - feature_mean) / feature_std, np.ones(len(current_status)))
    )
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ target_xy.reshape(len(current_status), -1),
    )
    return {
        "mean": feature_mean,
        "std": feature_std,
        "coefficients": coefficients,
        "lambda": ridge_lambda,
    }


def apply_fitted_ridge(fitted, current_status):
    design = np.column_stack(
        (
            (current_status - fitted["mean"]) / fitted["std"],
            np.ones(len(current_status)),
        )
    )
    return (design @ fitted["coefficients"]).reshape(len(current_status), 8, 2)


def select_ridge_lambda_train_only(training_windows, lambda_candidates):
    status = np.stack(
        [
            window["online_inputs"]["current_ego_status"].numpy()
            for window in training_windows
        ]
    ).astype(np.float64)
    target = np.stack(
        [
            window["training_targets"]["ego_trajectory_target"].numpy()[..., :2]
            for window in training_windows
        ]
    ).astype(np.float64)
    metadata = [window["metadata"] for window in training_windows]
    groups = sorted({row["recording_group"] for row in metadata})
    fold_by_group = {group: index % 3 for index, group in enumerate(groups)}
    fold_id = np.array([fold_by_group[row["recording_group"]] for row in metadata])
    rows = []
    for ridge_lambda in lambda_candidates:
        fold_scores = []
        for fold in range(3):
            fit_mask, validation_mask = fold_id != fold, fold_id == fold
            fitted = fit_ridge_from_training(
                status[fit_mask], target[fit_mask], ridge_lambda
            )
            fold_scores.append(
                scene_macro_xy_ade(
                    apply_fitted_ridge(fitted, status[validation_mask]),
                    target[validation_mask],
                    [
                        row
                        for index, row in enumerate(metadata)
                        if validation_mask[index]
                    ],
                )
            )
        rows.append(
            {
                "lambda": ridge_lambda,
                "fold_scene_macro_ade_meters": fold_scores,
                "mean_train_cv_ade_meters": float(np.mean(fold_scores)),
            }
        )
    chosen = min(
        rows, key=lambda row: (row["mean_train_cv_ade_meters"], row["lambda"])
    )["lambda"]
    return fit_ridge_from_training(status, target, chosen), {
        "selection_split": "train only",
        "method": "3 recording-group folds, standardization re-fitted within each training fold",
        "fold_by_recording": fold_by_group,
        "candidates": rows,
        "chosen_lambda": chosen,
        "description": "train-fitted simple learned model, NOT an untrained physical reference",
    }


def prediction_error_metrics(
    raw_prediction, raw_target, common_valid, normalization_std, target_name
):
    observations = []
    horizon_rows = []
    for future_step in range(raw_target.shape[-2]):
        valid = common_valid[:, future_step]
        count = int(valid.sum())
        row = {"future_seconds": (future_step + 1) * 0.5, "valid_count": count}
        if count:
            prediction, target = (
                raw_prediction[:, future_step][valid],
                raw_target[:, future_step][valid],
            )
            error = prediction - target
            row["normalized_mse"] = float((error / normalization_std).square().mean())
            observations.append(
                (
                    float((error / normalization_std).square().sum()),
                    count * raw_target.shape[-1],
                )
            )
            if target_name == "spatial":
                row["position_mean_error_meters"] = float(
                    error[:, :2].norm(dim=-1).mean() * 40
                )
                row["position_rmse_meters"] = float(
                    error[:, :2].square().sum(dim=-1).mean().sqrt() * 40
                )
                row["velocity_mean_error_meters_per_second"] = float(
                    error[:, 4:6].norm(dim=-1).mean() * 10
                )
                row["vx_mae_meters_per_second"] = float(error[:, 4].abs().mean() * 10)
                row["vy_mae_meters_per_second"] = float(error[:, 5].abs().mean() * 10)
                row["speed_mae_meters_per_second"] = float(
                    (prediction[:, 4:6].norm(dim=-1) - target[:, 4:6].norm(dim=-1))
                    .abs()
                    .mean()
                    * 10
                )
                yaw_difference = torch.atan2(
                    prediction[:, 2], prediction[:, 3]
                ) - torch.atan2(target[:, 2], target[:, 3])
                row["heading_mae_radians"] = float(
                    torch.atan2(yaw_difference.sin(), yaw_difference.cos()).abs().mean()
                )
        horizon_rows.append(row)
    return {
        "valid_slot_times": int(common_valid.sum()),
        "normalized_mse": sum(row[0] for row in observations)
        / max(sum(row[1] for row in observations), 1),
        "per_horizon": horizon_rows,
    }


def aggregate_prediction_rows(rows):
    report = {}
    for target_name in ("visual", "spatial"):
        for mask_name in ("common_valid", "native_valid"):
            key = f"{target_name}_{mask_name}"
            selected_rows = [row[key] for row in rows if key in row]
            if not selected_rows:
                continue
            count = sum(row["valid_slot_times"] for row in selected_rows)
            report[key] = {
                "valid_slot_times": count,
                "normalized_mse": sum(
                    row["normalized_mse"] * row["valid_slot_times"]
                    for row in selected_rows
                )
                / max(count, 1),
                "per_horizon": [],
            }
            for future_step in range(8):
                horizons = [row["per_horizon"][future_step] for row in selected_rows]
                horizon_count = sum(row["valid_count"] for row in horizons)
                horizon = {
                    "future_seconds": (future_step + 1) * 0.5,
                    "valid_count": horizon_count,
                }
                if horizon_count:
                    metric_names = set().union(*(row.keys() for row in horizons)) - {
                        "future_seconds",
                        "valid_count",
                    }
                    for metric in metric_names:
                        if metric == "position_rmse_meters":
                            horizon[metric] = float(
                                np.sqrt(
                                    sum(
                                        row.get(metric, 0) ** 2 * row["valid_count"]
                                        for row in horizons
                                    )
                                    / horizon_count
                                )
                            )
                        else:
                            horizon[metric] = (
                                sum(
                                    row.get(metric, 0) * row["valid_count"]
                                    for row in horizons
                                )
                                / horizon_count
                            )
                report[key]["per_horizon"].append(horizon)
    return report


@torch.no_grad()
def evaluate_reference_methods(
    windows, normalization, ridge, model=None, condition=None
):
    method_rows = defaultdict(list)
    for start in range(0, len(windows), 8):
        batch = windows[start : start + 8]
        online, targets = exploration.collate_cached_windows(batch, "cpu")
        if model is not None:
            output = model(
                **online,
                enable_future_branch=condition["enable_future_branch"],
                detach_future_for_planning=condition.get(
                    "detach_future_for_planning", False
                ),
            )
            selection = output.entity_selection
            ego_predictions = {condition["label"]: output.ego_trajectory}
        else:
            selection = select_nearest_current_entities(
                online["current_entity_features"],
                online["current_entity_valid_mask"],
                online["stable_entity_ids"],
            )
            ego_predictions = {
                name: predict_ego_kinematic_trajectory(
                    online["current_ego_status"], motion_model=name
                )
                for name in ("stationary", "constant_velocity", "constant_acceleration")
            }
            ridge_xy = torch.tensor(
                apply_fitted_ridge(ridge, online["current_ego_status"].numpy()),
                dtype=torch.float32,
            )
            ego_predictions["train_fitted_ego_status_ridge"] = torch.cat(
                (ridge_xy, torch.zeros_like(ridge_xy[..., :1])), dim=-1
            )
        selected = exploration.gather_selected_training_targets(selection, targets)
        current = selection.hard_selection_weights @ online["current_entity_features"]
        predicted_targets = {}
        if model is None:
            predicted_targets = {
                "entity_current_state_persistence": {
                    "spatial": predict_entity_constant_velocity_state(
                        current, extrapolate_position=False
                    )
                },
                "entity_constant_velocity": {
                    "spatial": predict_entity_constant_velocity_state(current)
                },
                "visual_current_roi_persistence": {
                    "visual": current[..., :1024, None]
                    .transpose(-1, -2)
                    .expand(-1, -1, 8, -1)
                },
                "visual_train_mean": {
                    "visual": normalization["visual"]["mean"].expand(
                        len(batch), 4, 8, -1
                    )
                },
            }
        elif condition["enable_future_branch"]:
            predicted_targets[condition["label"]] = {
                name: values * normalization[name]["std"] + normalization[name]["mean"]
                for name, values in (
                    ("visual", output.predicted_future_visual_latents),
                    ("spatial", output.predicted_future_spatial_states),
                )
            }
        for sample, window in enumerate(batch):
            for name, prediction in ego_predictions.items():
                error = (
                    prediction[sample, :, :2]
                    - targets["ego_trajectory_target"][sample, :, :2]
                ).norm(dim=-1)
                method_rows[name].append(
                    {
                        **window["metadata"],
                        "ade_meters": float(error.mean()),
                        "fde_meters": float(error[-1]),
                        "xy_position_errors_meters": error.tolist(),
                    }
                )
            for name, predictions in predicted_targets.items():
                if name in ego_predictions:
                    row = method_rows[name][-1]
                else:
                    row = {**window["metadata"]}
                    method_rows[name].append(row)
                for target_name, prediction in predictions.items():
                    for mask_name, mask in (
                        ("common_valid", selected["common_valid"]),
                        ("native_valid", selected[f"{target_name}_valid"]),
                    ):
                        row[f"{target_name}_{mask_name}"] = prediction_error_metrics(
                            prediction[sample],
                            selected[target_name][sample],
                            mask[sample],
                            normalization[target_name]["std"],
                            target_name,
                        )
    summary = {}
    for name, rows in method_rows.items():
        row = {"windows": len(rows), "prediction": aggregate_prediction_rows(rows)}
        if "ade_meters" in rows[0]:
            scenes, groups = defaultdict(list), defaultdict(list)
            for record in rows:
                scenes[(record["recording_group"], record["scene_token"])].append(
                    record["ade_meters"]
                )
                groups[record["recording_group"]].append(record["ade_meters"])
            row.update(
                scene_macro_ade_meters=float(
                    np.mean([np.mean(values) for values in scenes.values()])
                ),
                window_mean_ade_meters=float(
                    np.mean([record["ade_meters"] for record in rows])
                ),
                fde_meters=float(np.mean([record["fde_meters"] for record in rows])),
                per_recording_window_mean_ade_meters={
                    group: float(np.mean(values)) for group, values in groups.items()
                },
                scenes=len(scenes),
                recordings=len(groups),
            )
        summary[name] = row
    return summary, method_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists() or not output_directory.is_relative_to(
        exploration.PROJECT_ROOT / "outputs"
    ):
        raise ValueError("require a new project outputs directory")
    output_directory.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.perf_counter()
    configuration, windows, normalization, _ = load_existing_experiment("cpu")
    train = [window for window in windows if window["metadata"]["split"] == "train"]
    registered = json.loads(
        (
            exploration.PROJECT_ROOT
            / "configs/exploration/pilot_foundation_decision_v1.json"
        ).read_text()
    )
    ridge, ridge_record = select_ridge_lambda_train_only(
        train, registered["ridge"]["lambda_candidates"]
    )
    report = {
        "baseline_commit": "607da52",
        "ridge": ridge_record,
        "metrics": "scene-macro XY ADE in meters, 8 horizons; same recording+scene grouping as old runner",
        "unit_contract": {
            "ego_status": [
                "command0",
                "command1",
                "command2",
                "command3",
                "vx_m/s",
                "vy_m/s",
                "ax_m/s2",
                "ay_m/s2",
            ],
            "entity_tail": [
                "x/40",
                "y/40",
                "sin_heading",
                "cos_heading",
                "width/10",
                "length/10",
                "vx/10",
                "vy/10",
                "vehicle",
                "pedestrian",
            ],
            "future_spatial": [
                "x/40",
                "y/40",
                "sin_heading",
                "cos_heading",
                "vx/10",
                "vy/10",
            ],
            "frame": "fixed current rear-axle planar ego XY",
            "nominal_cadence_seconds": 0.5,
            "historical_timestamp_max_error_seconds": 0.012979,
            "past_velocity_acceleration_coordinates": "native EgoStatus fields used as current ego-frame vectors; source code/calibration audit recorded separately",
        },
        "splits": {},
    }
    fitted_path = output_directory / "train_fitted_ridge.pt"
    torch.save(
        {
            key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
            for key, value in ridge.items()
        },
        fitted_path,
    )
    conditions = {row["label"]: row for row in configuration["conditions"]}
    conditions["F"] = {
        **conditions["E"],
        "label": "F",
        "detach_future_for_planning": True,
    }
    for split in ("train", "development"):
        subset = [window for window in windows if window["metadata"]["split"] == split]
        summary, rows = evaluate_reference_methods(subset, normalization, ridge)
        for label in "ABCDEF":
            model = exploration.FixedDistanceFutureSupervisionPilot().eval()
            checkpoint_path = (
                exploration.PROJECT_ROOT
                / "outputs/future_prediction_diagnostics/bounded_followup_1000_v1"
                / f"seed29_{label}"
                / "checkpoint_update1000.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(checkpoint["model"], strict=True)
            model_summary, model_rows = evaluate_reference_methods(
                subset, normalization, ridge, model, conditions[label]
            )
            summary.update(model_summary)
            rows.update(model_rows)
        report["splits"][split] = summary
        (output_directory / f"{split}_per_window.json").write_text(
            json.dumps(rows, indent=2) + "\n"
        )
    report["wall_seconds"] = time.perf_counter() - started
    report["source_sha256"] = {
        str(
            Path(__file__).relative_to(exploration.PROJECT_ROOT)
        ): exploration.file_sha256(__file__)
    }
    report["cache_index_sha256"] = exploration.file_sha256(
        exploration.PROJECT_ROOT / configuration["cache_directory"] / "cache_index.json"
    )
    (output_directory / "reference_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "wall_seconds": report["wall_seconds"],
                "dev_ego_ade": {
                    name: row.get("scene_macro_ade_meters")
                    for name, row in report["splits"]["development"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
