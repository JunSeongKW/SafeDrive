"""Bounded, preregistered comparison on cached official Drive-JEPA features.

No held-out/navtest evaluation, no official source/weight modifications. The cache
is local and specific to this small extension experiment, not the old pilot.
"""

import argparse
import contextlib
import hashlib
import json
import os
import pickle
import signal
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from PIL import Image
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    gradient_report,
    parameter_sha256,
)

WORKSPACE = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stable_order(value):
    return hashlib.sha256(value.encode()).hexdigest()


def select_manifest_records(manifest, specification):
    """Select without future validity or scores; no held-out admission."""
    grouped = defaultdict(list)
    assignments = manifest["assignments"]["split_by_recording"]
    for record in manifest["records"]:
        recording = record["recording_group"]
        if record["split"] != assignments[recording]:
            raise ValueError("Manifest record disagrees with recording assignment")
        if recording in specification["excluded_recordings"]:
            continue
        if record["split"] in ("train", "development"):
            grouped[(record["split"], recording)].append(record)
    selected = []
    for split, count in specification["recording_counts"].items():
        if split not in ("train", "development"):
            raise ValueError("Held-out is forbidden for this comparison")
        recordings = sorted(
            [recording for entry_split, recording in grouped if entry_split == split],
            key=lambda recording: stable_order("selection-comparison-v1|" + recording),
        )[:count]
        if len(recordings) != count:
            raise ValueError("Insufficient recordings; do not silently change split")
        for recording in recordings:
            selected.extend(
                sorted(
                    grouped[(split, recording)],
                    key=lambda row: stable_order(row["current_frame_token"]),
                )[: specification["windows_per_recording"]]
            )
    return selected


def control_patch_indices(condition, batch_size, specification, generator):
    if condition.startswith("planning_conditioned"):
        return None
    if condition == "fixed_lattice":
        return (
            torch.tensor(specification["fixed_patch_indices"], dtype=torch.long)
            .expand(batch_size, -1)
            .cuda()
        )
    if condition != "seeded_random":
        raise ValueError(condition)
    return torch.stack(
        [
            torch.randperm(512, generator=generator)[
                : specification["selected_patch_budget"]
            ]
            for _ in range(batch_size)
        ]
    ).cuda()


def enforce_limits(started, specification):
    if specification.get(
        "deadline_iso"
    ) and datetime.now().astimezone() >= datetime.fromisoformat(
        specification["deadline_iso"]
    ):
        raise RuntimeError("Registered deadline reached")
    if (
        STOP_REQUESTED
        or time.perf_counter() - started > specification["maximum_wall_seconds"]
    ):
        raise RuntimeError("Stop requested or registered wall-time cap reached")
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Registered GPU allocated-memory cap exceeded")
    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
        raise RuntimeError("Shared GPU reserve below 6GiB; stopping our work")


def build_cache(
    agent,
    extension,
    official_root,
    specification,
    connection,
    output,
    started,
    reused_cache_index=None,
):
    from navsim.common.dataclasses import AgentInput, Scene

    manifest_path = WORKSPACE / specification["project_recording_split_manifest"]
    manifest = json.loads(manifest_path.read_text())
    selected = select_manifest_records(manifest, specification)
    write_json(output / "preregistered_windows.json", selected)
    official_segments = set(
        yaml.safe_load(
            (
                official_root
                / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
            ).read_text()
        )["log_names"]
    )
    navtest_tokens = set(
        json.loads(
            (
                WORKSPACE
                / "results/official_drive_jepa_reproduction/expected_scene_tokens.json"
            ).read_text()
        )["tokens"]
    )
    sensor_root = WORKSPACE / "dataset/sensor_blobs/trainval"
    log_root = WORKSPACE / "dataset/navsim_logs/trainval"
    cached_logs, source_hashes, records, skipped = {}, {}, [], []
    cache_bytes, reused_bytes, reused_count = 0, 0, 0
    reused_records = {}
    if reused_cache_index is not None:
        prior = json.loads(Path(reused_cache_index).read_text())
        if (
            prior["manifest_sha256"] != file_sha256(manifest_path)
            or not prior["cache_baseline_bitwise_equivalent"]
        ):
            raise RuntimeError("Prior cache manifest/equivalence mismatch")
        reused_records = {row["current_frame_token"]: row for row in prior["records"]}
        source_hashes.update(prior["source_hashes"])
    checked_online_equivalence = False
    (output / "feature_cache").mkdir()
    for index, record in enumerate(selected):
        enforce_limits(started, specification)
        log_path = log_root / record["segment_filename"]
        if (
            log_path.stem not in official_segments
            or record["current_frame_token"] in navtest_tokens
        ):
            raise RuntimeError("Unofficial training segment or navtest overlap")
        if record["current_frame_token"] in reused_records:
            previous = reused_records[record["current_frame_token"]]
            if any(previous[key] != value for key, value in record.items()):
                raise RuntimeError("Reused window metadata changed")
            previous_path = Path(previous["cache_file"])
            if file_sha256(previous_path) != previous["cache_sha256"]:
                raise RuntimeError("Reused cache checksum mismatch")
            cache_bytes += previous_path.stat().st_size
            if cache_bytes > specification["maximum_cache_gib"] * 2**30:
                raise RuntimeError("Registered total logical cache cap reached")
            reused_bytes += previous_path.stat().st_size
            reused_count += 1
            records.append(previous)
            continue
        if log_path.name not in cached_logs:
            with log_path.open("rb") as handle:
                cached_logs = {log_path.name: pickle.load(handle)}
            source_hashes[str(log_path)] = file_sha256(log_path)
        frames = cached_logs[log_path.name][
            record["start_index"] : record["start_index"] + 12
        ]
        if len(frames) != 12 or frames[3]["token"] != record["current_frame_token"]:
            raise RuntimeError("Manifest/frame time alignment mismatch")
        intervals = np.diff([int(frame["timestamp"]) for frame in frames]) / 1e6
        if np.any(np.abs(intervals - 0.5) > connection["timestamp_tolerance_seconds"]):
            raise RuntimeError("Cadence outside registered tolerance")
        paths = [
            sensor_root
            / next(
                camera["data_path"]
                for name, camera in frame["cams"].items()
                if name.lower() == "cam_f0"
            )
            for frame in frames
        ]
        if not all(paths[position].is_file() for position in (2, 3)):
            skipped.append(
                {
                    "token": record["current_frame_token"],
                    "reason": "missing_current_input",
                }
            )
            continue
        for path in paths[2:]:
            if path.is_file():
                source_hashes[str(path)] = file_sha256(path)
        current_input = AgentInput.from_scene_dict_list(
            frames[:4], sensor_root, 4, agent.get_sensor_config()
        )
        feature_builder = agent.get_feature_builders()[0]
        features = feature_builder.compute_features(current_input)
        current_clip = torch.stack(
            (features["camera_feature_2"], features["camera_feature_1"]), dim=1
        )[None].cuda()
        scene = Scene.from_scene_dict_list(
            frames, sensor_root, 4, 8, agent.get_sensor_config()
        )
        ego_target = (
            agent.get_target_builders()[0].compute_targets(scene)["trajectory"].float()
        )
        current_status = features["status_feature"].float()
        with torch.no_grad():
            current_latents = extension.encode_observed_clip(current_clip)
            # Real-image cached/online/original equivalence, not a synthetic substitute.
            if not checked_online_equivalence:
                encoder_outputs = []
                encoder_hook = agent._model.image_encoder.register_forward_hook(
                    lambda module, inputs, result, encoder_outputs=encoder_outputs: (
                        encoder_outputs.append(result.detach())
                    )
                )
                original = agent._model(current_clip, current_status[None].cuda())[
                    "trajectory"
                ]
                encoder_hook.remove()
                cached = extension.forward_from_current_patch_latents(
                    current_latents, current_status[None].cuda()
                )["trajectory"]
                disabled = extension.forward_from_current_patch_latents(
                    current_latents, current_status[None].cuda(), False
                )["trajectory"]
                write_json(
                    output / "cached_path_equivalence.json",
                    {
                        "encoder_max_abs_difference": float(
                            (current_latents - encoder_outputs[0]).abs().max()
                        ),
                        "enabled_max_abs_difference": float(
                            (original - cached).abs().max()
                        ),
                        "disabled_max_abs_difference": float(
                            (original - disabled).abs().max()
                        ),
                        "original_trajectory": original.cpu().tolist(),
                        "cached_trajectory": cached.cpu().tolist(),
                        "current_latent_stride": list(current_latents.stride()),
                        "original_latent_stride": list(encoder_outputs[0].stride()),
                    },
                )
                if not torch.equal(original, cached) or not torch.equal(
                    original, disabled
                ):
                    raise RuntimeError(
                        "Cached path failed bitwise baseline equivalence"
                    )
                checked_online_equivalence = True
            future_latents, future_valid = [], []
            for pair in connection["future_frame_pairs"]:
                valid = all(paths[position].is_file() for position in pair)
                future_valid.append(valid)
                if valid:
                    cameras = []
                    for position in pair:
                        with Image.open(paths[position]) as image:
                            cameras.append(
                                SimpleNamespace(
                                    cam_f0=SimpleNamespace(
                                        image=np.array(image.convert("RGB"))
                                    )
                                )
                            )
                    latest, earlier = feature_builder._get_camera_feature(
                        SimpleNamespace(cameras=cameras)
                    )
                    clip = torch.stack((earlier, latest), dim=1)[None].cuda()
                    future_latents.append(extension.encode_observed_clip(clip)[0].cpu())
                else:
                    future_latents.append(torch.zeros(512, 1024))
        tensors = {
            "current_patch_latents": current_latents[0].cpu(),
            "current_ego_status": current_status,
            "ego_trajectory_target": ego_target,
            "future_target_latents": torch.stack(future_latents),
            "future_target_valid_mask": torch.tensor(future_valid, dtype=torch.bool)[
                :, None
            ]
            .expand(-1, 512)
            .clone(),
        }
        cache_path = output / "feature_cache" / f"{record['current_frame_token']}.pt"
        projected_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        if (
            cache_bytes + projected_bytes + 16384
            > specification["maximum_cache_gib"] * 2**30
        ):
            raise RuntimeError("Registered cache disk cap reached")
        torch.save(tensors, cache_path)
        cache_bytes += cache_path.stat().st_size
        records.append(
            {
                **record,
                "future_tubelet_valid": future_valid,
                "cache_file": str(cache_path),
                "cache_sha256": file_sha256(cache_path),
                "actual_frame_intervals_seconds": intervals.tolist(),
            }
        )
        write_json(output / "cache_index.partial.json", records)
        if (index + 1) % 16 == 0:
            print(f"CACHE {index + 1}/{len(selected)} bytes={cache_bytes}", flush=True)
    # Reuse the source manifest's non-overlap contract and recheck selected intervals.
    for recording in {row["recording_group"] for row in records}:
        ordered = sorted(
            [row for row in records if row["recording_group"] == recording],
            key=lambda row: row["history_begin_timestamp_us"],
        )
        if any(
            left["future_end_timestamp_us"] >= right["history_begin_timestamp_us"]
            for left, right in zip(ordered, ordered[1:])  # noqa: RUF007 -- Python 3.9
        ):
            raise RuntimeError("Overlapping source recording windows")
    for split, expected in specification["recording_counts"].items():
        actual = {row["recording_group"] for row in records if row["split"] == split}
        if len(actual) != expected:
            raise RuntimeError(
                "Current-file filtering reduced preregistered recording count"
            )
    report = {
        "records": records,
        "skipped": skipped,
        "cache_bytes": cache_bytes,
        "reused_cache_bytes": reused_bytes,
        "reused_cache_window_count": reused_count,
        "new_cache_bytes": cache_bytes - reused_bytes,
        "source_hashes": source_hashes,
        "manifest_sha256": file_sha256(manifest_path),
        "future_validity_used_for_selection": False,
        "cache_baseline_bitwise_equivalent": True,
        "counts": dict(Counter(row["split"] for row in records)),
        "context_counts": {
            split: dict(
                Counter(
                    str(row["command_raw_index"])
                    for row in records
                    if row["split"] == split
                )
            )
            for split in ("train", "development")
        },
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(output / "cache_index.json", report)
    loaded = [
        torch.load(row["cache_file"], map_location="cpu", weights_only=True)
        for row in records
    ]
    cache = {key: torch.stack([row[key] for row in loaded]) for key in loaded[0]}
    return report, cache


def get_training_batch(cache, indices):
    return {key: tensor[indices].cuda() for key, tensor in cache.items()}


def summarize_rows(rows, predictions, targets):
    from navsim.agents.drive_jepa_perception_free.drive_jepa_agent import (
        l1_length_normalized_loss,
    )

    recording_values, scene_values = defaultdict(list), defaultdict(list)
    for row in rows:
        recording_values[row["recording"]].append(row["xy_ade_m"])
        scene_values[row["scene_token"]].append(row["xy_ade_m"])
    valid_count = sum(row.get("supervised_patch_times", 0) for row in rows)
    result = {
        "window_count": len(rows),
        "recording_count": len(recording_values),
        "scene_count": len(scene_values),
        "window_mean_xy_ade_m": float(np.mean([row["xy_ade_m"] for row in rows])),
        "scene_macro_xy_ade_m": float(
            np.mean([np.mean(values) for values in scene_values.values()])
        ),
        "recording_macro_xy_ade_m": float(
            np.mean([np.mean(values) for values in recording_values.values()])
        ),
        "official_il_loss_whole_split": float(
            l1_length_normalized_loss(predictions, targets, alpha=5)
        ),
        "supervised_patch_times": valid_count,
        "command_window_ade": {
            str(command): {
                "count": sum(row["command"] == command for row in rows),
                "mean": float(
                    np.mean(
                        [row["xy_ade_m"] for row in rows if row["command"] == command]
                    )
                ),
            }
            for command in sorted({row["command"] for row in rows})
        },
    }
    if valid_count:
        for key in ("future_mse", "persistence_mse"):
            result[key] = (
                sum(row[key] * row["supervised_patch_times"] for row in rows)
                / valid_count
            )
    return result


@torch.no_grad()
def evaluate(extension, cache, records, split, condition, specification, seed):
    extension.eval()
    evaluation_generator = torch.Generator().manual_seed(seed + 100000)
    selected_indices = [
        index for index, row in enumerate(records) if row["split"] == split
    ]
    rows, predictions, targets = [], [], []
    for offset in range(0, len(selected_indices), specification["batch_size"]):
        indices = selected_indices[offset : offset + specification["batch_size"]]
        batch = get_training_batch(cache, indices)
        control_ids = (
            None
            if condition == "original_no_branch"
            else control_patch_indices(
                condition, len(indices), specification, evaluation_generator
            )
        )
        result = extension.forward_from_current_patch_latents(
            batch["current_patch_latents"],
            batch["current_ego_status"],
            condition != "original_no_branch",
            selected_patch_indices=control_ids,
        )
        prediction = result["trajectory"]
        target = batch["ego_trajectory_target"]
        ade = (prediction[..., :2] - target[..., :2]).norm(dim=-1).mean(dim=-1)
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        for position, index in enumerate(indices):
            metadata = records[index]
            row = {
                "token": metadata["current_frame_token"],
                "recording": metadata["recording_group"],
                "scene_token": metadata["scene_token"],
                "command": metadata["command_raw_index"],
                "xy_ade_m": float(ade[position]),
            }
            if condition != "original_no_branch":
                selected_ids = result["patch_selection"].selected_patch_indices[
                    position
                ]
                selected_target = batch["future_target_latents"][
                    position, :, selected_ids
                ].transpose(0, 1)
                valid = batch["future_target_valid_mask"][
                    position, :, selected_ids
                ].transpose(0, 1)
                predicted = result["predicted_future_latents"][position]
                persistent = batch["current_patch_latents"][
                    position, selected_ids, None
                ]
                per_patch_error = (predicted - selected_target).square().mean(dim=-1)
                persistence_error = (persistent - selected_target).square().mean(dim=-1)
                row.update(
                    {
                        "selected_patch_ids": selected_ids.cpu().tolist(),
                        "supervised_patch_times": int(valid.sum()),
                        "future_mse": float(per_patch_error[valid].mean())
                        if valid.any()
                        else 0.0,
                        "persistence_mse": float(persistence_error[valid].mean())
                        if valid.any()
                        else 0.0,
                    }
                )
            rows.append(row)
    return {
        "summary": summarize_rows(rows, torch.cat(predictions), torch.cat(targets)),
        "windows": rows,
    }


def extension_state(extension):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in extension.state_dict().items()
        if not name.startswith("baseline_model.")
    }


@torch.no_grad()
def future_dependence_diagnostic(
    extension, cache, records, specification, condition, seed
):
    """Prediction-content intervention only; coordinates/current memory unchanged."""
    extension.eval()
    generator = torch.Generator().manual_seed(seed + 100000)
    indices = [
        index for index, row in enumerate(records) if row["split"] == "development"
    ]
    changes = {mode: [] for mode in ("zero", "within_batch_slot_swap")}
    for offset in range(0, len(indices), specification["batch_size"]):
        batch = get_training_batch(
            cache, indices[offset : offset + specification["batch_size"]]
        )
        control_ids = control_patch_indices(
            condition, batch["current_ego_status"].shape[0], specification, generator
        )

        def forward(batch=batch, control_ids=control_ids):
            return extension.forward_from_current_patch_latents(
                batch["current_patch_latents"],
                batch["current_ego_status"],
                selected_patch_indices=control_ids,
            )

        normal = forward()
        original_future = normal["predicted_future_latents"]
        for mode, measurements in changes.items():
            replacement = (
                torch.zeros_like(original_future)
                if mode == "zero"
                else original_future.roll(1, dims=0)
            )
            handle = extension.future_predictor.register_forward_hook(
                lambda module, inputs, output, replacement=replacement: replacement
            )
            try:
                intervened = forward()
            finally:
                handle.remove()
            feature_change = (
                (replacement - original_future).square().mean(dim=(1, 2, 3)).sqrt()
            )
            trajectory_change = (
                (intervened["trajectory"][..., :2] - normal["trajectory"][..., :2])
                .norm(dim=-1)
                .mean(dim=1)
            )
            measurements.extend(
                [
                    {
                        "future_input_rms_change": float(left),
                        "trajectory_xy_mean_change_m": float(right),
                    }
                    for left, right in zip(feature_change, trajectory_change)
                ]
            )
    return {
        mode: {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
        for mode, rows in changes.items()
    }


def train_condition(
    agent, cache, records, specification, condition, seed, output, started
):
    from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
        DriveJEPASelectivePatchFuture,
    )

    condition_started = time.perf_counter()
    torch.manual_seed(seed)
    extension = DriveJEPASelectivePatchFuture(agent._model).cuda()
    initial_state = extension_state(extension)
    initial_hashes = {
        name: parameter_sha256(module)
        for name, module in (
            ("selector", extension.patch_selector),
            ("predictor", extension.future_predictor),
            ("bridge", extension.future_bridge),
        )
    }
    if not condition.startswith("planning_conditioned"):
        extension.patch_selector.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in extension.parameters() if parameter.requires_grad],
        lr=specification["learning_rate"],
        weight_decay=specification["weight_decay"],
    )
    active_parameters = sum(
        parameter.numel()
        for parameter in extension.parameters()
        if parameter.requires_grad
    )
    train_indices = torch.tensor(
        [index for index, row in enumerate(records) if row["split"] == "train"]
    )
    sampler = torch.Generator().manual_seed(seed + 200000)
    batch_schedule = train_indices[
        torch.randint(
            len(train_indices),
            (specification["updates"], specification["batch_size"]),
            generator=sampler,
        )
    ]
    random_selector = torch.Generator().manual_seed(seed + 300000)
    auxiliary_weight = (
        0.0
        if condition.endswith("no_auxiliary")
        else specification["future_auxiliary_weight"]
    )
    report = {
        "condition": condition,
        "seed": seed,
        "initial_module_hashes": initial_hashes,
        "batch_schedule_sha256": hashlib.sha256(
            batch_schedule.numpy().tobytes()
        ).hexdigest(),
        "active_trainable_parameters": active_parameters,
        "auxiliary_weight": auxiliary_weight,
        "curve": [],
        "evaluations": {},
        "gradient_checks": {},
    }
    condition_root = output / f"{condition}_seed{seed}"
    condition_root.mkdir()
    completed_updates = 0
    try:
        for update in range(specification["updates"] + 1):
            enforce_limits(started, specification)
            if (
                time.perf_counter() - condition_started
                > specification["maximum_condition_wall_seconds"]
            ):
                raise RuntimeError("Registered per-condition time cap reached")
            if update in specification["evaluation_updates"]:
                report["evaluations"][str(update)] = {
                    split: evaluate(
                        extension, cache, records, split, condition, specification, seed
                    )
                    for split in ("train", "development")
                }
                write_json(condition_root / "progress.json", report)
                print(
                    f"EVAL {condition} seed={seed} update={update} dev_ADE={report['evaluations'][str(update)]['development']['summary']['scene_macro_xy_ade_m']:.6f}",
                    flush=True,
                )
            if update == specification["updates"]:
                break
            extension.train()
            batch = get_training_batch(cache, batch_schedule[update])
            control_ids = control_patch_indices(
                condition, specification["batch_size"], specification, random_selector
            )
            optimizer.zero_grad(set_to_none=True)
            result = extension.forward_from_current_patch_latents(
                batch["current_patch_latents"],
                batch["current_ego_status"],
                selected_patch_indices=control_ids,
            )
            planning_loss = agent.compute_loss(
                {}, {"trajectory": batch["ego_trajectory_target"]}, result
            )
            auxiliary_loss = (
                extension.compute_future_auxiliary_loss(
                    result,
                    batch["current_ego_status"],
                    batch["future_target_latents"],
                    batch["future_target_valid_mask"],
                )
                if auxiliary_weight
                else planning_loss.new_zeros(())
            )
            loss = planning_loss + auxiliary_weight * auxiliary_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            if update in (0, 1, specification["updates"] - 1):
                report["gradient_checks"][str(update + 1)] = gradient_report(extension)
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in extension.parameters()
                    if parameter.requires_grad
                ],
                specification["gradient_clip_norm"],
                error_if_nonfinite=True,
            )
            optimizer.step()
            completed_updates = update + 1
            report["curve"].append(
                {
                    "update": completed_updates,
                    "planning_loss": float(planning_loss.detach()),
                    "auxiliary_loss": float(auxiliary_loss.detach()),
                    "supervised_patch_times": int(
                        batch["future_target_valid_mask"][:, :, 0].sum()
                    )
                    * 4,
                }
            )
            del result, loss, planning_loss, auxiliary_loss, batch
        report["future_content_interventions"] = future_dependence_diagnostic(
            extension, cache, records, specification, condition, seed
        )
    finally:
        report["completed_updates"] = completed_updates
        report["wall_seconds"] = time.perf_counter() - condition_started
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        state = extension_state(extension)
        report["module_parameter_change_l2"] = {
            prefix: float(
                sum(
                    (value - initial_state[name]).square().sum()
                    for name, value in state.items()
                    if name.startswith(prefix)
                )
                ** 0.5
            )
            for prefix in ("patch_selector", "future_predictor", "future_bridge")
        }
        if str(specification["updates"]) in report["evaluations"]:
            before = report["evaluations"]["0"]["development"]["windows"]
            after = report["evaluations"][str(specification["updates"])]["development"][
                "windows"
            ]
            report["development_selected_set_changed_fraction"] = float(
                np.mean(
                    [
                        set(left["selected_patch_ids"])
                        != set(right["selected_patch_ids"])
                        for left, right in zip(before, after)
                    ]
                )
            )
            frequencies = Counter(
                identifier for row in after for identifier in row["selected_patch_ids"]
            )
            report["development_selected_patch_histogram"] = dict(frequencies)
        torch.save(
            {
                "extension_state_dict": state,
                "optimizer_state_dict": optimizer.state_dict(),
                "completed_updates": completed_updates,
                "batch_schedule": batch_schedule,
                "random_selector_generator_state": random_selector.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "specification": specification,
                "condition": condition,
                "seed": seed,
            },
            condition_root / "extension_checkpoint.pt",
        )
        write_json(condition_root / "results.json", report)
    return report


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--specification",
        type=Path,
        default=WORKSPACE
        / "configs/drive_jepa_selective_future/selection_comparison_v1.json",
    )
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    specification = json.loads(arguments.specification.read_text())
    connection = json.loads(
        (WORKSPACE / specification["base_connection_specification"]).read_text()
    )
    if output.exists():
        raise RuntimeError("Fresh output directory required; preserve prior runs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Select exactly one approved physical GPU")
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient GPU free memory")
    output.mkdir(parents=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    torch.set_num_threads(1)
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate

    from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
        DriveJEPASelectivePatchFuture,
    )

    baseline_spec, assets, configuration, official_root = official_configuration(
        WORKSPACE
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official_root, text=True
    ).strip()
    source_diff = subprocess.check_output(["git", "diff"], cwd=official_root)
    if source_commit != baseline_spec["official_source_commit"]:
        raise RuntimeError("Official source changed")
    for name in ("planning_checkpoint", "initialization_encoder"):
        if file_sha256(assets[name]["path"]) != assets[name]["sha256"]:
            raise RuntimeError("Official weight hash mismatch")
    with (
        (output / "strict_loading.log").open("w") as loading_log,
        contextlib.redirect_stdout(loading_log),
    ):
        agent = instantiate(configuration.agent)
        checkpoint = torch.load(
            configuration.agent.checkpoint_path, map_location="cpu", mmap=True
        )
        loading = agent.load_state_dict(
            {
                name.replace("agent.", ""): value
                for name, value in checkpoint["state_dict"].items()
            },
            strict=True,
        )
        del checkpoint
    agent.eval().cuda()
    baseline_hash = parameter_sha256(agent._model)
    torch.manual_seed(29)
    extension = DriveJEPASelectivePatchFuture(agent._model).cuda().eval()
    cache_report, cache = build_cache(
        agent, extension, official_root, specification, connection, output, started
    )
    baseline = {
        split: evaluate(
            extension,
            cache,
            cache_report["records"],
            split,
            "original_no_branch",
            specification,
            29,
        )
        for split in ("train", "development")
    }
    write_json(output / "original_baseline.json", baseline)
    del extension
    reports = []
    for seed in specification["seeds"]:
        for condition in specification["conditions"]:
            torch.cuda.reset_peak_memory_stats()
            reports.append(
                train_condition(
                    agent,
                    cache,
                    cache_report["records"],
                    specification,
                    condition,
                    seed,
                    output,
                    started,
                )
            )
    if parameter_sha256(agent._model) != baseline_hash:
        raise RuntimeError("Frozen baseline changed")
    if subprocess.check_output(["git", "diff"], cwd=official_root) != source_diff:
        raise RuntimeError("Official source diff changed")
    for seed in specification["seeds"]:
        matched = [report for report in reports if report["seed"] == seed]
        if any(
            report["initial_module_hashes"] != matched[0]["initial_module_hashes"]
            or report["batch_schedule_sha256"] != matched[0]["batch_schedule_sha256"]
            for report in matched
        ):
            raise RuntimeError("Unmatched initialization or batch schedule")
    summary = {
        "specification": specification,
        "official_source_commit": source_commit,
        "assets": assets,
        "strict_loading": str(loading),
        "baseline_hash_before_after": baseline_hash,
        "source_sha256": {
            str(path.relative_to(WORKSPACE)): file_sha256(path)
            for path in (
                Path(__file__),
                arguments.specification.resolve(),
                WORKSPACE
                / "src/planning_aware_future_prediction/models/drive_jepa_selective_patch_future.py",
            )
        },
        "cache": {
            key: value
            for key, value in cache_report.items()
            if key not in ("records", "source_hashes")
        },
        "baseline": {split: value["summary"] for split, value in baseline.items()},
        "conditions": [
            {
                key: value
                for key, value in report.items()
                if key not in ("curve", "evaluations")
            }
            | {
                "final": {
                    split: value["summary"]
                    for split, value in report["evaluations"]["200"].items()
                }
            }
            for report in reports
        ],
        "total_wall_seconds": time.perf_counter() - started,
        "initialization_and_batch_matching_verified": True,
        "held_out_and_navtest_evaluation_performed": False,
    }
    write_json(output / "comparison_results.json", summary)
    print(
        f"COMPARISON_COMPLETE seconds={summary['total_wall_seconds']:.2f}", flush=True
    )


if __name__ == "__main__":
    main()
