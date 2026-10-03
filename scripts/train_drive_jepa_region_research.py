"""Preregistered located/sparse/history-mask/motion controls on frozen Drive-JEPA."""

import argparse
import hashlib
import json
import math
import os
import pickle
from types import SimpleNamespace

import numpy as np
from PIL import Image
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from diagnose_drive_jepa_learning_limitations import load_official_agent, restore_delta
from run_drive_jepa_architecture_followup import (
    WORKSPACE,
    build_model,
    module_hashes,
    observed_forward,
)
from run_drive_jepa_selection_comparison import (
    get_training_batch,
    summarize_rows,
    write_json,
)
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    parameter_sha256,
)

from planning_aware_future_prediction.models.spatial_region_future import (
    SpatialRegionSelector,
    compute_region_future_auxiliary_loss,
    pooled_future_targets,
)

from planning_aware_future_prediction.models.region_future_research import (
    LocatedRegionFutureBridge, compute_history_reconstruction_losses,
    observed_history_motion_scores,
)

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def guard(specification, started, run_started=None):
    if STOP_REQUESTED:
        raise RuntimeError("Requested stop; preserve our checkpoint")
    if time.perf_counter() - started > specification["maximum_wall_seconds"]:
        raise RuntimeError("Registered total runtime cap")
    if (
        run_started
        and time.perf_counter() - run_started
        > specification["maximum_condition_wall_seconds"]
    ):
        raise RuntimeError("Registered condition runtime cap")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError("Shared GPU reserve reached; no automatic retry")
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Our GPU allocation cap reached")


def set_selection_control(model, records, indices, options, seed):
    explicit = None
    if options["selection_mode"] == "random":
        selected = []
        for index in indices:
            # Same current window and seed always yields the same random control.
            digest = hashlib.sha256(
                f"region-control:{seed}:{records[index]['current_frame_token']}".encode()
            ).digest()
            generator = torch.Generator().manual_seed(
                int.from_bytes(digest[:8], "little") % (2**63 - 1)
            )
            selected.append(
                torch.randperm(model.patch_selector.region_count, generator=generator)[
                    : options["region_budget"]
                ]
            )
        explicit = torch.stack(selected).cuda()
    if options["selection_mode"] == "motion":
        # Ties are resolved by native region ID, without targets or outcome labels.
        explicit = torch.tensor([sorted(range(model.patch_selector.region_count),
                                        key=lambda region: (-records[index]["observed_motion_region_scores"][region], region))
                                 [:options["region_budget"]] for index in indices], device="cuda")
        # Same hash-random fallback as the random control if observed history is absent.
        for position, index in enumerate(indices):
            if not records[index]["earlier_history_available"]:
                fallback_options = dict(options, selection_mode="random")
                set_selection_control(model, records, [index], fallback_options, seed)
                explicit[position] = model.patch_selector.explicit_region_indices[0]
    model.patch_selector.explicit_region_indices = explicit


@torch.no_grad()
def evaluate_regions(model, cache, records, split, options, seed, batch_size):
    model.eval()
    indices = [index for index, record in enumerate(records) if record["split"] == split]
    rows, predictions, trajectories = [], [], []
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start:start + batch_size]
        observed_batch = get_training_batch(cache, selected_indices)
        set_selection_control(model, records, selected_indices, options, seed)
        outputs = observed_forward(model, observed_batch)
        trajectory = outputs["trajectory"]
        predictions.append(trajectory.cpu())
        trajectories.append(observed_batch["ego_trajectory_target"].cpu())
        ade = (trajectory[..., :2] - observed_batch["ego_trajectory_target"][..., :2]).norm(dim=-1).mean(-1)
        selection = outputs["patch_selection"]
        targets, valid = pooled_future_targets(selection, observed_batch["future_target_latents"],
                                               observed_batch["future_target_valid_mask"])
        persistence = (selection.hard_selection_weights @ observed_batch["current_patch_latents"])[:, :, None]
        errors = (outputs["predicted_future_latents"] - targets).square().mean(-1)
        persistence_errors = (persistence - targets).square().mean(-1)
        for position, index in enumerate(selected_indices):
            record = records[index]
            region_valid = valid[position]
            rows.append({
                "token": record["current_frame_token"],
                "scene_token": record["scene_token"],
                "recording": record["recording_group"],
                "command": record["command_raw_index"],
                "ego_speed_meters_per_second": record["ego_speed_meters_per_second"],
                "xy_ade_m": float(ade[position]),
                "trajectory": trajectory[position].cpu().tolist(),
                "selected_region_ids": selection.selected_region_indices[position].cpu().tolist(),
                "supervised_patch_times": int(region_valid.sum()),
                "future_mse": float(errors[position][region_valid].mean()) if region_valid.any() else 0.0,
                "persistence_mse": float(persistence_errors[position][region_valid].mean()) if region_valid.any() else 0.0,
                "edge_density": float(model.future_bridge.last_hard_edges[position].float().mean()),
            })
    return {"summary": summarize_rows(rows, torch.cat(predictions), torch.cat(trajectories)), "windows": rows}


def auxiliary_losses(model, outputs, observed_batch, options):
    if options["history_objective"] == "none":
        future_loss = compute_region_future_auxiliary_loss(
            model, outputs, observed_batch["current_ego_status"],
            observed_batch["future_target_latents"], observed_batch["future_target_valid_mask"])
        return future_loss, future_loss.new_zeros(())
    region_budget = model.patch_selector.region_budget
    masked_slot_indices = torch.rand(observed_batch["current_patch_latents"].shape[0],
                                     region_budget, device="cuda").argsort(1)[:, :region_budget // 2]
    return compute_history_reconstruction_losses(
        model, outputs, observed_batch["current_ego_status"], observed_batch["earlier_patch_latents"],
        observed_batch["earlier_history_valid"], observed_batch["future_target_latents"],
        observed_batch["future_target_valid_mask"], masked_slot_indices,
        mask_current_history=options["history_objective"] == "masked")


def inspect_gradient_contract(model, agent, observed_batch, options):
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    report = {}
    for objective in (("planning",) if options.get("current_features_only") else ("planning", "future_auxiliary")):
        outputs = observed_forward(model, observed_batch)
        if objective == "planning":
            loss = agent.compute_loss({}, {"trajectory": observed_batch["ego_trajectory_target"]}, outputs)
        else:
            future_loss, reconstruction_loss = auxiliary_losses(model, outputs, observed_batch, options)
            loss = future_loss + reconstruction_loss
        gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
        totals = {"patch_selector": 0.0, "future_predictor": 0.0, "future_bridge": 0.0}
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None:
                totals[name.split(".")[0]] += float(gradient.square().sum())
        report[objective] = {name: value ** 0.5 for name, value in totals.items()}
    if "future_auxiliary" in report and (report["future_auxiliary"]["patch_selector"] or report["future_auxiliary"]["future_bridge"]):
        raise RuntimeError("Future/history targets crossed their gradient contract")
    return report


def prepare_observed_history_cache(agent, records, output, specification, started):
    history_directory = output / "observed_history_cache"
    history_directory.mkdir(exist_ok=True)
    encoder_wrapper = build_model(agent._model, "ego_query_residual", 29)
    feature_builder = agent.get_feature_builders()[0]
    previous_log_name, frames_in_log = None, None
    cached_history, manifest = [], []
    sensor_root = WORKSPACE / "dataset/sensor_blobs/trainval"
    for index, record in enumerate(records):
        guard(specification, started)
        history_path = history_directory / (record["current_frame_token"] + ".pt")
        metadata_path = history_path.with_suffix(".json")
        if history_path.exists():
            metadata = json.loads(metadata_path.read_text())
            if file_sha256(history_path) != metadata["cache_sha256"]:
                raise RuntimeError("Earlier history cache hash mismatch")
            history = torch.load(history_path, map_location="cpu", weights_only=True)
        else:
            if previous_log_name != record["segment_filename"]:
                log_path = WORKSPACE / "dataset/navsim_logs/trainval" / record["segment_filename"]
                with log_path.open("rb") as handle:
                    frames_in_log = pickle.load(handle)
                previous_log_name = record["segment_filename"]
            observed_frames = frames_in_log[record["start_index"]:record["start_index"] + 4]
            if len(observed_frames) != 4 or observed_frames[3]["token"] != record["current_frame_token"]:
                raise RuntimeError("Observed history alignment failed")
            image_paths = [sensor_root / next(camera["data_path"] for name, camera in frame["cams"].items()
                                             if name.lower() == "cam_f0") for frame in observed_frames]
            available = all(path.is_file() for path in image_paths)
            history = {"earlier_patch_latents": torch.zeros(512, 1024),
                       "earlier_history_valid": torch.tensor(available),
                       "observed_motion_region_scores": torch.zeros(128)}
            if available:
                cameras = []
                for path in image_paths:
                    with Image.open(path) as image:
                        cameras.append(SimpleNamespace(cam_f0=SimpleNamespace(image=np.array(image.convert("RGB")))))
                early_latest, early_earlier = feature_builder._get_camera_feature(SimpleNamespace(cameras=cameras[:2]))
                current_latest, current_earlier = feature_builder._get_camera_feature(SimpleNamespace(cameras=cameras[2:]))
                observed_images = torch.stack((early_earlier, early_latest, current_earlier, current_latest))
                early_clip = observed_images[:2].permute(1, 0, 2, 3)[None].cuda()
                history["earlier_patch_latents"] = encoder_wrapper.encode_observed_clip(early_clip)[0].cpu()
                history["observed_motion_region_scores"] = observed_history_motion_scores(observed_images)
            torch.save(history, history_path)
            metadata = {"token": record["current_frame_token"], "observed_frame_indices": [0, 1, 2, 3],
                        "observed_frame_timestamps_us": [int(frame["timestamp"]) for frame in observed_frames],
                        "source_image_hashes": {str(path): file_sha256(path) for path in image_paths if path.is_file()},
                        "history_available": available, "cache_sha256": file_sha256(history_path)}
            write_json(metadata_path, metadata)
        record["observed_motion_region_scores"] = history["observed_motion_region_scores"].tolist()
        record["earlier_history_available"] = bool(history["earlier_history_valid"])
        cached_history.append(history)
        manifest.append(metadata)
        if (index + 1) % 64 == 0:
            print(f"OBSERVED_HISTORY_CACHED {index + 1}/{len(records)}", flush=True)
    write_json(output / "observed_history_manifest.json", manifest)
    return {key: torch.stack([entry[key] for entry in cached_history]) for key in cached_history[0]}


def run_condition(
    agent, cache, records, specification, output, condition, seed, started, resume
):
    directory = output / f"{condition}_seed{seed}"
    if resume and (directory / "results.json").is_file():
        return json.loads((directory / "results.json").read_text())
    directory.mkdir(exist_ok=resume)
    run_started = time.perf_counter()
    options = specification["conditions"][condition]
    model = build_model(agent._model, "ego_query_residual", seed)
    warmup_path = (
        WORKSPACE
        / specification["warmup_root"]
        / f"ego_query_residual_seed{seed}/auxiliary_warmup_complete.pt"
    )
    restore_delta(model, warmup_path)
    # Hash before wrapping: identical learned scorer, predictor and bridge for every paired seed.
    common_hashes = module_hashes(model)
    model.patch_selector = SpatialRegionSelector(
        model.patch_selector, 16, 32, options["region_side"], options["region_budget"]
    ).cuda()
    torch.manual_seed(seed + 60000)
    model.future_bridge = LocatedRegionFutureBridge(
        model.patch_selector.region_pooling_weights,
        sparse_connections=options["sparse_connections"]).cuda()
    if options["history_objective"] != "none":
        model.future_predictor.current_time_embedding = torch.nn.Parameter(
            model.future_predictor.time_embeddings.weight.detach().mean(0, keepdim=True).clone())
    if options.get("current_features_only"):
        model.future_predictor.requires_grad_(False)
        def current_only_prediction(current_latents, ego_status, selection_weights):
            return (selection_weights @ current_latents)[:, :, None].expand(-1, -1, 4, -1)
        model._predict_selected = current_only_prediction
    if options["selection_mode"] != "learned":
        model.patch_selector.requires_grad_(False)
    train_indices = [
        index for index, record in enumerate(records) if record["split"] == "train"
    ]
    generator = torch.Generator().manual_seed(seed + 10000)
    schedule = [
        torch.tensor(train_indices)[
            torch.randperm(len(train_indices), generator=generator)[
                : specification["batch_size"]
            ]
        ].tolist()
        for _ in range(100 + specification["joint_updates"])
    ][100:]
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    reference = json.loads(
        (
            WORKSPACE
            / specification["reference_root"]
            / f"ego_lower_learning_rate_seed{seed}/results.json"
        ).read_text()
    )
    if (
        schedule_hash != reference["batch_schedule_sha256"]
        or common_hashes != reference["initial_module_hashes"]
    ):
        raise RuntimeError(
            "Paired batch order or common initialization differs from preserved reference"
        )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.future_predictor.parameters(),
                "lr": specification["learning_rate"],
            },
            {
                "params": model.future_bridge.parameters(),
                "lr": specification["learning_rate"],
            },
            {
                "params": [
                    parameter
                    for parameter in model.patch_selector.parameters()
                    if parameter.requires_grad
                ],
                "lr": specification["selector_learning_rate"],
            },
        ],
        weight_decay=specification["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: (
            0.1
            + 0.45
            * (
                1
                + math.cos(
                    math.pi
                    * min(update, specification["joint_updates"])
                    / specification["joint_updates"]
                )
            )
        ),
    )
    torch.manual_seed(seed + 30000)
    torch.cuda.reset_peak_memory_stats()
    curve, evaluations, completed_update = [], {}, 0
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    def checkpoint(filename):
        payload = {
            "condition": condition,
            "seed": seed,
            "specification": specification,
            "completed_update": completed_update,
            "curve": curve,
            "evaluations": evaluations,
            "extension_parameters": {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.named_parameters()
                if not name.startswith("baseline_model.")
            },
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(),
            "batch_schedule": schedule,
            "batch_schedule_sha256": schedule_hash,
        }
        partial = directory / (filename + ".partial")
        torch.save(payload, partial)
        partial.replace(directory / filename)

    def evaluate_update():
        return {
            split: evaluate_regions(
                model, cache, records, split, options, seed, specification["batch_size"]
            )
            for split in ("train", "development")
        }

    try:
        if resume and (directory / "latest.pt").exists():
            saved = torch.load(directory / "latest.pt", map_location="cpu")
            if (
                saved["specification"] != specification
                or saved["batch_schedule"] != schedule
            ):
                raise RuntimeError("Resume configuration changed")
            parameters = {
                name: parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("baseline_model.")
            }
            if set(parameters) != set(saved["extension_parameters"]):
                raise RuntimeError("Resume keys differ")
            with torch.no_grad():
                for name, parameter in parameters.items():
                    if parameter.shape != saved["extension_parameters"][name].shape:
                        raise RuntimeError("Resume shape differs")
                    parameter.copy_(saved["extension_parameters"][name])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state(saved["cuda_rng"])
            completed_update, curve, evaluations = (
                saved["completed_update"],
                saved["curve"],
                saved["evaluations"],
            )
        else:
            first_batch = get_training_batch(cache, schedule[0])
            set_selection_control(model, records, schedule[0], options, seed)
            with torch.no_grad():
                original = observed_forward(
                    model, first_batch, enable_future_branch=False
                )["trajectory"]
                if not torch.equal(
                    original, observed_forward(model, first_batch)["trajectory"]
                ):
                    raise RuntimeError(
                        "Update-zero region branch does not preserve original output"
                    )
            contract = inspect_gradient_contract(
                model, agent, first_batch, options
            )
            write_json(directory / "initial_gradient_contract.json", contract)
            evaluations["0"] = evaluate_update()
            checkpoint("initial.pt")
        for update in range(completed_update + 1, specification["joint_updates"] + 1):
            guard(specification, started, run_started)
            model.train()
            selected_indices = schedule[update - 1]
            observed_batch = get_training_batch(cache, selected_indices)
            set_selection_control(model, records, selected_indices, options, seed)
            optimizer.zero_grad(set_to_none=True)
            outputs = observed_forward(model, observed_batch)
            planning_loss = agent.compute_loss(
                {}, {"trajectory": observed_batch["ego_trajectory_target"]}, outputs
            )
            if options.get("current_features_only"):
                future_loss = planning_loss.new_zeros(())
                reconstruction_loss = planning_loss.new_zeros(())
            else:
                future_loss, reconstruction_loss = auxiliary_losses(model, outputs, observed_batch, options)
            sparsity_loss = (model.future_bridge.connection_sparsity_loss()
                             if options["sparse_connections"] else planning_loss.new_zeros(()))
            loss = (planning_loss + specification["future_auxiliary_weight"] * future_loss
                    + specification["current_reconstruction_weight"] * reconstruction_loss
                    + specification["connection_sparsity_weight"] * sparsity_loss)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite combined loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                trainable, specification["gradient_clip_norm"]
            )
            if not torch.isfinite(norm):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            scheduler.step()
            completed_update = update
            curve.append(
                {
                    "update": update,
                    "planning_loss": float(planning_loss.detach()),
                    "future_auxiliary_loss": float(future_loss.detach()),
                    "current_reconstruction_loss": float(reconstruction_loss.detach()),
                    "connection_sparsity_loss": float(sparsity_loss.detach()),
                    "gradient_norm": float(norm),
                }
            )
            if update in specification["evaluation_updates"]:
                evaluations[str(update)] = evaluate_update()
                print(
                    f"EVALUATED {condition} seed={seed} update={update} dev_ADE={evaluations[str(update)]['development']['summary']['scene_macro_xy_ade_m']:.6f}",
                    flush=True,
                )
            if update % 25 == 0:
                checkpoint("latest.pt")
                write_json(
                    directory / "progress.json",
                    {
                        "completed_update": update,
                        "wall_seconds": time.perf_counter() - run_started,
                    },
                )
        set_selection_control(model, records, schedule[0], options, seed)
        contract = inspect_gradient_contract(
            model,
            agent,
            get_training_batch(cache, schedule[0]),
            options,
        )
        checkpoint("complete.pt")
        report = {
            "condition": condition,
            "options": options,
            "seed": seed,
            "evaluations": evaluations,
            "curve": curve,
            "common_initial_hashes": common_hashes,
            "final_module_hashes": module_hashes(model),
            "batch_schedule_sha256": schedule_hash,
            "warmup_sha256": file_sha256(warmup_path),
            "gradient_contract": contract,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "wall_seconds": time.perf_counter() - run_started,
            "active_parameter_count": sum(parameter.numel() for parameter in trainable),
            "region_target_queries": options["region_budget"] * 4,
            "covered_native_cells": options["region_budget"]
            * options["region_side"] ** 2,
        }
        write_json(directory / "results.json", report)
        return report
    except BaseException as error:
        checkpoint("latest.pt")
        write_json(
            directory / "stopped.json",
            {"error": repr(error), "completed_update": completed_update},
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.output_directory = args.output_directory.resolve()
    if args.detach:
        if args.output_directory.exists() and not args.resume:
            raise RuntimeError("New output directory required")
        args.output_directory.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output_directory.with_suffix(".log")
        command = [
            "nohup",
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            "--config",
            str(args.config.resolve()),
            "--output-directory",
            str(args.output_directory.resolve()),
        ]
        if args.resume:
            command.append("--resume")
        with log_path.open("a" if args.resume else "x") as log_stream:
            child = subprocess.Popen(
                command,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                cwd=WORKSPACE,
            )
        print(
            json.dumps(
                {
                    "pid": child.pid,
                    "log": str(log_path),
                    "output": str(args.output_directory),
                }
            ),
            flush=True,
        )
        return
    specification = json.loads(args.config.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Exactly one approved physical GPU is required")
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("GPU launch reserve insufficient")
    args.output_directory.mkdir(parents=True, exist_ok=args.resume)
    snapshot = args.output_directory / "specification.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != specification:
        raise RuntimeError("Output specification differs")
    write_json(snapshot, specification)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    started = time.perf_counter()
    agent, source = load_official_agent(args.output_directory)
    baseline_hash = parameter_sha256(agent._model)
    if (
        baseline_hash
        != "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a"
    ):
        raise RuntimeError("Official planner hash mismatch")
    write_json(args.output_directory / "source.json", source)
    index = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    if index["counts"] != specification["expected_cache_counts"]:
        raise RuntimeError("Unexpected cache counts")
    records, loaded = index["records"], []
    for record in records:
        if (
            record["split"] not in ("train", "development")
            or file_sha256(record["cache_file"]) != record["cache_sha256"]
        ):
            raise RuntimeError("Cache hash/split mismatch")
        loaded.append(
            torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        )
    cache = {key: torch.stack([value[key] for value in loaded]) for key in loaded[0]}
    del loaded
    cache.update(prepare_observed_history_cache(agent, records, args.output_directory, specification, started))
    write_json(
        args.output_directory / "provenance.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
            ).strip(),
            "config_sha256": file_sha256(args.config),
            "cache_index_sha256": file_sha256(
                WORKSPACE / specification["reused_cache"]
            ),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_version": torch.__version__,
        },
    )
    if sum(bool(row["earlier_history_available"]) for row in records) < specification["minimum_observed_history_count"]:
        raise RuntimeError("Insufficient observed history for registered masking comparison")
    completed = []
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            guard(specification, started)
            report = run_condition(
                agent,
                cache,
                records,
                specification,
                args.output_directory,
                condition,
                seed,
                started,
                args.resume,
            )
            completed.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "summary": report["evaluations"][
                        str(specification["joint_updates"])
                    ]["development"]["summary"],
                }
            )
            write_json(args.output_directory / "completed_runs.json", completed)
            torch.cuda.empty_cache()
    if parameter_sha256(agent._model) != baseline_hash:
        raise RuntimeError("Original planner weights changed")
    write_json(
        args.output_directory / "completion.json",
        {
            "complete": True,
            "baseline_hash_unchanged": baseline_hash,
            "wall_seconds": time.perf_counter() - started,
            "completed_runs": completed,
        },
    )
    print("REGION_RESEARCH_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
