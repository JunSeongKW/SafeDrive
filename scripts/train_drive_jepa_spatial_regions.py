"""Bounded paired region-size/count/teacher-retention experiment; cached inputs only."""

import argparse
import hashlib
import json
import math
import os
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
    compute_planner_retention_loss,
    compute_region_future_auxiliary_loss,
    planner_retention_outputs,
    pooled_future_targets,
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
    model.patch_selector.explicit_region_indices = explicit


@torch.no_grad()
def evaluate_regions(model, cache, records, split, options, seed, batch_size):
    model.eval()
    indices = [
        index for index, record in enumerate(records) if record["split"] == split
    ]
    rows, predictions, trajectories = [], [], []
    for start in range(0, len(indices), batch_size):
        selected_indices = indices[start : start + batch_size]
        observed_batch = get_training_batch(cache, selected_indices)
        set_selection_control(model, records, selected_indices, options, seed)
        outputs = observed_forward(model, observed_batch)
        trajectory = outputs["trajectory"]
        predictions.append(trajectory.cpu())
        trajectories.append(observed_batch["ego_trajectory_target"].cpu())
        ade = (
            (trajectory[..., :2] - observed_batch["ego_trajectory_target"][..., :2])
            .norm(dim=-1)
            .mean(-1)
        )
        selection = outputs["patch_selection"]
        targets, valid = pooled_future_targets(
            selection,
            observed_batch["future_target_latents"],
            observed_batch["future_target_valid_mask"],
        )
        persistence = (
            selection.hard_selection_weights @ observed_batch["current_patch_latents"]
        )[:, :, None]
        errors = (outputs["predicted_future_latents"] - targets).square().mean(-1)
        persistence_errors = (persistence - targets).square().mean(-1)
        retained, original = planner_retention_outputs(
            model,
            observed_batch["current_patch_latents"],
            observed_batch["current_ego_status"],
            selection,
        )
        readout_error = (retained[..., :2] - original[..., :2]).norm(dim=-1).mean(-1)
        for position, index in enumerate(selected_indices):
            record = records[index]
            region_valid = valid[position]
            rows.append(
                {
                    "token": record["current_frame_token"],
                    "recording": record["recording_group"],
                    "scene_token": record["scene_token"],
                    "command": record["command_raw_index"],
                    "xy_ade_m": float(ade[position]),
                    "selected_region_ids": selection.selected_region_indices[position]
                    .cpu()
                    .tolist(),
                    "supervised_patch_times": int(region_valid.sum()),
                    "future_mse": float(errors[position][region_valid].mean())
                    if region_valid.any()
                    else 0.0,
                    "persistence_mse": float(
                        persistence_errors[position][region_valid].mean()
                    )
                    if region_valid.any()
                    else 0.0,
                    "retained_current_planner_xy_difference_m": float(
                        readout_error[position]
                    ),
                    "future_mse_by_horizon": [
                        float(errors[position, :, step][region_valid[:, step]].mean())
                        if region_valid[:, step].any()
                        else None
                        for step in range(4)
                    ],
                }
            )
    summary = summarize_rows(rows, torch.cat(predictions), torch.cat(trajectories))
    summary["retained_current_planner_xy_difference_m"] = sum(
        row["retained_current_planner_xy_difference_m"] for row in rows
    ) / len(rows)
    return {"summary": summary, "windows": rows}


def inspect_gradient_contract(model, agent, observed_batch, learned):
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    report = {}
    for objective in (
        ["planning", "future_auxiliary", "current_retention"]
        if learned
        else ["planning", "future_auxiliary"]
    ):
        outputs = observed_forward(model, observed_batch)
        if objective == "planning":
            loss = agent.compute_loss(
                {}, {"trajectory": observed_batch["ego_trajectory_target"]}, outputs
            )
        elif objective == "future_auxiliary":
            loss = compute_region_future_auxiliary_loss(
                model,
                outputs,
                observed_batch["current_ego_status"],
                observed_batch["future_target_latents"],
                observed_batch["future_target_valid_mask"],
            )
        else:
            loss = compute_planner_retention_loss(
                model,
                observed_batch["current_patch_latents"],
                observed_batch["current_ego_status"],
                outputs["patch_selection"],
            )
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in named], allow_unused=True
        )
        totals = {"patch_selector": 0.0, "future_predictor": 0.0, "future_bridge": 0.0}
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None:
                totals[name.split(".")[0]] += float(gradient.square().sum())
        report[objective] = {name: value**0.5 for name, value in totals.items()}
    if (
        report["future_auxiliary"]["patch_selector"]
        or report["future_auxiliary"]["future_bridge"]
    ):
        raise RuntimeError("Future auxiliary gradient crossed its contract")
    if learned and (
        report["current_retention"]["future_predictor"]
        or report["current_retention"]["future_bridge"]
        or not report["current_retention"]["patch_selector"]
    ):
        raise RuntimeError("Retention must reach selector only")
    return report


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
    if options["selection_mode"] == "random":
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
                model, agent, first_batch, options["selection_mode"] == "learned"
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
            future_loss = compute_region_future_auxiliary_loss(
                model,
                outputs,
                observed_batch["current_ego_status"],
                observed_batch["future_target_latents"],
                observed_batch["future_target_valid_mask"],
            )
            retention_loss = (
                compute_planner_retention_loss(
                    model,
                    observed_batch["current_patch_latents"],
                    observed_batch["current_ego_status"],
                    outputs["patch_selection"],
                )
                if options["retention_weight"]
                else planning_loss.new_zeros(())
            )
            loss = (
                planning_loss
                + specification["future_auxiliary_weight"] * future_loss
                + options["retention_weight"] * retention_loss
            )
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
                    "retention_loss": float(retention_loss.detach()),
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
            options["selection_mode"] == "learned",
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
    if args.detach:
        if args.output_directory.exists() and not args.resume:
            raise RuntimeError("New output directory required")
        args.output_directory.parent.mkdir(parents=True, exist_ok=True)
        log_path = args.output_directory.with_suffix(".log")
        command = [
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
    write_json(
        args.output_directory / "provenance.json",
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "config_sha256": file_sha256(args.config),
            "cache_index_sha256": file_sha256(
                WORKSPACE / specification["reused_cache"]
            ),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "torch_version": torch.__version__,
        },
    )
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
    print("SPATIAL_REGION_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
