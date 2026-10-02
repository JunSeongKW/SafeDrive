"""Bounded, resumable hypothesis tests on a preserved official planner.

The older optimizer experiment is not resumed: each condition starts at the
identical per-seed auxiliary-warmup boundary and a new, preregistered schedule.
--resume restores THIS runner's optimizer/scheduler/RNG and completed update.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import signal
import time
from datetime import datetime
from pathlib import Path

import torch
from diagnose_drive_jepa_learning_limitations import load_official_agent, restore_delta
from run_drive_jepa_architecture_followup import (
    WORKSPACE,
    branch_dependence,
    build_model,
    evaluate,
    load_reused_cache,
    module_hashes,
    observed_forward,
)
from run_drive_jepa_selection_comparison import get_training_batch, write_json
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    parameter_sha256,
)

from planning_aware_future_prediction.models.future_gradient_routing import (
    freeze_parameter_gradients,
    project_auxiliary_gradient,
    uniform_xy_ade_with_heading,
)

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def enforce_safety(specification, started, run_started=None):
    if STOP_REQUESTED:
        raise RuntimeError("Stop requested: saving only our run")
    if datetime.now().astimezone() >= datetime.fromisoformat(
        specification["deadline_iso"]
    ):
        raise RuntimeError("Predeclared morning deadline reached")
    if time.perf_counter() - started > specification["maximum_wall_seconds"]:
        raise RuntimeError("Overall wall-time bound")
    if (
        run_started is not None
        and time.perf_counter() - run_started
        > specification["maximum_condition_wall_seconds"]
    ):
        raise RuntimeError("Per-condition wall-time bound")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError(
            "Shared GPU free-memory reserve reached; no automatic restart"
        )
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Our allocated-memory cap reached")


def planning_forward(model, observed_batch, condition):
    parameter_context = (
        freeze_parameter_gradients(model.future_predictor)
        if condition == "ego_predictor_aux_only"
        else contextlib.nullcontext()
    )
    with parameter_context:
        return observed_forward(model, observed_batch)


def planning_objective(agent, result, batch, condition, specification):
    if condition == "ego_uniform_ade":
        return uniform_xy_ade_with_heading(
            result["trajectory"],
            batch["ego_trajectory_target"],
            specification["heading_weight"],
        )
    return agent.compute_loss(
        {}, {"trajectory": batch["ego_trajectory_target"]}, result
    )


def gradient_contract(model, agent, batch, condition, specification):
    report = {}
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    for kind in ("planning", "auxiliary"):
        result = planning_forward(model, batch, condition)
        loss = (
            planning_objective(agent, result, batch, condition, specification)
            if kind == "planning"
            else model.compute_future_auxiliary_loss(
                result,
                batch["current_ego_status"],
                batch["future_target_latents"],
                batch["future_target_valid_mask"],
            )
        )
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in named], allow_unused=True
        )
        totals = {"patch_selector": 0.0, "future_predictor": 0.0, "future_bridge": 0.0}
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None:
                totals[name.split(".")[0]] += float(gradient.square().sum())
        report[kind] = {name: value**0.5 for name, value in totals.items()}
    if (
        report["auxiliary"]["patch_selector"] != 0
        or report["auxiliary"]["future_bridge"] != 0
    ):
        raise RuntimeError("Auxiliary selector/bridge gradient leaked")
    if (
        condition == "ego_predictor_aux_only"
        and report["planning"]["future_predictor"] != 0
    ):
        raise RuntimeError("Predictor parameter freeze violated")
    if condition != "ego_selector_frozen" and report["planning"]["patch_selector"] <= 0:
        raise RuntimeError(
            "Selector planning gradient disconnected after bridge warmup"
        )
    return report


def run_condition(
    agent, cache, records, specification, output, condition, seed, started, resume
):
    directory = output / f"{condition}_seed{seed}"
    if (directory / "results.json").is_file():
        if not resume:
            raise RuntimeError("Existing completed run; use explicit --resume to reuse")
        return json.loads((directory / "results.json").read_text())
    directory.mkdir(exist_ok=resume)
    run_started = time.perf_counter()
    architecture = (
        "mlp_recipe_control" if condition == "mlp_reference" else "ego_query_residual"
    )
    model = build_model(agent._model, architecture, seed)
    warmup_path = (
        WORKSPACE
        / specification["warmup_root"]
        / f"{architecture}_seed{seed}/auxiliary_warmup_complete.pt"
    )
    restore_delta(model, warmup_path)
    initial_hashes = module_hashes(model)
    if condition == "ego_selector_frozen":
        model.patch_selector.requires_grad_(False)
    bridge_handle = (
        model.future_bridge.register_forward_hook(
            lambda module, inputs, result: result * 0.5
        )
        if condition == "ego_bridge_half"
        else None
    )
    train_indices = [
        index for index, row in enumerate(records) if row["split"] == "train"
    ]
    batch_generator = torch.Generator().manual_seed(seed + 10000)
    # Consume the original 100 warmup draws so the first 200 joint batches match.
    schedule = [
        torch.tensor(train_indices)[
            torch.randperm(len(train_indices), generator=batch_generator)[
                : specification["batch_size"]
            ]
        ].tolist()
        for _ in range(100 + specification["joint_updates"])
    ][100:]
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.future_predictor.parameters()),
                "lr": specification["learning_rate"],
            },
            {
                "params": list(model.future_bridge.parameters()),
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
        lambda step: (
            0.1
            + 0.45
            * (
                1
                + math.cos(
                    math.pi
                    * min(step, specification["joint_updates"])
                    / specification["joint_updates"]
                )
            )
        ),
    )
    torch.manual_seed(seed + 30000)
    torch.cuda.reset_peak_memory_stats()
    completed_update, curve, evaluations = 0, [], {}
    predictor_parameters = list(model.future_predictor.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]

    def checkpoint(filename):
        payload = {
            "condition": condition,
            "seed": seed,
            "completed_update": completed_update,
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
            "specification": specification,
            "curve": curve,
            "evaluations": evaluations,
        }
        destination = directory / filename
        temporary = destination.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(destination)

    try:
        if resume and (directory / "latest.pt").is_file():
            saved = torch.load(directory / "latest.pt", map_location="cpu")
            if (
                saved["specification"] != specification
                or saved["batch_schedule"] != schedule
            ):
                raise RuntimeError("Resume configuration or batch sequence changed")
            expected = {
                name: parameter
                for name, parameter in model.named_parameters()
                if not name.startswith("baseline_model.")
            }
            if set(expected) != set(saved["extension_parameters"]):
                raise RuntimeError("Resume parameter key mismatch")
            with torch.no_grad():
                for name, parameter in expected.items():
                    value = saved["extension_parameters"][name]
                    if parameter.shape != value.shape:
                        raise RuntimeError("Resume parameter shape mismatch")
                    parameter.copy_(value)
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
            batch = get_training_batch(cache, schedule[0])
            with torch.no_grad():
                disabled = observed_forward(model, batch, enable_future_branch=False)[
                    "trajectory"
                ]
                enabled = observed_forward(model, batch)["trajectory"]
                if not torch.equal(disabled, enabled):
                    raise RuntimeError(
                        "Auxiliary-warmup checkpoint should retain zero bridge"
                    )
            evaluations["0"] = {
                split: evaluate(
                    model, cache, records, split, specification["batch_size"]
                )
                for split in ("train", "development")
            }
            checkpoint("latest.pt")
        for update in range(completed_update + 1, specification["joint_updates"] + 1):
            enforce_safety(specification, started, run_started)
            batch = get_training_batch(cache, schedule[update - 1])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = planning_forward(model, batch, condition)
            planning_loss = planning_objective(
                agent, result, batch, condition, specification
            )
            auxiliary_loss = model.compute_future_auxiliary_loss(
                result,
                batch["current_ego_status"],
                batch["future_target_latents"],
                batch["future_target_valid_mask"],
            )
            weighted_auxiliary = (
                specification["future_auxiliary_weight"] * auxiliary_loss
            )
            combined_loss = planning_loss + weighted_auxiliary
            if not torch.isfinite(combined_loss):
                raise RuntimeError("Nonfinite loss")
            projection = None
            if condition == "ego_planning_priority_projection":
                planning_gradients = torch.autograd.grad(
                    planning_loss,
                    predictor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                auxiliary_gradients = torch.autograd.grad(
                    weighted_auxiliary,
                    predictor_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                projected, projection = project_auxiliary_gradient(
                    planning_gradients, auxiliary_gradients, predictor_parameters
                )
            combined_loss.backward()
            if projection is not None:
                for parameter, gradient in zip(predictor_parameters, projected):
                    parameter.grad = gradient
            norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters, specification["gradient_clip_norm"]
            )
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite gradient")
            optimizer.step()
            scheduler.step()
            completed_update = update
            curve.append(
                {
                    "update": update,
                    "planning_loss": float(planning_loss.detach()),
                    "auxiliary_loss": float(auxiliary_loss.detach()),
                    "gradient_norm": float(norm),
                    "projection": projection,
                }
            )
            if update in specification["evaluation_updates"]:
                evaluations[str(update)] = {
                    split: evaluate(
                        model, cache, records, split, specification["batch_size"]
                    )
                    for split in ("train", "development")
                }
                checkpoint(f"joint_update_{update}.pt")
                print(
                    f"EVALUATED {condition} seed={seed} update={update} dev_ADE={evaluations[str(update)]['development']['summary']['scene_macro_xy_ade_m']:.6f}",
                    flush=True,
                )
            if update % 25 == 0:
                checkpoint("latest.pt")
                write_json(
                    directory / "progress.json",
                    {
                        "condition": condition,
                        "seed": seed,
                        "completed_update": update,
                        "elapsed_seconds": time.perf_counter() - run_started,
                    },
                )
        report = {
            "condition": condition,
            "seed": seed,
            "evaluations": evaluations,
            "curve": curve,
            "initial_module_hashes": initial_hashes,
            "final_module_hashes": module_hashes(model),
            "warmup_checkpoint": str(warmup_path),
            "warmup_sha256": file_sha256(warmup_path),
            "batch_schedule_sha256": schedule_hash,
            "gradient_contract": gradient_contract(
                model,
                agent,
                get_training_batch(cache, schedule[0]),
                condition,
                specification,
            ),
            "branch_dependence": branch_dependence(
                model, get_training_batch(cache, schedule[0])
            ),
            "wall_seconds": time.perf_counter() - run_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "active_parameter_count": sum(
                parameter.numel() for parameter in trainable_parameters
            ),
        }
        checkpoint("complete.pt")
        write_json(directory / "results.json", report)
        return report
    except BaseException as error:
        # Even pressure/interrupt preserves CPU deltas and exact optimizer state.
        checkpoint("latest.pt")
        write_json(
            directory / "stopped.json",
            {
                "reason": repr(error),
                "completed_update": completed_update,
                "checkpoint": str(directory / "latest.pt"),
            },
        )
        raise
    finally:
        if bridge_handle is not None:
            bridge_handle.remove()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path, output = args.config.resolve(), args.output_directory.resolve()
    specification = json.loads(config_path.read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Exactly one approved physical GPU 0 or 1 is required")
    output.mkdir(parents=True, exist_ok=args.resume)
    snapshot = output / "specification.json"
    if snapshot.is_file() and json.loads(snapshot.read_text()) != specification:
        raise RuntimeError("Output configuration mismatch")
    write_json(snapshot, specification)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient shared GPU launch headroom")
    started = time.perf_counter()
    agent, source = load_official_agent(output)
    records, cache = load_reused_cache(specification)
    baseline_hash = parameter_sha256(agent._model)
    if (
        baseline_hash
        != "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a"
    ):
        raise RuntimeError("Unexpected official baseline")
    write_json(output / "source.json", source)
    results = []
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            enforce_safety(specification, started)
            result = run_condition(
                agent,
                cache,
                records,
                specification,
                output,
                condition,
                seed,
                started,
                args.resume,
            )
            results.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "summary": result["evaluations"][
                        str(specification["joint_updates"])
                    ]["development"]["summary"],
                }
            )
            write_json(output / "completed_runs.json", results)
    if parameter_sha256(agent._model) != baseline_hash:
        raise RuntimeError("Official weights changed")
    write_json(
        output / "completion.json",
        {
            "complete": True,
            "baseline_hash_unchanged": baseline_hash,
            "wall_seconds": time.perf_counter() - started,
            "results": results,
        },
    )
    print("CAUSAL_FOLLOWUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
