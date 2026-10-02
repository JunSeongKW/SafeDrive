"""Bounded architectural comparison, reused 192 windows, untouched official model.

Run from a dedicated Conda environment, one approved GPU, no benchmark tuning.
Checkpoint trainable deltas + optimizer/scheduler/RNG on regular and stop boundaries.
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
import pickle
import signal
import subprocess
import time
from pathlib import Path

import torch
from run_drive_jepa_selection_comparison import (
    get_training_batch,
    summarize_rows,
    write_json,
)
from torch import nn
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    parameter_sha256,
)

from planning_aware_future_prediction.models.drive_jepa_adaptive_future import (
    ContextualResidualFuturePredictor,
    DriveJEPAAdaptiveFuture,
    EgoQueryPatchSelector,
    FutureBranchLoRAEncoderTail,
)
from planning_aware_future_prediction.models.drive_jepa_selective_patch_future import (
    FutureMemoryResidualBridge,
    LightweightPatchFuturePredictor,
    PatchSelection,
    PlanningConditionedPatchSelector,
)

WORKSPACE = Path(__file__).resolve().parents[1]
STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def check_limits(specification, started, condition_started=None):
    if STOP_REQUESTED:
        raise RuntimeError("User or system stop requested")
    if time.perf_counter() - started > specification["maximum_wall_seconds"]:
        raise RuntimeError("Registered overall time cap")
    if (
        condition_started
        and time.perf_counter() - condition_started
        > specification["maximum_condition_wall_seconds"]
    ):
        raise RuntimeError("Registered condition time cap")
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Registered memory cap")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError("Shared GPU reserve reached")


def build_model(baseline, condition, seed):
    # Independent module seeds keep shared modules identical across architectures.
    torch.manual_seed(seed)
    model = DriveJEPAAdaptiveFuture(
        baseline, contextual_predictor=False, ego_query_selector=False
    )
    torch.manual_seed(seed + 10)
    selector_type = (
        PlanningConditionedPatchSelector
        if condition in ("mlp_recipe_control", "contextual_residual")
        else EgoQueryPatchSelector
    )
    model.patch_selector = selector_type(1024, 128, 4)
    torch.manual_seed(seed + 20)
    predictor_type = (
        LightweightPatchFuturePredictor
        if condition == "mlp_recipe_control"
        else ContextualResidualFuturePredictor
    )
    model.future_predictor = predictor_type(1024, 128, 4)
    torch.manual_seed(seed + 30)
    model.future_bridge = FutureMemoryResidualBridge(1024, 256, 4)
    if "lora" in condition:
        torch.manual_seed(seed + 40)
        model.encoder_lora_tail = FutureBranchLoRAEncoderTail(baseline.image_encoder)
    return model.cuda().eval()


def load_reused_cache(specification):
    cache_path = WORKSPACE / specification["reused_cache"]
    cache_report = json.loads(cache_path.read_text())
    records, tensors = cache_report["records"], []
    if len(records) != 192 or {row["split"] for row in records} != {
        "train",
        "development",
    }:
        raise RuntimeError("Unexpected reused split; no expansion permitted")
    for row in records:
        if file_sha256(row["cache_file"]) != row["cache_sha256"]:
            raise RuntimeError("Prior cache modified")
        tensors.append(
            torch.load(row["cache_file"], map_location="cpu", weights_only=True)
        )
    cache = {key: torch.stack([row[key] for row in tensors]) for key in tensors[0]}
    return records, cache


def cache_current_prefix(agent, model, records, cache, output, specification, started):
    from navsim.common.dataclasses import AgentInput

    directory = output / "current_encoder_prefix_cache"
    directory.mkdir()
    captures, logs, rows, prefixes = [], {}, [], []
    handle = agent._model.image_encoder.blocks[20].register_forward_pre_hook(
        lambda module, inputs: captures.append(inputs[0].detach())
    )
    cache_bytes, maximum_difference = 0, 0.0
    try:
        for index, record in enumerate(records):
            check_limits(specification, started)
            segment = record["segment_filename"]
            if segment not in logs:
                with (WORKSPACE / "dataset/navsim_logs/trainval" / segment).open(
                    "rb"
                ) as stream:
                    logs = {segment: pickle.load(stream)}
            frames = logs[segment][record["start_index"] : record["start_index"] + 4]
            if frames[-1]["token"] != record["current_frame_token"]:
                raise RuntimeError("Prefix current-frame alignment mismatch")
            agent_input = AgentInput.from_scene_dict_list(
                frames,
                WORKSPACE / "dataset/sensor_blobs/trainval",
                4,
                agent.get_sensor_config(),
            )
            features = agent.get_feature_builders()[0].compute_features(agent_input)
            if not torch.equal(
                features["status_feature"].float(), cache["current_ego_status"][index]
            ):
                raise RuntimeError("Reused observed status mismatch")
            clip = torch.stack(
                (features["camera_feature_2"], features["camera_feature_1"]), dim=1
            )[None].cuda()
            captures.clear()
            with torch.no_grad():
                encoded = model.encode_observed_clip(clip)[0].cpu()
            if len(captures) != 1:
                raise RuntimeError("Expected one current prefix capture")
            difference = float(
                (encoded - cache["current_patch_latents"][index]).abs().max()
            )
            maximum_difference = max(maximum_difference, difference)
            torch.testing.assert_close(
                encoded,
                cache["current_patch_latents"][index],
                atol=specification["prefix_equivalence_atol"],
                rtol=specification["prefix_equivalence_rtol"],
            )
            prefix = captures[0][0].cpu()
            path = directory / (record["current_frame_token"] + ".pt")
            if (
                cache_bytes + prefix.numel() * 4 + 16384
                > specification["maximum_prefix_cache_gib"] * 2**30
            ):
                raise RuntimeError("Prefix cache disk cap reached")
            torch.save(prefix, path)
            cache_bytes += path.stat().st_size
            prefixes.append(prefix)
            rows.append(
                {
                    "token": record["current_frame_token"],
                    "file": str(path),
                    "sha256": file_sha256(path),
                }
            )
            if (index + 1) % 32 == 0:
                print(f"PREFIX {index + 1}/192 bytes={cache_bytes}", flush=True)
    finally:
        handle.remove()
    cache["current_prefix_latents"] = torch.stack(prefixes)
    report = {
        "records": rows,
        "bytes": cache_bytes,
        "max_original_feature_difference": maximum_difference,
        "source_cache_sha256": file_sha256(WORKSPACE / specification["reused_cache"]),
    }
    write_json(output / "prefix_cache_index.json", report)
    return report


def observed_forward(model, batch, **kwargs):
    return model.forward_cached_observations(
        batch["current_patch_latents"],
        batch["current_ego_status"],
        batch.get("current_prefix_latents"),
        **kwargs,
    )


def target_for_condition(batch, condition):
    if condition == "lora_current_target_control":
        return batch["current_patch_latents"][:, None].expand(-1, 4, -1, -1)
    return batch["future_target_latents"]


@torch.no_grad()
def evaluate(model, cache, records, split, batch_size):
    model.eval()
    indices = [index for index, row in enumerate(records) if row["split"] == split]
    rows, predictions, targets = [], [], []
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        batch = get_training_batch(cache, selected)
        result = observed_forward(model, batch)
        prediction, target = result["trajectory"], batch["ego_trajectory_target"]
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        ade = (prediction[..., :2] - target[..., :2]).norm(dim=-1).mean(dim=-1)
        for position, index in enumerate(selected):
            metadata = records[index]
            patch_ids = result["patch_selection"].selected_patch_indices[position]
            future_target = batch["future_target_latents"][
                position, :, patch_ids
            ].transpose(0, 1)
            valid = batch["future_target_valid_mask"][position, :, patch_ids].transpose(
                0, 1
            )
            future_prediction = result["predicted_future_latents"][position]
            persistence = batch["current_patch_latents"][position, patch_ids, None]
            error = (future_prediction - future_target).square().mean(-1)
            persistence_error = (persistence - future_target).square().mean(-1)
            rows.append(
                {
                    "token": metadata["current_frame_token"],
                    "recording": metadata["recording_group"],
                    "scene_token": metadata["scene_token"],
                    "command": metadata["command_raw_index"],
                    "xy_ade_m": float(ade[position]),
                    "selected_patch_ids": patch_ids.cpu().tolist(),
                    "supervised_patch_times": int(valid.sum()),
                    "future_mse": float(error[valid].mean()) if valid.any() else 0.0,
                    "persistence_mse": float(persistence_error[valid].mean())
                    if valid.any()
                    else 0.0,
                    "future_mse_by_horizon": [
                        float(error[:, step][valid[:, step]].mean())
                        if valid[:, step].any()
                        else None
                        for step in range(4)
                    ],
                    "persistence_mse_by_horizon": [
                        float(persistence_error[:, step][valid[:, step]].mean())
                        if valid[:, step].any()
                        else None
                        for step in range(4)
                    ],
                    "prediction_delta_rms_from_frozen_current": float(
                        (future_prediction - persistence).square().mean().sqrt()
                    ),
                }
            )
    return {
        "summary": summarize_rows(rows, torch.cat(predictions), torch.cat(targets)),
        "windows": rows,
    }


def trainable_state(model):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def module_hashes(model):
    modules = {
        "selector": model.patch_selector,
        "predictor": model.future_predictor,
        "bridge": model.future_bridge,
    }
    if model.encoder_lora_tail is not None:
        modules["lora_tail"] = model.encoder_lora_tail
    return {name: parameter_sha256(module) for name, module in modules.items()}


def loss_gradient_contract(model, batch, agent, condition):
    report = {}
    for kind in ("planning", "auxiliary"):
        result = observed_forward(model, batch)
        loss = (
            agent.compute_loss(
                {}, {"trajectory": batch["ego_trajectory_target"]}, result
            )
            if kind == "planning"
            else model.compute_future_auxiliary_loss(
                result,
                batch["current_ego_status"],
                target_for_condition(batch, condition),
                batch["future_target_valid_mask"],
            )
        )
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            loss, [parameter for _, parameter in named], allow_unused=True
        )
        totals = {
            "patch_selector": 0.0,
            "future_predictor": 0.0,
            "future_bridge": 0.0,
            "encoder_lora_tail": 0.0,
        }
        for (name, _), gradient in zip(named, gradients):
            if gradient is not None:
                totals[name.split(".")[0]] += float(gradient.square().sum())
        report[kind] = {name: value**0.5 for name, value in totals.items()}
    if (
        report["auxiliary"]["patch_selector"] != 0
        or report["auxiliary"]["future_bridge"] != 0
    ):
        raise RuntimeError("Auxiliary gradient boundary violated")
    if any(
        parameter.grad is not None for parameter in model.baseline_model.parameters()
    ):
        raise RuntimeError("Frozen official parameter received gradient")
    return report


@torch.no_grad()
def branch_dependence(model, batch):
    normal = observed_forward(model, batch)
    predicted = normal["predicted_future_latents"]
    report = {}
    for name, replacement in (
        ("zero", torch.zeros_like(predicted)),
        ("batch_swap", predicted.roll(1, 0)),
    ):
        hook = model.future_predictor.register_forward_hook(
            lambda module, inputs, output, replacement=replacement: replacement
        )
        try:
            changed = observed_forward(model, batch)
        finally:
            hook.remove()
        report[name] = {
            "future_input_rms_change": float(
                (replacement - predicted).square().mean().sqrt()
            ),
            "trajectory_xy_change_m": float(
                (changed["trajectory"][..., :2] - normal["trajectory"][..., :2])
                .norm(dim=-1)
                .mean()
            ),
        }
    return report


def train_condition(
    agent, cache, records, specification, condition, seed, output, started
):
    condition_started = time.perf_counter()
    directory = output / f"{condition}_seed{seed}"
    directory.mkdir()
    model = build_model(agent._model, condition, seed)
    initial_hashes = module_hashes(model)
    initial_trainable = trainable_state(model)
    first_batch = get_training_batch(cache, list(range(specification["batch_size"])))
    with torch.no_grad():
        original = observed_forward(model, first_batch, enable_future_branch=False)[
            "trajectory"
        ]
        enabled = observed_forward(model, first_batch)["trajectory"]
        if not torch.equal(original, enabled):
            raise RuntimeError(
                "Initial branch must preserve official cached output bitwise"
            )
        tail_difference = None
        if model.encoder_lora_tail is not None:
            tail = model.encoder_lora_tail(first_batch["current_prefix_latents"])
            tail_difference = float(
                (tail - first_batch["current_patch_latents"]).abs().max()
            )
            torch.testing.assert_close(
                tail,
                first_batch["current_patch_latents"],
                atol=specification["prefix_equivalence_atol"],
                rtol=specification["prefix_equivalence_rtol"],
            )
    initial_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    training_indices = [
        index for index, row in enumerate(records) if row["split"] == "train"
    ]
    batch_generator = torch.Generator().manual_seed(seed + 10000)
    schedule = [
        torch.tensor(training_indices)[
            torch.randperm(len(training_indices), generator=batch_generator)[
                : specification["batch_size"]
            ]
        ].tolist()
        for _ in range(
            specification["auxiliary_warmup_updates"] + specification["joint_updates"]
        )
    ]
    schedule_hash = hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
    random_generator = torch.Generator().manual_seed(seed + 20000)
    groups = []
    for name, module, rate in (
        ("predictor", model.future_predictor, specification["learning_rate"]),
        ("bridge", model.future_bridge, specification["learning_rate"]),
        ("selector", model.patch_selector, specification["selector_learning_rate"]),
        ("lora", model.encoder_lora_tail, specification["lora_learning_rate"]),
    ):
        if module is not None:
            groups.append(
                {
                    "params": [
                        parameter
                        for parameter in module.parameters()
                        if parameter.requires_grad
                    ],
                    "lr": rate,
                    "name": name,
                }
            )
    curve, evaluations = [], {}
    stage, completed_update, optimizer, scheduler = "not_started", 0, None, None

    def checkpoint(label):
        torch.save(
            {
                "condition": condition,
                "seed": seed,
                "stage": stage,
                "completed_update": completed_update,
                "trainable_state": trainable_state(model),
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "random_selection_rng": random_generator.get_state(),
                "batch_schedule": schedule,
                "specification": specification,
                "curve": curve,
            },
            directory / f"{label}.pt",
        )

    try:
        for stage, updates in (
            ("auxiliary_warmup", specification["auxiliary_warmup_updates"]),
            ("joint", specification["joint_updates"]),
        ):
            completed_update = 0
            # Explicit new optimizer at stage boundary, shared by every condition.
            for group in groups:
                group.pop("initial_lr", None)
                group["lr"] = specification[
                    {
                        "predictor": "learning_rate",
                        "bridge": "learning_rate",
                        "selector": "selector_learning_rate",
                        "lora": "lora_learning_rate",
                    }[group["name"]]
                ]
            optimizer = torch.optim.AdamW(
                groups, weight_decay=specification["weight_decay"]
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step, total_updates=updates: (
                    0.1
                    + 0.9
                    * 0.5
                    * (1 + math.cos(math.pi * min(step, total_updates) / total_updates))
                ),
            )
            if stage == "joint":
                evaluations["0"] = {
                    split: evaluate(
                        model, cache, records, split, specification["batch_size"]
                    )
                    for split in ("train", "development")
                }
            for update in range(1, updates + 1):
                check_limits(specification, started, condition_started)
                offset = (
                    0
                    if stage == "auxiliary_warmup"
                    else specification["auxiliary_warmup_updates"]
                )
                batch = get_training_batch(cache, schedule[offset + update - 1])
                model.train()
                optimizer.zero_grad(set_to_none=True)
                if stage == "auxiliary_warmup":
                    context = batch["current_patch_latents"]
                    if model.encoder_lora_tail is not None:
                        context = model.encoder_lora_tail(
                            batch["current_prefix_latents"]
                        )
                    ids = torch.stack(
                        [
                            torch.randperm(512, generator=random_generator)[:4]
                            for _ in range(len(context))
                        ]
                    ).cuda()
                    hard = nn.functional.one_hot(ids, 512).to(context.dtype)
                    result = {
                        "current_prediction_context_latents": context,
                        "patch_selection": PatchSelection(
                            hard, hard, ids, hard @ model.patch_coordinates
                        ),
                    }
                    planning_loss = torch.zeros((), device="cuda")
                else:
                    result = observed_forward(model, batch)
                    planning_loss = agent.compute_loss(
                        {}, {"trajectory": batch["ego_trajectory_target"]}, result
                    )
                auxiliary_loss = model.compute_future_auxiliary_loss(
                    result,
                    batch["current_ego_status"],
                    target_for_condition(batch, condition),
                    batch["future_target_valid_mask"],
                )
                weight = (
                    specification["warmup_auxiliary_weight"]
                    if stage == "auxiliary_warmup"
                    else specification["future_auxiliary_weight"]
                )
                loss = planning_loss + weight * auxiliary_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    specification["gradient_clip_norm"],
                )
                if not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite training gradient")
                optimizer.step()
                scheduler.step()
                completed_update = update
                curve.append(
                    {
                        "stage": stage,
                        "update": update,
                        "planning_loss": float(planning_loss.detach()),
                        "auxiliary_loss": float(auxiliary_loss.detach()),
                        "gradient_norm_before_clip": float(norm),
                    }
                )
                if (
                    stage == "joint"
                    and update in specification["evaluation_joint_updates"]
                ):
                    evaluations[str(update)] = {
                        split: evaluate(
                            model, cache, records, split, specification["batch_size"]
                        )
                        for split in ("train", "development")
                    }
                    checkpoint(f"joint_update_{update}")
                if update % 25 == 0:
                    write_json(
                        directory / "progress.json",
                        {
                            "stage": stage,
                            "update": update,
                            "elapsed_seconds": time.perf_counter() - condition_started,
                            "latest": curve[-1],
                        },
                    )
                    print(
                        f"TRAIN {condition} seed={seed} {stage}={update}/{updates} loss={float(loss):.6f}",
                        flush=True,
                    )
            checkpoint(stage + "_complete")
        model.eval()
        gradient_contract = loss_gradient_contract(model, first_batch, agent, condition)
        dev_indices = [
            index for index, row in enumerate(records) if row["split"] == "development"
        ][: specification["batch_size"]]
        dependence = branch_dependence(model, get_training_batch(cache, dev_indices))
        report = {
            "condition": condition,
            "seed": seed,
            "initial_module_hashes": initial_hashes,
            "batch_schedule_sha256": schedule_hash,
            "trainable_parameters": initial_parameter_count,
            "lora_parameters": sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if name.startswith("encoder_lora_tail") and parameter.requires_grad
            ),
            "initial_baseline_output_bitwise_equal": True,
            "initial_lora_tail_max_abs_difference": tail_difference,
            "gradient_contract": gradient_contract,
            "first_dev_batch_dependence": dependence,
            "final_parameter_delta_norms": {
                name: float((parameter.detach().cpu() - initial_trainable[name]).norm())
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            },
            "evaluations": evaluations,
            "curve": curve,
            "wall_seconds": time.perf_counter() - condition_started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        }
        write_json(directory / "results.json", report)
        print(
            f"DONE {condition} seed={seed} dev_ADE={evaluations[str(specification['joint_updates'])]['development']['summary']['scene_macro_xy_ade_m']:.6f} elapsed={report['wall_seconds']:.1f}s",
            flush=True,
        )
        return report
    except BaseException:
        checkpoint("interrupted_state")
        write_json(
            directory / "interrupted_progress.json",
            {
                "stage": stage,
                "completed_update": completed_update,
                "evaluations": evaluations,
                "curve": curve,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--specification",
        type=Path,
        default=WORKSPACE
        / "configs/drive_jepa_selective_future/architecture_followup_v1.json",
    )
    arguments = parser.parse_args()
    output, specification_path = (
        arguments.output_directory.resolve(),
        arguments.specification.resolve(),
    )
    specification = json.loads(specification_path.read_text())
    if output.exists():
        raise RuntimeError("Fresh output required; never overwrite prior results")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Exactly one authorized physical GPU required")
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient shared GPU memory at admission")
    output.mkdir(parents=True)
    started = time.perf_counter()
    for event in (signal.SIGINT, signal.SIGTERM):
        signal.signal(event, request_stop)
    from audit_official_drive_jepa_evaluation import official_configuration
    from hydra.utils import instantiate

    baseline_spec, assets, configuration, official_root = official_configuration(
        WORKSPACE
    )
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=official_root, text=True
    ).strip()
    source_diff = subprocess.check_output(["git", "diff"], cwd=official_root)
    if source_commit != baseline_spec["official_source_commit"]:
        raise RuntimeError("Official source commit mismatch")
    for name in ("planning_checkpoint", "initialization_encoder"):
        if file_sha256(assets[name]["path"]) != assets[name]["sha256"]:
            raise RuntimeError("Official weight changed")
    with (
        (output / "strict_loading.log").open("w") as stream,
        contextlib.redirect_stdout(stream),
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
    records, cache = load_reused_cache(specification)
    model = build_model(agent._model, "mlp_recipe_control", 29)
    prefix = cache_current_prefix(
        agent, model, records, cache, output, specification, started
    )
    del model
    write_json(
        output / "execution_specification.json",
        {
            "specification": specification,
            "strict_loading": str(loading),
            "source_commit": source_commit,
            "baseline_hash": baseline_hash,
            "assets": assets,
            "specification_sha256": file_sha256(specification_path),
            "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
            "software": {"torch": torch.__version__, "cuda": torch.version.cuda},
            "source_hashes": {
                str(path.relative_to(WORKSPACE)): file_sha256(path)
                for path in (
                    Path(__file__),
                    WORKSPACE
                    / "src/planning_aware_future_prediction/models/drive_jepa_adaptive_future.py",
                    WORKSPACE
                    / "src/planning_aware_future_prediction/models/drive_jepa_selective_patch_future.py",
                )
            },
        },
    )
    reports = []
    for seed in specification["seeds"]:
        for condition in specification["conditions"]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            reports.append(
                train_condition(
                    agent,
                    cache,
                    records,
                    specification,
                    condition,
                    seed,
                    output,
                    started,
                )
            )
            write_json(
                output / "completed_conditions.json",
                [
                    {
                        "condition": report["condition"],
                        "seed": report["seed"],
                        "final": report["evaluations"]["200"],
                    }
                    for report in reports
                ],
            )
    if (
        parameter_sha256(agent._model) != baseline_hash
        or subprocess.check_output(["git", "diff"], cwd=official_root) != source_diff
    ):
        raise RuntimeError("Frozen official assets modified")
    for seed in specification["seeds"]:
        matched = [report for report in reports if report["seed"] == seed]
        assert len({report["batch_schedule_sha256"] for report in matched}) == 1
        assert (
            len({report["initial_module_hashes"]["bridge"] for report in matched}) == 1
        )
        assert (
            len(
                {
                    report["initial_module_hashes"]["predictor"]
                    for report in matched
                    if report["condition"] != "mlp_recipe_control"
                }
            )
            == 1
        )
        assert (
            len(
                {
                    report["initial_module_hashes"]["selector"]
                    for report in matched
                    if report["condition"]
                    not in ("mlp_recipe_control", "contextual_residual")
                }
            )
            == 1
        )
    write_json(
        output / "completion.json",
        {
            "completed": True,
            "condition_count": len(reports),
            "total_updates": len(reports) * 300,
            "wall_seconds": time.perf_counter() - started,
            "baseline_hash_before_after": baseline_hash,
            "prefix_cache_bytes": prefix["bytes"],
            "original_source_diff_preserved": True,
        },
    )
    print("ARCHITECTURE_FOLLOWUP_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
