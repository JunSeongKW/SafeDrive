"""Read-only checkpoint diagnosis before a bounded, hypothesis-driven retraining.

All interventions are sensitivity checks, not deployable/oracle performance claims.
No held-out/navtest, no edits to original cache, weights or official source.
"""

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from audit_official_drive_jepa_evaluation import official_configuration
from hydra.utils import instantiate
from run_drive_jepa_architecture_followup import (
    WORKSPACE,
    build_model,
    load_reused_cache,
    observed_forward,
)
from run_drive_jepa_selection_comparison import (
    get_training_batch,
    summarize_rows,
    write_json,
)
from validate_drive_jepa_selective_future_connection import parameter_sha256


def restore_delta(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(parameters) != set(checkpoint["trainable_state"]):
        raise RuntimeError("Checkpoint trainable keys mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = checkpoint["trainable_state"][name]
            if value.shape != parameter.shape:
                raise RuntimeError("Checkpoint shape mismatch")
            parameter.copy_(value)
    return checkpoint


def load_official_agent(output):
    specification, assets, configuration, official_root = official_configuration(
        WORKSPACE
    )
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
    agent._model.requires_grad_(False)
    return agent, {
        "specification": specification,
        "assets": assets,
        "loading": str(loading),
        "official_root": str(official_root),
    }


def cache_statistics(cache, records):
    summaries = {}
    for split in ("train", "development"):
        indices = [index for index, row in enumerate(records) if row["split"] == split]
        trajectory = cache["ego_trajectory_target"][indices]
        arclength = (
            (trajectory[:, 1:, :2] - trajectory[:, :-1, :2]).norm(dim=-1).sum(dim=-1)
        )
        weights = (5 + arclength).reciprocal()
        weights = weights / weights.mean()
        summaries[split] = {
            "windows": len(indices),
            "recordings": len({records[index]["recording_group"] for index in indices}),
            "weighted_effective_window_count": float(
                weights.sum().square() / weights.square().sum()
            ),
            "weight_min_median_max": [
                float(weights.min()),
                float(weights.median()),
                float(weights.max()),
            ],
            "arc_length_min_median_max_m": [
                float(arclength.min()),
                float(arclength.median()),
                float(arclength.max()),
            ],
            "fraction_arclength_under_one_m": float((arclength < 1).float().mean()),
            "future_valid_fraction_by_horizon": cache["future_target_valid_mask"][
                indices
            ]
            .float()
            .mean(dim=(0, 2))
            .tolist(),
            "status_per_channel_mean": cache["current_ego_status"][indices]
            .mean(dim=0)
            .tolist(),
            "status_per_channel_std": cache["current_ego_status"][indices]
            .std(dim=0)
            .tolist(),
        }
    return summaries


@torch.no_grad()
def intervention_evaluation(model, cache, records, split, mode):
    indices = [index for index, row in enumerate(records) if row["split"] == split]
    rows, predictions, targets = [], [], []
    for offset in range(0, len(indices), 8):
        selected = indices[offset : offset + 8]
        batch = get_training_batch(cache, selected)
        selection = model.patch_selector(
            batch["current_patch_latents"],
            batch["current_ego_status"],
            model.patch_coordinates,
        )
        hard = selection.hard_selection_weights
        persistence = (hard @ batch["current_patch_latents"])[:, :, None].expand(
            -1, -1, 4, -1
        )
        selected_target = torch.einsum(
            "bkn,btnd->bktd", hard, batch["future_target_latents"]
        )
        selected_valid = torch.einsum(
            "bkn,btn->bkt", hard, batch["future_target_valid_mask"].float()
        ).bool()
        memory_ratio = []

        def change_bridge(module, inputs, output, memory_ratio=memory_ratio):
            memory_ratio.append(
                (
                    output.square().mean(dim=(1, 2)).sqrt()
                    / inputs[0].square().mean(dim=(1, 2)).sqrt().clamp_min(1e-8)
                ).cpu()
            )
            gain = {"gain_zero": 0.0, "gain_quarter": 0.25, "gain_half": 0.5}.get(
                mode, 1.0
            )
            return output * gain

        def change_future(
            module,
            inputs,
            output,
            persistence=persistence,
            selected_valid=selected_valid,
            selected_target=selected_target,
        ):
            if mode == "current_persistence":
                return persistence
            if mode == "available_future_gt":
                return torch.where(selected_valid[..., None], selected_target, output)
            return output

        bridge_hook = model.future_bridge.register_forward_hook(change_bridge)
        future_hook = model.future_predictor.register_forward_hook(change_future)
        try:
            result = observed_forward(model, batch)
        finally:
            bridge_hook.remove()
            future_hook.remove()
        prediction, target = result["trajectory"], batch["ego_trajectory_target"]
        predictions.append(prediction.cpu())
        targets.append(target.cpu())
        xy_ade = (prediction[..., :2] - target[..., :2]).norm(dim=-1).mean(dim=-1)
        for position, index in enumerate(selected):
            metadata = records[index]
            rows.append(
                {
                    "token": metadata["current_frame_token"],
                    "recording": metadata["recording_group"],
                    "scene_token": metadata["scene_token"],
                    "command": metadata["command_raw_index"],
                    "xy_ade_m": float(xy_ade[position]),
                    "memory_residual_rms_ratio": float(memory_ratio[0][position]),
                    "selected_patch_ids": result["patch_selection"]
                    .selected_patch_indices[position]
                    .cpu()
                    .tolist(),
                }
            )
    summary = summarize_rows(rows, torch.cat(predictions), torch.cat(targets))
    summary["memory_residual_rms_ratio_mean"] = float(
        np.mean([row["memory_residual_rms_ratio"] for row in rows])
    )
    summary["memory_residual_rms_ratio_max"] = max(
        row["memory_residual_rms_ratio"] for row in rows
    )
    return {"summary": summary, "windows": rows}


def gradient_alignment(model, agent, cache, records, batch_schedule):
    measurements = []
    for selected in batch_schedule[:8]:
        if any(records[index]["split"] != "train" for index in selected):
            raise RuntimeError("Only training batches for gradient diagnosis")
        batch = get_training_batch(cache, selected)
        result = observed_forward(model, batch)
        planning = agent.compute_loss(
            {}, {"trajectory": batch["ego_trajectory_target"]}, result
        )
        auxiliary = model.compute_future_auxiliary_loss(
            result,
            batch["current_ego_status"],
            batch["future_target_latents"],
            batch["future_target_valid_mask"],
        )
        parameters = list(model.future_predictor.parameters())
        planning_gradients = torch.autograd.grad(
            planning, parameters, retain_graph=True, allow_unused=True
        )
        auxiliary_gradients = torch.autograd.grad(
            auxiliary, parameters, allow_unused=True
        )
        dot = sum(
            (left * right).sum()
            for left, right in zip(planning_gradients, auxiliary_gradients)
            if left is not None and right is not None
        )
        planning_norm = sum(
            value.square().sum() for value in planning_gradients if value is not None
        ).sqrt()
        auxiliary_norm = sum(
            value.square().sum() for value in auxiliary_gradients if value is not None
        ).sqrt()
        prediction = result["trajectory"].detach().requires_grad_(True)
        official_loss = agent.compute_loss(
            {},
            {"trajectory": batch["ego_trajectory_target"]},
            {"trajectory": prediction},
        )
        uniform_ade = (
            (prediction[..., :2] - batch["ego_trajectory_target"][..., :2])
            .norm(dim=-1)
            .mean()
        )
        official_gradient = torch.autograd.grad(
            official_loss, prediction, retain_graph=True
        )[0]
        ade_gradient = torch.autograd.grad(uniform_ade, prediction)[0]
        measurements.append(
            {
                "predictor_planning_gradient_norm": float(planning_norm),
                "predictor_scaled_auxiliary_gradient_norm": float(
                    0.01 * auxiliary_norm
                ),
                "predictor_gradient_cosine": float(
                    dot / (planning_norm * auxiliary_norm).clamp_min(1e-12)
                ),
                "official_vs_xy_ade_output_gradient_cosine": float(
                    (official_gradient * ade_gradient).sum()
                    / (official_gradient.norm() * ade_gradient.norm()).clamp_min(1e-12)
                ),
            }
        )
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output = arguments.output_directory.resolve()
    if output.exists():
        raise RuntimeError("Fresh diagnostic directory required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("One authorized GPU only")
    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        raise RuntimeError("Need 12GiB free at admission")
    output.mkdir(parents=True)
    torch.set_num_threads(1)
    started = time.perf_counter()
    specification = json.loads(
        (
            WORKSPACE
            / "configs/drive_jepa_selective_future/architecture_followup_v1.json"
        ).read_text()
    )
    agent, source = load_official_agent(output)
    baseline_hash = parameter_sha256(agent._model)
    records, cache = load_reused_cache(specification)
    previous = (
        WORKSPACE
        / "outputs/drive_jepa_selective_future/architecture_followup_v1_20261002"
    )
    reports = []
    for condition in ("mlp_recipe_control", "ego_query_residual"):
        for seed in (29, 47, 83):
            model = build_model(agent._model, condition, seed)
            directory = previous / f"{condition}_seed{seed}"
            restore_delta(model, directory / "auxiliary_warmup_complete.pt")
            warmup_ids = {}
            with torch.no_grad():
                for offset in range(0, len(records), 8):
                    batch = get_training_batch(
                        cache, list(range(offset, min(offset + 8, len(records))))
                    )
                    selection = model.patch_selector(
                        batch["current_patch_latents"],
                        batch["current_ego_status"],
                        model.patch_coordinates,
                    )
                    for position, row in enumerate(
                        selection.selected_patch_indices.cpu().tolist()
                    ):
                        warmup_ids[
                            records[offset + position]["current_frame_token"]
                        ] = set(row)
            checkpoint_reports = {}
            for update in (50, 200):
                checkpoint = restore_delta(
                    model, directory / f"joint_update_{update}.pt"
                )
                modes = (
                    "normal",
                    "gain_zero",
                    "gain_quarter",
                    "gain_half",
                    "current_persistence",
                    "available_future_gt",
                )
                evaluations = {split: {} for split in ("train", "development")}
                for split, split_evaluations in evaluations.items():
                    for mode in modes:
                        if (
                            torch.cuda.mem_get_info()[0] < 6 * 2**30
                            or time.perf_counter() - started > 1200
                        ):
                            raise RuntimeError("Diagnostic reserve/time cap")
                        split_evaluations[mode] = intervention_evaluation(
                            model, cache, records, split, mode
                        )
                    rows = split_evaluations["normal"]["windows"]
                    split_evaluations["normal"]["summary"][
                        "mean_patch_retention_from_warmup"
                    ] = float(
                        np.mean(
                            [
                                len(
                                    set(row["selected_patch_ids"])
                                    & warmup_ids[row["token"]]
                                )
                                / 4
                                for row in rows
                            ]
                        )
                    )
                checkpoint_reports[str(update)] = {
                    "evaluations": evaluations,
                    "training_gradient_alignment": gradient_alignment(
                        model, agent, cache, records, checkpoint["batch_schedule"][100:]
                    ),
                }
                print(f"DIAGNOSED {condition} seed={seed} update={update}", flush=True)
            reports.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "checkpoints": checkpoint_reports,
                }
            )
            write_json(output / f"{condition}_seed{seed}.json", reports[-1])
            del model
            torch.cuda.empty_cache()
    if parameter_sha256(agent._model) != baseline_hash:
        raise RuntimeError("Original baseline changed")
    summary = {
        "source": source,
        "baseline_hash_unchanged": baseline_hash,
        "cache_statistics": cache_statistics(cache, records),
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "conditions": [
            {
                "condition": report["condition"],
                "seed": report["seed"],
                "checkpoints": {
                    step: {
                        "evaluations": {
                            split: {
                                mode: value["summary"] for mode, value in modes.items()
                            }
                            for split, modes in details["evaluations"].items()
                        },
                        "training_gradient_alignment": details[
                            "training_gradient_alignment"
                        ],
                    }
                    for step, details in report["checkpoints"].items()
                },
            }
            for report in reports
        ],
        "interpretation": "posthoc_sensitivity_gradient_diagnostics_not_causal_identification_or_tuned_evaluation",
    }
    write_json(output / "diagnosis_summary.json", summary)
    print("DIAGNOSIS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
