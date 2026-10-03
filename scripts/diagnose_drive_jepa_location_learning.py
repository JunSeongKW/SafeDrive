"""Fixed-budget read-only pipeline diagnosis, not another training experiment."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from diagnose_drive_jepa_learning_limitations import load_official_agent
from run_drive_jepa_architecture_followup import build_model, observed_forward
from run_drive_jepa_selection_comparison import file_sha256, write_json
from train_drive_jepa_spatial_regions import inspect_gradient_contract
from validate_drive_jepa_selective_future_connection import parameter_sha256

from planning_aware_future_prediction.models.spatial_region_future import (
    SpatialRegionSelector,
    compute_planner_retention_loss,
)
from planning_aware_future_prediction.selector_location_diagnostics import (
    capture_ego_query_scores,
    gradient_cosine_or_none,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def restore_region_model(agent, checkpoint_path, seed):
    saved = torch.load(checkpoint_path, map_location="cpu")
    if saved["completed_update"] != 800:
        raise RuntimeError("Only registered last checkpoint")
    options = saved["specification"]["conditions"][saved["condition"]]
    if (options["region_side"], options["region_budget"]) != (2, 8):
        raise RuntimeError("Fixed region size and count required")
    model = build_model(agent._model, "ego_query_residual", seed)
    model.patch_selector = SpatialRegionSelector(
        model.patch_selector, 16, 32, 2, 8
    ).cuda()
    parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("baseline_model.")
    }
    if set(parameters) != set(saved["extension_parameters"]):
        raise RuntimeError("Strict extension key mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != saved["extension_parameters"][name].shape:
                raise RuntimeError("Strict extension shape mismatch")
            parameter.copy_(saved["extension_parameters"][name])
    return model.eval()


def guard(specification, started):
    if time.perf_counter() - started > specification["maximum_wall_seconds"]:
        raise RuntimeError("Read-only diagnosis time limit")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError("Shared GPU reserve reached")
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Own memory cap reached")


@torch.no_grad()
def attention_diagnostics(model, current, outputs):
    observed_grid = current.reshape(-1, 16, 32, 1024).permute(0, 3, 1, 2)
    pooled = (
        model.baseline_model.avg_pool(observed_grid).flatten(-2, -1).permute(0, 2, 1)
    )
    current_memory = model.baseline_model.image_fc(pooled)
    selection = outputs["patch_selection"]
    coordinates = selection.selection_weights @ model.patch_coordinates
    bridge = model.future_bridge
    memory = (
        bridge.future_projection(outputs["predicted_future_latents"])
        + bridge.coordinate_projection(coordinates)[:, :, None]
        + bridge.time_embeddings.weight[None, None]
    ).flatten(1, 2)
    _, weights = bridge.attention(
        current_memory, memory, memory, need_weights=True, average_attn_weights=False
    )
    probabilities = weights.clamp_min(1e-12)
    normalized_entropy = -(probabilities * probabilities.log()).sum(-1) / math.log(
        weights.shape[-1]
    )
    residual = outputs["future_memory_residual"]
    return {
        "bridge_attention_normalized_entropy": float(normalized_entropy.mean()),
        "bridge_attention_mean_max_weight": float(weights.max(-1).values.mean()),
        "uniform_weight": 1 / weights.shape[-1],
        "residual_to_current_memory_rms_ratio": float(
            residual.square().mean().sqrt() / current_memory.square().mean().sqrt()
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    # Official imports temporarily change cwd; never retain relative output paths.
    args.config = args.config.resolve()
    args.output_directory = args.output_directory.resolve()
    specification = json.loads(args.config.read_text())
    args.output_directory.mkdir(parents=True, exist_ok=False)
    write_json(args.output_directory / "specification.json", specification)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Exactly one approved GPU required")
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("GPU launch headroom insufficient")
    started = time.perf_counter()
    run = WORKSPACE / specification["region_run"]
    training_specification = json.loads((run / "specification.json").read_text())
    index = json.loads((WORKSPACE / training_specification["reused_cache"]).read_text())
    grouped = defaultdict(list)
    for record in index["records"]:
        if record["split"] == "train":
            grouped[record["recording_group"]].append(record)
    stable_hash = lambda value: hashlib.sha256(
        f"{specification['sampling_hash_salt']}:{value}".encode()
    ).hexdigest()
    chosen = [
        min(grouped[group], key=lambda row: stable_hash(row["current_frame_token"]))
        for group in sorted(grouped, key=stable_hash)[
            : specification["training_recording_count"]
        ]
    ]
    write_json(args.output_directory / "resolved_training_windows.json", chosen)
    cached_windows = []
    for record in chosen:
        if file_sha256(record["cache_file"]) != record["cache_sha256"]:
            raise RuntimeError("Cache hash changed")
        cached_windows.append(
            torch.load(record["cache_file"], map_location="cpu", weights_only=True)
        )
    agent, source = load_official_agent(args.output_directory)
    original_hash = parameter_sha256(agent._model)
    write_json(args.output_directory / "source.json", source)
    summaries = {}
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            checkpoint_path = run / f"{condition}_seed{seed}/complete.pt"
            model = restore_region_model(agent, checkpoint_path, seed)
            model_hash_before = parameter_sha256(model)
            rows, swaps, contracts = [], [], []
            for record, cached in zip(chosen, cached_windows):
                guard(specification, started)
                observed_batch = {
                    key: value[None].cuda() for key, value in cached.items()
                }
                # A read-only gradient audit may read GT in loss, never selector forward.
                model.patch_selector.explicit_region_indices = None
                reference = observed_forward(model, observed_batch)
                with capture_ego_query_scores(model.patch_selector.scorer) as captured:
                    outputs = observed_forward(model, observed_batch)
                    scores = captured["scores"]
                    score_values = scores.detach()[0]
                    difference = float(
                        (reference["trajectory"] - outputs["trajectory"])
                        .abs()
                        .max()
                        .detach()
                    )
                    if difference > specification[
                        "forward_tolerance"
                    ] or not torch.equal(
                        reference["patch_selection"].selected_region_indices,
                        outputs["patch_selection"].selected_region_indices,
                    ):
                        raise RuntimeError("Score tracing changed production forward")
                    loss = agent.compute_loss(
                        {},
                        {"trajectory": observed_batch["ego_trajectory_target"]},
                        outputs,
                    )
                    score_gradient = torch.autograd.grad(
                        loss, scores, retain_graph=True
                    )[0][0].detach()
                    retention = compute_planner_retention_loss(
                        model,
                        observed_batch["current_patch_latents"],
                        observed_batch["current_ego_status"],
                        outputs["patch_selection"],
                    )
                    retention_gradient = torch.autograd.grad(retention, scores)[0][
                        0
                    ].detach()
                baseline_prediction = outputs["trajectory"].detach()
                original_loss = float(loss.detach())
                selected = (
                    outputs["patch_selection"].selected_region_indices[0].tolist()
                )
                model.patch_selector.explicit_region_indices = torch.tensor(
                    [selected], device="cuda"
                )
                matched_explicit = observed_forward(model, observed_batch)["trajectory"]
                explicit_error = float(
                    (matched_explicit - baseline_prediction).abs().max().detach()
                )
                if explicit_error > specification["forward_tolerance"]:
                    raise RuntimeError(
                        "Explicit-ID reference changed production forward"
                    )
                model.patch_selector.explicit_region_indices = None
                probability = (
                    score_values / model.patch_selector.scorer.temperature
                ).softmax(-1)
                sorted_scores = score_values.sort(descending=True).values
                row = {
                    "token": record["current_frame_token"],
                    "recording": record["recording_group"],
                    "selected_region_ids": selected,
                    "score_trace_forward_max_error": difference,
                    "explicit_reference_forward_max_error": explicit_error,
                    "planning_score_gradient_norm": float(score_gradient.norm()),
                    "weighted_retention_score_gradient_norm": float(
                        0.1 * retention_gradient.norm()
                    ),
                    "planning_retention_score_gradient_cosine": gradient_cosine_or_none(
                        score_gradient, retention_gradient
                    ),
                    "score_softmax_normalized_entropy": float(
                        -(probability * probability.clamp_min(1e-12).log()).sum()
                        / math.log(len(probability))
                    ),
                    "selection_cutoff_margin": float(
                        sorted_scores[7] - sorted_scores[8]
                    ),
                    **attention_diagnostics(
                        model, observed_batch["current_patch_latents"], outputs
                    ),
                }
                # Same grad-enabled execution mode for all forward perturbations.
                disabled = observed_forward(
                    model, observed_batch, enable_future_branch=False
                )["trajectory"]
                row["branch_off_trajectory_xy_change_m"] = float(
                    (disabled[..., :2] - baseline_prediction[..., :2])
                    .norm(dim=-1)
                    .mean()
                    .detach()
                )
                predicted = outputs["predicted_future_latents"].detach()
                for name, replacement in (
                    (
                        "future_slot_mean",
                        predicted.mean(1, keepdim=True).expand_as(predicted),
                    ),
                    ("future_slot_shuffle", predicted.roll(1, 1)),
                ):
                    handle = model.future_predictor.register_forward_hook(
                        lambda module, inputs, result, replacement=replacement: (
                            replacement
                        )
                    )
                    try:
                        changed = observed_forward(model, observed_batch)["trajectory"]
                    finally:
                        handle.remove()
                    row[name + "_future_rms_change"] = float(
                        (replacement - predicted).square().mean().sqrt()
                    )
                    row[name + "_trajectory_xy_change_m"] = float(
                        (changed[..., :2] - baseline_prediction[..., :2])
                        .norm(dim=-1)
                        .mean()
                        .detach()
                    )
                unselected = [
                    candidate
                    for candidate in score_values.argsort(descending=True).tolist()
                    if candidate not in selected
                ]
                generator = np.random.default_rng(
                    int(stable_hash(record["current_frame_token"])[:8], 16)
                )
                candidates = unselected[: specification["near_boundary_substitutions"]]
                remaining = [
                    candidate for candidate in unselected if candidate not in candidates
                ]
                candidates += generator.choice(
                    remaining, size=specification["random_substitutions"], replace=False
                ).tolist()
                outgoing = selected[-1]
                for incoming in candidates:
                    variant = [*selected[:-1], incoming]
                    model.patch_selector.explicit_region_indices = torch.tensor(
                        [variant], device="cuda"
                    )
                    changed = observed_forward(model, observed_batch)
                    changed_loss = float(
                        agent.compute_loss(
                            {},
                            {"trajectory": observed_batch["ego_trajectory_target"]},
                            changed,
                        ).detach()
                    )
                    incoming_delta = (
                        score_values[outgoing] + 0.001 - score_values[incoming]
                    )
                    outgoing_delta = (
                        score_values[unselected[0]] - 0.001 - score_values[outgoing]
                    )
                    swaps.append(
                        {
                            "token": record["current_frame_token"],
                            "incoming": incoming,
                            "outgoing": outgoing,
                            "actual_planning_loss_change": changed_loss - original_loss,
                            "score_linearized_loss_change": float(
                                score_gradient[incoming] * incoming_delta
                                + score_gradient[outgoing] * outgoing_delta
                            ),
                            "trajectory_xy_change_m": float(
                                (
                                    changed["trajectory"][..., :2]
                                    - baseline_prediction[..., :2]
                                )
                                .norm(dim=-1)
                                .mean()
                                .detach()
                            ),
                        }
                    )
                model.patch_selector.explicit_region_indices = None
                # Actual same-size arbitrary relocation of all 8 sites.
                random_ids = generator.choice(128, size=8, replace=False).tolist()
                model.patch_selector.explicit_region_indices = torch.tensor(
                    [random_ids], device="cuda"
                )
                random_prediction = observed_forward(model, observed_batch)[
                    "trajectory"
                ]
                row["random_relocation_trajectory_xy_change_m"] = float(
                    (random_prediction[..., :2] - baseline_prediction[..., :2])
                    .norm(dim=-1)
                    .mean()
                    .detach()
                )
                model.patch_selector.explicit_region_indices = None
                if not contracts:
                    contracts.append(
                        inspect_gradient_contract(model, agent, observed_batch, True)
                    )
                rows.append(row)
                del reference, outputs, loss, retention, changed, random_prediction
            if parameter_sha256(model) != model_hash_before:
                raise RuntimeError("Read-only diagnosis changed model weights")
            effective = [
                item
                for item in swaps
                if abs(item["actual_planning_loss_change"])
                > specification["loss_change_noise_floor"]
                and abs(item["score_linearized_loss_change"])
                > specification["loss_change_noise_floor"]
            ]
            summary = {
                key: float(np.mean([row[key] for row in rows if row[key] is not None]))
                for key in rows[0]
                if key not in ("token", "recording", "selected_region_ids")
            }
            summary.update(
                {
                    "hard_swaps": len(swaps),
                    "nontrivial_hard_swaps": len(effective),
                    "score_surrogate_sign_agreement": float(
                        np.mean(
                            [
                                np.sign(item["actual_planning_loss_change"])
                                == np.sign(item["score_linearized_loss_change"])
                                for item in effective
                            ]
                        )
                    )
                    if effective
                    else None,
                    "hard_swap_xy_change_mean_m": float(
                        np.mean([item["trajectory_xy_change_m"] for item in swaps])
                    ),
                    "fraction_tested_swaps_improving_planning_loss": float(
                        np.mean(
                            [
                                item["actual_planning_loss_change"]
                                < -specification["loss_change_noise_floor"]
                                for item in swaps
                            ]
                        )
                    ),
                }
            )
            report = {
                "condition": condition,
                "seed": seed,
                "summary": summary,
                "windows": rows,
                "hard_swap_probes": swaps,
                "gradient_contract": contracts[0],
                "model_hash_unchanged": model_hash_before,
                "checkpoint_sha256": file_sha256(checkpoint_path),
            }
            write_json(args.output_directory / f"{condition}_seed{seed}.json", report)
            summaries[f"{condition}_seed{seed}"] = summary
            print(
                json.dumps({"condition": condition, "seed": seed, "summary": summary}),
                flush=True,
            )
            del model
            torch.cuda.empty_cache()
    if parameter_sha256(agent._model) != original_hash:
        raise RuntimeError("Official planner changed")
    write_json(
        args.output_directory / "summary.json",
        {
            "summaries": summaries,
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "baseline_hash_unchanged": original_hash,
            "optimizer_updates": 0,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True
            ).strip(),
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "limitations": [
                "Local logits surrogate vs finite hard transitions, not an unbiased derivative proof.",
                "Inference perturbations are dependence probes, not retrained no-future controls.",
                "Attention weights and teacher retention are not semantic ground truth.",
            ],
        },
    )


if __name__ == "__main__":
    main()
