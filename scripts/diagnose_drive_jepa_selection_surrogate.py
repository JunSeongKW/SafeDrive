"""Read-only, train-only comparison of local routing gradients and hard swaps."""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from diagnose_drive_jepa_learning_limitations import load_official_agent
from run_drive_jepa_architecture_followup import build_model, observed_forward
from run_drive_jepa_selection_comparison import file_sha256, write_json
from validate_drive_jepa_selective_future_connection import parameter_sha256

from planning_aware_future_prediction.selection_surrogate_diagnostics import (
    correlation_or_none,
    generate_single_slot_substitutions,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def restore_complete_extension(model, checkpoint_path):
    saved = torch.load(checkpoint_path, map_location="cpu")
    if saved["completed_update"] != 800:
        raise RuntimeError("Only preregistered final800 checkpoint is admitted")
    parameters = {
        name: value
        for name, value in model.named_parameters()
        if not name.startswith("baseline_model.")
    }
    if set(parameters) != set(saved["extension_parameters"]):
        raise RuntimeError("Extension checkpoint key mismatch")
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = saved["extension_parameters"][name]
            if parameter.shape != value.shape:
                raise RuntimeError("Extension checkpoint shape mismatch")
            parameter.copy_(value)


def check_resources(specification, started):
    if datetime.now().astimezone() >= datetime.fromisoformat(
        specification["deadline_iso"]
    ):
        raise RuntimeError("Morning deadline reached")
    if time.monotonic() - started > specification["maximum_wall_seconds"]:
        raise RuntimeError("Diagnostic wall-time limit")
    if torch.cuda.mem_get_info()[0] < specification["running_reserve_gib"] * 2**30:
        raise RuntimeError("Shared GPU reserve reached")
    if (
        torch.cuda.max_memory_allocated()
        > specification["maximum_peak_allocated_gib"] * 2**30
    ):
        raise RuntimeError("Diagnostic allocated-memory cap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    specification = json.loads(arguments.config.resolve().read_text())
    output = arguments.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "specification.json", specification)
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("One approved GPU required")
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient GPU admission headroom")
    started = time.monotonic()
    torch.set_num_threads(1)
    cache_index = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    grouped = defaultdict(list)
    for row in cache_index["records"]:
        if row["split"] == "train":
            grouped[row["recording_group"]].append(row)
    salt = specification["sampling_hash_salt"]
    stable_key = lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()
    recordings = sorted(grouped, key=stable_key)[
        : specification["training_recording_count"]
    ]
    chosen = [
        min(grouped[recording], key=lambda row: stable_key(row["current_frame_token"]))
        for recording in recordings
    ]
    write_json(output / "resolved_training_windows.json", chosen)
    current_inputs = []
    for row in chosen:
        if file_sha256(row["cache_file"]) != row["cache_sha256"]:
            raise RuntimeError("Cache checksum mismatch")
        cached = torch.load(row["cache_file"], map_location="cpu", weights_only=True)
        # No future target feature or future validity is passed to online forward.
        current_inputs.append(
            {
                key: cached[key]
                for key in (
                    "current_patch_latents",
                    "current_ego_status",
                    "ego_trajectory_target",
                )
            }
        )
    agent, source = load_official_agent(output)
    write_json(output / "source.json", source)
    original_hash = parameter_sha256(agent._model)
    all_rows, summaries = [], {}
    for condition, architecture in specification["condition_architectures"].items():
        for seed in specification["seeds"]:
            model = build_model(agent._model, architecture, seed).eval()
            checkpoint_path = (
                WORKSPACE
                / specification["checkpoint_root"]
                / f"{condition}_seed{seed}/complete.pt"
            )
            restore_complete_extension(model, checkpoint_path)
            rows = []
            for row, cached in zip(chosen, current_inputs):
                check_resources(specification, started)
                batch = {
                    key: value.unsqueeze(0).cuda() for key, value in cached.items()
                }
                prediction = observed_forward(model, batch)
                loss = agent.compute_loss(
                    {}, {"trajectory": batch["ego_trajectory_target"]}, prediction
                )
                selection = prediction["patch_selection"]
                weight_gradient = (
                    torch.autograd.grad(loss, selection.selection_weights)[0][0]
                    .detach()
                    .cpu()
                )
                original_ids = selection.selected_patch_indices[0].tolist()
                sampling_seed = int(
                    stable_key(f"{row['current_frame_token']}:{seed}")[:16], 16
                )
                variants, replacements = generate_single_slot_substitutions(
                    original_ids,
                    512,
                    specification["replacements_per_slot"],
                    sampling_seed,
                )
                repeated = {
                    key: value.expand(len(variants), *value.shape[1:])
                    for key, value in batch.items()
                }
                with torch.no_grad():
                    counterfactual = observed_forward(
                        model,
                        repeated,
                        selected_patch_indices=torch.tensor(
                            variants, device="cuda", dtype=torch.long
                        ),
                    )["trajectory"]
                    discrepancy = float(
                        (prediction["trajectory"] - counterfactual[:1]).abs().max()
                    )
                    if discrepancy > specification["forward_comparison_tolerance"]:
                        raise RuntimeError(
                            f"Gradient/batched-reference mismatch: {discrepancy}"
                        )
                    actual_losses = [
                        float(
                            agent.compute_loss(
                                {},
                                {"trajectory": batch["ego_trajectory_target"]},
                                {"trajectory": trajectory[None]},
                            )
                        )
                        for trajectory in counterfactual
                    ]
                for replacement, actual_loss in zip(replacements, actual_losses[1:]):
                    slot, previous_id, new_id = replacement
                    rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "token": row["current_frame_token"],
                            "recording": row["recording_group"],
                            "slot": slot,
                            "previous_patch_id": previous_id,
                            "replacement_patch_id": new_id,
                            "linearized_loss_change": float(
                                weight_gradient[slot, new_id]
                                - weight_gradient[slot, previous_id]
                            ),
                            "actual_loss_change": actual_loss - actual_losses[0],
                            "reference_forward_max_difference": discrepancy,
                        }
                    )
                del prediction, selection, loss, batch, repeated, counterfactual
            noise_floor = specification["loss_change_noise_floor"]
            informative = [
                row
                for row in rows
                if abs(row["linearized_loss_change"]) > noise_floor
                and abs(row["actual_loss_change"]) > noise_floor
            ]
            summaries[f"{condition}_seed{seed}"] = {
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "substitutions": len(rows),
                "informative_substitutions": len(informative),
                "sign_agreement": float(
                    np.mean(
                        [
                            np.sign(row["linearized_loss_change"])
                            == np.sign(row["actual_loss_change"])
                            for row in informative
                        ]
                    )
                )
                if informative
                else None,
                "pearson_loss_change_correlation": correlation_or_none(
                    [row["linearized_loss_change"] for row in rows],
                    [row["actual_loss_change"] for row in rows],
                ),
                "fraction_of_sampled_substitutions_improving_training_loss": float(
                    np.mean([row["actual_loss_change"] < -noise_floor for row in rows])
                ),
            }
            all_rows.extend(rows)
            write_json(
                output / "partial_results.json",
                {"summaries": summaries, "rows": all_rows},
            )
            del model
            torch.cuda.empty_cache()
    if parameter_sha256(agent._model) != original_hash:
        raise RuntimeError("Original model changed")
    write_json(
        output / "results.json",
        {
            "complete": True,
            "summaries": summaries,
            "rows": all_rows,
            "wall_seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "original_parameter_hash_unchanged": original_hash,
            "interpretation": "dL/d(selection_weights) predicts a finite one-hot displacement; not the complete score-gradient update or a causal/world importance estimate. Train-only repeated-model sensitivity, no optimizer updates.",
        },
    )


if __name__ == "__main__":
    main()
