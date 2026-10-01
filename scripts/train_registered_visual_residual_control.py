"""ONE registered visual residual contrast on old cache, not an A–F sweep."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import train_target_supervision_ablation as exploration
from diagnose_visual_future_prediction_variance import load_existing_experiment
from evaluate_kinematic_and_ridge_references import evaluate_reference_methods
from train_bounded_future_prediction_followup import (
    capture_rng_states,
    generate_recording_balanced_batch_sequence,
)

from planning_aware_future_prediction.models.residual_future_supervision import (
    ResidualFutureSupervisionPilot,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    output_directory = arguments.output_directory.resolve()
    if output_directory.exists() or not output_directory.is_relative_to(
        exploration.PROJECT_ROOT / "outputs"
    ):
        raise ValueError("require new workspace outputs, never replace prior runs")
    output_directory.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    registered = json.loads(
        (
            exploration.PROJECT_ROOT
            / "configs/exploration/pilot_foundation_decision_v1.json"
        ).read_text()
    )
    specification = registered["small_cache_visual_comparison"]
    if (
        specification["maximum_new_updates"] != 5000
        or specification["optimizer_updates"] != 1000
    ):
        raise ValueError("only registered bounded comparison supported")
    started = time.perf_counter()
    _, windows, normalization, _ = load_existing_experiment("cpu")
    subsets = {
        split: [window for window in windows if window["metadata"]["split"] == split]
        for split in ("train", "development")
    }
    old_directory = (
        exploration.PROJECT_ROOT
        / "outputs/future_prediction_diagnostics/bounded_followup_1000_v1"
    )
    report = {
        "baseline_commit": "607da52",
        "device": "cpu",
        "gpu_use": False,
        "registered_comparison": specification,
        "runs": [],
        "new_optimizer_updates": 0,
        "cache_index_sha256": exploration.file_sha256(
            exploration.PROJECT_ROOT
            / registered["old_cache_directory"]
            / "cache_index.json"
        ),
        "source_sha256": {
            str(
                Path(__file__).relative_to(exploration.PROJECT_ROOT)
            ): exploration.file_sha256(__file__)
        },
    }
    condition = {
        "label": "C",
        "enable_future_branch": True,
        "visual_weight": 0.1,
        "spatial_weight": 0.0,
    }
    for seed in specification["seeds"]:
        initial = torch.load(
            old_directory / f"shared_initial_seed{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        sequence = generate_recording_balanced_batch_sequence(
            subsets["train"],
            seed,
            specification["optimizer_updates"],
            specification["batch_size"],
        )
        for residual in (False, True):
            name = "C_visual_residual" if residual else "C_absolute"
            directory = output_directory / f"seed{seed}_{name}"
            directory.mkdir()
            model = ResidualFutureSupervisionPilot(residual_visual=residual)
            model.load_state_dict(initial, strict=True)
            model.set_train_target_normalization(normalization)
            model.configure_active_modules(True)
            if residual:
                model.initialize_residual_output_heads()
            row = {
                "seed": seed,
                "condition": name,
                "common_initial_source": str(
                    old_directory / f"shared_initial_seed{seed}.pt"
                ),
                "common_initial_sha256": exploration.state_sha256(initial),
                "effective_initial_sha256": exploration.state_sha256(
                    model.state_dict()
                ),
                "same_batch_sequence_sha256": __import__("hashlib")
                .sha256(json.dumps(sequence).encode())
                .hexdigest(),
                "only_architecture_difference": "visual current ROI skip + zeroed visual output rows",
                "ego_spatial_residual_enabled": False,
                "evaluations": [],
            }
            run_started = time.perf_counter()
            if not residual and seed == 29:
                checkpoint = torch.load(
                    old_directory / "seed29_C/checkpoint_update1000.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model.load_state_dict(checkpoint["model"], strict=True)
                row.update(
                    reused_completed_run=True,
                    completed_updates=1000,
                    original_checkpoint=str(
                        old_directory / "seed29_C/checkpoint_update1000.pt"
                    ),
                )
                for split, subset in subsets.items():
                    summary, records = evaluate_reference_methods(
                        subset, normalization, None, model, condition
                    )
                    row["evaluations"].append(
                        {"update": 1000, "split": split, **summary["C"]}
                    )
                    (directory / f"{split}_per_window_update1000.json").write_text(
                        json.dumps(records["C"], indent=2) + "\n"
                    )
            else:
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                optimizer = torch.optim.AdamW(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    lr=specification["optimizer"]["learning_rate"],
                    weight_decay=specification["optimizer"]["weight_decay"],
                )
                row["training_curve"] = []
                row["completed_updates"] = 0
                for update in range(specification["optimizer_updates"] + 1):
                    if (
                        time.perf_counter() - started
                        > specification["whole_comparison_seconds"]
                        or time.perf_counter() - run_started
                        > specification["per_run_seconds"]
                    ):
                        row["stopped_by_time_cap"] = True
                        break
                    if update in specification["evaluation_updates"]:
                        model.eval()
                        for split, subset in subsets.items():
                            summary, records = evaluate_reference_methods(
                                subset, normalization, None, model, condition
                            )
                            row["evaluations"].append(
                                {"update": update, "split": split, **summary["C"]}
                            )
                            if update in (0, 1000):
                                (
                                    directory
                                    / f"{split}_per_window_update{update:04d}.json"
                                ).write_text(json.dumps(records["C"], indent=2) + "\n")
                        torch.save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "optimizer_updates": update,
                                "normalization": normalization,
                                "sampler_sequence": sequence,
                                "sampler_next_update": update + 1,
                                "rng_states": capture_rng_states(),
                                "registered_configuration": registered,
                                "condition": name,
                                "seed": seed,
                            },
                            directory / f"checkpoint_update{update:04d}.pt",
                        )
                        print(
                            f"EVAL seed={seed} condition={name} update={update} dev_ADE={row['evaluations'][-1]['scene_macro_ade_meters']:.4f}",
                            flush=True,
                        )
                    if update == specification["optimizer_updates"]:
                        break
                    model.train()
                    online, targets = exploration.collate_cached_windows(
                        [subsets["train"][index] for index in sequence[update]], "cpu"
                    )
                    output = model(**online)
                    loss, components, selected = exploration.compute_losses(
                        output, targets, normalization, condition
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            "nonfinite loss; no automatic hyperparameter repair"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        specification["optimizer"]["gradient_clip_norm"],
                    )
                    if not torch.isfinite(gradient_norm):
                        raise RuntimeError("nonfinite gradient")
                    optimizer.step()
                    row["completed_updates"] = update + 1
                    report["new_optimizer_updates"] += 1
                    row["training_curve"].append(
                        {
                            "update": update + 1,
                            "total_loss": float(loss.detach()),
                            **{
                                name: float(value.detach())
                                for name, value in components.items()
                            },
                            "common_valid_slot_times": int(
                                selected["common_valid"].sum()
                            ),
                            "gradient_norm_before_clip": float(gradient_norm),
                        }
                    )
                row["last_fixed_checkpoint_update"] = max(
                    record["update"] for record in row["evaluations"]
                )
            row["wall_seconds"] = time.perf_counter() - run_started
            (directory / "condition_result.json").write_text(
                json.dumps(row, indent=2) + "\n"
            )
            report["runs"].append(row)
            report["wall_seconds"] = time.perf_counter() - started
            (output_directory / "comparison_summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if row.get("stopped_by_time_cap"):
                print("REGISTERED_TIME_CAP_STOP", flush=True)
                return
    report["complete"] = (
        report["new_optimizer_updates"] == specification["maximum_new_updates"]
    )
    report["wall_seconds"] = time.perf_counter() - started
    (output_directory / "comparison_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "new_optimizer_updates": report["new_optimizer_updates"],
                "wall_seconds": report["wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
