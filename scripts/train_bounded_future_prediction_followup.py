"""Preregistered small follow-up: A seed variability, longer curves, paired E/F.

Reuses the committed cache and old runner metrics without altering old artifacts.
Old checkpoints restore model/AdamW, but lack RNG snapshots: not exact resume.
"""

import argparse
import copy
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import train_target_supervision_ablation as exploration
from diagnose_visual_future_prediction_variance import (
    collect_predictions,
    load_existing_experiment,
    perturbation_amplitudes,
    visual_prediction_diagnostics,
)

PROJECT_ROOT = exploration.PROJECT_ROOT


class PlanningGradientControlledPilot(exploration.FixedDistanceFutureSupervisionPilot):
    """Same weights/forward values, optional planner-only detached input boundary."""

    def forward(self, *args, **kwargs):
        kwargs.setdefault("detach_future_for_planning", self.detach_future_for_planning)
        return super().forward(*args, **kwargs)


def generate_recording_balanced_batch_sequence(
    training_windows, seed, num_updates, batch_size
):
    recording_indices = defaultdict(list)
    for index, window in enumerate(training_windows):
        recording_indices[window["metadata"]["recording_group"]].append(index)
    generator = random.Random(seed)
    recording_names = sorted(recording_indices)
    return [
        [
            generator.choice(recording_indices[generator.choice(recording_names)])
            for _ in range(batch_size)
        ]
        for _ in range(num_updates)
    ]


def capture_rng_states():
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all_visible": torch.cuda.get_rng_state_all(),
        "python": random.getstate(),
        "numpy": {
            "generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        type=Path,
        default=PROJECT_ROOT
        / "configs/exploration/future_prediction_diagnostic_followup_v1.json",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.physical_gpu)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("approved physical GPU unavailable")
    device = torch.device("cuda:0")
    configuration = json.loads(arguments.configuration.read_text())
    if (
        configuration["max_new_optimizer_updates"] != 7400
        or configuration["common_total_updates"] != 1000
        or configuration["selected_entity_budget"] != 4
    ):
        raise ValueError("runner restricted to registered bounded study")
    output_directory = arguments.output_directory.resolve()
    if (
        not output_directory.is_relative_to(PROJECT_ROOT / "outputs")
        or output_directory.exists()
    ):
        raise ValueError("require NEW project outputs directory")
    output_directory.mkdir(parents=True)
    old_configuration, windows, normalization, original_directory = (
        load_existing_experiment(device)
    )
    training_windows = [
        window for window in windows if window["metadata"]["split"] == "train"
    ]
    development_windows = [
        window for window in windows if window["metadata"]["split"] == "development"
    ]
    original_batch_record = json.loads(
        (original_directory / "shared_batch_sequence.json").read_text()
    )
    if original_batch_record["training_window_order"] != [
        window["metadata"]["current_frame_token"] for window in training_windows
    ]:
        raise RuntimeError("original sampler window order differs")
    conditions = {row["label"]: row for row in old_configuration["conditions"]}
    conditions["F"] = configuration["F"]
    run_specs = [
        {"seed": seed, "label": "A", "end_update": 200, "start_update": 0}
        for seed in configuration["baseline_variability_seeds"]
    ]
    run_specs += [
        {"seed": 29, "label": label, "end_update": 1000, "start_update": 200}
        for label in configuration["continued_seed29_conditions"]
    ]
    run_specs += [
        {**specification, "end_update": 1000, "start_update": 0}
        for specification in configuration["fresh_long_runs"]
    ]
    expected_new_updates = sum(
        row["end_update"] - row["start_update"] for row in run_specs
    )
    if expected_new_updates != configuration["max_new_optimizer_updates"]:
        raise RuntimeError("registered update count changed")
    report = {
        "baseline_commit": "9353acf",
        "execution_parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "configuration": configuration,
        "configuration_sha256": exploration.file_sha256(arguments.configuration),
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): exploration.file_sha256(path)
            for path in (
                Path(__file__),
                PROJECT_ROOT / "scripts/train_target_supervision_ablation.py",
                PROJECT_ROOT / "scripts/diagnose_visual_future_prediction_variance.py",
                PROJECT_ROOT
                / "src/planning_aware_future_prediction/models/fixed_distance_future_supervision.py",
            )
        },
        "cache_index_sha256": exploration.file_sha256(
            PROJECT_ROOT / configuration["cache_directory"] / "cache_index.json"
        ),
        "gpu_name": torch.cuda.get_device_name(0),
        "physical_gpu": arguments.physical_gpu,
        "torch_version": torch.__version__,
        "runs": [],
        "new_optimizer_updates": 0,
        "interpretation": "Bounded diagnostics only. Paired E/F seeds29/11; longer A-E seed29 exploratory. No official planning score or target/novelty winner.",
    }
    write_json(output_directory / "registered_execution.json", report)
    invocation_start = time.perf_counter()
    initial_states, batch_sequences = {}, {}
    for run in run_specs:
        seed, label = run["seed"], run["label"]
        run_start = time.perf_counter()
        condition = conditions[label]
        directory = output_directory / f"seed{seed}_{label}"
        directory.mkdir()
        if seed not in initial_states:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            initial_model = PlanningGradientControlledPilot()
            initial_states[seed] = copy.deepcopy(initial_model.state_dict())
            batch_sequences[seed] = generate_recording_balanced_batch_sequence(
                training_windows, seed, 1000, 8
            )
            torch.save(
                initial_states[seed], output_directory / f"shared_initial_seed{seed}.pt"
            )
            write_json(
                output_directory / f"batch_sequence_seed{seed}.json",
                batch_sequences[seed],
            )
        shared_initial_state = initial_states[seed]
        initial_hash = exploration.state_sha256(shared_initial_state)
        batch_sequence = batch_sequences[seed]
        if seed == 29:
            if batch_sequence[:200] != original_batch_record["batch_indices"]:
                raise RuntimeError("old sampler prefix differs")
            if initial_hash != exploration.state_sha256(
                torch.load(
                    original_directory / "shared_initial_state.pt", weights_only=True
                )
            ):
                raise RuntimeError("old initial weights differ")
        model = PlanningGradientControlledPilot().to(device)
        model.detach_future_for_planning = condition.get(
            "detach_future_for_planning", False
        )
        model.load_state_dict(shared_initial_state, strict=True)
        model.configure_active_modules(condition["enable_future_branch"])
        active_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(active_parameters, lr=0.0001, weight_decay=0.0001)
        result = {
            "seed": seed,
            "condition": condition,
            "start_update": run["start_update"],
            "end_update": run["end_update"],
            "initial_state_sha256": initial_hash,
            "active_parameters": sum(
                parameter.numel() for parameter in active_parameters
            ),
            "evaluations": [],
            "new_training_updates": [],
            "continuation": None,
        }
        if run["start_update"]:
            source_path = original_directory / label / "last_update_200.pt"
            checkpoint = torch.load(source_path, weights_only=True)
            if (
                checkpoint["optimizer_updates"] != 200
                or checkpoint["shared_initial_state_sha256"] != initial_hash
                or checkpoint["cache_index_sha256"] != report["cache_index_sha256"]
            ):
                raise RuntimeError("continuation provenance mismatch")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_steps = sorted(
                {int(state["step"]) for state in optimizer.state.values()}
            )
            if optimizer_steps != [200]:
                raise RuntimeError("AdamW state was not at step200")
            if exploration.state_sha256(model.state_dict()) != exploration.state_sha256(
                checkpoint["model"]
            ):
                raise RuntimeError("restored model differs")
            old_result = json.loads(
                (original_directory / label / "condition_result.json").read_text()
            )
            result["evaluations"] = old_result["evaluations"]
            result["continuation"] = {
                "source_checkpoint_sha256": exploration.file_sha256(source_path),
                "model_and_optimizer_restored": True,
                "verified_sampler_prefix_updates": 200,
                "scheduler": "absent in original and follow-up",
                "old_rng_snapshot_available": False,
                "bitwise_exact_resume_claim": False,
                "rng_relevance": "Current dropout=0 and sampler precomputed; still no historical RNG snapshot.",
            }
        else:
            model.eval()
            for split_name, split_windows in (
                ("train", training_windows),
                ("development", development_windows),
            ):
                summary, rows = exploration.evaluate_model(
                    model, split_windows, normalization, condition, device, 8
                )
                result["evaluations"].append(
                    {"update": 0, "split": split_name, **summary}
                )
                write_json(directory / f"{split_name}_update000.json", rows)
        torch.cuda.reset_peak_memory_stats()
        for update in range(run["start_update"] + 1, run["end_update"] + 1):
            model.train()
            online, targets = exploration.collate_cached_windows(
                [training_windows[index] for index in batch_sequence[update - 1]],
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                **online, enable_future_branch=condition["enable_future_branch"]
            )
            total_loss, components, selected = exploration.compute_losses(
                prediction, targets, normalization, condition
            )
            if not torch.isfinite(total_loss):
                raise RuntimeError("nonfinite loss")
            if update == run["start_update"] + 1:
                result["first_new_update_gradient_contract"] = {
                    name: {
                        "weighted_predictor_gradient_norm": exploration.tensor_gradient_norm(
                            loss
                            * (
                                1 if name == "planning" else condition[f"{name}_weight"]
                            ),
                            model.future_predictor.parameters(),
                        )
                        if condition["enable_future_branch"]
                        else 0,
                        "weighted_planner_gradient_norm": exploration.tensor_gradient_norm(
                            loss
                            * (
                                1 if name == "planning" else condition[f"{name}_weight"]
                            ),
                            model.ego_planner.parameters(),
                        )
                        if condition["enable_future_branch"]
                        else None,
                    }
                    for name, loss in components.items()
                }
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(active_parameters, 1.0)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError("nonfinite gradient")
            optimizer.step()
            result["new_training_updates"].append(
                {
                    "update": update,
                    "total_loss": float(total_loss.detach()),
                    **{
                        f"{name}_loss": float(loss.detach())
                        for name, loss in components.items()
                    },
                    "common_valid_slot_times": int(selected["common_valid"].sum()),
                    "global_gradient_norm_before_clip": float(gradient_norm),
                }
            )
            report["new_optimizer_updates"] += 1
            if (
                time.perf_counter() - run_start
                > configuration["time_limits_seconds"]["per_run_including_evaluation"]
                or time.perf_counter() - invocation_start
                > configuration["time_limits_seconds"]["whole_invocation"]
                or report["new_optimizer_updates"] > 7400
            ):
                raise RuntimeError("registered execution cap exceeded")
            if update % 100 == 0:
                print(
                    f"TRAIN seed={seed} condition={label} update={update} loss={float(total_loss.detach()):.6f}",
                    flush=True,
                )
            if (
                update in configuration["evaluation_updates"]
                or update == run["end_update"]
            ):
                model.eval()
                for split_name, split_windows in (
                    ("train", training_windows),
                    ("development", development_windows),
                ):
                    summary, rows = exploration.evaluate_model(
                        model, split_windows, normalization, condition, device, 8
                    )
                    result["evaluations"].append(
                        {"update": update, "split": split_name, **summary}
                    )
                    write_json(
                        directory / f"{split_name}_update{update:04d}.json", rows
                    )
                checkpoint_path = directory / f"checkpoint_update{update:04d}.pt"
                if checkpoint_path.exists():
                    raise RuntimeError("checkpoint collision")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "optimizer_updates": update,
                        "normalization": normalization,
                        "configuration": configuration,
                        "condition": condition,
                        "seed": seed,
                        "rng_states": capture_rng_states(),
                        "sampler_next_update": update + 1,
                        "sampler_sequence": batch_sequence,
                        "initial_state_sha256": initial_hash,
                        "cache_index_sha256": report["cache_index_sha256"],
                    },
                    checkpoint_path,
                )
                write_json(directory / "condition_result.json", result)
        model.eval()
        if condition["enable_future_branch"] and label in ("C", "E", "F"):
            result["visual_prediction_diagnostics"] = {
                split_name: visual_prediction_diagnostics(
                    collect_predictions(model, split_windows, device), normalization
                )
                for split_name, split_windows in (
                    ("train", training_windows),
                    ("development", development_windows),
                )
            }
            result["perturbation_amplitudes"] = perturbation_amplitudes(
                model, development_windows, device
            )
        result["wall_seconds_including_evaluation"] = time.perf_counter() - run_start
        result["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["parameter_update_l2_from_shared_initial"] = {
            module_name: float(
                sum(
                    (tensor.detach().cpu() - shared_initial_state[name])
                    .double()
                    .square()
                    .sum()
                    for name, tensor in model.state_dict().items()
                    if name.startswith(module_name + ".")
                ).sqrt()
            )
            for module_name in ("future_predictor", "ego_planner", "entity_scorer")
        }
        write_json(directory / "condition_result.json", result)
        report["runs"].append(result)
        write_json(output_directory / "comparison_report.json", report)
        print(f"RUN_DONE seed={seed} condition={label}", flush=True)
        del model, optimizer, active_parameters, prediction, total_loss, components
        torch.cuda.empty_cache()
    if report["new_optimizer_updates"] != 7400:
        raise RuntimeError("incomplete registered execution")
    report["whole_wall_seconds"] = time.perf_counter() - invocation_start
    report["status"] = "bounded_followup_complete"
    write_json(output_directory / "comparison_report.json", report)
    print("BOUNDED_FOLLOWUP_DONE", flush=True)


if __name__ == "__main__":
    main()
