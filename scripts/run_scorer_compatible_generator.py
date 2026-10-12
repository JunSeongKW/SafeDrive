#!/usr/bin/env python
"""CPU teacher labels, native generator-head learning, scoring and diagnostics."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
from datetime import datetime, timezone
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import METRIC_NAMES, write_json, file_sha256, official_scene_scores
from calibrate_official_toad_navsim_v1_scorer import protected_files_audit

CONDITION_LABELS = {"A_original": "Original", "B_quality": "Quality matching", "C_compatible": "Quality + scoring reliability"}


def inputs(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]
    registration = json.loads((output / "registration.json").read_text())
    assert registration["configuration"] == configuration
    cache = {path.stem: np.load(path, mmap_mode="r") for path in (output / "cache").glob("*.npy")}
    return output, registration, cache


def reference_configuration(configuration):
    comparison = json.loads((PROJECT_ROOT / configuration["comparison_configuration"]).read_text())
    return json.loads((PROJECT_ROOT / comparison["reference_configuration"]).read_text())


def official_job(job):
    record, trajectories, rankings = job
    # Existing validated helper expects two nonempty pools. Only bookkeeping is
    # split here; pairwise official PDMS is independent of this split.
    midpoint = len(trajectories) // 2
    return official_scene_scores((record, trajectories, rankings[:midpoint], rankings[midpoint:], rankings[midpoint:midpoint+16]))


def evaluate_official_jobs(configuration, jobs, destination, progress_path):
    started = time.perf_counter(); metrics, count, error = [], 0, 0.
    with ProcessPoolExecutor(max_workers=configuration["official_evaluation_workers"], mp_context=multiprocessing.get_context("spawn")) as workers:
        for completed, result in enumerate(workers.map(official_job, jobs, chunksize=1), start=1):
            metrics.append(result["official_metrics"]); count += result["checked_candidate_count"]
            error = max(error, result["pair_evaluator_maximum_error"])
            if completed % 32 == 0:
                write_json(progress_path, {"scenes": completed, "total": len(jobs), "seconds": time.perf_counter() - started})
                print(json.dumps({"official_scored_scenes": completed, "total": len(jobs)}), flush=True)
    values = np.asarray(metrics, dtype=np.float64); assert np.isfinite(values).all()
    np.save(destination, values)
    return {"official_candidates": int(values.shape[0] * values.shape[1]), "unchanged_pair_checks": count,
        "maximum_pair_evaluator_difference": error, "cpu_wall_seconds": time.perf_counter() - started, "gpu_seconds": 0}


def label_native(configuration):
    output, registration, cache = inputs(configuration)
    records = [record for record in registration["records"] if record["pilot_split"] != "locked_development"]
    jobs = [(record, np.array(cache["native_trajectories"][record["pilot_row"]]), np.array(cache["native_log_scores"][record["pilot_row"]])) for record in records]
    measured = evaluate_official_jobs(configuration, jobs, output / "adaptation_native_metrics.npy", output / "native_label_progress.json")
    development_metrics = np.load(PROJECT_ROOT / configuration["development_source"] / "official_metrics.npy")[:, :64]
    np.save(output / "native_metrics_all.npy", np.concatenate((np.load(output / "adaptation_native_metrics.npy"), development_metrics)))
    write_json(output / "native_labels_complete.json", {"complete": True, **measured})


def native_head(configuration):
    import torch
    repository = PROJECT_ROOT / "reference_repositories/TOADNav1OfficialCalibration"
    sys.path.insert(0, str(repository))
    from navsim.agents.drivoR.layers.utils.mlp import MLP
    head = MLP(256, 1024, 24)
    saved = torch.load(PROJECT_ROOT / configuration["output_directory"] / "native_head_initialization.pt", map_location="cpu", weights_only=True)
    head.load_state_dict(saved, strict=True); return head


def construct_teachers(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_compatible_generator import construct_teacher_distributions, distribution_matching_losses
    output, registration, cache = inputs(configuration)
    torch.set_num_threads(configuration["cpu_threads"])
    labels = np.load(output / "native_metrics_all.npy")
    # Only adaptation rows are ever passed to the teacher construction.
    adaptation_count = configuration["training_scenes"] + configuration["tuning_scenes"]
    actual = torch.from_numpy(labels[:adaptation_count, :, -1].astype(np.float32))
    predicted = torch.from_numpy(np.exp(np.array(cache["native_log_scores"][:adaptation_count])) / 12)
    quality, compatible, mixtures = construct_teacher_distributions(actual, predicted, configuration["quality_temperature"],
        configuration["reliability_temperature"], configuration["maximum_reliability_mixture"], configuration["maximum_target_quality_loss"])
    np.savez(output / "teacher_distributions.npz", quality=quality.numpy(), compatible=compatible.numpy(), mixtures=mixtures.numpy())
    training_rows = np.asarray([record["pilot_row"] for record in registration["records"] if record["pilot_split"] == "train"])
    distribution_stats = {}
    for name, weights in [("B_quality", quality), ("C_compatible", compatible)]:
        weights = weights[training_rows]; errors = (predicted[training_rows] - actual[training_rows]).abs()
        distribution_stats[name] = {"target_mean_actual_pdms": float((weights * actual[training_rows]).sum(dim=1).mean() * 100),
            "target_mean_absolute_prediction_error_pdms_points": float((weights * errors).sum(dim=1).mean() * 100),
            "median_effective_candidate_count": float(weights.square().sum(dim=1).reciprocal().median()),
            "mean_effective_candidate_count": float(weights.square().sum(dim=1).reciprocal().mean())}
    actual_drop = (quality * actual).sum(dim=1) - (compatible * actual).sum(dim=1)
    assert actual_drop.max() <= configuration["maximum_target_quality_loss"] + 1e-6
    head = native_head(configuration).eval()
    tokens = torch.from_numpy(np.array(cache["trajectory_tokens"][:2])); native = torch.from_numpy(np.array(cache["native_trajectories"][:2]))
    prediction = head(tokens).reshape(2, 64, 8, 3)
    identity_error = float((prediction - native).abs().max()); assert identity_error < 1e-3
    losses = distribution_matching_losses(prediction, native, compatible[:2], quality[:2], configuration["kernel_bandwidths_m"], configuration["heading_distance_scale_m"], configuration["coverage_temperature_m2"])
    total = losses["distribution_mmd"] + configuration["anchor_loss_weight"] * losses["native_anchor"] + configuration["coverage_loss_weight"] * losses["quality_coverage"]
    total.backward()
    gradients = {name: float(parameter.grad.norm()) for name, parameter in head.named_parameters()}
    assert all(np.isfinite(value) and value > 0 for value in gradients.values())
    # A nondegenerate synthetic target must exert a trajectory-coordinate gradient.
    generator = torch.Generator().manual_seed(0)
    synthetic_native = torch.randn(2, 8, 8, 3, generator=generator)
    synthetic_generated = synthetic_native.clone().requires_grad_(True)
    synthetic_weights = torch.zeros(2, 8); synthetic_weights[:, 0] = 1
    synthetic_loss = distribution_matching_losses(synthetic_generated, synthetic_native, synthetic_weights, synthetic_weights)["distribution_mmd"]
    synthetic_loss.backward(); assert synthetic_generated.grad.norm() > 0 and torch.isfinite(synthetic_generated.grad).all()
    # Identical uniform empirical sets yield zero distribution distance.
    uniform = torch.full((2, 8), 1 / 8)
    uniform_distance = float(distribution_matching_losses(synthetic_native, synthetic_native, uniform, uniform)["distribution_mmd"])
    assert abs(uniform_distance) < 1e-6
    write_json(output / "teacher_and_gradient_complete.json", {"complete": True, "training_target_statistics": distribution_stats,
        "target_quality_guard_maximum_drop_pdms_points": float(actual_drop.max() * 100),
        "native_head_identity_error_m_or_rad": identity_error, "uniform_empirical_mmd_identity_error": uniform_distance,
        "synthetic_coordinate_gradient_norm": float(synthetic_generated.grad.norm()), "actual_head_gradient_norms": gradients,
        "native_head_trainable_parameters": sum(parameter.numel() for parameter in head.parameters()),
        "navtest_rows_in_teacher_construction": 0, "encoder_calls": 0, "gpu_seconds": 0,
        "frozen_original_checkpoint_sha256_unchanged": file_sha256(PROJECT_ROOT / registration["checkpoint"]) == registration["checkpoint_sha256"]})
    render_teachers(configuration, registration, cache, labels, quality.numpy(), compatible.numpy(), distribution_stats)
    print(json.dumps({"stage1": distribution_stats, "gradient_validation": "passed", "parameters": sum(parameter.numel() for parameter in head.parameters())}), flush=True)


def render_teachers(configuration, registration, cache, labels, quality, compatible, statistics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    destination = PROJECT_ROOT / configuration["results_directory"]
    train_rows = [record["pilot_row"] for record in registration["records"] if record["pilot_split"] == "train"]
    # Select by observed command/order, never best improvement.
    examples = [next(record for record in registration["records"] if record["pilot_split"] == "train" and record["command_index"] == command) for command in range(3)]
    figure, axes = plt.subplots(3, 3, figsize=(16, 13))
    for row_index, record in enumerate(examples):
        scene_row = record["pilot_row"]; actual = labels[scene_row, :, -1] * 100
        predicted = np.exp(cache["native_log_scores"][scene_row]) / 12 * 100
        axis = axes[row_index, 0]; axis.plot(actual, label="Actual PDMS", color="#1a9850"); axis.plot(predicted, label="Scorer estimate", color="#e95d22")
        axis.set_ylim(0, 105); axis.set_ylabel("PDMS points"); axis.set_title(record["command_name"] + " | " + record["token"]); axis.legend(fontsize=8)
        axis = axes[row_index, 1]; axis.plot(quality[scene_row], label="B: quality", color="#4c78a8"); axis.plot(compatible[scene_row], label="C: quality + reliability", color="#d04467")
        axis.set_xlabel("Native candidate index"); axis.set_ylabel("Teacher probability"); axis.legend(fontsize=8)
        paths = cache["native_trajectories"][scene_row]
        for candidate_index, path in enumerate(paths):
            axes[row_index, 2].plot(path[:, 1], path[:, 0], color="#bbbbbb", linewidth=.5)
        axes[row_index, 2].scatter(paths[:, -1, 1], paths[:, -1, 0], s=50 + 2000 * compatible[scene_row], c=actual, cmap="viridis", vmin=0, vmax=100, alpha=.8)
        axes[row_index, 2].invert_xaxis(); axes[row_index, 2].set_xlabel("Ego lateral (m), left positive"); axes[row_index, 2].set_ylabel("Ego forward (m)")
        axes[row_index, 2].set_title("C teacher mass (dot area) / actual PDMS (color)")
    figure.suptitle("Stage 1: high-quality and accurately scored TRAINING-only target distributions\nOfficial labels define an offline teacher; this is not an inference policy or a measured planning improvement", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .94)); figure.savefig(destination / "stage1_teacher_distributions.png", dpi=140); plt.close(figure)


def train_heads(configuration):
    import torch
    from planning_aware_future_prediction.models.scorer_compatible_generator import distribution_matching_losses
    output, registration, cache = inputs(configuration); assert (output / "teacher_and_gradient_complete.json").exists()
    torch.set_num_threads(configuration["cpu_threads"]); torch.use_deterministic_algorithms(True)
    training_rows = [record["pilot_row"] for record in registration["records"] if record["pilot_split"] == "train"]
    teachers = np.load(output / "teacher_distributions.npz")
    training_tokens = torch.from_numpy(np.array(cache["trajectory_tokens"][training_rows])); native = torch.from_numpy(np.array(cache["native_trajectories"][training_rows]))
    quality = torch.from_numpy(teachers["quality"][training_rows]); compatible = torch.from_numpy(teachers["compatible"][training_rows])
    random = np.random.default_rng(configuration["seed"])
    batch_order = []; order = []
    while len(batch_order) < configuration["training_steps"]:
        order = random.permutation(len(training_rows))
        batch_order.extend([order[start:start + configuration["batch_size"]].tolist() for start in range(0, len(order), configuration["batch_size"])])
    batch_order = batch_order[:configuration["training_steps"]]; write_json(output / "batch_order.json", {"training_rows": training_rows, "batches": batch_order})
    measured = {}
    for condition, target in [("B_quality", quality), ("C_compatible", compatible)]:
        torch.manual_seed(configuration["seed"]); head = native_head(configuration).train()
        optimizer = torch.optim.AdamW(head.parameters(), lr=configuration["learning_rate"], weight_decay=configuration["weight_decay"])
        condition_directory = output / condition; condition_directory.mkdir(exist_ok=True)
        training_log = []; started = time.perf_counter()
        for step, rows in enumerate(batch_order, start=1):
            prediction = head(training_tokens[rows]).reshape(len(rows), 64, 8, 3)
            losses = distribution_matching_losses(prediction, native[rows], target[rows], quality[rows], configuration["kernel_bandwidths_m"], configuration["heading_distance_scale_m"], configuration["coverage_temperature_m2"])
            loss = losses["distribution_mmd"] + configuration["anchor_loss_weight"] * losses["native_anchor"] + configuration["coverage_loss_weight"] * losses["quality_coverage"]
            optimizer.zero_grad(set_to_none=True); loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), configuration["gradient_clip_norm"])
            assert torch.isfinite(loss) and torch.isfinite(gradient_norm)
            assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())
            optimizer.step()
            training_log.append({"step": step, "loss": float(loss.detach()), **{name: float(value.detach()) for name, value in losses.items()}, "gradient_norm": float(gradient_norm), "wall_seconds": time.perf_counter() - started})
            if step in configuration["checkpoint_steps"]:
                torch.save({"state_dict": head.state_dict(), "condition": condition, "completed_steps": step, "optimizer": optimizer.state_dict()}, condition_directory / f"step{step}.pt")
            if step % 100 == 0:
                print(json.dumps({"condition": condition, **training_log[-1]}), flush=True)
        write_json(condition_directory / "training_log.json", training_log)
        measured[condition] = {"cpu_wall_seconds": time.perf_counter() - started, "steps": len(training_log), "parameters": sum(parameter.numel() for parameter in head.parameters()), "finite_gradients_every_step": True}
    write_json(output / "training_complete.json", {"complete": True, "conditions": measured, "encoder_calls": 0, "gpu_seconds": 0,
        "same_batch_order": True, "same_initial_weights": True, "original_checkpoint_sha256_unchanged": file_sha256(PROJECT_ROOT / registration["checkpoint"]) == registration["checkpoint_sha256"]})


def predict_and_rescore(configuration):
    import torch
    from compare_public_drivor_toad_scorers import model_configuration_for, load_official_model, score_metrics
    from diagnose_residual_control_sampler import module_digest
    output, registration, cache = inputs(configuration)
    comparison = json.loads((PROJECT_ROOT / configuration["comparison_configuration"]).read_text())
    model, loading = load_official_model(model_configuration_for(comparison, "drivor_public"), "cpu")
    torch.set_num_threads(configuration["cpu_threads"]); initial_digest = module_digest(model)
    backbone_calls = 0
    def reject_encoder(_module, _arguments):
        nonlocal backbone_calls
        backbone_calls += 1; raise AssertionError("Cached generator learning/evaluation must not run encoder")
    guard = model.image_backbone.register_forward_pre_hook(reject_encoder)
    evaluation_records = [record for record in registration["records"] if record["pilot_split"] != "train"]
    specifications = [("A_original", 0), ("B_quality", 100), ("C_compatible", 100), ("B_quality", 300), ("C_compatible", 300)]
    paths, scores, probabilities = [], [], []; timings = []
    started = time.perf_counter()
    for condition, step in specifications:
        head = native_head(configuration).eval()
        if condition != "A_original":
            head.load_state_dict(torch.load(output / condition / f"step{step}.pt", map_location="cpu", weights_only=True)["state_dict"], strict=True)
        condition_paths, condition_scores, condition_probabilities = [], [], []
        with torch.inference_mode():
            for record in evaluation_records:
                row = record["pilot_row"]
                tokens = torch.from_numpy(np.array(cache["trajectory_tokens"][row:row+1]))
                visual = torch.from_numpy(np.array(cache["scene_features"][row:row+1])); ego = torch.from_numpy(np.array(cache["ego_tokens"][row:row+1]))
                tick = time.perf_counter(); generated = head(tokens).reshape(1, 64, 8, 3)
                decoder_seconds = time.perf_counter() - tick
                if condition == "A_original":
                    assert float((generated - torch.from_numpy(np.array(cache["native_trajectories"][row:row+1]))).abs().max()) < 1e-3
                    generated = torch.from_numpy(np.array(cache["native_trajectories"][row:row+1]))
                ranking, probability, _ = score_metrics(model, generated, visual, ego)
                if condition == "A_original":
                    assert float((ranking - torch.from_numpy(np.array(cache["native_log_scores"][row:row+1]))).abs().max()) < 1e-4
                condition_paths.append(generated[0].numpy()); condition_scores.append(ranking[0].numpy()); condition_probabilities.append(probability[0].numpy())
                timings.append({"condition": condition, "head_cpu_seconds": decoder_seconds})
        paths.append(condition_paths); scores.append(condition_scores); probabilities.append(condition_probabilities)
        print(json.dumps({"rescored": condition, "step": step, "scenes": len(evaluation_records)}), flush=True)
    guard.remove(); assert backbone_calls == 0 and initial_digest == module_digest(model)
    assert all(parameter.grad is None for parameter in model.parameters())
    np.savez_compressed(output / "evaluation_candidates.npz", tokens=[record["token"] for record in evaluation_records],
        trajectories=np.asarray(paths), log_scores=np.asarray(scores), metric_probabilities=np.asarray(probabilities))
    write_json(output / "prediction_complete.json", {"complete": True, "specifications": specifications, "records": evaluation_records,
        "cpu_wall_seconds": time.perf_counter() - started, "encoder_calls": backbone_calls, "all_frozen_scorer_weights_unchanged": True,
        "scorer_weight_gradients_absent": True, "timings": timings, "gpu_seconds": 0, "loading": loading,
        "protected_training": protected_files_audit(reference_configuration(configuration))})


def label_predictions(configuration):
    output, registration, cache = inputs(configuration)
    prediction_metadata = json.loads((output / "prediction_complete.json").read_text())
    records = prediction_metadata["records"]; arrays = np.load(output / "evaluation_candidates.npz")
    jobs = []
    for row, record in enumerate(records):
        # All five generator conditions; official helper restores reference+
        # individual candidate normalization, not a joint-progress oracle.
        jobs.append((record, arrays["trajectories"][:, row].reshape(-1, 8, 3), arrays["log_scores"][:, row].reshape(-1)))
    measured = evaluate_official_jobs(configuration, jobs, output / "evaluation_metrics_flat.npy", output / "evaluation_label_progress.json")
    values = np.load(output / "evaluation_metrics_flat.npy").reshape(len(records), 5, 64, 7).transpose(1, 0, 2, 3)
    original_labels = np.load(output / "native_metrics_all.npy")[[record["pilot_row"] for record in records]]
    difference = float(np.abs(original_labels - values[0]).max()); assert difference < 1e-8, difference
    np.save(output / "evaluation_metrics.npy", values)
    write_json(output / "evaluation_labels_complete.json", {"complete": True, **measured, "original_same_paths_official_metric_difference": difference})


def baseline_precision(configuration):
    output = PROJECT_ROOT / configuration["output_directory"]; destination = PROJECT_ROOT / configuration["results_directory"]
    request = json.loads((output / "baseline_precision_request.json").read_text()); arrays = np.load(output / "baseline_precision_paths.npz")
    measured = evaluate_official_jobs(configuration, [(request["record"], arrays["trajectories"], arrays["log_scores"])], output / "baseline_precision_metrics.npy", output / "baseline_precision_progress.json")
    metrics = np.load(output / "baseline_precision_metrics.npy")[0]; chosen = request["original_selected_index"]
    maximum = float(np.abs(metrics[:64] - metrics[64:]).max())
    binary_identical = bool(np.array_equal(metrics[:64, [0, 1, 3, 4]], metrics[64:, [0, 1, 3, 4]]))
    assert binary_identical and maximum < 1e-3
    report = {"complete": True, **measured, "token": request["record"]["token"],
        "maximum_unchanged_head_coordinate_replay_error_m_or_rad": request["maximum_unchanged_head_gpu_cache_vs_cpu_error_m_or_rad"],
        "binary_nc_dac_ttc_comfort_identical_for_all64": binary_identical, "maximum_all_metric_difference": maximum,
        "original_selected_path_gpu_cache_pdms": float(metrics[chosen, -1] * 100),
        "original_selected_path_unchanged_head_cpu_pdms": float(metrics[64 + chosen, -1] * 100)}
    write_json(destination / "baseline_precision_validation.json", report)
    with (destination / "report.md").open("a") as stream:
        stream.write(f"\n### 원head CPU 재생 정밀도 확인\n\n도로경계악화사례에서학습하지않은원head를CPU로재생한64개경로와원GPU캐시64개를공식평가했다. NC/DAC/TTC/comfort모두동일,전체지표최대차{maximum:.3g},선택query37은양쪽모두100점이다. 따라서이번100→0실패를원head의CPU/GPU정밀도차로설명할근거는없다. 추가학습·GPU0,검사범위는이한장면이다.\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--phase", choices=["label_native", "teachers", "train", "predict", "label_predictions", "baseline_precision"], required=True)
    arguments = parser.parse_args(); configuration = json.loads(arguments.configuration.read_text())
    phases = {"label_native": label_native, "teachers": construct_teachers, "train": train_heads, "predict": predict_and_rescore, "label_predictions": label_predictions, "baseline_precision": baseline_precision}
    phases[arguments.phase](configuration)
