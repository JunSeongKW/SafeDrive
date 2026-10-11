#!/usr/bin/env python
"""Read-only CPU checks of metric-query interactions and scoring shortcuts.

Uses the original 12 fixed development scenes and trained checkpoints. This
does not train a new model, label sky, or estimate causal object importance.
"""
import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from run_metric_visual_evidence_pilot import (
    cache_tensors, initialize_reader, load_artifacts, native_labels,
    verify_registration,
)
from calibrate_frozen_drivor_scorer import file_sha256, protected_training_audit, write_json
from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses
from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score


def verify_nonlinear_joint_query_releases_constraint():
    """Synthetic capacity check only; not a trained replacement model."""
    generator = torch.Generator().manual_seed(20261011)
    torch.manual_seed(20261011)
    candidate_conditions = torch.randn(1, 4, 6, generator=generator)
    metric_conditions = torch.randn(2, 5, generator=generator)
    keys = torch.randn(1, 13, 8, generator=generator)
    additive_candidate_query = torch.nn.Linear(6, 8)
    additive_metric_query = torch.nn.Linear(5, 8)
    joint_query_network = torch.nn.Sequential(torch.nn.Linear(11, 16), torch.nn.SiLU(), torch.nn.Linear(16, 8))
    additive_queries = additive_candidate_query(candidate_conditions)[:, :, None] + additive_metric_query(metric_conditions)[None, None]
    joined_conditions = torch.cat((candidate_conditions[:, :, None].expand(-1, -1, 2, -1),
                                   metric_conditions[None, None].expand(1, 4, -1, -1)), -1)
    joint_queries = joint_query_network(joined_conditions)
    variations = {}
    for name, queries in (("additive", additive_queries), ("nonlinear_joint", joint_queries)):
        logits = torch.einsum("bmkd,bnd->bmkn", queries, keys) / math.sqrt(8)
        contrast = logits[:, :, 0] - logits[:, :, 1]
        contrast = contrast - contrast[..., :1]
        variations[name] = (contrast - contrast[:, :1]).abs().max().item()
    assert variations["additive"] < 1e-5
    assert variations["nonlinear_joint"] > 1e-4
    return {"maximum_candidate_dependence_of_metric_relative_odds": variations,
            "optimizer_steps": 0, "interpretation": "Nonlinearly joining candidate and metric inputs removes this algebraic constraint. It establishes capacity, not semantic learning or planning benefit."}


def inspect_additive_metric_constraint(model, inputs, attention):
    """Verify relative NC/DAC token odds have no candidate interaction."""
    with torch.no_grad():
        safe_tokens = inputs["visual_tokens"].masked_fill(~inputs["valid_token_mask"][..., None], 0.)
        positions = inputs["positions"]
        position_inputs = positions[None] if positions.ndim == 2 else positions
        keys = model.key_projection(model.visual_normalization(safe_tokens)) + model.position_projection(position_inputs)
        metric_key_contrast = torch.einsum(
            "d,bnd->bn", model.metric_embedding[0] - model.metric_embedding[1], keys
        ) / math.sqrt(model.reader_dimension)
        # Subtract token0 to cancel each softmax normalization constant.
        observed_log_odds = attention[:, :, 0].log() - attention[:, :, 1].log()
        observed_log_odds = observed_log_odds - observed_log_odds[..., :1]
        expected_log_odds = metric_key_contrast - metric_key_contrast[..., :1]
        formula_error = (observed_log_odds - expected_log_odds[:, None]).abs().max().item()
        candidate_interaction_error = (observed_log_odds - observed_log_odds[:, :1]).abs().max().item()
        assert formula_error < 1e-5
        assert candidate_interaction_error < 1e-5
    return {
        "relative_metric_token_log_odds_formula_max_error": formula_error,
        "candidate_interaction_max_error": candidate_interaction_error,
        "constraint": "For fixed scene keys, NC/DAC relative token-pair odds depend only on metric embedding difference, not candidate/ego query.",
    }


def summarize_input_groups(adapter_input, gradient, reader_dimension):
    boundaries = [0, reader_dimension, reader_dimension + 256,
                  reader_dimension + 267, reader_dimension + 291,
                  reader_dimension + 297]
    names = ("visual_readout", "original_scored_feature", "direct_ego",
             "direct_trajectory", "original_component_logit")
    return {
        name: {
            "channels": end - start,
            "mean_absolute_input_gradient": gradient[..., start:end].abs().mean().item(),
            "mean_absolute_gradient_times_input_per_channel": (gradient[..., start:end] * adapter_input[..., start:end]).abs().mean().item(),
        }
        for name, start, end in zip(names, boundaries[:-1], boundaries[1:])
    }


def main(configuration_path, results_directory):
    configuration = json.loads(configuration_path.read_text())
    registration = verify_registration(configuration)
    torch.set_num_threads(2)
    torch.manual_seed(0)
    source, records, _, score_weights, output, pilot_result = load_artifacts(configuration)
    diagnostic_subset = json.loads((pilot_result / "diagnostic_subset.json").read_text())
    selected_rows = diagnostic_subset["rows"]
    labels = native_labels(configuration)
    baseline_input_labels = labels[:configuration["training_scenes"], :, :2]
    results_directory.mkdir(parents=True, exist_ok=True)
    protected_before = protected_training_audit(configuration)
    started = time.perf_counter()
    checkpoint_results = []
    for training_step in (100, 500):
        checkpoint_path = output / f"metric_specific_step{training_step}.pt"
        model = initialize_reader(configuration, "metric_specific").eval()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["model"], strict=True)
        model.requires_grad_(False)
        scene_results = []
        for row in selected_rows:
            inputs = cache_tensors(configuration, [row], torch.device("cpu"))
            inputs["visual_tokens"].requires_grad_(True)
            captured = {}

            def capture_adapter_inputs(module, arguments):
                captured["adapter_input"] = arguments[0]

            hook = model.residual_adapter.register_forward_pre_hook(capture_adapter_inputs)
            predicted, attention = model(**inputs, capture_attention=True)
            hook.remove()
            reference_scores = compose_drivor_log_score(predicted, score_weights).detach()
            constraints = inspect_additive_metric_constraint(model, inputs, attention.detach())
            pairs = torch.randint(0, 64, (1, 128, 2), generator=torch.Generator().manual_seed(8100 + row))
            official = torch.from_numpy(labels[row:row+1])
            component_loss, ranking_loss = compute_metric_evidence_losses(
                predicted, official, score_weights, pairs, configuration["ranking_temperature"]
            )
            total_loss = (configuration["component_loss_weight"] * component_loss
                          + configuration["ranking_loss_weight"] * ranking_loss)
            input_gradient, attention_gradient, token_gradient = torch.autograd.grad(
                total_loss, (captured["adapter_input"], attention, inputs["visual_tokens"])
            )
            assert all(torch.isfinite(value).all() for value in (predicted, attention, total_loss, input_gradient, attention_gradient, token_gradient))
            assert all(parameter.grad is None for parameter in model.parameters())
            input_groups = summarize_input_groups(captured["adapter_input"].detach(), input_gradient, model.reader_dimension)
            interventions = {}
            # Removing a head input is a functional/OOD intervention; native
            # score addition, visual query and all other blocks are unchanged.
            intervention_blocks = {
                "zero_visual_readout": (0, model.reader_dimension),
                "zero_original_scored_feature": (model.reader_dimension, model.reader_dimension + 256),
                "zero_original_logits_in_correction_only": (model.reader_dimension + 291, model.reader_dimension + 297),
                "zero_all_direct_inputs": (model.reader_dimension, model.reader_dimension + 297),
            }
            for name, (first_channel, last_channel) in intervention_blocks.items():
                def replace_adapter_inputs(module, arguments):
                    changed = arguments[0].clone()
                    changed[..., first_channel:last_channel] = 0.
                    return (changed,)

                hook = model.residual_adapter.register_forward_pre_hook(replace_adapter_inputs)
                with torch.no_grad():
                    modified, _ = model(**inputs)
                    modified_scores = compose_drivor_log_score(modified, score_weights)
                hook.remove()
                interventions[name] = {
                    "mean_absolute_NC_DAC_logit_change": (modified[..., :2] - predicted.detach()[..., :2]).abs().mean().item(),
                    "mean_absolute_log_score_change": (modified_scores - reference_scores).abs().mean().item(),
                    "selected_candidate_changed": bool((modified_scores.argmax(-1) != reference_scores.argmax(-1)).item()),
                    "native_other_four_components_unchanged": bool(torch.equal(modified[..., 2:], inputs["component_logits"][..., 2:])),
                }
            # Coordinate diagnostic only: upper/lower thirds are NOT sky/road labels.
            density = attention.detach()[0, :, :, 703:].reshape(64, 2, 19, 37) * 1406
            scene_results.append({
                "pilot_row": row, "sample_id": records[row]["token"],
                "query_constraint": constraints,
                "component_loss": component_loss.item(), "ranking_loss": ranking_loss.item(),
                "attention_gradient_norm": attention_gradient.norm().item(),
                "visual_token_gradient_norm": token_gradient.norm().item(),
                "adapter_input_gradient_groups": input_groups,
                "input_interventions": interventions,
                "current_frame_upper_six_rows_density_NC_DAC": density[:, :, :6].mean((0, 2, 3)).tolist(),
                "current_frame_lower_six_rows_density_NC_DAC": density[:, :, -6:].mean((0, 2, 3)).tolist(),
            })
        checkpoint_results.append({
            "training_step": training_step, "checkpoint_sha256": file_sha256(checkpoint_path),
            "scenes": scene_results,
            "maximum_candidate_interaction_error": max(record["query_constraint"]["candidate_interaction_max_error"] for record in scene_results),
            "mean_attention_gradient_norm": float(np.mean([record["attention_gradient_norm"] for record in scene_results])),
            "mean_visual_token_gradient_norm": float(np.mean([record["visual_token_gradient_norm"] for record in scene_results])),
            "intervention_summary": {
                name: {
                    "mean_absolute_NC_DAC_logit_change": float(np.mean([record["input_interventions"][name]["mean_absolute_NC_DAC_logit_change"] for record in scene_results])),
                    "selected_candidate_changes": sum(record["input_interventions"][name]["selected_candidate_changed"] for record in scene_results),
                }
                for name in intervention_blocks
            },
        })
    protected_after = protected_training_audit(configuration)
    summary = {
        "complete": True, "diagnostic_scenes": len(selected_rows),
        "selected_rows": selected_rows, "fixed_subset_source": str(pilot_result / "diagnostic_subset.json"),
        "optimizer_steps": 0, "gpu_seconds": 0., "encoder_calls": 0,
        "python": platform.python_version(), "torch": torch.__version__,
        "script_sha256": file_sha256(Path(__file__)), "original_sources": registration["sources"],
        "protected_before": protected_before, "protected_after": protected_after,
        "synthetic_nonlinear_query_capacity_check": verify_nonlinear_joint_query_releases_constraint(),
        "training_labels": {
            "NC_all_safe_scenes": int((baseline_input_labels[:, :, 0] == 1).all(1).sum()),
            "DAC_all_safe_scenes": int((baseline_input_labels[:, :, 1] == 1).all(1).sum()),
            "NC_candidate_fraction_fully_safe": float((baseline_input_labels[:, :, 0] == 1).mean()),
            "DAC_candidate_fraction_fully_safe": float((baseline_input_labels[:, :, 1] == 1).mean()),
        },
        "checkpoints": checkpoint_results, "cpu_wall_seconds": time.perf_counter() - started,
        "interpretation": "Additive query constraint is algebraic. Input zeroing is OOD functional diagnosis, not causal attribution or a newly trained ablation. Coordinate bands are not semantic sky/road masks.",
    }
    write_json(results_directory / "structure_diagnostics.json", summary)
    print(json.dumps({
        "complete": True, "diagnostic_scenes": len(selected_rows),
        "cpu_wall_seconds": summary["cpu_wall_seconds"], "optimizer_steps": 0,
        "summary": [{key: value for key, value in checkpoint.items() if key != "scenes"} for checkpoint in checkpoint_results],
        "training_labels": summary["training_labels"], "protected_after": protected_after,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, default=Path("configs/metric_visual_evidence/pilot_v1.json"))
    parser.add_argument("--results-directory", type=Path, default=Path("results/metric_visual_reader_structure_v1"))
    arguments = parser.parse_args()
    main(arguments.configuration, arguments.results_directory)
