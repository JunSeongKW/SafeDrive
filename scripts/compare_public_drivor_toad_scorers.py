#!/usr/bin/env python
"""Frozen author checkpoints on common candidate pools, with official oracle."""
import argparse
import ast
import copy
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import time
from types import MethodType

import numpy as np

from calibrate_official_toad_navsim_v1_scorer import (
    PROJECT_ROOT, load_official_model, build_inputs, protected_files_audit,
    gpu_memory_status, write_json, file_sha256,
)
from calibrate_frozen_drivor_scorer import METRIC_NAMES, score as evaluate_candidates


def reference_configuration(configuration):
    return json.loads((PROJECT_ROOT / configuration["reference_configuration"]).read_text())


def prepare(configuration):
    reference = reference_configuration(configuration)
    previous_registration = json.loads((PROJECT_ROOT / configuration["reference_output_directory"] / "registration.json").read_text())
    assert len(previous_registration["records"]) == configuration["scene_count"]
    assert file_sha256(PROJECT_ROOT / configuration["drivor_checkpoint"]) == configuration["drivor_checkpoint_sha256"]
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT / configuration["drivor_repository"], text=True).strip() == configuration["drivor_repository_commit"]
    registration = {"configuration": configuration, "registered_utc": datetime.now(timezone.utc).isoformat(),
        "records": previous_registration["records"], "before_new_model_inference": True,
        "subset_reused_from_previous_development_diagnostic": True, "independent_final_test": False,
        "protected_training": protected_files_audit(reference)}
    output = PROJECT_ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "registration.json", registration)
    write_json(PROJECT_ROOT / configuration["results_directory"] / "subset_manifest.json", registration)
    write_json(PROJECT_ROOT / configuration["results_directory"] / "configuration.json", configuration)


def model_configuration_for(configuration, name):
    result = reference_configuration(configuration)
    result["official_random_bouquet_count"] = 0  # Native learned proposals only; common pools added externally.
    if name == "drivor_public":
        result.update(checkpoint=configuration["drivor_checkpoint"],
            expected_checkpoint_epoch=configuration["drivor_expected_checkpoint_epoch"],
            expected_checkpoint_global_step=configuration["drivor_expected_checkpoint_global_step"])
    return result


def original_drivor_forward(model, configuration):
    """Unchanged original forward body validates the compatibility checkout."""
    import torch
    source = PROJECT_ROOT / configuration["drivor_repository"] / "navsim/agents/drivoR/drivor_model.py"
    parsed = ast.parse(source.read_text())
    model_class = next(node for node in parsed.body if isinstance(node, ast.ClassDef) and node.name == "DrivoRModel")
    forward = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"torch": torch, "Dict": dict, "log": logging.getLogger("official-drivor-validation")}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), "exec"), namespace)
    return MethodType(namespace["forward"], model)


def score_metrics(model, trajectories, visual_features, ego_token):
    import torch
    captured = []
    def capture(_module, _arguments, returned):
        captured.append(returned[0])
    hook = model.scorer.register_forward_hook(capture)
    ranking = model._score_trajectories(trajectories, visual_features, ego_token)
    hook.remove()
    assert len(captured) == 1
    probabilities = torch.stack([captured[0][name].sigmoid() for name in METRIC_NAMES[:-1]], dim=-1)
    predicted_pdms = probabilities[..., 0] * probabilities[..., 1] * (
        5 * probabilities[..., 2] + 5 * probabilities[..., 3] + 2 * probabilities[..., 4]) / 12
    error = float((predicted_pdms - ranking.exp() / 12).abs().max())
    assert error <= 1e-6
    assert torch.isfinite(ranking).all() and torch.isfinite(probabilities).all()
    return ranking, probabilities, error


def infer(configuration):
    import torch
    from diagnose_residual_control_sampler import module_digest
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"])
    assert torch.cuda.is_available(), "Run this authorized small diagnostic outside the CUDA sandbox"
    torch.cuda.set_per_process_memory_fraction(configuration["maximum_process_bytes"] / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    reference = reference_configuration(configuration)
    output = PROJECT_ROOT / configuration["output_directory"]
    previous = np.load(PROJECT_ROOT / configuration["reference_output_directory"] / "scorer_cache_and_candidates.npz")
    records = json.loads((output / "registration.json").read_text())["records"]
    assert previous["tokens"].tolist() == [record["token"] for record in records]
    all_paths = np.zeros((len(records), 256, 8, 3), dtype=np.float32)
    all_paths[:, 64:128] = previous["trajectories"][:, :64]
    all_paths[:, 128:192] = previous["trajectories"][:, 64:128]
    all_paths[:, 192:256] = previous["trajectories"][:, 128:192]
    model_scores, model_metric_probabilities, loading_reports, model_checks = [], [], {}, {}
    original_cached_visual = original_cached_ego = None
    maximum_card_used = 0
    for name in configuration["models"]:
        model, loading = load_official_model(model_configuration_for(configuration, name), "cuda")
        loading_reports[name] = loading
        initial_digest = module_digest(model)
        current_scores, current_probabilities, current_checks = [], [], []
        current_visual, current_ego = [], []
        for scene_index, record in enumerate(records):
            if time.perf_counter() - started >= configuration["maximum_new_gpu_phase_seconds"]:
                raise RuntimeError("Small GPU diagnostic budget exceeded")
            maximum_card_used = max(maximum_card_used, gpu_memory_status(configuration)["card_used_bytes"])
            preprocessing_checks, original_forward_error = {}, None
            if name == "drivor_public":
                features, preprocessing_checks = build_inputs(record, model._config, scene_index < 3)
                features = {key: value.cuda() for key, value in features.items()}
                captured = []
                def capture_backbone(_module, _arguments, returned):
                    captured.append(returned.detach())
                hook = model.image_backbone.register_forward_hook(capture_backbone)
                with torch.no_grad():
                    prediction = model(features)
                    ego_token = model.hist_encoding(features["ego_status"][:, -1])[:, None]
                    assert len(captured) == 1
                    visual_features = captured[0]
                    if scene_index < 3:
                        native_original = original_drivor_forward(model, configuration)(features)
                        original_forward_error = max(float((native_original[key] - prediction[key]).abs().max())
                            for key in ("proposals", "pdm_score", "trajectory"))
                        assert original_forward_error <= 2e-6
                    native_replay, _, _ = score_metrics(model, prediction["proposals"], visual_features, ego_token)
                    native_replay_error = float((native_replay - prediction["pdm_score"]).abs().max())
                    assert native_replay_error <= 2e-6
                hook.remove()
                assert prediction["proposals"].shape == (1, 64, 8, 3)
                all_paths[scene_index, :64] = prediction["proposals"][0].cpu().numpy()
                current_visual.append(visual_features[0].cpu().numpy())
                current_ego.append(ego_token[0].cpu().numpy())
            else:
                visual_features = torch.from_numpy(previous["scene_features"][scene_index:scene_index+1]).cuda()
                ego_token = torch.from_numpy(previous["ego_tokens"][scene_index:scene_index+1]).cuda()
                native_replay_error = None
            cached_backbone_calls = []
            def reject_encoder(_module, _arguments):
                cached_backbone_calls.append(1)
                raise RuntimeError("Image backbone called during cached common-pool scoring")
            guard = model.image_backbone.register_forward_pre_hook(reject_encoder)
            pool_scores, pool_probabilities, conversion_errors = [], [], []
            with torch.no_grad():
                for begin, end in configuration["candidate_pools"].values():
                    paths = torch.from_numpy(all_paths[scene_index:scene_index+1, begin:end]).cuda()
                    ranking, probabilities, error = score_metrics(model, paths, visual_features, ego_token)
                    pool_scores.append(ranking[0].cpu().numpy())
                    pool_probabilities.append(probabilities[0].cpu().numpy())
                    conversion_errors.append(error)
            guard.remove()
            assert not cached_backbone_calls
            current_scores.append(np.concatenate(pool_scores))
            current_probabilities.append(np.concatenate(pool_probabilities))
            current_checks.append({"token": record["token"], "preprocessing": preprocessing_checks,
                "original_drivor_forward_maximum_error": original_forward_error,
                "native_replay_maximum_error": native_replay_error, "cached_backbone_calls": 0,
                "pdms_conversion_maximum_error": max(conversion_errors)})
            if (scene_index+1) % 16 == 0:
                print(json.dumps({"model": name, "scenes": scene_index+1,
                    "elapsed_seconds": time.perf_counter()-started}), flush=True)
        assert initial_digest == module_digest(model)
        assert all(parameter.grad is None for parameter in model.parameters())
        model_scores.append(np.stack(current_scores))
        model_metric_probabilities.append(np.stack(current_probabilities))
        model_checks[name] = current_checks
        if name == "drivor_public":
            original_cached_visual, original_cached_ego = np.stack(current_visual), np.stack(current_ego)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(output / "scorer_cache_and_candidates.npz", tokens=previous["tokens"], trajectories=all_paths,
        neural_log_scores=np.stack(model_scores), metric_probabilities=np.stack(model_metric_probabilities),
        native_rankings=model_scores[0][:, :128], nearby_rankings=model_scores[0][:, 128:192],
        secondary_rankings=model_scores[0][:, 128:144],
        drivor_scene_features=original_cached_visual, drivor_ego_tokens=original_cached_ego)
    summary = {"complete": True, "models": loading_reports, "checks": model_checks,
        "all_weights_unchanged": True, "all_weights_frozen": True, "all_weight_gradients_absent": True,
        "maximum_card_used_bytes": maximum_card_used, "memory": gpu_memory_status(configuration),
        "gpu_phase_wall_seconds": time.perf_counter() - started, "optimizer_steps": 0,
        "protected_training": protected_files_audit(reference), "source_sha256": file_sha256(Path(__file__)),
        "official_scorer_source_matches_original_drivor": file_sha256(PROJECT_ROOT / reference["official_repository"] / "navsim/agents/drivoR/score_module/scorer.py") == file_sha256(PROJECT_ROOT / configuration["drivor_repository"] / "navsim/agents/drivoR/score_module/scorer.py"),
        "official_scorer_decoder_source_matches_original_drivor": file_sha256(PROJECT_ROOT / reference["official_repository"] / "navsim/agents/drivoR/transformer_decoder.py") == file_sha256(PROJECT_ROOT / configuration["drivor_repository"] / "navsim/agents/drivoR/transformer_decoder.py"),
        "environment": {"torch": torch.__version__, "timm": __import__("timm").__version__}}
    write_json(output / "inference_complete.json", summary)
    print(json.dumps({"complete": True, "gpu_phase_wall_seconds": summary["gpu_phase_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--phase", choices=("prepare", "infer", "score"), required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text())
    if arguments.phase == "score":
        evaluate_candidates(configuration)
    else:
        globals()[arguments.phase](configuration)
