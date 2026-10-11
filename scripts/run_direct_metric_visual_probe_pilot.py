#!/usr/bin/env python
"""Bounded shared/specific NC/DAC probe over immutable existing spatial caches."""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from calibrate_frozen_drivor_scorer import file_sha256, protected_training_audit, recording_bootstrap, write_json
from run_metric_visual_evidence_pilot import (
    load_artifacts, metric_summary, native_labels, ranking_diagnostics,
    verify_registration, whole_card_memory,
)
from planning_aware_future_prediction.models.direct_metric_visual_probe import (
    DirectMetricVisualProbe, replace_nc_dac_logits,
)
from planning_aware_future_prediction.models.metric_visual_evidence import compute_metric_evidence_losses
from planning_aware_future_prediction.models.scorer_selection_learning import compose_drivor_log_score

CONDITIONS = ("shared", "metric_specific")


class CachedMetricProbeScenes:
    def __init__(self, configuration):
        self.configuration = configuration
        _, self.records, original_cache, self.score_weights, self.output, self.result = load_artifacts(configuration)
        source_cache = PROJECT_ROOT / configuration["source_visual_cache_directory"]
        self.spatial_tokens = np.load(source_cache / "spatial_tokens.npy", mmap_mode="r")
        self.valid_token_mask = np.load(source_cache / "valid_token_mask.npy", mmap_mode="r")
        self.positions = np.load(source_cache / "token_positions.npy")
        self.ego_status = original_cache["ego_status"]
        self.trajectories = original_cache["native_trajectories"]
        self.original_logits = original_cache["native_component_logits"]
        self.targets = native_labels(configuration)

    def rows(self, subset):
        return [index for index, record in enumerate(self.records) if record["pilot_split"] == subset]

    def batch(self, selected_rows, device, constant_visual=None):
        tokens = torch.from_numpy(np.array(self.spatial_tokens[selected_rows])).to(device).float()
        if constant_visual is not None:
            tokens = constant_visual[None, None].expand_as(tokens)
        model_inputs = {
            "visual_tokens": tokens,
            "valid_token_mask": torch.from_numpy(np.array(self.valid_token_mask[selected_rows])).to(device),
            "positions": torch.from_numpy(self.positions).to(device),
            "ego_status": torch.from_numpy(np.array(self.ego_status[selected_rows])).to(device),
            "trajectories": torch.from_numpy(np.array(self.trajectories[selected_rows])).to(device),
        }
        return (model_inputs,
                torch.from_numpy(np.array(self.original_logits[selected_rows])).to(device),
                torch.from_numpy(np.array(self.targets[selected_rows])).to(device))


def initialize_probe(configuration, condition):
    return DirectMetricVisualProbe(
        384, configuration["reader_dimension"], configuration["hidden_dimension"],
        configuration["metric_dimension"], condition,
    )


def environment_snapshot(device):
    return {"python": platform.python_version(), "torch": torch.__version__,
            "device": str(device), "physical_gpu": 0 if device.type == "cuda" else None,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "bitwise_replay_verified": False}


def prepare(configuration, configuration_path):
    previous_configuration = json.loads((PROJECT_ROOT / configuration["source_visual_pilot_configuration"]).read_text())
    previous_registration = verify_registration(previous_configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    scene_cache.output.mkdir(parents=True, exist_ok=True)
    scene_cache.result.mkdir(parents=True, exist_ok=True)
    source_cache = PROJECT_ROOT / configuration["source_visual_cache_directory"]
    metadata = json.loads((source_cache / "patch_cache_complete.json").read_text())
    assert metadata["future_input"] is False
    assert metadata["frozen_parameters_unchanged"] is True
    assert metadata["shape"] == [1280, 1406, 384]
    cache_started = time.perf_counter()
    for filename, digest in metadata["cache_sha256"].items():
        assert file_sha256(source_cache / filename) == digest, filename
    cache_verification_seconds = time.perf_counter() - cache_started
    assert len(scene_cache.rows("train")) == configuration["training_scenes"] == 1024
    assert len(scene_cache.rows("tuning")) == configuration["tuning_scenes"] == 64
    assert len(scene_cache.rows("locked_development")) == configuration["evaluation_scenes"] == 192
    recording_groups = {subset: {scene_cache.records[row]["recording_group"] for row in scene_cache.rows(subset)}
                        for subset in ("train", "tuning", "locked_development")}
    assert not recording_groups["train"].intersection(recording_groups["tuning"] | recording_groups["locked_development"])
    assert not recording_groups["tuning"].intersection(recording_groups["locked_development"])
    assert configuration["training_steps"] == 500 and configuration["batch_size"] == 8
    torch.manual_seed(configuration["seed"])
    common_initialization = initialize_probe(configuration, "metric_specific").state_dict()
    torch.save(common_initialization, scene_cache.output / "common_initialization.pt")
    # Copy actual previously executed minibatch/pair sequence, rather than only
    # specifying the same seed. No profiling state will enter training.
    for filename in ("batch_order.npy", "pair_indices.npy"):
        content = np.load(source_cache / filename)
        np.save(scene_cache.output / filename, content)
        assert file_sha256(scene_cache.output / filename) == file_sha256(source_cache / filename)
    source_files = [configuration_path, Path(__file__),
                    PROJECT_ROOT / "src/planning_aware_future_prediction/models/direct_metric_visual_probe.py",
                    PROJECT_ROOT / "tests/test_direct_metric_visual_probe.py",
                    PROJECT_ROOT / "scripts/diagnose_direct_metric_visual_probe.py",
                    PROJECT_ROOT / "scripts/run_direct_metric_visual_probe_pilot.sh"]
    input_files = [PROJECT_ROOT / configuration["source_visual_pilot_configuration"],
                   source_cache / "registration.json", source_cache / "patch_cache_complete.json",
                   source_cache / "batch_order.npy", source_cache / "pair_indices.npy",
                   scene_cache.output / "common_initialization.pt"]
    input_hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in input_files}
    # Prior registration includes checkpoint, sample identities and official targets.
    input_hashes.update(previous_registration["inputs"])
    registration = {
        "registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "sources": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_files},
        "inputs": input_hashes, "records": scene_cache.records,
        "recording_counts": {name: len(groups) for name, groups in recording_groups.items()},
        "cache_sha256": metadata["cache_sha256"], "cache_metadata": metadata,
        "cache_hash_verification_seconds": cache_verification_seconds,
        "cache_generation_calls": 0, "official_candidate_label_calls": 0,
        "protected_training": protected_training_audit(configuration),
    }
    write_json(scene_cache.output / "registration.json", registration)
    write_json(scene_cache.result / "subset_manifest.json", registration)
    write_json(scene_cache.result / "configuration.json", configuration)
    write_json(scene_cache.result / "cache_reuse.json", {
        "source": str(source_cache), "metadata": metadata,
        "new_encoder_calls": 0, "same_sample_order_as_prior_pilot": True,
        "same_batch_and_pair_order_as_prior_pilot": True,
        "cache_hash_verification_seconds": cache_verification_seconds,
    })
    unit = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_direct_metric_visual_probe.py"],
                          cwd=PROJECT_ROOT, capture_output=True, text=True,
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    write_json(scene_cache.result / "unit_validation.json", {
        "passed": unit.returncode == 0, "stdout": unit.stdout, "stderr": unit.stderr,
    })
    assert unit.returncode == 0, unit.stdout + unit.stderr
    protocol = """# 직접 NC·DAC 시각 probe: 실행 전 등록\n\n
같은 frozen 공간 cache와 고정64후보에서 기존 scorer / shared 직접reader / metric-specific 직접reader를 비교한다.
두 reader는 같은 초기 state·90,402개 파라미터·batch8·seed0·500step·AdamW3e-4·BCE+PDMS-gap ranking을 사용한다.
이전 pilot의 실제 배치·pair 순서를 복사한다. Profile 가중치는 폐기한다. 튜닝64의 PDMS로100/250/500 중 선택한다.
모델은 공간feature·위치·ego·후보만 받는다. Head 입력은 visual readout만이며 원 scorer feature/logit과 직접 ego/path는 없다.
Metric/candidate query는 비선형 결합하고 position은key에만 사용한다. NC/DAC map을 강제로 분리하는 loss는 없다.
원 scorer의 다른4지표와 후보64개는 고정한다. OFF는 원6logit과 선택을 정확히 복원한다.
Planning gate와 visual-reading gate는 configuration.json에 사전 등록한 기준을 각각 평가한다.
기존 NAVTRAIN1024·튜닝NAVVAL64·개발NAVVAL192; recording 중복0. 이전 개발셋 재사용/단일seed/축소DrivoR baseline이다.
NC/DAC 정답은 후보별 공식score이며 semantic box/segmentation 정답이 아니다.
새학습은 이 두 reader의500step으로 종료한다. Fullscale/추가seed/새semantic감독을 자동 시작하지 않는다.
GPU는 물리0만, process2.5GiB allocator/전체카드48decimalGB. 원LPWM 학습과 등록289source를 보존한다.
"""
    (scene_cache.result / "preregistered_protocol.md").write_text(protocol)
    write_json(scene_cache.output / "prepare_complete.json", {"complete": True})
    print(json.dumps({"phase": "prepare", "cache_reused": True, "unit_tests_passed": True}), flush=True)


def validate(configuration):
    verify_registration(configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    device = torch.device("cpu")
    initialization = torch.load(scene_cache.output / "common_initialization.pt", map_location=device, weights_only=True)
    specific = initialize_probe(configuration, "metric_specific")
    shared = initialize_probe(configuration, "shared")
    specific.load_state_dict(initialization)
    shared.load_state_dict(initialization)
    counts = [sum(parameter.numel() for parameter in model.parameters()) for model in (specific, shared)]
    assert counts[0] == counts[1]
    model_inputs, baseline_logits, targets = scene_cache.batch([0, 1], device)
    prediction, attention = specific(**model_inputs, capture_attention=True)
    relative_odds = attention.log()[:, :, 0] - attention.log()[:, :, 1]
    relative_odds = relative_odds - relative_odds[..., :1]
    candidate_dependence = (relative_odds - relative_odds[:, :1]).abs().max().item()
    assert candidate_dependence > 1e-7
    component, ranking = compute_metric_evidence_losses(
        replace_nc_dac_logits(baseline_logits, prediction), targets, scene_cache.score_weights,
        torch.randint(0, 64, (2, 128, 2), generator=torch.Generator().manual_seed(43)),
    )
    (component + ranking).backward()
    gradient_groups = {
        name: math.sqrt(sum(parameter.grad.detach().square().sum().item() for parameter in module.parameters()))
        for name, module in (("query", specific.candidate_metric_query), ("keys", specific.key_projection),
                             ("values", specific.value_projection), ("head", specific.metric_prediction_head))
    }
    gradient_groups["metric_embedding"] = specific.metric_embedding.grad.norm().item()
    assert all(np.isfinite(value) and value > 0 for value in gradient_groups.values())
    assert all(torch.isfinite(parameter.grad).all() for parameter in specific.parameters())
    assert all(tensor.grad is None for tensor in (*model_inputs.values(), baseline_logits))
    with torch.no_grad():
        _, shared_attention = shared(**model_inputs, capture_attention=True)
        assert torch.equal(shared_attention[:, :, 0], shared_attention[:, :, 1])
        constant_inputs = {**model_inputs, "visual_tokens": model_inputs["visual_tokens"].mean((0, 1), keepdim=True).expand_as(model_inputs["visual_tokens"])}
        constant_prediction, _ = specific(**constant_inputs)
        changed_prediction, _ = specific(**{**constant_inputs,
                                           "ego_status": model_inputs["ego_status"] * 3.,
                                           "trajectories": model_inputs["trajectories"] * -2.,
                                           "positions": model_inputs["positions"] * 5.})
        constant_condition_error = (constant_prediction - changed_prediction).abs().max().item()
        assert constant_condition_error < 2e-5
        off = replace_nc_dac_logits(baseline_logits, prediction, probe_enabled=False)
        assert torch.equal(off, baseline_logits)
        other_metrics_exact = torch.equal(replace_nc_dac_logits(baseline_logits, prediction)[..., 2:], baseline_logits[..., 2:])
        assert other_metrics_exact
    write_json(scene_cache.result / "actual_cache_validation.json", {
        "passed": True, "samples": [scene_cache.records[row]["token"] for row in (0, 1)],
        "trainable_parameters": counts[0], "shared_specific_parameter_count_equal": True,
        "first_update_gradient_groups": gradient_groups,
        "metric_relative_odds_candidate_dependence": candidate_dependence,
        "constant_values_condition_intervention_max_logit_change": constant_condition_error,
        "original_logits_off_exact": True, "other_four_metrics_exact": other_metrics_exact,
        "original_scorer_and_encoder_modules_in_model": False, "encoder_calls": 0,
        "cache_inputs_have_no_gradient": True, "environment": environment_snapshot(device),
        "prior_live_cache_equivalence": json.loads((PROJECT_ROOT / configuration["source_visual_cache_directory"] / "patch_cache_complete.json").read_text())["same_forward_original_route_errors"],
        "protected_training": protected_training_audit(configuration),
    })
    write_json(scene_cache.output / "validation_complete.json", {"complete": True})
    print(json.dumps({"phase": "validate", "passed": True, "parameters": counts[0]}), flush=True)


def require_gpu(configuration):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == str(configuration["physical_gpu"]) == "0"
    assert torch.cuda.is_available(), "Host GPU access is required; CPU validation remains separate"
    torch.cuda.set_per_process_memory_fraction(
        configuration["allocator_limit_gib"] * 1024**3 / torch.cuda.get_device_properties(0).total_memory,
        device=0,
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    whole_card_memory(configuration)
    return torch.device("cuda:0")


def probe_loss(configuration, scene_cache, model, selected_rows, pair_indices, device):
    model_inputs, original_logits, targets = scene_cache.batch(selected_rows, device)
    prediction, _ = model(**model_inputs)
    combined = replace_nc_dac_logits(original_logits, prediction)
    pairs = pair_indices.to(device).clone()
    pairs[:, 0, 0] = compose_drivor_log_score(original_logits, scene_cache.score_weights).argmax(1)
    pairs[:, 0, 1] = targets[:, :, 6].argmax(1)
    component, ranking = compute_metric_evidence_losses(combined, targets, scene_cache.score_weights, pairs,
                                                      configuration["ranking_temperature"])
    loss = configuration["component_loss_weight"] * component + configuration["ranking_loss_weight"] * ranking
    return loss, component, ranking


def predict(configuration, scene_cache, model, selected_rows, device, intervention=None, constant_visual=None):
    scores, logits = [], []
    with torch.no_grad():
        for first in range(0, len(selected_rows), configuration["batch_size"]):
            inputs, baseline, _ = scene_cache.batch(selected_rows[first:first+configuration["batch_size"]], device, constant_visual)
            prediction, _ = model(**inputs, intervention=intervention)
            combined = replace_nc_dac_logits(baseline, prediction)
            scores.append(compose_drivor_log_score(combined, scene_cache.score_weights).cpu().numpy())
            logits.append(combined.cpu().numpy())
    return np.concatenate(scores), np.concatenate(logits)


def profile(configuration):
    verify_registration(configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    device = require_gpu(configuration)
    model = initialize_probe(configuration, "metric_specific").to(device)
    model.load_state_dict(torch.load(scene_cache.output / "common_initialization.pt", map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=configuration["learning_rate"])
    batches = np.load(scene_cache.output / "batch_order.npy")
    pairs = torch.from_numpy(np.load(scene_cache.output / "pair_indices.npy"))
    timings = []
    maximum_card = whole_card_memory(configuration)
    for step in range(configuration["profile_steps"]):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss, _, _ = probe_loss(configuration, scene_cache, model, batches[step], pairs[step], device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
    median = float(np.median(timings[2:]))
    estimated = median * configuration["training_steps"] * 2
    assert estimated < configuration["maximum_combined_training_gpu_hours"] * 3600
    maximum_card = max(maximum_card, whole_card_memory(configuration))
    profile_result = {
        "complete": True, "profile_steps": len(timings), "median_step_seconds": median,
        "estimated_combined_training_seconds": estimated, "gpu_profile_seconds": sum(timings),
        "profile_weights_used_for_training": False, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "maximum_whole_card_bytes": maximum_card, "environment": environment_snapshot(device),
    }
    write_json(scene_cache.result / "profiling.json", profile_result)
    write_json(scene_cache.output / "profile_complete.json", profile_result)
    print(json.dumps({"phase": "profile", "estimated_training_seconds": estimated}), flush=True)


def train(configuration):
    verify_registration(configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    device = require_gpu(configuration)
    initialization = torch.load(scene_cache.output / "common_initialization.pt", map_location="cpu", weights_only=True)
    batches = np.load(scene_cache.output / "batch_order.npy")
    pairs = torch.from_numpy(np.load(scene_cache.output / "pair_indices.npy"))
    tuning_rows = scene_cache.rows("tuning")
    summaries = {}
    for condition in CONDITIONS:
        model = initialize_probe(configuration, condition).to(device)
        model.load_state_dict(copy.deepcopy(initialization))
        optimizer = torch.optim.AdamW(model.parameters(), lr=configuration["learning_rate"])
        model.train()
        timings, logs = [], []
        best_pdms, best_step = -float("inf"), None
        maximum_card = whole_card_memory(configuration)
        started = time.perf_counter()
        for step, selected_rows in enumerate(batches, 1):
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            loss, component, ranking = probe_loss(configuration, scene_cache, model, selected_rows, pairs[step-1], device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.).item()
            if step == 1 or step % 25 == 0:
                reader_gradient = math.sqrt(sum(parameter.grad.detach().square().sum().item() for name, parameter in model.named_parameters()
                                               if parameter.grad is not None and not name.startswith("metric_prediction_head")))
                logs.append({"step": step, "loss": loss.item(), "component_loss": component.item(), "ranking_loss": ranking.item(),
                             "gradient_norm_before_clip": norm, "reader_gradient_norm": reader_gradient})
                write_json(scene_cache.output / f"{condition}_progress.json", logs[-1])
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - step_started)
            if step in configuration["checkpoint_steps"]:
                maximum_card = max(maximum_card, whole_card_memory(configuration))
                model.eval()
                tuning_started = time.perf_counter()
                scores, _ = predict(configuration, scene_cache, model, tuning_rows, device)
                tuning_pdms = float(scene_cache.targets[tuning_rows, scores.argmax(1), 6].mean() * 100)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "completed_steps": step,
                            "environment": environment_snapshot(device)}, scene_cache.output / f"{condition}_step{step}.pt")
                logs[-1]["tuning_pdms"] = tuning_pdms
                logs[-1]["tuning_wall_seconds"] = time.perf_counter() - tuning_started
                if tuning_pdms > best_pdms:
                    best_pdms, best_step = tuning_pdms, step
                print(json.dumps({"phase": "train", "condition": condition, "step": step, "tuning_pdms": tuning_pdms}), flush=True)
                model.train()
        summaries[condition] = {
            "completed_steps": len(timings), "best_tuning_step": best_step, "best_tuning_pdms": best_pdms,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "gpu_training_step_seconds": sum(timings), "median_step_seconds": float(np.median(timings)),
            "phase_wall_seconds": time.perf_counter() - started, "maximum_whole_card_bytes": maximum_card,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "logs": logs,
            "environment": environment_snapshot(device),
        }
        write_json(scene_cache.output / f"{condition}_training.json", summaries[condition])
        del optimizer, model
        torch.cuda.empty_cache()
    assert sum(value["gpu_training_step_seconds"] for value in summaries.values()) < configuration["maximum_combined_training_gpu_hours"] * 3600
    summary = {"complete": True, "conditions": summaries,
               "encoder_calls": 0, "original_scorer_calls": 0,
               "batch_order_sha256": file_sha256(scene_cache.output / "batch_order.npy"),
               "pair_order_sha256": file_sha256(scene_cache.output / "pair_indices.npy"),
               "initialization_sha256": file_sha256(scene_cache.output / "common_initialization.pt"),
               "protected_training": protected_training_audit(configuration)}
    write_json(scene_cache.output / "training_complete.json", summary)
    print(json.dumps({"phase": "train", "complete": True}), flush=True)


def load_probe(configuration, condition, device, step=None):
    scene_cache = CachedMetricProbeScenes(configuration)
    if step is None:
        step = json.loads((scene_cache.output / "training_complete.json").read_text())["conditions"][condition]["best_tuning_step"]
    model = initialize_probe(configuration, condition).to(device)
    checkpoint = torch.load(scene_cache.output / f"{condition}_step{step}.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval(), step


def report(configuration):
    import csv
    verify_registration(configuration)
    scene_cache = CachedMetricProbeScenes(configuration)
    device = require_gpu(configuration)
    selected_rows = scene_cache.rows("locked_development")
    records = [scene_cache.records[row] for row in selected_rows]
    labels = scene_cache.targets[selected_rows]
    original_logits = np.array(scene_cache.original_logits[selected_rows])
    all_logits = {"unchanged": original_logits}
    all_scores = {"unchanged": compose_drivor_log_score(torch.from_numpy(original_logits), scene_cache.score_weights).numpy()}
    # The replacement reference uses train samples only, never validation images.
    mean_feature = np.zeros(384, dtype="float64")
    training_rows = scene_cache.rows("train")
    for first in range(0, len(training_rows), 32):
        mean_feature += np.array(scene_cache.spatial_tokens[training_rows[first:first+32]], dtype="float32").sum((0, 1), dtype="float64")
    mean_feature = torch.from_numpy((mean_feature / (len(training_rows) * 1406)).astype("float32")).to(device)
    np.save(scene_cache.output / "training_mean_visual_feature.npy", mean_feature.cpu().numpy())
    interventions, latency, selected_steps, last_step_scores = {}, {}, {}, {}
    maximum_card = whole_card_memory(configuration)
    started = time.perf_counter()
    for condition in CONDITIONS:
        model, step = load_probe(configuration, condition, device)
        selected_steps[condition] = step
        all_scores[condition], all_logits[condition] = predict(configuration, scene_cache, model, selected_rows, device)
        assert np.array_equal(all_logits[condition][..., 2:], original_logits[..., 2:])
        if condition == "metric_specific":
            interventions[condition] = {}
            for intervention in ("uniform_attention", "swap_metric_attention", "patch_constant"):
                scores, logits = predict(configuration, scene_cache, model, selected_rows, device,
                                         None if intervention == "patch_constant" else intervention,
                                         mean_feature if intervention == "patch_constant" else None)
                name = condition + "_" + intervention
                all_scores[name], all_logits[name] = scores, logits
                interventions[condition][intervention] = {
                    "selected_index_changed_fraction": float((scores.argmax(1) != all_scores[condition].argmax(1)).mean()),
                    "mean_absolute_log_score_change": float(np.abs(scores - all_scores[condition]).mean()),
                }
        inputs, baseline, _ = scene_cache.batch(selected_rows[:configuration["batch_size"]], device)
        with torch.no_grad():
            predicted, _ = model(**inputs)
            off = replace_nc_dac_logits(baseline, predicted, probe_enabled=False)
            assert torch.equal(off, baseline)
            for _ in range(5):
                model(**inputs)
            torch.cuda.synchronize()
            latency_started = time.perf_counter()
            for _ in range(30):
                model(**inputs)
            torch.cuda.synchronize()
            latency[condition] = 1000 * (time.perf_counter() - latency_started) / (30 * len(inputs["visual_tokens"]))
        last_model, _ = load_probe(configuration, condition, device, step=500)
        last_step_scores[condition], _ = predict(configuration, scene_cache, last_model, selected_rows, device)
        maximum_card = max(maximum_card, whole_card_memory(configuration))
        del last_model, model
    # A train-only prior is a task-level sanity reference, not an equal-capacity
    # or candidate-conditioned baseline.
    train_probability = scene_cache.targets[training_rows, :, :2].mean((0, 1)).clip(1e-6, 1-1e-6)
    prior_logits = np.log(train_probability / (1-train_probability))
    all_logits["train_only_metric_prior"] = original_logits.copy()
    all_logits["train_only_metric_prior"][..., :2] = prior_logits[None, None]
    all_scores["train_only_metric_prior"] = compose_drivor_log_score(torch.from_numpy(all_logits["train_only_metric_prior"]), scene_cache.score_weights).numpy()
    summaries, selected_pdms, selected_indices = {}, {}, {}
    for name, scores in all_scores.items():
        summaries[name], selected_pdms[name], selected_indices[name] = metric_summary(scores, labels, records)
        summaries[name]["ranking_calibration"] = ranking_diagnostics(scores, all_logits[name], labels)
        summaries[name]["selected_index_changed_from_original_fraction"] = float((scores.argmax(1) != all_scores["unchanged"].argmax(1)).mean())
        summaries[name]["mean_NC_DAC_BCE"] = float(np.mean([summaries[name]["ranking_calibration"]["calibration"][metric]["BCE"] for metric in ("NC", "DAC")]))
    difference = selected_pdms["metric_specific"] - selected_pdms["shared"]
    groups = [record["recording_group"] for record in records]
    confidence_interval = recording_bootstrap(difference, groups,
                                             seed=configuration["seed"],
                                             repetitions=configuration["bootstrap_repetitions"])
    safety_ok = all(summaries["metric_specific"][metric] <= summaries[other][metric] + .01
                    for metric in ("NC_failure_rate", "DAC_failure_rate") for other in ("unchanged", "shared"))
    planning_gate = confidence_interval[0] > 0 and safety_ok
    specific_bce = summaries["metric_specific"]["mean_NC_DAC_BCE"]
    visual_gate = (specific_bce < summaries["train_only_metric_prior"]["mean_NC_DAC_BCE"]
                   and specific_bce < summaries["metric_specific_patch_constant"]["mean_NC_DAC_BCE"])
    rows = []
    for name in ("unchanged", *CONDITIONS):
        for index, record in enumerate(records):
            rows.append({"condition": name, "sample_id": record["token"], "recording_group": record["recording_group"],
                         "command_index": record["command_index"], "selected_candidate": int(selected_indices[name][index]),
                         "selected_pdms": float(selected_pdms[name][index])})
    with (scene_cache.result / "scene_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (scene_cache.result / "comparison.csv").open("w", newline="") as stream:
        fields = ["condition", "selected_pdms", "oracle_pdms", "selection_regret", "NC_failure_rate", "DAC_failure_rate", "mean_NC_DAC_BCE"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for name in ("unchanged", *CONDITIONS, "train_only_metric_prior"):
            writer.writerow({"condition": name, **{field: summaries[name][field] for field in fields[1:]}})
    np.savez_compressed(scene_cache.result / "evaluation_predictions.npz",
                        sample_ids=np.array([record["token"] for record in records]),
                        **{name + "_scores": scores for name, scores in all_scores.items()},
                        **{name + "_logits": logits for name, logits in all_logits.items()})
    registration = verify_registration(configuration)
    results = {
        "complete": True, "summaries": summaries, "selected_steps": selected_steps,
        "metric_specific_minus_shared_pdms": float(difference.mean()),
        "metric_specific_minus_shared_recording_95ci": confidence_interval,
        "planning_gate_passed": planning_gate, "visual_reading_gate_passed": visual_gate,
        "interventions": interventions, "original_off_exact": True,
        "last_step500_pdms_without_dev_selection": {
            name: metric_summary(scores, labels, records)[0]["selected_pdms"] for name, scores in last_step_scores.items()},
        "cached_probe_gpu_forward_ms_per_scene_batched8": latency,
        "gpu_report_wall_seconds": time.perf_counter() - started,
        "maximum_whole_card_bytes": maximum_card,
        "training": json.loads((scene_cache.output / "training_complete.json").read_text()),
        "profile": json.loads((scene_cache.output / "profile_complete.json").read_text()),
        "validation": json.loads((scene_cache.result / "actual_cache_validation.json").read_text()),
        "sources": registration["sources"], "input_hashes": registration["inputs"],
        "official_candidate_label_calls": 0, "cache_generation_calls": 0,
        "evaluation_scope": configuration["evaluation_scope"], "environment": environment_snapshot(device),
        "protected_training": protected_training_audit(configuration),
    }
    write_json(scene_cache.result / "results.json", results)
    write_json(scene_cache.output / "report_complete.json", {"complete": True})
    print(json.dumps({"phase": "report", "pdms": {name: summaries[name]["selected_pdms"] for name in ("unchanged", *CONDITIONS)},
                      "planning_gate": planning_gate, "visual_reading_gate": visual_gate}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, default=Path("configs/direct_metric_visual_probe/pilot_v1.json"))
    parser.add_argument("--phase", choices=("prepare", "validate", "profile", "train", "report"), required=True)
    arguments = parser.parse_args()
    configuration = json.loads(arguments.configuration.read_text())
    torch.set_num_threads(configuration["cpu_threads"])
    if arguments.phase == "prepare":
        prepare(configuration, arguments.configuration.resolve())
    else:
        {"validate": validate, "profile": profile, "train": train, "report": report}[arguments.phase](configuration)
