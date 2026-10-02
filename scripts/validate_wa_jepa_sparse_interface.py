"""Native checkpoint all-ID equality, then untrained sparse inference profiling.

One official model is loaded. Temporary method/class substitution references a
separate worktree and shares all original parameter tensors, avoiding duplicate
model allocation. Original source files and the dense benchmark stay unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from evaluate_official_wa_jepa import (
    configure_official_runtime, create_official_scene_loader,
    initialize_official_agent, save_json,
)

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
from planning_aware_future_prediction.wa_jepa_sparse_token_ids import (
    build_canonical_future_token_ids, derive_scene_selection_seed,
)


def load_sparse_native_module(workspace):
    module_path = workspace / "reference_repositories/WAJEPASparseInterfaceValidation/models/multiview_causal_future_jepa.py"
    module_specification = importlib.util.spec_from_file_location("wa_jepa_sparse_native", module_path)
    module = importlib.util.module_from_spec(module_specification)
    sys.modules[module_specification.name] = module
    module_specification.loader.exec_module(module)
    return module


@contextmanager
def use_sparse_interface_classes(model, sparse_module):
    substitutions = [
        (model, sparse_module.MultiViewCausalFutureMaskedJEPA),
        (model.predictor, sparse_module.SceneTrajectoryFlowPredictor),
        (model.predictor.scene_positioner, sparse_module.SceneInputPositioner),
    ]
    original_classes = [(module, module.__class__) for module, _ in substitutions]
    try:
        for module, new_class in substitutions:
            module.__class__ = new_class
        yield
    finally:
        for module, original_class in original_classes:
            module.__class__ = original_class


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare_tensors(reference, candidate, name):
    identical = reference.shape == candidate.shape and torch.equal(reference, candidate)
    maximum_difference = float((reference.float() - candidate.float()).abs().max()) if reference.shape == candidate.shape else None
    return {"name": name, "shape": list(reference.shape), "bitwise_equal": identical, "maximum_absolute_difference": maximum_difference}


@contextmanager
def capture_native_denoising_trace(model):
    trace = {"steps": [], "positioner_outputs": []}
    def before_predictor(module, positional, keyword):
        trace["steps"].append({
            "scene_state_before": keyword["noisy_future_scene"].detach().cpu().clone(),
            "ego_state_before": keyword["trajectory_inputs"]["noisy_trajectory"].detach().cpu().clone(),
            "time": keyword["t_cont"].detach().cpu().clone(),
        })
    def after_predictor(module, positional, keyword, output):
        step = trace["steps"][-1]
        step["predicted_scene"] = output[0].detach().cpu().clone()
        step["predicted_trajectory"] = output[1].detach().cpu().clone()
        # Recompute the official GPU Euler operation for diagnostic post-states.
        # For all but the last step, compare these with the actual next pre-state.
        denominator = (1.0 - keyword["t_cont"]).clamp_min(1e-3).view(-1, 1, 1)
        step_size = 1.0 / float(model.flow_num_inference_steps)
        for state_name, previous, prediction in (
            ("scene_state_after_recomputed", keyword["noisy_future_scene"], output[0]),
            ("ego_state_after_recomputed", keyword["trajectory_inputs"]["noisy_trajectory"], output[1]),
        ):
            step[state_name] = (previous + step_size * (prediction - previous) / denominator).detach().cpu().clone()
    def after_positioner(module, positional, keyword, output):
        trace["positioner_outputs"].append(output.detach().cpu().clone())
    handles = [
        model.predictor.register_forward_pre_hook(before_predictor, with_kwargs=True),
        model.predictor.register_forward_hook(after_predictor, with_kwargs=True),
        model.predictor.scene_positioner.register_forward_hook(after_positioner, with_kwargs=True),
    ]
    try:
        yield trace
    finally:
        for handle in handles:
            handle.remove()


def validate_trace_transitions(trace):
    for index in range(len(trace["steps"]) - 1):
        for prefix in ("scene", "ego"):
            if not torch.equal(trace["steps"][index][f"{prefix}_state_after_recomputed"], trace["steps"][index + 1][f"{prefix}_state_before"]):
                raise RuntimeError(f"Diagnostic Euler state does not match actual next step: {index} {prefix}")


def compute_selected_ids(model, condition, scene_token, benchmark):
    if condition == "original_dense":
        return None
    quota = model.predictor.scene_tokens_per_step if condition == "packed_all" else benchmark["spatial_tubes_per_camera"]
    return build_canonical_future_token_ids(
        num_cameras=model.num_cameras, num_future_tubelets=model.predictor.scene_future_steps,
        spatial_grid_rows=model.predictor.scene_grid_h, spatial_grid_columns=model.predictor.scene_grid_w,
        selection_policy=condition, selected_spatial_tubes_per_camera=quota,
        selection_seed=derive_scene_selection_seed(scene_token, benchmark["random_selection_seed"]),
    ).to(device="cuda")


def run_condition(agent, agent_input, sparse_module, selected_ids):
    if selected_ids is None:
        return agent.compute_trajectory(agent_input)
    from navsim.common.dataclasses import Trajectory
    features = agent.get_feature_builders()[0].compute_features(agent_input)
    features = {key: value.unsqueeze(0).to(agent._device) if torch.is_tensor(value) else value for key, value in features.items()}
    with use_sparse_interface_classes(agent.model, sparse_module):
        prediction = agent.model.predict_trajectory(features, future_token_ids=selected_ids)
    poses = prediction.squeeze(0).detach().float().cpu().numpy().astype(np.float32)
    return Trajectory(poses)


def run_all_id_equality(workspace, agent, scene_loader, sparse_module, specification, scene_tokens):
    benchmark = specification["sparse_benchmark"]
    reports = []
    for scene_token in scene_tokens:
        agent_input = scene_loader.get_agent_input_from_token(scene_token)
        selected_ids = compute_selected_ids(agent.model, "packed_all", scene_token, benchmark)
        with capture_native_denoising_trace(agent.model) as original_trace:
            original_trajectory = run_condition(agent, agent_input, sparse_module, None)
        with capture_native_denoising_trace(agent.model) as packed_trace:
            packed_trajectory = run_condition(agent, agent_input, sparse_module, selected_ids)
        validate_trace_transitions(original_trace)
        validate_trace_transitions(packed_trace)
        comparisons = [compare_tensors(torch.from_numpy(original_trajectory.poses), torch.from_numpy(packed_trajectory.poses), "final_ego_trajectory")]
        for step_index, (original_step, packed_step) in enumerate(zip(original_trace["steps"], packed_trace["steps"])):
            for component in original_step:
                comparisons.append(compare_tensors(original_step[component], packed_step[component], f"step{step_index}:{component}"))
        for position_index, (original_position, packed_position) in enumerate(zip(original_trace["positioner_outputs"], packed_trace["positioner_outputs"])):
            comparisons.append(compare_tensors(original_position, packed_position, f"positioner_output{position_index}"))
        positioner = agent.model.predictor.scene_positioner
        original_position = positioner._position_embeddings(agent.model.predictor.future_token_indices, time_offset=agent.model.predictor.scene_history_steps, dtype=torch.float32, device=selected_ids.device).reshape(1, -1, positioner.scene_dim)
        with use_sparse_interface_classes(agent.model, sparse_module):
            canonical_position = positioner.canonical_future_position_embeddings(selected_ids, time_offset=agent.model.predictor.scene_history_steps, dtype=torch.float32, device=selected_ids.device)
        comparisons.append(compare_tensors(original_position.cpu(), canonical_position.cpu(), "raw_canonical_positional_embedding"))
        report = {
            "token": scene_token, "all_comparisons_bitwise_equal": all(item["bitwise_equal"] for item in comparisons),
            "original_steps": len(original_trace["steps"]), "packed_steps": len(packed_trace["steps"]),
            "initial_scene_noise_sha256": tensor_sha256(original_trace["steps"][0]["scene_state_before"]),
            "initial_ego_noise_sha256": tensor_sha256(original_trace["steps"][0]["ego_state_before"]),
            "comparisons": comparisons,
            "actual_transition_check": "Recomputed GPU Euler post-state equals actual next pre-state for11 transitions; final post-state recomputed because original method does not expose final scene state",
        }
        reports.append(report)
        save_json(workspace / "results/official_wa_jepa_reproduction/all_id_equivalence.json", {
            "tolerance_registered_before_inference": benchmark["all_id_tolerance"],
            "scene_records": reports, "all_id_gate_passed": len(reports) == len(scene_tokens) and all(item["all_comparisons_bitwise_equal"] for item in reports),
        })
        print(f"WAJEPA_ALL_ID {len(reports)}/{len(scene_tokens)} equal={report['all_comparisons_bitwise_equal']}", flush=True)
        if not report["all_comparisons_bitwise_equal"] or report["original_steps"] != 12 or report["packed_steps"] != 12:
            first_difference = next((item for item in comparisons if not item["bitwise_equal"]), None)
            raise RuntimeError(f"All-ID divergence, sparse benchmark prohibited: {first_difference}")
        del original_trace, packed_trace
    return reports


@contextmanager
def instrument_predictor_cost(model):
    instrumentation = {"predictor_events": [], "linear_shapes": [], "future_pipeline_events": None}
    handles = []
    def before_predictor(module, positional, keyword):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        instrumentation["predictor_events"].append((start, end))
    def after_predictor(module, positional, keyword, output):
        instrumentation["predictor_events"][-1][1].record()
    def record_linear_shape(module_name):
        def before_linear(module, inputs):
            instrumentation["linear_shapes"].append({"module": module_name, "input_shape": list(inputs[0].shape)})
        return before_linear
    handles.extend([
        model.predictor.register_forward_pre_hook(before_predictor, with_kwargs=True),
        model.predictor.register_forward_hook(after_predictor, with_kwargs=True),
    ])
    for name, module in model.predictor.named_modules():
        if isinstance(module, torch.nn.Linear) and (name.endswith(".qkv") or ".scene_ffn." in name or name in ("noisy_scene_proj", "future_condition_proj")):
            handles.append(module.register_forward_pre_hook(record_linear_shape(name)))
    original_future_condition = model._future_mask_condition
    def timed_future_condition(*arguments, **keyword):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        instrumentation["future_pipeline_events"] = (start, end)
        return original_future_condition(*arguments, **keyword)
    model._future_mask_condition = timed_future_condition
    try:
        yield instrumentation
    finally:
        model._future_mask_condition = original_future_condition
        for handle in handles:
            handle.remove()


def validate_cost_shapes(instrumentation, num_future_tokens):
    for item in instrumentation["linear_shapes"]:
        name, shape = item["module"], item["input_shape"]
        expected = 4096 if "context_proj.qkv" in name else 8 if "traj_proj.qkv" in name else num_future_tokens
        if shape[1] != expected:
            raise RuntimeError(f"Incorrect pre-linear token shape: {name}: {shape}, expected {expected}")


def run_sparse_cost_benchmark(workspace, agent, scene_loader, sparse_module, specification, configuration, scene_tokens):
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    benchmark = specification["sparse_benchmark"]
    metric_cache_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    simulator, scorer = instantiate(configuration.simulator), instantiate(configuration.scorer)
    scene_inputs = {token: scene_loader.get_agent_input_from_token(token) for token in scene_tokens}
    trial_records = []
    reference_trajectories = {}
    for condition in benchmark["conditions"]:
        for _ in range(benchmark["warmup_calls_per_condition"]):
            warmup_ids = compute_selected_ids(agent.model, condition, scene_tokens[0], benchmark)
            run_condition(agent, scene_inputs[scene_tokens[0]], sparse_module, warmup_ids)
            torch.cuda.synchronize()
        for scene_token in scene_tokens:
            for repeat_index in range(benchmark["timed_repeats_per_scene_condition"]):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                inference_start = time.perf_counter()
                selection_start = time.perf_counter()
                selected_ids = compute_selected_ids(agent.model, condition, scene_token, benchmark)
                selection_cpu_seconds = time.perf_counter() - selection_start
                num_future_tokens = 8192 if selected_ids is None else selected_ids.size(1)
                with instrument_predictor_cost(agent.model) as instrumentation:
                    trajectory = run_condition(agent, scene_inputs[scene_token], sparse_module, selected_ids)
                    instrumentation["future_pipeline_events"][1].record()
                    torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_start
                validate_cost_shapes(instrumentation, num_future_tokens)
                if condition == "original_dense" and repeat_index == 0:
                    reference_trajectories[scene_token] = trajectory.poses.copy()
                trajectory_difference = trajectory.poses - reference_trajectories[scene_token]
                scores = asdict(pdm_score(metric_cache_loader.get_from_token(scene_token), trajectory, simulator.proposal_sampling, simulator, scorer))
                record = {
                    "condition": condition, "token": scene_token, "repeat_index": repeat_index,
                    "current_context_tokens": 4096, "future_tokens": num_future_tokens, "ego_tokens": 8,
                    "total_joint_tokens": 4096 + num_future_tokens + 8,
                    "selection_generation_seconds_cpu_wall_including_id_to_device": selection_cpu_seconds,
                    "predictor_cuda_seconds": sum(start.elapsed_time(end) for start, end in instrumentation["predictor_events"]) / 1000,
                    "future_pipeline_cuda_seconds_including_noise_packing_euler": instrumentation["future_pipeline_events"][0].elapsed_time(instrumentation["future_pipeline_events"][1]) / 1000,
                    "overall_inference_seconds_including_selection_preprocessing_encoder_prediction": inference_seconds,
                    "sensor_io_excluded_from_timing": True,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "trajectory_xy_mean_difference_from_dense_m": float(np.linalg.norm(trajectory_difference[:, :2], axis=1).mean()),
                    "trajectory_max_absolute_difference_from_dense": float(np.abs(trajectory_difference).max()),
                    "trajectory_poses": trajectory.poses.tolist(), "scores_fraction": scores,
                    "linear_shape_checks_passed": True,
                    "representative_first_step_linear_shapes": instrumentation["linear_shapes"][:62],
                    "selected_ids_sha256": tensor_sha256(selected_ids) if selected_ids is not None else None,
                }
                trial_records.append(record)
            print(f"WAJEPA_SPARSE_COST {condition} scene={scene_token}", flush=True)
            save_json(workspace / "results/official_wa_jepa_reproduction/sparse_cost_trials.json", {
                "registered_benchmark": benchmark, "trial_records": trial_records,
                "interpretation": "Untrained token removal includes distribution shift; no learned-selection performance claim",
            })
    summaries = {}
    for condition in benchmark["conditions"]:
        records = [record for record in trial_records if record["condition"] == condition]
        metric_keys = ["predictor_cuda_seconds", "future_pipeline_cuda_seconds_including_noise_packing_euler", "overall_inference_seconds_including_selection_preprocessing_encoder_prediction", "peak_cuda_allocated_bytes", "peak_cuda_reserved_bytes", "trajectory_xy_mean_difference_from_dense_m"]
        summaries[condition] = {
            "scene_count": len(scene_tokens), "timed_trial_count": len(records),
            "future_tokens": records[0]["future_tokens"], "total_joint_tokens": records[0]["total_joint_tokens"],
            "metrics": {key: {"mean": float(np.mean([record[key] for record in records])), "median": float(np.median([record[key] for record in records])), "minimum": min(record[key] for record in records), "maximum": max(record[key] for record in records)} for key in metric_keys},
            "official_scores_fraction_scene_mean": {key: float(np.mean([record["scores_fraction"][key] for record in records if record["repeat_index"] == 0])) for key in records[0]["scores_fraction"]},
        }
    save_json(workspace / "results/official_wa_jepa_reproduction/sparse_cost_summary.json", {
        "conditions": summaries, "source_reference": specification["official_source_commit"],
        "dtype_actual": "float32 without autocast", "smoke_scenes_only_not_full_benchmark": True,
        "no_training_or_learned_selector": True,
        "timing_limitations": "CUDA events measure queued device intervals; overall wall includes ID generation/transfer, official preprocessing and encoder but excludes preloaded sensor disk I/O and scorer. Future-pipeline interval includes noise draw, packing,12predictor calls and Euler/decode; selection CPU wall reported separately.",
        "allocation_policy": "One loaded parameter set, one evaluation process on GPU1; CPU-only traces, no duplicate full model; no concurrent GPU1 full evaluation during benchmark",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--equality-only", action="store_true")
    parser.add_argument("--cost-only", action="store_true", help="Reuse completed all-ID equality if the native patch file has not changed")
    arguments = parser.parse_args()
    workspace = arguments.workspace.resolve()
    specification, source_root, _, configuration = configure_official_runtime(workspace)
    smoke = json.loads((workspace / "results/official_wa_jepa_reproduction/smoke_results.json").read_text())
    if not smoke["all_scenes_successful"]:
        raise RuntimeError("Original smoke did not pass")
    scene_tokens = json.loads((workspace / "results/official_wa_jepa_reproduction/evaluation_preflight.json").read_text())["smoke_scene_tokens_fixed_before_gpu_inference"]
    agent = initialize_official_agent(workspace, specification, source_root)
    sparse_module = load_sparse_native_module(workspace)
    scene_loader = create_official_scene_loader(configuration, agent.get_sensor_config())
    if arguments.cost_only:
        equality_path = workspace / "results/official_wa_jepa_reproduction/all_id_equivalence.json"
        equality = json.loads(equality_path.read_text())
        native_patch = workspace / "reference_repositories/WAJEPASparseInterfaceValidation/models/multiview_causal_future_jepa.py"
        if not equality["all_id_gate_passed"] or native_patch.stat().st_mtime > equality_path.stat().st_mtime:
            raise RuntimeError("All-ID gate missing or native patch changed; rerun equality first")
    else:
        run_all_id_equality(workspace, agent, scene_loader, sparse_module, specification, scene_tokens)
    if not arguments.equality_only:
        run_sparse_cost_benchmark(workspace, agent, scene_loader, sparse_module, specification, configuration, scene_tokens)
    print("WAJEPA_SPARSE_VALIDATION_DONE", flush=True)


if __name__ == "__main__":
    main()
