"""Bounded read-only profiling of the official and already-validated sparse paths.

Use one model, one preregistered smoke scene, no scorer, and no evaluation writes.
CUDA elapsed intervals include competition from other processes on shared GPUs.
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import statistics
import textwrap
import time
import types
from contextlib import contextmanager
from pathlib import Path

import evaluate_official_wa_jepa as official_evaluation
import numpy as np
import torch
from validate_wa_jepa_sparse_interface import (
    compute_selected_ids,
    load_sparse_native_module,
    use_sparse_interface_classes,
)


@contextmanager
def timed_cuda_region(records, region_name):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    try:
        yield
    finally:
        end_event.record()
        records.append({"region": region_name, "start": start_event, "end": end_event,
                        "cpu_enqueue_wall_seconds": time.perf_counter() - wall_start})


@contextmanager
def instrument_modules(model, records):
    handles = []
    for module, region_name in [(model.encoder, "encoder"), (model.predictor, "joint_predictor_step")]:
        active_regions = []

        def before_module(module, positional, keyword, region_name=region_name, active_regions=active_regions):
            interval = timed_cuda_region(records, region_name)
            active_regions.append(interval)
            interval.__enter__()

        def after_module(module, positional, keyword, output, active_regions=active_regions):
            active_regions.pop().__exit__(None, None, None)

        handles.extend([module.register_forward_pre_hook(before_module, with_kwargs=True),
                        module.register_forward_hook(after_module, with_kwargs=True)])
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def instrument_sparse_packing(model, sparse_module, records):
    """Wrap the existing packing block in memory; do not change its operations.

The measured block includes ID validation/device transfer/indices/two gathers.
Canonical positional lookups inside predictor are included in predictor time.
"""
    original_method = sparse_module.MultiViewCausalFutureMaskedJEPA.predict_trajectory
    method_ast = ast.parse(textwrap.dedent(inspect.getsource(original_method)))
    matching_blocks = [node for node in ast.walk(method_ast) if isinstance(node, ast.If)
                       and ast.unparse(node.test) == "future_token_ids is not None"]
    if len(matching_blocks) != 1:
        raise RuntimeError("Sparse packing boundary changed; do not guess the timing region")
    packing_block = matching_blocks[0]
    packing_block.body = [ast.With(items=[ast.withitem(context_expr=ast.Call(
        func=ast.Name(id="profile_packing_region", ctx=ast.Load()), args=[], keywords=[]))],
        body=packing_block.body)]
    ast.fix_missing_locations(method_ast)
    # no_grad's wrapper lives in torch.utils._contextlib, not the model module.
    function_globals = dict(inspect.unwrap(original_method).__globals__)
    function_globals["profile_packing_region"] = lambda: timed_cuda_region(records, "future_id_validation_and_packing")
    # Only trusted pinned reference source is compiled, not external input.
    exec(compile(method_ast, "<diagnostic_sparse_packing_only>", "exec"), function_globals)  # noqa: S102
    previous_instance_method = model.__dict__.get("predict_trajectory")
    model.predict_trajectory = types.MethodType(function_globals["predict_trajectory"], model)
    try:
        yield
    finally:
        if previous_instance_method is None:
            del model.predict_trajectory
        else:
            model.predict_trajectory = previous_instance_method


def profile_one_call(agent, scene_input, sparse_module, specification, token, condition):
    if torch.cuda.mem_get_info()[0] < 6 * 2**30:
        raise RuntimeError("Less than6GiB free; diagnostic stops without altering other jobs")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    region_records = []
    agent_wall_start = time.perf_counter()
    selection_start = time.perf_counter()
    future_ids = compute_selected_ids(agent.model, condition, token, specification["sparse_benchmark"])
    torch.cuda.synchronize()
    selection_wall_seconds = time.perf_counter() - selection_start
    preprocessing_start = time.perf_counter()
    features = agent.get_feature_builders()[0].compute_features(scene_input)
    features = {key: value.unsqueeze(0).to(agent._device) if isinstance(value, torch.Tensor) else value
                for key, value in features.items()}
    torch.cuda.synchronize()
    preprocessing_wall_seconds = time.perf_counter() - preprocessing_start
    model_wall_start = time.perf_counter()
    with torch.no_grad(), instrument_modules(agent.model, region_records), timed_cuda_region(region_records, "whole_model"):
        if future_ids is None:
            prediction = agent.model.predict_trajectory(features)
        else:
            with use_sparse_interface_classes(agent.model, sparse_module), instrument_sparse_packing(agent.model, sparse_module, region_records):
                prediction = agent.model.predict_trajectory(features, future_token_ids=future_ids)
    torch.cuda.synchronize()
    model_wall_seconds = time.perf_counter() - model_wall_start
    trajectory = prediction.squeeze(0).detach().float().cpu().numpy()
    agent_wall_seconds = time.perf_counter() - agent_wall_start
    assert trajectory.shape == (8, 3) and np.isfinite(trajectory).all()
    timings = [{"region": record["region"], "cuda_elapsed_seconds": record["start"].elapsed_time(record["end"]) / 1000,
                "cpu_enqueue_wall_seconds": record["cpu_enqueue_wall_seconds"]} for record in region_records]
    predictor_steps = [record["cuda_elapsed_seconds"] for record in timings if record["region"] == "joint_predictor_step"]
    assert len(predictor_steps) == 12
    return {
        "condition": condition, "scene_token": token, "future_tokens": 8192 if future_ids is None else future_ids.size(1),
        "encoder_cuda_seconds": sum(record["cuda_elapsed_seconds"] for record in timings if record["region"] == "encoder"),
        "joint_predictor_cuda_seconds": sum(predictor_steps),
        "predictor_per_step_cuda_seconds": predictor_steps,
        "predictor_mean_step_cuda_seconds": statistics.mean(predictor_steps),
        "token_selection_wall_seconds_including_transfer_and_sync": selection_wall_seconds,
        "packing_cuda_seconds": sum(record["cuda_elapsed_seconds"] for record in timings if record["region"] == "future_id_validation_and_packing"),
        "packing_cpu_enqueue_wall_seconds": sum(record["cpu_enqueue_wall_seconds"] for record in timings if record["region"] == "future_id_validation_and_packing"),
        "whole_model_cuda_seconds": next(record["cuda_elapsed_seconds"] for record in timings if record["region"] == "whole_model"),
        "whole_model_wall_seconds": model_wall_seconds,
        "preprocessing_and_transfer_wall_seconds": preprocessing_wall_seconds,
        "whole_agent_wall_seconds_including_selection_preprocessing_model": agent_wall_seconds,
        "sensor_io_and_scorer_excluded": True,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "trajectory_poses": trajectory.tolist(), "timing_regions": timings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--reuse-dense-report", type=Path, help="Preserve/reuse an already completed dense timing; run only missing packing conditions")
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "module_timing_results.json"
    if report_path.exists():
        raise RuntimeError("Existing diagnostic report must not be overwritten")
    # Redirect initialization metadata; do not overwrite the full-evaluation artifacts.
    original_save = official_evaluation.save_json
    official_evaluation.save_json = lambda path, report: original_save(output_directory / "initialization_metadata" / path.name, report)
    specification, source_root, _, configuration = official_evaluation.configure_official_runtime(workspace)
    preflight = json.loads((workspace / "results/official_wa_jepa_reproduction/evaluation_preflight.json").read_text())
    token = preflight["smoke_scene_tokens_fixed_before_gpu_inference"][0]
    agent = official_evaluation.initialize_official_agent(workspace, specification, source_root)
    scene_loader = official_evaluation.create_official_scene_loader(configuration, agent.get_sensor_config())
    scene_input = scene_loader.get_agent_input_from_token(token)
    sparse_module = load_sparse_native_module(workspace)
    trials = []
    warmup_dense_trajectory = None
    conditions = ["original_dense", "packed_all", "fixed_spatial_lattice"]
    if arguments.reuse_dense_report:
        previous_report = json.loads(arguments.reuse_dense_report.read_text())
        if previous_report["official_source_commit"] != specification["official_source_commit"] or previous_report["inference_steps"] != 12:
            raise RuntimeError("Existing dense measurement uses a different official model")
        trials = [record for record in previous_report["trial_records"] if record["condition"] == "original_dense"]
        if len(trials) != 1 or trials[0]["scene_token"] != token:
            raise RuntimeError("Expected one completed dense measurement for the fixed smoke scene")
        warmup_dense_trajectory = np.array(trials[0]["trajectory_poses"])
        conditions = ["packed_all", "fixed_spatial_lattice"]
    for condition in conditions:
        warmup = profile_one_call(agent, scene_input, sparse_module, specification, token, condition)
        if condition == "original_dense":
            warmup_dense_trajectory = np.array(warmup["trajectory_poses"])
        measured = profile_one_call(agent, scene_input, sparse_module, specification, token, condition)
        if condition in ("original_dense", "packed_all"):
            assert np.array_equal(np.array(measured["trajectory_poses"]), warmup_dense_trajectory), "Timing instrumentation changed the dense trajectory"
        trials.append(measured)
        report = {
            "official_source_commit": specification["official_source_commit"], "inference_steps": 12,
            "precision": "Official FP32/TF32, no autocast", "scene_count": 1, "warmup_calls_per_condition": 1,
            "timed_calls_per_condition": 1, "trial_records": trials,
            "dense_and_packed_all_trajectory_bitwise_equal": len(trials) >= 2,
            "reused_dense_report": str(arguments.reuse_dense_report) if arguments.reuse_dense_report else None,
            "limitations": "Shared GPU0 with running evaluation/other users. CUDA elapsed intervals include contention and CPU launch gaps, not isolated kernel compute. One scene/one timed call per condition; use earlier isolated six-scene benchmark for stable standalone latency. No scorer or disk I/O included.",
            "model_weights_settings_and_evaluation_records_changed": False,
        }
        original_save(report_path, report)
        print(f"MODULE_TIMING_COMPLETE condition={condition} encoder={measured['encoder_cuda_seconds']:.6f}s predictor={measured['joint_predictor_cuda_seconds']:.6f}s model={measured['whole_model_wall_seconds']:.6f}s packing={measured['packing_cuda_seconds']:.6f}s", flush=True)
    print("WAJEPA_MODULE_PROFILING_DONE", flush=True)


if __name__ == "__main__":
    main()
