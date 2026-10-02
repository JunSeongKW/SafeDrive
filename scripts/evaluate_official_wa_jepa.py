"""Pinned official WA-JEPA agent/scorer with strict preflight and resumable records.

This harness changes evaluation scheduling/reporting only. It does not import the
pilot, implement its preprocessing, or modify the upstream agent or PDMS scorer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def compute_file_sha256(file_path):
    checksum = hashlib.sha256()
    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def save_json(result_path, report):
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def configure_official_runtime(workspace):
    specification = json.loads((workspace / "configs/official_wa_jepa/reproduction_v1.json").read_text())
    source_root = workspace / specification["official_source_worktree"]
    navsim_root = workspace / specification["navsim_source_worktree"]
    for repository, expected in ((source_root, specification["official_source_commit"]), (navsim_root, specification["navsim_source_commit"])):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
        if actual != expected:
            raise RuntimeError(f"Unexpected source: {repository}: {actual} != {expected}")
    # Explicit precedence prevents importing preserved SafeDrive/Drive-JEPA code.
    sys.path.insert(0, str(navsim_root))
    sys.path.insert(0, str(source_root))
    os.environ.update({
        "OPENSCENE_DATA_ROOT": str(workspace / "dataset"),
        "NAVSIM_EXP_ROOT": str(workspace / "outputs/official_wa_jepa_reproduction"),
        "NUPLAN_MAPS_ROOT": str(workspace / "dataset/maps"),
        "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
        "VJEPA2_CKPT": str(workspace / specification["checkpoint_directory"] / "vjepa2_1_vitl_dist_vitG_384.pt"),
    })
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(navsim_root / "navsim/planning/script/config/pdm_scoring"), version_base=None):
        configuration = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=navtest",
            f"metric_cache_path={workspace / specification['metric_cache_root']}",
        ])
    return specification, source_root, navsim_root, configuration


def create_official_scene_loader(configuration, sensor_config=None):
    from hydra.utils import instantiate
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SensorConfig
    return SceneLoader(
        data_path=Path(configuration.navsim_log_path),
        sensor_blobs_path=Path(configuration.sensor_blobs_path),
        scene_filter=instantiate(configuration.train_test_split.scene_filter),
        sensor_config=sensor_config or SensorConfig.build_no_sensors(),
    )


def run_preflight(workspace, specification, source_root, navsim_root, configuration):
    import torch
    from hydra.utils import instantiate
    from navsim.common.dataloader import MetricCacheLoader
    from omegaconf import OmegaConf
    results_root = workspace / "results/official_wa_jepa_reproduction"
    scene_loader = create_official_scene_loader(configuration)
    cache_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    expected_tokens = set(configuration.train_test_split.scene_filter.tokens)
    actual_tokens = set(scene_loader.tokens)
    required_images = set()
    observed_cadences = set()
    for frames in scene_loader.scene_frames_dicts.values():
        for history_index in range(4):
            camera_entries = {name.lower(): entry for name, entry in frames[history_index]["cams"].items()}
            for camera in specification["camera_order"]:
                required_images.add(str(Path(configuration.sensor_blobs_path) / camera_entries[camera]["data_path"]))
        observed_cadences.update(int(frames[index]["timestamp"]) - int(frames[index - 1]["timestamp"]) for index in range(1, 4))
    missing_images = [path for path in sorted(required_images) if not Path(path).is_file()]
    missing_cache_files = [str(cache_loader.metric_cache_paths[token]) for token in sorted(actual_tokens & set(cache_loader.tokens)) if not Path(cache_loader.metric_cache_paths[token]).is_file()]
    smoke_tokens = [min(tokens) for _, tokens in sorted(scene_loader.get_tokens_list_per_log().items())[:6]]
    # Dataclass/scorer/proposal settings are byte-identical to the preserved v1
    # reproduction; evaluator helper additions and whitespace changes are not.
    previous_navsim = workspace / "reference_repositories/DriveJEPAOfficialEvaluation/navsim_v1"
    critical_files = [
        "navsim/planning/metric_caching/metric_cache.py",
        "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
        "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml",
    ]
    compatibility_files = [{"file": relative_path, "previous_sha256": compute_file_sha256(previous_navsim / relative_path), "official_sha256": compute_file_sha256(navsim_root / relative_path)} for relative_path in critical_files]
    identical_critical_files = all(item["previous_sha256"] == item["official_sha256"] for item in compatibility_files)
    # Actual unpickling under the new official module namespace is mandatory.
    cache_sample = cache_loader.get_from_token(smoke_tokens[0])
    simulator = instantiate(configuration.simulator)
    scorer = instantiate(configuration.scorer)
    assert simulator.proposal_sampling == scorer.proposal_sampling
    cache_report = {
        "critical_files": compatibility_files,
        "critical_files_identical": identical_critical_files,
        "sample_cache_type": str(type(cache_sample)),
        "sample_unpickle_under_official_navsim_succeeded": True,
        "simulator_configuration": OmegaConf.to_container(configuration.simulator, resolve=True),
        "scorer_configuration": OmegaConf.to_container(configuration.scorer, resolve=True),
        "other_source_differences": "Drive fork adds unused fast trajectory/map helpers and index aliases; simulator changes whitespace only. Original official scorer/trajectory conversion retained.",
    }
    assets = {}
    for filename, published in specification["checkpoint_files"].items():
        file_path = workspace / specification["checkpoint_directory"] / filename
        measured = {"path": str(file_path), "bytes": file_path.stat().st_size, "sha256": compute_file_sha256(file_path), "source": published}
        if measured["bytes"] != published["bytes"] or (published.get("sha256") and measured["sha256"] != published["sha256"]):
            raise RuntimeError(f"Downloaded artifact integrity failure: {filename}")
        assets[filename] = measured
    checkpoint_metadata = torch.load(assets["state.pt"]["path"], map_location="cpu", weights_only=False)
    saved_model_config = checkpoint_metadata["config"]["model"]
    preset = OmegaConf.to_container(OmegaConf.load(source_root / specification["official_model_config"]), resolve=True)
    compared_fields = ["camera_names", "num_history_frames", "num_future_frames", "mv_scene_dim", "flow_hidden_dim", "flow_num_layers", "flow_num_heads", "num_inference_steps", "trajectory_horizon", "ego_status_dim", "input_hw", "patch_size", "tubelet_size"]
    configuration_matches = {field: {"preset": preset["model"][field], "checkpoint": saved_model_config[field], "matches": preset["model"][field] == saved_model_config[field]} for field in compared_fields}
    checkpoint_state_report = {
        "step": checkpoint_metadata["step"], "epoch": checkpoint_metadata["epoch"],
        "checkpoint_preset_agreement": configuration_matches,
        "training_best_metric": checkpoint_metadata["config"]["trainer"]["best_metric"],
        "training_validation_split": "navtest according to published config.data.val and trainer.navsim_eval; not independent held-out",
        "public_evaluation_claim": "Official installation documents the same stage-2 artifact for v1 PDMS and v2 EPDMS; raw run-specific scores/10-seed list not verified",
    }
    report = {
        "official_source_commit": specification["official_source_commit"],
        "navsim_source_commit": specification["navsim_source_commit"],
        "expected_scene_count": len(expected_tokens), "loaded_scene_count": len(actual_tokens),
        "missing_scene_tokens": sorted(expected_tokens - actual_tokens),
        "unexpected_scene_tokens": sorted(actual_tokens - expected_tokens),
        "missing_metric_cache_tokens": sorted(actual_tokens - set(cache_loader.tokens)),
        "missing_metric_cache_files": missing_cache_files, "missing_current_camera_files": missing_images,
        "unique_required_current_camera_files": len(required_images),
        "history_input_cadences_microseconds": sorted(observed_cadences),
        "future_camera_files_required_for_inference": False,
        "smoke_scene_tokens_fixed_before_gpu_inference": smoke_tokens,
        "metric_cache_compatibility": cache_report, "checkpoint_metadata": checkpoint_state_report,
    }
    report["complete"] = (identical_critical_files and all(item["matches"] for item in configuration_matches.values()) and not any(report[key] for key in ["missing_scene_tokens", "unexpected_scene_tokens", "missing_metric_cache_tokens", "missing_metric_cache_files", "missing_current_camera_files"]))
    save_json(results_root / "verified_assets.json", assets)
    save_json(results_root / "evaluation_preflight.json", report)
    save_json(results_root / "expected_scene_tokens.json", {"tokens": sorted(expected_tokens)})
    save_json(results_root / "resolved_official_model_config.json", preset)
    print(json.dumps({"complete": report["complete"], "expected_scenes": len(expected_tokens), "smoke_tokens": smoke_tokens}), flush=True)
    if not report["complete"]:
        raise RuntimeError("Preflight incomplete; full evaluation prohibited")


def initialize_official_agent(workspace, specification, source_root):
    import torch
    from eval.navsim_agent import WorldModelNavsimAgent
    from training.checkpoint import _state_dict_key_diff
    import psutil
    free_device_bytes, total_device_bytes = torch.cuda.mem_get_info()
    available_host_bytes = psutil.virtual_memory().available
    safety = specification["memory_safety"]
    save_json(workspace / "results/official_wa_jepa_reproduction" / f"memory_preflight_gpu{os.environ.get('CUDA_VISIBLE_DEVICES', 'unknown')}.json", {
        "free_device_bytes": free_device_bytes, "total_device_bytes": total_device_bytes,
        "host_available_bytes": available_host_bytes, "policy": safety,
    })
    if free_device_bytes < safety["minimum_free_device_gib_before_model_load"] * 2**30 or available_host_bytes < safety["minimum_host_available_gib_before_model_load"] * 2**30:
        raise RuntimeError("Insufficient free device/host memory; no model allocation attempted")
    torch.set_num_threads(1)
    torch.manual_seed(specification["execution_seed"])
    from omegaconf import OmegaConf
    requested_tf32 = bool(OmegaConf.load(source_root / specification["official_model_config"]).get("tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = requested_tf32
    torch.backends.cudnn.allow_tf32 = requested_tf32
    dependency_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    save_json(workspace / "results/official_wa_jepa_reproduction/runtime_environment.json", {
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "pip_freeze": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "pip_check_exit_code": dependency_check.returncode,
        "pip_check_output": dependency_check.stdout + dependency_check.stderr,
        "library_path_process_local": os.environ.get("LD_LIBRARY_PATH"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "tf32_control_provenance": "Harness explicitly applies the pinned YAML tf32 field; bare upstream NAVSIM launcher does not explicitly set these global flags. This is an execution-control difference, not a post-score adjustment.",
        "existing_drive_environment_modified": False,
    })
    agent = WorldModelNavsimAgent(
        config_path=str(source_root / specification["official_model_config"]),
        checkpoint_path=str(workspace / specification["checkpoint_directory"]), device="cuda",
    )
    agent.initialize()
    checkpoint_state = torch.load(str(workspace / specification["checkpoint_directory"] / "model_state_dict.pt"), map_location="cpu", weights_only=False, mmap=True)
    if "state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["state_dict"]
    elif "model_state_dict" in checkpoint_state:
        checkpoint_state = checkpoint_state["model_state_dict"]
    missing, unexpected, mismatches = _state_dict_key_diff(agent.model.state_dict(), checkpoint_state)
    if missing or unexpected or mismatches:
        raise RuntimeError(f"Strict full planning checkpoint mismatch: {missing}, {unexpected}, {mismatches}")
    agent.model.load_state_dict(checkpoint_state, strict=True)
    save_json(workspace / "results/official_wa_jepa_reproduction/strict_loading.json", {
        "missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatches,
        "explicit_pytorch_strict_true_completed": True, "model_state_key_count": len(checkpoint_state),
        "encoder_loading_policy": "Official require_pretrained=true, allow_partial_load=false, min_checkpoint_load_ratio=1.0",
        "model_parameter_dtype_counts": {str(dtype): sum(parameter.numel() for parameter in agent.model.parameters() if parameter.dtype == dtype) for dtype in {parameter.dtype for parameter in agent.model.parameters()}},
    })
    return agent


def evaluate_official_scenes(workspace, specification, source_root, configuration, full_evaluation=False, execution_shard=None):
    import numpy as np
    import torch
    from hydra.utils import instantiate
    import navsim
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.evaluate.pdm_score import pdm_score
    results_root = workspace / "results/official_wa_jepa_reproduction"
    preflight = json.loads((results_root / "evaluation_preflight.json").read_text())
    if not preflight["complete"]:
        raise RuntimeError("Preflight did not pass")
    if full_evaluation:
        smoke = json.loads((results_root / "smoke_results.json").read_text())
        if not smoke["all_scenes_successful"] or smoke["estimated_full_single_gpu_hours"] > specification["full_evaluation_gate"]["max_estimated_single_gpu_hours"]:
            raise RuntimeError("Smoke/cost gate did not pass")
    agent = initialize_official_agent(workspace, specification, source_root)
    scene_loader = create_official_scene_loader(configuration, agent.get_sensor_config())
    cache_loader = MetricCacheLoader(Path(configuration.metric_cache_path))
    simulator = instantiate(configuration.simulator)
    scorer = instantiate(configuration.scorer)
    tokens = sorted(scene_loader.tokens) if full_evaluation else preflight["smoke_scene_tokens_fixed_before_gpu_inference"]
    if execution_shard is not None:
        num_execution_shards = specification["parallel_evaluation"]["execution_shards"]
        if not full_evaluation or not 0 <= execution_shard < num_execution_shards:
            raise ValueError("Execution shard outside registered dense full-evaluation plan")
        tokens = tokens[execution_shard::num_execution_shards]
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    output_root.mkdir(parents=True, exist_ok=True)
    shard_suffix = f"_workers{specification['parallel_evaluation']['execution_shards']}_shard{execution_shard}" if execution_shard is not None else ""
    record_path = output_root / (f"dense_full_scene_records{shard_suffix}.jsonl" if full_evaluation else "dense_smoke_scene_records.jsonl")
    previous_records = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
    if full_evaluation and execution_shard is not None:
        # Preserve/reuse the original4-worker plan's records without copying or
        # overwriting its files. Partition completed scenes under the new plan.
        assigned_tokens = set(tokens)
        for earlier_path in sorted(output_root.glob("dense_full_scene_records_shard*.jsonl")):
            earlier_records = [json.loads(line) for line in earlier_path.read_text().splitlines()]
            previous_records.extend(record for record in earlier_records if record["token"] in assigned_tokens)
    if len({record["token"] for record in previous_records}) != len(previous_records):
        raise RuntimeError("Duplicate saved tokens; do not silently overwrite")
    completed_tokens = {record["token"] for record in previous_records}
    if completed_tokens - set(tokens):
        raise RuntimeError("Saved records contain unexpected tokens")
    records = list(previous_records)
    launch_time = time.perf_counter()
    with record_path.open("a", buffering=1) as record_file:
        for scene_token in tokens:
            if scene_token in completed_tokens:
                continue
            torch.cuda.reset_peak_memory_stats()
            scene_start = time.perf_counter()
            record = {"token": scene_token, "valid": True, "flow_inference_seed": specification["flow_inference_seed"]}
            try:
                agent_input = scene_loader.get_agent_input_from_token(scene_token)
                features = agent.get_feature_builders()[0].compute_features(agent_input)
                torch.cuda.synchronize()
                inference_start = time.perf_counter()
                trajectory = agent.compute_trajectory(agent_input)
                torch.cuda.synchronize()
                inference_seconds = time.perf_counter() - inference_start
                if trajectory.poses.shape != (8, 3) or not np.isfinite(trajectory.poses).all():
                    raise RuntimeError("Invalid trajectory shape or non-finite output")
                scoring_start = time.perf_counter()
                scores = asdict(pdm_score(cache_loader.get_from_token(scene_token), trajectory, simulator.proposal_sampling, simulator, scorer))
                record.update({
                    "feature_shapes": {key: list(value.shape) for key, value in features.items()},
                    "trajectory_poses": trajectory.poses.tolist(),
                    "trajectory_sampling": asdict(trajectory.trajectory_sampling),
                    "inference_steps": agent.model.flow_num_inference_steps,
                    "inference_seconds_including_official_feature_builder": inference_seconds,
                    "scoring_seconds": time.perf_counter() - scoring_start,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "scores_fraction": scores,
                })
            except Exception as error:
                record.update({"valid": False, "error": repr(error), "traceback": traceback.format_exc()})
                print(record["traceback"], flush=True)
            record["scene_total_seconds"] = time.perf_counter() - scene_start
            record_file.write(json.dumps(record, allow_nan=False) + "\n")
            records.append(record)
            if not record["valid"] and ("out of memory" in record.get("error", "").lower()):
                save_json(results_root / f"oom_abort{shard_suffix}.json", record)
                raise RuntimeError("CUDA OOM: scene preserved; this worker stops without retry/config sweep")
            if not full_evaluation or len(records) % 100 == 0:
                print(f"WAJEPA_DENSE_PROGRESS {len(records)}/{len(tokens)} valid={record['valid']} seconds={record['scene_total_seconds']:.3f}", flush=True)
            if not full_evaluation and not record["valid"]:
                break
    valid_records = [record for record in records if record["valid"]]
    aggregate = {key: float(np.mean([record["scores_fraction"][key] for record in valid_records])) for key in valid_records[0]["scores_fraction"]} if valid_records else {}
    report = {
        "purpose": "official_full_navtest_single_preset_seed" if full_evaluation else "normal_operation_and_resource_smoke_not_paper_reproduction_score",
        "execution_shard": execution_shard,
        "expected_scenes": len(tokens), "processed_scenes": len(records), "successful_scenes": len(valid_records),
        "failed_scene_tokens": [record["token"] for record in records if not record["valid"]],
        "missing_scene_tokens": sorted(set(tokens) - {record["token"] for record in records}),
        "all_scenes_successful": len(valid_records) == len(tokens),
        "aggregate_fraction_not_percent": aggregate,
        "estimated_full_single_gpu_hours": float(np.mean([record["scene_total_seconds"] for record in valid_records]) * preflight["expected_scene_count"] / 3600) if valid_records else None,
        "elapsed_this_invocation_seconds_excluding_initialization": time.perf_counter() - launch_time,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "actual_navsim_module": navsim.__file__, "conda_python": sys.executable, "python": platform.python_version(),
        "model_source_commit": specification["official_source_commit"], "official_steps": 12,
        "scene_records": str(record_path), "evaluation_schedule_difference": "Sorted sequential scenes, one initialized official agent/scorer; per-scene incremental persistence. Official generator resets each call so ordering does not change model noise.",
        "paper_comparison_limitation": "Single preset seed versus reported 10-seed paper mean; no numerical reproduction tolerance claimed",
    }
    save_json(results_root / (f"full_navtest_results{shard_suffix}.json" if full_evaluation else "smoke_results.json"), report)
    if full_evaluation:
        csv_path = results_root / f"official_scene_scores{shard_suffix}.csv"
        columns = ["token", "valid"] + list(aggregate)
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for record in records:
                writer.writerow({"token": record["token"], "valid": record["valid"], **record.get("scores_fraction", {})})
        print("WAJEPA_DENSE_FULL_DONE", flush=True)
    else:
        print(json.dumps(report, indent=2), flush=True)


def aggregate_completed_official_evaluation(workspace, specification):
    """Only publish the dense benchmark aggregate when every shard is complete."""
    import numpy as np
    results_root = workspace / "results/official_wa_jepa_reproduction"
    num_workers = specification["parallel_evaluation"]["execution_shards"]
    shard_paths = [results_root / f"full_navtest_results_workers{num_workers}_shard{index}.json" for index in range(num_workers)]
    if not all(path.is_file() for path in shard_paths):
        raise RuntimeError("Full evaluation not complete; aggregate withheld")
    shard_reports = [json.loads(path.read_text()) for path in shard_paths]
    if any(report["missing_scene_tokens"] for report in shard_reports):
        raise RuntimeError("An execution shard has missing scenes")
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    record_paths = sorted(output_root.glob("dense_full_scene_records_shard*.jsonl")) + sorted(output_root.glob(f"dense_full_scene_records_workers{num_workers}_shard*.jsonl"))
    records = [json.loads(line) for path in record_paths for line in path.read_text().splitlines()]
    actual_tokens = [record["token"] for record in records]
    expected_tokens = set(json.loads((results_root / "expected_scene_tokens.json").read_text())["tokens"])
    if len(set(actual_tokens)) != len(actual_tokens) or set(actual_tokens) != expected_tokens:
        raise RuntimeError("Full evaluation has duplicate/missing/extra scene records; aggregate withheld")
    valid_records = [record for record in records if record["valid"]]
    metric_names = list(valid_records[0]["scores_fraction"]) if valid_records else []
    aggregate = {metric: float(np.mean([record["scores_fraction"][metric] for record in valid_records])) for metric in metric_names}
    with (results_root / "official_scene_scores.csv").open("w", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=["token", "valid"] + metric_names)
        writer.writeheader()
        for record in sorted(records, key=lambda item: item["token"]):
            writer.writerow({"token": record["token"], "valid": record["valid"], **record.get("scores_fraction", {})})
    maximum_device_memory_bytes = {}
    telemetry_path = output_root / "memory_guard_telemetry.jsonl"
    if telemetry_path.is_file():
        telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        maximum_device_memory_bytes = {gpu: max(record["gpu_used_bytes"][gpu] for record in telemetry) for gpu in ("0", "1")}
    report = {
        "purpose": "official_dense_full_navtest_single_preset_seed",
        "expected_scenes": len(expected_tokens), "processed_scenes": len(records), "successful_scenes": len(valid_records),
        "failed_scene_tokens": [record["token"] for record in records if not record["valid"]],
        "missing_scene_tokens": [], "extra_scene_tokens": [], "duplicate_scene_tokens": [],
        "aggregate_fraction_not_percent": aggregate,
        "aggregate_denominator": "Successful scenes; failures explicitly listed, not silently omitted",
        "paper_reference_percent": specification["paper_result"],
        "single_seed_pdms_minus_paper_mean_points": aggregate["score"] * 100 - specification["paper_result"]["PDMS"] if valid_records else None,
        "inference_source": specification["official_source_commit"], "navsim_source": specification["navsim_source_commit"],
        "steps": 12, "flow_seed": specification["flow_inference_seed"], "actual_dtype": "float32; official agent has no autocast",
        "paper_comparison_limitations": "Single preset seed vs10-seed published mean; exact reproduction tolerance unknown. Do not compare v1PDMS with v2EPDMS.",
        "execution_shards": num_workers, "reused_completed_four_worker_scene_count": sum(len(path.read_text().splitlines()) for path in output_root.glob("dense_full_scene_records_shard*.jsonl")),
        "maximum_monitored_total_device_used_bytes": maximum_device_memory_bytes,
        "scene_scores_sha256": compute_file_sha256(results_root / "official_scene_scores.csv"),
    }
    save_json(results_root / "full_navtest_results.json", report)
    print("WAJEPA_DENSE_AGGREGATE_DONE", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "smoke", "full", "aggregate", "wait-and-aggregate"])
    parser.add_argument("--execution-shard", type=int)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    arguments = parser.parse_args()
    workspace = arguments.workspace.resolve()
    if arguments.action in ("smoke", "full") and (workspace / "outputs/official_wa_jepa_reproduction/evaluation_pause.json").exists():
        raise RuntimeError("Evaluation is explicitly user-paused; GPU inference is prohibited until the user requests resume and the pause is acknowledged by the launcher.")
    specification, source_root, navsim_root, configuration = configure_official_runtime(workspace)
    if arguments.action in ("aggregate", "wait-and-aggregate"):
        while True:
            try:
                aggregate_completed_official_evaluation(workspace, specification)
                break
            except (RuntimeError, json.JSONDecodeError) as error:
                if arguments.action == "aggregate":
                    raise
                print(f"WAJEPA_AGGREGATE_WAIT {error}", flush=True)
                time.sleep(30)
    elif arguments.action == "preflight":
        run_preflight(workspace, specification, source_root, navsim_root, configuration)
    else:
        evaluate_official_scenes(workspace, specification, source_root, configuration, full_evaluation=arguments.action == "full", execution_shard=arguments.execution_shard)


if __name__ == "__main__":
    main()
