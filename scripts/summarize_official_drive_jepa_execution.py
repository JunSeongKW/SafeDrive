"""Preserve measured execution costs and verify the two actual Hydra configurations.

Only existing evaluation artifacts are read; no inference or training is repeated.
"""

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

from audit_official_drive_jepa_evaluation import official_configuration, write_result
from omegaconf import OmegaConf


def elapsed_text_to_seconds(elapsed_text):
    elapsed_parts = [float(part) for part in elapsed_text.split(":")]
    if len(elapsed_parts) not in (2, 3):
        raise ValueError(f"Unsupported GNU time elapsed value: {elapsed_text}")
    elapsed_seconds = 0.0
    for elapsed_part in elapsed_parts:
        elapsed_seconds = elapsed_seconds * 60 + elapsed_part
    return elapsed_seconds


def parse_completed_execution_log(log_path):
    log_text = log_path.read_text()
    if "OFFICIAL_DRIVE_JEPA_EVALUATION_DONE" not in log_text:
        raise RuntimeError(f"Incomplete execution log: {log_path}")
    required_fields = {
        "wall_seconds": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
        "cpu_user_seconds": r"User time \(seconds\):\s*([\d.]+)",
        "cpu_system_seconds": r"System time \(seconds\):\s*([\d.]+)",
        "maximum_reported_child_rss_kib_not_sum_of_workers": r"Maximum resident set size \(kbytes\):\s*(\d+)",
        "exit_status": r"Exit status:\s*(\d+)",
    }
    execution_cost = {}
    for field_name, field_pattern in required_fields.items():
        field_match = re.search(field_pattern, log_text)
        if field_match is None:
            raise RuntimeError(f"Missing GNU time field {field_name}: {log_path}")
        field_text = field_match.group(1).strip()
        execution_cost[field_name] = elapsed_text_to_seconds(field_text) if field_name == "wall_seconds" else float(field_text)
    timestamps = re.findall(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", log_text, re.MULTILINE)
    if len(timestamps) != 2 or execution_cost["exit_status"] != 0:
        raise RuntimeError(f"Unexpected completion or timestamp evidence: {log_path}")
    execution_cost.update({
        "started_at": timestamps[0],
        "completed_at": timestamps[1],
        "log_path": str(log_path),
        "log_sha256": hashlib.sha256(log_path.read_bytes()).hexdigest(),
    })
    return execution_cost


def main():
    workspace = Path(__file__).resolve().parents[1]
    result_root = workspace / "results/official_drive_jepa_reproduction"
    specification, _, official_config, _ = official_configuration(workspace)
    full_result = json.loads((result_root / "full_navtest_results.json").read_text())
    if not full_result["evaluation_complete"]:
        raise RuntimeError("Preserve incomplete results, but do not report completed evaluation cost")
    execution_shards = []
    model_and_scoring_keys = ("agent", "proposal_sampling", "simulator", "scorer")
    common_configuration = {
        config_key: OmegaConf.to_container(official_config[config_key], resolve=True)
        for config_key in model_and_scoring_keys
    }
    for shard_index in (0, 1):
        run_root = workspace / f"outputs/official_drive_jepa_reproduction/full_navtest_gpu{shard_index}_v1"
        hydra_path = run_root / "code/hydra/config.yaml"
        actual_config = OmegaConf.load(hydra_path)
        manifest = json.loads((result_root / f"evaluation_shard_{shard_index}.json").read_text())
        for config_key in model_and_scoring_keys:
            if OmegaConf.to_container(actual_config[config_key], resolve=True) != common_configuration[config_key]:
                raise RuntimeError(f"Actual shard {shard_index} differs from pinned {config_key}")
        for path_key in ("navsim_log_path", "sensor_blobs_path", "metric_cache_path"):
            if actual_config[path_key] != official_config[path_key]:
                raise RuntimeError(f"Actual shard {shard_index} differs in {path_key}")
        actual_filter = actual_config.train_test_split.scene_filter
        expected_filter = official_config.train_test_split.scene_filter
        if sorted(actual_filter.log_names) != manifest["log_names"]:
            raise RuntimeError("Actual log shard differs from the precommitted execution manifest")
        for filter_key in expected_filter:
            if filter_key != "log_names" and actual_filter[filter_key] != expected_filter[filter_key]:
                raise RuntimeError(f"Actual shard {shard_index} differs in scene filter {filter_key}")
        if actual_config.worker.max_workers != manifest["worker_count"] or not actual_config.worker.use_process_pool:
            raise RuntimeError("Actual worker configuration differs from the recorded execution")
        execution_cost = parse_completed_execution_log(workspace / f"outputs/official_drive_jepa_reproduction/logs/full_navtest_gpu{shard_index}_v1.log")
        execution_cost.update({
            "physical_gpu": shard_index,
            "workers": manifest["worker_count"],
            "scene_count": manifest["scene_count"],
            "actual_hydra_config_path": str(hydra_path),
            "actual_hydra_config_sha256": hashlib.sha256(hydra_path.read_bytes()).hexdigest(),
            "model_and_scoring_match_pinned_official_configuration": True,
            "execution_log_filter_matches_precommitted_manifest": True,
        })
        execution_shards.append(execution_cost)
    telemetry = json.loads((result_root / "execution_telemetry.json").read_text())
    if not telemetry["both_execution_shards_complete"]:
        raise RuntimeError("GPU telemetry has not observed completion yet")
    first_start = min(datetime.fromisoformat(shard["started_at"]) for shard in execution_shards)
    last_end = max(datetime.fromisoformat(shard["completed_at"]) for shard in execution_shards)
    concurrent_wall_seconds = (last_end - first_start).total_seconds()
    report = {
        "source_commit": specification["official_source_commit"],
        "planning_checkpoint_sha256": specification["planning_checkpoint"]["sha256"],
        "execution_shards": execution_shards,
        "common_actual_model_and_scoring_configuration": common_configuration,
        "started_at": first_start.isoformat(),
        "completed_at": last_end.isoformat(),
        "concurrent_wall_seconds_timestamp_resolution_1s": concurrent_wall_seconds,
        "maximum_per_shard_gnu_time_wall_seconds": max(shard["wall_seconds"] for shard in execution_shards),
        "sum_shard_wall_seconds_divided_by_3600_gpu_allocation_hours_not_device_active_hours": sum(shard["wall_seconds"] for shard in execution_shards) / 3600,
        "successful_scenes_per_concurrent_wall_second": full_result["successful_scenes"] / concurrent_wall_seconds,
        "sampled_peak_device_memory_mib_by_gpu": telemetry["sampled_peak_device_memory_mib_by_gpu"],
        "memory_measurement_limitations": "Telemetry started after launch and is sampled; GPU values include all device allocations, and GNU time RSS is not the sum of concurrent processes.",
        "duration_scope": "Full evaluation launch to completion only; excludes environment preparation, downloads, preflight and smoke.",
        "worker_count_changed_during_run": False,
        "worker_scaling_benchmark_performed": False,
        "shared_original_dataset_modified": False,
        "training_or_finetuning_performed": False,
    }
    write_result(result_root / "execution_cost.json", report)


if __name__ == "__main__":
    main()
