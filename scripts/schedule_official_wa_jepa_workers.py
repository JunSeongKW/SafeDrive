"""Queue unchanged evaluation shards with bounded shared-GPU concurrency.

No model/scorer/seed changes, failed-worker retries, or foreign-process signals.
The existing memory guard must be running before this CPU scheduler is started.
"""
import argparse
import datetime
import hashlib
import json
import runpy
import subprocess
import sys
import time
from pathlib import Path

import psutil


def pending_shards_by_gpu(expected_tokens, completed_tokens, physical_gpu_by_shard):
    """Keep sorted-token modulo partitioning, including older preserved rows."""
    if completed_tokens - set(expected_tokens):
        raise RuntimeError("Saved records include unexpected scenes")
    pending = {0: [], 1: []}
    for shard_index, physical_gpu in enumerate(physical_gpu_by_shard):
        assigned_tokens = set(expected_tokens[shard_index::len(physical_gpu_by_shard)])
        if assigned_tokens - completed_tokens:
            pending[physical_gpu].append(shard_index)
    return pending


def has_launch_capacity(free_device_bytes, host_available_bytes, worker_usage_bytes, safety, physical_gpu=None):
    worker_limit = safety.get("maximum_concurrent_workers_by_gpu", {}).get(str(physical_gpu), safety["maximum_concurrent_workers_per_gpu"])
    if len(worker_usage_bytes) >= worker_limit:
        return False
    # Reserve the remaining peak budget for workers still loading their models.
    peak_worker_bytes = safety["estimated_peak_worker_device_gib"] * 2**30
    startup_reservation = sum(max(0, peak_worker_bytes - usage) for usage in worker_usage_bytes)
    return (
        free_device_bytes - startup_reservation >= safety["minimum_free_device_gib_before_launch"] * 2**30
        and host_available_bytes >= safety["minimum_host_available_gib_before_launch"] * 2**30
    )


def registered_live_workers(workspace, manifest):
    expected_script = str(workspace / "scripts/evaluate_official_wa_jepa.py")
    live_workers = {}
    for worker in manifest["registered_workers"]:
        try:
            process = psutil.Process(worker["pane_pid"])
            process_arguments = process.cmdline()
            if (
                process.uids().real == psutil.Process().uids().real
                and expected_script in process_arguments
                and process_arguments[process_arguments.index("--execution-shard") + 1] == str(worker["shard"])
                and process.environ().get("CUDA_VISIBLE_DEVICES") == str(worker["physical_gpu"])
            ):
                live_workers[process.pid] = worker
        except (psutil.Error, ValueError):
            continue
    return list(live_workers.values())


def gpu_device_and_process_memory():
    device_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
    free_bytes = {int(row.split(",")[0]): int(row.split(",")[1]) * 2**20 for row in device_rows}
    process_rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"], text=True).splitlines()
    process_usage_bytes = {}
    for row in process_rows:
        process_pid, memory_mib = [value.strip() for value in row.split(",")]
        if memory_mib.isdigit():
            process_usage_bytes[int(process_pid)] = int(memory_mib) * 2**20
    return free_bytes, process_usage_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-profile", type=Path, required=True)
    parser.add_argument("--resume-user-paused", action="store_true", help="Only after an explicit user request and renewed GPU availability check")
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    safety = json.loads(arguments.safety_profile.read_text())
    specification = json.loads((workspace / safety["evaluation_configuration"]).read_text())
    assignments = specification["parallel_evaluation"]["physical_gpu_by_shard"]
    if len(assignments) != safety["execution_shards"] or set(assignments) != set(safety["physical_gpus"]):
        raise RuntimeError("Scheduling profile would change the registered evaluation partition")
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    pause_marker = output_root / "evaluation_pause.json"
    if pause_marker.exists():
        if not arguments.resume_user_paused:
            raise RuntimeError("Explicitly acknowledge user pause only after verifying GPU allocation")
        pause_summary = json.loads((workspace / "results/official_wa_jepa_reproduction/paused_evaluation_state.json").read_text())
        evaluation_configuration = workspace / safety["evaluation_configuration"]
        if hashlib.sha256(evaluation_configuration.read_bytes()).hexdigest() != pause_summary["configuration_sha256"]:
            raise RuntimeError("Official evaluation configuration changed since the pause")
        for record_name, expected_sha256 in pause_summary["scene_record_sha256_by_file"].items():
            if hashlib.sha256((output_root / record_name).read_bytes()).hexdigest() != expected_sha256:
                raise RuntimeError(f"Saved scene file differs from the pause inventory: {record_name}")
    pressure_marker = output_root / "memory_pressure_stop.json"
    if pressure_marker.exists():
        raise RuntimeError("Memory-pressure stop requires review, not automatic retry")
    expected_tokens = json.loads((workspace / "results/official_wa_jepa_reproduction/expected_scene_tokens.json").read_text())["tokens"]
    completed_rows = [json.loads(line) for record_path in output_root.glob("dense_full_scene_records*.jsonl") for line in record_path.read_text().splitlines()]
    completed_tokens = {record["token"] for record in completed_rows}
    if len(completed_tokens) != len(completed_rows) or any(not record["valid"] for record in completed_rows):
        raise RuntimeError("Duplicate or failed saved records require review before resume")
    pending = pending_shards_by_gpu(expected_tokens, completed_tokens, assignments)
    if pause_marker.exists():
        acknowledge = runpy.run_path(str(workspace / "scripts/launch_official_wa_jepa_workers.py"))["acknowledge_user_pause_for_explicit_resume"]
        acknowledge(pause_marker, True)
    manifest_path = output_root / "parallel_worker_manifest.json"
    # CPU-only scheduler restarts adopt live workers; never duplicate their shards.
    initial_live_workers = registered_live_workers(workspace, json.loads(manifest_path.read_text()))
    for physical_gpu in safety["physical_gpus"]:
        limit = safety.get("maximum_concurrent_workers_by_gpu", {}).get(str(physical_gpu), safety["maximum_concurrent_workers_per_gpu"])
        if sum(worker["physical_gpu"] == physical_gpu for worker in initial_live_workers) > limit:
            raise RuntimeError("Existing workers exceed the new limit; do not stop them implicitly")
    launch_in_progress = None
    launch_started_at = None
    launched_shards = {worker["shard"] for worker in initial_live_workers}
    pending = {physical_gpu: [shard_index for shard_index in shards if shard_index not in launched_shards] for physical_gpu, shards in pending.items()}
    status_path = output_root / "bounded_scheduler_status.json"
    while True:
        if pause_marker.exists() or pressure_marker.exists():
            status_path.write_text(json.dumps({"status": "stopped_by_pause_or_memory_pressure", "pending_shards": pending, "automatic_retry": False}) + "\n")
            print("WAJEPA_QUEUE_STOP pause or memory-pressure marker; no automatic retry", flush=True)
            break
        manifest = json.loads(manifest_path.read_text())
        live_workers = registered_live_workers(workspace, manifest)
        active_shards = {worker["shard"] for worker in live_workers}
        for shard_index in launched_shards - active_shards:
            report_path = workspace / f"results/official_wa_jepa_reproduction/full_navtest_results_workers{len(assignments)}_shard{shard_index}.json"
            if not (report_path.exists() and json.loads(report_path.read_text()).get("all_scenes_successful")) and launch_in_progress != shard_index:
                raise RuntimeError(f"Shard {shard_index} exited without a successful report; no automatic retry")
        if not any(pending.values()) and not live_workers and launch_in_progress is None:
            status_path.write_text(json.dumps({"status": "completed_bounded_shared_gpu_evaluation", "active_workers": [], "pending_shards": pending}) + "\n")
            print("WAJEPA_QUEUE_COMPLETE all launched shards succeeded", flush=True)
            break
        if subprocess.run(["tmux", "-L", "planning-aware-wa-jepa", "has-session", "-t", "official_wa_jepa_memory_guard"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode:
            raise RuntimeError("Memory guard is not running; no additional worker admission")
        free_bytes, process_usage_bytes = gpu_device_and_process_memory()
        if launch_in_progress is not None:
            if launch_in_progress in active_shards:
                launch_in_progress = None
            else:
                # tmux pane exec is briefly visible as env/bash before Python.
                if time.monotonic() - launch_started_at > 30:
                    raise RuntimeError(f"Shard {launch_in_progress} did not become a verified worker; no retry")
                time.sleep(safety["poll_interval_seconds"])
                continue
        status = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "running_bounded_shared_gpu_evaluation",
            "maximum_concurrent_workers_per_gpu": safety["maximum_concurrent_workers_per_gpu"],
            "maximum_concurrent_workers_by_gpu": safety.get("maximum_concurrent_workers_by_gpu", {}),
            "active_workers": live_workers, "pending_shards": pending,
            "gpu_free_bytes": {str(gpu): free_bytes[gpu] for gpu in safety["physical_gpus"]},
            "automatic_failed_worker_retry": False,
        }
        if not any(pending.values()) and not live_workers:
            status["status"] = "completed_bounded_shared_gpu_evaluation"
        staged_status = status_path.with_suffix(".pending")
        staged_status.write_text(json.dumps(status, indent=2) + "\n")
        staged_status.replace(status_path)
        if not any(pending.values()) and not live_workers:
            print("WAJEPA_QUEUE_COMPLETE all launched shards succeeded", flush=True)
            break
        for physical_gpu in safety["physical_gpus"]:
            if not pending[physical_gpu]:
                continue
            workers_on_gpu = [worker for worker in live_workers if worker["physical_gpu"] == physical_gpu]
            usages = [process_usage_bytes.get(worker["pane_pid"], 0) for worker in workers_on_gpu]
            if not has_launch_capacity(free_bytes[physical_gpu], psutil.virtual_memory().available, usages, safety, physical_gpu):
                continue
            shard_index = pending[physical_gpu][0]
            if shard_index in active_shards:
                raise RuntimeError("Pending shard already has an active worker")
            subprocess.run([sys.executable, str(workspace / "scripts/launch_official_wa_jepa_workers.py"), "--gpu", str(physical_gpu), "--execution-shard", str(shard_index)], check=True)
            pending[physical_gpu].pop(0)
            launched_shards.add(shard_index)
            launch_in_progress = shard_index
            launch_started_at = time.monotonic()
            print(f"WAJEPA_QUEUE_STARTED gpu={physical_gpu} shard={shard_index}", flush=True)
            break  # One launch per poll; reserve for incomplete model allocations.
        time.sleep(safety["poll_interval_seconds"])


if __name__ == "__main__":
    main()
