"""Read-only telemetry; SIGINT only explicitly registered own evaluation workers.

No worker is restarted automatically. The guard never touches foreign processes
or changes precision/model settings. Per-scene files survive a guard stop.
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-profile", type=Path)
    arguments = parser.parse_args()
    safety = json.loads((workspace / "configs/official_wa_jepa/reproduction_v1.json").read_text())["memory_safety"]
    shared_safety = json.loads(arguments.safety_profile.read_text()) if arguments.safety_profile else {}
    polling_seconds = shared_safety.get("poll_interval_seconds", 10)
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    manifest_path = output_root / "parallel_worker_manifest.json"
    with (output_root / "memory_guard_telemetry.jsonl").open("a", buffering=1) as telemetry:
        while True:
            if not manifest_path.exists():
                time.sleep(polling_seconds)
                continue
            manifest = json.loads(manifest_path.read_text())
            verified_workers = []
            for worker in manifest["registered_workers"]:
                try:
                    process = psutil.Process(worker["pane_pid"])
                    arguments = process.cmdline()
                    if process.uids().real != os.getuid() or str(workspace / "scripts/evaluate_official_wa_jepa.py") not in arguments:
                        continue
                    shard_index = arguments[arguments.index("--execution-shard") + 1]
                    if shard_index == str(worker["shard"]) and process.environ().get("CUDA_VISIBLE_DEVICES") == str(worker["physical_gpu"]):
                        verified_workers.append(worker)
                except (psutil.Error, ValueError):
                    continue
            gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).splitlines()
            gpu_usage = {int(row.split(",")[0]): int(row.split(",")[1]) * 2**20 for row in gpu_rows}
            gpu_free = {int(row.split(",")[0]): int(row.split(",")[2]) * 2**20 for row in gpu_rows}
            available_host_bytes = psutil.virtual_memory().available
            record = {"timestamp_unix": time.time(), "gpu_used_bytes": {gpu: gpu_usage[gpu] for gpu in (0, 1)}, "gpu_free_bytes": {gpu: gpu_free[gpu] for gpu in (0, 1)}, "host_available_bytes": available_host_bytes, "active_registered_workers": len(verified_workers), "stopped_workers": []}
            for gpu in (0, 1):
                candidates = [worker for worker in verified_workers if worker["physical_gpu"] == gpu]
                minimum_free_bytes = shared_safety.get("minimum_device_free_gib_while_running", 0) * 2**30
                if candidates and (gpu_usage[gpu] > safety["user_gpu_memory_cap_bytes"] or gpu_free[gpu] < minimum_free_bytes or available_host_bytes < safety["minimum_host_available_gib_before_model_load"] * 2**30):
                    worker = candidates[-1]
                    # Stop future queue admission before interrupting a registered worker.
                    (output_root / "memory_pressure_stop.json").write_text(json.dumps({"timestamp_unix": time.time(), "gpu": gpu, "worker": worker, "gpu_free_bytes": gpu_free[gpu], "automatic_resume": False}) + "\n")
                    os.kill(worker["pane_pid"], signal.SIGINT)
                    record["stopped_workers"].append(worker)
                    print(f"MEMORY_GUARD_STOP owned pid={worker['pane_pid']} gpu={gpu}", flush=True)
            telemetry.write(json.dumps(record) + "\n")
            if (workspace / "results/official_wa_jepa_reproduction/full_navtest_results.json").exists():
                print("MEMORY_GUARD_EVALUATION_COMPLETE", flush=True)
                break
            time.sleep(polling_seconds)


if __name__ == "__main__":
    main()
