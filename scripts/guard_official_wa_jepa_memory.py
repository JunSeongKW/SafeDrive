"""Read-only telemetry; SIGINT only explicitly registered own evaluation workers.

No worker is restarted automatically. The guard never touches foreign processes
or changes precision/model settings. Per-scene files survive a guard stop.
"""
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import psutil


def main():
    workspace = Path(__file__).resolve().parents[1]
    safety = json.loads((workspace / "configs/official_wa_jepa/reproduction_v1.json").read_text())["memory_safety"]
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    manifest_path = output_root / "parallel_worker_manifest.json"
    with (output_root / "memory_guard_telemetry.jsonl").open("a", buffering=1) as telemetry:
        while True:
            if not manifest_path.exists():
                time.sleep(10)
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
                    if shard_index == str(worker["shard"]):
                        verified_workers.append(worker)
                except (psutil.Error, ValueError):
                    continue
            gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines()
            gpu_usage = {int(row.split(",")[0]): int(row.split(",")[1]) * 2**20 for row in gpu_rows}
            available_host_bytes = psutil.virtual_memory().available
            record = {"timestamp_unix": time.time(), "gpu_used_bytes": {gpu: gpu_usage[gpu] for gpu in (0, 1)}, "host_available_bytes": available_host_bytes, "active_registered_workers": len(verified_workers), "stopped_workers": []}
            for gpu in (0, 1):
                candidates = [worker for worker in verified_workers if worker["physical_gpu"] == gpu]
                if candidates and (gpu_usage[gpu] > safety["user_gpu_memory_cap_bytes"] or available_host_bytes < safety["minimum_host_available_gib_before_model_load"] * 2**30):
                    worker = candidates[-1]
                    os.kill(worker["pane_pid"], signal.SIGINT)
                    record["stopped_workers"].append(worker)
                    print(f"MEMORY_GUARD_STOP owned pid={worker['pane_pid']} gpu={gpu}", flush=True)
            telemetry.write(json.dumps(record) + "\n")
            if (workspace / "results/official_wa_jepa_reproduction/full_navtest_results.json").exists():
                print("MEMORY_GUARD_EVALUATION_COMPLETE", flush=True)
                break
            time.sleep(10)


if __name__ == "__main__":
    main()
