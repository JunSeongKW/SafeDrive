"""Read-only GPU0/1 telemetry until our two official evaluation completion markers."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    log_paths = [workspace / f"outputs/official_drive_jepa_reproduction/logs/full_navtest_gpu{index}_v1.log" for index in (0, 1)]
    telemetry_path = workspace / "results/official_drive_jepa_reproduction/execution_telemetry.json"
    samples = []
    start_time = time.monotonic()
    while True:
        gpu_query = subprocess.run([
            "nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits",
        ], capture_output=True, text=True, check=True)
        records = []
        for line in gpu_query.stdout.splitlines():
            gpu_index, memory_used, utilization = [int(value.strip()) for value in line.split(",")]
            records.append({"physical_gpu": gpu_index, "device_total_memory_used_mib": memory_used, "gpu_utilization_percent": utilization})
        samples.append({"utc_time": datetime.now(timezone.utc).isoformat(), "gpu_records": records})
        completion = ["OFFICIAL_DRIVE_JEPA_EVALUATION_DONE" in log_path.read_text() for log_path in log_paths]
        report = {
            "scope": "read-only samples of approved GPUs0/1; no process mutations",
            "sampling_started_after_evaluation_launch": True,
            "sampled_peak_not_continuous_cuda_allocator_peak": True,
            "poll_seconds": arguments.poll_seconds,
            "sampling_elapsed_seconds": time.monotonic() - start_time,
            "both_execution_shards_complete": all(completion),
            "sampled_peak_device_memory_mib_by_gpu": {
                str(gpu_index): max(record["device_total_memory_used_mib"] for sample in samples for record in sample["gpu_records"] if record["physical_gpu"] == gpu_index)
                for gpu_index in (0, 1)
            },
            "samples": samples,
        }
        telemetry_path.write_text(json.dumps(report, indent=2) + "\n")
        if all(completion):
            print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
            return
        if time.monotonic() - start_time > 3600:
            raise RuntimeError("Telemetry stopped after one hour without both completion markers; inspect official logs")
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
