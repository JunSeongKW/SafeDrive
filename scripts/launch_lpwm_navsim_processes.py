"""Start two bounded, independently locked GPU queues with durable PID records."""
import json
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[1]
artifact_root = root / "outputs/lpwm_navsim_adaptation_v1"
process_records = []
for variant, gpu_index in (("raw", 0), ("rotation_stabilized", 1)):
    with (artifact_root / f"{variant}_queue.log").open("a") as log_stream:
        process = subprocess.Popen(["nohup", "bash", "scripts/launch_lpwm_navsim_pilot.sh", variant, str(gpu_index)],
                                   cwd=root, stdin=subprocess.DEVNULL, stdout=log_stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    time.sleep(.5)
    if process.poll() is not None:
        raise RuntimeError(f"{variant} queue exited immediately with {process.returncode}")
    process_records.append({"variant": variant, "gpu": gpu_index, "pid": process.pid})
(artifact_root / "queue_processes.json").write_text(json.dumps(process_records, indent=2) + "\n")
print(json.dumps(process_records), flush=True)
