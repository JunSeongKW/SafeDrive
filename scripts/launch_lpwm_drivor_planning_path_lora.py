"""Launch registered full-data training, final-evaluation queue and diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
MONITOR = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"


def launch(command, directory, environment, record_name="launch.json", log_name="launch.log"):
    directory.mkdir(parents=True, exist_ok=True)
    assert not (directory / record_name).exists(), "Check existing launch before starting any replacement"
    with (directory / log_name).open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": list(map(str, command)), "started_unix": time.time()}
    (directory / record_name).write_text(json.dumps(record, indent=2) + "\n")
    return record


def main():
    assert not (OUTPUT / "pause.requested").exists()
    registration = json.loads((OUTPUT / "registration.json").read_text())
    for filename, expected in registration["sources"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    for gpu in (0, 1):
        used = int(subprocess.check_output(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)) * 1024**2
        assert used < 6_000_000_000, "Batch16 profile requires enough whole-card headroom"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    execution = json.loads((OUTPUT / "parallelism_selection.json").read_text())["execution_configuration"]
    training = launch([str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
        "scripts/train_lpwm_drivor_planning_path_lora.py", "--config", "configs/lpwm_drivor_planning_path_lora/navsim_v1.json",
        "--execution", execution], OUTPUT / "navsim_v1", environment)
    queue = launch([str(PYTHON), "-u", "scripts/queue_lpwm_drivor_planning_path_lora.py"], OUTPUT,
        {**environment, "CUDA_VISIBLE_DEVICES": ""}, "queue_launch.json", "queue.log")
    monitor = launch([str(PYTHON), "-u", "scripts/monitor_lpwm_drivor_planning_path_representations.py", "--watch"],
        MONITOR, {**environment, "CUDA_VISIBLE_DEVICES": "0"}, "watch_launch.json", "watch.log")
    print(json.dumps({"training": training, "queue": queue, "monitor": monitor}, indent=2))


if __name__ == "__main__":
    main()
