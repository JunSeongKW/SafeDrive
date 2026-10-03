"""Automatically audit and summarize the registered experiment after PDM completes."""
import json
import os
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"


def main():
    started = time.monotonic()
    while not all((OUTPUT_ROOT / name).exists() for name in ("worker0_complete.json", "worker1_complete.json", "pdm_complete.json")):
        if time.monotonic() - started > 25200:
            raise RuntimeError("Registered completion wait exceeded")
        time.sleep(20)
    python = PROJECT_ROOT / "runtime/environments/lpwm_navsim_adaptation/bin/python"
    for script in ("audit_lpwm_planning_execution.py", "summarize_lpwm_planning.py"):
        subprocess.run([str(python), str(PROJECT_ROOT / "scripts" / script)], cwd=PROJECT_ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}, check=True)
    print("LPWM_PLANNING_FINALIZATION_DONE", flush=True)


if __name__ == "__main__":
    main()
