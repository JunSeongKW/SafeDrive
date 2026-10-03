"""Launch one bounded LPWM task worker per authorized GPU and automatic CPU PDM."""
import json
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"


def main():
    destination = OUTPUT_ROOT / "processes.json"
    if destination.exists():
        raise FileExistsError("Inspect existing workers before launching again")
    python_path = PROJECT_ROOT / "runtime/environments/lpwm_navsim_adaptation/bin/python"
    official_python = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python"
    jobs = [(f"worker{worker}", [str(python_path), str(PROJECT_ROOT / "scripts/run_lpwm_planning.py"), "--gpu", str(worker), "--worker", str(worker)]) for worker in (0, 1)]
    jobs.append(("pdm", [str(official_python), str(PROJECT_ROOT / "scripts/evaluate_lpwm_planning_pdm.py"), "--wait"]))
    processes = []
    for name, command in jobs:
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}
        if name == "pdm":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        log_path = OUTPUT_ROOT / f"{name}.log"
        with log_path.open("a") as stream:
            process = subprocess.Popen(["nohup", *command], cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append({"name": name, "pid": process.pid, "command": command, "log": str(log_path)})
    destination.write_text(json.dumps(processes, indent=2) + "\n")
    print(json.dumps(processes, indent=2))


if __name__ == "__main__":
    main()
