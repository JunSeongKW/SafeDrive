"""Launch the approved frozen representation diagnostic as a durable, isolated queue."""

import json
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = PROJECT_ROOT / "configs/lpwm_navsim_adaptation/object_readout_validation_v1.json"
    configuration = json.loads(config_path.read_text())
    output_root = PROJECT_ROOT / configuration["output_directory"]
    if (output_root / "launch.json").exists():
        raise FileExistsError("Diagnostic was already launched; inspect its state before resuming")
    if not (output_root / "observed_object_manifest.json").exists():
        raise FileNotFoundError("Prepare the observed annotations before launching GPU extraction")
    alias = PROJECT_ROOT / "runtime/environments/kjs-lpwm-object-validation"
    if not alias.exists():
        alias.symlink_to("lpwm_navsim_adaptation", target_is_directory=True)
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[variable] = str(configuration["resources"]["cpu_threads"])
    command = ["nohup", str(alias / "bin/python"), str(PROJECT_ROOT / "scripts/validate_lpwm_object_readouts.py"),
               "--config", str(config_path), "--mode", "queue"]
    with (output_root / "queue.log").open("a") as stream:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    payload = {"pid": process.pid, "command": command, "lpwm_trainable": False,
               "stage2_started": False, "allowed_gpus": configuration["resources"]["allowed_gpus"]}
    (output_root / "launch.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
