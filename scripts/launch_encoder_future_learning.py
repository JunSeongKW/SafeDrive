"""Owned-process launcher; GPU admission and stop guards live in the worker."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "train", "evaluate"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_directory = workspace / "outputs/encoder_future_learning_v1"
    output_directory.mkdir(parents=True, exist_ok=True)
    interpreter = "/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python"
    workers = (0, 1) if args.mode == "train" else (0,)
    for worker_index in workers:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.mode == "evaluate" else str(worker_index),
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(workspace / "src"))
        command = ["nohup", interpreter, "-u", str(workspace / "scripts/run_encoder_future_learning.py"),
                   args.mode, "--worker-index", str(worker_index), "--worker-count", str(len(workers))]
        if args.mode == "evaluate":
            command = ["nohup", interpreter, "-u", str(workspace / "scripts/evaluate_encoder_future_learning.py"), "--wait-for-training"]
        process_record = output_directory / f"{args.mode}_worker{worker_index}_process.json"
        if process_record.exists():
            previous = json.loads(process_record.read_text())
            process_path = Path(f"/proc/{previous['pid']}/cmdline")
            if process_path.exists() and b"encoder_future_learning.py" in process_path.read_bytes():
                raise RuntimeError("Existing owned worker still running; no duplicate launch")
        log_path = output_directory / f"{args.mode}_worker{worker_index}.log"
        with log_path.open("a") as stream:
            process = subprocess.Popen(command, cwd=workspace, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        process_record.write_text(json.dumps({"pid": process.pid, "command": command, "log": str(log_path)}, indent=2) + "\n")
        print(process_record.read_text())


if __name__ == "__main__":
    main()
