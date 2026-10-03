"""Owned-process launcher; GPU admission and stop guards live in the worker."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "smoke", "train", "evaluate", "postflight", "matched_train", "matched_evaluate"))
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_directory = workspace / "outputs/encoder_future_learning_v1"
    if args.mode.startswith("matched_"):
        output_directory = workspace / "outputs/encoder_future_matched_targets_v1"
    output_directory.mkdir(parents=True, exist_ok=True)
    interpreter = "/rhome/junseong/envs/kjs-drive-jepa-extension/bin/python"
    workers = (0, 1) if args.mode in ("train", "matched_train") else (0,)
    for worker_index in workers:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="" if args.mode in ("evaluate", "matched_evaluate") else str(worker_index),
            OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(workspace / "src"))
        command = ["nohup", interpreter, "-u", str(workspace / "scripts/run_encoder_future_learning.py"),
                   args.mode, "--worker-index", str(worker_index), "--worker-count", str(len(workers))]
        if args.mode == "evaluate":
            command = ["nohup", interpreter, "-u", str(workspace / "scripts/evaluate_encoder_future_learning.py"), "--wait-for-training"]
        if args.mode == "postflight":
            command = ["nohup", interpreter, "-u", str(workspace / "scripts/verify_trained_encoder_inference.py"), "--wait-for-training"]
        if args.mode == "matched_train":
            command = ["nohup", interpreter, "-u", str(workspace / "scripts/run_encoder_future_learning.py"),
                       "train", "--worker-index", str(worker_index), "--worker-count", "2",
                       "--configuration", str(workspace / "configs/encoder_future_learning/matched_target_controls_v1.json"),
                       "--output-directory", str(output_directory)]
        if args.mode == "matched_evaluate":
            command = ["nohup", interpreter, "-u", str(workspace / "scripts/evaluate_encoder_future_learning.py"),
                       "--wait-for-training", "--configuration", str(workspace / "configs/encoder_future_learning/matched_target_controls_v1.json"),
                       "--run-directory", str(output_directory), "--share-directory", str(workspace / "results/encoder_future_matched_targets_v1")]
        process_record = output_directory / f"{args.mode}_worker{worker_index}_process.json"
        if process_record.exists():
            previous = json.loads(process_record.read_text())
            process_path = Path(f"/proc/{previous['pid']}/cmdline")
            if process_path.exists() and any(filename in process_path.read_bytes() for filename in (
                    b"encoder_future_learning.py", b"verify_trained_encoder_inference.py")):
                raise RuntimeError("Existing owned worker still running; no duplicate launch")
        log_path = output_directory / f"{args.mode}_worker{worker_index}.log"
        with log_path.open("a") as stream:
            process = subprocess.Popen(command, cwd=workspace, env=environment, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        process_record.write_text(json.dumps({"pid": process.pid, "command": command, "log": str(log_path)}, indent=2) + "\n")
        print(process_record.read_text())


if __name__ == "__main__":
    main()
