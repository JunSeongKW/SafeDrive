"""Checkpoint and resume CPU-worker execution without editing sealed sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = OUTPUT / "navsim_v1"
TRENDS = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def alive(process_id):
    process_path = Path("/proc") / str(process_id)
    return process_path.exists() and process_path.joinpath("stat").read_text().split()[2] != "Z"


def launch(command, log_path, environment, record_path):
    with log_path.open("a") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    record = {"pid": process.pid, "command": list(map(str, command)), "started_unix": time.time()}
    write_json(record_path, record)
    return record


def main(arguments):
    history = arguments.output_directory.resolve()
    history.mkdir(parents=True, exist_ok=False)
    old_training = read_json(TRAINING / "launch.json")
    old_queue = read_json(OUTPUT / "queue_launch.json")
    assert alive(old_training["pid"]) and alive(old_queue["pid"])
    assert any(Path(argument).name == "train_lpwm_drivor_planning_path_lora.py"
               for argument in old_training["command"])
    assert read_json(TRENDS / "status.json")["active_update"] is None, "Wait for the active diagnostic to finish"
    pause = OUTPUT / "pause.requested"
    assert not pause.exists() and not (TRAINING / "pause.requested").exists()
    original_registration = read_json(OUTPUT / "parallelism_registration.json")
    for name, expected in original_registration["sources"].items():
        assert digest(ROOT / name) == expected, name
    execution = read_json(ROOT / arguments.execution)
    previous_execution = read_json(TRAINING / "active_execution.json")
    assert execution["overrides"]["microbatch_per_gpu"] == previous_execution["overrides"]["microbatch_per_gpu"] == 16
    assert execution["overrides"]["loader_workers_per_rank"] == 2
    assert execution["effective_batch_unchanged"] == previous_execution["effective_batch"] == 64
    for source, destination in ((TRAINING / "launch.json", history / "previous_training_launch.json"),
                                (OUTPUT / "queue_launch.json", history / "previous_queue_launch.json"),
                                (TRAINING / "active_execution.json", history / "previous_execution.json"),
                                (TRENDS / "status.json", history / "previous_trend_status.json")):
        shutil.copyfile(source, destination)
    progress_before = read_json(TRAINING / "progress.json")
    write_json(history / "progress_before.json", progress_before)
    request_text = "Engineering checkpoint: CPU oracle parallelism; retain effective batch64 and all learning settings.\n"
    pause.write_text(request_text)
    print("Checkpoint requested at an update boundary", flush=True)
    deadline = time.monotonic() + 240
    while alive(old_training["pid"]) or alive(old_queue["pid"]) or read_json(TRENDS / "status.json")["stage"] != "paused":
        assert time.monotonic() < deadline, "Safe stop timed out; leave pause marker and inspect before resuming"
        time.sleep(2)
    assert pause.read_text() == request_text
    checkpoint = TRAINING / "latest.pt"
    os.link(checkpoint, history / "resume_checkpoint.pt")
    state = torch.load(history / "resume_checkpoint.pt", map_location="cpu", weights_only=False)
    completed = int(state["completed_updates"])
    optimizer_steps = [int(value["step"]) for value in state["optimizer"]["state"].values() if "step" in value]
    assert min(optimizer_steps) == max(optimizer_steps) == completed
    assert len(state["rng_by_rank"]) == 2
    preserved = {"completed_updates": completed, "epoch": state["epoch"],
                 "next_update_in_epoch": state["next_update_in_epoch"],
                 "checkpoint_sha256": digest(history / "resume_checkpoint.pt"),
                 "optimizer_steps_min": min(optimizer_steps), "optimizer_steps_max": max(optimizer_steps),
                 "scheduler_state_preserved": True, "both_rank_rng_states_preserved": True,
                 "frozen_native_sha256": state["frozen_native_sha256"],
                 "effective_batch": 64, "microbatch_per_gpu": 16}
    del state
    write_json(history / "preserved_resume_state.json", preserved)
    shutil.move(str(TRAINING / "paused.json"), history / "paused_before_resume.json")
    shutil.move(str(pause), history / "engineering_pause.requested")
    assert not (TRAINING / "pause.requested").exists()
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    command = list(old_training["command"])
    command[command.index("--execution") + 1] = arguments.execution
    training = launch(command, TRAINING / "launch.log", environment, TRAINING / "launch.json")
    queue = launch([str(PYTHON), "-u", arguments.queue_script], OUTPUT / "queue.log",
                   {**environment, "CUDA_VISIBLE_DEVICES": ""}, OUTPUT / "queue_launch.json")
    monitor = launch([str(PYTHON), "-u", "scripts/monitor_lpwm_particle_trends_every500.py", "--config",
                      "configs/lpwm_drivor_review/particle_trends_every500.json"], TRENDS / "watch.log",
                     {**environment, "CUDA_VISIBLE_DEVICES": "0"}, history / "trend_monitor_launch.json")
    write_json(history / "resumed_launches.json", {"training": training, "queue": queue, "monitor": monitor})
    print(json.dumps({"resumed_at": completed, "training_pid": training["pid"],
                      "queue_pid": queue["pid"], "monitor_pid": monitor["pid"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution", required=True)
    parser.add_argument("--queue-script", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    main(parser.parse_args())
