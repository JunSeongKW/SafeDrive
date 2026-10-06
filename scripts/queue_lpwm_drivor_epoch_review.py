"""Preserve the exact epoch resume state, then pause for intermediate review.

The registered trainer is unchanged. Its epoch-end model file is written after
the complete optimizer/RNG checkpoint. Opening that checkpoint before requesting
pause preserves the epoch boundary even if one next-epoch update is in flight.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def capture_checkpoint_and_request_pause(source, destination, pause_path, reason):
    """Keep an open inode across the trainer's atomic replacement of latest.pt."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert not destination.exists(), "Preserve existing review checkpoint"
    with source.open("rb") as checkpoint_stream:
        # Exclusive creation respects a separate user pause request.
        with pause_path.open("x") as pause_stream:
            json.dump(reason, pause_stream, indent=2)
            pause_stream.write("\n")
        pending = destination.with_suffix(".pending.pt")
        with pending.open("wb") as snapshot_stream:
            shutil.copyfileobj(checkpoint_stream, snapshot_stream, length=8 * 1024**2)
        pending.replace(destination)
    return digest(destination)


def inspect_resume_state(path, target_epoch, target_update):
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert state["completed_updates"] == target_update, "Missed the exact epoch checkpoint; do not call this an epoch-boundary resume"
    assert state["epoch"] == target_epoch and state["next_update_in_epoch"] == 0
    assert state["total_updates"] == 40350, "Preserve the original25epoch schedule"
    assert all(name in state for name in ("model", "optimizer", "scheduler", "rng_by_rank"))
    assert len(state["rng_by_rank"]) == 2
    optimizer_steps = [int(value["step"]) for value in state["optimizer"]["state"].values() if "step" in value]
    assert optimizer_steps and max(optimizer_steps) == target_update
    result = {"completed_updates": state["completed_updates"], "resume_epoch_zero_based": state["epoch"],
        "next_update_in_epoch": state["next_update_in_epoch"], "total_updates": state["total_updates"],
        "remaining_updates": state["total_updates"] - target_update, "optimizer_steps_min": min(optimizer_steps),
        "optimizer_steps_max": max(optimizer_steps), "optimizer_state_count": len(optimizer_steps),
        "scheduler_last_epoch": state["scheduler"]["last_epoch"], "rank_rng_count": len(state["rng_by_rank"]),
        "configuration_sha256": state["configuration_sha256"], "frozen_native_sha256": state["frozen_native_sha256"]}
    del state
    return result


def process_alive(pid):
    try:
        return (Path("/proc") / str(pid) / "stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main(arguments):
    ctypes.CDLL(None).prctl(15, b"kjs-epoch-review", 0, 0, 0)
    torch.set_num_threads(1)
    configuration = json.loads(arguments.config.read_text())
    output = ROOT / configuration["output_directory"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "control.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    registration = json.loads((output / "registration.json").read_text())
    for name, expected in registration["sources"].items():
        assert digest(ROOT / name) == expected, name
    training = ROOT / configuration["training_directory"]
    target_epoch = configuration["target_epoch"]
    target_update = configuration["target_update"]
    epoch_model = training / f"epoch_{target_epoch:02d}.pt"
    pause_path = training / "pause.requested"
    launch = json.loads((training / "launch.json").read_text())
    assert launch["pid"] == configuration["training_pid"]
    reason = {"owner": "lpwm_drivor_epoch1_review_v1", "user_request": "Review after epoch1 before the remaining24epochs",
        "target_epoch": target_epoch, "target_update": target_update, "requested_utc": datetime.now(timezone.utc).isoformat()}
    while not epoch_model.exists():
        assert process_alive(launch["pid"]), "Training stopped before epoch boundary"
        assert not (training.parent / "pause.requested").exists() and not pause_path.exists(), "Respect existing pause"
        progress = json.loads((training / "progress.json").read_text())
        assert progress["completed_updates"] <= target_update, "Epoch model marker missing beyond requested boundary"
        write_json(output / "status.json", {"stage": "waiting_for_epoch_boundary", "target_update": target_update,
            "training_update": progress["completed_updates"], "updated_unix": time.time()})
        time.sleep(1)
    assert not (training.parent / "pause.requested").exists()
    snapshot = output / "epoch_01_resume.pt"
    checkpoint_hash = capture_checkpoint_and_request_pause(training / "latest.pt", snapshot, pause_path, reason)
    resume = inspect_resume_state(snapshot, target_epoch, target_update)
    write_json(output / "resume_state.json", resume | {"checkpoint": str(snapshot), "checkpoint_sha256": checkpoint_hash})
    waiting_started = time.time()
    while any(process_alive(pid) for pid in (configuration["training_pid"], configuration["queue_pid"], configuration["monitor_pid"])):
        write_json(output / "status.json", {"stage": "waiting_for_workers_to_stop", "checkpoint_updates": target_update,
            "elapsed_wait_seconds": time.time() - waiting_started, "updated_unix": time.time()})
        assert time.time() - waiting_started < 1800, "Pause cleanup timeout; review checkpoint remains safe"
        time.sleep(5)
    paused = json.loads((training / "paused.json").read_text())
    assert not paused["memory_exceeded"], "Training stopped for memory pressure, not just review"
    assert paused["completed_updates"] >= target_update
    # Preserve possible in-flight updates separately. They must not replace the
    # requested epoch1 state when the remaining24epochs are resumed.
    stopped_snapshot = output / "stopped_process_resume.pt"
    assert not stopped_snapshot.exists()
    shutil.copyfile(training / "latest.pt", stopped_snapshot)
    write_json(output / "training_held.json", {"exact_review_update": target_update,
        "stopped_process_update": paused["completed_updates"], "in_flight_updates_to_discard_on_epoch_resume": paused["completed_updates"]-target_update,
        "resume_checkpoint_sha256": checkpoint_hash, "stopped_checkpoint_sha256": digest(stopped_snapshot),
        "original_training_configuration_unchanged": True, "automatic_remaining_epochs_started": False})
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    command = [str(ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"), "-u",
               str(ROOT / "scripts/monitor_lpwm_drivor_representations.py"), "--checkpoint", str(snapshot), "--device", "cuda"]
    write_json(output / "status.json", {"stage": "epoch1_representation_diagnostics", "checkpoint_updates": target_update, "updated_unix": time.time()})
    with (output / "diagnostics.log").open("a") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    assert completed.returncode == 0, "Diagnostics failed; preserve the training hold"
    diagnostic = ROOT / "outputs/lpwm_drivor_representation_monitor_v1" / f"update_{target_update:06d}"
    assert json.loads((diagnostic / "complete.json").read_text())["checkpoint_sha256"] == checkpoint_hash
    for name in ("complete.json", "summary.json", "readouts.json", "intent_sensitivity.json", "interventions.json"):
        shutil.copyfile(diagnostic / name, output / ("epoch1_" + name))
    write_json(output / "status.json", {"stage": "held_for_drivor_comparison_and_review", "checkpoint_updates": target_update,
        "representation_diagnostics_complete": True, "drivor_comparison_complete": False,
        "independent_benchmark_complete": False, "remaining_epochs": 24, "updated_unix": time.time()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        configuration = json.loads(arguments.config.read_text())
        write_json(ROOT / configuration["output_directory"] / "status.json",
                   {"stage": "stopped_on_error", "error": repr(error), "updated_unix": time.time()})
        raise
