"""Restore primary microbatch16 after the additional Adapter epochs and evaluations."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from resume_lpwm_drivor_oracle_parallelism import alive, digest, launch, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / "outputs/lpwm_adapter_original_batch_rebalance_v1"
OUTPUT = HISTORY / "restore_primary_after_adapter"
PRIMARY_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
PRIMARY = PRIMARY_ROOT / "navsim_v1"
ADAPTER = ROOT / "outputs/lpwm_adapter_original_batch_shared_v4"
TRENDS = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
SMALL_EXECUTION = "configs/lpwm_adapter_original_batch_rebalance/primary_batch4.json"
ORIGINAL_EXECUTION = "configs/lpwm_drivor_optimized_execution/batched_checks_oracle16.json"
PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"


def status(stage, **values):
    write_json(OUTPUT / "status.json", {"stage": stage, "updated_unix": time.time(), **values})


def card_usage():
    return [int(value) * 1024**2 for value in subprocess.check_output(
        ["nvidia-smi", "--id=0,1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]


def main():
    import fcntl
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = (OUTPUT / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUTPUT / "resumed_launches.json").exists():
        return
    expected = read_json(OUTPUT / "registration.json")
    for name, checksum in expected["sources"].items():
        assert digest(ROOT / name) == checksum, name
    stable_since = None
    while True:
        if (PRIMARY / "training_complete.json").exists():
            status("primary_already_complete")
            return
        active = read_json(PRIMARY / "active_execution.json")
        if active["execution_sha256"] != digest(ROOT / SMALL_EXECUTION):
            status("superseded_by_another_execution")
            return
        if (PRIMARY_ROOT / "pause.requested").exists() or (ADAPTER / "pause.requested").exists():
            status("waiting_for_user_resume")
            stable_since = None
        elif not (ADAPTER / "complete.json").exists():
            status("waiting_for_adapter_training_and_evaluation")
            stable_since = None
        elif max(card_usage()) >= 18_000_000_000 or read_json(TRENDS / "status.json").get("active_update") is not None:
            status("waiting_for_stable_memory_or_diagnostic")
            stable_since = None
        else:
            stable_since = stable_since or time.monotonic()
            if time.monotonic() - stable_since >= 60:
                break
        time.sleep(30)
    summary = read_json(ADAPTER / "complete.json")
    assert summary["complete"] and all(summary["epochs"][str(epoch)]["complete"] for epoch in (2, 3))
    old_training = read_json(PRIMARY / "launch.json")
    old_queue = read_json(PRIMARY_ROOT / "queue_launch.json")
    assert alive(old_training["pid"]) and alive(old_queue["pid"])
    pause = PRIMARY_ROOT / "pause.requested"
    request = "Restore original primary batch16 after completed Adapter training and evaluations.\n"
    assert not pause.exists()
    pause.write_text(request)
    status("checkpoint_requested")
    deadline = time.monotonic() + 600
    while alive(old_training["pid"]) or alive(old_queue["pid"]) or read_json(TRENDS / "status.json").get("stage") != "paused":
        assert time.monotonic() < deadline
        time.sleep(2)
    assert pause.read_text() == request
    assert max(card_usage()) < 6_000_000_000, "Keep checkpoint paused if shared memory changed"
    import torch
    preserved = OUTPUT / "preserved_resume.pt"
    os.link(PRIMARY / "latest.pt", preserved)
    saved = torch.load(preserved, map_location="cpu", weights_only=False)
    completed = saved["completed_updates"]
    assert len(saved["optimizer"]["state"]) == 796 and len(saved["rng_by_rank"]) == 2
    assert {int(value["step"]) for value in saved["optimizer"]["state"].values()} == {completed}
    assert saved["scheduler"]["last_epoch"] == completed
    write_json(OUTPUT / "checkpoint_integrity.json", {"completed_updates": completed,
        "optimizer_states": 796, "scheduler_step": completed, "rank_rng_states": 2,
        "checkpoint_sha256": digest(preserved), "native_sha256": saved["frozen_native_sha256"]})
    del saved
    for name, checksum in expected["sources"].items():
        assert digest(ROOT / name) == checksum, name
    shutil.move(str(PRIMARY / "paused.json"), OUTPUT / "previous_paused.json")
    shutil.move(str(pause), OUTPUT / "completed_transition.requested")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
                   "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    command = old_training["command"]
    command[command.index("--execution") + 1] = ORIGINAL_EXECUTION
    training = launch(command, PRIMARY / "launch.log", environment, PRIMARY / "launch.json")
    queue = launch([str(PYTHON), "-u", "scripts/queue_lpwm_drivor_optimized_execution.py", "--execution", ORIGINAL_EXECUTION],
                   PRIMARY_ROOT / "queue.log", {**environment, "CUDA_VISIBLE_DEVICES": ""}, PRIMARY_ROOT / "queue_launch.json")
    monitor = launch([str(PYTHON), "-u", "scripts/monitor_lpwm_particle_trends_every500.py", "--config", "configs/lpwm_drivor_review/particle_trends_every500.json"],
                     TRENDS / "watch.log", {**environment, "CUDA_VISIBLE_DEVICES": "0"}, OUTPUT / "monitor_launch.json")
    write_json(OUTPUT / "resumed_launches.json", {"training": training, "queue": queue, "monitor": monitor,
        "completed_updates": completed, "microbatch": 16, "accumulation": 2, "effective_batch": 64})
    status("restored_original_primary_batch", completed_updates=completed)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
