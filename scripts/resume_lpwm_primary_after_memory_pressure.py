"""Restore the saved primary run after the observed GPU1 memory-pressure stop."""
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from resume_lpwm_drivor_oracle_parallelism import alive, launch, read_json, write_json, digest

ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = TRAINING_ROOT / "navsim_v1"
TRENDS = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
HISTORY = ROOT / "outputs/lpwm_adapter_epoch_extension_v1/primary_recovery_20261007"
PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"


def main():
    HISTORY.mkdir(parents=True, exist_ok=False)
    old_training = read_json(TRAINING / "launch.json")
    old_queue = read_json(TRAINING_ROOT / "queue_launch.json")
    old_monitor = read_json(ROOT / "outputs/lpwm_drivor_optimized_execution_v1/monitor_resume_launch.json")
    assert all(not alive(record["pid"]) for record in (old_training, old_queue, old_monitor))
    assert not (TRAINING_ROOT / "pause.requested").exists() and not (TRAINING / "pause.requested").exists()
    paused = read_json(TRAINING / "paused.json")
    assert paused["completed_updates"] == 4988
    for rank in (0, 1):
        last = json.loads((TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()[-1])
        assert last["completed_updates"] == 4988
        write_json(HISTORY / f"rank{rank}_pressure_update.json", last)
        if rank == 1:
            assert last["card_used_bytes"] > 48_000_000_000
    registration = read_json(ROOT / "outputs/lpwm_drivor_optimized_execution_v1/execution_registration.json")
    for name, expected in registration["sources"].items():
        assert digest(ROOT / name) == expected, name
    preserved = HISTORY / "resume_checkpoint.pt"
    os.link(TRAINING / "latest.pt", preserved)
    state = torch.load(preserved, map_location="cpu", weights_only=False)
    assert state["completed_updates"] == 4988 and len(state["rng_by_rank"]) == 2
    assert len(state["optimizer"]["state"]) == 796
    assert {int(item["step"]) for item in state["optimizer"]["state"].values()} == {4988}
    assert state["scheduler"]["last_epoch"] == 4988
    checksum = hashlib.sha256()
    prefix = "planner.image_backbone.world_model."
    entries = 0
    for name, value in sorted(state["model"].items()):
        if name.startswith(prefix) and "lora_input_projection" not in name and "lora_output_projection" not in name:
            canonical_name = name[len(prefix):].replace(".pretrained_linear.", ".").replace(".pretrained_convolution.", ".")
            checksum.update(canonical_name.encode())
            checksum.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            entries += 1
    assert entries == 1070 and checksum.hexdigest() == state["frozen_native_sha256"]
    write_json(HISTORY / "checkpoint_integrity.json", {
        "completed_updates": 4988, "optimizer_states": 796, "scheduler_step": 4988,
        "rank_rng_states": 2, "native_entries": entries, "native_sha256": checksum.hexdigest(),
        "checkpoint_sha256": digest(preserved), "passed": True,
        "reason": "Rank1 exceeded48decimalGB after an external process entered; rank0's paused.json memory flag is local and false",
    })
    del state
    for source, destination in [(TRAINING / "launch.json", "previous_training_launch.json"),
                                (TRAINING_ROOT / "queue_launch.json", "previous_queue_launch.json"),
                                (TRENDS / "status.json", "previous_monitor_status.json")]:
        shutil.copy2(source, HISTORY / destination)
    shutil.move(str(TRAINING / "paused.json"), HISTORY / "primary_paused_before_resume.json")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    training = launch(old_training["command"], TRAINING / "launch.log", environment, TRAINING / "launch.json")
    queue = launch(old_queue["command"], TRAINING_ROOT / "queue.log", {**environment, "CUDA_VISIBLE_DEVICES": ""}, TRAINING_ROOT / "queue_launch.json")
    monitor = launch([str(PYTHON), "-u", "scripts/monitor_lpwm_particle_trends_every500.py", "--config",
                      "configs/lpwm_drivor_review/particle_trends_every500.json"], TRENDS / "watch.log",
                     {**environment, "CUDA_VISIBLE_DEVICES": "0"}, HISTORY / "monitor_launch.json")
    write_json(HISTORY / "resumed_launches.json", {"training": training, "queue": queue, "monitor": monitor})
    print(json.dumps({"restored_update": 4988, "training_pid": training["pid"], "queue_pid": queue["pid"], "monitor_pid": monitor["pid"]}), flush=True)


if __name__ == "__main__":
    main()
