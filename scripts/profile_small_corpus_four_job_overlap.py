"""Bounded disposable concurrency measurement; never changes active trainers."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import queue_four_model_small_corpus_overlap_launch_fix as base

DIRECTORY = base.OUTPUT / "scheduling_v5_joint_overlap"


def main():
    base.verify_original_sources()
    audit = json.loads((DIRECTORY / "gradient_equivalence/complete.json").read_text())
    assert audit["passed"]
    destination = DIRECTORY / "profile_four_jobs"
    destination.mkdir(parents=True, exist_ok=True)
    assert not (destination / "launch.json").exists()
    protocol = dict(loss_relative_cap=1e-4, gradient_relative_l2_cap=1e-3,
        maximum_sampled_card_bytes=45_500_000_000, minimum_estimated_speedup=1.05,
        updates=8, profile_weights_discarded=True,
        timing_scope="Short, two-phase estimate: overlap until other jobs finish, then native joint activation storage. Not a guaranteed ETA.")
    base.atomic_json(DIRECTORY / "protocol.json", protocol)
    before = {}
    for kind in ("drivor", "jepa", "lpwm_sequential"):
        rows = base.timing_rows(base.OUTPUT / kind)
        before[kind] = dict(rows=len(rows), update=rows[-1]["completed_updates"],
            median_seconds=statistics.median(row["seconds"] for row in rows[-100:]))
    base.atomic_json(destination / "before.json", before)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="0,1", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        NCCL_P2P_DISABLE="1", PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    command = base.original_queue.distributed_command("train_small_corpus_joint_adaptive_checkpointing.py",
        ["--output", destination, "--micro-batch", "2", "--profile-updates", "8",
         "--native-marker", DIRECTORY / "profile_native_marker_never_created.json"])
    with (destination / "run.log").open("a") as log:
        process = subprocess.Popen([str(value) for value in command], cwd=base.ROOT, env=environment,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        base.atomic_json(destination / "launch.json", dict(pid=process.pid, command=[str(value) for value in command],
            started_unix=time.time(), allocator=environment["PYTORCH_CUDA_ALLOC_CONF"],
            wrapper_sha256=hashlib.sha256((base.ROOT / "scripts/train_small_corpus_joint_adaptive_checkpointing.py").read_bytes()).hexdigest()))
        maximum_bytes = 0
        samples = []
        while process.poll() is None:
            output = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            values = [int(value) * 1024**2 for value in output.splitlines()]
            samples.append(dict(time=time.time(), card_bytes=values))
            maximum_bytes = max(maximum_bytes, *values)
            if maximum_bytes > protocol["maximum_sampled_card_bytes"] or (base.OUTPUT / "pause.requested").exists():
                base.atomic_json(destination / "pause.requested", dict(reason="Profile memory ceiling or user pause"))
            time.sleep(.5)
    base.atomic_json(destination / "memory_samples.json", samples)
    if process.returncode or not (destination / "complete.json").exists():
        base.atomic_json(DIRECTORY / "admission.json", dict(approved=False, reason="Disposable profile failed or paused", maximum_sampled_card_bytes=maximum_bytes))
        return
    rows = base.timing_rows(destination)
    ordinary = [row["seconds"] for row in rows[1:-1]]
    # Last update intentionally exercises twice the usual SSL batch (75/3200).
    joint_seconds = statistics.median(ordinary) * (3125 / 3200) + rows[-1]["seconds"] * (75 / 3200)
    other_speeds = {}
    for kind, initial in before.items():
        recent = base.timing_rows(base.OUTPUT / kind)[initial["rows"]:]
        other_speeds[kind] = dict(updates=len(recent), median_seconds=statistics.median(row["seconds"] for row in recent))
    native = base.timing_rows(base.OUTPUT / "profiles/lpwm_joint_micro2_maxload")
    native_seconds = statistics.median(row["seconds"] for row in native)
    serial_tail = max((3200 - value["update"]) * value["median_seconds"] for value in before.values())
    overlap_tail = max((3200 - before[kind]["update"]) * value["median_seconds"] for kind, value in other_speeds.items())
    overlapped_updates = min(3200, overlap_tail / joint_seconds)
    sequential_seconds = serial_tail + 3200 * native_seconds
    concurrent_seconds = min(overlap_tail, 3200 * joint_seconds) + max(0, 3200 - overlapped_updates) * native_seconds
    speedup = sequential_seconds / concurrent_seconds
    checks = dict(numerical_audit_passed=audit["passed"], maxload_completed=rows[-1]["ssl_presentations"] == 144,
        memory_safe=maximum_bytes <= protocol["maximum_sampled_card_bytes"],
        companions_progressing=all(value["updates"] >= 3 for value in other_speeds.values()),
        projected_completion_faster=speedup >= protocol["minimum_estimated_speedup"])
    base.atomic_json(DIRECTORY / "admission.json", dict(approved=all(checks.values()), checks=checks,
        maximum_sampled_card_bytes=maximum_bytes, joint_overlap_seconds=joint_seconds,
        historical_native_joint_seconds=native_seconds, other_speeds=other_speeds,
        projected_remaining_sequential_hours=sequential_seconds/3600,
        projected_remaining_concurrent_hours=concurrent_seconds/3600,
        estimated_completion_speedup=speedup, limitation=protocol["timing_scope"]))


if __name__ == "__main__":
    main()
