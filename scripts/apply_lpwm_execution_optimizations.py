"""Checkpoint, compare execution variants, and resume the preserved experiment."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import torch

from resume_lpwm_drivor_oracle_parallelism import alive, digest, launch, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lpwm_drivor_optimized_execution_v1"
TRAINING_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = TRAINING_ROOT / "navsim_v1"
TRENDS = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
PYTHON = ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"
ENVIRONMENT = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def status(stage, **values):
    write_json(OUTPUT / "status.json", {"stage": stage, "updated_unix": time.time(), **values})
    print(json.dumps({"stage": stage, **values}), flush=True)


def verify_old_registration():
    for name in ("registration.json", "oracle8_execution_registration.json"):
        registration = read_json(TRAINING_ROOT / name)
        for filename, checksum in registration["sources"].items():
            assert digest(ROOT / filename) == checksum, filename


def hold_training():
    assert read_json(OUTPUT / "runtime_compatibility.json")["passed"]
    assert read_json(TRENDS / "status.json")["active_update"] is None
    verify_old_registration()
    pause = TRAINING_ROOT / "pause.requested"
    assert not pause.exists() and not (TRAINING / "paused.json").exists()
    old_training = read_json(TRAINING / "launch.json")
    old_queue = read_json(TRAINING_ROOT / "queue_launch.json")
    for description in (old_training, old_queue):
        assert alive(description["pid"])
        assert (Path("/proc") / str(description["pid"])).stat().st_uid == os.getuid()
    assert "scripts/train_lpwm_drivor_planning_path_lora.py" in old_training["command"]
    write_json(OUTPUT / "previous_training_launch.json", old_training)
    write_json(OUTPUT / "previous_queue_launch.json", old_queue)
    shutil.copyfile(TRAINING / "active_execution.json", OUTPUT / "previous_execution.json")
    request = "Engineering runtime optimization checkpoint; preserve all research settings and resume state.\n"
    pause.write_text(request)
    status("checkpoint_requested", training_pid=old_training["pid"])
    deadline = time.monotonic() + 300
    while alive(old_training["pid"]) or alive(old_queue["pid"]) or read_json(TRENDS / "status.json")["stage"] != "paused":
        assert time.monotonic() < deadline, "Checkpoint deadline exceeded; preserve pause for inspection"
        time.sleep(2)
    assert pause.read_text() == request
    checkpoint = OUTPUT / "preserved_resume.pt"
    os.link(TRAINING / "latest.pt", checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    steps = [int(value["step"]) for value in state["optimizer"]["state"].values()]
    assert min(steps) == max(steps) == state["completed_updates"]
    assert len(state["rng_by_rank"]) == 2
    preserved = {"checkpoint_sha256": digest(checkpoint), "completed_updates": state["completed_updates"],
                 "epoch": state["epoch"], "next_update_in_epoch": state["next_update_in_epoch"],
                 "optimizer_state_count": len(steps), "optimizer_steps_min": min(steps),
                 "optimizer_steps_max": max(steps), "scheduler_state": state["scheduler"],
                 "both_rank_rng_states_preserved": True, "native_hash": state["frozen_native_sha256"]}
    del state
    write_json(OUTPUT / "preserved_resume_state.json", preserved)
    status("checkpoint_preserved", completed_updates=preserved["completed_updates"])
    return old_training, request, preserved


def benchmark(label, configuration_name, updates):
    directory = OUTPUT / "benchmarks" / label
    directory.mkdir(parents=True, exist_ok=False)
    execution = "configs/lpwm_drivor_optimized_execution/" + configuration_name + ".json"
    command = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "scripts/train_lpwm_drivor_optimized_execution.py", "--config",
               "configs/lpwm_drivor_planning_path_lora/navsim_v1.json", "--execution", execution,
               "--benchmark-updates", str(updates), "--benchmark-output", str(directory),
               "--resume-checkpoint", str(OUTPUT / "preserved_resume.pt")]
    with (directory / "console.log").open("w") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=ENVIRONMENT, stdout=log, stderr=subprocess.STDOUT)
        status("benchmark_running", condition=label, benchmark_pid=child.pid, expected_updates=updates)
        deadline = time.monotonic() + 1200
        while child.poll() is None:
            if time.monotonic() > deadline:
                child.terminate()
                child.wait(timeout=60)
                raise TimeoutError("Bounded DDP benchmark exceeded twenty minutes")
            time.sleep(2)
    assert child.returncode == 0, f"Benchmark {label} failed: {directory / 'console.log'}"
    passed = read_json(directory / "passed.json")
    assert passed["passed"] and passed["benchmark_updates"] == updates
    rows = [[json.loads(line) for line in (directory / f"rank{rank}_training.jsonl").read_text().splitlines()]
            for rank in (0, 1)]
    assert all(len(rank_rows) == updates for rank_rows in rows)
    durations = [max(rank_rows[index]["seconds_this_update"] for rank_rows in rows) for index in range(3, updates)]
    wall = [max((rank_rows[-1]["elapsed_seconds"] - rank_rows[2]["elapsed_seconds"]) / (updates-3) for rank_rows in rows)]
    for rank_rows in rows:
        assert all(row["card_used_bytes"] < 48_000_000_000 for row in rank_rows)
        assert all(value > 0 for value in rank_rows[0]["gradient_norms"].values())
    result = {"condition": label, "execution_configuration": execution, "updates": updates,
              "median_update_seconds": statistics.median(durations), "mean_update_seconds": statistics.mean(durations),
              "wall_seconds_per_update": wall[0],
              "maximum_card_bytes": max(row["card_used_bytes"] for rank_rows in rows for row in rank_rows),
              "rank_scene_hashes": [[row["rank_scene_tokens_sha256"] for row in rank_rows] for rank_rows in rows],
              "rank_losses": [[row["loss"] for row in rank_rows] for rank_rows in rows],
              "all_gradient_groups_positive": True, "native_hash_preserved": True,
              "resume_updates": passed["resume_updates"], "benchmark_weights_discarded": True}
    write_json(directory / "summary.json", result)
    status("benchmark_complete", **result)
    return result


def seal_execution(selected, measurements, preserved):
    verify_old_registration()
    sources = dict(read_json(TRAINING_ROOT / "oracle8_execution_registration.json")["sources"])
    additions = [ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_optimized_execution.py",
                 ROOT / "scripts/train_lpwm_drivor_optimized_execution.py",
                 ROOT / "scripts/queue_lpwm_drivor_optimized_execution.py",
                 ROOT / "scripts/apply_lpwm_execution_optimizations.py",
                 ROOT / "scripts/benchmark_lpwm_oracle_worker_counts.py"]
    additions += list((ROOT / "configs/lpwm_drivor_optimized_execution").glob("*.json"))
    sources.update({str(path.relative_to(ROOT)): digest(path) for path in additions})
    registration = {"sources": sources, "created_unix": time.time(), "parent_registration": str(TRAINING_ROOT / "registration.json"),
                    "authorization": "User requested applying measured runtime improvements and resuming training.",
                    "selected_execution": selected["execution_configuration"], "preserved_resume": preserved,
                    "matched_official_loss_ddp_benchmarks": measurements,
                    "effective_batch_64_and_research_settings_unchanged": True,
                    "sdpa_training_only_canonical_evaluation_preserved": True,
                    "dropout_probability_preserved_but_random_masks_and_arithmetic_may_differ": True}
    assert not (OUTPUT / "execution_registration.json").exists()
    write_json(OUTPUT / "execution_registration.json", registration)


def resume_training(old_training, request, selected=None):
    pause = TRAINING_ROOT / "pause.requested"
    assert pause.read_text() == request, "A different pause request must remain pending"
    assert digest(TRAINING / "latest.pt") == read_json(OUTPUT / "preserved_resume_state.json")["checkpoint_sha256"]
    shutil.move(str(TRAINING / "paused.json"), str(OUTPUT / "training_paused_before_resume.json"))
    shutil.move(str(pause), str(OUTPUT / "engineering_pause.requested"))
    command = list(old_training["command"])
    if selected:
        command[command.index("scripts/train_lpwm_drivor_planning_path_lora.py")] = "scripts/train_lpwm_drivor_optimized_execution.py"
        command[command.index("--execution") + 1] = selected["execution_configuration"]
        queue_command = [str(PYTHON), "-u", "scripts/queue_lpwm_drivor_optimized_execution.py", "--execution", selected["execution_configuration"]]
    else:
        queue_command = [str(PYTHON), "-u", "scripts/queue_lpwm_drivor_oracle8_execution.py"]
    training = launch(command, TRAINING / "launch.log", ENVIRONMENT, TRAINING / "launch.json")
    queue = launch(queue_command, TRAINING_ROOT / "queue.log", {**ENVIRONMENT, "CUDA_VISIBLE_DEVICES": ""}, TRAINING_ROOT / "queue_launch.json")
    monitor = launch([str(PYTHON), "-u", "scripts/monitor_lpwm_particle_trends_every500.py", "--config",
                      "configs/lpwm_drivor_review/particle_trends_every500.json"], TRENDS / "watch.log",
                     {**ENVIRONMENT, "CUDA_VISIBLE_DEVICES": "0"}, OUTPUT / "monitor_resume_launch.json")
    write_json(OUTPUT / "resumed_launches.json", {"training": training, "queue": queue, "monitor": monitor})
    status("resumed", optimized=bool(selected), training_pid=training["pid"], queue_pid=queue["pid"], monitor_pid=monitor["pid"])


def main(arguments):
    ctypes.CDLL(None).prctl(15, b"kjs-lpwm-speed", 0, 0, 0)
    assert not (OUTPUT / "preserved_resume.pt").exists()
    old_training, request, preserved = hold_training()
    try:
        measurements = []
        conditions = [("original_before", "original_oracle8", arguments.updates),
                      ("oracle16", "batched_checks_oracle16", arguments.updates),
                      ("sdpa_oracle16", "sdpa_batched_checks_oracle16", arguments.updates),
                      ("original_after", "original_oracle8", max(8, arguments.updates//2))]
        for label, configuration_name, updates in conditions:
            result = benchmark(label, configuration_name, updates)
            if measurements:
                for rank in (0, 1):
                    assert result["rank_scene_hashes"][rank] == measurements[0]["rank_scene_hashes"][rank][:updates]
                assert result["resume_updates"] == preserved["completed_updates"]
            measurements.append(result)
            write_json(OUTPUT / "benchmark_comparison.json", {"measurements": measurements})
        baseline = statistics.mean(measurements[index]["median_update_seconds"] for index in (0, 3))
        candidates = measurements[1:3]
        selected = min(candidates, key=lambda result: result["median_update_seconds"])
        assert selected["median_update_seconds"] < baseline * .97, "No measured improvement above three percent"
        # Avoid adding SDPA when the worker-only option is within measurement noise.
        if selected is measurements[2] and selected["median_update_seconds"] >= measurements[1]["median_update_seconds"] * .985:
            selected = measurements[1]
        seal_execution(selected, measurements, preserved)
        write_json(OUTPUT / "selection.json", {"selected": selected, "baseline_median_average_seconds": baseline,
            "time_reduction_percent": 100*(1-selected["median_update_seconds"]/baseline), "benchmark_weights_discarded": True})
        resume_training(old_training, request, selected)
    except Exception as error:
        status("optimization_failed_restoring_original", error=repr(error))
        # Benchmark failures never substitute their weights for the original
        # full checkpoint. Restore the previously running training immediately.
        if (TRAINING_ROOT / "pause.requested").exists():
            resume_training(old_training, request)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=12)
    main(parser.parse_args())
