"""Profile on GPU1 only during the existing Adapter evaluation on GPU0."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from evaluate_lpwm_full_planning import write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
QUEUE_ROOT = PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/queue"


def gpu_used_bytes():
    raw = subprocess.check_output(["nvidia-smi", "--id=1", "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True)
    return int(raw.strip()) * 1024**2


def main(arguments):
    destination = arguments.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "1", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    state = {"started_unix": time.time(), "supervisor_pid": os.getpid(), "status": "waiting_for_idle_gpu1", "profiles": []}
    write_json(destination / "status.json", state)
    deadline = time.monotonic() + 1800
    while True:
        queue = json.loads((QUEUE_ROOT / "queue_state.json").read_text())
        if queue["status"] != "running" or (QUEUE_ROOT / "pause.requested").exists():
            raise RuntimeError("Registered queue is no longer running")
        if queue["current_stage"] == "residual_adapter_evaluation" and gpu_used_bytes() < 1_000_000_000:
            break
        if queue["current_stage"].startswith("full_low_learning_rate"):
            raise RuntimeError("Missed the idle evaluation window; do not overlap full training")
        if time.monotonic() > deadline:
            raise RuntimeError("Idle GPU wait timed out; no training changed")
        time.sleep(5)
    reference_peak = None
    # First quantify added recomputation cost at the unchanged effective batch.
    for batch_size, recompute in ((8, "original"), (8, "none"), (8, "future_only"), (12, "original"), (16, "original")):
        queue = json.loads((QUEUE_ROOT / "queue_state.json").read_text())
        if queue["current_stage"] != "residual_adapter_evaluation" or gpu_used_bytes() >= 1_000_000_000:
            state["stopped_reason"] = "Idle evaluation window ended; original queue has priority"
            break
        label = f"batch{batch_size}_{recompute}"
        record = {"label": label, "planning_batch_size": batch_size, "recompute_policy": recompute}
        if batch_size > 8 and reference_peak is not None and reference_peak * batch_size / 8 + 2_000_000_000 > 47_000_000_000:
            record.update(status="skipped", reason="Conservative baseline memory extrapolation exceeds47GB profile allowance")
            state["profiles"].append(record)
            write_json(destination / "status.json", state)
            continue
        command = [str(PYTHON), "-u", "scripts/profile_lpwm_available_vram.py", "--config",
            "configs/lpwm_planning/card_budget_measured_v4/full_low_learning_rate_batch8.json",
            "--output", str(destination / (label + ".json")), "--physical-gpu", "1",
            "--batch-size", str(batch_size), "--recompute", recompute]
        with (destination / (label + ".log")).open("w") as stream:
            child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="profiling", current_profile=label, child_pid=child.pid)
            write_json(destination / "status.json", state)
            maximum_used = 0
            try:
                while child.poll() is None:
                    memory = gpu_used_bytes()
                    maximum_used = max(maximum_used, memory)
                    queue = json.loads((QUEUE_ROOT / "queue_state.json").read_text())
                    if memory > 48_000_000_000 or queue["current_stage"] != "residual_adapter_evaluation" or (QUEUE_ROOT / "pause.requested").exists():
                        record["guard_stop"] = {"used_bytes": memory, "queue_stage": queue["current_stage"]}
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=30)
                        break
                    time.sleep(.5)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
            record.update(returncode=child.wait(), maximum_sampled_card_used_bytes=maximum_used)
        report_path = destination / (label + ".json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            record.update(report=report)
        record["status"] = "passed" if record["returncode"] == 0 and "guard_stop" not in record else "failed"
        state["profiles"].append(record)
        write_json(destination / "status.json", state)
        print(json.dumps({key: value for key, value in record.items() if key != "report"}), flush=True)
        if label == "batch8_original":
            if record["status"] != "passed":
                raise RuntimeError("Baseline profile failed; do not attempt larger configurations")
            reference_peak = maximum_used
    state.update(status="complete", finished_unix=time.time(), training_or_queue_changed=False)
    write_json(destination / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        main(arguments)
    except BaseException as error:
        status_path = arguments.output.resolve() / "status.json"
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        state.update(status="failed", error=repr(error), finished_unix=time.time(),
            training_or_queue_changed=False)
        write_json(status_path, state)
        raise
