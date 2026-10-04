"""Run a separate one-epoch partial-tuning and validation queue on GPU0/1."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_partial_protocol import prepare_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"


def queue(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    if arguments.detach:
        with (root / "queue.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": "0,1",
                    "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"})
        print(json.dumps({"queue_pid": child.pid, "output_directory": str(root)}), flush=True)
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "queue_completion.json").exists():
        return
    if (root / "pause.requested").exists() or (root / "queue_failed.json").exists():
        raise RuntimeError("Explicitly resolve this queue's pause/failure before resuming")
    inherited = json.loads((PROJECT_ROOT / "outputs/lpwm_object_future_planning_v3/queue_registration.json").read_text())
    sources = dict(inherited["source_sha256"])
    for relative in ["scripts/train_lpwm_partial_planning.py", "scripts/evaluate_lpwm_partial_planning.py",
            "scripts/audit_lpwm_partial_finetuning.py", "scripts/lpwm_partial_protocol.py",
            "scripts/queue_lpwm_partial_planning.py",
            "src/planning_aware_future_prediction/object_centric/lpwm_partial_finetuning.py"]:
        sources[relative] = digest(PROJECT_ROOT / relative)
    registration = {"configuration_sha256": digest(arguments.config), "source_sha256": sources,
        "old_full_training_remains_paused": True, "stage1_checkpoint_sha256":
            digest(PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt")}
    registration_path = root / "queue_registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write_json(registration_path, registration)
    protocol = prepare_protocol(arguments.config)
    write_json(PROJECT_ROOT / specification["shared_results_directory"] / "evaluation_protocol.json", protocol)
    state = {"queue_pid": os.getpid(), "status": "running", "current_stage": "starting", "nodes": {},
        "configuration_sha256": registration["configuration_sha256"], "created_unix": time.time()}
    children = []
    def save():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)
    def stop_requested(number, _frame):
        raise KeyboardInterrupt(f"Partial queue signal {number}")
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def usage():
        rows = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        return {int(values[0]): {"used_bytes": int(values[1]) * 1024**2, "free_gib": int(values[2]) / 1024}
            for values in ([part.strip() for part in row.split(",")] for row in rows.splitlines())}
    def execute(name, command, gpu=True, distributed=False):
        sentinel = root / "completed_nodes" / (name + ".json")
        if sentinel.exists():
            assert json.loads(sentinel.read_text())["configuration_sha256"] == registration["configuration_sha256"]
            state["nodes"][name] = {"status": "previously_complete"}
            return
        assert digest(arguments.config) == registration["configuration_sha256"]
        assert all(digest(PROJECT_ROOT / relative) == expected for relative, expected in sources.items())
        if (root / "pause.requested").exists():
            raise RuntimeError("Partial queue explicitly paused")
        started_wait = time.monotonic()
        while gpu:
            resources = usage()
            required = specification["minimum_training_admission_free_gib"] if distributed else 12
            if all(resources[index]["free_gib"] >= required for index in ((0, 1) if distributed else (0,))):
                break
            state.update(current_stage="waiting_for_gpu:" + name, resources=resources)
            save()
            if time.monotonic() - started_wait > 86400 or (root / "pause.requested").exists():
                raise RuntimeError("Admission paused or exceeded one day")
            time.sleep(10)
        with (root / (name + ".log")).open("a") as stream:
            child = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1" if gpu else "", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            children.append(child)
            state["current_stage"] = name
            state["nodes"][name] = {"status": "running", "pid": child.pid, "started_unix": time.time()}
            save()
            while child.poll() is None:
                state["resources"] = usage()
                save()
                if any(row["used_bytes"] > specification["resource_limits"]["maximum_gpu_used_bytes"] for row in state["resources"].values()):
                    raise RuntimeError("GPU total VRAM limit exceeded; own child will checkpoint and stop")
                if (root / "pause.requested").exists():
                    raise RuntimeError("Partial queue explicitly paused")
                time.sleep(5)
        state["nodes"][name].update(status="complete" if child.returncode == 0 else "failed",
            returncode=child.returncode, finished_unix=time.time())
        save()
        if child.returncode:
            raise RuntimeError(f"Partial queue stage {name} exited {child.returncode}")
        write_json(sentinel, {"configuration_sha256": registration["configuration_sha256"], "finished_unix": time.time()})
    try:
        for condition in specification["conditions"]:
            common = ["--config", str(arguments.config), "--condition", condition]
            execute(condition + "_gradient_audit", ["scripts/audit_lpwm_partial_finetuning.py", *common,
                "--device", "cuda:0", "--checkpoint", str(PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"),
                "--output", str(root / (condition + "_gradient_audit.json"))])
            execute(condition + "_training", ["-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                "--nproc-per-node=2", "scripts/train_lpwm_partial_planning.py", *common, "--resume"], distributed=True)
            # Validation always completes before the next condition begins.
            # A scientific non-improvement is recorded, not converted to success.
            execute(condition + "_evaluation", ["scripts/evaluate_lpwm_partial_planning.py", *common])
        execute("paired_trend_report", ["scripts/evaluate_lpwm_partial_planning.py", "--config", str(arguments.config), "--summarize"], gpu=False)
        state.update(status="complete", current_stage="complete")
        save()
        write_json(root / "queue_completion.json", {"complete": True, "finished_unix": time.time()})
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        state.update(status="paused" if (root / "pause.requested").exists() else "failed", error=repr(error))
        save()
        write_json(root / "queue_failed.json", {"stage": state["current_stage"], "error": repr(error),
            "time_unix": time.time(), "dependent_jobs_blocked": True})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    queue(arguments)
