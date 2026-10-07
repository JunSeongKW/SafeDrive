"""Resume historical Adapter batch8 on both GPUs, yielding to primary diagnostics."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            checksum.update(chunk)
    return checksum.hexdigest()



def run(arguments):
    specification = read_json(arguments.config)
    root = PROJECT_ROOT / specification["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    if arguments.detach:
        with (root / "queue.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), "-u", str(Path(__file__).resolve()), "--config", str(arguments.config)],
                cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(root / "launch.json", {"queue_pid": child.pid, "started_unix": time.time()})
        print(json.dumps({"queue_pid": child.pid, "output_directory": str(root)}), flush=True)
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "complete.json").exists():
        return
    registration = read_json(root / "registration.json")
    assert read_json(root / "baseline_replay_check.json")["passed"]
    assert read_json(root / "parallel_profile_report.json")["passed"]
    gpu_uuids = set(subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=uuid", "--format=csv,noheader"], text=True).split())
    stop_requested = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop_requested.append(True))
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    state = {"queue_pid": os.getpid(), "started_unix": time.time(), "stage": "starting"}
    yield_path = root / "yield_for_primary.requested"

    def verify():
        assert digest(arguments.config) == registration["configuration_sha256"]
        assert all(digest(PROJECT_ROOT / name) == expected for name, expected in registration["sources"].items())

    def save_state(stage, **values):
        state.update(stage=stage, heartbeat_unix=time.time(), **values)
        write_json(root / "queue_state.json", state)

    def memory():
        result = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        return {int(parts[0]): int(parts[1]) * 1024**2 for parts in (line.split(",") for line in result.splitlines())}

    def external_gpu_processes():
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        unexpected = []
        allowed_commands = (b"train_lpwm_drivor_optimized_execution.py", b"train_lpwm_adapter_original_batch.py",
                            b"evaluate_lpwm_adapter_epoch_extension_single_gpu.py", b"monitor_lpwm_drivor_planning_path_representations.py")
        for line in raw.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or fields[1] not in gpu_uuids or not fields[0].isdigit():
                continue
            pid = int(fields[0])
            process = Path(f"/proc/{pid}/cmdline")
            try:
                command = process.read_bytes()
            except FileNotFoundError:
                continue
            if not any(allowed in command for allowed in allowed_commands):
                unexpected.append({"pid": pid, "command": command.replace(b"\0", b" ").decode(errors="replace")[:250]})
        return unexpected

    def primary_needs_gpu():
        monitor = read_json(PROJECT_ROOT / specification["primary_monitor_directory"] / "status.json")
        progress = read_json(PROJECT_ROOT / specification["primary_training_directory"] / "progress.json")
        milestone = monitor.get("next_milestone")
        return monitor.get("stage") in ("evaluating", "waiting_for_memory") or (
            milestone is not None and progress["completed_updates"] >= milestone - 2)

    def paused():
        return bool(stop_requested) or (root / "pause.requested").exists()

    def execute(command, label, training):
        verify()
        stable_since = None
        while True:
            if paused():
                save_state("paused")
                return False
            used = memory()
            needs_primary = primary_needs_gpu()
            cards = (0, 1) if training else (0,)
            allowed = not needs_primary and all(used[index] < specification["secondary_admission_card_bytes"] for index in cards)
            if allowed:
                stable_since = stable_since or time.monotonic()
                if time.monotonic() - stable_since >= specification["stable_admission_seconds"]:
                    break
            else:
                stable_since = None
            save_state("waiting_for_primary_or_memory", pending=label, card_used_bytes=used)
            time.sleep(10)
        if yield_path.exists():
            yield_path.unlink()
        admitted_external = external_gpu_processes()
        admitted_external_ids = {record["pid"] for record in admitted_external}
        with (root / f"{label}.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), "-u", *command], cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_state(label, child_pid=child.pid, card_used_bytes=used, admitted_external_processes=admitted_external)
            maximum = dict(used)
            preempted = False
            last_heartbeat = 0.
            while child.poll() is None:
                used = memory()
                maximum = {index: max(maximum[index], used[index]) for index in (0, 1)}
                external = external_gpu_processes()
                new_external = [record for record in external if record["pid"] not in admitted_external_ids]
                if new_external or any(used[index] >= specification["secondary_stop_card_bytes"] for index in (0, 1)):
                    assert os.getpgid(child.pid) == child.pid
                    os.killpg(child.pid, signal.SIGKILL)
                    preempted = True
                    write_json(root / f"resource_preemption_{time.time_ns()}.json", {"child_pid": child.pid,
                        "external_processes": external, "card_used_bytes": used, "resume": "last complete checkpoint; partial update discarded"})
                if primary_needs_gpu() or paused():
                    yield_path.touch(exist_ok=True)
                if paused():
                    (root / "pause.requested").touch(exist_ok=True)
                if time.monotonic() - last_heartbeat >= 5:
                    save_state(label, child_pid=child.pid, card_used_bytes=used, maximum_sampled_card_bytes=maximum,
                               yielding=yield_path.exists())
                    last_heartbeat = time.monotonic()
                time.sleep(.5)
            assert child.returncode == 0 or preempted, f"{label} failed; see{root / (label + '.log')}"
        save_state(label + "_exited", child_pid=None, maximum_sampled_card_bytes=maximum)
        return not paused()

    try:
        for epoch in specification["target_epochs"]:
            while not (root / f"epoch{epoch:02d}_training_complete.json").exists():
                if not execute(["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                    "scripts/train_lpwm_adapter_original_batch.py", "--config", str(arguments.config), "--target-epoch", str(epoch)],
                    f"epoch{epoch:02d}_training", True):
                    save_state("paused")
                    return
            while not (root / f"epoch{epoch:02d}_evaluation/summary.json").exists():
                if not execute(["scripts/evaluate_lpwm_adapter_epoch_extension_single_gpu.py", "--config", str(arguments.config), "--epoch", str(epoch)],
                    f"epoch{epoch:02d}_evaluation", False):
                    save_state("paused")
                    return
            evaluation = read_json(root / f"epoch{epoch:02d}_evaluation/summary.json")
            assert evaluation["complete"] and evaluation["valid_pdms_scenes"] == 1021
            assert evaluation["frozen_native_sha256"] == registration["frozen_native_sha256"]
            save_state("epoch_verified", completed_epoch=epoch, evaluation_means=evaluation["means"],
                       delta_from_epoch1=evaluation["trained_minus_epoch1"])
        summary = {"complete": True, "epochs": {str(epoch): read_json(root / f"epoch{epoch:02d}_evaluation/summary.json")
                   for epoch in specification["target_epochs"]}, "finished_unix": time.time()}
        write_json(root / "complete.json", summary)
        write_json(PROJECT_ROOT / specification["shared_results_directory"] / "complete.json", summary)
        save_state("complete")
    except Exception as error:
        write_json(root / "queue_failed.json", {"error": repr(error), "state": state, "time_unix": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
