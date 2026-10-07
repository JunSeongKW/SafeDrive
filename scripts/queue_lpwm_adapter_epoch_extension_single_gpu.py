"""Continue Adapter epochs2/3, yielding GPU0 to the existing particle monitor."""
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


def prepare(configuration_path, specification):
    import torch
    root = PROJECT_ROOT / specification["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / "registration.json").exists(), "Keep an existing registration immutable"
    profile_root = PROJECT_ROOT / specification["memory_profile_directory"]
    profiles = [read_json(profile_root / "profile_batch1_chunked_lpips" / f"rank{rank}.json") for rank in (0, 1)]
    assert all(row["passed"] and row["native_weights_and_buffers_unchanged"] for row in profiles)
    assert read_json(profile_root / "lpips_chunk_gradient_check.json")["gradient_close"]
    source = PROJECT_ROOT / specification["source_checkpoint"]
    preserved = root / "source_epoch01_resume.pt"
    if not preserved.exists():
        os.link(source, preserved)
    source_hash = digest(source)
    assert digest(preserved) == source_hash
    saved = torch.load(preserved, map_location="cpu", weights_only=False)
    assert saved["completed_updates"] == 4707
    assert {int(state["step"]) for state in saved["optimizer"]["state"].values()} == {4707}
    original_config = PROJECT_ROOT / specification["base_configuration"]
    assert digest(original_config) == saved["configuration_sha256"]
    base = read_json(original_config)
    checkpoint_model = torch.load(PROJECT_ROOT / base["output_directory"] / "metric_plus_world/checkpoint.pt", map_location="cpu", weights_only=True)
    assert saved["model"].keys() == checkpoint_model.keys()
    assert all(torch.equal(value, checkpoint_model[name]) for name, value in saved["model"].items())
    native = hashlib.sha256()
    count = 0
    for name, value in saved["model"].items():
        if name.startswith("world_model.") and ".residual_adapter." not in name:
            native.update(name[len("world_model."):].encode())
            native.update(value.detach().cpu().contiguous().numpy().tobytes())
            count += 1
    assert count > 1000
    stage1 = read_json(PROJECT_ROOT / base["stage1_config"])
    source_names = set(read_json(PROJECT_ROOT / "outputs/lpwm_drivor_optimized_execution_v1/execution_registration.json")["sources"])
    source_names.update(str(path.relative_to(PROJECT_ROOT)) for path in (PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric").glob("*.py"))
    source_names.update([
        "scripts/train_lpwm_partial_planning.py", "scripts/evaluate_lpwm_partial_planning.py",
        "scripts/run_lpwm_navsim_posttraining.py", "scripts/train_lpwm_navtrain_distributed.py",
        "scripts/lpwm_partial_protocol.py", "scripts/lpwm_stage2_admission.py",
        "scripts/summarize_lpwm_posttraining.py", "scripts/evaluate_lpwm_full_planning.py",
        "scripts/profile_lpwm_adapter_concurrent_training.py", "scripts/lpwm_adapter_memory_execution.py",
        "scripts/train_lpwm_adapter_epoch_extension.py", "scripts/evaluate_lpwm_adapter_epoch_extension.py",
        "scripts/queue_lpwm_adapter_epoch_extension.py",
        "scripts/train_lpwm_adapter_epoch_extension_single_gpu.py", "scripts/evaluate_lpwm_adapter_epoch_extension_single_gpu.py",
        "scripts/queue_lpwm_adapter_epoch_extension_single_gpu.py", str(configuration_path.relative_to(PROJECT_ROOT)),
        str(original_config.relative_to(PROJECT_ROOT)), base["stage1_config"],
    ])
    registration = {
        "configuration_sha256": digest(configuration_path), "base_configuration_sha256": digest(original_config),
        "source_checkpoint": str(source), "source_checkpoint_sha256": source_hash,
        "stage1_checkpoint_sha256": saved["stage1_checkpoint_sha256"],
        "evaluation_protocol_sha256": digest(PROJECT_ROOT / base["output_directory"] / "evaluation_protocol.json"),
        "teacher_metrics_sha256": digest(PROJECT_ROOT / base["teacher_directory"] / "candidate_metrics.npy"),
        "trajectory_vocabulary_sha256": digest(PROJECT_ROOT / base["teacher_directory"] / "trajectory_vocabulary.npy"),
        "planning_manifest_sha256": digest(PROJECT_ROOT / stage1["output_directory"] / "planning_manifest.json"),
        "frozen_native_sha256": native.hexdigest(), "frozen_native_state_count": count,
        "optimizer_states": len(saved["optimizer"]["state"]),
        "continued_learning_rates": [group["lr"] for group in saved["optimizer"]["param_groups"]],
        "sources": {name: digest(PROJECT_ROOT / name) for name in sorted(source_names)},
        "profiles": profiles, "profile_weights_discarded": True,
        "source_latest_matches_evaluated_checkpoint": True, "created_unix": time.time(),
        "authorization": "User requested additional Adapter Stage2 epochs concurrently on GPUs0/1 if feasible;48decimalGB/card retained",
    }
    write_json(root / "registration.json", registration)
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    write_json(shared / "registration.json", registration)
    for name in ("profile_batch2", "profile_batch2_chunked_lpips", "profile_batch1_chunked_lpips"):
        for rank in (0, 1):
            path = profile_root / name / f"rank{rank}.json"
            if path.exists():
                write_json(shared / name / path.name, read_json(path))
    write_json(shared / "lpips_chunk_gradient_check.json", read_json(profile_root / "lpips_chunk_gradient_check.json"))
    print(json.dumps({"prepared": True, "source_sha256": source_hash, "sources": len(source_names),
                      "native_sha256": native.hexdigest(), "optimizer_states": registration["optimizer_states"]}), flush=True)


def run(arguments):
    specification = read_json(arguments.config)
    root = PROJECT_ROOT / specification["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    if arguments.prepare:
        prepare(arguments.config, specification)
        return
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
    gpu0_uuid = subprocess.check_output(["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
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

    def external_gpu0_processes():
        raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"], text=True)
        unexpected = []
        allowed_commands = (b"train_lpwm_drivor_optimized_execution.py", b"train_lpwm_adapter_epoch_extension_single_gpu.py",
                            b"evaluate_lpwm_adapter_epoch_extension_single_gpu.py", b"monitor_lpwm_drivor_planning_path_representations.py")
        for line in raw.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or fields[1] != gpu0_uuid or not fields[0].isdigit():
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
            cards = (0,)
            allowed = not needs_primary and not external_gpu0_processes() and all(used[index] < specification["secondary_admission_card_bytes"] for index in cards)
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
        with (root / f"{label}.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), "-u", *command], cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            save_state(label, child_pid=child.pid, card_used_bytes=used)
            maximum = dict(used)
            preempted = False
            last_heartbeat = 0.
            while child.poll() is None:
                used = memory()
                maximum = {index: max(maximum[index], used[index]) for index in (0, 1)}
                external = external_gpu0_processes()
                if external or used[0] >= specification["secondary_stop_card_bytes"]:
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
                if not execute(["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1",
                    "scripts/train_lpwm_adapter_epoch_extension_single_gpu.py", "--config", str(arguments.config), "--target-epoch", str(epoch)],
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
