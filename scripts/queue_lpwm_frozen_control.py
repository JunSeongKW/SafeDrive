"""Append the frozen-LPWM planner control after the active four-method queue."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_frozen_control_protocol import PROJECT_ROOT, verify_control_configuration
from lpwm_partial_protocol import prepare_protocol
from lpwm_48gb_execution import gpu_memory_snapshot
from queue_lpwm_adaptation_methods import process_identity, identity_alive, signal_owned_group

PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
EXTRA_SOURCES = (
    "src/planning_aware_future_prediction/object_centric/lpwm_frozen_control.py",
    "scripts/lpwm_frozen_control_protocol.py", "scripts/train_lpwm_frozen_control.py",
    "scripts/run_lpwm_frozen_control_stage.py", "scripts/audit_lpwm_frozen_control.py",
    "scripts/queue_lpwm_frozen_control.py", "scripts/summarize_lpwm_frozen_control.py",
)


def predecessor_ready(directory):
    if (directory / "queue_failed.json").exists():
        raise RuntimeError("Predecessor failed; frozen control will not start")
    if not (directory / "queue_completion.json").exists():
        return False
    completion = json.loads((directory / "queue_completion.json").read_text())
    comparison = json.loads((directory / "four_method_summary.json").read_text())
    assert completion["complete"] and comparison["complete"]
    assert set(comparison["reports"]) == {
        "partial_output_layers", "attention_lora", "residual_adapter", "full_low_learning_rate"}
    for report in comparison["reports"].values():
        assert not report["engineering_only"] and report["checks"]["complete_registered_epoch"]
    return True


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    root.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": str(PROJECT_ROOT / "src"),
        "PYTORCH_CUDA_ALLOC_CONF": specification["cuda_allocator_configuration"]}
    if arguments.detach:
        with (root / "queue.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env=environment)
        print(json.dumps({"queue_pid": child.pid, "output_directory": str(root)}), flush=True)
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "queue_completion.json").exists():
        return
    if (root / "pause.requested").exists() or (root / "queue_failed.json").exists():
        raise RuntimeError("Resolve saved pause/failure explicitly before restarting")
    predecessor = PROJECT_ROOT / specification["predecessor_queue_directory"]
    prior_registration = json.loads((predecessor / "queue_registration.json").read_text())
    sources = dict(prior_registration["source_sha256"])
    sources.update({name: digest(PROJECT_ROOT / name) for name in EXTRA_SOURCES})
    configurations = dict(prior_registration["configuration_sha256"])
    profiles = [PROJECT_ROOT / relative for relative in specification["profiles"]]
    configurations.update({str(path): digest(path) for path in [arguments.config, *profiles]})
    registration = {"source_sha256": sources, "configuration_sha256": configurations,
        "stage1_checkpoint_sha256": prior_registration["stage1_checkpoint_sha256"],
        "predecessor_registration_sha256": digest(predecessor / "queue_registration.json"),
        "frozen_control": True, "direct_object_gt_auxiliary": False, "maximum_gpu_used_bytes": 48_000_000_000}
    registration_path = root / "queue_registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write_json(registration_path, registration)
    write_json(shared / "queue_registration.json", registration)

    def verify():
        assert all(digest(PROJECT_ROOT / name) == expected for name, expected in sources.items())
        assert all(digest(Path(name)) == expected for name, expected in configurations.items())
        assert digest(PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt") == registration["stage1_checkpoint_sha256"]

    verify()
    cpu_audit = json.loads((shared / "cpu_audit.json").read_text())
    assert cpu_audit["audit_passed"] and cpu_audit["device"] == "cpu"
    assert cpu_audit["configuration_sha256"] == digest(profiles[0])
    assert all(sources[name] == value for name, value in cpu_audit["source_sha256"].items())
    reference_protocol = json.loads((PROJECT_ROOT / "outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/evaluation_protocol.json").read_text())
    for path in profiles:
        config = verify_control_configuration(path)
        protocol = prepare_protocol(path)
        assert protocol["planning_tokens"] == reference_protocol["planning_tokens"]
        assert protocol["world_tokens"] == reference_protocol["world_tokens"]
        assert (PROJECT_ROOT / config["output_directory"] / "stage1_validation_gate.json").exists()
    state = {"queue_pid": os.getpid(), "status": "waiting", "current_stage": "waiting_for_full_training_and_validation",
        "nodes": {}, "created_unix": time.time(), "predecessor_directory": str(predecessor),
        "scope": "One frozen-representation planner control, then same development evaluation and paired report"}
    owned = []

    def save():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)

    def paused():
        return (root / "pause.requested").exists() or (predecessor / "pause.requested").exists()

    def stop_requested(number, _frame):
        raise KeyboardInterrupt(f"Frozen control queue signal {number}")

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def execute(label, command, distributed=False, allow_memory_failure=False, gpu=True):
        sentinel = root / "completed_nodes" / (label + ".json")
        if sentinel.exists():
            record = json.loads(sentinel.read_text())
            assert record["queue_configuration_sha256"] == digest(arguments.config)
            state["nodes"][label] = {"status": "previously_complete", **record}
            save()
            return record
        verify()
        waiting_since = time.monotonic()
        while gpu:
            if paused():
                raise RuntimeError("Frozen control queue explicitly paused")
            current = gpu_memory_snapshot()
            admission = specification["minimum_training_admission_free_gib"] if distributed else specification["minimum_evaluation_admission_free_gib"]
            if all(current[index]["free_gib"] >= admission for index in ((0, 1) if distributed else (0,))):
                break
            state.update(status="waiting", current_stage="waiting_for_gpu:" + label, resources=current)
            save()
            if time.monotonic() - waiting_since > 86400:
                raise RuntimeError("GPU admission timeout; no unsafe launch")
            time.sleep(10)
        if paused():
            raise RuntimeError("Frozen control queue explicitly paused")
        pressure, maximum_used = None, 0
        with (root / (label + ".log")).open("a") as stream:
            child = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**environment, "CUDA_VISIBLE_DEVICES": "0,1" if gpu else ""})
            identity = process_identity(child.pid)
            assert identity is not None and identity["uid"] == os.getuid()
            owned.append(identity)
            state.update(status="running", current_stage=label)
            state["nodes"][label] = {"status": "running", "pid": child.pid, "started_unix": time.time()}
            save()
            last_saved = time.monotonic()
            while child.poll() is None:
                if paused():
                    raise RuntimeError("Frozen control queue explicitly paused")
                if gpu:
                    current = gpu_memory_snapshot()
                    state["resources"] = current
                    active = [current[index] for index in ((0, 1) if distributed else (0,))]
                    maximum_used = max(maximum_used, *(row["used_bytes"] for row in active))
                    if any(row["used_bytes"] > 48_000_000_000 for row in active):
                        if pressure is None:
                            pressure = {"resources": current, "time_unix": time.time()}
                            signal_owned_group(identity)
                        elif time.time() - pressure["time_unix"] > 120:
                            raise RuntimeError("Memory guard could not stop own child")
                if time.monotonic() - last_saved >= 10:
                    save()
                    last_saved = time.monotonic()
                time.sleep(.5)
            returncode = child.wait()
        record = {"returncode": returncode, "memory_guard": pressure, "maximum_sampled_gpu_used_bytes": maximum_used,
            "queue_configuration_sha256": digest(arguments.config), "finished_unix": time.time()}
        passed = returncode == 0 and pressure is None
        state["nodes"][label].update(status="complete" if passed else "failed", **record)
        save()
        if not passed and not allow_memory_failure:
            raise RuntimeError(f"Frozen-control stage failed: {label}: {record}")
        if passed:
            write_json(sentinel, record)
        return record

    try:
        save()
        while not predecessor_ready(predecessor):
            if paused():
                raise RuntimeError("Frozen control paused before predecessor completion")
            previous = json.loads((predecessor / "queue_state.json").read_text())
            if previous["status"] in ("failed", "paused", "superseded"):
                raise RuntimeError("Predecessor requires intervention; no dependent GPU launch")
            identity = process_identity(previous["queue_pid"])
            if identity is None or identity["state"] == "Z":
                raise RuntimeError("Predecessor disappeared without successful completion")
            assert identity["uid"] == os.getuid()
            assert any(argument.endswith("queue_lpwm_measured_card_budget.py") for argument in identity["arguments"])
            state.update(predecessor_stage=previous["current_stage"], predecessor_status=previous["status"])
            save()
            time.sleep(10)
        verify()
        state.update(status="running", current_stage="predecessor_validated")
        save()
        execute("frozen_control_gpu_audit", ["scripts/audit_lpwm_frozen_control.py", "--config", str(profiles[0]),
            "--device", "cuda:0", "--output", str(shared / "gpu_audit.json")])
        records, selected_path = [], None
        for path in profiles:
            config = verify_control_configuration(path)
            batch = config["microbatch_size_per_gpu"]
            label = f"frozen_control_profile_batch{batch}"
            execution = execute(label, ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                "scripts/run_lpwm_frozen_control_stage.py", "--action", "train", "--config", str(path), "--profile", "--resume"],
                distributed=True, allow_memory_failure=True)
            passed = execution["returncode"] == 0 and execution["memory_guard"] is None
            record = {"configuration": str(path), "microbatch_size_per_gpu": batch, "passed": passed, **execution}
            if passed:
                profile_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile"
                summary = json.loads((profile_root / "training_summary.json").read_text())
                assert summary["profile_only"] and summary["completed_updates"] == config["profile_updates"]
                assert summary["frozen_state_unchanged"]
                rows = [json.loads(line) for line in (profile_root / "training_log.jsonl").read_text().splitlines()]
                record.update(training_summary=summary, steady_update_seconds=sum(row["update_seconds"] for row in rows[2:]) / len(rows[2:]))
                selected_path = path
            else:
                failure = (root / (label + ".log")).read_text().lower()
                if not execution["memory_guard"] and not any(term in failure for term in ("out of memory", "outofmemoryerror", "allocated-memory cap", "48gb gpu memory guard")):
                    raise RuntimeError("Non-memory profile failure; smaller batches must not hide it")
            records.append(record)
            write_json(shared / "execution_profiles.json", records)
            if passed:
                break
        if selected_path is None:
            raise RuntimeError("No batch passed the48GB profile")
        config = verify_control_configuration(selected_path)
        selection = {"selected_configuration": str(selected_path), "profiles": records,
            "selection_basis": "User batch8 priority;4/2 fallback only after memory failure, planning16/world8 unchanged."}
        write_json(root / "execution_selection.json", selection)
        write_json(shared / "execution_selection.json", selection)
        condition_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world"
        common = ["--config", str(selected_path), "--condition", "metric_plus_world"]
        execute("frozen_control_evaluation_engineering", ["scripts/run_lpwm_frozen_control_stage.py", "--action", "evaluate",
            *common, "--engineering-checkpoint", str(condition_root / "profile/checkpoint.pt")])
        # Fresh Stage1+seed47 planner; never transfer profile or adapted weights.
        execute("frozen_control_training", ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
            "scripts/run_lpwm_frozen_control_stage.py", "--action", "train", *common, "--resume"], distributed=True)
        training = json.loads((condition_root / "training_summary.json").read_text())
        assert training["completed_updates"] == 4707 and not training["profile_only"] and training["frozen_state_unchanged"]
        assert training["configuration_sha256"] == digest(selected_path)
        execute("frozen_control_evaluation", ["scripts/run_lpwm_frozen_control_stage.py", "--action", "evaluate", *common])
        execute("adaptation_vs_frozen_comparison", ["scripts/summarize_lpwm_frozen_control.py", "--config", str(arguments.config)], gpu=False)
        state.update(status="complete", current_stage="complete")
        save()
        write_json(root / "queue_completion.json", {"complete": True, "finished_unix": time.time()})
    except BaseException as error:
        for identity in reversed(owned):
            with contextlib.suppress(Exception):
                signal_owned_group(identity)
        state.update(status="failed", error=repr(error))
        save()
        write_json(root / "queue_failed.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
