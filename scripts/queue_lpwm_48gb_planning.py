"""Checkpoint-preserving execution amendment and four-method training queue."""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_partial_protocol import prepare_protocol
from lpwm_48gb_execution import (choose_profile, fork_execution_checkpoint,
    gpu_memory_snapshot, verify_execution_configuration)
from queue_lpwm_adaptation_methods import process_identity, identity_alive, signal_owned_group

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
STAGE1 = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"
OLD_PARTIAL = PROJECT_ROOT / "outputs/lpwm_partial_planning_v1"


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    root.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
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
        raise RuntimeError("Explicitly resolve saved pause/failure before resuming")
    registration = json.loads((PROJECT_ROOT / "outputs/lpwm_four_method_queue_v1/queue_registration.json").read_text())
    for relative in ("scripts/lpwm_48gb_execution.py", "scripts/run_lpwm_48gb_stage.py",
            "scripts/queue_lpwm_48gb_planning.py", "scripts/summarize_lpwm_48gb_methods.py"):
        registration["source_sha256"][relative] = digest(PROJECT_ROOT / relative)
    configurations = [PROJECT_ROOT / relative for method in specification["methods"] for relative in method["profiles"]]
    for path in [arguments.config, *configurations]:
        registration["configuration_sha256"][str(path)] = digest(path)
    registration.update(maximum_gpu_used_bytes=48_000_000_000, minimum_free_gib=3,
        method_order=specification["method_order"], object_gt_deferred=True)
    registration_path = root / "queue_registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write_json(registration_path, registration)
    write_json(shared / "queue_registration.json", registration)

    def verify():
        assert all(digest(PROJECT_ROOT / name) == value for name, value in registration["source_sha256"].items())
        assert all(digest(Path(name)) == value for name, value in registration["configuration_sha256"].items())
        assert digest(STAGE1) == registration["stage1_checkpoint_sha256"]

    verify()
    original_protocol = json.loads((OLD_PARTIAL / "evaluation_protocol.json").read_text())
    for path in configurations:
        config = json.loads(path.read_text())
        verify_execution_configuration(config, PROJECT_ROOT)
        protocol = prepare_protocol(path)
        assert protocol["planning_tokens"] == original_protocol["planning_tokens"]
        assert protocol["world_tokens"] == original_protocol["world_tokens"]
        shutil.copy2(OLD_PARTIAL / "stage1_validation_gate.json",
            PROJECT_ROOT / config["output_directory"] / "stage1_validation_gate.json")
    state = {"queue_pid": os.getpid(), "status": "running", "current_stage": "preparing_handoff",
        "nodes": {}, "method_order": specification["method_order"], "created_unix": time.time()}
    owned = []

    def save():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)

    def paused():
        directories = [root, OLD_PARTIAL, PROJECT_ROOT / "outputs/lpwm_four_method_queue_v1",
            PROJECT_ROOT / "outputs/lpwm_adaptation_method_comparison_v1"]
        directories.extend(PROJECT_ROOT / json.loads(path.read_text())["output_directory"] for path in configurations)
        return any((directory / "pause.requested").exists() for directory in directories)

    def stop_requested(number, _frame):
        raise KeyboardInterrupt(f"48GB queue signal {number}")

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def execute(label, command, distributed=False, allow_memory_failure=False, gpu=True):
        sentinel = root / "completed_nodes" / (label + ".json")
        if sentinel.exists():
            record = json.loads(sentinel.read_text())
            assert record["queue_configuration_sha256"] == digest(arguments.config)
            return record
        verify()
        waiting_since = time.monotonic()
        while gpu:
            if paused():
                raise RuntimeError("Queue explicitly paused")
            current = gpu_memory_snapshot()
            admission = specification["minimum_training_admission_free_gib"] if distributed else specification["minimum_evaluation_admission_free_gib"]
            if all(current[index]["free_gib"] >= admission for index in ((0, 1) if distributed else (0,))):
                break
            state.update(current_stage="waiting_for_gpu:" + label, resources=current)
            save()
            if time.monotonic() - waiting_since > 86400:
                raise RuntimeError("GPU admission timed out after one day")
            time.sleep(10)
        if paused():
            raise RuntimeError("Queue explicitly paused")
        minimum_free, maximum_used, pressure = float("inf"), 0, None
        with (root / (label + ".log")).open("a") as stream:
            child = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**environment, "CUDA_VISIBLE_DEVICES": "0,1" if gpu else ""})
            identity = process_identity(child.pid)
            assert identity is not None and identity["uid"] == os.getuid()
            owned.append(identity)
            state.update(current_stage=label)
            state["nodes"][label] = {"status": "running", "pid": child.pid, "started_unix": time.time()}
            save()
            last_saved = time.monotonic()
            while child.poll() is None:
                if paused():
                    raise RuntimeError("Queue explicitly paused")
                if gpu:
                    current = gpu_memory_snapshot()
                    state["resources"] = current
                    active_rows = [current[index] for index in ((0, 1) if distributed else (0,))]
                    minimum_free = min(minimum_free, *(row["free_gib"] for row in active_rows))
                    maximum_used = max(maximum_used, *(row["used_bytes"] for row in active_rows))
                    if any(row["used_bytes"] > 48_000_000_000 or row["free_gib"] < 3 for row in active_rows):
                        if pressure is None:
                            pressure = {"resources": current, "time_unix": time.time()}
                            signal_owned_group(identity)
                        elif time.time() - pressure["time_unix"] > 120:
                            raise RuntimeError("Memory guard could not stop owned child within120s")
                if time.monotonic() - last_saved >= 10:
                    save()
                    last_saved = time.monotonic()
                time.sleep(.5 if allow_memory_failure else 2)
            returncode = child.wait()
        record = {"returncode": returncode, "memory_guard": pressure, "maximum_sampled_gpu_used_bytes": maximum_used,
            "minimum_free_gib": minimum_free if gpu else None,
            "minimum_cap_headroom_gib": (48_000_000_000 - maximum_used) / 1024**3 if gpu else None,
            "queue_configuration_sha256": digest(arguments.config), "finished_unix": time.time()}
        passed = returncode == 0 and pressure is None
        state["nodes"][label].update(status="complete" if passed else "failed", **record)
        save()
        if not passed and not allow_memory_failure:
            raise RuntimeError(f"Execution failed: {label}: {record}")
        if passed:
            write_json(sentinel, record)
        return record

    def stop_previous_execution():
        handoff_path = root / "execution_handoff.json"
        if handoff_path.exists():
            return json.loads(handoff_path.read_text())
        supervisor_specs = [(specification["old_followup_queue_pid"], "queue_lpwm_followup_methods.py",
            "outputs/lpwm_four_method_queue_v1"), (specification["old_comparison_queue_pid"],
            "queue_lpwm_adaptation_methods.py", "outputs/lpwm_adaptation_method_comparison_v1")]
        training = process_identity(specification["old_training_pid"])
        assert training and training["uid"] == os.getuid() and "scripts/train_lpwm_partial_planning.py" in training["arguments"]
        workers = [process_identity(pid) for pid in specification["old_worker_pids"]]
        assert all(worker and worker["uid"] == os.getuid() and worker["parent_pid"] == training["pid"]
            and "scripts/train_lpwm_partial_planning.py" in worker["arguments"] for worker in workers)
        supervisors = []
        for pid, expected, directory in supervisor_specs:
            identity = process_identity(pid)
            assert identity and identity["uid"] == os.getuid() and any(argument.endswith(expected) for argument in identity["arguments"])
            supervisors.append((identity, directory))
        record = {"reason": "User requested48GB VRAM cap and batch increase", "training": training,
            "workers": workers, "supervisors": supervisors, "new_queue_pid": os.getpid(), "time_unix": time.time(),
            "strategy": "Stop only owned CPU supervisors with SIGKILL, then SIGINT each training rank for an atomic model+AdamW checkpoint"}
        write_json(root / "execution_handoff_started.json", record)
        # Their SIGTERM handlers would also stop adopted training; kill CPU supervisors only.
        for identity, directory in supervisors:
            assert identity_alive(identity)
            os.kill(identity["pid"], signal.SIGKILL)
            old_root = PROJECT_ROOT / directory
            old_state = json.loads((old_root / "queue_state.json").read_text())
            write_json(old_root / "queue_superseded_48gb.json", {**record, "previous_state": old_state})
            write_json(old_root / "queue_state.json", {**old_state, "status": "superseded", "new_queue_pid": os.getpid(),
                "new_queue_directory": str(root)})
        for worker in workers:
            assert identity_alive(worker)
            os.kill(worker["pid"], signal.SIGINT)
        stopping_since = time.monotonic()
        while identity_alive(training) or any(identity_alive(worker) for worker in workers):
            state.update(current_stage="checkpointing_previous_partial")
            save()
            if time.monotonic() - stopping_since > 180:
                raise RuntimeError("Graceful training stop exceeded180s; do not discard checkpoint")
            time.sleep(1)
        import torch
        source = OLD_PARTIAL / "metric_plus_world/latest.pt"
        saved = torch.load(source, map_location="cpu", weights_only=False)
        assert saved["reason"] == "signal"
        assert saved["configuration_sha256"] == digest(PROJECT_ROOT / "configs/lpwm_planning/partial_output_layers_v1.json")
        assert {int(value["step"]) for value in saved["optimizer"]["state"].values()} == {saved["completed_updates"]}
        record.update(completed_updates=saved["completed_updates"], optimizer_states=len(saved["optimizer"]["state"]),
            checkpoint_sha256=digest(source))
        del saved
        destination = root / "partial_resume_source.pt"
        shutil.copy2(source, destination)
        assert digest(destination) == record["checkpoint_sha256"]
        record["checkpoint"] = str(destination)
        write_json(handoff_path, record)
        write_json(shared / "execution_handoff.json", record)
        return record

    try:
        save()
        handoff = stop_previous_execution()
        for method in specification["methods"]:
            name, reports = method["method"], []
            for relative in method["profiles"]:
                path = PROJECT_ROOT / relative
                config = json.loads(path.read_text())
                batch = config["microbatch_size_per_gpu"]
                profile_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile"
                if name == "partial_output_layers" and not (profile_root / "latest.pt").exists():
                    fork_execution_checkpoint(Path(handoff["checkpoint"]), profile_root / "latest.pt", path, PROJECT_ROOT, profile_only=True)
                label = name + "_profile_batch" + str(batch)
                command = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                    "scripts/run_lpwm_48gb_stage.py", "--action", "train", "--config", str(path), "--profile", "--resume"]
                record = execute(label, command, distributed=True, allow_memory_failure=True)
                report = {"configuration": str(path), "microbatch_size_per_gpu": batch,
                    "passed": record["returncode"] == 0 and record["memory_guard"] is None, **record}
                if report["passed"]:
                    summary = json.loads((profile_root / "training_summary.json").read_text())
                    assert summary["profile_only"] and summary["completed_updates"] == config["profile_updates"]
                    assert summary["peak_allocated_gib"] <= config["maximum_allocated_gib"]
                    rows = [json.loads(line) for line in (profile_root / "training_log.jsonl").read_text().splitlines()]
                    assert len(rows) == config["profile_updates"]
                    report.update(training_summary=summary,
                        steady_update_seconds=sum(row["update_seconds"] for row in rows[2:]) / len(rows[2:]))
                else:
                    failure = (root / (label + ".log")).read_text().lower()
                    if not record["memory_guard"] and not any(term in failure for term in
                            ("out of memory", "outofmemoryerror", "allocated-memory cap", "48gb gpu memory guard")):
                        raise RuntimeError(f"Non-memory error in {label}; dependent work blocked")
                reports.append(report)
                write_json(root / (name + "_execution_profiles.json"), reports)
                write_json(shared / (name + "_execution_profiles.json"), reports)
                if not report["passed"]:
                    break  # Larger allocations cannot solve a failed smaller batch.
            selected = choose_profile(reports, specification["profile_selection_cap_headroom_gib"])
            path = Path(selected["configuration"])
            config = json.loads(path.read_text())
            selection = {"selected_configuration": str(path), "profiles": reports, "selection_basis": specification["profile_selection"]}
            write_json(root / (name + "_execution_selection.json"), selection)
            write_json(shared / (name + "_execution_selection.json"), selection)
            common = ["--config", str(path), "--condition", "metric_plus_world"]
            if name != "partial_output_layers":
                audit_script = "scripts/audit_lpwm_lora_finetuning.py" if name == "attention_lora" else "scripts/audit_lpwm_adapter_full_finetuning.py"
                execute(name + "_gradient_audit", [audit_script, *common, "--device", "cuda:0", "--checkpoint", str(STAGE1),
                    "--output", str(root / (name + "_gpu_audit.json"))])
            profile_checkpoint = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile/checkpoint.pt"
            execute(name + "_evaluation_engineering", ["scripts/run_lpwm_48gb_stage.py", "--action", "evaluate",
                *common, "--engineering-checkpoint", str(profile_checkpoint)])
            run_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world"
            if name == "partial_output_layers" and not (run_root / "latest.pt").exists():
                fork_execution_checkpoint(Path(handoff["checkpoint"]), run_root / "latest.pt", path, PROJECT_ROOT)
                for filename in ("training_log.jsonl", "validation_log.jsonl"):
                    shutil.copy2(OLD_PARTIAL / "metric_plus_world" / filename, run_root / filename)
                source_visualization = OLD_PARTIAL / "metric_plus_world/visualization"
                if source_visualization.exists():
                    shutil.copytree(source_visualization, run_root / "visualization", dirs_exist_ok=True)
            execute(name + "_training", ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                "scripts/run_lpwm_48gb_stage.py", "--action", "train", *common, "--resume"], distributed=True)
            training = json.loads((run_root / "training_summary.json").read_text())
            assert training["completed_updates"] == 4707 and not training["profile_only"]
            assert training["configuration_sha256"] == digest(path)
            execute(name + "_evaluation", ["scripts/run_lpwm_48gb_stage.py", "--action", "evaluate", *common])
            report = json.loads((run_root / "trend_evaluation/summary.json").read_text())
            assert not report["engineering_only"] and report["checks"]["complete_registered_epoch"]
        execute("four_method_summary", ["scripts/summarize_lpwm_48gb_methods.py", "--config", str(arguments.config)], gpu=False)
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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
