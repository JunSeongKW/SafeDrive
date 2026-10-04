"""Adopt the running partial baseline, defer object GT, then train/evaluate LoRA."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_partial_protocol import prepare_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
CHECKPOINT = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"


def process_identity(pid):
    root = Path(f"/proc/{pid}")
    try:
        fields = (root / "stat").read_text().rsplit(") ", 1)[1].split()
        return {"pid": pid, "state": fields[0], "parent_pid": int(fields[1]), "start_ticks": fields[19],
            "uid": root.stat().st_uid, "arguments": (root / "cmdline").read_bytes().decode().split("\0")[:-1]}
    except FileNotFoundError:
        return None


def identity_alive(identity):
    current = process_identity(identity["pid"])
    return current is not None and current["state"] != "Z" and current["start_ticks"] == identity["start_ticks"]


def signal_owned_group(identity):
    if identity_alive(identity):
        assert identity["uid"] == os.getuid()
        assert os.getpgid(identity["pid"]) == identity["pid"]
        os.killpg(identity["pid"], signal.SIGINT)


def select_execution_profile(profile_records):
    passed = [record for record in profile_records if record.get("passed")]
    if not passed:
        raise RuntimeError("No LoRA execution profile passed the finite-loss and memory checks")
    return min(passed, key=lambda record: record["last_three_mean_update_seconds"])


def projected_upsize_peak_gib(previous_peak_gib, previous_batch, proposed_batch, workspace_margin_gib):
    """Conservative linear batch extrapolation, with extra kernel workspace."""
    return previous_peak_gib * proposed_batch / previous_batch + workspace_margin_gib


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    root.mkdir(parents=True, exist_ok=True)
    if arguments.detach:
        with (root / "queue.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)],
                cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
        print(json.dumps({"queue_pid": child.pid, "output_directory": str(root)}), flush=True)
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "queue_completion.json").exists():
        return
    if (root / "pause.requested").exists() or (root / "queue_failed.json").exists():
        raise RuntimeError("Explicitly resolve this queue's saved pause/failure before resuming")
    partial_config_path = PROJECT_ROOT / specification["partial_config"]
    partial_specification = json.loads(partial_config_path.read_text())
    partial_root = PROJECT_ROOT / partial_specification["output_directory"]
    previous_registration = json.loads((partial_root / "queue_registration.json").read_text())
    sources = dict(previous_registration["source_sha256"])
    for name in ("src/planning_aware_future_prediction/object_centric/lpwm_lora_finetuning.py",
            "scripts/train_lpwm_lora_planning.py", "scripts/evaluate_lpwm_lora_planning.py",
            "scripts/audit_lpwm_lora_finetuning.py", "scripts/queue_lpwm_adaptation_methods.py",
            "scripts/summarize_lpwm_adaptation_methods.py"):
        sources[name] = digest(PROJECT_ROOT / name)
    configuration_paths = [arguments.config, partial_config_path, *[PROJECT_ROOT / name for name in specification["lora_profiles"]]]
    configurations = {str(path): digest(path) for path in configuration_paths}
    registration = {"source_sha256": sources, "configuration_sha256": configurations,
        "stage1_checkpoint_sha256": digest(CHECKPOINT), "object_gt_condition_deferred": True}
    registration_path = root / "queue_registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write_json(registration_path, registration)
    def verify_registration():
        assert all(digest(PROJECT_ROOT / name) == expected for name, expected in sources.items())
        assert all(digest(Path(name)) == expected for name, expected in configurations.items())
    verify_registration()
    audit = json.loads((root / "lora_cpu_engineering_audit.json").read_text())
    assert audit["frozen_native_weights_identical_after_optimizer_step"]
    assert audit["ssl_gradient_norms"]["dynamics"] > 0
    assert audit["lora_model_source_sha256"] == sources["src/planning_aware_future_prediction/object_centric/lpwm_lora_finetuning.py"]
    partial_protocol = prepare_protocol(partial_config_path)
    for path in configuration_paths[2:]:
        protocol = prepare_protocol(path)
        assert protocol["planning_tokens"] == partial_protocol["planning_tokens"]
        assert protocol["world_tokens"] == partial_protocol["world_tokens"]
    write_json(shared / "queue_registration.json", registration)
    write_json(shared / "comparison_configuration.json", specification)
    state = {"queue_pid": os.getpid(), "status": "running", "current_stage": "adopting_partial_training",
        "created_unix": time.time(), "nodes": {}, "object_gt_condition_deferred": True}
    owned = []
    def save():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)
    def stop_requested(number, _frame):
        raise KeyboardInterrupt(f"Method-comparison queue signal {number}")
    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    def resources():
        output = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        return {int(values[0]): {"used_bytes": int(values[1]) * 1024**2, "free_gib": int(values[2]) / 1024}
            for values in ([part.strip() for part in line.split(",")] for line in output.splitlines())}
    def paused():
        return (root / "pause.requested").exists() or (partial_root / "pause.requested").exists()
    def memory_pressure(current):
        return any(row["used_bytes"] > specification["maximum_gpu_used_bytes"] or
            row["free_gib"] < specification["minimum_free_gib"] for row in current.values())
    def watch(identity, label, child=None, allow_memory_failure=False):
        pressure = None
        while identity_alive(identity) if child is None else child.poll() is None:
            state["resources"] = resources()
            save()
            if paused():
                raise RuntimeError("Method-comparison queue explicitly paused")
            if memory_pressure(state["resources"]):
                if not allow_memory_failure:
                    raise RuntimeError("GPU total VRAM or minimum-free guard exceeded")
                if pressure is None:
                    pressure = {"reason": "GPU memory guard", "resources": state["resources"], "time_unix": time.time()}
                    signal_owned_group(identity)
                elif time.time() - pressure["time_unix"] > 120:
                    raise RuntimeError("Profile did not stop after memory guard")
            time.sleep(5)
        return pressure
    def execute(label, command, distributed=False, gpu=True, allow_memory_failure=False):
        sentinel = root / "completed_nodes" / (label + ".json")
        if sentinel.exists():
            record = json.loads(sentinel.read_text())
            assert record["comparison_config_sha256"] == digest(arguments.config)
            state["nodes"][label] = {"status": "previously_complete"}
            return record
        verify_registration()
        waiting_since = time.monotonic()
        while gpu:
            if paused():
                raise RuntimeError("Method-comparison queue explicitly paused")
            current = resources()
            required = specification["minimum_training_admission_free_gib"] if distributed else specification["minimum_evaluation_admission_free_gib"]
            if all(current[index]["free_gib"] >= required for index in ((0, 1) if distributed else (0,))):
                break
            state.update(current_stage="waiting_for_gpu:" + label, resources=current)
            save()
            if time.monotonic() - waiting_since > 86400:
                raise RuntimeError("GPU admission exceeded one day")
            time.sleep(10)
        if paused():
            raise RuntimeError("Method-comparison queue explicitly paused")
        with (root / (label + ".log")).open("a") as stream:
            child = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1" if gpu else "", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            identity = process_identity(child.pid)
            assert identity is not None and identity["uid"] == os.getuid()
            owned.append(identity)
            state.update(current_stage=label)
            state["nodes"][label] = {"status": "running", "pid": child.pid, "started_unix": time.time()}
            save()
            pressure = watch(identity, label, child, allow_memory_failure)
            returncode = child.wait()
        record = {"returncode": returncode, "memory_guard": pressure,
            "comparison_config_sha256": digest(arguments.config), "finished_unix": time.time()}
        state["nodes"][label].update(status="complete" if returncode == 0 and pressure is None else "failed", **record)
        save()
        if (returncode or pressure) and not allow_memory_failure:
            raise RuntimeError(f"Queue stage {label} failed: {record}")
        if returncode == 0 and pressure is None:
            write_json(sentinel, record)
        return record
    try:
        save()
        supersession_path = root / "partial_queue_supersession.json"
        if not supersession_path.exists():
            old_state = json.loads((partial_root / "queue_state.json").read_text())
            supervisor = process_identity(specification["superseded_queue_pid"])
            training = process_identity(specification["adopt_training_pid"])
            assert old_state["current_stage"] == "metric_plus_world_training"
            assert supervisor is not None and training is not None
            assert supervisor["uid"] == training["uid"] == os.getuid()
            assert any(argument.endswith("scripts/queue_lpwm_partial_planning.py") for argument in supervisor["arguments"])
            assert "scripts/train_lpwm_partial_planning.py" in training["arguments"]
            assert "metric_plus_world" in training["arguments"]
            assert training["parent_pid"] == supervisor["pid"]
            assert os.getpgid(training["pid"]) == training["pid"]
            assert not (partial_root / "metric_object_future_plus_world").exists()
            amendment = {"reason": "User prioritizes partial-vs-LoRA before direct object GT supervision",
                "superseded_supervisor": supervisor, "adopted_training": training,
                "previous_queue_state": old_state, "new_queue_pid": os.getpid(), "time_unix": time.time(),
                "old_source_and_configuration_unchanged": True,
                "supervisor_signal": "SIGKILL to supervisor PID only; its SIGTERM handler would stop training",
                "training_signal_sent": False, "object_auxiliary_condition": "deferred; not automatically scheduled"}
            write_json(supersession_path, amendment)
            write_json(shared / "partial_queue_supersession.json", amendment)
            owned.append(training)
            # The training process has its own session. Replacing only its CPU
            # supervisor prevents the obsolete GT job from being launched.
            os.kill(supervisor["pid"], signal.SIGKILL)
            for _ in range(50):
                if not identity_alive(supervisor):
                    break
                time.sleep(.1)
            assert not identity_alive(supervisor) and identity_alive(training)
            write_json(partial_root / "queue_superseded.json", amendment)
            write_json(partial_root / "queue_state.json", {**old_state, "status": "superseded",
                "new_queue_directory": str(root), "new_queue_pid": os.getpid(), "training_continues": True})
        else:
            amendment = json.loads(supersession_path.read_text())
            training = amendment["adopted_training"]
            owned.append(training)
        state.update(current_stage="partial_training_adopted")
        state["nodes"]["partial_training"] = {"status": "running", "pid": training["pid"], "adopted_without_restart": True}
        save()
        watch(training, "partial_training")
        completion = json.loads((partial_root / "metric_plus_world/training_summary.json").read_text())
        assert completion["completed_updates"] == 4707 and completion["epochs"] == 1 and not completion["profile_only"]
        assert completion["configuration_sha256"] == digest(partial_config_path)
        state["nodes"]["partial_training"].update(status="complete", completed_updates=completion["completed_updates"])
        save()
        execute("partial_evaluation", ["scripts/evaluate_lpwm_partial_planning.py", "--config", str(partial_config_path),
            "--condition", "metric_plus_world"])
        profiles = []
        for profile_index, relative in enumerate(specification["lora_profiles"]):
            config_path = PROJECT_ROOT / relative
            config = json.loads(config_path.read_text())
            successful = [record for record in profiles if record.get("passed")]
            if successful:
                previous = successful[-1]
                previous_config = json.loads(Path(previous["configuration"]).read_text())
                projected = projected_upsize_peak_gib(previous["training_summary"]["peak_allocated_gib"],
                    previous_config["microbatch_size_per_gpu"], config["microbatch_size_per_gpu"],
                    specification["batch_upsize_workspace_margin_gib"])
                if projected > config["maximum_allocated_gib"]:
                    profiles.append({"configuration": str(config_path), "passed": False,
                        "skipped_before_gpu_execution": True, "projected_peak_allocated_gib": projected,
                        "failure_reason": "Conservative batch-size memory projection exceeds allocated cap"})
                    write_json(root / "lora_execution_profiles.json", profiles)
                    write_json(shared / "lora_execution_profiles.json", profiles)
                    continue
            record = execute(f"lora_execution_profile_{profile_index + 1}", ["-m", "torch.distributed.run", "--standalone",
                "--nnodes=1", "--nproc-per-node=2", "scripts/train_lpwm_lora_planning.py", "--config", str(config_path),
                "--condition", "metric_plus_world", "--profile"], distributed=True, allow_memory_failure=True)
            profile_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile"
            passed = record["returncode"] == 0 and record["memory_guard"] is None and (profile_root / "training_summary.json").exists()
            report = {"configuration": str(config_path), "passed": passed, **record}
            if passed:
                summary = json.loads((profile_root / "training_summary.json").read_text())
                assert summary["profile_only"] and summary["completed_updates"] == config["profile_updates"]
                assert summary["peak_allocated_gib"] <= config["maximum_allocated_gib"]
                rows = [json.loads(line) for line in (profile_root / "training_log.jsonl").read_text().splitlines()]
                report.update(training_summary=summary, last_three_mean_update_seconds=sum(row["update_seconds"] for row in rows[-3:]) / 3)
            else:
                stopped = profile_root / "stopped.json"
                reason = json.loads(stopped.read_text()).get("error", "") if stopped.exists() else ""
                profile_log = (root / f"lora_execution_profile_{profile_index + 1}.log").read_text()
                failure_text = (reason + "\n" + profile_log).lower()
                if not record["memory_guard"] and not any(term in failure_text for term in ("out of memory", "outofmemoryerror", "allocated-memory cap", "memory limit")):
                    raise RuntimeError(f"Non-memory profile failure blocks comparison: {reason}")
                report["failure_reason"] = reason or "GPU memory guard"
            profiles.append(report)
            write_json(root / "lora_execution_profiles.json", profiles)
            write_json(shared / "lora_execution_profiles.json", profiles)
        selected = select_execution_profile(profiles)
        selected_path = Path(selected["configuration"])
        selected_config = json.loads(selected_path.read_text())
        selection = {"selected_configuration": str(selected_path), "selection_basis": specification["profile_selection"],
            "profiles": profiles, "selected_before_main_training": True}
        write_json(root / "lora_execution_selection.json", selection)
        write_json(shared / "lora_execution_selection.json", selection)
        common = ["--config", str(selected_path), "--condition", "metric_plus_world"]
        execute("lora_gradient_audit", ["scripts/audit_lpwm_lora_finetuning.py", *common, "--device", "cuda:0",
            "--checkpoint", str(CHECKPOINT), "--output", str(root / "lora_gpu_engineering_audit.json")])
        profile_checkpoint = PROJECT_ROOT / selected_config["output_directory"] / "metric_plus_world/profile/checkpoint.pt"
        execute("lora_evaluation_engineering", ["scripts/evaluate_lpwm_lora_planning.py", *common,
            "--engineering-checkpoint", str(profile_checkpoint)])
        execute("lora_training", ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
            "scripts/train_lpwm_lora_planning.py", *common, "--resume"], distributed=True)
        execute("lora_evaluation", ["scripts/evaluate_lpwm_lora_planning.py", *common])
        execute("paired_method_report", ["scripts/summarize_lpwm_adaptation_methods.py", "--config", str(arguments.config)], gpu=False)
        state.update(status="complete", current_stage="complete")
        save()
        write_json(root / "queue_completion.json", {"complete": True, "object_gt_condition_deferred": True, "finished_unix": time.time()})
    except BaseException as error:
        for identity in owned:
            signal_owned_group(identity)
        state.update(status="paused" if paused() else "failed", error=repr(error))
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
    run(arguments)
