"""Wait for partial/LoRA validation, then run adapter and full-weight methods."""
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
from queue_lpwm_adaptation_methods import (
    process_identity, identity_alive, signal_owned_group, select_execution_profile, projected_upsize_peak_gib)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
CHECKPOINT = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt"


def predecessor_ready(predecessor_root):
    """Require a successful completed comparison, not merely a vanished PID."""
    if (predecessor_root / "queue_failed.json").exists():
        raise RuntimeError("Predecessor execution failed; follow-up methods remain blocked")
    if not (predecessor_root / "queue_completion.json").exists():
        return False
    completion = json.loads((predecessor_root / "queue_completion.json").read_text())
    report = json.loads((predecessor_root / "method_comparison_summary.json").read_text())
    assert completion["complete"] and report["complete"]
    for condition in report["reports"].values():
        assert not condition["engineering_only"] and condition["checks"]["complete_registered_epoch"]
    return True


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
    if (root / "queue_failed.json").exists() or (root / "pause.requested").exists():
        raise RuntimeError("Saved failure/pause requires explicit resolution before resuming")
    predecessor = PROJECT_ROOT / specification["predecessor_queue_directory"]
    earlier = json.loads((predecessor / "queue_registration.json").read_text())
    sources, configurations = dict(earlier["source_sha256"]), dict(earlier["configuration_sha256"])
    for name in ("src/planning_aware_future_prediction/object_centric/lpwm_adapter_full_finetuning.py",
            "scripts/train_lpwm_adapter_full_planning.py", "scripts/evaluate_lpwm_adapter_full_planning.py",
            "scripts/audit_lpwm_adapter_full_finetuning.py", "scripts/queue_lpwm_followup_methods.py",
            "scripts/summarize_lpwm_four_methods.py", "scripts/train_lpwm_resumed_full_planning.py",
            "scripts/lpwm_full_continuation.py"):
        sources[name] = digest(PROJECT_ROOT / name)
    configurations[str(arguments.config)] = digest(arguments.config)
    profiles = [PROJECT_ROOT / path for method in specification["followup_methods"] for path in method["profiles"]]
    for path in profiles:
        configurations[str(path)] = digest(path)
    registration = {"source_sha256": sources, "configuration_sha256": configurations,
        "stage1_checkpoint_sha256": digest(CHECKPOINT), "object_gt_deferred": True,
        "method_order": specification["method_order"]}
    full_specification = json.loads(next(path for path in profiles if
        json.loads(path.read_text())["adaptation_method"] == "full_low_learning_rate").read_text())
    continuation = full_specification["resume_from_prior_full"]
    assert digest(PROJECT_ROOT / continuation["checkpoint"]) == continuation["checkpoint_sha256"]
    registration["preserved_full_resume_checkpoint_sha256"] = continuation["checkpoint_sha256"]
    registration_path = root / "queue_registration.json"
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write_json(registration_path, registration)

    def verify():
        assert all(digest(PROJECT_ROOT / name) == value for name, value in sources.items())
        assert all(digest(Path(name)) == value for name, value in configurations.items())

    verify()
    previous_specification = json.loads((PROJECT_ROOT / specification["predecessor_config"]).read_text())
    previous_protocol = prepare_protocol(PROJECT_ROOT / previous_specification["partial_config"])
    for path in profiles:
        protocol = prepare_protocol(path)
        assert protocol["planning_tokens"] == previous_protocol["planning_tokens"]
        assert protocol["world_tokens"] == previous_protocol["world_tokens"]
    for method in specification["followup_methods"]:
        audit = json.loads((root / (method["method"] + "_cpu_audit.json")).read_text())
        assert audit["audit_passed"]
        assert audit["model_source_sha256"] == sources[
            "src/planning_aware_future_prediction/object_centric/lpwm_adapter_full_finetuning.py"]
        assert audit["configuration_sha256"] == digest(PROJECT_ROOT / method["profiles"][0])
        write_json(shared / (method["method"] + "_cpu_audit.json"), audit)
    write_json(shared / "queue_registration.json", registration)
    write_json(shared / "sequence_configuration.json", specification)
    state = {"queue_pid": os.getpid(), "status": "waiting", "current_stage": "waiting_for_partial_and_lora",
        "method_order": specification["method_order"], "nodes": {}, "created_unix": time.time(),
        "object_gt_deferred": True}
    owned = []

    def save():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)

    def paused():
        return any((directory / "pause.requested").exists() for directory in
            (root, predecessor, PROJECT_ROOT / "outputs/lpwm_partial_planning_v1"))

    def stop_requested(number, _frame):
        raise KeyboardInterrupt(f"Follow-up queue signal {number}")

    signal.signal(signal.SIGINT, stop_requested)
    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def resources():
        output = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used,memory.free",
            "--format=csv,noheader,nounits"], text=True)
        return {int(values[0]): {"used_bytes": int(values[1]) * 1024**2, "free_gib": int(values[2]) / 1024}
            for values in ([part.strip() for part in line.split(",")] for line in output.splitlines())}

    def execute(label, command, distributed=False, gpu=True, allow_memory_failure=False):
        sentinel = root / "completed_nodes" / (label + ".json")
        if sentinel.exists():
            record = json.loads(sentinel.read_text())
            assert record["sequence_config_sha256"] == digest(arguments.config)
            state["nodes"][label] = {"status": "previously_complete"}
            return record
        verify()
        waiting_since = time.monotonic()
        while gpu:
            if paused():
                raise RuntimeError("Follow-up queue paused")
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
            raise RuntimeError("Follow-up queue paused")
        with (root / (label + ".log")).open("a") as stream:
            child = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1" if gpu else "", "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
            identity = process_identity(child.pid)
            assert identity is not None and identity["uid"] == os.getuid()
            owned.append(identity)
            state.update(status="running", current_stage=label)
            state["nodes"][label] = {"status": "running", "pid": child.pid, "started_unix": time.time()}
            pressure = None
            while child.poll() is None:
                if paused():
                    raise RuntimeError("Follow-up queue paused")
                if gpu:
                    current = resources()
                    state["resources"] = current
                    if any(row["used_bytes"] > specification["maximum_gpu_used_bytes"] or
                            row["free_gib"] < specification["minimum_free_gib"] for row in current.values()):
                        if not allow_memory_failure:
                            raise RuntimeError("GPU memory guard exceeded")
                        if pressure is None:
                            pressure = {"resources": current, "time_unix": time.time()}
                            signal_owned_group(identity)
                        elif time.time() - pressure["time_unix"] > 120:
                            raise RuntimeError("Profile did not stop after memory guard")
                save()
                time.sleep(5)
            returncode = child.wait()
        record = {"returncode": returncode, "memory_guard": pressure,
            "sequence_config_sha256": digest(arguments.config), "finished_unix": time.time()}
        state["nodes"][label].update(status="complete" if returncode == 0 and pressure is None else "failed", **record)
        save()
        if (returncode or pressure) and not allow_memory_failure:
            raise RuntimeError(f"Queue stage {label} failed: {record}")
        if returncode == 0 and pressure is None:
            write_json(sentinel, record)
        return record

    try:
        while not predecessor_ready(predecessor):
            if paused():
                raise RuntimeError("Follow-up queue paused while waiting")
            previous_state = json.loads((predecessor / "queue_state.json").read_text())
            state["predecessor_stage"] = previous_state["current_stage"]
            state["predecessor_status"] = previous_state["status"]
            if previous_state["status"] in ("failed", "paused"):
                raise RuntimeError("Predecessor is failed or paused")
            identity = process_identity(previous_state["queue_pid"])
            if identity is None or identity["state"] == "Z":
                raise RuntimeError("Predecessor disappeared without completing validation")
            assert identity["uid"] == os.getuid()
            assert any(argument.endswith("scripts/queue_lpwm_adaptation_methods.py") for argument in identity["arguments"])
            save()
            time.sleep(10)
        verify()
        state.update(status="running", current_stage="predecessor_validated")
        save()
        for method in specification["followup_methods"]:
            name = method["method"]
            reports = []
            for index, relative in enumerate(method["profiles"]):
                path = PROJECT_ROOT / relative
                config = json.loads(path.read_text())
                successful = [row for row in reports if row.get("passed")]
                if reports and not successful:
                    raise RuntimeError(f"Smallest {name} profile failed memory admission; larger batches blocked")
                if successful:
                    previous = successful[-1]
                    projected = projected_upsize_peak_gib(previous["training_summary"]["peak_allocated_gib"],
                        previous["microbatch_size_per_gpu"], config["microbatch_size_per_gpu"],
                        specification["batch_upsize_workspace_margin_gib"])
                    if projected > config["maximum_allocated_gib"]:
                        reports.append({"configuration": str(path), "passed": False, "skipped_before_launch": True,
                            "reason": "Conservative larger-batch memory estimate exceeds cap", "projected_peak_gib": projected})
                        write_json(root / (name + "_execution_profiles.json"), reports)
                        continue
                label = f"{name}_execution_profile_{index + 1}"
                common = ["--config", str(path), "--condition", "metric_plus_world"]
                record = execute(label, ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                    "scripts/train_lpwm_adapter_full_planning.py", *common, "--profile"], distributed=True, allow_memory_failure=True)
                profile_root = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile"
                report = {"configuration": str(path), "passed": record["returncode"] == 0 and record["memory_guard"] is None,
                    "microbatch_size_per_gpu": config["microbatch_size_per_gpu"], "execution": record}
                if report["passed"]:
                    summary = json.loads((profile_root / "training_summary.json").read_text())
                    assert summary["profile_only"] and summary["completed_updates"] == config["profile_updates"]
                    assert summary["peak_allocated_gib"] <= config["maximum_allocated_gib"]
                    rows = [json.loads(line) for line in (profile_root / "training_log.jsonl").read_text().splitlines()]
                    report.update(training_summary=summary,
                        last_three_mean_update_seconds=sum(row["update_seconds"] for row in rows[-3:]) / 3)
                else:
                    failure = (root / (label + ".log")).read_text().lower()
                    if not record["memory_guard"] and not any(term in failure for term in
                            ("out of memory", "outofmemoryerror", "allocated-memory cap", "memory limit")):
                        raise RuntimeError(f"Non-memory failure in {label}; dependent jobs blocked")
                reports.append(report)
                write_json(root / (name + "_execution_profiles.json"), reports)
            selected = select_execution_profile(reports)
            path = Path(selected["configuration"])
            config = json.loads(path.read_text())
            selection = {"selected_configuration": str(path), "profiles": reports, "selection_basis": specification["profile_selection"]}
            write_json(root / (name + "_execution_selection.json"), selection)
            write_json(shared / (name + "_execution_selection.json"), selection)
            common = ["--config", str(path), "--condition", "metric_plus_world"]
            execute(name + "_gradient_audit", ["scripts/audit_lpwm_adapter_full_finetuning.py", *common,
                "--device", "cuda:0", "--checkpoint", str(CHECKPOINT), "--output", str(root / (name + "_gpu_audit.json"))])
            profile_checkpoint = PROJECT_ROOT / config["output_directory"] / "metric_plus_world/profile/checkpoint.pt"
            execute(name + "_evaluation_engineering", ["scripts/evaluate_lpwm_adapter_full_planning.py", *common,
                "--engineering-checkpoint", str(profile_checkpoint)])
            execute(name + "_training", ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                "scripts/train_lpwm_adapter_full_planning.py", *common, "--resume"], distributed=True)
            execute(name + "_evaluation", ["scripts/evaluate_lpwm_adapter_full_planning.py", *common])
            summary = json.loads((PROJECT_ROOT / config["output_directory"] /
                "metric_plus_world/trend_evaluation/summary.json").read_text())
            assert not summary["engineering_only"] and summary["checks"]["complete_registered_epoch"]
            state["nodes"][name + "_evaluation"]["scientific_checks"] = summary["checks"]
            save()
        execute("four_method_report", ["scripts/summarize_lpwm_four_methods.py", "--config", str(arguments.config)], gpu=False)
        state.update(status="complete", current_stage="complete")
        save()
        write_json(root / "queue_completion.json", {"complete": True, "finished_unix": time.time(), "object_gt_deferred": True})
    except BaseException as error:
        for identity in owned:
            signal_owned_group(identity)
        state.update(status="paused" if paused() else "failed", error=repr(error))
        save()
        write_json(root / "queue_failed.json", {"stage": state["current_stage"], "error": repr(error),
            "dependent_jobs_blocked": True, "time_unix": time.time()})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detach", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
