"""Durable CPU preparation -> adopted stage1 -> gated stage2/evaluation queue.

Never owns or interrupts the already running stage1 process. The old stage1
supervisor's subsequent planner invocation joins this queue instead of starting
a second GPU job. A failed gate records diagnostics and blocks its dependents.
"""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"
OFFICIAL_PYTHON = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python"


def join_queue(root):
    while True:
        if (root / "queue_completion.json").exists():
            assert json.loads((root / "queue_completion.json").read_text())["complete"]
            return
        if (root / "queue_failed.json").exists():
            raise RuntimeError("Validated LPWM queue failed; see " + str(root / "queue_failed.json"))
        state = json.loads((root / "queue_state.json").read_text())
        os.kill(state["supervisor_pid"], 0)
        time.sleep(15)


def run(arguments):
    specification = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    if arguments.join:
        return join_queue(root)
    if arguments.detach:
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        with (root / "queue.log").open("a") as stream:
            child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)],
                cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"queue_pid": child.pid, "log": str(root / "queue.log")}), flush=True)
        return
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "queue_completion.json").exists():
        return
    if (root / "queue_failed.json").exists():
        raise RuntimeError("Explicitly resolve and archive queue_failed.json before resuming; no automatic failed-job retry")
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    stage1_config = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_config["output_directory"]
    teacher_root = PROJECT_ROOT / specification["teacher_directory"]
    state = {"supervisor_pid": os.getpid(), "configuration_sha256": digest(arguments.config), "created_unix": time.time(),
        "current_stage": "prepare_candidate_teacher_while_stage1_runs", "adopted_stage1": str(stage1_root), "nodes": {},
        "planned_sequence": ["candidate_teacher", "stage1_full_training_and_evaluation_external", "stage1_causality", "stage1_gate", "teacher_gate", "planning_gradient_audit"]
            + [condition + " -> full_training -> development -> world_retention -> validation_gate" for condition in specification["conditions"]]
            + (["locked_full_navtest", "paired_report"] if specification.get("automatic_navtest", True) else ["paired_development_ablation_report"])}
    source_paths = [Path(__file__).resolve(), PROJECT_ROOT / "scripts/train_lpwm_full_planning.py",
        PROJECT_ROOT / "scripts/evaluate_lpwm_full_planning.py", PROJECT_ROOT / "scripts/prepare_lpwm_candidate_teacher.py",
        PROJECT_ROOT / "scripts/validate_lpwm_stage_transition.py", PROJECT_ROOT / "scripts/audit_lpwm_planning_finetuning.py",
        PROJECT_ROOT / "scripts/lpwm_refinement_oracle.py",
        PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_candidate_planner.py",
        PROJECT_ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_planning_finetuning.py"]
    if "stage1_admission_amendment" in specification:
        source_paths.extend(PROJECT_ROOT / name for name in (
            "scripts/lpwm_stage2_admission.py", "scripts/prepare_lpwm_stage2_object_targets.py",
            "scripts/summarize_lpwm_object_auxiliary_ablation.py",
            "src/planning_aware_future_prediction/object_centric/lpwm_object_supervision.py",
            specification["stage1_admission_amendment"]))
    identity = {"configuration_sha256": digest(arguments.config), "source_sha256": {str(path.relative_to(PROJECT_ROOT)): digest(path) for path in source_paths}}
    if (root / "queue_registration.json").exists():
        assert json.loads((root / "queue_registration.json").read_text()) == identity
    else:
        write_json(root / "queue_registration.json", identity)
    children, streams = [], []
    def resource_usage():
        import psutil
        process = psutil.Process(os.getpid())
        cpu_rss = 0
        for member in [process, *process.children(recursive=True)]:
            try:
                cpu_rss += member.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        used_bytes = {gpu: int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
            "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).strip()) * 1024**2 for gpu in (0, 1)}
        state["resource_usage"] = {"training_process_tree_rss_bytes": cpu_rss, "gpu_total_used_bytes": used_bytes,
            "cpu_rss_note": "Conservative sum including shared mappings; other users' CPU processes excluded"}
        return cpu_rss, used_bytes
    def save_state():
        state["heartbeat_unix"] = time.time()
        write_json(root / "queue_state.json", state)
    def verify_sources():
        assert digest(arguments.config) == identity["configuration_sha256"]
        assert all(digest(PROJECT_ROOT / path) == expected for path, expected in identity["source_sha256"].items()), "Unregistered queue source change"
    def start(name, command, interpreter=PYTHON, cpu=False):
        verify_sources()
        if (root / "pause.requested").exists():
            raise RuntimeError("Explicit queue pause marker present")
        if not cpu:
            distributed_training = "torch.distributed.run" in command
            required_free_gib = specification.get("minimum_training_admission_free_gib", 44) if distributed_training else specification.get("minimum_evaluation_admission_free_gib", 24)
            devices = (0, 1) if distributed_training else (0,)
            waiting_started = time.time()
            while True:
                available = {gpu: int(subprocess.check_output(["nvidia-smi", f"--id={gpu}",
                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip()) / 1024 for gpu in devices}
                if all(value >= required_free_gib for value in available.values()):
                    break
                state["current_stage"] = "waiting_for_gpu_memory:" + name
                state["gpu_admission"] = {"free_gib": available, "required_free_gib": required_free_gib}
                save_state()
                if (root / "pause.requested").exists() or time.time() - waiting_started > 86400:
                    raise RuntimeError("GPU admission paused or exceeded one day; other users' processes are untouched")
                time.sleep(15)
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "" if cpu else "0,1", "OMP_NUM_THREADS": "1" if cpu else "4", "MKL_NUM_THREADS": "1" if cpu else "4", "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
        stream = (root / (name + ".log")).open("a")
        streams.append(stream)
        child = subprocess.Popen([str(interpreter), *command], cwd=PROJECT_ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        state["nodes"][name] = {"status": "running", "pid": child.pid, "started_unix": time.time(), "command": command}
        state["current_stage"] = name
        save_state()
        return child
    def finish(name, child):
        while child.poll() is None:
            if "resource_limits" in specification:
                cpu_rss, gpu_usage = resource_usage()
                limits = specification["resource_limits"]
                cpu_exceeded = ("maximum_training_cpu_rss_bytes" in limits and
                    cpu_rss > limits["maximum_training_cpu_rss_bytes"] - limits.get("cpu_stop_reserve_bytes", 0))
                gpu_exceeded = any(value > limits["maximum_gpu_used_bytes"] for value in gpu_usage.values())
                if cpu_exceeded or gpu_exceeded:
                    state["resource_stop"] = {"cpu_limit_approached": cpu_exceeded, "gpu_limit_exceeded": gpu_exceeded}
                    save_state()
                    os.killpg(child.pid, signal.SIGINT)
                    raise RuntimeError("User memory limit approached; training signalled to checkpoint and stop")
            save_state()
            if (root / "pause.requested").exists():
                os.killpg(child.pid, signal.SIGINT)
                raise RuntimeError("Queue paused by marker")
            time.sleep(15)
        state["nodes"][name].update(status="complete" if child.returncode == 0 else "failed", returncode=child.returncode, finished_unix=time.time())
        save_state()
        if child.returncode:
            raise RuntimeError(f"Queue node {name} failed with exit {child.returncode}; dependent nodes blocked")
    def checked(name, command, interpreter=PYTHON, cpu=False):
        sentinel = root / "completed_nodes" / (name + ".json")
        if sentinel.exists():
            state["nodes"][name] = {"status": "previously_complete"}
            return
        finish(name, start(name, command, interpreter, cpu))
        write_json(sentinel, {"completed_unix": time.time(), "configuration_sha256": identity["configuration_sha256"]})
    def signal_stop(number, _frame):
        raise KeyboardInterrupt(f"Queue signal {number}; adopted stage1 is not a child")
    signal.signal(signal.SIGINT, signal_stop)
    signal.signal(signal.SIGTERM, signal_stop)
    config_arguments = ["--config", str(arguments.config)]
    validation = ["scripts/validate_lpwm_stage_transition.py", *config_arguments]
    inference = ["scripts/evaluate_lpwm_full_planning.py", *config_arguments]
    save_state()
    try:
        teacher_process = None
        if not (teacher_root / "completion.json").exists():
            teacher_process = start("candidate_teacher", ["scripts/prepare_lpwm_candidate_teacher.py", *config_arguments,
                "--workers", str(specification["teacher_workers"])], OFFICIAL_PYTHON, cpu=True)
        # Existing supervisor performs full training and both full evaluations.
        state["current_stage"] = "waiting_for_stage1_full_training_and_validation"
        while not (stage1_root / "stage1_completion.json").exists():
            if teacher_process is not None and teacher_process.poll() not in (None, 0):
                finish("candidate_teacher", teacher_process)
            active = json.loads((stage1_root / "active_stage.json").read_text())
            os.kill(active["supervisor_pid"], 0)
            state["stage1_progress"] = json.loads((stage1_root / "stage1/progress.json").read_text())
            if (root / "pause.requested").exists():
                raise RuntimeError("Queue pause marker present")
            save_state()
            time.sleep(15)
        if "stage1_admission_amendment" in specification:
            import shutil
            amendment = json.loads((PROJECT_ROOT / specification["stage1_admission_amendment"]).read_text())
            shutil.copyfile(PROJECT_ROOT / amendment["historical_causality_audit"], root / "stage1_causality_audit.json")
            state["nodes"]["stage1_causality"] = {"status": "reused_verified_evidence", "source": amendment["historical_causality_audit"]}
        else:
            checked("stage1_causality", [*validation, "--mode", "stage1-causality"])
        checked("stage1_gate", [*validation, "--mode", "stage1-gate"], cpu=True)
        if teacher_process is not None:
            finish("candidate_teacher", teacher_process)
        checked("teacher_gate", [*validation, "--mode", "teacher-gate"], cpu=True)
        checked("planning_gradient_audit", ["scripts/audit_lpwm_planning_finetuning.py", *config_arguments, "--device", "cuda:0",
            "--checkpoint", str(stage1_root / "stage1/checkpoint.pt"), "--output", str(root / "planning_gradient_audit.json")])
        distributed = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
        for condition in specification["conditions"]:
            condition_arguments = ["--condition", condition]
            training = [*distributed, "scripts/train_lpwm_full_planning.py", *config_arguments, *condition_arguments]
            if "object_future" in condition:
                checked("object_target_preparation", ["scripts/prepare_lpwm_stage2_object_targets.py", *config_arguments], cpu=True)
            if "refinement" in condition or "object_future" in condition:
                checked(condition + "_gradient_audit", ["scripts/audit_lpwm_planning_finetuning.py", *config_arguments,
                    "--condition", condition, "--device", "cuda:0", "--checkpoint", str(stage1_root / "stage1/checkpoint.pt"),
                    "--output", str(root / (condition + "_gradient_audit.json"))])
            checked(condition + "_profile", [*training, "--profile"])
            if not specification.get("initial_evaluation_after_training", False):
                checked(condition + "_initial_development", [*inference, "--mode", "predict", *condition_arguments, "--initial"])
            checked(condition + "_training", [*training, "--resume"])
            if specification.get("initial_evaluation_after_training", False):
                checked(condition + "_initial_development", [*inference, "--mode", "predict", *condition_arguments, "--initial"])
            checked(condition + "_development", [*inference, "--mode", "predict", *condition_arguments])
            if "refinement" in condition:
                checked(condition + "_initial_development_pdm", [*inference, "--mode", "score", *condition_arguments, "--initial"], OFFICIAL_PYTHON, cpu=True)
                checked(condition + "_development_pdm", [*inference, "--mode", "score", *condition_arguments], OFFICIAL_PYTHON, cpu=True)
            checked(condition + "_persistent_future", [*inference, "--mode", "predict", *condition_arguments, "--intervention", "persistent_future"])
            if "refinement" in condition:
                checked(condition + "_persistent_future_pdm", [*inference, "--mode", "score", *condition_arguments, "--intervention", "persistent_future"], OFFICIAL_PYTHON, cpu=True)
            checked(condition + "_world_retention", [*validation, "--mode", "world-retention", *condition_arguments])
            checked(condition + "_gate", [*validation, "--mode", "planning-gate", *condition_arguments], cpu=True)
        if specification.get("automatic_navtest", True):
            checked("navtest_inputs", [*inference, "--mode", "prepare-navtest"], OFFICIAL_PYTHON, cpu=True)
            for condition in specification["conditions"]:
                checked(condition + "_navtest", [*inference, "--mode", "predict", "--condition", condition, "--subset", "navtest"])
                checked(condition + "_navtest_score", [*inference, "--mode", "score", "--condition", condition, "--subset", "navtest"], OFFICIAL_PYTHON, cpu=True)
            checked("paired_planning_report", [*inference, "--mode", "summarize"], cpu=True)
        else:
            checked("paired_object_auxiliary_report", ["scripts/summarize_lpwm_object_auxiliary_ablation.py", *config_arguments], cpu=True)
        state["current_stage"] = "complete"
        save_state()
        write_json(root / "queue_completion.json", {"complete": True, "conditions": specification["conditions"], "time_unix": time.time()})
        print("VALIDATED_LPWM_QUEUE_DONE", flush=True)
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        gates = {}
        for path in [root / "stage1_validation_gate.json", root / "teacher_validation_gate.json", *root.glob("*/validation_gate.json")]:
            if path.exists():
                gates[str(path.relative_to(root))] = json.loads(path.read_text())
        write_json(root / "queue_failed.json", {"failed_stage": state["current_stage"], "error": repr(error), "time_unix": time.time(),
            "validation_diagnostics": gates, "dependent_training_blocked": True,
            "recovery_policy": "Preserve checkpoints; diagnose failed criterion; no automatic gate relaxation or unregistered retraining"})
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--join", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    run(arguments)
