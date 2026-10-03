"""Durable stage1 training -> final evaluation -> adaptation gate, without stage mixing."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/lpwm_navsim_adaptation/bin/python"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2) + "\n")
    pending.replace(path)


def supervise(arguments):
    specification = json.loads(arguments.config.read_text())
    output_root = PROJECT_ROOT / specification["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    if arguments.detach:
        if (output_root / "completion.json").exists():
            raise FileExistsError("Completed pipeline cannot be relaunched")
        with (output_root / "supervisor.log").open("a") as stream:
            command = [str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)]
            if arguments.resume:
                command.append("--resume")
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"supervisor_pid": process.pid, "log": str(output_root / "supervisor.log")}), flush=True)
        return
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (output_root / "supervisor.lock").open("a") as lock_stream:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output_root / "completion.json").exists():
            print("PIPELINE_ALREADY_COMPLETE", flush=True)
            return
        profile = json.loads((output_root / "profile/training_summary.json").read_text())
        assert profile["profile_only"] and profile["completed_updates"] == 3
        assert all(value["norm"] > 0 for value in profile["module_gradients"].values())
        profile_evaluation = json.loads((output_root / "evaluation/profile_published/metrics.json").read_text())
        assert profile_evaluation["profile_only"]
        source_paths = [
            "scripts/prepare_lpwm_navsim_posttraining.py", "scripts/run_lpwm_navsim_posttraining.py",
            "scripts/visualize_lpwm_posttraining_progress.py", "scripts/summarize_lpwm_posttraining.py",
            "scripts/launch_lpwm_posttraining.py", "src/planning_aware_future_prediction/object_centric/lpwm_bridge.py",
            "scripts/prepare_lpwm_driving_risk_diagnostics.py", "configs/lpwm_navsim_adaptation/driving_risks_v1.json",
            "configs/lpwm_planning/postadaptation_joint_v1.json",
        ]
        provenance = {"configuration_sha256": digest(arguments.config),
            "manifest_sha256": digest(output_root / "manifest.json"),
            "initial_checkpoint_sha256": profile["initial_checkpoint_sha256"],
            "source_sha256": {name: digest(PROJECT_ROOT / name) for name in source_paths},
            "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "profile": profile, "gpu_training": 0, "gpu_published_evaluation": 1,
            "stage2_requires_adaptation_gate": True,
            "downstream_direction": "merged stage2: low learning rate LPWM fine-tuning and full planner training"}
        registration_path = output_root / "registration.json"
        if registration_path.exists():
            previous = json.loads(registration_path.read_text())
            assert previous["configuration_sha256"] == provenance["configuration_sha256"]
            assert previous["source_sha256"] == provenance["source_sha256"]
        else:
            write_json(registration_path, provenance)
        children = []
        log_streams = []
        def launch(name, command):
            stream = (output_root / (name + ".log")).open("a")
            log_streams.append(stream)
            process = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT,
                env={**os.environ, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"},
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            children.append((name, process))
            write_json(output_root / "active_processes.json", {"supervisor_pid": os.getpid(),
                "processes": [{"name": label, "pid": child.pid, "returncode": child.poll()} for label, child in children]})
            return process
        def stop_children(_signal, _frame):
            for _, process in children:
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
            write_json(output_root / "pipeline_stopped.json", {"reason": "user_signal", "signal": _signal})
            raise SystemExit(130)
        signal.signal(signal.SIGINT, stop_children)
        signal.signal(signal.SIGTERM, stop_children)
        runner = str(PROJECT_ROOT / "scripts/run_lpwm_navsim_posttraining.py")
        config_arguments = ["--config", str(arguments.config)]
        initial_evaluation = launch("published_evaluation", [runner, *config_arguments, "--mode", "evaluate", "--gpu", "1", "--name", "published"])
        training_arguments = [runner, *config_arguments, "--mode", "train", "--gpu", "0"]
        if arguments.resume:
            training_arguments.append("--resume")
        training = launch("stage1_training", training_arguments)
        training_code = training.wait()
        if training_code:
            if initial_evaluation.poll() is None:
                initial_evaluation.send_signal(signal.SIGINT)
            write_json(output_root / "pipeline_stopped.json", {"reason": "training_failed_or_stopped", "returncode": training_code})
            raise SystemExit(training_code)
        if initial_evaluation.wait():
            write_json(output_root / "pipeline_stopped.json", {"reason": "published_evaluation_failed"})
            raise SystemExit(1)
        final_evaluation = launch("posttrained_evaluation", [runner, *config_arguments, "--mode", "evaluate", "--gpu", "0", "--name", "posttrained", "--checkpoint", str(output_root / "stage1/checkpoint.pt")])
        if final_evaluation.wait():
            write_json(output_root / "pipeline_stopped.json", {"reason": "posttrained_evaluation_failed"})
            raise SystemExit(1)
        report = launch("adaptation_report", [str(PROJECT_ROOT / "scripts/summarize_lpwm_posttraining.py"), *config_arguments])
        if report.wait():
            raise RuntimeError("Adaptation report failed")
        gate = json.loads((output_root / "adaptation_gate.json").read_text())
        write_json(output_root / "completion.json", {"stage1_completed": True,
            "adaptation_gate_passed": gate["adaptation_gate_passed"], "stage2_status": gate["stage2_status"],
            "unmet_criteria": gate["unmet_criteria"], "time_unix": time.time()})
        print("STAGE1_PIPELINE_DONE", json.dumps({"gate_passed": gate["adaptation_gate_passed"], "stage2_status": gate["stage2_status"]}), flush=True)
        for stream in log_streams:
            stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/posttraining_v1.json")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    supervise(arguments)
