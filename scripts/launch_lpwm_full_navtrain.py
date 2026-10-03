"""Durable full-navtrain stage1 -> diagnostics -> conditional merged-stage2 pipeline."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-stage1/bin/python"
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from launch_lpwm_posttraining import digest, write_json


def run(arguments):
    configuration = json.loads(arguments.config.read_text())
    output_root = PROJECT_ROOT / configuration["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    if arguments.detach:
        with (output_root / "pipeline.log").open("a") as stream:
            command = [str(PYTHON), str(Path(__file__).resolve()), "--config", str(arguments.config)]
            if arguments.resume:
                command.append("--resume")
            if arguments.execution_config:
                command.extend(["--execution-config", str(arguments.execution_config)])
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pipeline_pid": process.pid, "log": str(output_root / "pipeline.log")}), flush=True)
        return
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    with (output_root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        children, streams = [], []
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
        def start(name, command):
            stream = (output_root / (name + ".log")).open("a")
            streams.append(stream)
            process = subprocess.Popen([str(PYTHON), *command], cwd=PROJECT_ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((name, process))
            write_json(output_root / "active_stage.json", {"stage": name, "supervisor_pid": os.getpid(), "time_unix": time.time(),
                "processes": [{"name": label, "pid": child.pid, "returncode": child.poll()} for label, child in children]})
            print("START", name, process.pid, flush=True)
            return process
        def checked(name, command):
            process = start(name, command)
            code = process.wait()
            if code:
                raise RuntimeError(f"{name} stopped with code {code}; no automatic retry")
        def stop(_signal, _frame):
            for _, process in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            write_json(output_root / "pipeline_stopped.json", {"reason": "user_signal", "signal": _signal})
            raise SystemExit(130)
        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        config_arguments = ["--config", str(arguments.config)]
        distributed_prefix = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
        try:
            if not (output_root / "manifest.json").exists():
                checked("prepare_rgb", ["scripts/prepare_lpwm_navsim_posttraining.py", *config_arguments])
            risks = start("prepare_driving_risks", ["scripts/prepare_lpwm_driving_risk_diagnostics.py", *config_arguments])
            targets = start("prepare_planning_targets", ["scripts/prepare_lpwm_full_planning_targets.py", *config_arguments])
            if not (output_root / "distributed_profile/training_summary.json").exists():
                checked("distributed_profile", [*distributed_prefix, "scripts/train_lpwm_navtrain_distributed.py", *config_arguments, "--profile"])
            checked("before_training_visualization", ["scripts/run_lpwm_navsim_posttraining.py", *config_arguments, "--mode", "visualize", "--gpu", "0"])
            if risks.wait():
                raise RuntimeError("Driving risk metadata preparation failed")
            checked("causal_evaluation_profile", ["scripts/run_lpwm_navsim_posttraining.py", *config_arguments,
                "--mode", "evaluate", "--gpu", "0", "--profile", "--name", "profile_published"])
            source_paths = ["scripts/train_lpwm_navtrain_distributed.py", "scripts/run_lpwm_navsim_posttraining.py",
                "scripts/prepare_lpwm_navsim_posttraining.py", "scripts/prepare_lpwm_driving_risk_diagnostics.py",
                "scripts/visualize_lpwm_posttraining_progress.py", "scripts/summarize_lpwm_posttraining.py",
                "scripts/prepare_lpwm_full_planning_targets.py", "scripts/launch_lpwm_full_navtrain.py",
                "scripts/lpwm_prefetched_video_batches.py",
                "configs/lpwm_navsim_adaptation/driving_risks_v1.json"]
            registration = {"configuration": configuration, "configuration_sha256": digest(arguments.config),
                "manifest_sha256": digest(output_root / "manifest.json"),
                "source_sha256": {path: digest(PROJECT_ROOT / path) for path in source_paths},
                "initial_checkpoint_sha256": json.loads((output_root / "distributed_profile/training_summary.json").read_text())["initial_checkpoint_sha256"],
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
                "latest_user_scope": "full navtrain available inputs, epoch-based stage1 and low-LR LPWM/full-planner merged stage2"}
            registration_path = output_root / "registration.json"
            if registration_path.exists():
                previous = json.loads(registration_path.read_text())
                assert previous["configuration_sha256"] == registration["configuration_sha256"]
                expected_sources = previous["source_sha256"]
                # Preserve the original registration and require an explicit,
                # auditable hash chain for engineering corrections.
                for amendment_path in sorted((output_root / "source_amendments").glob("*.json")):
                    amendment = json.loads(amendment_path.read_text())
                    assert amendment["previous_source_sha256"] == expected_sources
                    assert amendment["configuration_sha256"] == previous["configuration_sha256"]
                    expected_sources = amendment["updated_source_sha256"]
                assert expected_sources == registration["source_sha256"], "Unregistered source changes"
            else:
                write_json(registration_path, registration)
            training_command = [*distributed_prefix, "scripts/train_lpwm_navtrain_distributed.py", *config_arguments]
            if arguments.resume:
                training_command.append("--resume")
            if arguments.execution_config:
                training_command.extend(["--execution-config", str(arguments.execution_config)])
            checked("stage1_full_training", training_command)
            assert (output_root / "stage1/training_summary.json").exists(), "Stage1 stopped before completion"
            initial = start("published_full_evaluation", ["scripts/run_lpwm_navsim_posttraining.py", *config_arguments,
                "--mode", "evaluate", "--gpu", "0", "--name", "published"])
            adapted = start("posttrained_full_evaluation", ["scripts/run_lpwm_navsim_posttraining.py", *config_arguments,
                "--mode", "evaluate", "--gpu", "1", "--name", "posttrained", "--checkpoint", str(output_root / "stage1/checkpoint.pt")])
            if initial.wait() or adapted.wait():
                raise RuntimeError("Stage1 full evaluation failed")
            checked("stage1_adaptation_report", ["scripts/summarize_lpwm_posttraining.py", *config_arguments])
            gate = json.loads((output_root / "adaptation_gate.json").read_text())
            write_json(output_root / "stage1_completion.json", {"stage1_complete": True, "adaptation_gate_passed": gate["adaptation_gate_passed"],
                "unmet_criteria": gate["unmet_criteria"], "time_unix": time.time()})
            if not gate["adaptation_gate_passed"]:
                write_json(output_root / "completion.json", {"stage1_complete": True, "stage2_started": False,
                    "reason": "adaptation criteria not met", "unmet_criteria": gate["unmet_criteria"]})
                print("STAGE1_DONE_STAGE2_GATE_NOT_MET", flush=True)
                return
            if targets.wait():
                raise RuntimeError("Planning targets failed official-target validation")
            checked("stage2_full_training_and_evaluation", ["scripts/launch_lpwm_full_planning.py",
                "--config", "configs/lpwm_planning/full_joint_training_v1.json"])
            write_json(output_root / "completion.json", {"stage1_complete": True, "stage2_complete": True, "time_unix": time.time()})
            print("FULL_LPWM_PIPELINE_DONE", flush=True)
        except Exception as error:
            for _, process in children:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            write_json(output_root / "pipeline_stopped.json", {"reason": repr(error), "time_unix": time.time()})
            raise
        finally:
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_navsim_adaptation/full_posttraining_v2.json")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execution-config", type=Path)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    if arguments.execution_config:
        arguments.execution_config = arguments.execution_config.resolve()
    run(arguments)
