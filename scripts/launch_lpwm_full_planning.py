"""Durable, gated merged-stage2 training and full planning evaluation."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from evaluate_lpwm_full_planning import digest, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/lpwm_navsim_adaptation/bin/python"
OFFICIAL_PYTHON = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation/bin/python"


def main(arguments):
    route_path = PROJECT_ROOT / "configs/lpwm_planning/active_pipeline.json"
    if route_path.exists():
        route = json.loads(route_path.read_text())
        if route["enabled"] and arguments.config.resolve() == (PROJECT_ROOT / route["superseded_config"]).resolve():
            # The already running stage1 parent will reach this code later.
            # Join the registered queue; never launch the superseded planner.
            os.execv(str(PYTHON), [str(PYTHON), str(PROJECT_ROOT / "scripts/queue_lpwm_validated_training.py"),
                "--config", str(PROJECT_ROOT / route["active_config"]), "--join"])
    configuration = json.loads(arguments.config.read_text())
    output_root = PROJECT_ROOT / configuration["output_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    stage1_configuration = json.loads((PROJECT_ROOT / configuration["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_configuration["output_directory"]
    gate = json.loads((stage1_root / "adaptation_gate.json").read_text())
    assert gate["adaptation_gate_passed"], "Do not start planning before successful adaptation"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    children, streams = [], []
    def start(name, command, interpreter=PYTHON):
        stream = (output_root / (name + ".log")).open("a")
        streams.append(stream)
        child = subprocess.Popen([str(interpreter), *command], cwd=PROJECT_ROOT, env=environment,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(child)
        write_json(output_root / "active_stage.json", {"stage": name, "child_pid": child.pid, "supervisor_pid": os.getpid(), "time_unix": time.time()})
        print("STAGE2_START", name, child.pid, flush=True)
        return child
    def checked(name, command, interpreter=PYTHON):
        child = start(name, command, interpreter)
        code = child.wait()
        if code:
            raise RuntimeError(f"{name} failed with {code}; no automatic training retry")
    def stop(signal_number, _frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        write_json(output_root / "stopped.json", {"reason": "signal", "signal": signal_number})
        raise SystemExit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    with (output_root / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            # Engineering checks use the actual adapted weight, not benchmark weights.
            checked("planning_gradient_and_causality_audit", ["scripts/audit_lpwm_planning_finetuning.py", "--device", "cuda:0",
                "--checkpoint", str(stage1_root / "stage1/checkpoint.pt"), "--output", str(output_root / "adapted_gradient_audit.json")])
            audit = json.loads((output_root / "adapted_gradient_audit.json").read_text())
            assert audit["future_target_intervention_max_prediction_difference"] < 1e-5
            sources = ["scripts/train_lpwm_full_planning.py", "scripts/visualize_lpwm_planning_progress.py",
                "scripts/evaluate_lpwm_full_planning.py", "scripts/launch_lpwm_full_planning.py",
                "src/planning_aware_future_prediction/object_centric/lpwm_planning_finetuning.py"]
            registration = {"configuration_sha256": digest(arguments.config), "stage1_checkpoint_sha256": digest(stage1_root / "stage1/checkpoint.pt"),
                "source_sha256": {source: digest(PROJECT_ROOT / source) for source in sources},
                "planning_manifest_sha256": digest(stage1_root / "planning_manifest.json")}
            registration_path = output_root / "registration.json"
            if registration_path.exists():
                assert json.loads(registration_path.read_text()) == registration
            else:
                write_json(registration_path, registration)
            distributed = ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2"]
            for condition in configuration["conditions"]:
                training_command = [*distributed, "scripts/train_lpwm_full_planning.py", "--config", str(arguments.config), "--condition", condition]
                checked(condition + "_profile", [*training_command, "--profile"])
                checked(condition + "_full_training", [*training_command, "--resume"])
                assert (output_root / condition / "training_summary.json").exists()
            evaluation_prefix = ["scripts/evaluate_lpwm_full_planning.py", "--config", str(arguments.config)]
            checked("prepare_navtest_inputs", [*evaluation_prefix, "--mode", "prepare-navtest"], OFFICIAL_PYTHON)
            for subset, intervention in (("development", "none"), ("development", "persistent_future"), ("navtest", "none")):
                jobs = [start(f"{condition}_{subset}_{intervention}", [*evaluation_prefix, "--mode", "predict",
                    "--condition", condition, "--subset", subset, "--gpu", str(gpu), "--intervention", intervention])
                    for gpu, condition in enumerate(configuration["conditions"])]
                if any(child.wait() for child in jobs):
                    raise RuntimeError("Full planning inference failed")
            for subset in ("development", "navtest"):
                jobs = [start(f"{condition}_{subset}_pdm", [*evaluation_prefix, "--mode", "score", "--condition", condition,
                    "--subset", subset], OFFICIAL_PYTHON) for condition in configuration["conditions"]]
                if any(child.wait() for child in jobs):
                    raise RuntimeError("Official PDM scoring failed")
            checked("planning_report", [*evaluation_prefix, "--mode", "summarize"])
            write_json(output_root / "completion.json", {"complete": True, "time_unix": time.time(), "conditions": configuration["conditions"]})
            print("MERGED_STAGE2_DONE", flush=True)
        except Exception as error:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
            write_json(output_root / "stopped.json", {"error": repr(error), "time_unix": time.time()})
            raise
        finally:
            for stream in streams:
                stream.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/lpwm_planning/full_joint_training_v1.json")
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    main(arguments)
