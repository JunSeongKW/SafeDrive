"""Sequential epoch training -> fixed dev validation -> next epoch, resumable."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPU_PYTHON = PROJECT_ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"
ORACLE_ENVIRONMENT = PROJECT_ROOT / "runtime/environments/drive_jepa_official_evaluation"


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def run(arguments):
    import fcntl
    configuration = json.loads(arguments.config.read_text())
    root = PROJECT_ROOT / configuration["output_directory"]
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "queue.pid").write_text(str(os.getpid()) + "\n")
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
                   "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONPATH": str(PROJECT_ROOT / "src")}
    passed = json.loads((root / "ddp_engineering/execution_microbatch8/passed.json").read_text())
    assert passed["passed"] and passed["updates"] == 2
    audit = json.loads((root / "ddp_engineering/execution_microbatch8/validation/update_000002/prediction_complete.json").read_text())
    assert audit["complete"] and audit["frozen_native_unchanged"]

    def paused():
        return (root / "pause.requested").exists() or (root.parent / "pause.requested").exists()

    def execute(label, command, execution_environment):
        with (root / f"{label}.log").open("a") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            write_json(root / "queue_state.json", {"status": label, "queue_pid": os.getpid(),
                "active_pid": process.pid, "command": command, "started_at_unix": time.time()})
            result = process.wait()
        if result:
            write_json(root / "queue_failed.json", {"step": label, "exit_code": result, "log": str(root / f"{label}.log")})
            raise RuntimeError(f"{label} failed with exit code {result}")

    for epoch in range(1, configuration["epochs"] + 1):
        if paused():
            write_json(root / "queue_state.json", {"status": "paused", "queue_pid": os.getpid()})
            return
        epoch_checkpoint = root / f"epoch_{epoch:02d}.pt"
        if not epoch_checkpoint.exists():
            execute(f"training_epoch_{epoch:02d}", [str(GPU_PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                "scripts/train_lpwm_front_history_stage1_lora.py", "--config", str(arguments.config),
                "--execution", str(arguments.execution), "--stop-after-epoch"], environment)
            if paused() or ((root / "paused.json").exists() and not epoch_checkpoint.exists()):
                write_json(root / "queue_state.json", {"status": "paused", "epoch": epoch})
                return
            assert epoch_checkpoint.exists(), "Epoch checkpoint absent; cannot proceed to validation"
        updates_per_epoch = (75297 + configuration["effective_batch"] - 1) // configuration["effective_batch"]
        validation_root = root / "validation" / f"update_{epoch*updates_per_epoch:06d}"
        if not (validation_root / "prediction_complete.json").exists():
            execute(f"validation_prediction_epoch_{epoch:02d}", [str(GPU_PYTHON), "-u",
                "scripts/validate_lpwm_front_history_stage1_lora.py", "--mode", "predict", "--config", str(arguments.config),
                "--checkpoint", str(epoch_checkpoint)], environment | {"CUDA_VISIBLE_DEVICES": "0"})
        if not (validation_root / "validation_complete.json").exists():
            execute(f"validation_score_epoch_{epoch:02d}", [str(ORACLE_ENVIRONMENT / "bin/python"), "-u",
                "scripts/validate_lpwm_front_history_stage1_lora.py", "--mode", "score",
                "--predictions", str(validation_root / "predictions.npz")], environment | {
                    "CUDA_VISIBLE_DEVICES": "", "LD_LIBRARY_PATH": str(ORACLE_ENVIRONMENT / "lib"),
                    "PYTHONPATH": str(PROJECT_ROOT / "reference_repositories/DrivoR")})
        validation = json.loads((validation_root / "validation_complete.json").read_text())
        assert validation["passed_execution_and_finiteness"] and validation["pdms_count"] == 1021
        write_json(root / "latest_validation.json", {"epoch": epoch, "completed_updates": epoch*updates_per_epoch,
            "validation": str(validation_root), "pdms": validation["pdms"], "ade_meters": validation["ade_meters"],
            "fde_meters": validation["fde_meters"], "learning_quality_requires_review": True})
    write_json(root / "training_complete.json", {"epochs": configuration["epochs"], "completed_updates": configuration["epochs"]*updates_per_epoch,
        "epoch_validations_complete": True, "checkpoint": str(root / "latest.pt")})
    write_json(root / "queue_state.json", {"status": "complete", "queue_pid": os.getpid()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    run(parser.parse_args())
