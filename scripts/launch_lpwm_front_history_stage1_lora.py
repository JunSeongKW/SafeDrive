"""Detach the authorized Stage2 queue after data/gradient/RGB admission checks."""
import json
import os
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    study_root = PROJECT_ROOT / "outputs/lpwm_front_history_stage1_lora_v1"
    admission = json.loads((study_root / "admission_report.json").read_text())
    assert admission["passed"] and admission["main_initializes_from_stage1_not_audit"]
    assert not (study_root / "pause.requested").exists()
    python = PROJECT_ROOT / "runtime/environments/kjs-lpwm-drivor-joint/bin/python"
    command = [str(python), "-u", "scripts/queue_lpwm_front_history_stage1_lora.py", "--config",
               str(PROJECT_ROOT / "configs/lpwm_front_history_stage1_lora/experiment.json"), "--execution",
               str(PROJECT_ROOT / "configs/lpwm_front_history_stage1_lora/execution_microbatch8.json")]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "PYTHONPATH": str(PROJECT_ROOT / "src"),
                   "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (study_root / "queue_launch.log").open("a") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    report = {"queue_pid": process.pid, "command": command, "new_stage2_from_completed_stage1": True,
              "historical_adapter_and_joint_remain_paused": True, "gpus": [0, 1]}
    (study_root / "launch.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
