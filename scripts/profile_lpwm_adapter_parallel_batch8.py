"""Bound a disposable Adapter batch8 profile alongside the resumed primary run."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/lpwm_adapter_original_batch_rebalance_v1"


def memory():
    lines = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used",
                                     "--format=csv,noheader,nounits"], text=True).splitlines()
    return {int(line.split(",")[0]): int(line.split(",")[1]) * 1024**2 for line in lines}


if __name__ == "__main__":
    primary = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1/progress.json"
    deadline = time.monotonic() + 900
    while True:
        state = json.loads(primary.read_text())
        if state["completed_updates"] > 5084:
            assert state["execution_overrides"]["microbatch_per_gpu"] == 4
            if max(memory().values()) < 21_000_000_000:
                break
        assert time.monotonic() < deadline
        time.sleep(2)
    destination = OUTPUT / "adapter_batch8_parallel_profile"
    command = [str(ROOT / "runtime/environments/kjs-lpwm-stage2/bin/python"), "-m", "torch.distributed.run",
               "--standalone", "--nproc_per_node=2", "scripts/train_lpwm_adapter_original_batch.py",
               "--config", "configs/lpwm_adapter_original_batch_rebalance/adapter_batch8.json",
               "--target-epoch", "2", "--profile-updates", "4", "--profile-output", str(destination)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "2",
                   "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    with (OUTPUT / "adapter_batch8_parallel_profile.log").open("w") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        peak = memory()
        stopped = False
        while child.poll() is None:
            current = memory()
            peak = {index: max(peak[index], current[index]) for index in (0, 1)}
            if max(current.values()) >= 46_500_000_000 or time.monotonic() >= deadline:
                os.killpg(child.pid, signal.SIGKILL)
                stopped = True
                break
            time.sleep(.5)
        result = child.wait(timeout=30)
    report = {"passed": result == 0 and not stopped, "maximum_card_used_bytes": peak,
              "returncode": result, "guard_stopped": stopped, "profile_weights_discarded": True}
    (OUTPUT / "parallel_profile_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    assert report["passed"]
