"""Compare full-gradient stage1 execution settings without reusing benchmark weights."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / "runtime/environments/lpwm_navsim_adaptation/bin/python"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_navsim_full_posttraining_v2/throughput_profiles"


def main(arguments):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if arguments.detach:
        with (OUTPUT_ROOT / "benchmark.log").open("a") as stream:
            process = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), "--profiles", *arguments.profiles,
                "--report-name", arguments.report_name], cwd=PROJECT_ROOT,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"benchmark_pid": process.pid}), flush=True)
        return
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    reports = []
    execution_root = PROJECT_ROOT / "configs/lpwm_navsim_adaptation/execution"
    for name in arguments.profiles:
        directory = OUTPUT_ROOT / name
        directory.mkdir(exist_ok=True)
        configuration = execution_root / (name + ".json")
        if not (directory / "training_summary.json").exists():
            free_memory = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits"], text=True)
            required = 36 if "batch4" in name else 22
            if min(int(value) / 1024 for value in free_memory.split()) < required:
                reports.append({"name": name, "skipped": "Insufficient GPU memory headroom at launch"})
                continue
            command = [str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=2",
                "scripts/train_lpwm_navtrain_distributed.py", "--config", "configs/lpwm_navsim_adaptation/full_posttraining_v2.json",
                "--profile", "--profile-updates", "8", "--execution-config", str(configuration)]
            with (directory / "console.log").open("a") as stream:
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": "0,1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
                print("PROFILE", name, child.pid, flush=True)
                try:
                    deadline = time.monotonic() + 600
                    while child.poll() is None:
                        memory_fields = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemTotal:', 'MemAvailable:'))}
                        remaining_mib = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                        if memory_fields['MemAvailable'] < 32 * 1024**2 or min(map(int, remaining_mib.split())) < 6 * 1024:
                            child.send_signal(signal.SIGINT)
                            child.wait(timeout=60)
                            raise RuntimeError("Memory reserve reached; worker benchmark stopped")
                        if time.monotonic() > deadline:
                            raise subprocess.TimeoutExpired(command, 600)
                        time.sleep(2)
                    code = child.returncode
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGINT)
                    child.wait(timeout=60)
                    code = 124
            if code:
                reports.append({"name": name, "failed_returncode": code})
                continue
        ranks = [[json.loads(line) for line in (directory / f"rank{rank}_training_log.jsonl").read_text().splitlines()] for rank in (0, 1)]
        # Exclude initial allocator/kernel/worker startup. Both ranks process the same number of clips.
        step_seconds = [max(ranks[0][index]["update_seconds"], ranks[1][index]["update_seconds"])
            for index in range(2, len(ranks[0]))]
        median_seconds = statistics.median(step_seconds)
        reports.append({"name": name, "execution_config": str(configuration), "effective_batch_size": 16,
            "median_update_seconds": median_seconds, "clips_per_second": 16 / median_seconds,
            "median_data_wait_seconds": statistics.median(row["data_wait_seconds"] for rows in ranks for row in rows[2:]),
            "maximum_peak_allocated_gib": max(row["peak_allocated_gib"] for rows in ranks for row in rows),
            "estimated_training_hours_excluding_evaluation": 28920 * median_seconds / 3600,
            "measured_updates": len(step_seconds), "profile_weights_are_discarded": True})
        print(json.dumps(reports[-1]), flush=True)
    eligible = [row for row in reports if "median_update_seconds" in row]
    if not eligible:
        raise RuntimeError("No execution configuration passed")
    fastest = min(eligible, key=lambda row: row["median_update_seconds"])
    result = {"profiles": reports, "recommended": fastest, "note": "Short shared-GPU benchmark; long-run speed and co-tenants can change"}
    (OUTPUT_ROOT / (arguments.report_name + ".json")).write_text(json.dumps(result, indent=2) + "\n")
    print("THROUGHPUT_COMPARISON_DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--profiles", nargs="+", default=["batch2_accumulation4_workers0", "batch4_accumulation2_workers0", "batch4_accumulation2_workers2"])
    parser.add_argument("--report-name", default="comparison")
    main(parser.parse_args())
