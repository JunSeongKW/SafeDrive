"""Gracefully pause only registered, UID/command/shard-verified evaluation workers."""
import datetime
import json
import os
import signal
from pathlib import Path

import psutil


def main():
    workspace = Path(__file__).resolve().parents[1]
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    report_path = output_root / "kjs_process_label_pause.json"
    if report_path.exists():
        raise RuntimeError("Previous pause report exists; preserve it before another pause")
    manifest = json.loads((output_root / "parallel_worker_manifest.json").read_text())
    expected_script = str(workspace / "scripts/evaluate_official_wa_jepa.py")
    verified_processes = []
    verified_workers = []
    for worker in manifest["registered_workers"]:
        try:
            process = psutil.Process(worker["pane_pid"])
            arguments = process.cmdline()
            if process.uids().real != os.getuid() or expected_script not in arguments:
                continue
            if arguments[arguments.index("--execution-shard") + 1] != str(worker["shard"]):
                continue
            if process.environ().get("CUDA_VISIBLE_DEVICES") != str(worker["physical_gpu"]):
                continue
            if process.pid in {candidate.pid for candidate in verified_processes}:
                continue
            verified_processes.append(process)
            verified_workers.append({**worker, "command_before_pause": arguments})
        except (psutil.Error, ValueError):
            continue
    for process in verified_processes:
        process.send_signal(signal.SIGINT)
    gone, alive = psutil.wait_procs(verified_processes, timeout=30)
    if alive:
        raise RuntimeError(f"Workers have not stopped; no force-kill or restart: {[process.pid for process in alive]}")
    scene_paths = sorted(output_root.glob("dense_full_scene_records*.jsonl"))
    # Exclude the smoke file and keep every earlier full-evaluation record intact.
    records = []
    file_counts = {}
    for path in scene_paths:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        file_counts[path.name] = len(rows)
        records.extend(rows)
    unique_tokens = {record["token"] for record in records}
    if len(unique_tokens) != len(records):
        raise RuntimeError("Duplicate saved full-evaluation scene; do not restart silently")
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "User-requested process-label-only pause/resume; original Conda prefix preserved",
        "stopped_verified_workers": verified_workers,
        "stopped_worker_count": len(gone),
        "completed_unique_scene_count": len(unique_tokens),
        "failed_scene_count": sum(not record["valid"] for record in records),
        "scene_record_counts_by_file": file_counts,
        "completed_scene_tokens": sorted(unique_tokens),
        "resume_policy": "Same14shards/config/seed/weights; skip completed tokens, recompute only any interrupted scene",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("timestamp_utc", "stopped_worker_count", "completed_unique_scene_count", "failed_scene_count")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
