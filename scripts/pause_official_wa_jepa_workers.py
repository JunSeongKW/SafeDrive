"""Gracefully pause only registered, UID/command/shard-verified evaluation workers."""
import argparse
import datetime
import hashlib
import json
import os
import signal
from pathlib import Path

import psutil


def verify_preserved_pause_state(workspace, pause_report):
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    expected_tokens = set(json.loads((workspace / "results/official_wa_jepa_reproduction/expected_scene_tokens.json").read_text())["tokens"])
    preserved_tokens = set()
    for filename, expected_checksum in pause_report["scene_record_sha256_by_file"].items():
        saved_bytes = (output_root / filename).read_bytes()
        if hashlib.sha256(saved_bytes).hexdigest() != expected_checksum:
            raise RuntimeError(f"Saved evaluation file changed since pause: {filename}")
        rows = [json.loads(line) for line in saved_bytes.splitlines() if line.strip()]
        if len(rows) != pause_report["scene_record_counts_by_file"][filename]:
            raise RuntimeError(f"Saved row count changed: {filename}")
        for row in rows:
            if row["token"] in preserved_tokens:
                raise RuntimeError("Duplicate saved scene")
            preserved_tokens.add(row["token"])
    if preserved_tokens != set(pause_report["completed_scene_tokens"]) or preserved_tokens - expected_tokens:
        raise RuntimeError("Pause inventory does not match expected scene IDs")
    shared_summary = {
        "status": "user_paused_no_automatic_resume",
        "pause_timestamp_utc": pause_report["timestamp_utc"],
        "reason": pause_report["reason"],
        "expected_scene_count": len(expected_tokens),
        "preserved_completed_scene_count": len(preserved_tokens),
        "remaining_scene_count": len(expected_tokens - preserved_tokens),
        "completion_percent": 100 * len(preserved_tokens) / len(expected_tokens),
        "failed_scene_count": pause_report["failed_scene_count"],
        "duplicate_scene_count": 0,
        "saved_file_sha256_verified": True,
        "scene_record_counts_by_file": pause_report["scene_record_counts_by_file"],
        "scene_record_sha256_by_file": pause_report["scene_record_sha256_by_file"],
        "configuration_sha256": hashlib.sha256((workspace / "configs/official_wa_jepa/reproduction_v1.json").read_bytes()).hexdigest(),
        "preserved_execution_shards": 14,
        "resume_requires_explicit_user_request_and_renewed_gpu_allocation": True,
        "resume_policy": pause_report["resume_policy"],
        "local_inventory": str(output_root / "evaluation_pause.json"),
        "original_conda_prefix": "runtime/environments/wa_jepa_official_evaluation",
        "python_entrypoint": "/rhome/junseong/envs/kjs-wa-jepa-eval/bin/python",
        "full_evaluation_complete": False,
    }
    shared_path = workspace / "results/official_wa_jepa_reproduction/paused_evaluation_state.json"
    shared_path.write_text(json.dumps(shared_summary, indent=2) + "\n")
    print(json.dumps({key: shared_summary[key] for key in ("status", "preserved_completed_scene_count", "remaining_scene_count", "completion_percent", "saved_file_sha256_verified")}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reason", default="user_requested_gpu_release")
    parser.add_argument("--verify-only", action="store_true", help="Verify the recorded pause inventory; never signal or launch processes")
    pause_arguments = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    if pause_arguments.verify_only:
        marker = json.loads((output_root / "evaluation_pause.json").read_text())
        verify_preserved_pause_state(workspace, json.loads(Path(marker["pause_report"]).read_text()))
        return
    requested_at = datetime.datetime.now(datetime.timezone.utc)
    report_path = output_root / f"evaluation_pause_{requested_at.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    if report_path.exists():
        raise RuntimeError("Previous pause report exists; preserve it before another pause")
    pause_marker = output_root / "evaluation_pause.json"
    previous_pause_marker = json.loads(pause_marker.read_text()) if pause_marker.exists() else None
    pause_marker.write_text(json.dumps({"user_requested_pause": True, "requested_at_utc": requested_at.isoformat(), "reason": pause_arguments.reason, "resume_requires_explicit_user_request": True, "pause_report": str(report_path)}, indent=2) + "\n")
    manifest = json.loads((output_root / "parallel_worker_manifest.json").read_text())
    expected_script = str(workspace / "scripts/evaluate_official_wa_jepa.py")
    verified_processes = []
    verified_workers = []
    for worker in manifest["registered_workers"]:
        try:
            process = psutil.Process(worker["pane_pid"])
            worker_arguments = process.cmdline()
            if process.uids().real != os.getuid() or expected_script not in worker_arguments:
                continue
            if worker_arguments[worker_arguments.index("--execution-shard") + 1] != str(worker["shard"]):
                continue
            if process.environ().get("CUDA_VISIBLE_DEVICES") != str(worker["physical_gpu"]):
                continue
            if process.pid in {candidate.pid for candidate in verified_processes}:
                continue
            verified_processes.append(process)
            verified_workers.append({**worker, "command_before_pause": worker_arguments})
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
    file_checksums = {}
    for path in scene_paths:
        saved_bytes = path.read_bytes()
        rows = [json.loads(line) for line in saved_bytes.splitlines() if line.strip()]
        file_counts[path.name] = len(rows)
        file_checksums[path.name] = hashlib.sha256(saved_bytes).hexdigest()
        records.extend(rows)
    unique_tokens = {record["token"] for record in records}
    if len(unique_tokens) != len(records):
        raise RuntimeError("Duplicate saved full-evaluation scene; do not restart silently")
    report = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "purpose": "User-requested evaluation pause; GPU0/1 released and completed scenes preserved",
        "reason": pause_arguments.reason,
        "previous_pause_marker": previous_pause_marker,
        "resume_requires_explicit_user_request": True,
        "stopped_verified_workers": verified_workers,
        "stopped_worker_count": len(gone),
        "completed_unique_scene_count": len(unique_tokens),
        "failed_scene_count": sum(not record["valid"] for record in records),
        "scene_record_counts_by_file": file_counts,
        "scene_record_sha256_by_file": file_checksums,
        "completed_scene_tokens": sorted(unique_tokens),
        "resume_policy": "Same14shards/config/seed/weights; skip completed tokens, recompute only any interrupted scene",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("timestamp_utc", "stopped_worker_count", "completed_unique_scene_count", "failed_scene_count")}, indent=2), flush=True)
    verify_preserved_pause_state(workspace, report)


if __name__ == "__main__":
    main()
