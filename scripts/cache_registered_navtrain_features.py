"""Config/manifest-based cache. Fail CLOSED on unresolved image provenance or GPU occupancy."""

import argparse
import json
import pickle
import resource
import shutil
import subprocess
import time
from pathlib import Path

import torch
from cache_target_supervision_features import cache_window
from train_target_supervision_ablation import PROJECT_ROOT, file_sha256

from planning_aware_future_prediction.models.frozen_driving_video_encoder import (
    FrozenDrivingVideoEncoder,
)


def approved_gpu_is_unoccupied(physical_gpu):
    try:
        gpu_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).splitlines()
        gpu_row = next(
            row.split(",") for row in gpu_rows if int(row.split(",")[0]) == physical_gpu
        )
        uuid = gpu_row[1].strip()
        # Graphics processes (e.g. CARLA) may not appear in compute-apps.
        # Conservative guard: neither utilization=0 nor no compute-apps is enough.
        if int(gpu_row[2].strip()) > 256:
            return False
        process_rows = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).splitlines()
        return not any(row.split(",")[0].strip() == uuid for row in process_rows)
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configuration",
        type=Path,
        default=PROJECT_ROOT / "configs/exploration/pilot_foundation_decision_v1.json",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preprocessing-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--completed-profile", type=Path)
    parser.add_argument("--physical-gpu", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    directory = arguments.output_directory.resolve()
    if directory.exists() or not directory.is_relative_to(PROJECT_ROOT / "outputs"):
        raise ValueError("new project output required")
    directory.mkdir(parents=True)
    configuration = json.loads(arguments.configuration.read_text())["expanded_data"]
    manifest = json.loads(arguments.manifest.read_text())
    evidence = json.loads(arguments.preprocessing_evidence.read_text())
    blockers = []
    status_path = (
        PROJECT_ROOT
        / "configs/exploration/pilot_foundation_decision_followup_status.json"
    )
    if status_path.exists() and not json.loads(status_path.read_text()).get(
        "expanded_pilot_execution_authorized", False
    ):
        blockers.append(
            "latest user priority paused expanded pilot; a new authorized protocol is required"
        )
    if evidence.get("resolved") is not True or evidence.get("image_mode") not in (
        "stored_pinhole",
        "undistort_stored_brown_conrady",
    ):
        blockers.append(
            "stored-JPEG geometry/provenance unresolved; no encoder profile or expanded cache"
        )
    if (
        manifest.get("recording_counts", {}).get("development", 0)
        < configuration["minimum_development_groups"]
    ):
        blockers.append("not enough nonempty independent development groups")
    if shutil.disk_usage(directory).free < configuration["minimum_free_disk_bytes"]:
        blockers.append("registered free-disk guard failed")
    if not arguments.profile_only:
        if not arguments.completed_profile:
            blockers.append("full cache requires completed measured 200-window profile")
        else:
            profile = json.loads(arguments.completed_profile.read_text())
            if profile.get("windows") != configuration[
                "profile_windows"
            ] or not profile.get("within_registered_bounds"):
                blockers.append("invalid/incomplete measured profile")
            if profile.get("manifest_sha256") != file_sha256(
                arguments.manifest
            ) or profile.get("preprocessing_evidence_sha256") != file_sha256(
                arguments.preprocessing_evidence
            ):
                blockers.append("profile source/split/preprocessing differs")
    if not blockers and not approved_gpu_is_unoccupied(arguments.physical_gpu):
        blockers.append(
            "approved GPU occupied or inaccessible; never take another GPU or stop another process"
        )
    gate = {
        "can_execute": not blockers,
        "blockers": blockers,
        "cache_generated": False,
        "encoder_profile_executed": False,
        "held_out_model_evaluation_performed": False,
    }
    (directory / "execution_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    if blockers:
        print(json.dumps(gate, indent=2), flush=True)
        return
    started = time.perf_counter()
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(arguments.physical_gpu)
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    encoder = FrozenDrivingVideoEncoder(PROJECT_ROOT, device)
    records = manifest["records"]
    if arguments.profile_only:
        records = [record for record in records if record["split"] == "train"][
            : configuration["profile_windows"]
        ]
    if arguments.profile_only and len(records) != configuration["profile_windows"]:
        raise RuntimeError("less than registered 200 profile windows")
    log_root = PROJECT_ROOT / "dataset/navsim_logs" / configuration["source_split"]
    sensor_root = PROJECT_ROOT / "dataset/sensor_blobs" / configuration["source_split"]
    indexed, image_hashes, loaded_segment, frames = [], {}, None, None
    disk_bytes = 0
    limit = (
        configuration["maximum_profile_seconds"]
        if arguments.profile_only
        else configuration["maximum_cache_seconds"]
    )
    for record in records:
        if time.perf_counter() - started > limit:
            raise RuntimeError(
                "registered cache time cap; preserve indexed partial output, no automatic continuation"
            )
        if loaded_segment != record["segment_filename"]:
            log_path = log_root / record["segment_filename"]
            if (
                file_sha256(log_path)
                != manifest["source_log_sha256"][record["segment_filename"]]
            ):
                raise RuntimeError("source log integrity failure")
            with log_path.open("rb") as stream:
                frames = pickle.load(stream)
            loaded_segment = record["segment_filename"]
        cached = cache_window(
            frames,
            record["start_index"],
            sensor_root,
            encoder,
            device,
            image_hashes,
            rectify_stored_image=evidence["image_mode"]
            == "undistort_stored_brown_conrady",
        )
        cached["metadata"].update(record)
        filename = f"{Path(record['segment_filename']).stem}__window_{record['start_index']:05d}.pt"
        torch.save(cached, directory / filename)
        disk_bytes += (directory / filename).stat().st_size
        if disk_bytes > configuration["maximum_cache_bytes"]:
            raise RuntimeError("registered cache disk cap")
        indexed.append(
            {
                **record,
                "cache_file": filename,
                "cache_sha256": file_sha256(directory / filename),
                "cache_bytes": (directory / filename).stat().st_size,
            }
        )
        (directory / "cache_index.json").write_text(
            json.dumps(
                {
                    "configuration": configuration,
                    "manifest_sha256": file_sha256(arguments.manifest),
                    "preprocessing_evidence_sha256": file_sha256(
                        arguments.preprocessing_evidence
                    ),
                    "encoder_provenance": encoder.provenance,
                    "records": indexed,
                    "complete": len(indexed) == len(records),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"REGISTERED_CACHE windows={len(indexed)}/{len(records)}", flush=True)
    seconds = time.perf_counter() - started
    projected_bytes = disk_bytes / len(indexed) * len(manifest["records"])
    projected_seconds = seconds / len(indexed) * len(manifest["records"])
    report = {
        "windows": len(indexed),
        "wall_seconds": seconds,
        "cached_bytes": disk_bytes,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * 1024,
        "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(device),
        "projected_full_cache_bytes": projected_bytes,
        "projected_full_cache_seconds": projected_seconds,
        "within_registered_bounds": projected_bytes
        <= configuration["maximum_cache_bytes"]
        and projected_seconds <= configuration["maximum_cache_seconds"],
        "manifest_sha256": file_sha256(arguments.manifest),
        "preprocessing_evidence_sha256": file_sha256(arguments.preprocessing_evidence),
        "held_out_model_evaluation_performed": False,
    }
    (
        directory
        / ("profile_report.json" if arguments.profile_only else "cache_report.json")
    ).write_text(json.dumps(report, indent=2) + "\n")
    (directory / "source_image_sha256.json").write_text(
        json.dumps(image_hashes, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
