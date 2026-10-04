"""Auditable execution-only changes and checkpoint forks under a48GB card cap."""
import json
from pathlib import Path
import subprocess

import torch

from evaluate_lpwm_full_planning import digest, write_json


EXECUTION_FIELDS = {
    "output_directory", "shared_results_directory", "microbatch_size_per_gpu", "gradient_accumulation",
    "world_auxiliary_clips_per_gpu_microbatch", "minimum_free_gib", "maximum_allocated_gib",
    "maximum_reserved_gib", "resource_limits", "profile_updates", "execution_amendment", "adaptation_method",
}


def verify_execution_configuration(configuration, project_root):
    amendment = configuration["execution_amendment"]
    parent_path = Path(project_root) / amendment["parent_configuration"]
    assert digest(parent_path) == amendment["parent_configuration_sha256"]
    parent = json.loads(parent_path.read_text())
    unchanged = lambda values: {key: value for key, value in values.items() if key not in EXECUTION_FIELDS}
    assert unchanged(configuration) == unchanged(parent), "A scientific setting changed in an execution-only amendment"
    for values in (parent, configuration):
        assert values["microbatch_size_per_gpu"] * values["gradient_accumulation"] * values["world_size"] == 16
        assert values["world_auxiliary_clips_per_gpu_microbatch"] * values["gradient_accumulation"] * values["world_size"] == 8
    assert configuration["workers_per_rank"] == 0
    assert configuration["resource_limits"]["maximum_gpu_used_bytes"] == 48_000_000_000
    return parent


def gpu_memory_snapshot():
    raw = subprocess.check_output(["nvidia-smi", "--id=0,1", "--query-gpu=index,memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True)
    return {int(fields[0]): {"used_bytes": int(fields[1]) * 1024**2, "free_gib": int(fields[2]) / 1024}
        for fields in ([part.strip() for part in line.split(",")] for line in raw.splitlines())}


def check_gpu_limits(gpu, minimum_free_gib=3):
    row = gpu_memory_snapshot()[gpu]
    if row["used_bytes"] > 48_000_000_000 or row["free_gib"] < minimum_free_gib:
        raise RuntimeError(f"48GB GPU memory guard: GPU{gpu}: {row}")
    return int(row["free_gib"] * 1024)


def fork_execution_checkpoint(source_path, destination, configuration_path, project_root, profile_only=False):
    """Copy model+AdamW verbatim; only execution metadata and diagnostic counters change."""
    specification = json.loads(configuration_path.read_text())
    verify_execution_configuration(specification, project_root)
    source_hash = digest(source_path)
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    source_config_hash = specification["execution_amendment"]["parent_configuration_sha256"]
    assert saved["configuration_sha256"] == source_config_hash
    completed = saved["completed_updates"]
    assert 0 < completed < 4707
    steps = {int(state["step"]) for state in saved["optimizer"]["state"].values()}
    assert steps == {completed}
    provenance = {"source_checkpoint": str(source_path), "source_checkpoint_sha256": source_hash,
        "source_configuration_sha256": source_config_hash, "source_completed_updates": completed,
        "optimizer_states": len(saved["optimizer"]["state"]), "profile_only": profile_only,
        "model_and_optimizer_unchanged": True, "profile_updates_are_discarded": True}
    saved["execution_parent"] = provenance
    saved["configuration_sha256"] = digest(configuration_path)
    if profile_only:
        saved["completed_updates"] = 0
        saved["elapsed_seconds"] = 0.
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending.pt")
    torch.save(saved, pending)
    pending.replace(destination)
    write_json(destination.parent / "execution_checkpoint_parent.json", provenance)
    assert digest(source_path) == source_hash
    return provenance


def choose_profile(records, minimum_reserved_margin_gib=.5):
    eligible = [row for row in records if row.get("passed") and
        row["minimum_cap_headroom_gib"] >= minimum_reserved_margin_gib and row["minimum_free_gib"] >= 3]
    if not eligible:
        raise RuntimeError("No profile passed finite loss, memory and headroom checks")
    return min(eligible, key=lambda row: row["steady_update_seconds"])
