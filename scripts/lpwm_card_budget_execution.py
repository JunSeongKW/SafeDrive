"""Derive CUDA allocation allowance from the total48GB card budget."""
import json
from pathlib import Path

import torch

from evaluate_lpwm_full_planning import digest, write_json
from lpwm_48gb_execution import gpu_memory_snapshot, verify_execution_configuration

CARD_LIMIT_BYTES = 48_000_000_000
NON_ALLOCATOR_GROWTH_BYTES = 512 * 1024**2
ROUNDING_HEADROOM_BYTES = 128 * 1024**2


def allocator_budget_bytes(card_used_bytes, own_reserved_bytes):
    """Current NVML usage includes our context; add back only our allocator pool."""
    allowance = CARD_LIMIT_BYTES - card_used_bytes + own_reserved_bytes
    allowance -= NON_ALLOCATOR_GROWTH_BYTES + ROUNDING_HEADROOM_BYTES
    if allowance <= 0:
        raise RuntimeError("48GB GPU memory guard: no available CUDA allocation budget")
    return allowance


def check_card_budget(gpu, minimum_free_gib=0):
    row = gpu_memory_snapshot()[gpu]
    if row["used_bytes"] > CARD_LIMIT_BYTES:
        raise RuntimeError(f"48GB GPU memory guard: GPU{gpu}: {row}")
    return int(row["free_gib"] * 1024)


def install_allocator_budget(configuration, profile_only=False):
    """Override only resource setup; model, optimizer and registered trainer stay intact."""
    original = torch.cuda.set_per_process_memory_fraction
    project_root = Path(__file__).resolve().parents[1]
    directory = project_root / configuration["output_directory"] / "metric_plus_world"
    if profile_only:
        directory /= "profile"

    def configure(_requested_fraction, device=None):
        index = torch.cuda.current_device() if device is None else torch.device(device).index
        row = gpu_memory_snapshot()[index]
        reserved = torch.cuda.memory_reserved(index)
        allowance = allocator_budget_bytes(row["used_bytes"], reserved)
        total = torch.cuda.get_device_properties(index).total_memory
        fraction = min(1., allowance / total)
        original(fraction, index)
        report = {"gpu_index": index, "card_limit_bytes": CARD_LIMIT_BYTES,
            "card_used_bytes_before_model": row["used_bytes"], "own_reserved_bytes": reserved,
            "allocator_allowance_bytes": allowance, "allocator_allowance_gib": allowance / 1024**3,
            "non_allocator_growth_bytes": NON_ALLOCATOR_GROWTH_BYTES,
            "rounding_headroom_bytes": ROUNDING_HEADROOM_BYTES,
            "interpretation": "No fixed23.2GiB process cap; actual shared-card remaining budget minus measured context and small workspace reserve"}
        write_json(directory / f"allocator_budget_gpu{index}.json", report)
        print("CARD_ALLOCATOR_BUDGET", json.dumps(report), flush=True)

    torch.cuda.set_per_process_memory_fraction = configure


def choose_largest_safe_profile(records, _unused_margin=0):
    eligible = [record for record in records if record.get("passed")
        and record["maximum_sampled_gpu_used_bytes"] <= CARD_LIMIT_BYTES]
    if not eligible:
        raise RuntimeError("No batch passed the48GB card-budget profile")
    return max(eligible, key=lambda record: record["microbatch_size_per_gpu"])


def fork_registered_checkpoint(source_path, destination, configuration_path, project_root,
                               source_configuration_path, profile_only=False):
    configuration = json.loads(configuration_path.read_text())
    verify_execution_configuration(configuration, project_root)
    source_hash = digest(source_path)
    saved = torch.load(source_path, map_location="cpu", weights_only=False)
    assert saved["configuration_sha256"] == digest(source_configuration_path)
    completed = saved["completed_updates"]
    assert 0 < completed < 4707
    assert {int(value["step"]) for value in saved["optimizer"]["state"].values()} == {completed}
    provenance = {"source_checkpoint": str(source_path), "source_checkpoint_sha256": source_hash,
        "source_configuration_sha256": digest(source_configuration_path), "source_completed_updates": completed,
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
