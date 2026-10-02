"""Freeze sample IDs on CPU; optionally build a bounded official-feature cache."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

import run_drive_jepa_selection_comparison as cache_builder
import torch
from diagnose_drive_jepa_learning_limitations import load_official_agent
from run_drive_jepa_architecture_followup import build_model
from validate_drive_jepa_selective_future_connection import (
    file_sha256,
    parameter_sha256,
)

WORKSPACE = Path(__file__).resolve().parents[1]


def resolved_sampling_specification(specification):
    """Exclude known development exposure without changing any train assignment."""
    reused = json.loads((WORKSPACE / specification["reused_cache"]).read_text())
    mini = json.loads(
        (
            WORKSPACE
            / "results/data_surveys/navsim_recording_split_manifest_20261001.json"
        ).read_text()
    )
    prior_development = {
        row["recording_group"]
        for row in reused["records"]
        if row["split"] == "development"
    }
    prior_development.update(
        row["recording_group"]
        for row in mini["groups"]
        if row["split"] == "development"
    )
    manifest = json.loads(
        (WORKSPACE / specification["project_recording_split_manifest"]).read_text()
    )
    # Some historical mini assignments may have changed in the later three-way
    # manifest; excluding an already-evaluated recording remains conservative.
    resolved = {
        **specification,
        "excluded_recordings": sorted(
            set(specification["excluded_recordings"]) | prior_development
        ),
    }
    selected = cache_builder.select_manifest_records(manifest, resolved)
    old_train = {
        row["current_frame_token"]
        for row in reused["records"]
        if row["split"] == "train"
    }
    new_train = {
        row["current_frame_token"] for row in selected if row["split"] == "train"
    }
    if not old_train.issubset(new_train):
        raise RuntimeError("Nested training-subset contract failed")
    if {
        row["recording_group"] for row in selected if row["split"] == "development"
    } & prior_development:
        raise RuntimeError("Known development exposure leaked into added recordings")
    return resolved, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    specification = json.loads(args.config.resolve().read_text())
    resolved, selected = resolved_sampling_specification(specification)
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "resolved_specification.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != resolved:
        raise RuntimeError("Preserved sampling specification changed")
    cache_builder.write_json(snapshot, resolved)
    cache_builder.write_json(output / "preregistered_windows.json", selected)
    cache_builder.write_json(
        output / "sampling_preflight.json",
        {
            "window_count": len(selected),
            "split_counts": {
                split: sum(row["split"] == split for row in selected)
                for split in ("train", "development")
            },
            "recording_counts": {
                split: len(
                    {
                        row["recording_group"]
                        for row in selected
                        if row["split"] == split
                    }
                )
                for split in ("train", "development")
            },
            "manifest_sha256": file_sha256(
                WORKSPACE / specification["project_recording_split_manifest"]
            ),
            "known_development_excluded": resolved["excluded_recordings"],
            "held_out_evaluated": False,
        },
    )
    print(f"PREREGISTERED {len(selected)} windows before new model scores", flush=True)
    if not args.build:
        return
    if (output / "cache_index.json").is_file():
        print("CACHE_ALREADY_COMPLETE_REUSING", flush=True)
        return
    if (output / "feature_cache").exists():
        raise RuntimeError(
            "Partial cache is preserved; inspect before an explicit recovery"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("0", "1"):
        raise RuntimeError("Only one approved GPU 0/1")
    torch.set_num_threads(1)
    if torch.cuda.mem_get_info()[0] < specification["minimum_free_gpu_gib"] * 2**30:
        raise RuntimeError("Insufficient GPU launch reserve")
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, cache_builder.request_stop)
    started = time.perf_counter()
    agent, source = load_official_agent(output)
    baseline_hash = parameter_sha256(agent._model)
    if (
        baseline_hash
        != "05af3ccd8a8e7db1b9b2ae2e77f81ef753446a3309989cef08c161ce0614476a"
    ):
        raise RuntimeError("Official source/checkpoint mismatch")
    cache_builder.write_json(output / "source.json", source)
    model = build_model(agent._model, "mlp_recipe_control", 29)
    # Warm the exact frozen execution path before numerical identity checks.
    connection = json.loads(
        (WORKSPACE / specification["base_connection_specification"]).read_text()
    )
    report, cached_tensors = cache_builder.build_cache(
        agent,
        model,
        Path(source["official_root"]),
        resolved,
        connection,
        output,
        started,
        reused_cache_index=WORKSPACE / specification["reused_cache"],
    )
    if parameter_sha256(agent._model) != baseline_hash:
        raise RuntimeError("Frozen baseline changed during cache creation")
    cache_builder.write_json(
        output / "completion.json",
        {
            "complete": True,
            "baseline_hash_unchanged": baseline_hash,
            "wall_seconds": time.perf_counter() - started,
            "counts": report["counts"],
            "new_cache_bytes": report["new_cache_bytes"],
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        },
    )
    del cached_tensors
    print("RECORDING_COVERAGE_CACHE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
