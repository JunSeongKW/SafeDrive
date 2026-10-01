"""Verify NEW follow-up checkpoint state/RNG contracts; no training or old recovery rerun."""

import json
import random

import numpy as np
import torch
import train_target_supervision_ablation as exploration
from train_bounded_future_prediction_followup import (
    generate_recording_balanced_batch_sequence,
)

PROJECT_ROOT = exploration.PROJECT_ROOT


def restore_cpu_rng_states(states):
    torch.set_rng_state(states["torch_cpu"])
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state(
        (
            numpy_state["generator"],
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )


def main():
    directory = (
        PROJECT_ROOT / "outputs/future_prediction_diagnostics/bounded_followup_1000_v1"
    )
    report = json.loads((directory / "comparison_report.json").read_text())
    if report.get("status") != "bounded_followup_complete":
        raise RuntimeError("run incomplete")
    verification = {
        "new_final_checkpoints": [],
        "historical_outputs_overwritten": False,
        "old_checkpoint_bitwise_exact_resume_verified": False,
        "cuda_rng_restoration": "Snapshot present and finite byte tensors; CUDA RNG replay not executed in this CPU verification.",
    }
    for run in report["runs"]:
        seed, label, update = run["seed"], run["condition"]["label"], run["end_update"]
        path = directory / f"seed{seed}_{label}" / f"checkpoint_update{update:04d}.pt"
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        model = exploration.FixedDistanceFutureSupervisionPilot()
        model.load_state_dict(checkpoint["model"], strict=True)
        if not all(
            torch.isfinite(tensor).all() for tensor in checkpoint["model"].values()
        ):
            raise RuntimeError("nonfinite state")
        optimizer_steps = sorted(
            {int(state["step"]) for state in checkpoint["optimizer"]["state"].values()}
        )
        if (
            optimizer_steps != [update]
            or checkpoint["sampler_next_update"] != update + 1
        ):
            raise RuntimeError("optimizer/sampler step mismatch")
        expected_batches = json.loads(
            (directory / f"batch_sequence_seed{seed}.json").read_text()
        )
        if checkpoint["sampler_sequence"] != expected_batches:
            raise RuntimeError("sampler snapshot mismatch")
        states = checkpoint["rng_states"]
        restore_cpu_rng_states(states)
        first = (
            torch.rand(5),
            [random.random() for _ in range(5)],
            np.random.random(5),
        )
        restore_cpu_rng_states(states)
        second = (
            torch.rand(5),
            [random.random() for _ in range(5)],
            np.random.random(5),
        )
        if (
            not torch.equal(first[0], second[0])
            or first[1] != second[1]
            or not np.array_equal(first[2], second[2])
        ):
            raise RuntimeError("CPU RNG snapshot replay differs")
        if len(states["torch_cuda_all_visible"]) != 1:
            raise RuntimeError("unexpected visible CUDA devices")
        verification["new_final_checkpoints"].append(
            {
                "seed": seed,
                "label": label,
                "optimizer_update": update,
                "checkpoint_sha256": exploration.file_sha256(path),
                "strict_loading_finite_verified": True,
                "cpu_rng_replay_verified": True,
                "cuda_rng_snapshot_bytes": states["torch_cuda_all_visible"][0].numel(),
                "sampler_next_update": checkpoint["sampler_next_update"],
            }
        )
    # Sampler-only reconstruction contract is independent of learned weights.
    synthetic_windows = [
        {"metadata": {"recording_group": "recording_a"}},
        {"metadata": {"recording_group": "recording_b"}},
    ]
    short = generate_recording_balanced_batch_sequence(synthetic_windows, 29, 200, 8)
    long = generate_recording_balanced_batch_sequence(synthetic_windows, 29, 1000, 8)
    if short != long[:200]:
        raise RuntimeError("sampler prefix contract broken")
    verification["sampler_prefix_contract_verified"] = True
    output_path = (
        PROJECT_ROOT
        / "results/future_prediction_diagnostics/new_checkpoint_contracts_v1.json"
    )
    if output_path.exists():
        raise RuntimeError("refusing overwrite")
    output_path.write_text(json.dumps(verification, indent=2) + "\n")
    print("NEW_CHECKPOINT_CONTRACTS_DONE", len(verification["new_final_checkpoints"]))


if __name__ == "__main__":
    main()
