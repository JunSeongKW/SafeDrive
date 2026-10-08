"""Compare particle budgets, then train one local SSL epoch with the smallest viable one."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts/train_lpwm_reduced_particle_stage1.py"


def write_json(path, content):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def run_condition(root, particle_count, updates, output):
    if (output / "complete.json").exists():
        return
    if (root / "pause.requested").exists() or (output / "paused.json").exists():
        raise RuntimeError("Queue paused; explicitly inspect and resume after resource/user pause")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", str(TRAINER), "--particles", str(particle_count),
        "--updates", str(updates), "--accumulation", "4", "--validation-clips", "32",
        "--workers", "4", "--output", str(output)]
    write_json(root / "queue_state.json", {"status": "training", "particles": particle_count,
        "output": str(output), "command": command, "started_unix": time.time()})
    with (output / "console.log").open("a") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
    assert (output / "complete.json").exists(), "Training paused or incomplete; dependent work blocked"


def main(arguments):
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    sources = [TRAINER, Path(__file__), ROOT / "src/planning_aware_future_prediction/object_centric/lpwm_rectangular.py"]
    registration = {"sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "candidates": [8, 16, 32, 64], "updates_each": 200, "clips_per_update": 4,
        "candidate_quality_tolerance": 1.15, "selection_is_provisional": True,
        "object_preservation_or_planning_noninferiority_proven": False,
        "after_selection": "one_full_available_OpenScene_train_epoch_from_public_initialization",
        "full_330h_run_blocked_on_gated_dataset_access": True, "physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"]}
    if (root / "registration.json").exists():
        assert json.loads((root / "registration.json").read_text()) == registration
    else:
        write_json(root / "registration.json", registration)
    candidates = {}
    for particles in registration["candidates"]:
        output = root / f"candidate_particles{particles}"
        run_condition(root, particles, 200, output)
        candidates[particles] = json.loads((output / "complete.json").read_text())["validation"]
    reference = candidates[64]
    checked_metrics = ["reconstruction_mse", "reconstruction_lpips", "forecast_mse", "forecast_lpips", "forecast_lower_half_mse"]
    decisions = {}
    for particles, measured in candidates.items():
        ratios = {metric: measured[metric] / max(reference[metric], 1e-10) for metric in checked_metrics}
        noncollapse = measured["mean_presence"] > .05 and measured["position_std"] > .05 and measured["appearance_std"] > .001
        decisions[particles] = {"ratios_to_64": ratios, "noncollapse": noncollapse,
            "passes_provisional_screen": max(ratios.values()) <= 1.15 and noncollapse}
    selected = next((count for count in (8, 16, 32) if decisions[count]["passes_provisional_screen"]), None)
    write_json(root / "particle_selection.json", {"selected_foreground_particles": selected,
        "decisions": decisions, "validation": candidates, "purpose": "early trend exploration",
        "small_object_information_verified": False, "planning_performance_verified": False})
    if selected is None:
        write_json(root / "queue_state.json", {"status": "held_for_quality_review", "reason": "No reduced candidate passed even the exploratory screen"})
        return
    run_condition(root, selected, 0, root / f"local_stage1_particles{selected}")
    write_json(root / "queue_state.json", {"status": "local_stage1_complete_external_corpus_pending",
        "selected_particles": selected, "full_330h_training_complete": False,
        "stage2_started": False, "dataset_cleanup_authorized_after_full_training": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/particle_budget_study")
    arguments = parser.parse_args()
    try:
        main(arguments)
    except Exception as error:
        write_json(arguments.output.resolve() / "queue_failed.json", {"error": repr(error), "time": time.time()})
        raise
