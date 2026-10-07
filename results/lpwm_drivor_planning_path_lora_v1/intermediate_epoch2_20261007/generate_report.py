"""Archive epoch-two diagnostics and official PDMS without changing training."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = TRAINING_ROOT / "navsim_v1"
CONTROLLER = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
REPRESENTATIONS = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
EVALUATION = ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch2_v1"
PREVIOUS_REPORT = ROOT / "results/lpwm_drivor_planning_path_lora_v1/intermediate_update3000_20261007"
OVERLAYS = ROOT / "outputs/lpwm_drivor_particle_geometry_overlays_v1/update_003228"


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_pdms_change(before_rows, after_rows):
    before_by_token = {row["token"]: row for row in before_rows}
    assert set(before_by_token) == {row["token"] for row in after_rows}
    differences_by_recording = {}
    for row in after_rows:
        before = before_by_token[row["token"]]
        assert row["recording_group"] == before["recording_group"]
        differences_by_recording.setdefault(row["recording_group"], []).append(
            (row["score"] - before["score"]) * 100)
    recordings = [differences_by_recording[name] for name in sorted(differences_by_recording)]
    sums = np.array([sum(values) for values in recordings])
    counts = np.array([len(values) for values in recordings])
    draws = np.random.default_rng(71).integers(0, len(recordings), (5000, len(recordings)))
    bootstrap_changes = sums[draws].sum(1) / counts[draws].sum(1)
    return {"pdms_change": float(sums.sum() / counts.sum()),
            "paired_recording_bootstrap_95_ci": np.quantile(bootstrap_changes, [.025, .975]).tolist(),
            "recordings": len(recordings), "bootstrap_repeats": 5000, "seed": 71}


def main():
    assert not (DESTINATION / "report.json").exists(), "Keep completed reports immutable"
    checked_at = datetime.now(ZoneInfo("Asia/Seoul"))
    completed_evaluation = read_json(EVALUATION / "evaluation_complete.json")
    assert completed_evaluation["complete"]
    latest_pdms = completed_evaluation["rows"][0]
    assert latest_pdms["completed_updates"] == 3228 and latest_pdms["failed"] == 0
    previous = read_json(PREVIOUS_REPORT / "report.json")
    pdms_rows = previous["pdms_rows"] + [latest_pdms]
    after_scores = read_json(EVALUATION / "update_003228/scores.json")
    previous_scores = {
        1614: ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch1_v1/update_001614/scores.json",
        2500: PREVIOUS_REPORT / "update_002500/scores.json",
        3000: PREVIOUS_REPORT / "update_003000/scores.json",
    }
    paired_changes = [{"before_update": update, "after_update": 3228,
                       **paired_pdms_change(read_json(path), after_scores)}
                      for update, path in previous_scores.items()]
    training_rows = {rank: [json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
                     for rank in (0, 1)}
    progress = read_json(TRAINING / "progress.json")
    checks = {str(rank): {
        "logged_updates": len(rows), "last_logged_update": rows[-1]["completed_updates"],
        "nonfinite_loss_rows": sum(not all(np.isfinite(row[key]) for key in ("loss", "trajectory_loss", "final_score_loss")) for row in rows),
        "native_digests": sorted({row["frozen_native_sha256"] for row in rows}),
        "execution_digests": sorted({row["execution_sha256"] for row in rows})}
        for rank, rows in training_rows.items()}
    gradient_samples = {str(rank): next(row for row in reversed(rows) if row.get("gradient_norms"))
                        for rank, rows in training_rows.items()}
    for sample in gradient_samples.values():
        assert all(np.isfinite(value) and value > 0 for value in sample["gradient_norms"].values())
    registration_checks = {}
    for name in ("registration.json", "oracle8_execution_registration.json"):
        registered = read_json(TRAINING_ROOT / name)
        mismatches = [relative for relative, expected in registered["sources"].items()
                      if sha256(ROOT / relative) != expected]
        assert not mismatches, mismatches
        registration_checks[name] = {"source_count": len(registered["sources"]), "mismatches": mismatches}
    execution_rows = [row for row in training_rows[0]
                      if row["execution_sha256"] == progress["execution_sha256"] and row["completed_updates"] >= 3161]
    eta = {}
    for window in (50, 100):
        start, end = execution_rows[-window - 1], execution_rows[-1]
        seconds_per_update = (end["elapsed_seconds"] - start["elapsed_seconds"]) / (end["completed_updates"] - start["completed_updates"])
        eta[str(window)] = {"seconds_per_update": seconds_per_update,
            "targets": {str(target): (checked_at + timedelta(seconds=(target - progress["completed_updates"]) * seconds_per_update)).isoformat()
                        for target in (3500, 4000, 4842, 40350)},
            "scope": "V1 training only; excludes diagnostic processing, final evaluation, and V2 training"}
    representation_rows = []
    for update in (0, 1614, 2500, 3000, 3228):
        report = read_json(CONTROLLER / f"reports/update_{update:06d}.json")
        representation_rows.append({"update": update, "metrics": report["metrics"],
            "from_initial": report["from_initial"], "from_previous_diagnostic": report["from_previous_diagnostic"],
            "readouts": report["readouts"], "diagnostics": report["representation_diagnostics"], "checkpoint": report["checkpoint"]})
    result = {"checked_at": checked_at.isoformat(), "report_source_sha256": sha256(Path(__file__)),
        "progress": progress, "training_checks": checks, "gradient_samples": gradient_samples,
        "source_registration_checks": registration_checks, "eta": eta,
        "warmup_updates": 3322, "warmup_complete": progress["completed_updates"] >= 3322,
        "pdms_scope": completed_evaluation["scope"], "pdms_rows": pdms_rows, "paired_changes": paired_changes,
        "representation_rows": representation_rows, "replay_checks": read_json(EVALUATION / "replay_checks.json"),
        "monitor_status": read_json(CONTROLLER / "status.json"),
        "interpretation": "Epoch2 PDMS is similar to update3000, with opposite changes for left and right turns. Geometry and appearance change, but concentration in driving regions, stable future-information improvement, LPWM-only benefit, and held-out generalization are not established.",
        "training_changes": False, "new_training_or_baseline_launched": False}
    for name in ("evaluation_complete.json", "registration.json", "replay_checks.json"):
        shutil.copy2(EVALUATION / name, DESTINATION / name)
    for name in ("summary.json", "scores.json", "scores.csv"):
        shutil.copy2(EVALUATION / "update_003228" / name, DESTINATION / name)
    shutil.copy2(CONTROLLER / "reports/update_003228.json", DESTINATION / "representation_report_update3228.json")
    shutil.copy2(OVERLAYS / "overlay_report.json", DESTINATION / "particle_overlay_report_update3228.json")
    shutil.copy2(OVERLAYS / "particle_geometry_comparison_with_overlay.png", DESTINATION / "particles_before_after_and_overlay_update3228.png")
    shutil.copy2(OVERLAYS / "particle_geometry_overlay.png", DESTINATION / "particles_overlay_update3228.png")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    axes[0].plot([row["completed_updates"] for row in pdms_rows], [row["pdms"] for row in pdms_rows], "o-", color="#2563eb")
    for row in pdms_rows:
        if row["completed_updates"] in (0, 1000, 1614, 2500, 3000, 3228):
            axis_offset = (22, -18) if row["completed_updates"] == 3228 else (0, 8)
            axes[0].annotate(f"{row['pdms']:.2f}", (row["completed_updates"], row["pdms"]),
                             xytext=axis_offset, textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(xlabel="Optimizer updates", ylabel="PDMS (0-100)", title="Same 95 training-panel scenes", ylim=(20, 90), xlim=(-100, 3700))
    scenarios = ("straight", "left_turn", "right_turn")
    for index, update in enumerate((3000, 3228)):
        row = next(row for row in pdms_rows if row["completed_updates"] == update)
        bars = axes[1].bar(np.arange(3) + (index - .5) * .36,
            [row["by_scene_type"][scenario]["pdms"] for scenario in scenarios], width=.36,
            label=f"{update:,} updates", color=("#94a3b8", "#2563eb")[index])
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(xticks=np.arange(3), xticklabels=("Straight (41)", "Left (30)", "Right (24)"),
                ylabel="PDMS", ylim=(0, 100), title="3,000 updates versus epoch 2")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("Official NAVSIM v1 PDMS | training panel, not navtest")
    figure.savefig(DESTINATION / "pdms_progress_and_scenarios.png", dpi=160)
    plt.close(figure)
    (DESTINATION / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checked_at": result["checked_at"], "progress_updates": progress["completed_updates"],
        "pdms": latest_pdms["pdms"], "paired_changes": paired_changes, "training_checks": checks,
        "eta": eta, "source_registration_checks": registration_checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
