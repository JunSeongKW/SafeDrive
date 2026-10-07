"""Archive the fixed-panel update4000 evaluation and representation diagnostics."""
from datetime import datetime, timedelta
import hashlib
import importlib.util
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
EVALUATION = ROOT / "outputs/lpwm_drivor_intermediate_pdms_update4000_v1"
OVERLAYS = ROOT / "outputs/lpwm_drivor_particle_geometry_overlays_v1/update_004000"


def read_json(path):
    return json.loads(path.read_text())


def main():
    assert not (DESTINATION / "report.json").exists(), "Preserve completed reports"
    checked_at = datetime.now(ZoneInfo("Asia/Seoul"))
    progress = read_json(TRAINING / "progress.json")
    training_checks, gradient_samples = {}, {}
    training_rows = {}
    for rank in (0, 1):
        rows = [json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
        training_rows[rank] = rows
        training_checks[str(rank)] = {"last_logged_update": rows[-1]["completed_updates"],
            "nonfinite_loss_rows": sum(not all(np.isfinite(row[key]) for key in ("loss", "trajectory_loss", "final_score_loss")) for row in rows),
            "native_digests": sorted({row["frozen_native_sha256"] for row in rows})}
        gradient_samples[str(rank)] = next(row for row in reversed(rows) if row.get("gradient_norms"))
        assert all(np.isfinite(value) and value > 0 for value in gradient_samples[str(rank)]["gradient_norms"].values())
    source_checks = {}
    for name in ("registration.json", "oracle8_execution_registration.json"):
        registered = read_json(TRAINING_ROOT / name)
        mismatches = [relative for relative, expected in registered["sources"].items()
                      if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
        assert not mismatches, mismatches
        source_checks[name] = {"source_count": len(registered["sources"]), "mismatches": mismatches}
    prior_report = ROOT / "results/lpwm_drivor_planning_path_lora_v1/intermediate_epoch2_20261007"
    specification = importlib.util.spec_from_file_location("prior_panel_reporting", prior_report / "generate_report.py")
    reporting = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reporting)
    summaries = read_json(prior_report / "report.json")["pdms_rows"]
    summaries += read_json(ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3500_v1/evaluation_complete.json")["rows"]
    completed = read_json(EVALUATION / "evaluation_complete.json")
    assert completed["complete"] and completed["rows"][0]["completed_updates"] == 4000
    summaries += completed["rows"]
    after_scores = read_json(EVALUATION / "update_004000/scores.json")
    paired_changes = []
    for update, location in (
        (1614, ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch1_v1/update_001614/scores.json"),
        (3000, ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3000_v1/update_003000/scores.json"),
        (3500, ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3500_v1/update_003500/scores.json")):
        paired_changes.append({"before_update": update, "after_update": 4000,
                               **reporting.paired_pdms_change(read_json(location), after_scores)})
    representations = [read_json(CONTROLLER / f"reports/update_{update:06d}.json")
                       for update in (0, 1614, 3000, 3228, 3500, 4000)]
    execution_rows = [row for row in training_rows[0]
                      if row["execution_sha256"] == progress["execution_sha256"] and row["completed_updates"] >= 3161]
    eta = {}
    for window in (50, 100, 200):
        start, end = execution_rows[-window - 1], execution_rows[-1]
        seconds_per_update = (end["elapsed_seconds"] - start["elapsed_seconds"]) / (end["completed_updates"] - start["completed_updates"])
        eta[str(window)] = {"seconds_per_update": seconds_per_update,
            "targets": {str(target): (checked_at + timedelta(seconds=(target - progress["completed_updates"]) * seconds_per_update)).isoformat()
                        for target in (4500, 4842, 40350)},
            "scope": "V1 training only; diagnostic processing and subsequent evaluation/V2 excluded"}
    report = {"checked_at": checked_at.isoformat(), "progress": progress, "training_checks": training_checks,
        "gradient_samples": gradient_samples, "source_registration_checks": source_checks,
        "pdms_scope": completed["scope"], "pdms_rows": summaries, "paired_pdms_changes": paired_changes,
        "representations": representations, "eta": eta, "monitor_status": read_json(CONTROLLER / "status.json"),
        "replay_checks": read_json(EVALUATION / "replay_checks.json"), "training_changes": False,
        "interpretation": "Training remains healthy. PDMS is similar to update3500, with higher straight score and lower right-turn score. Appearance readout improves. The earlier significant 12-scene future-repeat-current effect is not statistically sustained at update4000; physical future accuracy, LPWM-only benefit, and held-out generalization remain unconfirmed."}
    for name in ("registration.json", "evaluation_complete.json", "replay_checks.json"):
        shutil.copy2(EVALUATION / name, DESTINATION / name)
    for name in ("scores.json", "scores.csv", "summary.json"):
        shutil.copy2(EVALUATION / "update_004000" / name, DESTINATION / name)
    shutil.copy2(CONTROLLER / "reports/update_004000.json", DESTINATION / "representation_report_update4000.json")
    shutil.copy2(OVERLAYS / "overlay_report.json", DESTINATION / "particle_overlay_report_update4000.json")
    shutil.copy2(OVERLAYS / "particle_geometry_comparison_with_overlay.png", DESTINATION / "particles_before_after_and_overlay_update4000.png")
    shutil.copy2(OVERLAYS / "particle_geometry_overlay.png", DESTINATION / "particles_overlay_update4000.png")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    axes[0].plot([row["completed_updates"] for row in summaries], [row["pdms"] for row in summaries], "o-", color="#2563eb")
    for update in (1614, 3000, 4000):
        row = next(row for row in summaries if row["completed_updates"] == update)
        axes[0].annotate(f"{row['pdms']:.2f}", (update, row["pdms"]), xytext=(0, 10),
                         textcoords="offset points", ha="center", fontsize=10)
    axes[0].set(xlabel="Optimizer updates", ylabel="PDMS", title="Same 95 training-panel scenes", ylim=(20, 90))
    scenarios = ("straight", "left_turn", "right_turn")
    for index, update in enumerate((3500, 4000)):
        row = next(row for row in summaries if row["completed_updates"] == update)
        bars = axes[1].bar(np.arange(3) + (index - .5) * .36,
            [row["by_scene_type"][scenario]["pdms"] for scenario in scenarios], width=.36,
            label=f"{update:,} updates", color=("#94a3b8", "#2563eb")[index])
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(xticks=np.arange(3), xticklabels=("Straight (41)", "Left (30)", "Right (24)"), ylabel="PDMS", ylim=(0, 100), title="3,500 versus 4,000 updates")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("Official NAVSIM v1 PDMS | training panel, not navtest")
    figure.savefig(DESTINATION / "pdms_progress_and_scenarios.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.1), constrained_layout=True)
    for axis, metric, title in (
        (axes[0], "ade_m_change", "ADE change (m): positive favors original future"),
        (axes[1], "selected_trajectory_oracle_score_change", "Training oracle score change: negative favors original future")):
        comparisons = [row["representation_diagnostics"]["paired_interventions"]["future_repeat_current"][metric]
                       for row in representations[2:]]
        values = np.array([value["difference"] for value in comparisons])
        intervals = np.array([value["confidence_interval_95"] for value in comparisons])
        updates = [row["metrics"]["completed_updates"] for row in representations[2:]]
        axis.errorbar(updates, values, yerr=np.stack((values - intervals[:, 0], intervals[:, 1] - values)), fmt="o-", capsize=4)
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="Optimizer updates", ylabel="Replacement minus original", title=title)
        axis.grid(alpha=.2)
    figure.suptitle("Future-repeat-current intervention | same 12 training scenes, 95% intervals")
    figure.savefig(DESTINATION / "future_intervention_trend.png", dpi=160)
    plt.close(figure)
    (DESTINATION / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checked_at": report["checked_at"], "completed_updates": progress["completed_updates"],
        "pdms": completed["rows"][0]["pdms"], "paired_pdms_changes": paired_changes,
        "source_checks": source_checks, "training_checks": training_checks, "eta": eta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
