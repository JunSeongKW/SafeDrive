"""Archive completed diagnostics and report paired changes on the fixed training panel."""
from datetime import datetime, timedelta
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
CONTROLLER = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
REPRESENTATIONS = ROOT / "outputs/lpwm_drivor_planning_path_representation_monitor_v1"
TRAINING = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1"
EVALUATIONS = (
    ROOT / "outputs/lpwm_drivor_intermediate_pdms_v1",
    ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch1_v1",
    ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3000_v1",
)


def read_json(path):
    return json.loads(path.read_text())


def paired_pdms_change(before_rows, after_rows):
    before_by_token = {row["token"]: row for row in before_rows}
    assert set(before_by_token) == {row["token"] for row in after_rows}
    group_differences = {}
    for row in after_rows:
        before = before_by_token[row["token"]]
        assert row["recording_group"] == before["recording_group"]
        group_differences.setdefault(row["recording_group"], []).append((row["score"] - before["score"]) * 100)
    groups = [group_differences[name] for name in sorted(group_differences)]
    group_sums = np.array([sum(values) for values in groups])
    group_counts = np.array([len(values) for values in groups])
    sampled_groups = np.random.default_rng(71).integers(0, len(groups), size=(5000, len(groups)))
    sampled_changes = group_sums[sampled_groups].sum(1) / group_counts[sampled_groups].sum(1)
    return {"pdms_change": float(group_sums.sum() / group_counts.sum()),
            "paired_recording_bootstrap_95_ci": np.quantile(sampled_changes, [.025, .975]).tolist(),
            "recordings": len(groups), "bootstrap_repeats": 5000, "seed": 71}


def main():
    checked_at = datetime.now(ZoneInfo("Asia/Seoul"))
    pdms_rows, scores_by_update = [], {}
    for evaluation in EVALUATIONS:
        complete = read_json(evaluation / "evaluation_complete.json")
        assert complete["complete"]
        for row in complete["rows"]:
            update = row["completed_updates"]
            pdms_rows.append(row)
            scores_by_update[update] = read_json(evaluation / f"update_{update:06d}/scores.json")
    pdms_rows.sort(key=lambda row: row["completed_updates"])
    paired_changes = [{"before_update": before, "after_update": after,
                       **paired_pdms_change(scores_by_update[before], scores_by_update[after])}
                      for before, after in [(1614, 2000), (1614, 2500), (1614, 3000), (1000, 3000), (2500, 3000)]]
    training_rows = {rank: [json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
                     for rank in (0, 1)}
    progress = read_json(TRAINING / "progress.json")
    training_checks = {str(rank): {
        "logged_updates": len(rows), "last_logged_update": rows[-1]["completed_updates"],
        "nonfinite_loss_rows": sum(not all(np.isfinite(row[key]) for key in ("loss", "trajectory_loss", "final_score_loss")) for row in rows),
        "native_digest_count": len({row["frozen_native_sha256"] for row in rows}),
        "execution_digest_count": len({row["execution_sha256"] for row in rows})}
        for rank, rows in training_rows.items()}
    last_gradient_sample = next(row for row in reversed(training_rows[0]) if row["gradient_norms"])
    eta = {}
    for window in (50, 100, 200):
        end, start = training_rows[0][-1], training_rows[0][-window - 1]
        seconds_per_update = (end["elapsed_seconds"] - start["elapsed_seconds"]) / (end["completed_updates"] - start["completed_updates"])
        eta[str(window)] = {"seconds_per_update": seconds_per_update,
                            "targets": {str(target): (checked_at + timedelta(seconds=(target - progress["completed_updates"]) * seconds_per_update)).isoformat()
                                        for target in (3228, 3322, 3500, 4000, 40350)}}
    representation_rows = []
    for update in (0, 500, 1000, 1500, 1614, 2000, 2500, 3000):
        report = read_json(CONTROLLER / f"reports/update_{update:06d}.json")
        representation_rows.append({"update": update, "metrics": report["metrics"],
            "fraction_center_shift_over_one_input_pixel": report["from_initial"]["fraction_center_shift_over_one_input_pixel"],
            "fraction_particles_any_size_axis_change_over_five_percent": report["from_initial"]["fraction_particles_any_size_axis_change_over_five_percent"],
            "distribution": report["representation_diagnostics"]["distribution"]["all"],
            "intent_sensitivity": report["representation_diagnostics"]["intent_sensitivity"],
            "classification_macro_f1": {name: value["macro_f1_present_classes"] for name, value in report["readouts"]["classification"].items()},
            "future_displacement": report["readouts"]["future_displacement"],
            "future_repeat_current_intervention": report["representation_diagnostics"]["paired_interventions"]["future_repeat_current"],
            "checkpoint": report["checkpoint"]})
    result = {"checked_at": checked_at.isoformat(), "progress": progress, "training_checks": training_checks,
              "latest_gradient_sample": last_gradient_sample, "eta": eta, "warmup_updates": 3322,
              "pdms_scope": "Official NAVSIM v1 metric on the same 95 scenes / 24 recordings from upstream training distribution; not navtest",
              "pdms_rows": pdms_rows, "paired_changes": paired_changes, "representation_rows": representation_rows,
              "latest_evaluation_replay_checks": read_json(EVALUATIONS[-1] / "replay_checks.json"),
              "interpretation": "Joint planning PDMS rises on the training panel. Geometry and intent-conditioned output changes are confirmed, but object readouts are mixed and the epoch1 future-information improvement does not persist. LPWM-only benefit and held-out generalization are not established.",
              "training_changes": False}
    (DESTINATION / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    for name in ("registration.json", "launch.json", "evaluation_complete.json", "replay_checks.json"):
        shutil.copy2(EVALUATIONS[-1] / name, DESTINATION / name)
    for update in (2000, 2500, 3000):
        destination = DESTINATION / f"update_{update:06d}"
        destination.mkdir(exist_ok=True)
        for name in ("summary.json", "scores.csv", "scores.json"):
            shutil.copy2(EVALUATIONS[-1] / f"update_{update:06d}" / name, destination / name)
        shutil.copy2(CONTROLLER / f"reports/update_{update:06d}.json", destination / "representation_report.json")
    shutil.copy2(CONTROLLER / "trend.csv", DESTINATION / "particle_trend.csv")
    shutil.copy2(CONTROLLER / "trend.png", DESTINATION / "particle_trend.png")
    shutil.copy2(REPRESENTATIONS / "update_003000/geometry/before_after_particle_geometry.png", DESTINATION / "particles_before_vs_update3000.png")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.3), constrained_layout=True)
    updates = [row["completed_updates"] for row in pdms_rows]
    scores = [row["pdms"] for row in pdms_rows]
    axes[0].plot(updates, scores, "o-", color="#2563eb")
    for update, score in zip(updates, scores):
        axes[0].annotate(f"{score:.2f}", (update, score), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(xlabel="Optimizer updates", ylabel="PDMS (0-100)", title="Same 95 training-panel scenes", ylim=(20, 85))
    compared_rows = [next(row for row in pdms_rows if row["completed_updates"] == update) for update in (1614, 3000)]
    scenarios = ("straight", "left_turn", "right_turn")
    for index, row in enumerate(compared_rows):
        positions = np.arange(3) + (index - .5) * .36
        bars = axes[1].bar(positions, [row["by_scene_type"][name]["pdms"] for name in scenarios], width=.36,
                           label=f"{row['completed_updates']} updates", color=("#94a3b8", "#2563eb")[index])
        axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set(xticks=np.arange(3), xticklabels=("Straight (41)", "Left (30)", "Right (24)"), ylabel="PDMS", ylim=(0, 100), title="First epoch versus 3,000 updates")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    figure.suptitle("Official NAVSIM v1 PDMS | training panel, not navtest")
    figure.savefig(DESTINATION / "pdms_progress.png", dpi=160)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, horizon in zip(axes, ("2", "4")):
        comparisons = [row["future_displacement"][horizon]["predicted_minus_current_error"] for row in representation_rows]
        values = np.array([comparison["difference"] for comparison in comparisons])
        intervals = np.array([comparison["confidence_interval_95"] for comparison in comparisons])
        axis.errorbar([row["update"] for row in representation_rows], values,
                       yerr=np.stack((values - intervals[:, 0], intervals[:, 1] - values)), fmt="o-", capsize=3)
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="Optimizer updates", ylabel="Future minus current readout error (m)", title=f"{horizon}s object displacement | negative is better")
        axis.grid(alpha=.25)
    figure.suptitle("Fixed 6-recording readout evaluation | upstream training distribution")
    figure.savefig(DESTINATION / "future_readout_trend.png", dpi=160)
    plt.close(figure)
    print(json.dumps({"checked_at": result["checked_at"], "completed_updates": progress["completed_updates"],
                      "paired_changes": paired_changes, "training_checks": training_checks, "eta": eta,
                      "replay_checks": {key: value for key, value in result["latest_evaluation_replay_checks"].items() if key != "checks"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
