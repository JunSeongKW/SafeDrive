"""Archive exact update4842 PDMS, paired comparisons and recorded particles."""

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


project_root = Path(__file__).resolve().parents[3]
output_directory = Path(__file__).resolve().parent
evaluation_directory = project_root / "outputs/lpwm_drivor_intermediate_pdms_epoch3_v1"
previous_directory = project_root / "results/lpwm_drivor_planning_path_lora_v1/intermediate_update4500_20261007"
training_directory = project_root / "outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1"
controller_directory = project_root / "outputs/lpwm_drivor_particle_trends_every500_v1"
overlay_directory = project_root / "outputs/lpwm_drivor_particle_geometry_overlays_v1/update_004842"


def read_json(path):
    return json.loads(path.read_text())


def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (output_directory / "report.json").exists(), "Preserve completed reports"
    completed_evaluation = read_json(evaluation_directory / "evaluation_complete.json")
    assert completed_evaluation["complete"]
    latest_pdms = completed_evaluation["rows"][0]
    assert latest_pdms["completed_updates"] == 4842 and latest_pdms["scene_count"] == 95 and latest_pdms["failed"] == 0
    previous_report = read_json(previous_directory / "report.json")
    pdms_rows = previous_report["pdms_rows"] + [latest_pdms]
    after_scores = read_json(evaluation_directory / "update_004842/scores.json")
    assert len(after_scores) == len({row["token"] for row in after_scores}) == 95

    bootstrap_script = project_root / "results/lpwm_drivor_planning_path_lora_v1/intermediate_epoch2_20261007/generate_report.py"
    specification = importlib.util.spec_from_file_location("recorded_panel_bootstrap", bootstrap_script)
    bootstrap_module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(bootstrap_module)
    before_locations = {
        1614: project_root / "outputs/lpwm_drivor_intermediate_pdms_epoch1_v1/update_001614/scores.json",
        3228: project_root / "outputs/lpwm_drivor_intermediate_pdms_epoch2_v1/update_003228/scores.json",
        3500: project_root / "outputs/lpwm_drivor_intermediate_pdms_update3500_v1/update_003500/scores.json",
        4000: project_root / "outputs/lpwm_drivor_intermediate_pdms_update4000_v1/update_004000/scores.json",
        4500: previous_directory / "scores.json",
    }
    paired_changes = [
        {"before_update": update, "after_update": 4842,
         **bootstrap_module.paired_pdms_change(read_json(path), after_scores)}
        for update, path in before_locations.items()
    ]
    registration_checks = {}
    for registration_path in [training_directory.parent / "registration.json",
                              training_directory.parent / "oracle8_execution_registration.json",
                              project_root / "outputs/lpwm_drivor_optimized_execution_v1/execution_registration.json"]:
        registration_name = str(registration_path.relative_to(project_root))
        registered = read_json(registration_path)
        mismatches = [name for name, expected in registered["sources"].items()
                      if digest_file(project_root / name) != expected]
        assert not mismatches, mismatches
        registration_checks[registration_name] = {"source_count": len(registered["sources"]), "mismatches": mismatches}
    progress = read_json(training_directory / "progress.json")
    training_health = {}
    for rank in [0, 1]:
        rank_rows = [json.loads(line) for line in (training_directory / f"rank{rank}_training.jsonl").read_text().splitlines() if line.strip()]
        gradient_row = next(row for row in reversed(rank_rows) if row.get("gradient_norms"))
        training_health[str(rank)] = {
            "last_logged_update": rank_rows[-1]["completed_updates"],
            "nonfinite_loss_rows": sum(not all(np.isfinite(row[key]) for key in ["loss", "trajectory_loss", "final_score_loss"]) for row in rank_rows),
            "native_digests": sorted({row["frozen_native_sha256"] for row in rank_rows}),
            "last_gradient_update": gradient_row["completed_updates"],
            "gradient_groups": gradient_row["gradient_norms"],
        }
        assert training_health[str(rank)]["nonfinite_loss_rows"] == 0
        assert all(np.isfinite(value) and value > 0 for value in gradient_row["gradient_norms"].values())
    checked_at = datetime.now(ZoneInfo("Asia/Seoul"))
    rank0_rows = [json.loads(line) for line in (training_directory / "rank0_training.jsonl").read_text().splitlines() if line.strip()]
    execution_rows = [row for row in rank0_rows if row.get("execution_sha256") == progress["execution_sha256"]]
    eta = {}
    for window in (50, 100, 200):
        start, end = execution_rows[-window-1], execution_rows[-1]
        seconds_per_update = (end["elapsed_seconds"] - start["elapsed_seconds"]) / (end["completed_updates"] - start["completed_updates"])
        eta[str(window)] = {
            "seconds_per_update": seconds_per_update,
            "remaining_training_days": (40350-progress["completed_updates"])*seconds_per_update/86400,
            "targets": {str(target): (checked_at + timedelta(seconds=(target-progress["completed_updates"])*seconds_per_update)).isoformat()
                        for target in (5000, 6456, 40350)},
            "scope": "V1 training only; final evaluation and V2 excluded, current contention assumed"
        }
    representation_history = {}
    for update in (0, 1614, 3228, 4000, 4500, 4842):
        recorded = read_json(controller_directory / f"reports/update_{update:06d}.json")
        representation_history[str(update)] = {key: recorded[key] for key in ("metrics", "readouts", "representation_diagnostics")}
    representation = read_json(controller_directory / "reports/update_004842.json")
    replay_checks = read_json(evaluation_directory / "replay_checks.json")
    assert replay_checks["maximum_ade_difference"] < 1e-6
    assert replay_checks["max_card_used_bytes"] < 48_000_000_000
    comparison_to4500 = next(row for row in paired_changes if row["before_update"] == 4500)
    interval = comparison_to4500["paired_recording_bootstrap_95_ci"]
    report = {
        "checked_at": checked_at.isoformat(),
        "eta": eta,
        "representation_history": representation_history,
        "evaluated_updates": 4842,
        "evaluated_epoch_equivalent": 4842 / 1614,
        "progress_at_report": progress,
        "pdms_scope": completed_evaluation["scope"],
        "pdms_rows": pdms_rows,
        "paired_pdms_changes": paired_changes,
        "source_registration_checks": registration_checks,
        "training_health": training_health,
        "representation_metrics": representation["metrics"],
        "future_repeat_current_intervention": representation["representation_diagnostics"]["paired_interventions"]["future_repeat_current"],
        "replay_checks": replay_checks,
        "monitor_status": read_json(controller_directory / "status.json"),
        "training_changed": False,
        "interpretation": "Aggregate, left-turn and right-turn PDMS increased from4500, while straight PDMS declined. " + ("The paired difference from4500 has an interval excluding zero." if interval[0] > 0 else "The paired difference from4500 remains uncertain; its interval includes zero.") + " Future replacement worsens ADE on12 training scenes, while object readout is not consistently improving and geometry concentration in road regions is not established. This does not establish independent generalization, saturation, physical future accuracy or LPWM-only benefit.",
        "bootstrap_definition": "5000 paired recording-cluster resamples, seed71; not training-seed uncertainty, exploratory repeated diagnostic comparisons.",
        "source_sha256": {str(bootstrap_script.relative_to(project_root)): digest_file(bootstrap_script), str(Path(__file__).relative_to(project_root)): digest_file(Path(__file__))},
        "initial_sandbox_attempt": "Before GPU model initialization, NVML subprocess returned exit9. The same evaluation was rerun with host permission and completed.",
    }
    for name in ["registration.json", "evaluation_complete.json", "replay_checks.json"]:
        shutil.copy2(evaluation_directory / name, output_directory / name)
    for name in ["scores.json", "scores.csv", "summary.json"]:
        shutil.copy2(evaluation_directory / "update_004842" / name, output_directory / name)
    shutil.copy2(controller_directory / "reports/update_004842.json", output_directory / "representation_report_update4842.json")
    for source_name, destination_name in [
        ("overlay_report.json", "particle_overlay_report_update4842.json"),
        ("particle_geometry_comparison_with_overlay.png", "particles_before_after_and_overlay_update4842.png"),
        ("particle_geometry_overlay.png", "particles_overlay_update4842.png"),
    ]:
        shutil.copy2(overlay_directory / source_name, output_directory / destination_name)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    axes[0].plot([row["completed_updates"] for row in pdms_rows], [row["pdms"] for row in pdms_rows], "o-", color="#2563eb")
    for update in [1614, 3228, 4842]:
        row = next(row for row in pdms_rows if row["completed_updates"] == update)
        axes[0].annotate(f"{row['pdms']:.2f}", xy=(update, row["pdms"]), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(xlabel="Optimizer updates", ylabel="PDMS", ylim=(20, 95), title="Same 95 training-panel scenes")
    scenarios = ["straight", "left_turn", "right_turn"]
    for index, update in enumerate([4500, 4842]):
        row = next(row for row in pdms_rows if row["completed_updates"] == update)
        bars = axes[1].bar(np.arange(3)+(index-.5)*.36, [row["by_scene_type"][name]["pdms"] for name in scenarios], width=.36, color=["#94a3b8", "#2563eb"][index], label=f"{update:,} updates")
        axes[1].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    axes[1].set(xticks=np.arange(3), xticklabels=["Straight (41)", "Left (30)", "Right (24)"], ylabel="PDMS", ylim=(0, 105), title="4,500 updates versus epoch 3")
    axes[1].legend(loc="upper right", fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    figure.suptitle("Official NAVSIM v1 PDMS | training panel, not navtest")
    figure.savefig(output_directory / "pdms_progress_and_scenarios.png", dpi=160)
    figure.savefig(output_directory / "pdms_progress_and_scenarios.pdf")
    plt.close(figure)
    (output_directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"pdms":latest_pdms["pdms"],"by_scene_type":latest_pdms["by_scene_type"],"paired_changes":paired_changes,"progress":progress["completed_updates"],"eta":eta,"training_health":training_health,"source_checks":registration_checks,"replay_checks":replay_checks}, ensure_ascii=False))


if __name__ == "__main__":
    main()
