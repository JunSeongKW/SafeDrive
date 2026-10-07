"""Summarize learning progress and distinguish branch use from future accuracy."""
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = Path(__file__).resolve().parent
TRAINING_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = TRAINING_ROOT / "navsim_v1"
CONTROLLER = ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1"
EVALUATION = ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3500_v1"


def read_json(path):
    return json.loads(path.read_text())


def main():
    assert not (DESTINATION / "assessment.json").exists(), "Preserve completed assessments"
    checked_at = datetime.now(ZoneInfo("Asia/Seoul"))
    progress = read_json(TRAINING / "progress.json")
    training_checks, loss_windows, gradient_samples = {}, {}, {}
    for rank in (0, 1):
        rows = [json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
        training_checks[str(rank)] = {"last_logged_update": rows[-1]["completed_updates"],
            "nonfinite_loss_rows": sum(not all(np.isfinite(row[key]) for key in ("loss", "trajectory_loss", "final_score_loss")) for row in rows),
            "native_digests": sorted({row["frozen_native_sha256"] for row in rows})}
        gradient_samples[str(rank)] = next(row for row in reversed(rows) if row.get("gradient_norms"))
        assert all(np.isfinite(value) and value > 0 for value in gradient_samples[str(rank)]["gradient_norms"].values())
        loss_windows[str(rank)] = {}
        for first, last in ((1601, 1700), (2901, 3000), (3401, 3500), (3501, 3600)):
            window = [row for row in rows if first <= row["completed_updates"] <= last]
            assert len(window) == 100
            loss_windows[str(rank)][f"{first}-{last}"] = {
                key: float(np.mean([row[key] for row in window]))
                for key in ("loss", "trajectory_loss", "final_score_loss")}
    source_checks = {}
    for name in ("registration.json", "oracle8_execution_registration.json"):
        registered = read_json(TRAINING_ROOT / name)
        mismatches = [relative for relative, expected in registered["sources"].items()
                      if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected]
        assert not mismatches, mismatches
        source_checks[name] = {"source_count": len(registered["sources"]), "mismatches": mismatches}
    completed = read_json(EVALUATION / "evaluation_complete.json")
    assert completed["complete"] and completed["rows"][0]["completed_updates"] == 3500
    prior_report = ROOT / "results/lpwm_drivor_planning_path_lora_v1/intermediate_epoch2_20261007"
    specification = importlib.util.spec_from_file_location("prior_panel_reporting", prior_report / "generate_report.py")
    reporting = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(reporting)
    after_scores = read_json(EVALUATION / "update_003500/scores.json")
    paired_changes = []
    for update, location in (
        (1614, ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch1_v1/update_001614/scores.json"),
        (3000, ROOT / "outputs/lpwm_drivor_intermediate_pdms_update3000_v1/update_003000/scores.json"),
        (3228, ROOT / "outputs/lpwm_drivor_intermediate_pdms_epoch2_v1/update_003228/scores.json")):
        paired_changes.append({"before_update": update, "after_update": 3500,
                               **reporting.paired_pdms_change(read_json(location), after_scores)})
    representation_rows = []
    for update in (0, 1614, 3000, 3228, 3500):
        report = read_json(CONTROLLER / f"reports/update_{update:06d}.json")
        representation_rows.append({"update": update, "metrics": report["metrics"],
            "readouts": report["readouts"], "distribution": report["representation_diagnostics"]["distribution"]["all"],
            "future_repeat_current": report["representation_diagnostics"]["paired_interventions"]["future_repeat_current"],
            "checkpoint": report["checkpoint"]})
    assessment = {"checked_at": checked_at.isoformat(), "progress": progress,
        "training_checks": training_checks, "gradient_samples": gradient_samples,
        "source_registration_checks": source_checks, "loss_windows": loss_windows,
        "loss_comparison_limitation": "Different training batches and online oracle targets; not a fixed-sample optimization comparison",
        "official_pdms": completed, "paired_pdms_changes": paired_changes,
        "evaluation_replay_checks": read_json(EVALUATION / "replay_checks.json"),
        "representations": representation_rows, "monitor_status": read_json(CONTROLLER / "status.json"),
        "interpretation": {
            "training_health": "Finite losses and positive planning gradients; native weights and registered sources preserved",
            "planning": "Same training-panel PDMS improves relative to epoch1; recent small changes remain uncertain and do not isolate LPWM adaptation",
            "future_branch": "At update3500 replacing future attributes with current attributes worsens 12-scene oracle score and ADE; this is an exploratory branch-use signal, with possible distribution shift and repeated testing",
            "future_accuracy": "Future displacement readouts do not establish improvement; errors exceed zero-displacement controls",
            "objective": "Planning-only training does not directly require predicted particle states to match physical future states",
            "scope": "Training distribution diagnostics, not held-out validation or navtest",
            "decision": "Keep the authorized 25-epoch run and every500/epoch diagnostics; check persistence at update4000 and epoch3 without changing the registered experiment"},
        "training_changes": False}
    for name in ("registration.json", "evaluation_complete.json", "replay_checks.json"):
        shutil.copy2(EVALUATION / name, DESTINATION / name)
    for name in ("scores.json", "scores.csv", "summary.json"):
        shutil.copy2(EVALUATION / "update_003500" / name, DESTINATION / name)
    (DESTINATION / "assessment.json").write_text(json.dumps(assessment, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"checked_at": assessment["checked_at"], "completed_updates": progress["completed_updates"],
        "pdms": completed["rows"][0]["pdms"], "paired_pdms_changes": paired_changes,
        "source_checks": source_checks, "training_checks": training_checks,
        "latest_future_intervention": representation_rows[-1]["future_repeat_current"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
