"""Archive runtime comparisons and verify the resumed production run."""
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "outputs/lpwm_drivor_optimized_execution_v1"
TRAINING_ROOT = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1"
TRAINING = TRAINING_ROOT / "navsim_v1"
RESULTS = ROOT / "results/lpwm_drivor_planning_path_lora_v1/optimized_execution_20261007"


def read_json(path):
    return json.loads(path.read_text())


def main():
    selection = read_json(AUDIT / "selection.json")
    execution = selection["selected"]["execution_configuration"]
    execution_hash = hashlib.sha256((ROOT / execution).read_bytes()).hexdigest()
    preserved = read_json(AUDIT / "preserved_resume_state.json")
    resumed = read_json(TRAINING / ("resume_" + Path(execution).stem + ".json"))
    assert resumed["checkpoint_sha256"] == preserved["checkpoint_sha256"]
    assert resumed["completed_updates"] == resumed["optimizer_steps_min"] == resumed["optimizer_steps_max"] == preserved["completed_updates"]
    assert resumed["scheduler_state_restored"] and resumed["rank_rng_restored_before_training"]
    rows_by_rank = [[json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
                    for rank in (0, 1)]
    rows_by_rank = [[row for row in rows if row["execution_sha256"] == execution_hash] for rows in rows_by_rank]
    count = min(map(len, rows_by_rank))
    assert count >= 8, "Observe at least eight real resumed updates"
    rows_by_rank = [rows[:count] for rows in rows_by_rank]
    for rows in rows_by_rank:
        assert rows[0]["completed_updates"] == preserved["completed_updates"] + 1
        assert all(row["completed_updates"] == preserved["completed_updates"] + index + 1 for index, row in enumerate(rows))
        assert all(row["card_used_bytes"] < 48_000_000_000 for row in rows)
        assert all(value > 0 for value in rows[0]["gradient_norms"].values())
    durations = [max(rows[index]["seconds_this_update"] for rows in rows_by_rank) for index in range(3, count)]
    wall_seconds = max((rows[-1]["elapsed_seconds"] - rows[2]["elapsed_seconds"]) / (count-3) for rows in rows_by_rank)
    baseline = read_json(AUDIT / "benchmarks/original_before/summary.json")
    repeat = read_json(AUDIT / "benchmarks/original_after/summary.json")
    worker_only = read_json(AUDIT / "benchmarks/oracle16/summary.json")
    numeric_variation = {}
    for name, candidate in (("original_repeat", repeat), ("worker_and_finite_check", worker_only)):
        numeric_variation[name] = {"first_loss_max_difference": max(abs(baseline["rank_losses"][rank][0]-candidate["rank_losses"][rank][0]) for rank in (0, 1)),
            "max_loss_difference_over_updates": max(abs(expected-observed) for rank in (0, 1) for expected, observed in zip(baseline["rank_losses"][rank], candidate["rank_losses"][rank]))}
    registration_checks = []
    for path in (TRAINING_ROOT / "registration.json", TRAINING_ROOT / "oracle8_execution_registration.json", AUDIT / "execution_registration.json"):
        registration = read_json(path)
        mismatches = [name for name, checksum in registration["sources"].items() if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum]
        assert not mismatches
        registration_checks.append({"file": str(path.relative_to(ROOT)), "sources": len(registration["sources"]), "mismatches": mismatches})
    now = datetime.now()
    remaining = rows_by_rank[0][-1]["total_updates"] - rows_by_rank[0][-1]["completed_updates"]
    report = {"checked_at_kst": now.isoformat(), "selection": selection, "preserved_resume": preserved,
              "restored_resume": resumed, "observed_production_updates": count,
              "production_update_range": [rows_by_rank[0][0]["completed_updates"], rows_by_rank[0][-1]["completed_updates"]],
              "production_steady_median_update_seconds": statistics.median(durations),
              "production_steady_wall_seconds_per_update": wall_seconds,
              "production_peak_card_bytes": max(row["card_used_bytes"] for rows in rows_by_rank for row in rows),
              "gradient_groups_positive_by_rank": [rows[0]["gradient_norms"] for rows in rows_by_rank],
              "epoch_hours_at_current_speed": wall_seconds * 1614 / 3600,
              "remaining_v1_training_days": wall_seconds * remaining / 86400,
              "estimated_v1_training_end_kst": (now + timedelta(seconds=wall_seconds*remaining)).isoformat(),
              "repeated_run_numeric_variation": numeric_variation,
              "source_checks": registration_checks, "queue": read_json(TRAINING_ROOT / "queue_status.json"),
              "monitor": read_json(ROOT / "outputs/lpwm_drivor_particle_trends_every500_v1/status.json"),
              "limitations": ["Short matched-checkpoint throughput benchmarks and initial production observations on shared hardware.",
                              "SDPA retains dropout probability but changes random mask draws and floating-point ordering.",
                              "CPU fixed-proposal scores were bitwise equal; repeated GPU training is not bitwise identical.",
                              "No new PDMS or long-run convergence claim; regular representation and final evaluation queue retained.",
                              "ETA excludes final evaluation, subsequent v2 training and future pauses."]}
    RESULTS.mkdir(parents=True, exist_ok=True)
    assert not (RESULTS / "report.json").exists(), "Preserve the completed report"
    for name in ("runtime_compatibility.json", "preserved_resume_state.json", "benchmark_comparison.json", "selection.json", "execution_registration.json", "resumed_launches.json"):
        shutil.copyfile(AUDIT / name, RESULTS / name)
    shutil.copyfile(AUDIT / "oracle_worker_benchmark/report.json", RESULTS / "oracle_worker_comparison.json")
    for folder in (AUDIT / "benchmarks").iterdir():
        destination = RESULTS / "benchmarks" / folder.name
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("summary.json", "passed.json", "rank0_training.jsonl", "rank1_training.jsonl"):
            shutil.copyfile(folder / name, destination / name)
    (RESULTS / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("production_update_range", "production_steady_median_update_seconds", "production_steady_wall_seconds_per_update", "production_peak_card_bytes", "remaining_v1_training_days", "estimated_v1_training_end_kst")}, indent=2))


if __name__ == "__main__":
    main()
