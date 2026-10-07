"""Compare rank-paired training timing before and after the CPU-worker change."""
import json
from pathlib import Path
import statistics
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[3]
TRAINING = ROOT / "outputs/lpwm_drivor_planning_path_lora_v1/navsim_v1"
DESTINATION = Path(__file__).resolve().parent


def summarize(rank_rows):
    updates = sorted(set(rank_rows[0]) & set(rank_rows[1]))
    paired = [[rows[update] for rows in rank_rows] for update in updates]
    wall_mean = max((rows[updates[-1]]["elapsed_seconds"] - rows[updates[0]]["elapsed_seconds"])
                    / (len(updates) - 1) for rows in rank_rows)
    return {"updates": [updates[0], updates[-1]], "samples_per_rank": len(updates),
            "mean_rank_max_update_seconds": statistics.mean(max(row["seconds_this_update"] for row in pair) for pair in paired),
            "wall_seconds_per_update": wall_mean,
            "timing_means_per_rank": [{key: statistics.mean(row["timing_seconds"][key] for row in rows.values())
                                       for key in next(iter(rows.values()))["timing_seconds"]} for rows in rank_rows],
            "maximum_whole_card_bytes": max(row["card_used_bytes"] for pair in paired for row in pair)}


def main():
    all_rows = [[json.loads(line) for line in (TRAINING / f"rank{rank}_training.jsonl").read_text().splitlines()]
                for rank in range(2)]
    before = [{row["completed_updates"]: row for row in rows if 3038 <= row["completed_updates"] <= 3137}
              for rows in all_rows]
    after = [{row["completed_updates"]: row for row in rows if 3161 <= row["completed_updates"] <= 3170
              and row["execution_overrides"]["oracle_workers_per_rank"] == 8} for rows in all_rows]
    assert all(len(rows) == 100 for rows in before)
    assert all(len(rows) == 10 for rows in after), "Wait until update3170; exclude the first resumed update3160."
    before_summary, after_summary = summarize(before), summarize(after)
    resume = json.loads((TRAINING / "resume_execution_batch16_loader2_oracle8.json").read_text())
    progress = json.loads((TRAINING / "progress.json").read_text())
    remaining_updates = progress["total_updates"] - progress["completed_updates"]
    old_remaining = remaining_updates * before_summary["wall_seconds_per_update"]
    new_remaining = remaining_updates * after_summary["wall_seconds_per_update"]
    report = {"before": before_summary, "after": after_summary,
              "wall_duration_reduction_percent": 100 * (1 - after_summary["wall_seconds_per_update"] / before_summary["wall_seconds_per_update"]),
              "resume_state": resume, "only_setting_changed": "CPU oracle workers per rank4 -> 8",
              "effective_batch": 64, "microbatch_per_gpu": 16, "accumulation": 2, "loader_workers_per_rank": 2,
              "latest_completed_updates": progress["completed_updates"],
              "estimated_remaining_days_before": old_remaining / 86400,
              "estimated_remaining_days_after": new_remaining / 86400,
              "estimated_hours_saved_remaining_training": (old_remaining - new_remaining) / 3600,
              "estimated_training_end_kst": (datetime.now() + timedelta(seconds=new_remaining)).isoformat(timespec="seconds"),
              "limitations": ["Ten steady resumed updates; verify long-window timing in subsequent status checks.",
                               "Sequential measurements on shared hardware; other load and scene cost can vary.",
                               "Full model/optimizer/scheduler/RNG state restored; restarting a DataLoader is not a bitwise-identical continuation guarantee.",
                               "Time estimate is NAVSIMv1 training only; later evaluation and v2 training excluded."]}
    (DESTINATION / "training_speed_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
