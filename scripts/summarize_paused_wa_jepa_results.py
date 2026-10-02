"""Publish explicitly PARTIAL WA results after a verified user pause; no inference."""

import csv
import datetime
import hashlib
import json
from pathlib import Path


def main():
    workspace = Path(__file__).resolve().parents[1]
    result_root = workspace / "results/official_wa_jepa_reproduction"
    pause = json.loads((result_root / "paused_evaluation_state.json").read_text())
    output_path = result_root / "partial_navtest_at_drive_extension_20261002.json"
    csv_path = output_path.with_suffix(".csv")
    if output_path.exists() or csv_path.exists():
        raise RuntimeError("Preserve published partial result; do not overwrite")
    records = []
    output_root = workspace / "outputs/official_wa_jepa_reproduction"
    for filename, checksum in pause["scene_record_sha256_by_file"].items():
        payload = (output_root / filename).read_bytes()
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise RuntimeError("Saved paused raw result changed")
        records.extend(
            json.loads(line) for line in payload.splitlines() if line.strip()
        )
    assert len(records) == len({record["token"] for record in records})
    assert len(records) == pause["preserved_completed_scene_count"]
    valid_records = [record for record in records if record["valid"]]
    metrics = list(valid_records[0]["scores_fraction"])
    with (
        workspace / "results/official_drive_jepa_reproduction/official_scene_scores.csv"
    ).open() as handle:
        drive = {
            row["token"]: row
            for row in csv.DictReader(handle)
            if row["valid"].lower() == "true"
        }
    assert all(record["token"] in drive for record in valid_records)
    comparisons = {}
    for metric in metrics:
        wa_mean = (
            sum(record["scores_fraction"][metric] for record in valid_records)
            / len(valid_records)
            * 100
        )
        drive_mean = (
            sum(float(drive[record["token"]][metric]) for record in valid_records)
            / len(valid_records)
            * 100
        )
        comparisons[metric] = {
            "wa_partial_percent": wa_mean,
            "drive_same_scenes_percent": drive_mean,
            "difference_points": wa_mean - drive_mean,
        }
    with csv_path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token", "valid", *metrics])
        writer.writeheader()
        for record in sorted(records, key=lambda value: value["token"]):
            writer.writerow(
                {
                    "token": record["token"],
                    "valid": record["valid"],
                    **record.get("scores_fraction", {}),
                }
            )
    report = {
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "status": "USER_PAUSED_PARTIAL_NOT_FULL_BENCHMARK",
        "expected_scenes": pause["expected_scene_count"],
        "completed_scenes": len(records),
        "successful_scenes": len(valid_records),
        "failed_tokens": [record["token"] for record in records if not record["valid"]],
        "remaining_scenes": pause["remaining_scene_count"],
        "protocol": "NAVSIM_v1.1_navtest_PDMS_official_12step_seed0",
        "comparison_same_scene_subset_only": comparisons,
        "scene_csv": str(csv_path.relative_to(workspace)),
        "scene_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "raw_scene_sha256": pause["scene_record_sha256_by_file"],
        "no_automatic_resume": True,
        "sparse_results_preserved": "results/official_wa_jepa_reproduction/sparse_cost_summary.json",
    }
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
