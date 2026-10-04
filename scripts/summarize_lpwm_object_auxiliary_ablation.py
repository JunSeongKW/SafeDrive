"""Paired development decision for optional object supervision, no test tuning."""
import argparse
import json
from pathlib import Path

from evaluate_lpwm_full_planning import write_json
from summarize_lpwm_posttraining import paired_recording_interval

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def summarize(config_path):
    specification = json.loads(config_path.read_text())
    root = PROJECT_ROOT / specification["output_directory"]
    shared = PROJECT_ROOT / specification["shared_results_directory"]
    baseline, auxiliary = specification["conditions"]
    assert baseline == "metric_plus_world" and auxiliary == "metric_object_future_plus_world"
    gates = {condition: json.loads((root / condition / "validation_gate.json").read_text()) for condition in (baseline, auxiliary)}
    rows = {}
    for condition in (baseline, auxiliary):
        pdm = json.loads((shared / "pdm" / (condition + "__development.json")).read_text())["windows"]
        planning = {row["token"]: row for row in json.loads((root / condition / "evaluations/development/results.json").read_text())["windows"]}
        rows[condition] = [{"token": row["token"], "recording_group": row["recording_group"],
            "metrics": {"score": row["score"], "ade_meters": planning[row["token"]]["ade_meters"]}} for row in pdm]
    comparison = {metric: paired_recording_interval(rows[auxiliary], rows[baseline], metric) for metric in ("score", "ade_meters")}
    world = {condition: json.loads((root / condition / "world_retention.json").read_text())["records"] for condition in (baseline, auxiliary)}
    world_comparison = {metric: paired_recording_interval(world[auxiliary], world[baseline], metric)
        for metric in ("reconstruction_lpips", "forecast_lpips")}
    retained = all(gates[auxiliary]["checks"][name] for name in ("reconstruction_lpips_retained", "forecast_lpips_retained"))
    supports_auxiliary = comparison["score"]["ci95"][0] > 0 and retained and gates[auxiliary]["passed"]
    report = {"complete": True, "conditions": gates, "auxiliary_minus_no_auxiliary": comparison,
        "world_auxiliary_minus_no_auxiliary": world_comparison,
        "object_gt_adoption": "supported_on_development_pending_independent_confirmation" if supports_auxiliary else "not_established_keep_GT_optional",
        "decision_rule": "Paired PDMS improvement CI > 0 with retained world-model performance and complete learning validation; no selection by a favorable individual seed/epoch",
        "limitations": ["One training seed per condition; CI clusters recordings, not seeds", "Development split was previously used for adaptation diagnostics",
            "Object auxiliary head loss is not an independent frozen feature probe", "Native mask/independent future-state readout and causal object interventions remain separate diagnostics"],
        "automatic_independent_test_started": False}
    write_json(shared / "ablation_summary.json", report)
    print(json.dumps({"object_gt_adoption": report["object_gt_adoption"], "comparison": comparison}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    summarize(parser.parse_args().config.resolve())
