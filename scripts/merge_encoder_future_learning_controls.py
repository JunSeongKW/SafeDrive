"""Preserve original reports and combine the additional matched-target controls."""

import csv
import json
from pathlib import Path

from evaluate_drive_jepa_region_research_pdm import write_json
from report_drive_jepa_region_research import compare_metric
from summarize_drive_jepa_architecture_followup import paired_recording_comparison


def main():
    workspace = Path(__file__).resolve().parents[1]
    original_directory = workspace / "results/encoder_future_learning_v1"
    control_directory = workspace / "results/encoder_future_additional_controls_v1"
    original = json.loads((original_directory / "summary.json").read_text())
    controls = json.loads((control_directory / "summary.json").read_text())
    original_pdm = json.loads((original_directory / "development_pdm_results.json").read_text())
    control_pdm = json.loads((control_directory / "development_pdm_results.json").read_text())
    if original_pdm["windows"]["original_frozen"] != control_pdm["windows"]["original_frozen"]:
        raise RuntimeError("Preserved baseline PDM differs between the two evaluation stages")
    combined = dict(original)
    combined["scope"] = "48 final512 trained encoders plus original on192development windows; matched-target and stronger-auxiliary controls included; exploratory unadjusted CIs"
    combined["table"] = original["table"] + [row for row in controls["table"] if row["condition"] != "original_frozen"]
    combined["methods"] = {**original["methods"], **controls["methods"]}
    combined["specification"] = {**original["specification"], "conditions": {
        **original["specification"]["conditions"], **controls["specification"]["conditions"]}}
    combined["control_specification"] = controls["specification"]
    combined["training_run_count"] = original["training_run_count"] + controls["training_run_count"]
    combined["total_gradient_updates"] = original["total_gradient_updates"] + controls["total_gradient_updates"]
    combined["total_training_gpu_seconds"] = original["total_training_gpu_seconds"] + controls["total_training_gpu_seconds"]
    for key in ("paired_ade_comparisons", "paired_pdm_comparisons", "ade_context_breakdown", "pdm_context_breakdown"):
        combined[key] = {**original[key], **controls[key]}
    seeds = original["specification"]["seeds"]
    def rows_for(condition):
        matched = condition in controls["methods"]
        run_directory = workspace / "outputs" / ("encoder_future_additional_controls_v1" if matched else "encoder_future_learning_v1")
        pdm = control_pdm if matched else original_pdm
        ade_rows = {seed: json.loads((run_directory / f"{condition}_seed{seed}/results.json").read_text())["evaluations"]["512"]["development"]["windows"] for seed in seeds}
        pdm_rows = {seed: pdm["windows"][f"{condition}_seed{seed}"] for seed in seeds}
        schedules = {seed: json.loads((run_directory / f"{condition}_seed{seed}/results.json").read_text())["batch_schedule_sha256"] for seed in seeds}
        return ade_rows, pdm_rows, schedules
    for proposed, reference in controls["specification"]["matched_parent_comparisons"]:
        proposed_ade, proposed_pdm, proposed_schedules = rows_for(proposed)
        reference_ade, reference_pdm, reference_schedules = rows_for(reference)
        if proposed_schedules != reference_schedules:
            raise RuntimeError("Matched supplemental batch schedule differs")
        key = f"{proposed}_minus_{reference}"
        combined["paired_ade_comparisons"][key] = paired_recording_comparison(proposed_ade, reference_ade)
        combined["paired_pdm_comparisons"][key] = compare_metric(proposed_pdm, reference_pdm, "score")
    combined["isolated_masking_comparisons"] = controls["specification"]["matched_parent_comparisons"][:2]
    combined["masking_confound_note"] = "Original masked vs resampled-unmasked comparisons combine input masking with target diversity; use fixed-target controls to isolate masking."
    combined["trained_inference_audits"] = {
        "original36": json.loads((original_directory / "trained_inference_audit.json").read_text()),
        "additional12": json.loads((control_directory / "trained_inference_audit.json").read_text())}
    write_json(original_directory / "combined_summary.json", combined)
    with (original_directory / "combined_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined["table"][0]))
        writer.writeheader()
        writer.writerows(combined["table"])
    print(json.dumps(combined["table"], indent=2))
    print("MATCHED_ENCODER_CONTROLS_MERGED")


if __name__ == "__main__":
    main()
