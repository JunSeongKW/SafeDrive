"""Explicit stage gates and whole-development visual-retention diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from evaluate_lpwm_full_planning import digest, write_json
from summarize_lpwm_posttraining import paired_recording_interval


def check_stage1(specification):
    configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / configuration["output_directory"]
    root = PROJECT_ROOT / specification["output_directory"]
    original = json.loads((stage1_root / "adaptation_gate.json").read_text())
    evaluation = json.loads((stage1_root / "evaluation/posttrained/metrics.json").read_text())
    causal = json.loads((root / "stage1_causality_audit.json").read_text())
    checks = {"registered_stage1_adaptation_gate": original["adaptation_gate_passed"],
        "all_causal_interventions_pass": len(causal["scenes"]) >= specification["validation_gates"]["minimum_causal_audit_scenes"]
            and all(row["observed_particle_max_difference"] < 1e-5 and row["forecast_max_difference"] < 1e-5 for row in causal["scenes"]),
        "causal_audit_checkpoint_matches": causal["checkpoint_sha256"] == evaluation["checkpoint_sha256"],
        "noncollapsed_particles": all(np.isfinite(row["metrics"]["particle_feature_std"]) and row["metrics"]["particle_feature_std"] > 1e-4
            and row["metrics"]["mean_visible_particles"] > 1 for row in evaluation["records"])}
    for risk, value in original["driving_risk_breakdown"].items():
        checks["risk_coverage_" + risk] = value["recordings"] >= specification["validation_gates"]["minimum_risk_recordings"]
    for scenario in ("straight", "turn", "projected_overlap"):
        records = [row for row in evaluation["records"] if row["scenario"] == scenario]
        checks["scenario_coverage_" + scenario] = len({row["recording_group"] for row in records}) >= 5
    if "stage1_admission_amendment" in specification:
        from lpwm_stage2_admission import require_stage2_admission
        require_stage2_admission(specification, PROJECT_ROOT, stage1_root, stage1_root / "stage1/checkpoint.pt")
        del checks["registered_stage1_adaptation_gate"]
        checks["documented_stage1_experiment_admission"] = True
    report = {"passed": all(checks.values()), "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "checkpoint_sha256": evaluation["checkpoint_sha256"], "original_gate_sha256": digest(stage1_root / "adaptation_gate.json"),
        "original_gate_passed": original["adaptation_gate_passed"],
        "admission_amendment": specification.get("stage1_admission_amendment"),
        "interpretation": "Operational adaptation criteria; not proof of perfect adaptation, object identity, or planning utility"}
    write_json(root / "stage1_validation_gate.json", report)
    return report


def audit_stage1_causality(specification, gpu):
    import torch
    from run_lpwm_navsim_posttraining import initialize_model, make_video_tensor, check_gpu_reserve
    from visualize_lpwm_posttraining_progress import select_visualization_records
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{gpu}")
    check_gpu_reserve(gpu)
    configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    root = PROJECT_ROOT / configuration["output_directory"]
    checkpoint = root / "stage1/checkpoint.pt"
    manifest = json.loads((root / "manifest.json").read_text())
    frames = np.load(root / "rgb_frames.npy", mmap_mode="r")
    model, _ = initialize_model(device, 12, checkpoint)
    model.eval()
    rows = []
    with torch.inference_mode():
        for record in select_visualization_records(manifest):
            video = make_video_tensor([record], frames, device)
            changed_video = video.clone()
            changed_video[:, 4:] = 1 - changed_video[:, 4:]
            original = model.encode_all(video, deterministic=True)
            changed = model.encode_all(changed_video, deterministic=True)
            original_forecast, _ = model.sample_from_x(video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            changed_forecast, _ = model.sample_from_x(changed_video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            rows.append({"token": record["current_frame_token"], "scenario": record["scenario"],
                "observed_particle_max_difference": float((original["z"][:, :4] - changed["z"][:, :4]).abs().max()),
                "forecast_max_difference": float((original_forecast - changed_forecast).abs().max())})
    write_json(PROJECT_ROOT / specification["output_directory"] / "stage1_causality_audit.json",
        {"checkpoint_sha256": digest(checkpoint), "scenes": rows})


def validate_teacher(specification):
    root = PROJECT_ROOT / specification["teacher_directory"]
    completion = json.loads((root / "completion.json").read_text())
    vocabulary = json.loads((root / "vocabulary.json").read_text())
    labels = np.load(root / "candidate_metrics.npy", mmap_mode="r")
    entries = json.loads((root / "metric_cache_manifest.json").read_text())
    dev_entries = [entry for entry in entries if entry["split"] == "development"]
    oracle_pdms = float(np.mean([labels[entry["index"], :, -1].max() for entry in dev_entries]))
    thresholds = specification["validation_gates"]
    stage1_configuration = json.loads((PROJECT_ROOT / specification["stage1_config"]).read_text())
    stage1_root = PROJECT_ROOT / stage1_configuration["output_directory"]
    checks = {"coverage": completion["teacher_gate_passed"],
        "train_only_vocabulary": vocabulary["train_only"],
        "vocabulary_hash": digest(root / "trajectory_vocabulary.npy") == completion["vocabulary_sha256"],
        "labels_hash": digest(root / "candidate_metrics.npy") == completion["labels_sha256"],
        "planning_manifest_hash": digest(stage1_root / "planning_manifest.json") == completion["identity"]["manifest_sha256"],
        "official_scorer_parity": bool(completion["official_pair_scoring_parity"]) and all(row["maximum_difference"] < 2e-5 for row in completion["official_pair_scoring_parity"]),
        "oracle_ade": vocabulary["coverage"]["development"]["oracle_ade_meters"] <= thresholds["maximum_vocabulary_dev_oracle_ade_meters"],
        "oracle_ade_p95": vocabulary["coverage"]["development"]["oracle_ade_p95_meters"] <= thresholds["maximum_vocabulary_dev_oracle_ade_p95_meters"],
        "oracle_pdms": oracle_pdms >= thresholds["minimum_candidate_oracle_pdms"]}
    report = {"passed": all(checks.values()), "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "oracle_pdms": oracle_pdms, "dev_scenes": len(dev_entries), "completion_sha256": digest(root / "completion.json")}
    write_json(PROJECT_ROOT / specification["output_directory"] / "teacher_validation_gate.json", report)
    return report


def evaluate_world_retention(arguments, specification):
    import os
    import torch
    from train_lpwm_full_planning import load_training_inputs, make_planning_inputs
    from planning_aware_future_prediction.object_centric.lpwm_candidate_planner import build_planning_model
    from run_lpwm_navsim_posttraining import check_gpu_reserve
    from planning_aware_future_prediction.object_centric.lpwm_bridge import ARTIFACT_ROOT
    torch.set_num_threads(4)
    device = torch.device(f"cuda:{arguments.gpu}")
    _, _, stage1_root, checkpoint, manifest, targets, frames, _ = load_training_inputs(arguments.config, arguments.condition)
    root = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    model = build_planning_model(checkpoint, specification, arguments.condition, PROJECT_ROOT).to(device)
    trained_checkpoint = root / "checkpoint.pt"
    model.load_state_dict(torch.load(trained_checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    os.environ["TORCH_HOME"] = str(ARTIFACT_ROOT / "torch")
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    perceptual = LossLPIPS(normalized_rgb=False).to(device).eval().perceptual_loss
    references = json.loads((stage1_root / "evaluation/posttrained/metrics.json").read_text())["records"]
    indices_by_token = {row["current_frame_token"]: index for index, row in enumerate(manifest["records"])}
    rows_path = root / "world_retention_records.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    completed = {row["token"] for row in rows}
    with torch.inference_mode(), rows_path.open("a") as stream:
        for index, reference in enumerate(references):
            if reference["token"] in completed:
                continue
            if index % 16 == 0:
                check_gpu_reserve(arguments.gpu)
            chosen = np.array([indices_by_token[reference["token"]]])
            video, status, _ = make_planning_inputs(manifest["records"], chosen, frames, targets, device, include_future=True)
            with model.encoder_command(status):
                encoded = model.world_model(video, deterministic=True)
                reconstruction = encoded["rec_rgb"].reshape_as(video)[0]
                forecast, _ = model.world_model.sample_from_x(video[:, :4].contiguous(), num_steps=8, cond_steps=4,
                    deterministic=True, use_all_ctx=False, return_z=True, n_pred_eq_gt=False)
            row = {"token": reference["token"], "recording_group": reference["recording_group"],
                "risk_flags": reference["risk_flags"], "metrics": {
                    "reconstruction_lpips": float(perceptual(reconstruction * 2 - 1, video[0] * 2 - 1).mean()),
                    "forecast_lpips": float(perceptual(forecast[0, -8:] * 2 - 1, video[0, 4:] * 2 - 1).mean())}}
            rows.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
    assert {row["token"] for row in rows} == {row["token"] for row in references}
    comparisons = {name: paired_recording_interval(rows, references, name) for name in ("reconstruction_lpips", "forecast_lpips")}
    risk_comparisons = {}
    for risk in sorted({name for row in rows for name in row["risk_flags"]}):
        selected = [row for row in rows if row["risk_flags"][risk]]
        risk_comparisons[risk] = {name: paired_recording_interval(selected, references, name) for name in comparisons}
    write_json(root / "world_retention.json", {"records": rows, "comparisons": comparisons,
        "risk_comparisons": risk_comparisons,
        "stage1_means": {name: float(np.mean([row["metrics"][name] for row in references])) for name in comparisons},
        "checkpoint_sha256": digest(trained_checkpoint)})


def validate_planning_condition(arguments, specification):
    from prepare_lpwm_candidate_teacher import METRIC_NAMES
    condition_root = PROJECT_ROOT / specification["output_directory"] / arguments.condition
    initial = json.loads((condition_root / "evaluations/development__initial/results.json").read_text())["windows"]
    trained = json.loads((condition_root / "evaluations/development/results.json").read_text())["windows"]
    persistent = json.loads((condition_root / "evaluations/development__persistent_future/results.json").read_text())["windows"]
    teacher_root = PROJECT_ROOT / specification["teacher_directory"]
    entries = {row["token"]: row for row in json.loads((teacher_root / "metric_cache_manifest.json").read_text()) if row["split"] == "development"}
    labels = np.load(teacher_root / "candidate_metrics.npy", mmap_mode="r")
    refined = "refinement" in arguments.condition
    result_root = PROJECT_ROOT / specification["shared_results_directory"]
    def with_metrics(rows, evaluation_name):
        independently_scored = None
        if refined:
            independently_scored = {row["token"]: row for row in json.loads((result_root / "pdm" / f"{arguments.condition}__{evaluation_name}.json").read_text())["windows"]}
        scored = []
        for row in rows:
            if row["token"] not in entries:
                continue
            selected = labels[entries[row["token"]]["index"], row["candidate_index"]] if independently_scored is None else np.array([independently_scored[row["token"]][name] for name in METRIC_NAMES])
            scored.append({**row, "metrics": {**dict(zip(METRIC_NAMES, selected.tolist())), "ade_meters": row["ade_meters"],
                "candidate_metric_bce": row["candidate_metric_bce"]}})
        return scored
    scored_initial, scored_trained = with_metrics(initial, "development__initial"), with_metrics(trained, "development")
    scored_persistent = with_metrics(persistent, "development__persistent_future")
    comparisons = {name: paired_recording_interval(scored_trained, scored_initial, name) for name in ("score", "ade_meters", "candidate_metric_bce")}
    training = json.loads((condition_root / "training_summary.json").read_text())
    checks = {"full_training": not training["profile_only"] and training["epochs"] == specification["epochs"],
        "same_complete_development_tokens": len(trained) == 27076 and len({row["token"] for row in trained}) == 27076
            and {row["token"] for row in initial} == {row["token"] for row in trained},
        "pdms_improved_over_initial": comparisons["score"]["ci95"][0] > specification["validation_gates"]["minimum_dev_pdms_gain_over_initial"],
        "ade_improved_over_initial": comparisons["ade_meters"]["ci95"][1] < 0,
        "planning_modules_have_gradients": all(training["module_gradients"][name] > 0 for name in ("image_encoder", "context", "dynamics", "planner_and_command"))}
    if not arguments.condition.startswith("imitation"):
        checks["metric_calibration_improved"] = comparisons["candidate_metric_bce"]["ci95"][1] < -specification["validation_gates"]["minimum_dev_metric_calibration_improvement"]
    retention = json.loads((condition_root / "world_retention.json").read_text())
    if arguments.condition.endswith("plus_world"):
        for name, configuration_name in (("reconstruction_lpips", "maximum_world_reconstruction_relative_degradation"), ("forecast_lpips", "maximum_world_forecast_relative_degradation")):
            checks[name + "_retained"] = retention["comparisons"][name]["ci95"][1] <= specification["validation_gates"][configuration_name] * retention["stage1_means"][name]
    report = {"passed": all(checks.values()), "checks": checks, "failed_checks": [name for name, passed in checks.items() if not passed],
        "comparisons_with_untrained_planner": comparisons, "pdms_percent": 100 * float(np.mean([row["metrics"]["score"] for row in scored_trained])),
        "scored_development_scenes": len(scored_trained), "world_retention": retention["comparisons"],
        "world_retention_risk_breakdown": retention["risk_comparisons"],
        "predicted_future_vs_persistent_future_pdms": paired_recording_interval(scored_trained, scored_persistent, "score"),
        "checkpoint_sha256": training["checkpoint_sha256"], "interpretation": "Learning and retention gate, not a claim of superiority over Drive-JEPA"}
    risks_by_token = {row["token"]: row["risk_flags"] for row in retention["records"]}
    report["development_risk_breakdown"] = {}
    for risk in sorted({name for flags in risks_by_token.values() for name in flags}):
        selected = [row for row in scored_trained if risks_by_token.get(row["token"], {}).get(risk, False)]
        if selected:
            report["development_risk_breakdown"][risk] = {"scenes": len(selected), "pdms_percent": 100 * float(np.mean([row["metrics"]["score"] for row in selected])),
                "vs_initial": paired_recording_interval(selected, scored_initial, "score"),
                "forecast_vs_persistence": paired_recording_interval(selected, scored_persistent, "score")}
    report["risk_scope"] = "Risk metadata is available on the 7745 world-model development clips; other planning dev scenes contribute aggregate metrics only"
    write_json(condition_root / "validation_gate.json", report)
    if not refined:
        write_json(result_root / "pdm" / f"{arguments.condition}__development.json", {"windows": [dict(token=row["token"], recording_group=row["recording_group"], **{name: row["metrics"][name] for name in METRIC_NAMES}) for row in scored_trained],
            "summary": {name: float(np.mean([row["metrics"][name] for row in scored_trained])) for name in METRIC_NAMES},
            "scenes": len(scored_trained), "scope": "All eligible development tokens; cached official candidate scoring with per-segment parity audits"})
    write_json(result_root / f"{arguments.condition}_validation_gate.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("stage1-causality", "stage1-gate", "teacher-gate", "world-retention", "planning-gate"), required=True)
    parser.add_argument("--condition")
    parser.add_argument("--gpu", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    specification = json.loads(arguments.config.read_text())
    if arguments.mode == "stage1-causality":
        audit_stage1_causality(specification, arguments.gpu)
    elif arguments.mode == "world-retention":
        evaluate_world_retention(arguments, specification)
    else:
        report = {"stage1-gate": lambda: check_stage1(specification), "teacher-gate": lambda: validate_teacher(specification),
            "planning-gate": lambda: validate_planning_condition(arguments, specification)}[arguments.mode]()
        print(json.dumps(report), flush=True)
        if not report["passed"]:
            if arguments.mode == "planning-gate" and specification.get("continue_ablation_after_scientific_gate_failure", False):
                engineering_checks = ("full_training", "same_complete_development_tokens", "planning_modules_have_gradients")
                if all(report["checks"][name] for name in engineering_checks):
                    print("SCIENTIFIC_CRITERIA_NOT_MET_CONTINUE_REGISTERED_COMPARISON", flush=True)
                    raise SystemExit(0)
            raise SystemExit(2)
