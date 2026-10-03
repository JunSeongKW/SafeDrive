"""Official development PDM, matched linear future probes, and paired reporting."""

import argparse
import csv
import json
import lzma
import os
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_official_drive_jepa_evaluation import official_configuration
from evaluate_drive_jepa_region_research_pdm import file_sha256, write_json
from report_drive_jepa_region_research import compare_metric, context_breakdown, scene_macro_ade
from summarize_drive_jepa_architecture_followup import paired_recording_comparison

WORKSPACE = Path(__file__).resolve().parents[1]
RUN_DIRECTORY = WORKSPACE / "outputs/encoder_future_learning_v1"
SHARE_DIRECTORY = WORKSPACE / "results/encoder_future_learning_v1"
CONFIGURATION = WORKSPACE / "configs/encoder_future_learning/controlled_comparison_v1.json"


def load_predictions(specification):
    baseline = json.loads((RUN_DIRECTORY / "original_baseline.json").read_text())
    predictions = {"original_frozen": baseline["windows"]}
    reports = {}
    for condition in specification["conditions"]:
        for seed in specification["seeds"]:
            name = f"{condition}_seed{seed}"
            report = json.loads((RUN_DIRECTORY / name / "results.json").read_text())
            if report["completed_updates"] != specification["joint_updates"] or report["configuration_sha256"] != file_sha256(CONFIGURATION):
                raise RuntimeError("Incomplete or mismatched trained model")
            reports[name] = report
            predictions[name] = report["evaluations"][str(specification["joint_updates"])]["development"]["windows"]
    return predictions, reports


def score_official_development(specification, predictions):
    from hydra.utils import instantiate

    _, _, configuration, official_root = official_configuration(WORKSPACE)
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score

    manifest = json.loads((WORKSPACE / specification["reused_metric_manifest"]).read_text())
    expected_tokens = {entry["token"] for entry in manifest}
    if len(manifest) != 192 or len(expected_tokens) != 192:
        raise RuntimeError("Expected192 unique development metric windows")
    by_token = {name: {row["token"]: row for row in rows} for name, rows in predictions.items()}
    if any(len(rows) != 192 or set(rows) != expected_tokens for rows in by_token.values()):
        raise RuntimeError("Prediction tokens differ from official development cache")
    simulator, scorer = instantiate(configuration.simulator), instantiate(configuration.scorer)
    if simulator.proposal_sampling != scorer.proposal_sampling:
        raise RuntimeError("Official scorer sampling mismatch")
    scoring_directory = RUN_DIRECTORY / "development_pdm"
    scoring_directory.mkdir(exist_ok=True)
    prediction_digest = __import__("hashlib").sha256(json.dumps(predictions, sort_keys=True).encode()).hexdigest()
    scores = {name: [] for name in predictions}
    started = time.perf_counter()
    for window_index, entry in enumerate(manifest):
        if file_sha256(entry["metric_cache_file"]) != entry["cache_sha256"]:
            raise RuntimeError("Preserved official metric cache changed")
        window_path = scoring_directory / "windows" / f"{entry['token']}.json"
        if window_path.exists():
            saved = json.loads(window_path.read_text())
            if saved["prediction_sha256"] != prediction_digest:
                raise RuntimeError("Resume PDM predictions differ")
            window_scores = saved["scores"]
        else:
            with lzma.open(entry["metric_cache_file"], "rb") as stream:
                metric_cache = pickle.load(stream)
            window_scores = {}
            for name, rows in by_token.items():
                trajectory = Trajectory(np.asarray(rows[entry["token"]]["trajectory"], dtype=np.float32))
                metrics = asdict(pdm_score(metric_cache, trajectory, simulator.proposal_sampling, simulator, scorer))
                if not all(np.isfinite(value) for value in metrics.values()):
                    raise RuntimeError("Non-finite official metric")
                window_scores[name] = {"token": entry["token"], "recording": entry["recording"], **metrics}
            write_json(window_path, {"prediction_sha256": prediction_digest, "scores": window_scores})
        for name, row in window_scores.items():
            scores[name].append(row)
        if (window_index + 1) % 8 == 0:
            write_json(scoring_directory / "progress.json", {"completed_windows": window_index + 1,
                "prediction_count": len(predictions), "wall_seconds": time.perf_counter() - started})
            print(f"ENCODER_PDM {window_index + 1}/192 x{len(predictions)}", flush=True)
    metrics = {name: {key: float(np.mean([row[key] for row in rows]))
        for key in rows[0] if key not in ("token", "recording")} for name, rows in scores.items()}
    result = {"summary": metrics, "windows": scores, "official_source": str(official_root),
        "prediction_sha256": prediction_digest, "wall_seconds": time.perf_counter() - started,
        "scope": "192 development windows; not independent test or navtest benchmark"}
    write_json(SHARE_DIRECTORY / "development_pdm_results.json", result)
    return result


def prepare_common_probe_targets(specification):
    if "prepared_cache_directory" in specification:
        parent_targets = WORKSPACE / specification["prepared_cache_directory"] / "common_representation_probe_targets.pt"
        wait_started = time.perf_counter()
        while not parent_targets.exists():
            if time.perf_counter() - wait_started > 1800:
                raise RuntimeError("Parent probe target preparation wait cap")
            time.sleep(10)
        return torch.load(parent_targets, map_location="cpu", weights_only=True)
    prepared_path = RUN_DIRECTORY / "common_representation_probe_targets.pt"
    if prepared_path.exists():
        return torch.load(prepared_path, map_location="cpu", weights_only=True)
    records = json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
    projection = torch.randn(1024, 16, generator=torch.Generator().manual_seed(9173)) / 32
    future_changes, valid_horizons, ego_statuses = [], [], []
    for record in records:
        cached_window = torch.load(str(RUN_DIRECTORY / "prefix_cache" / f"{record['current_frame_token']}.pt"),
                                   map_location="cpu", weights_only=True, mmap=True)
        def compress_targets(region_features):
            normalized_features = F.layer_norm(region_features, (1024,))
            projected_features = normalized_features @ projection
            return projected_features.reshape(-1, 4, 2, 8, 2, 16).mean((2, 4)).flatten(1)
        current_targets = compress_targets(cached_window["current_teacher_regions"])
        future_targets = compress_targets(cached_window["future_teacher_regions"])
        future_changes.append(future_targets - current_targets)
        valid_horizons.append(cached_window["future_region_valid_mask"].all(-1))
        ego_statuses.append(cached_window["ego_status"])
    result = {"future_teacher_change": torch.stack(future_changes), "future_valid_horizons": torch.stack(valid_horizons),
        "ego_status": torch.stack(ego_statuses),
        "train_indices": torch.tensor([index for index, record in enumerate(records) if record["split"] == "train"]),
        "development_indices": torch.tensor([index for index, record in enumerate(records) if record["split"] == "development"])}
    torch.save(result, prepared_path)
    return result


def evaluate_common_future_probes(specification, predictions):
    common = prepare_common_probe_targets(specification)
    train_indices, development_indices = common["train_indices"], common["development_indices"]
    results = {}
    for name in predictions:
        feature_path = (RUN_DIRECTORY / "original_representation_probe_features.pt" if name == "original_frozen"
                        else RUN_DIRECTORY / name / "representation_probe_features.pt")
        saved_features = torch.load(feature_path, map_location="cpu", weights_only=True)
        train_features = torch.cat((saved_features["train_features"], common["ego_status"][train_indices]), -1).double()
        development_features = torch.cat((saved_features["development_features"], common["ego_status"][development_indices]), -1).double()
        feature_mean = train_features.mean(0)
        feature_std = train_features.std(0).clamp_min(1e-3)
        train_features = (train_features - feature_mean) / feature_std
        development_features = (development_features - feature_mean) / feature_std
        horizon_results = []
        for future_step in range(4):
            train_valid = common["future_valid_horizons"][train_indices, future_step]
            development_valid = common["future_valid_horizons"][development_indices, future_step]
            train_targets = common["future_teacher_change"][train_indices, future_step][train_valid].double()
            development_targets = common["future_teacher_change"][development_indices, future_step][development_valid].double()
            # Closed-form ridge, fixed alpha10 and no dev hyperparameter selection.
            design = train_features[train_valid]
            design_mean, target_mean = design.mean(0), train_targets.mean(0)
            design = design - design_mean
            centered_targets = train_targets - target_mean
            dual_weights = torch.linalg.solve(design @ design.T + 10 * torch.eye(design.shape[0], dtype=torch.float64), centered_targets)
            predictions_delta = (development_features[development_valid] - design_mean) @ design.T @ dual_weights + target_mean
            errors = (predictions_delta - development_targets).square().mean(-1)
            persistence_errors = development_targets.square().mean(-1)
            horizon_results.append({"future_tubelet_index": future_step, "train_valid_windows": int(train_valid.sum()),
                "development_valid_windows": int(development_valid.sum()), "future_change_mse": float(errors.mean()),
                "persistence_mse": float(persistence_errors.mean()), "mse_over_persistence": float(errors.mean() / persistence_errors.mean()),
                "valid_development_indices": development_indices[development_valid].tolist(), "window_mse": errors.tolist()})
        valid_counts = np.asarray([entry["development_valid_windows"] for entry in horizon_results])
        results[name] = {"by_horizon": horizon_results,
            "future_change_mse": float(np.average([entry["future_change_mse"] for entry in horizon_results], weights=valid_counts)),
            "scope": "same520D linear readout; predict changes in fixed-teacher features; not object-state or causal-understanding proof"}
        print(f"ENCODER_PROBE {name} mse={results[name]['future_change_mse']:.6f}", flush=True)
    write_json(SHARE_DIRECTORY / "representation_probe_results.json", {"ridge_alpha": 10,
        "feature_projection_seed": 9173, "target": "normalized teacher region feature change from current to future",
        "methods": results})
    return results


def summarize_experiment(specification, predictions, reports, pdm_results, probe_results):
    records = json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
    records_by_token = {record["current_frame_token"]: record for record in records}
    seeds = specification["seeds"]
    ade_rows = {"original_frozen": {seed: predictions["original_frozen"] for seed in seeds}}
    pdm_rows = {"original_frozen": {seed: pdm_results["windows"]["original_frozen"] for seed in seeds}}
    table = [{"condition": "original_frozen", "development_ade_m": scene_macro_ade(predictions["original_frozen"]),
        "development_pdm_percent": pdm_results["summary"]["original_frozen"]["score"] * 100,
        "ade_seed_std_m": 0., "pdm_seed_std_pp": 0.,
        "future_probe_mse": probe_results["original_frozen"]["future_change_mse"]}]
    methods = {}
    for condition in specification["conditions"]:
        names = [f"{condition}_seed{seed}" for seed in seeds]
        ade_rows[condition] = {seed: predictions[name] for seed, name in zip(seeds, names)}
        pdm_rows[condition] = {seed: pdm_results["windows"][name] for seed, name in zip(seeds, names)}
        ade_values = [scene_macro_ade(predictions[name]) for name in names]
        pdm_values = [pdm_results["summary"][name]["score"] * 100 for name in names]
        table.append({"condition": condition, "development_ade_m": float(np.mean(ade_values)),
            "development_pdm_percent": float(np.mean(pdm_values)), "ade_seed_std_m": float(np.std(ade_values, ddof=1)),
            "pdm_seed_std_pp": float(np.std(pdm_values, ddof=1)),
            "future_probe_mse": float(np.mean([probe_results[name]["future_change_mse"] for name in names]))})
        methods[condition] = {"by_seed": {seed: {"development_ade_m": ade, "development_pdm_percent": pdm}
            for seed, ade, pdm in zip(seeds, ade_values, pdm_values)},
            "pdm_components_percent": {metric: float(np.mean([pdm_results["summary"][name][metric] * 100 for name in names]))
                for metric in pdm_results["summary"]["original_frozen"]},
            "intent_intervention": {seed: reports[name]["intent_intervention"] for seed, name in zip(seeds, names)},
            "gradient_contracts": {seed: reports[name]["gradient_contract"] for seed, name in zip(seeds, names)},
            "train_ade_m": float(np.mean([reports[name]["evaluations"][str(specification["joint_updates"])]["train"]["summary"]["scene_macro_xy_ade_m"] for name in names])),
            "training_seconds": [reports[name]["training_seconds"] for name in names],
            "peak_allocated_gib": max(reports[name]["peak_allocated_gib"] for name in names),
            "trainable_encoder_parameters": reports[names[0]]["trainable_encoder_parameters"]}
    comparisons = specification["primary_comparisons"] + [[condition, "original_frozen"] for condition in specification["conditions"]]
    result = {"scope": f"matched development-only encoder continuation, final512, all{len(reports)}runs; unadjusted exploratory multiple comparisons",
        "table": table, "methods": methods, "specification": specification,
        "paired_ade_comparisons": {f"{proposed}_minus_{reference}": paired_recording_comparison(ade_rows[proposed], ade_rows[reference])
                                   for proposed, reference in comparisons},
        "paired_pdm_comparisons": {f"{proposed}_minus_{reference}": compare_metric(pdm_rows[proposed], pdm_rows[reference], "score")
                                   for proposed, reference in comparisons},
        "ade_context_breakdown": {condition: context_breakdown(rows, ade_rows["original_frozen"], records_by_token)
                                   for condition, rows in ade_rows.items()},
        "pdm_context_breakdown": {condition: context_breakdown(rows, pdm_rows["original_frozen"], records_by_token, "score")
                                   for condition, rows in pdm_rows.items()},
        "training_run_count": len(reports), "total_gradient_updates": sum(report["completed_updates"] for report in reports.values()),
        "total_training_gpu_seconds": sum(report["training_seconds"] for report in reports.values())}
    for seed in seeds:
        schedule_hashes = {reports[f"{condition}_seed{seed}"]["batch_schedule_sha256"] for condition in specification["conditions"]}
        if len(schedule_hashes) != 1:
            raise RuntimeError("Paired batch schedules differ")
    write_json(SHARE_DIRECTORY / "summary.json", result)
    with (SHARE_DIRECTORY / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(SHARE_DIRECTORY / "validation.json", {"all_runs_complete": True, "paired_schedules_match": True,
        "all_encoders_changed": all(report["updated_encoder_hash"] != report["original_encoder_hash"] for report in reports.values()),
        "original_model_preserved": all(json.loads((RUN_DIRECTORY / f"train_worker{worker}/completion.json").read_text())["original_model_preserved"] for worker in (0, 1)),
        "input_mask_contract": json.loads((RUN_DIRECTORY / "input_mask_contract.json").read_text()),
        "window_score_count": sum(len(rows) for rows in pdm_results["windows"].values()),
        "configuration_sha256": file_sha256(CONFIGURATION)})
    print(json.dumps(table, indent=2), flush=True)


def main():
    global CONFIGURATION, RUN_DIRECTORY, SHARE_DIRECTORY
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-for-training", action="store_true")
    parser.add_argument("--configuration", type=Path, default=CONFIGURATION)
    parser.add_argument("--run-directory", type=Path, default=RUN_DIRECTORY)
    parser.add_argument("--share-directory", type=Path, default=SHARE_DIRECTORY)
    args = parser.parse_args()
    CONFIGURATION = args.configuration.resolve()
    RUN_DIRECTORY, SHARE_DIRECTORY = args.run_directory.resolve(), args.share_directory.resolve()
    torch.set_num_threads(2)
    specification = json.loads(CONFIGURATION.read_text())
    SHARE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.wait_for_training:
        while not all((RUN_DIRECTORY / f"train_worker{worker}/completion.json").exists() for worker in (0, 1)):
            if time.perf_counter() - started > specification["maximum_worker_wall_seconds"] + 600:
                raise RuntimeError("Training completion wait cap reached")
            time.sleep(20)
    predictions, reports = load_predictions(specification)
    pdm_results = score_official_development(specification, predictions)
    probe_results = evaluate_common_future_probes(specification, predictions)
    summarize_experiment(specification, predictions, reports, pdm_results, probe_results)
    write_json(SHARE_DIRECTORY / "completion.json", {"complete": True, "wall_seconds": time.perf_counter() - started})
    print("ENCODER_EVALUATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
