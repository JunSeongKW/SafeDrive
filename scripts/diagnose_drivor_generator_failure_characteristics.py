#!/usr/bin/env python
"""Relate generator set geometry/current observations to original scorer errors."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.stats import rankdata, spearmanr

from calibrate_official_toad_navsim_v1_scorer import PROJECT_ROOT, file_sha256, protected_files_audit, write_json


FEATURE_LABELS = {
    "median_path_length_m": "Median path length",
    "pairwise_rms_distance_median_m": "Candidate path dispersion",
    "endpoint_lateral_standard_deviation_m": "Lateral endpoint spread",
    "endpoint_heading_circular_standard_deviation_deg": "Heading spread",
    "terminal_speed_standard_deviation_m_per_s": "Terminal speed spread",
    "median_endpoint_heading_abs_deg": "Turn magnitude",
    "median_longitudinal_jerk_proxy_m_per_s3": "Waypoint jerk proxy",
    "ego_speed_m_per_s": "Current ego speed",
    "ego_longitudinal_acceleration_abs_m_per_s2": "Current acceleration",
    "past_yaw_rate_abs_rad_per_s": "Observed yaw rate",
    "vehicle_count_within30m": "Vehicles within 30 m",
    "vulnerable_actor_count_within30m": "Pedestrians/bicycles within 30 m",
    "forward_actor_count_within30m": "Actors in front corridor",
    "nearest_forward_actor_center_distance_capped_m": "Nearest front actor (capped)",
    "predicted_top1_top2_margin_pdms_points": "Predicted top-2 score gap",
}


def read_current_scene_metadata(records, configuration):
    metadata = [None] * len(records)
    known_names = set()
    grouped_logs = {}
    for scene_index, record in enumerate(records):
        grouped_logs.setdefault(record["log_name"], []).append((scene_index, record))
    for log_name, log_records in grouped_logs.items():
        with (PROJECT_ROOT / "dataset/navsim_logs/test" / (log_name + ".pkl")).open("rb") as stream:
            frames = pickle.load(stream)
        for scene_index, record in log_records:
            current = frames[record["current_frame_index"]]
            assert current["token"] == record["token"]
            np.testing.assert_allclose(current["ego_dynamic_state"], record["ego_history"][-1][3:7], atol=1e-6)
            assert int(np.argmax(current["driving_command"])) == record["command_index"]
            boxes = np.asarray(current["anns"]["gt_boxes"], dtype=np.float64)
            names = np.asarray(current["anns"]["gt_names"])
            known_names.update(names.tolist())
            vehicle_mask = names == "vehicle"
            vulnerable_mask = np.isin(names, ["pedestrian", "bicycle"])
            actor_mask = vehicle_mask | vulnerable_mask
            radial_distance = np.linalg.norm(boxes[:, :2], axis=1)
            near_mask = radial_distance <= configuration["nearby_actor_radius_m"]
            front_mask = actor_mask & (boxes[:, 0] > 0) & (np.abs(boxes[:, 1]) <= configuration["forward_actor_corridor_half_width_m"])
            front_distances = radial_distance[front_mask]
            current_ego = np.asarray(current["ego_dynamic_state"])
            metadata[scene_index] = {
                "vehicle_count_within30m": int(np.sum(vehicle_mask & near_mask)),
                "vulnerable_actor_count_within30m": int(np.sum(vulnerable_mask & near_mask)),
                "pedestrian_count_within30m": int(np.sum((names == "pedestrian") & near_mask)),
                "bicycle_count_within30m": int(np.sum((names == "bicycle") & near_mask)),
                "forward_actor_count_within30m": int(np.sum(front_mask & near_mask)),
                "nearest_forward_actor_center_distance_capped_m": min(float(front_distances.min()) if len(front_distances) else configuration["forward_actor_distance_cap_m"], configuration["forward_actor_distance_cap_m"]),
                "ego_speed_m_per_s": float(np.linalg.norm(current_ego[:2])),
                "ego_longitudinal_acceleration_abs_m_per_s2": float(abs(current_ego[2])),
                "past_yaw_rate_abs_rad_per_s": float(abs(record["boundary"]["past_yaw_rate"])),
                "current_gt_object_count": len(boxes), "map_location": current["map_location"],
                "current_traffic_light_record_count": len(current["traffic_lights"]),
                "current_annotation_timestamp": int(current["timestamp"]),
            }
        del frames
    return metadata, sorted(known_names)


def compute_set_geometry(paths, initial_speed):
    points = paths[..., :2].astype(np.float64)
    previous = np.concatenate((np.zeros_like(points[:, :1]), points[:, :-1]), axis=1)
    speed = np.linalg.norm(points - previous, axis=-1) / 0.5
    acceleration = (speed - np.concatenate((np.full_like(speed[:, :1], initial_speed), speed[:, :-1]), axis=1)) / 0.5
    jerk = np.diff(acceleration, axis=1) / 0.5
    distances = np.sqrt(np.square(points[:, None] - points[None]).sum(axis=-1).mean(axis=-1))
    pair_indices = np.triu_indices(len(paths), 1)
    heading = np.arctan2(np.sin(paths[:, -1, 2]), np.cos(paths[:, -1, 2]))
    circular_resultant = abs(np.exp(1j * heading).mean())
    return {
        "median_path_length_m": float(np.median(speed.sum(axis=1) * 0.5)),
        "pairwise_rms_distance_median_m": float(np.median(distances[pair_indices])),
        "endpoint_lateral_standard_deviation_m": float(np.std(points[:, -1, 1])),
        "endpoint_forward_standard_deviation_m": float(np.std(points[:, -1, 0])),
        "endpoint_heading_circular_standard_deviation_deg": float(np.rad2deg(np.sqrt(-2 * np.log(max(circular_resultant, 1e-12))))),
        "terminal_speed_standard_deviation_m_per_s": float(np.std(speed[:, -1])),
        "median_endpoint_heading_abs_deg": float(np.median(np.abs(np.rad2deg(heading)))),
        "median_longitudinal_jerk_proxy_m_per_s3": float(np.median(np.sqrt(np.square(jerk).mean(axis=1)))),
    }, distances, pair_indices


def recording_bootstrap_indices(recording_groups, repetitions, seed):
    grouped = {group: np.flatnonzero(recording_groups == group) for group in sorted(set(recording_groups))}
    group_names = list(grouped)
    random = np.random.default_rng(seed)
    return [np.concatenate([grouped[group_names[index]] for index in random.integers(0, len(group_names), len(group_names))]) for _ in range(repetitions)]


def rank_correlation(first, second):
    if np.ptp(first) == 0 or np.ptp(second) == 0:
        return np.nan
    first_rank, second_rank = rankdata(first), rankdata(second)
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def quantile_interval(values):
    finite = np.asarray(values)[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [None, None]


def summarize_scenes(scene_rows, bootstrap_indices, selection_threshold):
    actual_best = np.asarray([row["oracle_pdms"] for row in scene_rows])
    chosen = np.asarray([row["selected_actual_pdms"] for row in scene_rows])
    regret = actual_best - chosen
    mean_absolute_errors = np.asarray([row["candidate_prediction_mae_pdms_points"] for row in scene_rows])
    return {
        "scene_count": len(scene_rows), "recording_count": len(set(row["recording_group"] for row in scene_rows)),
        "mean_oracle_pdms": float(actual_best.mean()), "mean_selected_pdms": float(chosen.mean()),
        "mean_selection_regret_pdms_points": float(regret.mean()),
        "mean_prediction_mae_pdms_points": float(mean_absolute_errors.mean()),
        "regret_95ci_recording_bootstrap": quantile_interval([regret[indices].mean() for indices in bootstrap_indices]),
        "mae_95ci_recording_bootstrap": quantile_interval([mean_absolute_errors[indices].mean() for indices in bootstrap_indices]),
        "scenes_with_regret_over10": int(np.sum(regret > selection_threshold)),
        "selected_nc_failure_scene_count": int(sum(row["selected_no_at_fault_collisions"] < 1 for row in scene_rows)),
        "selected_dac_failure_scene_count": int(sum(row["selected_drivable_area_compliance"] < 1 for row in scene_rows)),
        "selected_ttc_failure_scene_count": int(sum(row["selected_time_to_collision_within_bound"] < 1 for row in scene_rows)),
    }


def benjamini_hochberg(probabilities):
    values = np.asarray(probabilities)
    order = np.argsort(values)
    ordered_adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    ordered_adjusted = np.minimum.accumulate(ordered_adjusted[::-1])[::-1]
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, ordered_adjusted)
    return adjusted


def analyze(configuration):
    started = time.perf_counter()
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    output.mkdir(parents=True, exist_ok=True); destination.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT / configuration["source_candidate_directory"]
    score_source = PROJECT_ROOT / configuration["source_score_directory"]
    reference = json.loads((PROJECT_ROOT / configuration["protected_training_configuration"]).read_text())
    records = json.loads((source / "registration.json").read_text())["records"]
    arrays = np.load(source / "scorer_cache_and_candidates.npz")
    score_cache = np.load(score_source / "distribution_context_scores.npz")
    metrics = np.load(source / "official_metrics.npy")
    assert arrays["tokens"].tolist() == score_cache["tokens"].tolist() == [record["token"] for record in records]
    registration = {"registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "source_sha256": file_sha256(Path(__file__)), "records": records,
        "source_files": {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in [source / "registration.json", source / "scorer_cache_and_candidates.npz", source / "official_metrics.npy", score_source / "distribution_context_scores.npz"]},
        "protected_training_before": protected_files_audit(reference),
        "exploratory_already_exposed_development_panel": True}
    write_json(output / "registration.json", registration)
    write_json(destination / "subset_manifest.json", registration)
    write_json(destination / "configuration.json", configuration)
    observations, annotation_names = read_current_scene_metadata(records, configuration)
    all_rows, pair_rows, results = [], [], {}
    recording_groups = np.asarray([record["recording_group"] for record in records])
    bootstrap_indices = recording_bootstrap_indices(recording_groups, configuration["bootstrap_repetitions"], configuration["bootstrap_seed"])
    for candidate_source, (begin, end) in configuration["candidate_sources"].items():
        scene_rows = []
        for scene_index, (record, observed) in enumerate(zip(records, observations)):
            paths = arrays["trajectories"][scene_index, begin:end]
            actual = metrics[scene_index, begin:end, -1].astype(np.float64) * 100
            predicted = np.exp(score_cache["separate_log_scores"][scene_index, begin:end].astype(np.float64)) * 100 / 12
            probabilities = score_cache["separate_metric_probabilities"][scene_index, begin:end]
            chosen, oracle = int(predicted.argmax()), int(actual.argmax())
            geometry, distances, pair_indices = compute_set_geometry(paths, observed["ego_speed_m_per_s"])
            ordered = np.sort(predicted)
            row = {"token": record["token"], "log_name": record["log_name"], "recording_group": record["recording_group"],
                "command_name": record["command_name"], "candidate_source": candidate_source, **observed, **geometry,
                "oracle_pdms": float(actual[oracle]), "selected_actual_pdms": float(actual[chosen]),
                "selected_predicted_pdms": float(predicted[chosen]), "selection_regret_pdms_points": float(actual[oracle] - actual[chosen]),
                "candidate_prediction_mae_pdms_points": float(np.abs(predicted - actual).mean()),
                "candidate_signed_error_pdms_points": float((predicted - actual).mean()),
                "selected_candidate_index": chosen, "oracle_candidate_index": oracle,
                "predicted_top1_top2_margin_pdms_points": float(ordered[-1] - ordered[-2]),
                "official_unsafe_candidate_fraction": float(((metrics[scene_index, begin:end, 0] < 1) | (metrics[scene_index, begin:end, 1] < 1)).mean()),
                "official_actual90plus_candidate_fraction": float((actual >= 90).mean()),
                "selected_no_at_fault_collisions": float(metrics[scene_index, begin + chosen, 0]),
                "selected_drivable_area_compliance": float(metrics[scene_index, begin + chosen, 1]),
                "selected_ego_progress": float(metrics[scene_index, begin + chosen, 2]),
                "selected_time_to_collision_within_bound": float(metrics[scene_index, begin + chosen, 3]),
                "selected_comfort": float(metrics[scene_index, begin + chosen, 4]),
                "selected_nc_prediction": float(probabilities[chosen, 0]),
                "selected_dac_prediction": float(probabilities[chosen, 1]),
                "selected_ttc_prediction": float(probabilities[chosen, 3]),
            }
            assert all(np.isfinite(value) for value in row.values() if isinstance(value, (int, float)))
            scene_rows.append(row); all_rows.append(row)
            first_indices, second_indices = pair_indices
            actual_gap = actual[first_indices] - actual[second_indices]
            predicted_gap = predicted[first_indices] - predicted[second_indices]
            meaningful = np.abs(actual_gap) > configuration["meaningful_pairwise_pdms_gap"]
            close = distances[pair_indices] <= configuration["close_candidate_pair_threshold_m"]
            for group_name, pair_mask in [("within1m", close), ("over1m", ~close)]:
                eligible = pair_mask & meaningful
                pair_rows.append({"token": record["token"], "recording_group": record["recording_group"],
                    "candidate_source": candidate_source, "distance_group": group_name,
                    "pair_count": int(pair_mask.sum()), "meaningful_pair_count": int(eligible.sum()),
                    "reversed_ranking_pair_count": int(np.sum(eligible & (actual_gap * predicted_gap <= 0))),
                    "actual_gap_over50_pair_count": int(np.sum(pair_mask & (np.abs(actual_gap) > 50))),
                    "predicted_gap_below5_among_actual_gap_over50_count": int(np.sum(pair_mask & (np.abs(actual_gap) > 50) & (np.abs(predicted_gap) < 5)))})
        summary = summarize_scenes(scene_rows, bootstrap_indices, configuration["selection_regret_threshold_pdms_points"])
        outcomes = ["selection_regret_pdms_points", "candidate_prediction_mae_pdms_points"]
        associations = []
        for predictor in configuration["predictor_names"]:
            predictor_values = np.asarray([row[predictor] for row in scene_rows])
            for outcome in outcomes:
                outcome_values = np.asarray([row[outcome] for row in scene_rows])
                correlation, naive_probability = spearmanr(predictor_values, outcome_values)
                bootstrap_correlations = [rank_correlation(predictor_values[indices], outcome_values[indices]) for indices in bootstrap_indices]
                leave_recording_out = [rank_correlation(predictor_values[recording_groups != group], outcome_values[recording_groups != group]) for group in sorted(set(recording_groups))]
                associations.append({"predictor": predictor, "outcome": outcome, "spearman": float(correlation),
                    "recording_bootstrap95ci": quantile_interval(bootstrap_correlations),
                    "leave_one_recording_out_min_max": [float(np.nanmin(leave_recording_out)), float(np.nanmax(leave_recording_out))],
                    "naive_scene_independent_p_value_not_for_claims": float(naive_probability)})
        adjusted = benjamini_hochberg([row["naive_scene_independent_p_value_not_for_claims"] for row in associations])
        for row, value in zip(associations, adjusted):
            row["naive_fdr_q_value_not_for_claims"] = float(value)
        strata = []
        strata_definitions = [("command_name", sorted(set(row["command_name"] for row in scene_rows))),
            ("pairwise_rms_distance_median_m", configuration["pairwise_rms_distance_edges_m"]),
            ("median_endpoint_heading_abs_deg", configuration["heading_edges_deg"]),
            ("ego_speed_m_per_s", configuration["ego_speed_edges_m_per_s"]),
            ("vehicle_count_within30m", configuration["vehicle_count_edges"]),
            ("vulnerable_actor_present_within30m", [False, True])]
        for feature, definitions in strata_definitions:
            for value_index, value in enumerate(definitions if isinstance(definitions[0], (str, bool)) else definitions[:-1]):
                if feature == "command_name":
                    selected_rows = [row for row in scene_rows if row[feature] == value]; label = str(value)
                elif feature == "vulnerable_actor_present_within30m":
                    selected_rows = [row for row in scene_rows if bool(row["vulnerable_actor_count_within30m"] > 0) == value]; label = str(value)
                else:
                    upper = definitions[value_index + 1]
                    selected_rows = [row for row in scene_rows if value <= row[feature] < upper]; label = f"[{value}, {upper})"
                if not selected_rows:
                    continue
                subgroup_recordings = np.asarray([row["recording_group"] for row in selected_rows])
                subgroup_bootstrap = recording_bootstrap_indices(subgroup_recordings, configuration["bootstrap_repetitions"], configuration["bootstrap_seed"])
                strata.append({"feature": feature, "stratum": label, **summarize_scenes(selected_rows, subgroup_bootstrap, configuration["selection_regret_threshold_pdms_points"])})
        ordered_rows = sorted(scene_rows, key=lambda row: -row["selection_regret_pdms_points"])
        total_regret = sum(row["selection_regret_pdms_points"] for row in ordered_rows)
        pair_summary = []
        for distance_group in ("within1m", "over1m"):
            pairs = [row for row in pair_rows if row["candidate_source"] == candidate_source and row["distance_group"] == distance_group]
            totals = {key: sum(row[key] for row in pairs) for key in ["pair_count", "meaningful_pair_count", "reversed_ranking_pair_count", "actual_gap_over50_pair_count", "predicted_gap_below5_among_actual_gap_over50_count"]}
            totals["meaningful_pair_ranking_error_fraction"] = totals["reversed_ranking_pair_count"] / max(1, totals["meaningful_pair_count"])
            pair_summary.append({"distance_group": distance_group, **totals})
        results[candidate_source] = {"summary": summary, "associations": associations, "strata": strata,
            "worst_scenes": ordered_rows[:8], "pairwise_boundary_summary": pair_summary,
            "worst_scene_fraction_of_total_regret": ordered_rows[0]["selection_regret_pdms_points"] / total_regret,
            "worst3_fraction_of_total_regret": sum(row["selection_regret_pdms_points"] for row in ordered_rows[:3]) / total_regret}
        print(json.dumps({"candidate_source": candidate_source, "summary": summary}), flush=True)
    primary_rows = [row for row in all_rows if row["candidate_source"] == "drivor_generated64"]
    examples = []
    used_tokens = set()
    for failed in sorted(primary_rows, key=lambda row: -row["selection_regret_pdms_points"])[:3]:
        eligible = [row for row in primary_rows if row["command_name"] == failed["command_name"] and row["selection_regret_pdms_points"] < 1 and row["token"] not in used_tokens]
        matched = min(eligible, key=lambda row: abs(row["ego_speed_m_per_s"] - failed["ego_speed_m_per_s"]) / 2 + abs(row["pairwise_rms_distance_median_m"] - failed["pairwise_rms_distance_median_m"]) / 5)
        used_tokens.add(matched["token"])
        examples.append({"failed": failed, "comparison": matched, "selection": "Post-hoc largest regret; comparison shares observed command and matches speed/dispersion approximately, not a causal control."})
    report = {"registered_utc": registration["registered_utc"], "scene_count": len(records),
        "recording_count": len(set(recording_groups)), "full_navtest": False, "independent_final_test": False,
        "candidate_count_per_source": 64, "annotation_names_observed": annotation_names,
        "current_metadata_used_only_for_diagnosis": True, "candidate_attention_context": "complete standalone64 for each source",
        "results": results, "examples": examples,
        "cost": {"cpu_analysis_wall_seconds": time.perf_counter() - started, "gpu_seconds": 0,
            "optimizer_steps": 0, "new_backbone_calls": 0, "new_scorer_calls": 0, "new_official_evaluator_calls": 0},
        "validation": {"token_order_and_current_observation_verified": True, "all_rows_finite": True,
            "source_candidate_arrays_unchanged": True, "protected_training_after": protected_files_audit(reference)}}
    write_json(destination / "results.json", report)
    for filename, rows in [("scene_characteristics.csv", all_rows), ("candidate_pair_diagnostics.csv", pair_rows)]:
        with (destination / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (destination / "feature_associations.csv").open("w", newline="") as stream:
        association_rows = [{"candidate_source": source_name, **row} for source_name, source_report in results.items() for row in source_report["associations"]]
        writer = csv.DictWriter(stream, fieldnames=list(association_rows[0])); writer.writeheader(); writer.writerows(association_rows)
    render_statistical_figures(primary_rows, report, destination)
    write_json(output / "analysis_complete.json", {"complete": True, "results": configuration["results_directory"], "gpu_seconds": 0})


def render_statistical_figures(scene_rows, report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"left_turn": "#b03b2c", "straight": "#167a96", "right_turn": "#9b6c00"}
    figure, axes = plt.subplots(2, 3, figsize=(17, 10))
    pairs = [("pairwise_rms_distance_median_m", "selection_regret_pdms_points"),
        ("median_endpoint_heading_abs_deg", "selection_regret_pdms_points"),
        ("vehicle_count_within30m", "selection_regret_pdms_points"),
        ("pairwise_rms_distance_median_m", "candidate_prediction_mae_pdms_points"),
        ("ego_speed_m_per_s", "candidate_prediction_mae_pdms_points"),
        ("nearest_forward_actor_center_distance_capped_m", "candidate_prediction_mae_pdms_points")]
    for axis, (predictor, outcome) in zip(axes.flat, pairs):
        for command, color in colors.items():
            rows = [row for row in scene_rows if row["command_name"] == command]
            axis.scatter([row[predictor] for row in rows], [row[outcome] for row in rows], color=color, alpha=.75, s=40)
        axis.set_xlabel(FEATURE_LABELS[predictor]); axis.set_ylabel("Selection regret (PDMS pts)" if outcome == "selection_regret_pdms_points" else "Candidate prediction MAE (PDMS pts)")
        axis.grid(alpha=.15)
    figure.suptitle("Original DrivoR generator + scorer: candidate-set / scene characteristics\nEach dot = one scene's complete 64-candidate set; 64 scenes / 38 recordings", fontsize=14)
    figure.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=color, label=command) for command, color in colors.items()], loc="lower center", ncol=3)
    figure.tight_layout(rect=(0, .055, 1, .92)); figure.savefig(destination / "scene_characteristics_vs_scorer_error.png", dpi=145); plt.close(figure)
    associations = report["results"]["drivor_generated64"]["associations"]
    figure, axes = plt.subplots(1, 2, figsize=(15, 10), sharey=True)
    for axis, outcome in zip(axes, ["selection_regret_pdms_points", "candidate_prediction_mae_pdms_points"]):
        rows = [row for row in associations if row["outcome"] == outcome]
        for position, row in enumerate(rows):
            lower, upper = row["recording_bootstrap95ci"]
            axis.plot([lower, upper], [position, position], color="#21618c", linewidth=2)
            axis.scatter(row["spearman"], position, color="#21618c", s=35)
        axis.axvline(0, color="#555555", linestyle="--", linewidth=1)
        axis.set_xticks([-.6, -.3, 0, .3, .6]); axis.set_xlim(-.8, .8)
        axis.set_title("Selection regret" if outcome == "selection_regret_pdms_points" else "Candidate prediction MAE")
        axis.set_xlabel("Spearman association + recording bootstrap 95% interval")
    axes[0].set_yticks(range(len(rows)), [FEATURE_LABELS[row["predictor"]] for row in rows]); axes[0].invert_yaxis()
    figure.suptitle("Exploratory associations, not causes or a validated failure detector", fontsize=14)
    figure.text(.5, .015, "15 predictors x 2 outcomes on an already exposed development panel. Multiple comparisons and command/speed confounding remain.", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .04, 1, .94)); figure.savefig(destination / "feature_associations_recording_bootstrap.png", dpi=145); plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", type=Path, required=True)
    arguments = parser.parse_args()
    analyze(json.loads(arguments.configuration.read_text()))
