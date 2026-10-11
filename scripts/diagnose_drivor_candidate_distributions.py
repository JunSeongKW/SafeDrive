#!/usr/bin/env python
"""CPU-only candidate-distribution diagnosis of the original public DrivoR scorer."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import spearmanr

from calibrate_official_toad_navsim_v1_scorer import PROJECT_ROOT, file_sha256, protected_files_audit, write_json
from calibrate_frozen_drivor_scorer import METRIC_NAMES, recording_bootstrap
from compare_public_drivor_toad_scorers import model_configuration_for, load_official_model, score_metrics


DISTRIBUTION_LABELS = {
    "drivor_generated64": "DrivoR generated",
    "toad_generated64": "TOAD generated",
    "random_bouquet64": "Random bouquet",
    "nearby64": "Nearby proposals",
    "constant_controls22": "Constant controls",
    "smooth_controls21": "Smooth controls",
    "freeform_controls21": "Freeform controls",
}


def load_registered_inputs(configuration):
    source = PROJECT_ROOT / configuration["source_output_directory"]
    registration = json.loads((source / "registration.json").read_text())
    arrays = np.load(source / "scorer_cache_and_candidates.npz")
    official_metrics = np.load(source / "official_metrics.npy")
    assert arrays["tokens"].tolist() == [record["token"] for record in registration["records"]]
    assert arrays["trajectories"].shape == (64, 256, 8, 3)
    assert arrays["drivor_scene_features"].shape == (64, 64, 256)
    assert official_metrics.shape == (64, 256, 7)
    assert np.isfinite(official_metrics).all()
    return registration["records"], arrays, official_metrics


def prepare_and_rescore_cached_candidates(configuration):
    import torch
    from diagnose_residual_control_sampler import module_digest

    assert configuration["device"] == "cpu"
    started = time.perf_counter()
    output = PROJECT_ROOT / configuration["output_directory"]
    destination = PROJECT_ROOT / configuration["results_directory"]
    output.mkdir(parents=True, exist_ok=True)
    comparison = json.loads((PROJECT_ROOT / configuration["comparison_configuration"]).read_text())
    reference = json.loads((PROJECT_ROOT / comparison["reference_configuration"]).read_text())
    records, arrays, official_metrics = load_registered_inputs(configuration)
    registration = {
        "registered_utc": datetime.now(timezone.utc).isoformat(), "configuration": configuration,
        "records": records, "description": "Retrospective development diagnostic; thresholds fixed before CPU rescoring",
        "analysis_source_sha256": file_sha256(Path(__file__)),
        "source_cache_sha256": file_sha256(PROJECT_ROOT / configuration["source_output_directory"] / "scorer_cache_and_candidates.npz"),
        "source_official_metrics_sha256": file_sha256(PROJECT_ROOT / configuration["source_output_directory"] / "official_metrics.npy"),
        "checkpoint_sha256": file_sha256(PROJECT_ROOT / comparison["drivor_checkpoint"]),
        "protected_training": protected_files_audit(reference),
    }
    assert registration["checkpoint_sha256"] == comparison["drivor_checkpoint_sha256"]
    write_json(output / "registration.json", registration)
    write_json(destination / "subset_manifest.json", {key: value for key, value in registration.items() if key != "configuration"})
    write_json(destination / "configuration.json", configuration)
    model_configuration = model_configuration_for(comparison, "drivor_public")
    model_configuration["cpu_threads"] = configuration["cpu_threads"]
    model, loading = load_official_model(model_configuration, "cpu")
    model_digest_before = module_digest(model)
    backbone_calls = 0

    def reject_backbone_forward(_module, _arguments):
        nonlocal backbone_calls
        backbone_calls += 1
        raise AssertionError("Cached scorer diagnosis must never execute image backbone")

    guard = model.image_backbone.register_forward_pre_hook(reject_backbone_forward)
    maximum_replay_error = 0.0
    replay_started = time.perf_counter()
    with torch.inference_mode():
        for scene_index in range(3):
            visual = torch.from_numpy(arrays["drivor_scene_features"][scene_index:scene_index+1])
            ego = torch.from_numpy(arrays["drivor_ego_tokens"][scene_index:scene_index+1])
            for begin, end in ((0, 128), (128, 192), (192, 256)):
                paths = torch.from_numpy(arrays["trajectories"][scene_index:scene_index+1, begin:end])
                scores, _, _ = score_metrics(model, paths, visual, ego)
                error = float(np.abs(scores.numpy()-arrays["neural_log_scores"][0, scene_index:scene_index+1, begin:end]).max())
                maximum_replay_error = max(maximum_replay_error, error)
    assert maximum_replay_error <= configuration["cache_replay_tolerance_log_score"], maximum_replay_error
    replay_seconds = time.perf_counter()-replay_started
    rescoring_started = time.perf_counter()
    separate_scores, separate_probabilities, shared_scores, shared_probabilities = [], [], [], []
    batch_sizes, batch_seconds = [], []
    for scene_begin in range(0, len(records), configuration["cpu_microbatch_scenes"]):
        scene_end = min(scene_begin+configuration["cpu_microbatch_scenes"], len(records))
        batch_started = time.perf_counter()
        visual = torch.from_numpy(arrays["drivor_scene_features"][scene_begin:scene_end])
        ego = torch.from_numpy(arrays["drivor_ego_tokens"][scene_begin:scene_end])
        paths = torch.from_numpy(arrays["trajectories"][scene_begin:scene_end])
        with torch.inference_mode():
            own_score, own_probability, _ = score_metrics(model, paths[:, :64], visual, ego)
            other_score, other_probability, _ = score_metrics(model, paths[:, 64:128], visual, ego)
            common_score, common_probability, _ = score_metrics(model, paths, visual, ego)
        # Random64 and nearby64 were already evaluated in their own complete 64-candidate context.
        separate_scores.append(np.concatenate((own_score.numpy(), other_score.numpy(), arrays["neural_log_scores"][0, scene_begin:scene_end, 128:]), axis=1))
        separate_probabilities.append(np.concatenate((own_probability.numpy(), other_probability.numpy(), arrays["metric_probabilities"][0, scene_begin:scene_end, 128:]), axis=1))
        shared_scores.append(common_score.numpy())
        shared_probabilities.append(common_probability.numpy())
        batch_sizes.append(scene_end-scene_begin)
        batch_seconds.append(time.perf_counter()-batch_started)
        write_json(output / "progress.json", {"completed_scenes": scene_end, "total_scenes": len(records), "device": "cpu", "gpu_seconds": 0})
        if scene_end % 16 == 0:
            print(json.dumps({"rescored_scenes": scene_end, "elapsed_seconds": time.perf_counter()-started}), flush=True)
    guard.remove()
    assert backbone_calls == 0
    assert module_digest(model) == model_digest_before
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters())
    np.savez_compressed(output / "distribution_context_scores.npz", tokens=arrays["tokens"],
        separate_log_scores=np.concatenate(separate_scores), separate_metric_probabilities=np.concatenate(separate_probabilities),
        shared_log_scores=np.concatenate(shared_scores), shared_metric_probabilities=np.concatenate(shared_probabilities))
    report = {"complete": True, "loading": loading, "maximum_cached_gpu_vs_cpu_replay_error": maximum_replay_error,
        "cached_backbone_calls": backbone_calls, "all_weights_frozen_and_unchanged": True, "all_parameter_gradients_absent": True,
        "optimizer_steps": 0, "device": "cpu", "gpu_seconds": 0, "new_official_evaluator_calls": 0,
        "reuse_official_candidate_count": int(official_metrics.shape[0]*official_metrics.shape[1]),
        "cpu_phase_wall_seconds": time.perf_counter()-started, "cached_replay_seconds": replay_seconds,
        "cpu_rescoring_seconds": time.perf_counter()-rescoring_started,
        "mean_seconds_per_scene_for_three_scorer_calls": float(np.sum(batch_seconds)/np.sum(batch_sizes)),
        "new_scorer_calls": 3*len(batch_sizes), "new_scorer_scene_forwards": 3*len(records),
        "cpu_threads": configuration["cpu_threads"], "microbatch_scenes": configuration["cpu_microbatch_scenes"],
        "protected_training": protected_files_audit(reference), "environment": {"torch": torch.__version__, "numpy": np.__version__}}
    write_json(output / "rescoring_complete.json", report)
    write_json(destination / "validation.json", report)
    print(json.dumps({"complete": True, "cpu_seconds": report["cpu_phase_wall_seconds"], "gpu_seconds": 0}), flush=True)


def compute_candidate_geometry(trajectories, records):
    """Coordinate/control proxies derived from planned waypoints, not simulated states."""
    points = trajectories[..., :2].astype(np.float64)
    preceding_points = np.concatenate((np.zeros_like(points[:, :, :1]), points[:, :, :-1]), axis=2)
    segment_speed = np.linalg.norm(points-preceding_points, axis=-1)/0.5
    initial_speed = np.asarray([record["ego_history"][-1][3] for record in records], dtype=np.float64)
    preceding_speed = np.concatenate((np.broadcast_to(initial_speed[:, None, None], segment_speed[:, :, :1].shape), segment_speed[:, :, :-1]), axis=2)
    acceleration_proxy = (segment_speed-preceding_speed)/0.5
    jerk_proxy = np.diff(acceleration_proxy, axis=2)/0.5
    distances = []
    for scene_points in points:
        squared = np.square(scene_points[:, None]-scene_points[None, :64]).sum(axis=-1).mean(axis=-1)
        distances.append(np.sqrt(squared.min(axis=1)))
    return {
        "nearest_drivor_proposal_rms_distance_m": np.stack(distances),
        "endpoint_heading_abs_deg": np.abs(np.rad2deg(np.arctan2(np.sin(trajectories[..., -1, 2]), np.cos(trajectories[..., -1, 2])))),
        "endpoint_lateral_displacement_abs_m": np.abs(points[..., -1, 1]),
        "terminal_speed_proxy_m_per_s": segment_speed[..., -1],
        "rms_longitudinal_jerk_proxy_m_per_s3": np.sqrt(np.square(jerk_proxy).mean(axis=-1)),
    }


def summarize_score_subset(log_scores, probabilities, official_metrics, recording_groups, configuration, allow_selection=True):
    estimate = np.exp(log_scores.astype(np.float64))*100/12
    actual = official_metrics[..., -1]*100
    error = estimate-actual
    unsafe = (official_metrics[..., 0] < 1) | (official_metrics[..., 1] < 1)
    predicted_safe = (probabilities[..., 0] >= configuration["prediction_safety_threshold"]) & (probabilities[..., 1] >= configuration["prediction_safety_threshold"])
    row = {
        "candidate_count_total": int(actual.size), "candidates_per_scene": int(actual.shape[1]), "scene_count": len(recording_groups),
        "prediction_mae_pdms_points": float(np.abs(error).mean()), "prediction_rmse_pdms_points": float(np.sqrt(np.square(error).mean())),
        "mean_signed_error_pdms_points": float(error.mean()), "predicted_mean_pdms": float(estimate.mean()), "actual_mean_pdms": float(actual.mean()),
        "prediction_within5_points_fraction": float((np.abs(error) <= 5).mean()),
        "unsafe_candidate_count": int(unsafe.sum()), "unsafe_candidate_fraction": float(unsafe.mean()),
        "false_safe_candidate_count": int((unsafe & predicted_safe).sum()),
        "false_safe_fraction_among_unsafe": float((unsafe & predicted_safe).sum()/unsafe.sum()) if unsafe.any() else None,
        "submetric_mae": {name: float(np.abs(probabilities[..., index]-official_metrics[..., index]).mean()) for index, name in enumerate(METRIC_NAMES[:-1])},
    }
    scene_mae = np.abs(error).mean(axis=1)
    row["mae_recording_bootstrap_95ci"] = recording_bootstrap(scene_mae, recording_groups, configuration["bootstrap_seed"], configuration["bootstrap_repetitions"])
    valid_correlations, discordant_count, eligible_count = [], 0, 0
    for scene_index in range(actual.shape[0]):
        if np.ptp(actual[scene_index]) > 1e-10 and np.ptp(log_scores[scene_index]) > 1e-12:
            valid_correlations.append(float(spearmanr(actual[scene_index], log_scores[scene_index]).statistic))
        actual_difference = actual[scene_index, :, None]-actual[scene_index, None, :]
        score_difference = log_scores[scene_index, :, None]-log_scores[scene_index, None, :]
        meaningful = np.triu(np.abs(actual_difference) > configuration["meaningful_pairwise_pdms_gap"], k=1)
        eligible_count += int(meaningful.sum())
        discordant_count += int((meaningful & (actual_difference*score_difference <= 0)).sum())
    row.update(within_scene_spearman_mean=float(np.mean(valid_correlations)) if valid_correlations else None,
        spearman_valid_scene_count=len(valid_correlations), meaningful_pair_count=eligible_count,
        discordant_pair_fraction=float(discordant_count/eligible_count) if eligible_count else None)
    if allow_selection:
        scene_indices = np.arange(actual.shape[0])
        selected_indices, oracle_indices = log_scores.argmax(axis=1), actual.argmax(axis=1)
        selected, oracle = actual[scene_indices, selected_indices], actual[scene_indices, oracle_indices]
        regret = oracle-selected
        selected_metrics = official_metrics[scene_indices, selected_indices]
        row.update(selected_pdms=float(selected.mean()), oracle_best_pdms=float(oracle.mean()), selection_regret_pdms_points=float(regret.mean()),
            regret_recording_bootstrap_95ci=recording_bootstrap(regret, recording_groups, configuration["bootstrap_seed"], configuration["bootstrap_repetitions"]),
            selected_nc_failure_count=int((selected_metrics[:, 0] < 1).sum()), selected_dac_failure_count=int((selected_metrics[:, 1] < 1).sum()),
            selected_ttc_failure_count=int((selected_metrics[:, 3] < 1).sum()))
    return row


def summarize_masked_errors(mask, estimate, actual, recording_groups):
    count = int(mask.sum())
    per_scene = np.divide((np.abs(estimate-actual)*mask).sum(axis=1), mask.sum(axis=1),
        out=np.full(mask.shape[0], np.nan), where=mask.sum(axis=1)>0)
    present = np.isfinite(per_scene)
    return {"candidate_count": count, "scene_count": int(present.sum()),
        "recording_count": len({group for group, valid in zip(recording_groups, present) if valid}),
        "candidate_weighted_mae": float(np.abs(estimate-actual)[mask].mean()) if count else None,
        "scene_macro_mae": float(per_scene[present].mean()) if present.any() else None,
        "signed_error": float((estimate-actual)[mask].mean()) if count else None,
        "actual_mean_pdms": float(actual[mask].mean()) if count else None,
        "prediction_within5_points_fraction": float((np.abs(estimate-actual)[mask]<=5).mean()) if count else None}


def analyze_distributions_and_plot(configuration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    started = time.perf_counter()
    output, destination = [PROJECT_ROOT / configuration[key] for key in ("output_directory", "results_directory")]
    records, source, official_metrics = load_registered_inputs(configuration)
    rescored = np.load(output / "distribution_context_scores.npz")
    assert source["tokens"].tolist() == rescored["tokens"].tolist()
    validation = json.loads((output / "rescoring_complete.json").read_text())
    grouping = [record["recording_group"] for record in records]
    geometry = compute_candidate_geometry(source["trajectories"], records)
    actual = official_metrics[..., -1]*100
    separate_scores = rescored["separate_log_scores"]
    shared_scores = rescored["shared_log_scores"]
    shared_estimate = np.exp(shared_scores.astype(np.float64))*100/12
    operational, shared, random_families = {}, {}, {}
    for name, (begin, end) in configuration["source_distributions"].items():
        operational[name] = summarize_score_subset(separate_scores[:, begin:end], rescored["separate_metric_probabilities"][:, begin:end], official_metrics[:, begin:end], grouping, configuration)
        shared[name] = summarize_score_subset(shared_scores[:, begin:end], rescored["shared_metric_probabilities"][:, begin:end], official_metrics[:, begin:end], grouping, configuration, allow_selection=False)
    for name, (begin, end) in configuration["random_family_offsets"].items():
        random_families[name] = summarize_score_subset(shared_scores[:, begin:end], rescored["shared_metric_probabilities"][:, begin:end], official_metrics[:, begin:end], grouping, configuration, allow_selection=False)
    conditional, score_bands, risk_slices = {}, {}, {}
    for source_name, (begin, end) in configuration["source_distributions"].items():
        rows = []
        for lower, upper in zip(configuration["actual_pdms_edges"][:-1], configuration["actual_pdms_edges"][1:]):
            mask = (actual[:, begin:end] >= lower) & (actual[:, begin:end] < upper)
            rows.append({"actual_pdms_lower": lower, "actual_pdms_upper": upper, **summarize_masked_errors(mask, shared_estimate[:, begin:end], actual[:, begin:end], grouping)})
        score_bands[source_name] = rows
        safe = (official_metrics[:, begin:end, 0] == 1) & (official_metrics[:, begin:end, 1] == 1)
        risk_slices[source_name] = {"nc_dac_compliant": summarize_masked_errors(safe, shared_estimate[:, begin:end], actual[:, begin:end], grouping),
            "nc_or_dac_noncompliant": summarize_masked_errors(~safe, shared_estimate[:, begin:end], actual[:, begin:end], grouping)}
    bin_definitions = {
        "nearest_drivor_proposal_rms_distance_m": configuration["nearest_native_distance_edges_m"],
        "endpoint_heading_abs_deg": configuration["endpoint_heading_edges_deg"],
        "rms_longitudinal_jerk_proxy_m_per_s3": configuration["control_proxy_jerk_edges_m_per_s3"],
    }
    for feature_name, edges in bin_definitions.items():
        feature = geometry[feature_name]
        rows = []
        for lower, upper in zip(edges[:-1], edges[1:]):
            mask = (feature >= lower) & (feature < upper)
            rows.append({"lower": lower, "upper": upper, **summarize_masked_errors(mask, shared_estimate, actual, grouping)})
        conditional[feature_name] = rows
    command_rows = {}
    for command in ("left_turn", "straight", "right_turn"):
        selected_scenes = np.asarray([record["command_name"]==command for record in records])
        command_rows[command] = {}
        for source_name, (begin, end) in configuration["source_distributions"].items():
            scores = separate_scores[selected_scenes, begin:end]
            values = actual[selected_scenes, begin:end]
            selection = values[np.arange(values.shape[0]), scores.argmax(axis=1)]
            command_rows[command][source_name] = {"scenes": int(selected_scenes.sum()), "selected_pdms": float(selection.mean()),
                "oracle_best_pdms": float(values.max(axis=1).mean()), "selection_regret_pdms_points": float((values.max(axis=1)-selection).mean())}
    context_effects = {}
    for source_name, (begin, end) in configuration["source_distributions"].items():
        separate_choice, shared_choice = separate_scores[:, begin:end].argmax(axis=1), shared_scores[:, begin:end].argmax(axis=1)
        indices = np.arange(len(records))
        separate_selection = actual[indices, begin+separate_choice]
        shared_selection = actual[indices, begin+shared_choice]
        context_effects[source_name] = {"selected_candidate_changed_scenes": int((separate_choice!=shared_choice).sum()),
            "mean_pdms_change_shared_minus_separate": float((shared_selection-separate_selection).mean()),
            "maximum_prediction_change_pdms_points": float(np.abs(shared_estimate[:, begin:end]-np.exp(separate_scores[:, begin:end].astype(np.float64))*100/12).max())}
    summary = {"complete": True, "benchmark": "NAVSIM v1 NAVTEST64 previously used development diagnostic", "scene_count": len(records),
        "recording_count": len(set(grouping)), "full_navtest": False, "independent_final_test": False,
        "model": "Original public DrivoR Scaling checkpoint (epoch index28, global53911)",
        "operational_separate64_context": operational, "fixed_shared256_context": shared,
        "random_generator_families_shared256": random_families, "actual_score_band_errors_shared256": score_bands,
        "safety_conditioned_errors_shared256": risk_slices, "geometric_error_slices_shared256": conditional,
        "by_current_driving_command_separate64": command_rows, "candidate_context_effect": context_effects,
        "score_conversion": "Predicted PDMS estimate = 100*exp(neural_log_score)/12; soft metric composition, not certified probability",
        "cost_and_checks": validation, "notes": configuration["notes"]}
    write_json(destination / "results.json", summary)
    write_json(destination / "random_families.json", random_families)
    with (destination / "distribution_comparison.csv").open("w") as stream:
        names = ["distribution", "candidate_count_total", "prediction_mae_pdms_points", "within_scene_spearman_mean", "discordant_pair_fraction", "oracle_best_pdms", "selected_pdms", "selection_regret_pdms_points", "unsafe_candidate_fraction", "selected_nc_failure_count", "selected_dac_failure_count"]
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for name, row in operational.items():
            writer.writerow({"distribution": name, **{key:row[key] for key in names if key in row}})
    with (destination / "candidate_diagnostics.csv").open("w") as stream:
        names = ["token", "recording_group", "current_command", "source_distribution", "candidate_index", "actual_pdms", "predicted_pdms_separate64", "predicted_pdms_shared256"]+list(geometry)
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        for source_name, (begin, end) in configuration["source_distributions"].items():
            for scene_index, record in enumerate(records):
                for candidate_index in range(begin,end):
                    writer.writerow({"token":record["token"], "recording_group":record["recording_group"], "current_command":record["command_name"],
                        "source_distribution":source_name, "candidate_index":candidate_index, "actual_pdms":actual[scene_index,candidate_index],
                        "predicted_pdms_separate64":float(np.exp(float(separate_scores[scene_index,candidate_index]))*100/12),
                        "predicted_pdms_shared256":shared_estimate[scene_index,candidate_index],
                        **{key:values[scene_index,candidate_index] for key,values in geometry.items()}})
    plot_distribution_diagnostics(configuration, summary, source, rescored, official_metrics, geometry, records, plt)
    write_distribution_report(configuration, summary, destination)
    write_json(output / "analysis_complete.json", {"complete": True, "cpu_analysis_seconds": time.perf_counter()-started, "gpu_seconds":0})
    print(json.dumps({"complete":True, "operational_distributions":operational, "families":random_families}),flush=True)


def plot_distribution_diagnostics(configuration, summary, source, rescored, official_metrics, geometry, records, plt):
    destination = PROJECT_ROOT/configuration["results_directory"]
    distributions = list(configuration["source_distributions"])
    labels = [DISTRIBUTION_LABELS[name] for name in distributions]
    colors = ["#2877b8", "#319466", "#dc812b", "#8b63ae"]
    actual = official_metrics[..., -1]*100
    estimate = np.exp(rescored["separate_log_scores"].astype(np.float64))*100/12
    shared_estimate = np.exp(rescored["shared_log_scores"].astype(np.float64))*100/12
    operational = summary["operational_separate64_context"]
    positions = np.arange(4)
    figure, axes = plt.subplots(2,3,figsize=(17,10))
    axes[0,0].bar(positions-.18,[operational[name]["oracle_best_pdms"] for name in distributions],width=.36,color="#42a25c",label="Oracle best candidate")
    axes[0,0].bar(positions+.18,[operational[name]["selected_pdms"] for name in distributions],width=.36,color="#2877b8",label="DrivoR-selected path")
    axes[0,0].set_ylim(0,110);axes[0,0].set_ylabel("Actual PDMS");axes[0,0].legend(fontsize=8,loc="lower right",bbox_to_anchor=(1,1.13),ncol=2)
    axes[0,0].set_title("Same budget: each distribution scored as 64 candidates")
    for index,name in enumerate(distributions):
        axes[0,0].text(index,operational[name]["oracle_best_pdms"]+2,f"gap {operational[name]['selection_regret_pdms_points']:.1f}",ha="center",fontsize=8)
    means=[operational[name]["prediction_mae_pdms_points"] for name in distributions]
    intervals=np.asarray([operational[name]["mae_recording_bootstrap_95ci"] for name in distributions])
    axes[0,1].bar(positions,means,color=colors,yerr=np.maximum(np.stack((np.asarray(means)-intervals[:,0],intervals[:,1]-means)),0),capsize=4)
    axes[0,1].set_ylabel("Prediction MAE (PDMS points)");axes[0,1].set_title("Absolute score accuracy; recording bootstrap 95% CI")
    axes[0,2].bar(positions,[operational[name]["within_scene_spearman_mean"] for name in distributions],color=colors)
    axes[0,2].set_ylim(0,1);axes[0,2].set_ylabel("Mean within-scene Spearman");axes[0,2].set_title("Ranking accuracy; higher is better")
    axes[1,0].bar(positions-.18,[operational[name]["selected_nc_failure_count"] for name in distributions],width=.36,color="#d55e00",label="NC < 1")
    axes[1,0].bar(positions+.18,[operational[name]["selected_dac_failure_count"] for name in distributions],width=.36,color="#0072b2",label="DAC < 1")
    axes[1,0].set_ylabel("Selected failures / 64 scenes");axes[1,0].set_title("Selected path safety");axes[1,0].legend(fontsize=8)
    axes[1,1].bar(positions,[summary["fixed_shared256_context"][name]["prediction_mae_pdms_points"] for name in distributions],color=colors)
    axes[1,1].set_ylabel("Prediction MAE (PDMS points)");axes[1,1].set_title("Context control: all candidates scored together")
    axes[1,2].violinplot([actual[:,begin:end].ravel() for begin,end in configuration["source_distributions"].values()],positions=positions,showmeans=True)
    axes[1,2].set_ylim(-5,105);axes[1,2].set_ylabel("Actual PDMS of ALL candidates");axes[1,2].set_title("Distributions contain different score / risk mixes")
    for axis in axes.flat:
        axis.set_xticks(positions,labels,rotation=15,ha="right");axis.grid(axis="y",alpha=.15)
    figure.suptitle("Original public DrivoR scorer | NAVTEST64 development diagnostic\nScore calibration and top-candidate selection are different properties",fontsize=14)
    figure.tight_layout(rect=(0,0,1,.94));figure.savefig(destination/"scorer_performance_by_distribution.png",dpi=160);plt.close(figure)
    figure,axes=plt.subplots(1,4,figsize=(17,4.5),sharex=True,sharey=True)
    for index,(name,(begin,end)) in enumerate(configuration["source_distributions"].items()):
        axes[index].hexbin(estimate[:,begin:end].ravel(),actual[:,begin:end].ravel(),gridsize=35,extent=(0,100,0,100),bins="log",mincnt=1,cmap="Blues")
        axes[index].plot([0,100],[0,100],"k--",linewidth=1)
        axes[index].set_title(f"{DISTRIBUTION_LABELS[name]}\nMAE {operational[name]['prediction_mae_pdms_points']:.2f}; regret {operational[name]['selection_regret_pdms_points']:.2f}")
        axes[index].set_xlabel("Predicted PDMS estimate");axes[index].set_xlim(-2,102);axes[index].set_ylim(-2,102)
    axes[0].set_ylabel("Actual official PDMS")
    figure.suptitle("Separate 64-candidate contexts | Lower right = optimistic prediction of a bad path\nHexbin darkness indicates candidate count; diagonal means accurate prediction",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.86));figure.savefig(destination/"predicted_vs_actual_by_distribution.png",dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,5,figsize=(20,9.5))
    from PIL import Image
    fixed_examples=[next(index for index,record in enumerate(records) if record["command_name"]==command) for command in ("straight","left_turn")]
    image_manifest=[]
    for row_index,scene_index in enumerate(fixed_examples):
        record=records[scene_index]
        axes[row_index,0].imshow(Image.open(record["current_camera_paths"][0]));axes[row_index,0].axis("off")
        axes[row_index,0].set_title(f"Current front image | {record['command_name']}\n{record['token']}",fontsize=10)
        paths=source["trajectories"][scene_index]
        maximum_lateral=max(10,float(np.abs(paths[...,1]).max())*1.05)
        forward_min=min(-5,float(paths[...,0].min())*1.05)
        forward_max=max(15,float(paths[...,0].max())*1.05)
        for column_index,(name,(begin,end)) in enumerate(configuration["source_distributions"].items(),start=1):
            axis=axes[row_index,column_index]
            for candidate_index in range(begin,end):
                axis.plot(np.r_[0,paths[candidate_index,:,1]],np.r_[0,paths[candidate_index,:,0]],color=plt.cm.viridis(actual[scene_index,candidate_index]/100),alpha=.40,linewidth=.9)
            axis.plot(0,0,"r^",markersize=7);axis.set_xlim(maximum_lateral,-maximum_lateral);axis.set_ylim(forward_min,forward_max)
            axis.set_aspect("equal",adjustable="box");axis.set_xlabel("Lateral (m); left positive",fontsize=9);axis.set_ylabel("Forward distance (m)",fontsize=9)
            axis.set_title(DISTRIBUTION_LABELS[name]+f"\nmean actual {actual[scene_index,begin:end].mean():.1f}",fontsize=10)
        image_manifest.append({"token":record["token"],"selection_rule":"First stored scene of observed command; not chosen by scorer error","current_command":record["command_name"]})
    figure.suptitle("Same scene, different candidate distributions | 64 trajectories per panel\nPurple = low actual PDMS; yellow = high actual PDMS. Coordinates are ego-frame planned paths, not map labels.",fontsize=13)
    figure.tight_layout(rect=(0,0,1,.92),h_pad=3.5,w_pad=2.0);figure.savefig(destination/"trajectory_candidate_distributions.png",dpi=150);plt.close(figure)
    write_json(destination/"trajectory_visualization_manifest.json",image_manifest)
    figure,axes=plt.subplots(1,3,figsize=(16,4.5))
    distance_edges=configuration["nearest_native_distance_edges_m"]
    heading_edges=configuration["endpoint_heading_edges_deg"]
    absolute_error=np.abs(shared_estimate-actual)
    heatmap=np.full((len(heading_edges)-1,len(distance_edges)-1),np.nan)
    counts=np.zeros_like(heatmap,dtype=int)
    for heading_index,(heading_low,heading_high) in enumerate(zip(heading_edges[:-1],heading_edges[1:])):
        for distance_index,(distance_low,distance_high) in enumerate(zip(distance_edges[:-1],distance_edges[1:])):
            mask=(geometry["endpoint_heading_abs_deg"]>=heading_low)&(geometry["endpoint_heading_abs_deg"]<heading_high)&(geometry["nearest_drivor_proposal_rms_distance_m"]>=distance_low)&(geometry["nearest_drivor_proposal_rms_distance_m"]<distance_high)
            counts[heading_index,distance_index]=int(mask.sum())
            if mask.any():heatmap[heading_index,distance_index]=absolute_error[mask].mean()
    plotted=axes[0].imshow(heatmap,aspect="auto",cmap="YlOrRd",vmin=0,vmax=max(30,float(np.nanmax(heatmap))))
    axes[0].set_xticks(range(5),["0-.5",".5-2","2-5","5-10",">10"]);axes[0].set_yticks(range(4),["0-10","10-30","30-60","60-180"])
    axes[0].set_xlabel("Nearest own proposal RMS distance (m)");axes[0].set_ylabel("Absolute endpoint heading (deg)")
    axes[0].set_title("Error by trajectory geometry | fixed context")
    for heading_index in range(4):
        for distance_index in range(5):
            if counts[heading_index,distance_index]:axes[0].text(distance_index,heading_index,f"{heatmap[heading_index,distance_index]:.1f}\nn={counts[heading_index,distance_index]}",ha="center",va="center",fontsize=8)
    figure.colorbar(plotted,ax=axes[0],label="PDMS prediction MAE")
    score_labels=["Actual 0","0-50","50-90","90-100"]
    for index,name in enumerate(distributions):
        values=[row["candidate_weighted_mae"] for row in summary["actual_score_band_errors_shared256"][name]]
        axes[1].plot(range(4),values,marker="o",label=DISTRIBUTION_LABELS[name],color=colors[index])
    axes[1].set_xticks(range(4),score_labels);axes[1].set_ylabel("PDMS prediction MAE");axes[1].set_title("Within actual-score bands | retrospective labels");axes[1].legend(fontsize=8)
    families=summary["random_generator_families_shared256"]
    axes[2].bar(range(3),[row["prediction_mae_pdms_points"] for row in families.values()],color=["#ebc146","#d88238","#bf5536"])
    axes[2].set_xticks(range(3),[DISTRIBUTION_LABELS[name] for name in families],rotation=15,ha="right")
    axes[2].set_ylabel("PDMS prediction MAE");axes[2].set_title("Random control families | identical 256 context")
    figure.suptitle("Conditional error diagnosis: shared 256-candidate context\nNative-distance is a geometry proxy, not proof of training density; no scorer is retrained",fontsize=12)
    figure.tight_layout(rect=(0,0,1,.86));figure.savefig(destination/"trajectory_geometry_and_error_slices.png",dpi=160);plt.close(figure)
    write_json(destination/"geometry_error_heatmap.json",{"mae":heatmap.tolist(),"candidate_counts":counts.tolist(),"distance_edges_m":distance_edges,"heading_edges_deg":heading_edges})


def write_distribution_report(configuration, summary, destination):
    operational=summary["operational_separate64_context"]
    lines=["# 원본 DrivoR scorer의 후보 분포별 성능 진단","","## 질문과 범위","",
        "연구의 하위 질문: 상황에 따라 탐색 후보를 조절하기 전에, 어떤 후보 분포에서 frozen scorer의 예측과 선택을 신뢰할 수 있는가?",
        "공식 원 DrivoR Scaling checkpoint(29epoch 완료/53911update), 동일 NAVTEST64/38recording, 기존의 16,384개 후보와 공식 NAVSIM v1 PDMS를 재사용했다. 새로운 학습·trajectory 생성·GPU·공식 평가기 호출은 0이다. 이전에 사용한 개발 패널이며 전체 NAVTEST/독립 최종 평가가 아니다.",
        "선택 지표는 각 분포를 같은 64후보 예산으로 따로 채점했다. 점수 예측과 geometry 분석에는 256경로를 한 번에 채점한 결과도 따로 사용해 후보 간 self-attention context를 고정했다.","",
        "## 후보 분포의 정의","",
        "- DrivoR generated64: 원 DrivoR trajectory generator가 낸 64경로. 이번 모델의 자기 후보 분포이며 학습 데이터의 확률 밀도라고 가정하지 않는다.",
        "- TOAD generated64: TOAD에서 배포한 DrivoR generator가 낸 64경로. 별도 공개 가중치에 의한 후보 분포다.",
        "- Random bouquet64: 공식 generator의 constant22/smooth21/freeform21 제어를 bicycle rollout한 경로. 물리 범위로 제한했지만 객체·지도와의 안전을 보장하지 않는다.",
        "- Nearby64: 이전 TOAD 선택 경로 1 + bicycle 역변환·재전개 중심 1 + TOAD 초기 Gaussian 주변 후보 62. 원 DrivoR 후보를 중심으로 생성한 분포가 아니므로 이 한계를 유지한다.","",
        "## 같은 후보 예산에서의 실제 성능","",
        "| 후보 분포 | 후보 수 | 예측 MAE ↓ | 순위 상관 ↑ | Oracle 최고 | Scorer 선택 | 선택손실 ↓ | NC<1 선택 | DAC<1 선택 |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name,row in operational.items():
        lines.append(f"| {DISTRIBUTION_LABELS[name]} | {row['candidate_count_total']} | {row['prediction_mae_pdms_points']:.3f} | {row['within_scene_spearman_mean']:.3f} | {row['oracle_best_pdms']:.3f} | {row['selected_pdms']:.3f} | {row['selection_regret_pdms_points']:.3f} | {row['selected_nc_failure_count']}/64 | {row['selected_dac_failure_count']}/64 |")
    lines += ["","MAE는 모든 후보의 예측 점수 절대오차, 순위 상관은 같은 장면 안에서 후보 순서를 얼마나 맞추는지, 선택손실은 같은 후보 집합의 최고 실제 PDMS−scorer가 선택한 실제 PDMS다. Oracle은 후보 내 상한이며 항상 100 또는 인간 GT 경로가 아니다.","",
        "## 핵심 관측과 해석","",
        "1. **선택 기준으로는 generator 후보에서 상대적으로 안정적이고, 광범위한 random 후보에서 취약했다.** 자기64후보의 선택손실6.27점 대비 random64는22.44점이었다. Nearby64도8.88점이며 자기 후보보다 선택손실이 컸다. 상대적 비교이며 자기 후보를 정확하게 채점한다는 결론은 아니다.",
        "2. **전체 MAE와 평균 순위 상관만으로 탐색 분포의 안전성을 판단하면 안 된다.** Random의 MAE15.24는 native17.40보다 작고 순위상관도 더 높지만, 실제 best 후보를 놓치는 손실과 안전 실패가 더 컸다. Random은 후보 평균 실제PDMS28.27, native는77.45로 실제 점수 구성이 크게 다르다.",
        "3. **Random 안의 좋은 후보를 과소평가했다.** 같은256context에서 실제90–100점 random 후보232개의 MAE41.38점/평균 signed error−41.32점이었다. Native의 같은 구간2508개는 MAE14.53점이었다. 분포 전체가 어려운 것보다, 유용한 random 후보를 찾아내는 문제가 두드러졌다.",
        "4. **Native에도 위험한 예외를 과대평가하는 문제가 남았다.** 실제NC/DAC미준수 native 후보655개의 실제평균은0점인데 환산예측은평균40.95점이었다. 정상적인 generator 경로와 비슷한 모양만으로 안전을 보장할 수 없다.",
        "5. **자기 후보에서 멀어질수록 오류가 커지는 단순 패턴은 없었다.** RMS0.5–2m 후보 MAE22.65,10m이상5.75점이었다. 후자는 실제평균4.36점으로 대부분 나쁜 후보라 비교가 쉬울 수 있다. 이 설명은 구성 차이를 근거로 한 추론이며 학습 밀도/OOD 원인 측정이 아니다.",
        "6. Constant/smooth/freeform의 MAE는15.66/14.62/15.17점으로, 제어 family만으로 큰 우열을 정할 근거는 부족하다. 서로의CI가 겹치며 causal training effect 비교가 아니다.","",
        "이전 보고의 DrivoR93.70은 두generator64개씩을합친128후보였고 이번93.61은자기64개만단독채점한값이다. 가중치나공식정답을바꾼것이아니라후보구성과scorer context가다른별도평가다.","",
        "**MAE가 낮아도 가장 좋은 후보를 잘 선택한다고 볼 수 없다.** 분포마다 실제 0점/고득점/위험 경로 비율이 달라, 낮은 점수의 쉬운 후보가 많으면 전체 MAE가 낮아질 수 있다. 선택손실·순위·조건부 오류를 함께 본다.","",
        "## 같은 256 후보 context에서 예측 정확도","","| 분포 | 예측 MAE | 오차 ±5점 이내 비율 | 실제 NC/DAC 미준수 후보 비율 |","|---|---:|---:|---:|"]
    for name,row in summary["fixed_shared256_context"].items():
        lines.append(f"| {DISTRIBUTION_LABELS[name]} | {row['prediction_mae_pdms_points']:.3f} | {row['prediction_within5_points_fraction']*100:.2f}% | {row['unsafe_candidate_fraction']*100:.2f}% |")
    lines += ["","## 무작위 제어 패턴별 차이","","| 공식 생성 family | 후보 수 | 예측 MAE | 순위 상관 | 평균 signed error |","|---|---:|---:|---:|---:|"]
    for name,row in summary["random_generator_families_shared256"].items():
        lines.append(f"| {DISTRIBUTION_LABELS[name]} | {row['candidate_count_total']} | {row['prediction_mae_pdms_points']:.3f} | {row['within_scene_spearman_mean']:.3f} | {row['mean_signed_error_pdms_points']:+.3f} |")
    lines += ["","상수 제어는 시간 전체 같은 가속도/yaw rate, smooth는 시작→끝 제어를 선형 변화, freeform은 시간별 noise를 더한 제어다. 원 공식 코드의 `family_mix` 연결 순서로 라벨을 붙였다. 서로 22/21/21개여서 family별 oracle 선택손실 우열을 비교하지 않았다.","",
        "## 기하학적 조건에 따른 예측오차","","| 자기 후보와의 RMS 거리 | 후보 수 | 장면 수 | recording 수 | MAE | 평균 signed error |","|---|---:|---:|---:|---:|---:|"]
    for row in summary["geometric_error_slices_shared256"]["nearest_drivor_proposal_rms_distance_m"]:
        high="inf" if row["upper"]>1e8 else str(row["upper"])
        lines.append(f"| [{row['lower']}, {high})m | {row['candidate_count']} | {row['scene_count']} | {row['recording_count']} | {row['candidate_weighted_mae']:.3f} | {row['signed_error']:+.3f} |")
    lines += ["","거리는 현재 장면의 DrivoR 생성64경로 중 가장 가까운 경로와 8개 waypoint의 평균 제곱 XY거리 제곱근이다. 현재의 geometry 유사성을 측정하며 실제 학습 밀도 또는 엄밀한 OOD 판정을 의미하지 않는다. 끝 heading·jerk proxy와 결합한 heatmap의 각 셀에 후보 수를 표시했다. 속도/가속도/jerk는 계획 waypoint로부터 역산한 proxy이며 실제 simulator의 comfort 지표와 다르다.","",
        "## 후보 context의 영향","","| 분포 | 64→256 context로 같은 분포의 선택 변경 | 선택 PDMS 변화 | 동일 후보 예측값 최대 변화 |","|---|---:|---:|---:|"]
    for name,row in summary["candidate_context_effect"].items():
        lines.append(f"| {DISTRIBUTION_LABELS[name]} | {row['selected_candidate_changed_scenes']}/64 | {row['mean_pdms_change_shared_minus_separate']:+.3f} | {row['maximum_prediction_change_pdms_points']:.3f} |")
    lines += ["","64와256은 candidate 수까지 다르므로 이 검사는 같은크기 context 안정성 검사가 아니다. Context 변화가 존재한다는 기능 진단이며 오류의 원인이라는 증명은 아니다.","",
        "## 그림과 산출물","",
        "- `scorer_performance_by_distribution.png`: 분포별 선택 점수/선택손실·MAE·순위·안전 실패·실제 점수 구성.",
        "- `predicted_vs_actual_by_distribution.png`: 오른쪽 아래는 과대평가한 나쁜 경로, 왼쪽 위는 과소평가한 좋은 경로. 색 농도는 후보 수.",
        "- `trajectory_candidate_distributions.png`: 같은 현재 영상에서 서로 다른 64후보의 실제 경로 분포. 보라색은 낮은 실제 PDMS, 노란색은 높은 실제 PDMS. 직진/좌회전 명령의 저장 순서 첫 장면을 사용해 오류가 큰 장면만 고르지 않았다.",
        "- `trajectory_geometry_and_error_slices.png`: geometry·실제 점수 구간·random family별 오류. 실제 점수 구간은 retrospective GT 분석이므로 inference-time 구분기로 사용하지 않는다.",
        "- `results.json`: 명령별 결과, score-band와 안전 조건부 오류, recording bootstrap CI 포함.",
        "- `candidate_diagnostics.csv`: 16,384개 개별 후보의 실제·예측·기하학 proxy와 sample/recording ID.","",
        "## 확인한 것과 아직 모르는 것","",
        "Frozen weights 불변/gradient0/backbone 호출0, GPU↔CPU cached score replay를 검증했다. 이번 고정된 후보 분포에서 scorer 오류를 분해했으며, 더 넓은 NAVTEST와 새로운 random seeds에서도 같은 특성이 유지되는지는 아직 검증하지 않았다. Random/near는 TOAD에서 유래했으며 현재 분포 차이로 학습 과정의 원인을 확정하지 않는다. Bootstrap은 recording 표본 변동만 반영하며 학습 seed 변동은 아니다. 위험 label은 공식 NC<1 또는 DAC<1로 정의했으며 NC 부분감점도 포함한다.","",
        "## 다음 확인 하나, 미실행","",
        "이번에관측된특성은자기후보와가까운거리의안전경계/유용한random후보에서score를다르게보정해야할가능성을시사한다. 우선같은점수구간과같은후보예산에서작은위치·속도변화가NC/DAC판단을어떻게바꾸는지paired검사를권고한다. 현재진단만으로geometry threshold기반필터나새학습의개선을확정하지않으며자동학습/후속queue는시작하지않았다.","",
        "## 비용·재실행","",
        f"CPU rescoring wall {summary['cost_and_checks']['cpu_phase_wall_seconds']:.2f}초, CPU {configuration['cpu_threads']}thread/microbatch {configuration['cpu_microbatch_scenes']}scene. 새GPU0초·학습0·공식평가0. 기존공식16,384후보 측정 재사용. 새CPU scorer scene-forward {summary['cost_and_checks']['new_scorer_scene_forwards']}회.","",
        "```bash","cd /rhome/junseong/PlanningAwareFuturePrediction","bash scripts/run_drivor_candidate_distribution_diagnosis.sh","```"]
    (destination/"report.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--configuration",type=Path,required=True)
    parser.add_argument("--phase",choices=("rescore","analyze"),required=True)
    arguments=parser.parse_args()
    configuration=json.loads(arguments.configuration.read_text())
    if arguments.phase=="rescore":prepare_and_rescore_cached_candidates(configuration)
    else:analyze_distributions_and_plot(configuration)
