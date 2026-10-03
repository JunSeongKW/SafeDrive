"""Matched PDMS, future-representation probes, and actual trajectory figures."""
import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from report_drive_jepa_region_research import compare_metric, context_breakdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs/lpwm_planning_v1"
RESULT_ROOT = PROJECT_ROOT / "results/lpwm_planning_v1"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def occupancy_targets(cache):
    # Current-visible annotated objects only; fixed current-ego BEV for every horizon.
    grid_x = torch.linspace(-5, 65, 8)
    grid_y = torch.linspace(-35, 35, 8)
    grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="ij"), -1).flatten(0, 1)
    future_positions = cache["object_future_xy"][:, :, [0, 1, 3, 7]]
    valid = cache["object_future_valid"][:, :, [0, 1, 3, 7]] & cache["object_valid"][:, :, None]
    squared_distance = (future_positions[..., None, :] - grid).square().sum(-1)
    occupancy = (-squared_distance / (2 * 5**2)).exp() * valid[..., None]
    future_occupancy = occupancy.amax(1).flatten(1)
    current_distance = (cache["object_current_xy"][..., None, :] - grid).square().sum(-1)
    current_occupancy = ((-current_distance / (2 * 5**2)).exp() * cache["object_valid"][..., None]).amax(1)
    return future_occupancy, future_occupancy - current_occupancy[:, None].expand(-1, 4, -1).flatten(1)


def fit_common_probe(train_features, dev_features, train_targets, dev_targets):
    train_features, dev_features = train_features.double(), dev_features.double()
    mean, scale = train_features.mean(0), train_features.std(0).clamp_min(.05)
    design = (train_features - mean) / scale
    development = (dev_features - mean) / scale
    # The projection dimension, seed, alpha, and target grid never depend on dev performance.
    if design.shape[-1] > 256:
        projection = torch.randn(design.shape[-1], 256, generator=torch.Generator().manual_seed(6317), dtype=torch.float64) / design.shape[-1]**.5
        design, development = design @ projection, development @ projection
    target_mean = train_targets.double().mean(0)
    weights = torch.linalg.solve(design @ design.T + 10 * torch.eye(len(design), dtype=torch.float64), train_targets.double() - target_mean)
    prediction = development @ design.T @ weights + target_mean
    return (prediction - dev_targets.double()).square().mean(-1)


def representation_probes(specification, records, reports):
    cache = torch.load(OUTPUT_ROOT / "supervised_cache.pt", map_location="cpu", weights_only=True)
    train_indices = [index for index, row in enumerate(records) if row["split"] == "train"]
    dev_indices = [index for index, row in enumerate(records) if row["split"] == "development"]
    target, change_target = occupancy_targets(cache)
    train_targets, dev_targets = target[train_indices], target[dev_indices]
    scores = {}
    baselines = {"zero_occupancy": dev_targets.square().mean(-1), "train_mean_occupancy": (dev_targets - train_targets.mean(0)).square().mean(-1),
                 "ego_status_probe": fit_common_probe(cache["ego_status"][train_indices], cache["ego_status"][dev_indices], train_targets, dev_targets)}
    for name, errors in baselines.items():
        scores[name] = {"mse": float(errors.mean()), "window_mse": errors.tolist()}
    change_scores = {"current_occupancy_persistence": {"mse": float(change_target[dev_indices].square().mean())}}
    ego_change_errors = fit_common_probe(cache["ego_status"][train_indices], cache["ego_status"][dev_indices], change_target[train_indices], change_target[dev_indices])
    change_scores["ego_status_probe"] = {"mse": float(ego_change_errors.mean()), "window_mse": ego_change_errors.tolist()}
    for name in reports:
        saved = torch.load(OUTPUT_ROOT / "runs" / name / "encoder_probe_features.pt", map_location="cpu", weights_only=True)
        train_features = torch.cat((saved["train"], cache["ego_status"][train_indices]), -1)
        dev_features = torch.cat((saved["development"], cache["ego_status"][dev_indices]), -1)
        errors = fit_common_probe(train_features, dev_features, train_targets, dev_targets)
        scores[name] = {"mse": float(errors.mean()), "window_mse": errors.tolist()}
        change_errors = fit_common_probe(train_features, dev_features, change_target[train_indices], change_target[dev_indices])
        change_scores[name] = {"mse": float(change_errors.mean()), "window_mse": change_errors.tolist()}
    result = {"target": "current-visible-object future BEV Gaussian occupancy at0.5/1/2/4seconds; 8x8grid; sigma5m; current-ego coordinates",
              "readout": "train-standardized flattened raw encoder attributes + status; fixed256D Gaussian projection seed6317; ridge alpha10; train only fit",
              "scope": "Probe diagnostic, not object tracking or standalone safety metric; change targets distinguish static reconstruction from object motion",
              "methods": scores, "future_change_methods": change_scores}
    write_json(RESULT_ROOT / "representation_probes.json", result)
    return result


def build_figures(summary, reports, records, baseline_predictions):
    directory = OUTPUT_ROOT / "visualization"
    directory.mkdir(exist_ok=True)
    labels = {"drive_jepa_reference": "Drive-JEPA reference", "frozen_particles": "Frozen LPWM + planner", "planning_joint": "LPWM + planning", "object_future_uniform": "+ uniform object futures", "object_future_risk": "+ planning-weighted futures", "object_future_risk_no_intent": "Weighted, no encoder intent", "ego_only": "Ego status only"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    table = summary["table"]
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.7), gridspec_kw={"wspace": .5})
    colors = ["#536878", "#9a9ea3", "#4277b5", "#db9833", "#2b9972", "#945baa", "#b96262"]
    positions = np.arange(len(table))
    for axis, key, error_key, title in zip(axes, ["pdm_percent", "development_ade_m", "future_probe_mse"], ["pdm_seed_std_pp", "ade_seed_std_m", None], ["Official development PDMS (higher is better)", "Ego trajectory ADE (m; lower is better)", "Frozen linear future-occupancy probe (MSE)"]):
        values = [row.get(key, np.nan) if row.get(key) is not None else np.nan for row in table]
        errors = [row.get(error_key, 0) for row in table] if error_key else None
        axis.barh(positions, values, xerr=errors, color=colors, capsize=3)
        axis.set_yticks(positions, [labels[row["condition"]] for row in table] if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.set_title(title, fontsize=10)
        axis.grid(axis="x", alpha=.2)
        for position, value in zip(positions, values):
            if np.isfinite(value):
                axis.text(value, position, f" {value:.3f}", va="center", fontsize=9)
        axis.margins(x=.2)
    figure.suptitle("LPWM encoder learning + standalone planner | 512 train / 192 development scenes\n3 seeds, final 1,000 updates; error bars = seed SD; development evaluation, not independent test", fontsize=13)
    figure.subplots_adjust(top=.8, left=.19, right=.97, bottom=.1)
    figure.savefig(directory / "01_planning_results.png", dpi=170)
    figures = [figure]
    records_by_token = {row["current_frame_token"]: row for row in records}
    by_model = {condition: {row["token"]: row for row in reports[f"{condition}_seed47"]["evaluations"]["development"]["windows"]}
                for condition in ("frozen_particles", "planning_joint", "object_future_risk")}
    by_model["drive_jepa_reference"] = {row["token"]: row for row in baseline_predictions}
    selected = []
    for scenario in ("straight", "turn", "projected_overlap"):
        candidates = [row for row in records if row["split"] == "development" and row["scenario"] == scenario]
        selected.append(sorted(candidates, key=lambda row: __import__("hashlib").sha256(("lpwm-planning-figures:" + row["current_frame_token"]).encode()).hexdigest())[0])
    cache = torch.load(OUTPUT_ROOT / "supervised_cache.pt", map_location="cpu", weights_only=True)
    indices_by_token = {row["current_frame_token"]: index for index, row in enumerate(records)}
    figure, axes = plt.subplots(3, 2, figsize=(14, 12), gridspec_kw={"width_ratios": [1.2, 1]})
    for row_index, record in enumerate(selected):
        token = record["current_frame_token"]
        cache_index = indices_by_token[token]
        with Image.open(record["observed_image_paths"][-1]) as photo:
            axes[row_index, 0].imshow(photo)
        axes[row_index, 0].set_axis_off()
        axes[row_index, 0].set_title(f"{record['scenario']} | {token}\nActual current front image; seed47 chosen before outcomes", fontsize=10)
        axis = axes[row_index, 1]
        target = cache["ego_trajectory_target"][cache_index].numpy()
        axis.plot(np.r_[0, target[:, 1]], np.r_[0, target[:, 0]], "k--o", lw=2.2, markersize=3, label="Expert GT")
        for condition, color in (("drive_jepa_reference", "#536878"), ("frozen_particles", "#9a9ea3"), ("planning_joint", "#4277b5"), ("object_future_risk", "#2b9972")):
            trajectory = np.array(by_model[condition][token]["trajectory"])
            axis.plot(np.r_[0, trajectory[:, 1]], np.r_[0, trajectory[:, 0]], "-o", lw=1.8, color=color, markersize=3, label=labels[condition])
        axis.scatter([0], [0], marker="^", c="black", s=60)
        axis.set_xlabel("Current-ego lateral y (m)")
        axis.set_ylabel("Current-ego forward x (m)")
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(alpha=.25)
        pdm_by_condition = summary["per_scene_seed47"]
        numbers = " | ".join(f"{short}: {pdm_by_condition[condition][token]*100:.1f}" for condition, short in (("planning_joint", "planning"), ("object_future_risk", "+future"), ("drive_jepa_reference", "Drive")))
        axis.set_title("PDMS: " + numbers, fontsize=10)
        if row_index == 0:
            axis.legend(fontsize=8, loc="best")
    figure.suptitle("Actual predicted trajectories on three outcome-independent development examples\nA single future branch and encoder are trained together; no future GT is fed to the planner", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, .95))
    figure.savefig(directory / "02_actual_trajectories.png", dpi=170)
    figures.append(figure)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    components = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort"]
    for condition, color in (("planning_joint", "#4277b5"), ("object_future_uniform", "#db9833"), ("object_future_risk", "#2b9972"), ("ego_only", "#b96262")):
        axes[0].plot(range(len(components)), [summary["methods"][condition]["pdm_components"][key] * 100 for key in components], "-o", label=labels[condition], color=color)
    axes[0].set_xticks(range(len(components)), ["No collision", "Drivable", "Progress", "TTC", "Comfort"], rotation=20)
    axes[0].set_ylabel("Official component (%)")
    axes[0].set_ylim(0, 105)
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8)
    scenarios = ["straight", "turn", "projected_overlap", "other"]
    for offset, (condition, color) in enumerate((("planning_joint", "#4277b5"), ("object_future_uniform", "#db9833"), ("object_future_risk", "#2b9972"))):
        values = [summary["methods"][condition]["scenario"][scenario]["pdm_percent"] for scenario in scenarios]
        axes[1].bar(np.arange(4) + (offset-1)*.24, values, width=.24, label=labels[condition], color=color)
    axes[1].set_xticks(range(4), ["Straight (12)", "Turn (25)", "Overlap proxy (147)", "Other (8)"], rotation=15)
    axes[1].set_ylabel("PDMS (%)")
    axes[1].set_ylim(0, 105)
    axes[1].grid(axis="y", alpha=.2)
    figure.suptitle("Safety/progress components and scenario breakdown | 192 development scenes")
    figure.tight_layout()
    figure.savefig(directory / "03_components_and_scenarios.png", dpi=170)
    figures.append(figure)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for condition, color in (("frozen_particles", "#9a9ea3"), ("planning_joint", "#4277b5"), ("object_future_uniform", "#db9833"), ("object_future_risk", "#2b9972")):
        logged = [list(map(json.loads, (OUTPUT_ROOT / "runs" / f"{condition}_seed{seed}" / "training_log.jsonl").read_text().splitlines())) for seed in (29, 47, 71)]
        steps = [row["update"] for row in logged[0]]
        planning_curve = np.mean([[row["planning"] for row in seed_log] for seed_log in logged], axis=0)
        axes[0].plot(steps, planning_curve, label=labels[condition], color=color)
        if condition.startswith("object_future"):
            future_curve = np.mean([[row["future_state"] for row in seed_log] for seed_log in logged], axis=0)
            axes[1].plot(steps, future_curve, label=labels[condition], color=color)
    axes[0].set_title("Sampled training planning loss")
    axes[1].set_title("Sampled training object-future loss")
    for axis in axes[:2]:
        axis.set_xlabel("Gradient updates")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    learned_rows = table[1:]
    axes[2].bar(np.arange(6)-.17, [row["train_ade_m"] for row in learned_rows], width=.34, label="Train")
    axes[2].bar(np.arange(6)+.17, [row["development_ade_m"] for row in learned_rows], width=.34, label="Development")
    axes[2].set_xticks(range(6), ["Frozen", "Planning", "Uniform", "Weighted", "No intent", "Ego only"], rotation=30)
    axes[2].set_ylabel("Final trajectory ADE (m)")
    axes[2].legend()
    figure.suptitle("Training-data diagnostics: 512 unique scenes, 8,000 sampled clips/model (~15.6 passes)\nTraining curves are sampled batches; no intermediate development checkpoints were selected", fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, .91))
    figure.savefig(directory / "04_training_diagnostics.png", dpi=170)
    figures.append(figure)
    with PdfPages(RESULT_ROOT / "lpwm_planning_figures.pdf") as pdf:
        for figure in figures:
            pdf.savefig(figure)
            plt.close(figure)
    write_json(RESULT_ROOT / "visualization_selection.json", {"seed": 47, "rule": "fixed SHA256 sorting within scenario, no metric-based selection", "tokens": [row["current_frame_token"] for row in selected]})
    (directory / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>LPWM planning evaluation</title><style>body{max-width:1500px;margin:auto;font-family:sans-serif}img{width:100%}</style><h1>LPWM planning: measured results</h1><p>512 train / 192 development; not an independent test.</p>' + ''.join(f'<img src="{name}">' for name in ("01_planning_results.png", "02_actual_trajectories.png", "03_components_and_scenarios.png", "04_training_diagnostics.png")))


def main():
    torch.set_num_threads(2)
    specification = json.loads((PROJECT_ROOT / "configs/lpwm_planning/controlled_v1.json").read_text())
    assert (OUTPUT_ROOT / "pdm_complete.json").exists()
    records = json.loads((OUTPUT_ROOT / "manifest.json").read_text())["records"]
    records_by_token = {row["current_frame_token"]: row for row in records}
    reports = {f"{condition}_seed{seed}": json.loads((OUTPUT_ROOT / "runs" / f"{condition}_seed{seed}" / "results.json").read_text()) for condition in specification["conditions"] for seed in specification["seeds"]}
    probes = representation_probes(specification, records, reports)
    baseline_predictions = json.loads((PROJECT_ROOT / "outputs/encoder_future_learning_v1/original_baseline.json").read_text())["windows"]
    pdm = {name: json.loads(path.read_text()) for path in sorted((RESULT_ROOT / "pdm").glob("*.json")) if (name := path.stem)}
    baseline_ade = float(np.mean([row["xy_ade_m"] for row in baseline_predictions]))
    table = [{"condition": "drive_jepa_reference", "pdm_percent": 100 * pdm["drive_jepa_reference"]["summary"]["score"], "pdm_seed_std_pp": 0.,
              "development_ade_m": baseline_ade, "ade_seed_std_m": 0., "train_ade_m": None, "future_probe_mse": None, "future_change_probe_mse": None}]
    methods, pdm_rows, ade_rows = {}, {}, {}
    for condition in specification["conditions"]:
        names = [f"{condition}_seed{seed}" for seed in specification["seeds"]]
        pdm_values = [100 * pdm[name]["summary"]["score"] for name in names]
        ade_values = [reports[name]["evaluations"]["development"]["mean_ade_m"] for name in names]
        train_ade_values = [reports[name]["evaluations"]["train"]["mean_ade_m"] for name in names]
        probe_values = [probes["methods"][name]["mse"] for name in names]
        table.append({"condition": condition, "pdm_percent": float(np.mean(pdm_values)), "pdm_seed_std_pp": float(np.std(pdm_values, ddof=1)),
                      "development_ade_m": float(np.mean(ade_values)), "ade_seed_std_m": float(np.std(ade_values, ddof=1)),
                      "train_ade_m": float(np.mean(train_ade_values)), "future_probe_mse": float(np.mean(probe_values)),
                      "future_change_probe_mse": float(np.mean([probes["future_change_methods"][name]["mse"] for name in names]))})
        pdm_rows[condition] = {seed: pdm[name]["windows"] for seed, name in zip(specification["seeds"], names)}
        ade_rows[condition] = {seed: reports[name]["evaluations"]["development"]["windows"] for seed, name in zip(specification["seeds"], names)}
        scenario = {}
        for label in ("straight", "turn", "projected_overlap", "other"):
            tokens = {row["current_frame_token"] for row in records if row["split"] == "development" and row["scenario"] == label}
            scenario[label] = {"count": len(tokens), "pdm_percent": 100 * float(np.mean([row["score"] for rows in pdm_rows[condition].values() for row in rows if row["token"] in tokens])),
                               "ade_m": float(np.mean([row["xy_ade_m"] for rows in ade_rows[condition].values() for row in rows if row["token"] in tokens]))}
        object_metrics = {}
        for metric in ("object_future_ade_m", "risk_weighted_object_future_ade_m", "matched_box_iou", "matched_box_recall_at_03"):
            values = [row[metric] for rows in ade_rows[condition].values() for row in rows if row.get(metric) is not None]
            object_metrics[metric] = float(np.mean(values)) if values else None
        methods[condition] = {"pdm_by_seed": dict(zip(map(str, specification["seeds"]), pdm_values)), "ade_by_seed": dict(zip(map(str, specification["seeds"]), ade_values)),
                              "pdm_components": {key: float(np.mean([pdm[name]["summary"][key] for name in names])) for key in pdm[names[0]]["summary"]},
                              "scenario": scenario, "object_metrics": object_metrics, "validation": {str(seed): reports[name]["validation"] for seed, name in zip(specification["seeds"], names)}}
    for seed in specification["seeds"]:
        assert len({reports[f"{condition}_seed{seed}"]["batch_schedule_sha256"] for condition in specification["conditions"]}) == 1
    pdm_rows["drive_jepa_reference"] = {seed: pdm["drive_jepa_reference"]["windows"] for seed in specification["seeds"]}
    ade_rows["drive_jepa_reference"] = {seed: baseline_predictions for seed in specification["seeds"]}
    comparisons = specification["primary_comparisons"] + [["object_future_risk", "ego_only"], ["object_future_risk", "drive_jepa_reference"]]
    interventions = {}
    for intervention in ("zero_predicted_futures", "shuffle_scene_particles"):
        intervention_pdm = {seed: pdm[f"object_future_risk_seed{seed}__{intervention}"]["windows"] for seed in specification["seeds"]}
        intervention_ade = {seed: reports[f"object_future_risk_seed{seed}"]["evaluations"]["development"][intervention] for seed in specification["seeds"]}
        interventions[intervention] = {"pdm_delta": compare_metric(intervention_pdm, pdm_rows["object_future_risk"], "score"),
                                      "ade_delta": compare_metric(intervention_ade, ade_rows["object_future_risk"], "xy_ade_m"),
                                      "trajectory_change_m": float(np.mean([row["trajectory_change_m"] for rows in intervention_ade.values() for row in rows]))}
    summary = {"scope": "512 navtrain-subset training /192 development scenes from disjoint recording groups; not full navtrain or navtest",
               "specification": specification, "table": table, "methods": methods,
               "paired_pdm": {f"{proposed}_minus_{reference}": compare_metric(pdm_rows[proposed], pdm_rows[reference], "score") for proposed, reference in comparisons},
               "paired_ade": {f"{proposed}_minus_{reference}": compare_metric(ade_rows[proposed], ade_rows[reference], "xy_ade_m") for proposed, reference in comparisons},
               "context_breakdown": {condition: context_breakdown(rows, pdm_rows["drive_jepa_reference"], records_by_token, "score") for condition, rows in pdm_rows.items()},
               "interventions": interventions, "probe_baselines": {key: value["mse"] for key, value in probes["methods"].items() if "seed" not in key},
               "future_change_probe_baselines": {key: value["mse"] for key, value in probes["future_change_methods"].items() if "seed" not in key},
               "training_run_count": len(reports), "total_updates": sum(report["completed_updates"] for report in reports.values()),
               "total_training_seconds": sum(report["training_seconds"] for report in reports.values()), "maximum_peak_allocated_gib": max(report["peak_allocated_gib"] for report in reports.values()),
               "per_scene_seed47": {condition: {row["token"]: row["score"] for row in pdm_rows[condition][47]} for condition in pdm_rows}}
    write_json(RESULT_ROOT / "summary.json", summary)
    with (RESULT_ROOT / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    write_json(RESULT_ROOT / "training_reports.json", {name: {key: value for key, value in report.items() if key != "evaluations"} for name, report in reports.items()})
    write_json(RESULT_ROOT / "prediction_rows.json", {condition: {str(seed): rows for seed, rows in by_seed.items()} for condition, by_seed in ade_rows.items()})
    for filename in ("manifest.json", "registration.json"):
        shutil.copy2(OUTPUT_ROOT / filename, RESULT_ROOT / filename)
    build_figures(summary, reports, records, baseline_predictions)
    write_json(RESULT_ROOT / "completion.json", {"complete": True, "runs": len(reports), "updates": summary["total_updates"], "paired_schedules_identical": True, "pdm_complete": True})
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
