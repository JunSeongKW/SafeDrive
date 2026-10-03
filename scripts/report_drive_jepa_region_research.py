"""Aggregate every preregistered final run; no checkpoint or condition selection."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from summarize_drive_jepa_architecture_followup import paired_recording_comparison

WORKSPACE = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scene_macro_ade(rows):
    scene_errors = {}
    for row in rows:
        scene_errors.setdefault(row["scene_token"], []).append(row["xy_ade_m"])
    return float(np.mean([np.mean(errors) for errors in scene_errors.values()]))


def compare_metric(proposed, reference, metric):
    differences_by_recording = {}
    seed_differences = []
    for seed, proposed_rows in proposed.items():
        reference_by_token = {row["token"]: row for row in reference[seed]}
        if set(reference_by_token) != {row["token"] for row in proposed_rows}:
            raise RuntimeError("Paired metric comparison token mismatch")
        differences = []
        for row in proposed_rows:
            difference = row[metric] - reference_by_token[row["token"]][metric]
            differences.append(difference)
            differences_by_recording.setdefault(row["recording"], {}).setdefault(row["token"], []).append(difference)
        seed_differences.append({"seed": seed, "difference": float(np.mean(differences))})
    recording_values = [np.asarray([np.mean(values) for values in tokens.values()])
                        for _, tokens in sorted(differences_by_recording.items())]
    generator = np.random.default_rng(20261003)
    bootstrap = [float(np.concatenate([recording_values[index] for index in generator.integers(
        0, len(recording_values), len(recording_values))]).mean()) for _ in range(2000)]
    return {"mean_difference": float(np.mean([row["difference"] for row in seed_differences])),
            "paired_seed_differences": seed_differences,
            "recording_cluster_bootstrap_95_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
            "scope": "seed-averaged window differences, recording resampling, unadjusted for multiple comparisons"}


def context_breakdown(rows_by_seed, baseline_by_seed, records_by_token, metric="xy_ade_m"):
    definitions = {
        "command": lambda record: str(record["command_raw_index"]),
        "speed_cutoffs_0.5_5_mps": lambda record: ("stationary" if record["ego_speed_meters_per_second"] <= .5
            else "slow" if record["ego_speed_meters_per_second"] <= 5 else "faster"),
        "speed_cutoffs_1_8_mps": lambda record: ("stationary" if record["ego_speed_meters_per_second"] <= 1
            else "slow" if record["ego_speed_meters_per_second"] <= 8 else "faster"),
    }
    result = {}
    for definition, label in definitions.items():
        grouped = {}
        for seed, rows in rows_by_seed.items():
            baseline = {row["token"]: row for row in baseline_by_seed[seed]}
            for row in rows:
                group = grouped.setdefault(label(records_by_token[row["token"]]), {"tokens": set(), "values": [], "differences": []})
                group["tokens"].add(row["token"])
                group["values"].append(row[metric])
                group["differences"].append(row[metric] - baseline[row["token"]][metric])
        result[definition] = {name: {"unique_windows": len(group["tokens"]),
                                    "metric": metric,
                                    "mean": float(np.mean(group["values"])),
                                    "mean_difference_from_original": float(np.mean(group["differences"]))}
                              for name, group in grouped.items()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--pdm-directory", type=Path, required=True)
    parser.add_argument("--preserved-control-directory", type=Path, required=True)
    parser.add_argument("--share-directory", type=Path, required=True)
    parser.add_argument("--wait-for-completion", action="store_true")
    args = parser.parse_args()
    if args.wait_for_completion:
        wait_started = time.perf_counter()
        required_completions = [directory / "completion.json" for directory in (
            args.run_directory, args.pdm_directory, args.preserved_control_directory)]
        while not all(path.exists() for path in required_completions):
            if time.perf_counter() - wait_started > 11400:
                raise RuntimeError("Registered comparison completion wait cap")
            time.sleep(30)
    specification = json.loads((args.run_directory / "specification.json").read_text())
    completion = json.loads((args.run_directory / "completion.json").read_text())
    if not completion["complete"]:
        raise RuntimeError("All training conditions must complete")
    args.share_directory.mkdir(parents=True, exist_ok=True)
    records = json.loads((WORKSPACE / specification["reused_cache"]).read_text())["records"]
    records_by_token = {record["current_frame_token"]: record for record in records}
    methods, rows_by_condition, table = {}, {}, []
    for condition in specification["conditions"]:
        reports = {seed: json.loads((args.run_directory / f"{condition}_seed{seed}/results.json").read_text())
                   for seed in specification["seeds"]}
        rows_by_condition[condition] = {seed: report["evaluations"]["800"]["development"]["windows"]
                                        for seed, report in reports.items()}
        if "original_frozen" not in rows_by_condition:
            rows_by_condition["original_frozen"] = {seed: report["evaluations"]["0"]["development"]["windows"]
                                                     for seed, report in reports.items()}
        metrics = {}
        for name in ("scene_macro_xy_ade_m", "official_il_loss_whole_split", "future_mse", "persistence_mse"):
            values = [report["evaluations"]["800"]["development"]["summary"][name] for report in reports.values()]
            metrics[name] = {"mean": float(np.mean(values)), "seed_std": float(np.std(values, ddof=1)), "by_seed": dict(zip(specification["seeds"], values))}
        metrics["train_scene_macro_xy_ade_m"] = float(np.mean([report["evaluations"]["800"]["train"]["summary"]["scene_macro_xy_ade_m"] for report in reports.values()]))
        metrics["future_mse_divided_by_own_persistence_mse"] = (
            metrics["future_mse"]["mean"] / metrics["persistence_mse"]["mean"])
        metrics["mean_edge_density"] = float(np.mean([row["edge_density"] for rows in rows_by_condition[condition].values() for row in rows]))
        metrics["mean_wall_seconds_per_seed"] = float(np.mean([report["wall_seconds"] for report in reports.values()]))
        metrics["maximum_peak_allocated_gib"] = max(report["peak_allocated_gib"] for report in reports.values())
        metrics["active_parameter_count"] = reports[specification["seeds"][0]]["active_parameter_count"]
        selection_replacement = []
        for seed, report in reports.items():
            initial = {row["token"]: set(row["selected_region_ids"]) for row in report["evaluations"]["0"]["development"]["windows"]}
            selection_replacement.extend(1 - len(initial[row["token"]] & set(row["selected_region_ids"])) / 8
                                         for row in rows_by_condition[condition][seed])
        metrics["mean_selected_set_replacement_fraction"] = float(np.mean(selection_replacement))
        methods[condition] = metrics
        table.append({"condition": condition, "dev_ade_m": metrics["scene_macro_xy_ade_m"]["mean"],
                      "seed_std_ade_m": metrics["scene_macro_xy_ade_m"]["seed_std"],
                      "train_ade_m": metrics["train_scene_macro_xy_ade_m"],
                      "official_il_loss": metrics["official_il_loss_whole_split"]["mean"],
                      "edge_density": metrics["mean_edge_density"],
                      "seconds_per_seed": metrics["mean_wall_seconds_per_seed"],
                      "peak_allocated_gib": metrics["maximum_peak_allocated_gib"]})
    preserved_root = WORKSPACE / "outputs/drive_jepa_selective_future/spatial_region_selection_v1_20261003"
    for label, condition in (("preserved_global_learned", "region8_planning"), ("preserved_global_random", "region8_random")):
        rows_by_condition[label] = {seed: json.loads((preserved_root / f"{condition}_seed{seed}/results.json").read_text())["evaluations"]["800"]["development"]["windows"]
                                    for seed in specification["seeds"]}
    comparisons = {}
    pairs = specification["primary_comparisons"] + specification["secondary_comparisons"] + [[condition, "original_frozen"] for condition in specification["conditions"]]
    for proposed, reference in pairs:
        comparisons[f"{proposed}_minus_{reference}"] = paired_recording_comparison(rows_by_condition[proposed], rows_by_condition[reference])
    selection_advantages = {}
    for label, learned, random_control in (
        ("global", "preserved_global_learned", "preserved_global_random"),
        ("located_dense", "located_dense_learned", "located_dense_random"),
        ("located_sparse", "located_sparse_learned", "located_sparse_random")):
        selection_advantages[label] = {}
        for seed, rows in rows_by_condition[learned].items():
            control = {row["token"]: row for row in rows_by_condition[random_control][seed]}
            selection_advantages[label][seed] = [dict(row, xy_ade_m=row["xy_ade_m"] - control[row["token"]]["xy_ade_m"])
                                                 for row in rows]
    exploratory_selection_interactions = {
        label: paired_recording_comparison(selection_advantages[label], selection_advantages["global"])
        for label in ("located_dense", "located_sparse")}
    contexts = {condition: context_breakdown(rows, rows_by_condition["original_frozen"], records_by_token)
                for condition, rows in rows_by_condition.items()}
    pdm_path = args.pdm_directory / "results.json"
    pdm_report = json.loads(pdm_path.read_text()) if pdm_path.exists() else None
    preserved_pdm_path = args.preserved_control_directory / "results.json"
    if pdm_report and preserved_pdm_path.exists():
        preserved_pdm = json.loads(preserved_pdm_path.read_text())
        for key in ("summary", "windows"):
            pdm_report[key].update(preserved_pdm[key])
    pdm_comparisons = {}
    pdm_contexts = {}
    if pdm_report:
        pdm_rows = {condition: {seed: pdm_report["windows"][f"{condition}_seed{seed}"] for seed in specification["seeds"]}
                    for condition in specification["conditions"]}
        pdm_rows["original_frozen"] = {seed: pdm_report["windows"]["original_frozen"] for seed in specification["seeds"]}
        for label in ("preserved_global_learned", "preserved_global_random"):
            if f"{label}_seed29" in pdm_report["windows"]:
                pdm_rows[label] = {seed: pdm_report["windows"][f"{label}_seed{seed}"] for seed in specification["seeds"]}
        for row in table:
            condition = row["condition"]
            values = [pdm_report["summary"][f"{condition}_seed{seed}"]["score"] * 100 for seed in specification["seeds"]]
            row["development_pdm_score_percent"] = float(np.mean(values))
            row["development_pdm_seed_std"] = float(np.std(values, ddof=1))
            methods[condition]["pdm_mean_percent"] = row["development_pdm_score_percent"]
            methods[condition]["pdm_by_seed_percent"] = dict(zip(specification["seeds"], values))
            methods[condition]["pdm_components_percent"] = {
                metric: float(np.mean([pdm_report["summary"][f"{condition}_seed{seed}"][metric] * 100
                                       for seed in specification["seeds"]]))
                for metric in pdm_report["summary"]["original_frozen"]}
        for proposed, reference in pairs:
            if proposed in pdm_rows and reference in pdm_rows:
                pdm_comparisons[f"{proposed}_minus_{reference}"] = compare_metric(pdm_rows[proposed], pdm_rows[reference], "score")
        pdm_contexts = {condition: context_breakdown(rows, pdm_rows["original_frozen"], records_by_token, "score")
                        for condition, rows in pdm_rows.items()}
    baseline_ade = scene_macro_ade(rows_by_condition["original_frozen"][specification["seeds"][0]])
    reference_metrics = {
        label: {"dev_ade_m": float(np.mean([scene_macro_ade(rows) for rows in rows_by_condition[label].values()])),
                "dev_pdm_percent": float(np.mean([pdm_report["summary"][f"{label}_seed{seed}"]["score"] * 100
                                                   for seed in specification["seeds"]]))
                if pdm_report and f"{label}_seed29" in pdm_report["summary"] else None}
        for label in ("preserved_global_learned", "preserved_global_random")}
    result = {"scope": "development only, fixed final800, exploratory multiple comparisons",
              "baseline_dev_ade_m": baseline_ade,
              "baseline_dev_pdm_percent": pdm_report["summary"]["original_frozen"]["score"] * 100 if pdm_report else None,
              "methods": methods, "paired_ade_comparisons": comparisons,
              "preserved_reference_metrics": reference_metrics,
              "exploratory_selection_difference_in_differences": exploratory_selection_interactions,
              "paired_pdm_comparisons": pdm_comparisons, "context_breakdown": contexts,
              "pdm_context_breakdown": pdm_contexts,
              "completion": completion, "specification": specification}
    write_json(args.share_directory / "summary.json", result)
    with (args.share_directory / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for condition, rows in rows_by_condition.items():
        write_json(args.share_directory / f"{condition}_development_windows.json", rows)
    if pdm_report:
        write_json(args.share_directory / "development_pdm_results.json", pdm_report)
    render_comparison_figure(result, args.share_directory)
    print(json.dumps({"baseline_dev_ade_m": baseline_ade, "conditions": table}, indent=2))


def render_comparison_figure(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(result["specification"]["conditions"])
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for index, condition in enumerate(names):
        comparison = result["paired_ade_comparisons"][f"{condition}_minus_original_frozen"]
        difference = comparison["mean_difference_m"] * 1000
        interval = np.asarray(comparison["recording_cluster_bootstrap_95_ci_m"]) * 1000
        axes[0].plot(interval, [index, index], color="#346888", linewidth=2)
        axes[0].plot(difference, index, "o", color="#346888")
        if result["paired_pdm_comparisons"]:
            comparison = result["paired_pdm_comparisons"][f"{condition}_minus_original_frozen"]
            difference = comparison["mean_difference"] * 100
            interval = np.asarray(comparison["recording_cluster_bootstrap_95_ci"]) * 100
            axes[1].plot(interval, [index, index], color="#a5602e", linewidth=2)
            axes[1].plot(difference, index, "o", color="#a5602e")
    axes[0].set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("ADE change from original (mm); lower is better")
    axes[1].set_xlabel("PDM change from original (percentage points); higher is better")
    for axis in axes:
        axis.axvline(0, color="#777777", linestyle="--", linewidth=1)
        axis.grid(axis="x", alpha=.2)
    figure.suptitle("Fixed final checkpoint · 3 paired seeds · 192 development windows / 24 recordings")
    figure.text(.52, .015, "95% recording-cluster intervals after seed averaging; exploratory, unadjusted comparisons",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .96))
    figure.savefig(output / "planning_comparison.png", dpi=180)
    figure.savefig(output / "planning_comparison.svg")
    plt.close(figure)


if __name__ == "__main__":
    main()
