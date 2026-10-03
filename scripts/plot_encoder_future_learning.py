"""Export paired planning comparisons with recording-bootstrap uncertainty."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    workspace = Path(__file__).resolve().parents[1]
    share_directory = workspace / "results/encoder_future_learning_v1"
    summary = json.loads((share_directory / "summary.json").read_text())
    conditions = list(summary["specification"]["conditions"])
    labels = {
        "encoder_planning": "Planning only / 2 blocks",
        "intent_planning": "Intent + planning / 2 blocks",
        "intent_uniform_future": "Uniform future / 2 blocks",
        "intent_motion_future": "Motion targets / 2 blocks",
        "intent_planning_future": "Planning targets / 2 blocks",
        "intent_uniform_masked_future": "Uniform future + mask / 2 blocks",
        "intent_planning_masked_future": "Planning future + mask / 2 blocks",
        "intent_planning_masked_current": "Current reconstruction + mask / 2 blocks",
        "encoder_planning_last6": "Planning only / 6 blocks",
        "intent_planning_last6": "Intent + planning / 6 blocks",
        "intent_uniform_future_last6": "Uniform future / 6 blocks",
        "intent_planning_masked_future_last6": "Planning future + mask / 6 blocks",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 7.5), sharey=True)
    for position, condition in enumerate(conditions):
        key = condition + "_minus_original_frozen"
        ade = summary["paired_ade_comparisons"][key]
        pdm = summary["paired_pdm_comparisons"][key]
        color = "#c56522" if condition.endswith("last6") else "#2476a8"
        for axis, mean, interval in (
            (axes[0], ade["mean_difference_m"] * 1000,
             np.asarray(ade["recording_cluster_bootstrap_95_ci_m"]) * 1000),
            (axes[1], pdm["mean_difference"] * 100,
             np.asarray(pdm["recording_cluster_bootstrap_95_ci"]) * 100),
        ):
            axis.hlines(position, interval[0], interval[1], color=color, linewidth=2)
            axis.scatter(mean, position, color=color, s=35, zorder=3)
    axes[0].set_yticks(range(len(conditions)), [labels[condition] for condition in conditions])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Change in scene-macro ADE (mm)\nNegative = lower error")
    axes[1].set_xlabel("Change in official development PDM (percentage points)\nPositive = higher score")
    for axis in axes:
        axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
        axis.grid(axis="x", color="#eeeeee")
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Planning-facing encoder learning: all registered final checkpoints", fontsize=14)
    figure.text(.02, .02, "192 development windows / 24 recordings; 3 paired seeds; fixed update 512.\n"
                "Bars: 95% recording-bootstrap intervals of seed-averaged differences; no multiple-comparison correction.\n"
                "Development evidence only; foundation-training exposure is not excluded.", fontsize=9)
    figure.tight_layout(rect=(0, .10, 1, .95))
    figure.savefig(share_directory / "paired_planning_comparison.svg")
    figure.savefig(share_directory / "paired_planning_comparison.pdf")
    figure.savefig(workspace / "outputs/encoder_future_learning_v1/paired_planning_comparison.png", dpi=150)
    print("ENCODER_COMPARISON_FIGURE_COMPLETE")


if __name__ == "__main__":
    main()
