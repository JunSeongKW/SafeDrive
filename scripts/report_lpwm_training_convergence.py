"""Summarize saved Stage 1 and LoRA curves without running model inference."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "results/lpwm_card_budget_measured_v4/convergence_review_20261005"


def read_log(relative_path):
    path = PROJECT_ROOT / relative_path
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    sources = {
        "stage1_validation": "outputs/lpwm_navsim_full_posttraining_v2/stage1/validation_log.jsonl",
        "stage1_training": "outputs/lpwm_navsim_full_posttraining_v2/stage1/training_log.jsonl",
        "lora_validation": "outputs/lpwm_card_budget_measured_v4/attention_lora/batch8/metric_plus_world/validation_log.jsonl",
        "lora_training": "outputs/lpwm_card_budget_measured_v4/attention_lora/batch8/metric_plus_world/training_log.jsonl",
    }
    logs = {name: read_log(path) for name, path in sources.items()}
    stage1_by_epoch = {row["epoch"]: row for row in logs["stage1_validation"]}
    lora_by_update = {row["update"]: row for row in logs["lora_validation"]}
    stage1_rows = [stage1_by_epoch[epoch] for epoch in sorted(stage1_by_epoch) if epoch > 0]
    lora_rows = [lora_by_update[update] for update in sorted(lora_by_update) if update > 0]
    assert len(stage1_rows) == 20 and stage1_rows[-1]["optimizer_update"] == 28920
    assert lora_rows[-1]["update"] == 4707
    summary = {
        "checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "scope": "Saved development monitors; no new inference or training; not official navtest.",
        "research_question": "Is the current optimization budget sufficient to assess planning-aware LPWM adaptation?",
        "sources": {name: {"path": path, "sha256": hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest()}
                    for name, path in sources.items()},
        "stage1_validation": stage1_rows,
        "stage1_loss_change_15_to_20_percent": 100 * (stage1_by_epoch[20]["loss"] / stage1_by_epoch[15]["loss"] - 1),
        "stage1_loss_change_10_to_20_percent": 100 * (stage1_by_epoch[20]["loss"] / stage1_by_epoch[10]["loss"] - 1),
        "stage1_sampled_training_loss_by_epoch": [
            {"epoch": epoch, "mean_logged_loss": statistics.mean(
                row["loss"] for row in logs["stage1_training"] if epoch - 1 < row["epoch_fraction"] <= epoch)}
            for epoch in range(1, 21)],
        "lora_validation": lora_rows,
        "lora_sampled_training_loss_by_update_window": [
            {"first_update": first, "last_update": last, "mean_logged_loss": statistics.mean(
                row["loss"] for row in logs["lora_training"] if first <= row["update"] <= last)}
            for first, last in [(1, 1024), (1025, 2048), (2049, 3072), (3073, 4096), (4097, 4707)]],
        "lora_final_learning_rates": {key: logs["lora_training"][-1][key] for key in ("lpwm_lr", "planner_lr")},
        "limitations": [
            "Stage1 monitor is the official temporal SSL objective, not a per-epoch past-only forecast evaluation.",
            "Stage1 improvements become small; twenty epochs do not establish optimal planning representations.",
            "LoRA monitor uses 128 scenes, 127 with valid cached official PDM scores; final 1024-scene scores are a different panel.",
            "LoRA late training losses still fall while development planning metrics fluctuate; neither convergence nor additional PDMS gains are established.",
            "Training loss averages summarize sparsely logged minibatches, not exact epoch averages.",
            "One training seed; no confidence interval for extrapolation beyond the observed epoch.",
            "LoRA learning rates decay to one tenth over one epoch; an extension requires a declared continuation schedule.",
        ],
        "runtime_changed": False,
    }
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    (REPORT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    axes[0].plot([row["epoch"] for row in stage1_rows], [row["loss"] for row in stage1_rows], "o-", color="#3569a8")
    axes[0].set(title="Stage 1: validation SSL objective", xlabel="Completed epoch", ylabel="Loss (lower is better)", ylim=(19, 24), xticks=[1, 5, 10, 15, 20])
    axes[0].text(.96, .91, "Epoch 15 to 20: -0.96%\n512 fixed development clips", ha="right", va="top", transform=axes[0].transAxes, fontsize=10)
    epoch_fractions = [row["epoch"] for row in lora_rows]
    axes[1].plot(epoch_fractions, [row["cached_official_pdms_percent"] for row in lora_rows], "o-", color="#248c76")
    axes[1].set(title="Stage 2 / LoRA: planning score", xlabel="Completed epoch fraction", ylabel="PDMS (higher is better)", ylim=(74, 87))
    axes[2].plot(epoch_fractions, [row["ade_meters"] for row in lora_rows], "o-", color="#ca7938")
    axes[2].set(title="Stage 2 / LoRA: trajectory error", xlabel="Completed epoch fraction", ylabel="ADE in meters (lower is better)", ylim=(1.0, 1.65))
    for axis in axes:
        axis.grid(alpha=.2)
    figure.suptitle("Saved learning curves: slowing Stage 1 improvement, uncertain Stage 2 convergence", fontsize=14)
    figure.text(.5, .025, "Stage 2: same 128-scene monitor (127 valid PDMS). Epoch 0 omitted. Zoomed vertical axes. Not navtest.", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, .06, 1, .92))
    figure.savefig(REPORT_ROOT / "learning_curves.png", dpi=160)
    figure.savefig(REPORT_ROOT / "learning_curves.pdf")
    plt.close(figure)
    print(json.dumps({"report": str(REPORT_ROOT), "stage1_15_to_20_percent": summary["stage1_loss_change_15_to_20_percent"],
                      "lora_final_learning_rates": summary["lora_final_learning_rates"]}, indent=2))


if __name__ == "__main__":
    main()
