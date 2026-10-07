"""Render recorded SSL and planning curves without inventing Stage1 PDMS."""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


project_root = Path.cwd()
output_directory = Path(__file__).resolve().parent
stage1_validation_path = project_root / "outputs/lpwm_navsim_full_posttraining_v2/stage1/validation_log.jsonl"
stage1_summary_path = project_root / "outputs/lpwm_navsim_full_posttraining_v2/stage1/training_summary.json"
joint_report_path = project_root / "results/lpwm_drivor_planning_path_lora_v1/intermediate_update4000_20261007/report.json"
stage1_effect_path = project_root / "results/lpwm_stage1_effect_v1/queue/stage1_effect_summary.json"

stage1_rows_by_epoch = {}
duplicate_initial_rows = 0
for line in stage1_validation_path.read_text().splitlines():
    if not line.strip():
        continue
    validation_row = json.loads(line)
    epoch_number = validation_row["epoch"]
    if epoch_number in stage1_rows_by_epoch:
        assert stage1_rows_by_epoch[epoch_number] == validation_row
        duplicate_initial_rows += 1
    stage1_rows_by_epoch[epoch_number] = validation_row
stage1_validation_rows = [stage1_rows_by_epoch[epoch_number] for epoch_number in sorted(stage1_rows_by_epoch)]
assert [row["epoch"] for row in stage1_validation_rows] == list(range(21))
assert not any("pdm" in key.lower() for row in stage1_validation_rows for key in row)
stage1_summary = json.loads(stage1_summary_path.read_text())
assert stage1_summary["official_elbo"] is True and stage1_summary["planning_loss"] is False
joint_report = json.loads(joint_report_path.read_text())
updates_per_joint_epoch = joint_report["progress"]["updates_per_epoch"]
joint_pdms_rows = [
    {
        "optimizer_updates": row["completed_updates"],
        "epoch_equivalent": row["completed_updates"] / updates_per_joint_epoch,
        "pdms": row["pdms"],
        "scene_count": row["scene_count"],
        "by_scene_type": row["by_scene_type"],
    }
    for row in joint_report["pdms_rows"]
]

with (output_directory / "stage1_epoch_validation.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["epoch", "optimizer_update", "validation_clips", "ssl_elbo_loss", "logged_psnr_db", "pdms"])
    writer.writeheader()
    for row in stage1_validation_rows:
        writer.writerow({
            "epoch": row["epoch"],
            "optimizer_update": row["optimizer_update"],
            "validation_clips": row["validation_clips"],
            "ssl_elbo_loss": row["loss"],
            "logged_psnr_db": row["psnr"],
            "pdms": "",
        })
with (output_directory / "joint_pdms_by_update.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["optimizer_updates", "epoch_equivalent", "pdms", "scene_count"])
    writer.writeheader()
    for row in joint_pdms_rows:
        writer.writerow({key: row[key] for key in writer.fieldnames})

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
figure, axes = plt.subplots(1, 3, figsize=(17, 5.7))
stage1_epochs = [row["epoch"] for row in stage1_validation_rows]
axes[0].plot(stage1_epochs, [row["loss"] for row in stage1_validation_rows], "o-", color="#186f96", markersize=4)
axes[0].set(title="Stage1: SSL validation loss", xlabel="Stage1 epoch", ylabel="Logged temporal ELBO loss (lower is better)", xlim=(-.5, 20.5))
axes[0].text(.98, .93, "No planner attached\nNo PDMS recorded", transform=axes[0].transAxes, ha="right", va="top", fontsize=12, bbox={"boxstyle":"round", "facecolor":"#eef5f8", "edgecolor":"#bed3df"})
axes[1].plot(stage1_epochs, [row["psnr"] for row in stage1_validation_rows], "o-", color="#278768", markersize=4)
axes[1].set(title="Stage1: logged video PSNR", xlabel="Stage1 epoch", ylabel="PSNR in official SSL forward (dB)", xlim=(-.5, 20.5))
axes[1].annotate("Epoch 1: 20.46 dB", xy=(1, stage1_validation_rows[1]["psnr"]), xytext=(4, 18.3), arrowprops={"arrowstyle":"->", "color":"#54616b"})
axes[1].annotate("Epoch 20: 21.08 dB", xy=(20, stage1_validation_rows[-1]["psnr"]), xytext=(10, 19.7), arrowprops={"arrowstyle":"->", "color":"#54616b"})
axes[2].plot([row["epoch_equivalent"] for row in joint_pdms_rows], [row["pdms"] for row in joint_pdms_rows], "o-", color="#d5811c", markersize=4)
axes[2].set(title="Current joint model: planning PDMS", xlabel="Current joint-training epoch equivalent", ylabel="Official PDMS on 95 training scenes", xlim=(-.08, 2.8), ylim=(25, 85))
for completed_epoch in [1, 2]:
    recorded_row = next(row for row in joint_pdms_rows if row["epoch_equivalent"] == completed_epoch)
    axes[2].annotate(f"Epoch {completed_epoch}: {recorded_row['pdms']:.2f}", xy=(completed_epoch, recorded_row["pdms"]), xytext=(completed_epoch-.65, recorded_row["pdms"]+5.0), fontsize=10, arrowprops={"arrowstyle":"->", "color":"#54616b"})
axes[2].annotate("4,000 updates: 75.83", xy=(joint_pdms_rows[-1]["epoch_equivalent"], joint_pdms_rows[-1]["pdms"]), xytext=(1.1, 53), fontsize=10, arrowprops={"arrowstyle":"->", "color":"#54616b"})
for axis in axes:
    axis.grid(alpha=.2)
figure.suptitle("Recorded learning curves: earlier Stage1 SSL and current joint planning", fontsize=17)
figure.text(.5, .035, "Stage1: 512 development clips. Current: 95 training scenes, not navtest.\nDifferent metrics, inputs and training budgets; these curves do not establish method superiority.", ha="center", fontsize=11)
figure.tight_layout(rect=(0, .12, 1, .92))
figure.savefig(output_directory / "stage1_ssl_and_current_joint_pdms.png", dpi=150)
figure.savefig(output_directory / "stage1_ssl_and_current_joint_pdms.pdf")
plt.close(figure)

stage1_effect = json.loads(stage1_effect_path.read_text())
report = {
    "checked_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
    "stage1_epoch_pdms_records_exist": False,
    "stage1_uses_official_ssl_without_planner": True,
    "stage1_validation_clips": 512,
    "stage1_validation_rows": stage1_validation_rows,
    "identical_initial_validation_duplicates_deduplicated_for_export_only": duplicate_initial_rows,
    "stage1_epoch_checkpoint_files_present": [path.name for path in sorted((project_root / "outputs/lpwm_navsim_full_posttraining_v2/stage1").glob("epoch*.pt"))],
    "joint_recorded_pdms_rows": joint_pdms_rows,
    "earlier_stage1_endpoint_planning_comparison": stage1_effect["planning_means"],
    "earlier_endpoint_interpretation": "Public-init or Stage1 epoch20 LPWM was frozen, then an identical planner was separately trained for one epoch. These are downstream planning results, not Stage1 training-time PDMS measurements.",
    "source_sha256": {str(path.relative_to(project_root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [stage1_validation_path, stage1_summary_path, joint_report_path, stage1_effect_path]},
    "limits": ["SSL loss/PSNR cannot determine planning PDMS or planning saturation.", "The earlier and current PDMS evaluation panels/planners/training budgets differ.", "Swapping earlier Stage1 checkpoints into a planner trained on epoch20 features would confound representation quality with distribution mismatch."],
    "compute_graph_audit": {
        "stage1_effective_clip_batch": 16,
        "stage1_camera_count": 1,
        "stage1_camera_sequences_per_optimizer_update": 16,
        "stage1_actual_video_frames_per_clip": 12,
        "stage1_native_trainable_parameters": 109545263,
        "joint_effective_scene_batch": 64,
        "joint_camera_count": 4,
        "joint_camera_sequences_per_optimizer_update": 256,
        "camera_sequence_count_ratio_is_not_a_flops_ratio": True,
        "joint_current_frame_count_per_camera": 1,
        "joint_recursive_future_steps_per_camera": 8,
        "joint_native_lpwm_weights_frozen": True,
        "joint_lpwm_lora_trainable_parameters": 4683650,
        "joint_activation_checkpoint_recomputation_during_backward": True,
        "both_use_preprocessed_rgb_pixel_npy_caches": True,
        "matched_factorwise_profiling_done": False,
    },
    "new_training_or_gpu_work_started": False,
}
(output_directory / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
print(json.dumps({"output_directory":str(output_directory), "stage1_epoch_pdms_records_exist":False,"stage1_validation_epochs":len(stage1_validation_rows),"joint_pdms_points":len(joint_pdms_rows)}, ensure_ascii=False))
