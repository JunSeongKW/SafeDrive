"""Publish quantitative particle overlays from saved validation arrays, using CPU only."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plot
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def draw_particles(axis, positions, scales, color, dashed=False):
    centers = (positions[:, ::-1] + 1) * np.array([256, 128]) - .5
    sizes = scales[:, ::-1] * np.array([512, 256])
    axis.scatter(centers[:, 0], centers[:, 1], s=14,
                 facecolors="none" if dashed else color, edgecolors=color, linewidths=.8)
    for center, size in zip(centers, sizes):
        axis.add_patch(Rectangle(center - size / 2, *size, fill=False, edgecolor=color,
                                 linestyle="--" if dashed else "-", linewidth=.65, alpha=.8))
    return centers


def publish(root, destination):
    destination.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for folder in sorted(root.iterdir()):
        before_path = folder / "before_training_visuals.npz"
        if not folder.is_dir() or not before_path.exists():
            continue
        before = np.load(before_path)
        for after_path in sorted(folder.glob("*_visuals.npz")):
            if after_path == before_path:
                continue
            label = after_path.stem.removesuffix("_visuals")
            output = destination / (folder.name + "_" + label + "_before_after_overlay.png")
            if output.exists():
                continue
            after = np.load(after_path)
            figure, axes = plot.subplots(3, 4, figsize=(18, 8))
            for row in range(3):
                frame = before[f"scene{row}_input"].transpose(1, 2, 0).clip(0, 1)
                assert np.array_equal(before[f"scene{row}_input"], after[f"scene{row}_input"])
                for column, title in enumerate(("Input", "Before", label, "Overlay: cyan before / orange after")):
                    axes[row, column].imshow(frame)
                    axes[row, column].set_title(title)
                    axes[row, column].axis("off")
                initial_positions, initial_scales = before[f"scene{row}_positions"], before[f"scene{row}_scales"]
                current_positions, current_scales = after[f"scene{row}_positions"], after[f"scene{row}_scales"]
                draw_particles(axes[row, 1], initial_positions, initial_scales, "cyan", True)
                draw_particles(axes[row, 2], current_positions, current_scales, "orange")
                initial_centers = draw_particles(axes[row, 3], initial_positions, initial_scales, "cyan", True)
                current_centers = draw_particles(axes[row, 3], current_positions, current_scales, "orange")
                for start, end in zip(initial_centers, current_centers):
                    axes[row, 3].annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "white", "lw": .6})
                for axis in axes[row]:
                    axis.set_xlim(-.5, 511.5)
                    axis.set_ylim(255.5, -.5)
            figure.suptitle(folder.name + ": same particle IDs; actual displacement; LPWM glimpses, not object boxes")
            figure.tight_layout()
            figure.savefig(output, dpi=140)
            plot.close(figure)
        complete = folder / "complete.json"
        if complete.exists():
            summaries[folder.name] = json.loads(complete.read_text())
    (destination / "completed_conditions.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/lpwm_driving_video_512x256_v1/particle_budget_study")
    parser.add_argument("--output", type=Path, default=ROOT / "results/lpwm_driving_video_512x256_v1/particle_budget_study")
    args = parser.parse_args()
    while True:
        publish(args.root, args.output)
        state = json.loads((args.root / "queue_state.json").read_text())
        if not args.watch or state["status"] != "training" or (args.root / "queue_failed.json").exists():
            break
        time.sleep(30)
