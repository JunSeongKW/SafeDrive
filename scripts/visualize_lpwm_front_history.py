"""Fixed development scenes: particle before/after/overlay and RGB diagnostics."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

from lpwm_front_history_data import FrontHistorySceneDataset
from train_lpwm_drivor_joint import device_inputs, rng_state, restore_rng, write_json


def draw_particles(rgb, attributes, selected, color):
    drawing = ImageDraw.Draw(rgb)
    side = rgb.width
    for particle in attributes:
        vertical, horizontal = (particle[:2] + 1) * (side / 2) - .5
        radius = 1 + 3 * float(np.clip(particle[4], 0, 1))
        drawing.ellipse((horizontal-radius, vertical-radius, horizontal+radius, vertical+radius),
                        outline=color, width=1)
    for index in selected:
        particle = attributes[index]
        vertical, horizontal = (particle[:2] + 1) * (side / 2) - .5
        half_height, half_width = particle[2:4] * side / 2
        drawing.rectangle((horizontal-half_width, vertical-half_height,
                           horizontal+half_width, vertical+half_height), outline=color, width=1)


def snapshot(model, records, device, output, label, benchmark):
    previous_rng = rng_state()
    was_training = model.training
    previous_recording = model.particle_encoder.record_particles
    model.eval()
    model.particle_encoder.record_particles = True
    path = output / "particles" / label
    path.mkdir(parents=True, exist_ok=True)
    examples = next(iter(DataLoader(FrontHistorySceneDataset(records[:4]), batch_size=4)))
    features, _ = device_inputs(examples, 0, 4, device, benchmark)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model(features)
        attributes = model.particle_encoder.latest_particle_attributes.numpy()
        np.save(path / "attributes.npy", attributes)
        before_path = output / "particles/before/attributes.npy"
        before = np.load(before_path) if before_path.exists() else attributes
        assert before.shape == attributes.shape == (4, 1, 9, 64, 14)
        columns = ("Current front input", "Before Stage 2", label, "Overlay: cyan before / orange after")
        canvas = Image.new("RGB", (4*256, 4*290+95), "white")
        drawing = ImageDraw.Draw(canvas)
        for column, caption in enumerate(columns):
            drawing.text((column*256+5, 5), caption, fill="black")
        for scene_index in range(4):
            image = Image.fromarray(examples["images"][scene_index, -1].numpy()).resize((256, 256))
            initial = before[scene_index, 0, 0]
            current = attributes[scene_index, 0, 0]
            selected = np.argsort(-initial[:, 4], kind="stable")[:16]
            panels = [image.copy() for _ in columns]
            draw_particles(panels[1], initial, selected, "#00c5e5")
            draw_particles(panels[2], current, selected, "#efa800")
            draw_particles(panels[3], initial, selected, "#00c5e5")
            draw_particles(panels[3], current, selected, "#efa800")
            for column, panel in enumerate(panels):
                canvas.paste(panel, (column*256, scene_index*290+35))
            drawing.text((5, scene_index*290+293), examples["token"][scene_index], fill="black")
        shifts = np.linalg.norm((attributes[:,:,0,:,:2]-before[:,:,0,:,:2])*64, axis=-1)
        size_change = np.abs(attributes[:,:,0,:,2:4]/np.maximum(before[:,:,0,:,2:4], 1e-6)-1)
        metrics = {"label": label, "tokens": examples["token"],
                   "mean_center_shift_input_pixels": float(shifts.mean()),
                   "mean_relative_size_change_percent": float(size_change.mean()*100),
                   "mean_absolute_presence_change": float(np.abs(attributes[:,:,0,:,4]-before[:,:,0,:,4]).mean()),
                   "scene_selection": "fixed first four of registered development panel",
                   "dots": "all 64 foreground particle centers; radius indicates presence",
                   "boxes": "LPWM glimpse extents, fixed initial top16 presence indices; not detected boxes"}
        drawing.text((5, 4*290+35), f"Mean center shift {shifts.mean():.4f} input px; size change {size_change.mean()*100:.4f}%", fill="black")
        drawing.text((5, 4*290+57), "Dots: all64; radius: presence. Boxes: fixed initial top16 LPWM glimpses, not object detections.", fill="black")
        canvas.save(path / "particles_before_after_overlay.png")
        write_json(path / "diagnostics.json", metrics)
    finally:
        model.particle_encoder.record_particles = previous_recording
        model.train(was_training)
        restore_rng(previous_rng)
