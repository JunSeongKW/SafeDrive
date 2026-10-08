"""Render saved LPWM RGB validation arrays on CPU, without model inference."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb_image(channels_first):
    assert channels_first.shape == (3, 256, 512)
    assert np.isfinite(channels_first).all()
    pixels = np.rint(channels_first.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels)


def render_comparison(destination, title, subtitle, columns, scenes, footer):
    """Paste native 512x256 outputs without sharpening, smoothing, or resizing."""
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font = ImageFont.truetype(font_path, 25)
    label_font = ImageFont.truetype(font_path, 19)
    note_font = ImageFont.truetype(font_path, 16)
    margin, gap, panel_width, panel_height = 20, 12, 512, 256
    row_height, header_height = 312, 116
    width = margin * 2 + len(columns) * panel_width + (len(columns) - 1) * gap
    height = header_height + len(scenes) * row_height + 70
    canvas = Image.new('RGB', (width, height), '#f3f5f7')
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 14), title, fill='#17212e', font=title_font)
    draw.text((margin, 52), subtitle, fill='#344254', font=note_font)
    for column_index, (heading, _) in enumerate(columns):
        draw.text((margin + column_index * (panel_width + gap), 84), heading,
                  fill='#17212e', font=label_font)
    for scene_index, scene in enumerate(scenes):
        top = header_height + scene_index * row_height
        draw.text((margin, top), scene['caption'], fill='#344254', font=note_font)
        for column_index, (_, key) in enumerate(columns):
            canvas.paste(rgb_image(scene[key]),
                         (margin + column_index * (panel_width + gap), top + 27))
    draw.multiline_text((margin, height - 65), footer, fill='#344254', font=note_font, spacing=7)
    canvas.save(destination)


def main(arguments):
    source = arguments.run_directory.resolve()
    destination = arguments.output_directory.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    initial_report = json.loads((source / 'before_training.json').read_text())
    latest_report = json.loads((source / f'{arguments.label}.json').read_text())
    initial_path = source / 'before_training_visuals.npz'
    latest_path = source / f'{arguments.label}_visuals.npz'
    scenes, frame_metrics = [], []
    with np.load(initial_path, allow_pickle=False) as initial, np.load(latest_path, allow_pickle=False) as latest:
        for scene_index in range(3):
            before_record = initial_report['records'][scene_index]
            after_record = latest_report['records'][scene_index]
            for key in ('recording', 'log', 'start'):
                assert before_record[key] == after_record[key]
            for key in ('input', 'truth'):
                assert np.array_equal(initial[f'scene{scene_index}_{key}'], latest[f'scene{scene_index}_{key}'])
            scene = dict(caption=f"Scene {scene_index + 1} | {after_record['recording']} | clip start {after_record['start']}")
            for key in ('input', 'truth', 'reconstruction', 'forecast'):
                scene[key] = latest[f'scene{scene_index}_{key}'].copy()
            scene['initial_reconstruction'] = initial[f'scene{scene_index}_reconstruction'].copy()
            scene['initial_forecast'] = initial[f'scene{scene_index}_forecast'].copy()
            scenes.append(scene)
            frame_metrics.append(dict(scene=scene_index + 1, recording=after_record['recording'],
                latest_observed_reconstruction_mse=float(np.mean((scene['input'] - scene['reconstruction']) ** 2)),
                latest_plus3s_forecast_mse=float(np.mean((scene['truth'] - scene['forecast']) ** 2)),
                plus3s_persistence_mse=float(np.mean((scene['truth'] - scene['input']) ** 2)),
                initial_plus3s_forecast_mse=float(np.mean((scene['truth'] - scene['initial_forecast']) ** 2))))
    latest_title = arguments.display_label
    footer = ('First 3 saved validation clips, fixed before training; not selected by reconstruction quality.\n'
              'Native 512x256 RGB panels. No enhancement. These examples do not establish object preservation or planning benefit.')
    render_comparison(destination / 'reconstruction_before_vs_epoch4.png',
        'LPWM | Observed-frame reconstruction: before vs after training',
        'Reconstruction uses encoded video; this panel is not a future prediction test.',
        [('Observed image (t = 0)', 'input'), ('Before training | reconstructed', 'initial_reconstruction'),
         (latest_title + ' | reconstructed', 'reconstruction')], scenes, footer)
    render_comparison(destination / 'future_prediction_before_vs_epoch4.png',
        'LPWM | Future prediction at +3 seconds: before vs after training',
        'Forecast sees only t = -0.5s and t = 0. Future ground truth is used for comparison only.',
        [('Actual future image (+3s)', 'truth'), ('Before training | predicted +3s', 'initial_forecast'),
         (latest_title + ' | predicted +3s', 'forecast')], scenes, footer)
    render_comparison(destination / 'observed_reconstruction_future_overview_epoch4.png',
        'LPWM | Observed image, reconstruction, and 3-second future prediction',
        latest_title + ' | 16 foreground + 1 background particles | same three held-out validation clips',
        [('Observed image (t = 0)', 'input'), ('Reconstruction (t = 0)', 'reconstruction'),
         ('Actual future image (+3s)', 'truth'), ('Predicted future image (+3s)', 'forecast')], scenes,
        'Forecast sees two observed frames only; no future ground-truth frames enter the forecast.\n' + footer.split('\n')[0])
    metadata = dict(created_at=datetime.now().astimezone().isoformat(), validation_label=arguments.label,
        source_sha256={str(path): file_digest(path) for path in (
            initial_path, latest_path, source / 'before_training.json', source / f'{arguments.label}.json')},
        selection='First three saved validation clips; same fixed scenes at both checkpoints.',
        display='Native 512x256 panels, values clipped to [0,1] and quantized to uint8; no enhancement.',
        forecast_observed_times_seconds=[-0.5, 0.0], displayed_forecast_time_seconds=3.0,
        no_model_inference_or_training_changes=True, frame_metrics=frame_metrics,
        all32_validation_means_before=initial_report['mean'], all32_validation_means_latest=latest_report['mean'])
    (destination / 'visualization_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(dict(output_directory=str(destination), frame_metrics=frame_metrics), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--label', default='update2620')
    parser.add_argument('--display-label', default='Epoch 4 (2,620 updates)')
    parser.add_argument('--output-directory', type=Path, required=True)
    main(parser.parse_args())
