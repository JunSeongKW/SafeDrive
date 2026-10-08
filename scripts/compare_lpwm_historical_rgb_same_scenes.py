"""Compare preserved LPWM checkpoints on the exact saved RGB validation scenes.

CPU-only reconstruction diagnostic. Different resolutions and training budgets
are displayed explicitly; this is not a controlled architecture ablation.
"""
from contextlib import nullcontext, redirect_stdout
from datetime import datetime
import gc
import io
import json
from pathlib import Path
import pickle
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning_aware_future_prediction.object_centric.lpwm_bridge import load_official_lpwm, checkpoint_digest
from planning_aware_future_prediction.object_centric.lpwm_adapter_full_finetuning import build_adapter_or_full_planning_model
from train_lpwm_reduced_particle_stage1 import prepare_records, load_clip
from visualize_small_corpus_lpwm_rgb import render_comparison, rgb_image

RUN = ROOT / 'outputs/four_model_small_corpus_v1/lpwm_ssl'
DESTINATION = ROOT / 'results/four_model_small_corpus_v1/historical_rgb_same_scenes'
STAGE1 = ROOT / 'outputs/lpwm_navsim_full_posttraining_v2/stage1/checkpoint.pt'
ADAPTER = ROOT / 'outputs/lpwm_card_budget_measured_v4/residual_adapter/batch8/metric_plus_world/epoch01.pt'
SCREENING = ROOT / 'outputs/lpwm_driving_video_512x256_v1/particle_budget_study/candidate_particles64'


def enlarge_for_comparison(channels_first):
    """Reverse square display distortion using nearest pixels, without new detail."""
    return np.asarray(rgb_image_square(channels_first).resize((512, 256), Image.Resampling.NEAREST)).transpose(2, 0, 1) / 255.


def rgb_image_square(channels_first):
    assert channels_first.shape == (3, 128, 128)
    return Image.fromarray(np.rint(channels_first.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8))


def load_scene_metadata(record):
    with (ROOT / 'dataset/navsim_logs/trainval' / (record['log'] + '.pkl')).open('rb') as stream:
        frames = pickle.load(stream)
    image_name = Path(record['paths'][1]).name
    matches = [frame for frame in frames if Path(frame['cams']['CAM_F0']['data_path']).name == image_name]
    assert len(matches) == 1
    current = matches[0]
    return np.r_[current['driving_command'], current['ego_dynamic_state']].astype(np.float32), current['token']


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    torch.manual_seed(47)
    registration = json.loads((RUN / 'registration.json').read_text())
    _, validation = prepare_records(Path(registration['manifest']), 32)
    records = validation[:3]
    current_report = json.loads((RUN / 'update2620.json').read_text())
    screening_report = json.loads((SCREENING / 'after_training.json').read_text())
    old_manifest = json.loads((ROOT / 'outputs/lpwm_navsim_full_posttraining_v2/manifest.json').read_text())
    old_training_recordings = {record['recording_group'] for record in old_manifest['records'] if record['split'] == 'train'}
    old_training_indices = {index for record in old_manifest['records'] if record['split'] == 'train' for index in record['frame_cache_indices']}
    old_training_images = {old_manifest['unique_image_paths'][index] for index in old_training_indices}
    scene_rows, square_clips, ego_statuses, provenance = [], [], [], []
    arrays = {}
    with np.load(RUN / 'update2620_visuals.npz') as current, np.load(SCREENING / 'after_training_visuals.npz') as screening:
        for scene_index, record in enumerate(records):
            for other in (current_report['records'][scene_index], screening_report['records'][scene_index]):
                assert all(record[key] == other[key] for key in ('recording', 'log', 'start'))
            rectangular_clip = load_clip(record['paths']).numpy()
            observed = current[f'scene{scene_index}_input']
            assert np.array_equal(rectangular_clip[1], np.rint(observed * 255).astype(np.uint8))
            assert np.array_equal(observed, screening[f'scene{scene_index}_input'])
            square_images = []
            for path in record['paths']:
                with Image.open(path) as opened:
                    square_images.append(cv2.resize(np.asarray(opened.convert('RGB'))[28:-28], (128, 128), interpolation=cv2.INTER_AREA))
            square_clip = torch.from_numpy(np.stack(square_images)).permute(0, 3, 1, 2).contiguous().float().div(255)[None]
            square_clips.append(square_clip)
            ego, token = load_scene_metadata(record)
            ego_statuses.append(torch.from_numpy(ego)[None])
            image_relative = str(Path(record['paths'][1]).relative_to(ROOT / 'dataset/sensor_blobs/trainval'))
            previously_seen = image_relative in old_training_images
            exposure_label = 'OLD MODEL TRAINING IMAGE' if previously_seen else 'HELD OUT FROM BOTH STAGE-1 RUNS'
            scene_rows.append(dict(caption=f"Scene {scene_index + 1} | {record['recording']} | {exposure_label}", input=observed.copy(),
                current=current[f'scene{scene_index}_reconstruction'].copy(),
                screening=screening[f'scene{scene_index}_reconstruction'].copy(),
                square_input=enlarge_for_comparison(square_clip[0, 1].numpy())))
            arrays[f'scene{scene_index}_historical_input'] = square_clip[0].numpy()
            provenance.append(dict(recording=record['recording'], log=record['log'], start=record['start'],
                raw_frame_paths=record['paths'], current_token=token, current_ego=ego.tolist(),
                historical_stage1_training_recording_overlap=record['recording'] in old_training_recordings,
                historical_stage1_training_image_overlap=previously_seen,
                input_uint8_exact_match_with_saved_current_validation=True))
    del old_manifest
    metrics = {}
    decoder_checks = {}
    for condition in ('stage1', 'adapter'):
        started = time.monotonic()
        with redirect_stdout(io.StringIO()):
            if condition == 'stage1':
                world_model, _ = load_official_lpwm('cpu', STAGE1)
                wrapper = None
            else:
                configuration = json.loads((ROOT / 'configs/lpwm_planning/card_budget_measured_v4/residual_adapter_batch8.json').read_text())
                wrapper = build_adapter_or_full_planning_model(STAGE1, configuration, 'metric_plus_world', ROOT)
                saved = torch.load(ADAPTER, map_location='cpu', weights_only=True, mmap=True)
                wrapper.load_state_dict(saved, strict=True)
                del saved
                wrapper.eval().requires_grad_(False)
                world_model = wrapper.world_model
        world_model.eval().requires_grad_(False)
        world_model.timestep_horizon = 7
        metrics[condition] = []
        with torch.inference_mode():
            for scene_index, clip in enumerate(square_clips):
                command_context = wrapper.encoder_command(ego_statuses[scene_index]) if wrapper is not None else nullcontext()
                with command_context:
                    encoded = world_model.encode_all(clip, deterministic=True)
                    attributes = [encoded[key] for key in ('z', 'z_scale', 'z_features', 'obj_on', 'z_depth', 'z_bg_features')]
                    filter_key = encoded['z_base_var'].sum(-1) if (
                        world_model.filter_particles_in_decoder and world_model.n_kp_enc != world_model.n_kp_dec) else None
                    decoded = world_model.decode_all(*attributes, z_ctx=encoded['z_context'], filter_key=filter_key)['rec_rgb'].reshape_as(clip)
                    if scene_index == 0:
                        # Verify that the saved rectangular evaluator's context slicing
                        # cannot change RGB for this public decoder configuration.
                        sliced = world_model.decode_all(*attributes, z_ctx=encoded['z_context'][:, 1:].contiguous(), filter_key=filter_key)['rec_rgb'].reshape_as(clip)
                        canonical = world_model(clip, deterministic=True)['rec_rgb'].reshape_as(clip)
                        decoder_checks[condition] = dict(normalize_rgb=world_model.normalize_rgb,
                            encoded_foreground_particles=world_model.n_kp_enc, decoded_foreground_particles=world_model.n_kp_dec,
                            manual_vs_forward_max_abs=float((decoded - canonical).abs().max()),
                            full_vs_sliced_context_max_abs=float((decoded - sliced).abs().max()))
                        print(json.dumps(dict(condition=condition,decoder_check=decoder_checks[condition])),flush=True)
                        assert decoder_checks[condition]['manual_vs_forward_max_abs'] < 1e-6
                        assert decoder_checks[condition]['full_vs_sliced_context_max_abs'] < 1e-6
                    current_frame = decoded[0, 1].numpy().copy()
                    scene_rows[scene_index][condition] = enlarge_for_comparison(current_frame)
                    arrays[f'scene{scene_index}_{condition}_reconstruction_native'] = current_frame
                    metrics[condition].append(dict(scene=scene_index + 1, reconstruction_mse_native128=float((decoded[0, 1] - clip[0, 1]).square().mean())))
                    assert np.isfinite(current_frame).all()
                print(json.dumps(dict(condition=condition, scene=scene_index + 1, seconds=time.monotonic() - started)), flush=True)
        del world_model, wrapper, encoded, decoded, sliced, canonical
        gc.collect()
    np.savez_compressed(DESTINATION / 'historical_native_outputs.npz', **arrays)
    footer = ('Same raw frames and top/bottom crop. Historical 128x128 outputs are expanded by nearest pixels to match the field of view.\n'
              'Scenes 1-2 were training images for the old Stage 1. Scene 3 was held out from both. This is not a controlled ablation.')
    render_comparison(DESTINATION / 'same_scenes_current_vs_historical_reconstruction.png',
        'LPWM reconstruction | Current and historical experiments on the same scenes',
        'Historical models: 64 encoded foreground particles, 30 selected for RGB decoding. Current: 16 encoded and decoded.',
        [('Observed image (512x256)', 'input'), ('Current: 512x256 / 16 FG / 4 ep', 'current'),
         ('Old Stage 1: 128x128 / 64 FG / 20 ep', 'stage1'), ('Old Stage 1 + Adapter: +1 planning ep', 'adapter')], scene_rows, footer)
    render_comparison(DESTINATION / 'historical_input_vs_reconstruction.png',
        'Historical LPWM | Input detail versus reconstructed detail',
        'All columns originate at 128x128; nearest-pixel enlargement. Historical decoder selects 30 of 64 foreground particles.',
        [('Actual historical 128x128 input', 'square_input'), ('Stage 1 after 20 epochs', 'stage1'),
         ('Stage 1 + Adapter after 1 epoch', 'adapter')], scene_rows, footer)
    render_comparison(DESTINATION / 'same_scenes_rectangular16_vs_rectangular64.png',
        'LPWM 512x256 | Current 16-particle model versus earlier 64-particle screening',
        'Equal RGB input, different training budgets: current 2,620 updates / batch16; old screening 200 updates / batch4.',
        [('Observed image', 'input'), ('Current: 16 FG / epoch 4', 'current'),
         ('Earlier screening: 64 FG / 200 updates', 'screening')], scene_rows,
        'This does not isolate particle count: the data exposure and training duration differ.\nAll image panels are native 512x256. No sharpening or other enhancement.')
    report = dict(created_at=datetime.now().astimezone().isoformat(), scenes=provenance,
        conditions=dict(current=dict(resolution=[512,256],foreground_particles=16,updates=2620,epochs=4),
            stage1=dict(resolution=[128,128],foreground_particles=64,decoded_foreground_particles=30,updates=28920,epochs=20),
            adapter=dict(resolution=[128,128],foreground_particles=64,decoded_foreground_particles=30,stage1_epochs=20,stage2_epochs=1,stage2_updates=4707),
            screening=dict(resolution=[512,256],foreground_particles=64,updates=200)),
        checkpoint_sha256={str(path):checkpoint_digest(path) for path in (STAGE1,ADAPTER)},
        source_array_sha256={str(path):checkpoint_digest(path) for path in (RUN/'update2620_visuals.npz',SCREENING/'after_training_visuals.npz')},
        native128_metrics=metrics,decoder_api_checks=decoder_checks,
        scope='RGB reconstruction on three fixed scenes; not planning evaluation, not a controlled ablation.',
        device='cpu',threads=2,no_training_updates=True,normalization='Original [0,1] RGB; normalize_rgb=false.',
        display='Native historical arrays saved; nearest-neighbor expansion only for side-by-side viewing.')
    (DESTINATION / 'comparison_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(complete=True,output=str(DESTINATION),decoder_api_checks=decoder_checks)),flush=True)


if __name__ == '__main__':
    main()
