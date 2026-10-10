"""Compare saved planning checkpoints while the original training queue continues.

Reuse the registered first-epoch model loading, inputs and official scorer.
Inference uses a separate, small allocation on GPU 0 and never trains a model.
"""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from compare_lpwm_front_history_epoch1 import (
    PROJECT_ROOT, context, create_model, prepare, read_json, score, sha256,
    verify_registration, write_json,
)


def prepare_comparison(arguments):
    prepare(arguments)
    configuration, output = context(arguments.config)
    registration = read_json(output / 'registration.json')
    registration['sources']['scripts/compare_planning_epoch_checkpoints.py'] = sha256(Path(__file__))
    registration['saved_baseline_checkpoint_sha256']['lpwm'] = sha256(
        PROJECT_ROOT / configuration['conditions']['lpwm']['checkpoint'])
    registration['lpwm_first_epoch_checkpoint_pending'] = False
    registration['planning_epoch'] = configuration['planning_epoch']
    write_json(output / 'registration.json', registration)


def infer(arguments):
    import torch
    from train_lpwm_drivor_joint import card_used_bytes

    configuration, output = context(arguments.config)
    verify_registration(arguments.config)
    folder = output / arguments.condition
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'prediction_complete.json').exists(), 'Preserve completed predictions'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    used_before = card_used_bytes(0)
    allowance = min(configuration['maximum_process_bytes'],
                    configuration['maximum_total_card_bytes'] - used_before - 1024**3)
    assert allowance >= 2_500_000_000, 'Insufficient budget for this evaluation; leave training intact'
    torch.cuda.set_per_process_memory_fraction(allowance / torch.cuda.get_device_properties(0).total_memory)
    model, metadata = create_model(configuration, arguments.condition)
    model.to('cuda')
    inputs = PROJECT_ROOT / configuration['inputs']
    images_name = 'historical_adapter_images.npy' if arguments.condition == 'lpwm' else 'rectangular_images.npy'
    ego_name = 'historical_joint_ego.npy' if arguments.condition == 'lpwm' else 'rectangular_ego.npy'
    images = np.load(inputs / images_name, mmap_mode='r')
    ego = np.load(inputs / ego_name, mmap_mode='r')
    targets = np.load(inputs / 'trajectory.npy', mmap_mode='r')
    records = read_json(PROJECT_ROOT / configuration['panel'])['records']
    trajectories = []
    started = time.time()
    largest_card_use = used_before
    # Match each saved model's original evaluation precision and batch size.
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if arguments.condition == 'lpwm' else contextlib.nullcontext()
    with torch.inference_mode(), autocast:
        for offset in range(0, len(records), configuration['inference_batch_size']):
            stop = offset + configuration['inference_batch_size']
            card_use = card_used_bytes(0)
            largest_card_use = max(largest_card_use, card_use)
            assert card_use < configuration['maximum_total_card_bytes'] - 512 * 1024**2, 'Evaluation yields to memory pressure'
            features = {
                'image': torch.from_numpy(np.array(images[offset:stop])).permute(0, 1, 4, 2, 3).to('cuda').float() / 255,
                'ego_status': torch.from_numpy(np.array(ego[offset:stop])).to('cuda').float()[:, None],
            }
            trajectories.append(model(features)['trajectory'].float().cpu().numpy())
    predictions = np.concatenate(trajectories)
    assert predictions.shape == (1024, 8, 3) and np.isfinite(predictions).all()
    distances = np.linalg.norm(predictions[..., :2] - targets[..., :2], axis=-1)
    pending = folder / 'predictions.pending.npz'
    np.savez(pending, tokens=np.array([row['token'] for row in records]), trajectories=predictions, targets=targets)
    pending.replace(folder / 'predictions.npz')
    write_json(folder / 'prediction_complete.json', metadata | {
        'complete': True, 'count': 1024, 'ade_meters': float(distances.mean()),
        'fde_meters': float(distances[:, -1].mean()), 'seconds': time.time() - started,
        'images': images_name, 'ego': ego_name, 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'largest_sampled_total_card_bytes': largest_card_use, 'allocator_limit_bytes': allowance,
        'precision_matches_original_evaluation': True, 'training_unchanged': True,
    })
    print(json.dumps({'condition': arguments.condition, 'prediction_complete': True}), flush=True)


def report(arguments):
    configuration, output = context(arguments.config)
    registration = verify_registration(arguments.config)
    results = {condition: read_json(output / condition / 'evaluation_complete.json') for condition in configuration['conditions']}
    assert all(result['count'] == 1024 and result['failed'] == 0 for result in results.values())
    assert all(result['completed_updates'] == configuration['conditions'][condition]['completed_updates']
               for condition, result in results.items())
    first_epoch = read_json(PROJECT_ROOT / configuration['first_epoch_comparison'])
    assert first_epoch['count'] == 1024
    summary = {
        'complete': True, 'planning_epoch': configuration['planning_epoch'], 'results': results,
        'count': 1024, 'full_navtest': False, 'conditions': configuration['conditions'],
        'overlap': registration['overlap'], 'interpretation': configuration['interpretation'],
        'selection_policy': configuration['selection_policy'],
        'lpwm_minus_baseline_pdms': {condition: results['lpwm']['pdms'] - results[condition]['pdms']
                                     for condition in ('drivor', 'jepa')},
        'first_epoch_pdms_same_panel': {condition: result['pdms'] for condition, result in first_epoch['results'].items()},
        'pdms_change_since_first_epoch': {condition: result['pdms'] - first_epoch['results'][condition]['pdms']
                                          for condition, result in results.items()},
        'historical_development_scores_not_same_panel': registration['historical_development_scores'],
    }
    write_json(output / 'comparison_complete.json', summary)
    lines = [f"# Planning epoch {configuration['planning_epoch']}: shared NAVTEST subset", '',
             'Same fixed 1,024 scenes; official NAVSIM v1 scorer; not full NAVTEST.', '',
             '| Model | Epoch 1 PDMS | Epoch 3 PDMS | ADE (m) | FDE (m) | Planning scenes / updates |',
             '|---|---:|---:|---:|---:|---|']
    for condition, result in results.items():
        specification = configuration['conditions'][condition]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} / {} |'.format(
            condition, first_epoch['results'][condition]['pdms'], result['pdms'],
            result['ade_meters'], result['fde_meters'], specification['planning_training_scenes'], result['completed_updates']))
    lines.extend(['', configuration['interpretation'], '', configuration['selection_policy'], ''])
    (output / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'complete': True, 'pdms': {condition: result['pdms'] for condition, result in results.items()}}), flush=True)


def run_comparison(arguments):
    configuration, output = context(arguments.config)
    verify_registration(arguments.config)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}
    gpu_python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    oracle_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    for condition in configuration['conditions']:
        for mode, completion in (('infer', 'prediction_complete.json'), ('score', 'evaluation_complete.json')):
            if (output / condition / completion).is_file():
                continue
            execution_environment = environment if mode == 'infer' else environment | {
                'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'LD_LIBRARY_PATH': str(oracle_root / 'lib'),
                'PYTHONPATH': str(PROJECT_ROOT / 'reference_repositories/DrivoR'),
            }
            python = gpu_python if mode == 'infer' else oracle_root / 'bin/python'
            command = [str(python), '-u', str(Path(__file__)), '--config', str(arguments.config),
                       '--mode', mode, '--condition', condition]
            with (output / f'{mode}_{condition}.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                write_json(output / 'controller_state.json', {'status': f'{mode}_{condition}',
                           'pid': os.getpid(), 'active_pid': process.pid, 'command': command, 'time': time.time()})
                assert process.wait() == 0, f'{mode} {condition} failed; original training remains intact'
    report(arguments)
    write_json(output / 'controller_state.json', {'status': 'complete', 'pid': os.getpid(), 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'infer', 'score', 'report', 'run'), required=True)
    parser.add_argument('--condition', choices=('lpwm', 'drivor', 'jepa'))
    arguments = parser.parse_args()
    try:
        {'prepare': prepare_comparison, 'infer': infer, 'score': score,
         'report': report, 'run': run_comparison}[arguments.mode](arguments)
    except Exception as error:
        configuration, output = context(arguments.config)
        write_json(output / 'failed.json', {'mode': arguments.mode, 'condition': arguments.condition,
                   'error': repr(error), 'time': time.time()})
        raise
