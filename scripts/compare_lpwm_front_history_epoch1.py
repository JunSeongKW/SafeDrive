"""Evaluate preserved first-epoch models on one previously fixed NAVTEST subset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def context(configuration_path):
    configuration = read_json(configuration_path)
    return configuration, PROJECT_ROOT / configuration['output_directory']


def verify_registration(configuration_path):
    configuration, output = context(configuration_path)
    registration = read_json(output / 'registration.json')
    for filename, expected in registration['sources'].items():
        assert sha256(PROJECT_ROOT / filename) == expected, filename
    for condition, expected in registration['saved_baseline_checkpoint_sha256'].items():
        assert sha256(PROJECT_ROOT / configuration['conditions'][condition]['checkpoint']) == expected
    return registration


def prepare(arguments):
    configuration, output = context(arguments.config)
    assert not (output / 'registration.json').exists(), 'Do not replace a registered comparison'
    records = read_json(PROJECT_ROOT / configuration['panel'])['records']
    assert len(records) == configuration['scene_count'] == len({row['token'] for row in records}) == 1024
    panel_tokens = {row['token'] for row in records}
    panel_recordings = {row['recording_group'] for row in records}
    training_configuration = read_json(PROJECT_ROOT / configuration['training_configuration'])
    training = [row for row in read_json(PROJECT_ROOT / training_configuration['manifest'])['records'] if row['split'] == 'train']
    stage1 = [row for row in read_json(PROJECT_ROOT / 'outputs/lpwm_navsim_full_posttraining_v2/manifest.json')['records'] if row['split'] == 'train']
    small_manifest = read_json(PROJECT_ROOT / 'outputs/four_model_small_corpus_v1/corpus/manifest.json')
    small_training = [row for row in small_manifest['records'] if row['study_split'] == 'train']
    ssl_recordings = {json.loads(line)['recording'] for line in Path(small_manifest['ssl_manifest']).read_text().splitlines()}
    overlap = {
        'lpwm_planning_tokens': len(panel_tokens & {row['token'] for row in training}),
        'lpwm_planning_recordings': len(panel_recordings & {row['recording_group'] for row in training}),
        'lpwm_stage1_recordings': len(panel_recordings & {row['recording_group'] for row in stage1}),
        'baseline_planning_tokens': len(panel_tokens & {row['token'] for row in small_training}),
        'baseline_planning_recordings': len(panel_recordings & {row['recording_group'] for row in small_training}),
        'jepa_stage1_recordings': len(panel_recordings & ssl_recordings),
    }
    assert not any(overlap.values()), overlap
    assert all(Path(row['metric_cache_file']).is_file() for row in records)
    inputs = PROJECT_ROOT / configuration['inputs']
    shapes = {'historical_adapter_images.npy': (1024, 4, 128, 128, 3),
              'historical_joint_ego.npy': (1024, 11), 'rectangular_images.npy': (1024, 2, 256, 512, 3),
              'rectangular_ego.npy': (1024, 11), 'trajectory.npy': (1024, 8, 3)}
    input_hashes = {}
    for filename, shape in shapes.items():
        assert np.load(inputs / filename, mmap_mode='r').shape == shape
        input_hashes[str((inputs / filename).relative_to(PROJECT_ROOT))] = sha256(inputs / filename)
    original = read_json(PROJECT_ROOT / training_configuration['registration'])
    sources = dict(original['sources'])
    for filename in (str(arguments.config.relative_to(PROJECT_ROOT)),
                     'scripts/compare_lpwm_front_history_epoch1.py',
                     'scripts/queue_lpwm_front_history_epoch1_comparison.py',
                     'scripts/launch_lpwm_front_history_epoch1_comparison.py',
                     configuration['panel'], small_manifest['ssl_manifest']):
        path = Path(filename)
        if path.is_absolute():
            path = path.relative_to(PROJECT_ROOT)
        sources[str(path)] = sha256(PROJECT_ROOT / path)
    sources.update(input_hashes)
    assert all(sha256(PROJECT_ROOT / filename) == expected for filename, expected in sources.items())
    historical = {condition: read_json(PROJECT_ROOT / specification['historical_development_scores'])
                  for condition, specification in configuration['conditions'].items()
                  if 'historical_development_scores' in specification}
    baseline_hashes = {condition: sha256(PROJECT_ROOT / specification['checkpoint'])
                       for condition, specification in configuration['conditions'].items() if condition != 'lpwm'}
    sources[configuration['conditions']['jepa']['stage1_checkpoint']] = sha256(PROJECT_ROOT / configuration['conditions']['jepa']['stage1_checkpoint'])
    write_json(output / 'registration.json', {'sources': sources, 'saved_baseline_checkpoint_sha256': baseline_hashes,
        'original_training_registration_sha256': sha256(PROJECT_ROOT / training_configuration['registration']),
        'prepared_at_unix': time.time(), 'count': 1024, 'recording_count': len(panel_recordings),
        'overlap': overlap, 'historical_development_scores': historical,
        'lpwm_first_epoch_checkpoint_pending': True, 'interpretation': configuration['interpretation']})
    print(json.dumps({'prepared': True, 'overlap': overlap, 'output': str(output)}), flush=True)


def create_model(configuration, condition):
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    specification = configuration['conditions'][condition]
    checkpoint = PROJECT_ROOT / specification['checkpoint']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['completed_updates'] == specification['completed_updates']
    if condition == 'lpwm':
        from planning_aware_future_prediction.object_centric.lpwm_front_history_stage1_lora import FrontHistoryStage1LoRAPlanner
        training_configuration = read_json(PROJECT_ROOT / configuration['training_configuration'])
        model = FrontHistoryStage1LoRAPlanner(PROJECT_ROOT / training_configuration['stage1_checkpoint'])
    else:
        from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
        stage1 = PROJECT_ROOT / specification['stage1_checkpoint'] if condition == 'jepa' else None
        model = CommonPlannerModel(condition, stage1)
        for filename, expected in saved['registration']['source_sha256'].items():
            assert sha256(PROJECT_ROOT / filename) == expected, filename
    model.load_state_dict(saved['model'], strict=True)
    if condition == 'lpwm':
        assert model.frozen_native_digest() == saved['frozen_native_sha256']
    model.eval().requires_grad_(False)
    return model, {'checkpoint_sha256': sha256(checkpoint), 'completed_updates': saved['completed_updates'],
                   'strict_state_load': True, 'frozen_native_sha256': saved.get('frozen_native_sha256')}


def audit_baselines(arguments):
    configuration, output = context(arguments.config)
    verify_registration(arguments.config)
    import torch
    torch.set_num_threads(2)
    details = {}
    for condition in ('drivor', 'jepa'):
        model, metadata = create_model(configuration, condition)
        details[condition] = metadata
        del model
    write_json(output / 'baseline_cpu_load_audit.json', {'passed': True, 'models': details, 'gpu_used': False})
    print(json.dumps({'passed': True, 'models': details}), flush=True)


def infer(arguments):
    import contextlib
    import torch
    from train_lpwm_drivor_joint import configure_allocator, card_used_bytes
    configuration, output = context(arguments.config)
    verify_registration(arguments.config)
    folder = output / arguments.condition
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder / 'prediction_complete.json').exists(), 'Preserve completed predictions'
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    configure_allocator(0)
    allowance = min(configuration['maximum_process_bytes'],
                    configuration['maximum_total_card_bytes'] - card_used_bytes(0) - 1024**3)
    assert allowance > 8 * 1024**3, 'Insufficient whole-card admission budget'
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
    # Preserve each trainer's original evaluation precision. Baseline backbones
    # autocast internally while their downstream planner evaluates in FP32.
    autocast = torch.autocast('cuda', dtype=torch.bfloat16) if arguments.condition == 'lpwm' else contextlib.nullcontext()
    with torch.inference_mode(), autocast:
        for offset in range(0, len(records), configuration['inference_batch_size']):
            stop = offset + configuration['inference_batch_size']
            features = {'image': torch.from_numpy(np.array(images[offset:stop])).permute(0, 1, 4, 2, 3).to('cuda').float() / 255,
                        'ego_status': torch.from_numpy(np.array(ego[offset:stop])).to('cuda').float()[:, None]}
            trajectories.append(model(features)['trajectory'].float().cpu().numpy())
    predictions = np.concatenate(trajectories)
    assert predictions.shape == (1024, 8, 3) and np.isfinite(predictions).all()
    distances = np.linalg.norm(predictions[..., :2] - targets[..., :2], axis=-1)
    tokens = np.array([record['token'] for record in records])
    pending = folder / 'predictions.pending.npz'
    np.savez(pending, tokens=tokens, trajectories=predictions, targets=targets)
    pending.replace(folder / 'predictions.npz')
    write_json(folder / 'prediction_complete.json', metadata | {'complete': True, 'count': 1024,
        'ade_meters': float(distances.mean()), 'fde_meters': float(distances[:, -1].mean()),
        'seconds': time.time() - started, 'images': images_name, 'ego': ego_name,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'precision_matches_original_evaluation': True})
    print(json.dumps({'condition': arguments.condition, 'prediction_complete': True}), flush=True)


def score(arguments):
    from score_lpwm_drivor_navtest import score_scene
    configuration, output = context(arguments.config)
    folder = output / arguments.condition
    records = read_json(PROJECT_ROOT / configuration['panel'])['records']
    predictions = np.load(folder / 'predictions.npz')
    assert predictions['tokens'].tolist() == [row['token'] for row in records]
    jobs = [(row['token'], trajectory, row['metric_cache_file']) for row, trajectory in zip(records, predictions['trajectories'])]
    with ProcessPoolExecutor(max_workers=configuration['scoring_workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = list(pool.map(score_scene, jobs, chunksize=8))
    assert len(rows) == 1024 and all(row['valid'] for row in rows)
    with (folder / 'scores.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = {key: float(np.mean([row[key] for row in rows])) for key in rows[0] if key not in ('token', 'log_name', 'valid')}
    by_scene = {}
    for scene_type in sorted({record['scene_type'] for record in records}):
        values = [row['score'] for row, record in zip(rows, records) if record['scene_type'] == scene_type]
        by_scene[scene_type] = {'count': len(values), 'pdms': float(np.mean(values)) * 100}
    write_json(folder / 'evaluation_complete.json', read_json(folder / 'prediction_complete.json') | {
        'pdms': metrics['score'] * 100, 'mean_metrics': metrics, 'by_scene_type': by_scene, 'failed': 0,
        'full_navtest': False, 'benchmark': 'NAVSIM v1 fixed independent navtest subset'})
    print(json.dumps({'condition': arguments.condition, 'pdms': metrics['score'] * 100}), flush=True)


def report(arguments):
    configuration, output = context(arguments.config)
    registration = verify_registration(arguments.config)
    results = {condition: read_json(output / condition / 'evaluation_complete.json') for condition in configuration['conditions']}
    for condition, result in results.items():
        assert result['count'] == 1024 and result['failed'] == 0
        assert result['completed_updates'] == configuration['conditions'][condition]['completed_updates']
    summary = {'complete': True, 'results': results, 'count': 1024, 'full_navtest': False,
        'historical_development_scores_not_same_panel': registration['historical_development_scores'],
        'conditions': configuration['conditions'], 'overlap': registration['overlap'],
        'lpwm_minus_baseline_pdms': {condition: results['lpwm']['pdms'] - results[condition]['pdms'] for condition in ('drivor', 'jepa')},
        'interpretation': configuration['interpretation'], 'selection_policy': configuration['selection_policy']}
    write_json(output / 'comparison_complete.json', summary)
    lines = ['# First planning epoch: shared NAVTEST subset', '',
             'Same 1,024 scenes and official NAVSIM v1 scorer; this is not full NAVTEST.', '',
             '| Model | PDMS | ADE (m) | FDE (m) | Planning scenes / updates | Input |',
             '|---|---:|---:|---:|---|---|']
    for condition, result in results.items():
        specification = configuration['conditions'][condition]
        lines.append('| {} | {:.4f} | {:.4f} | {:.4f} | {} / {} | front1, {} frames, {}×{} |'.format(
            condition, result['pdms'], result['ade_meters'], result['fde_meters'],
            specification['planning_training_scenes'], specification['completed_updates'], specification['observed_frames'],
            *specification['image_size_width_height']))
    lines.extend(['', configuration['interpretation'], '',
                  'Historical development scores (different panel): DrivoR 66.9414; JEPA 66.9890.', '',
                  configuration['selection_policy'], ''])
    (output / 'comparison.md').write_text('\n'.join(lines))
    print(json.dumps({'complete': True, 'pdms': {key: value['pdms'] for key, value in results.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'audit-baselines', 'infer', 'score', 'report'), required=True)
    parser.add_argument('--condition', choices=('lpwm', 'drivor', 'jepa'))
    arguments = parser.parse_args()
    if arguments.mode in ('infer', 'score'):
        assert arguments.condition
    {'prepare': prepare, 'audit-baselines': audit_baselines, 'infer': infer, 'score': score, 'report': report}[arguments.mode](arguments)
