"""Read-only candidate, interface and objective diagnostics for saved small runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import lzma
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = PROJECT_ROOT / 'outputs/four_model_small_corpus_v1'
DIAGNOSTIC_ROOT = PROJECT_ROOT / 'outputs/small_corpus_planning_diagnosis_20261010'
CONDITIONS = ('drivor', 'jepa', 'lpwm_sequential', 'lpwm_joint')
METRICS = ('no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress',
           'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance', 'score')


def write_json(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    pending.replace(path)


def file_digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def prepare_panel():
    manifest_path = STUDY_ROOT / 'corpus/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    ego = np.load(STUDY_ROOT / 'corpus/ego.npy', mmap_mode='r')
    records = [row for row in manifest['records'] if row['study_split'] == 'dev']
    selected = []
    for command in range(3):
        eligible = [row for row in records if int(ego[row['cache_row'], 7:11].argmax()) == command]
        random.Random(20261010 + command).shuffle(eligible)
        selected.extend(eligible[:64])
    assert len(selected) == 192 and len({row['token'] for row in selected}) == 192
    registration = {'records': selected, 'selection': '64 scenes per command; seeded before candidate scores',
                    'scene_count': 192, 'full_development_set': False, 'full_navtest': False,
                    'manifest_sha256': file_digest(manifest_path),
                    'checkpoints': {name: file_digest(STUDY_ROOT / name / 'pass3.pt') for name in CONDITIONS}}
    path = DIAGNOSTIC_ROOT / 'panel.json'
    if path.exists():
        assert json.loads(path.read_text()) == registration
    else:
        write_json(path, registration)
    return registration


def load_model(condition):
    import torch
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
    stage1 = STUDY_ROOT / ('jepa_ssl' if condition == 'jepa' else 'lpwm_ssl') / 'latest.pt'
    model = CommonPlannerModel(condition, stage1 if condition in ('jepa', 'lpwm_sequential') else None)
    saved = torch.load(STUDY_ROOT / condition / 'pass3.pt', map_location='cpu', mmap=True, weights_only=False)
    assert saved['completed_updates'] == 1920
    for filename, expected in saved['registration']['source_sha256'].items():
        assert file_digest(PROJECT_ROOT / filename) == expected, filename
    model.load_state_dict(saved['model'], strict=True)
    return model


def scene_inputs(record, device):
    import torch
    root = Path(record['cache_directory'])
    index = record['cache_row']
    arrays = {name: np.load(root / (name + '.npy'), mmap_mode='r') for name in
              ('images', 'ego', 'trajectory', 'trajectory_long')}
    features = {'image': torch.from_numpy(np.array(arrays['images'][index:index + 1])).permute(0, 1, 4, 2, 3).to(device).float() / 255,
                'ego_status': torch.from_numpy(np.array(arrays['ego'][index:index + 1])).to(device).float()[:, None]}
    if device == 'cpu':
        # Public LPWM uses view after its background CNN. CPU channels-last
        # convolutions require this storage normalization; values are identical.
        features['image'] = features['image'].contiguous()
    targets = {name: torch.from_numpy(np.array(arrays[name][index:index + 1])).to(device) for name in
               ('trajectory', 'trajectory_long')}
    return features, targets


def infer(condition):
    import torch
    torch.set_num_threads(2)
    panel = prepare_panel()
    torch.cuda.set_per_process_memory_fraction(4_000_000_000 / torch.cuda.get_device_properties(0).total_memory)
    model = load_model(condition).eval().requires_grad_(False).cuda()
    proposals, rankings, interface = [], [], []
    maximum_card = 0
    recorded = np.load(STUDY_ROOT / condition / 'validation/pass3.npz')
    recorded_by_token = dict(zip(recorded['tokens'].tolist(), recorded['trajectories']))
    maximum_replay_difference = 0.
    with torch.no_grad():
        for scene_index, record in enumerate(panel['records']):
            card_bytes = int(subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=memory.used',
                             '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2
            maximum_card = max(maximum_card, card_bytes)
            if card_bytes > 47_000_000_000:
                raise RuntimeError('Diagnostic yields before the shared 48GB card limit')
            features, _ = scene_inputs(record, 'cuda')
            if condition.startswith('lpwm'):
                model.backbone.record_particles = True
            output = model(features)
            proposals.append(output['proposals'][0].float().cpu().numpy())
            rankings.append(output['pdm_score'][0].float().cpu().numpy())
            selected = output['trajectory'][0].float().cpu().numpy()
            maximum_replay_difference = max(maximum_replay_difference,
                                           float(np.abs(selected - recorded_by_token[record['token']]).max()))
            if condition.startswith('lpwm') and scene_index < 3:
                attributes = model.backbone.latest_attributes
                assert attributes.shape[1:3] == (9, 16)
                model.backbone.command = features['ego_status'][:, -1, 7:11]
                memory = model.backbone(features['image'], None)
                model.backbone.command = None
                memory_squared_mean = memory.float().square().mean()
                attribute_sensitivity, memory_sensitivity = {}, {}
                for name, bounds in {'position': (0, 2), 'scale': (2, 4), 'presence': (4, 5),
                                     'foreground_appearance': (6, 10), 'background_appearance': (10, 14),
                                     'local_context': (14, 14 + model.backbone.context_dimension),
                                     'background_context': (14 + model.backbone.context_dimension, attributes.shape[-1])}.items():
                    perturbed = attributes.to('cuda').clone()
                    perturbed[..., bounds[0]:bounds[1]] = 0
                    projected = model.backbone.projection(perturbed.transpose(1, 2).flatten(2))
                    memory_sensitivity[name] = float((projected - memory).float().square().mean().sqrt())
                    attribute_sensitivity[name] = float(attributes[..., bounds[0]:bounds[1]].square().mean().sqrt())
                repeated = attributes.to('cuda').clone()
                repeated[:, 1:] = repeated[:, :1]
                projected = model.backbone.projection(repeated.transpose(1, 2).flatten(2))
                memory_sensitivity['future_repeat_current'] = float((projected - memory).float().square().mean().sqrt())
                interface.append({'token': record['token'], 'attributes_shape': list(attributes.shape),
                                  'planner_memory_shape': list(memory.shape),
                                  'memory_rms': float(memory_squared_mean.sqrt()),
                                  'attribute_rms': attribute_sensitivity, 'memory_change_rms': memory_sensitivity})
            if (scene_index + 1) % 32 == 0:
                print(json.dumps({'condition': condition, 'inferred': scene_index + 1}), flush=True)
    folder = DIAGNOSTIC_ROOT / condition
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'candidates.npz', proposals=np.stack(proposals), rankings=np.stack(rankings),
                        tokens=np.asarray([row['token'] for row in panel['records']]))
    write_json(folder / 'inference_complete.json', {'complete': True, 'checkpoint_updates': 1920,
               'scene_count': len(proposals), 'candidate_count': len(proposals[0]),
               'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'largest_sampled_card_bytes': maximum_card,
               'maximum_selected_trajectory_replay_difference': maximum_replay_difference, 'interface': interface})


def score_candidates(job):
    """Batch simulation, then reproduce reference-plus-single progress normalization."""
    sys.path.insert(0, str(PROJECT_ROOT / 'reference_repositories/DrivoR'))
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory, pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import WeightedMetricIndex
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    record, proposals, ranking = job
    with lzma.open(record['metric_cache_file'], 'rb') as stream:
        cache = pickle.load(stream)
    sampling = TrajectorySampling(num_poses=40, interval_length=.1)
    initial = cache.ego_state
    reference = get_trajectory_as_array(cache.trajectory, sampling, initial.time_point)
    states = np.stack([reference] + [get_trajectory_as_array(transform_trajectory(Trajectory(poses), initial),
                      sampling, initial.time_point) for poses in proposals])
    simulator, scorer = PDMSimulator(sampling), PDMScorer(sampling)
    simulated = simulator.simulate_proposals(states, initial)
    scorer.score_proposals(simulated, cache.observation, cache.centerline,
                           cache.route_lane_ids, cache.drivable_area_map)
    multiplicative = scorer._multi_metrics.prod(axis=0)
    raw_progress = scorer._progress_raw * multiplicative
    pair_maximum = np.maximum(raw_progress[0], raw_progress[1:])
    normalized = np.where(pair_maximum > scorer._config.progress_distance_threshold,
                          raw_progress[1:] / np.maximum(pair_maximum, 1e-12), (multiplicative[1:] != 0).astype(float))
    weighted = scorer._weighted_metrics[:, 1:].copy()
    weighted[WeightedMetricIndex.PROGRESS] = normalized
    final_scores = multiplicative[1:] * (weighted * scorer._config.weighted_metrics_array[:, None]).sum(0) / scorer._config.weighted_metrics_array.sum()
    selected_index, best_index = int(ranking.argmax()), int(final_scores.argmax())
    # Validate batching against the unchanged public evaluator for both extrema.
    checked = []
    for index in sorted({selected_index, best_index}):
        official = asdict(pdm_score(cache, Trajectory(proposals[index]), sampling,
                                   PDMSimulator(sampling), PDMScorer(sampling)))
        difference = abs(official['score'] - final_scores[index])
        assert difference < 1e-8, (record['token'], index, difference)
        checked.append(difference)
    return {'token': record['token'], 'recording_group': record['recording_group'],
            'selected_index': selected_index, 'oracle_best_index': best_index,
            'selected_pdms': float(final_scores[selected_index] * 100),
            'oracle_best_pdms': float(final_scores[best_index] * 100),
            'selection_regret': float((final_scores[best_index] - final_scores[selected_index]) * 100),
            'candidate_scores': final_scores.tolist(), 'pair_evaluator_maximum_error': max(checked)}


def score(condition):
    panel = prepare_panel()
    folder = DIAGNOSTIC_ROOT / condition
    candidates = np.load(folder / 'candidates.npz')
    assert candidates['tokens'].tolist() == [row['token'] for row in panel['records']]
    jobs = list(zip(panel['records'], candidates['proposals'], candidates['rankings']))
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        rows = []
        for index, row in enumerate(pool.map(score_candidates, jobs, chunksize=2)):
            rows.append(row)
            if (index + 1) % 32 == 0:
                print(json.dumps({'condition': condition, 'scored': index + 1}), flush=True)
    selected, best = np.array([row['selected_pdms'] for row in rows]), np.array([row['oracle_best_pdms'] for row in rows])
    historical = {row['token']: row for row in __import__('csv').DictReader(
                  (STUDY_ROOT / condition / 'validation/pass3.scores.csv').open())}
    replay_error = max(abs(row['selected_pdms'] - float(historical[row['token']]['score']) * 100) for row in rows)
    result = {'complete': True, 'count': len(rows), 'selected_pdms': float(selected.mean()),
              'oracle_best_pdms': float(best.mean()), 'selection_regret': float((best - selected).mean()),
              'oracle_best_below_80_fraction': float((best < 80).mean()),
              'oracle_perfect_but_selected_below_80_fraction': float(((best > 99.99) & (selected < 80)).mean()),
              'maximum_historical_selected_pdms_difference': replay_error,
              'score_definition': 'Official NAVSIM v1, each candidate paired with reference; not the training proxy',
              'by_command': {str(command): {'count': 64, 'selected_pdms': float(selected[command*64:(command+1)*64].mean()),
                             'oracle_best_pdms': float(best[command*64:(command+1)*64].mean())} for command in range(3)},
              'rows': rows}
    write_json(folder / 'candidate_analysis.json', result)


def gradients():
    """Measure separate, unclipped objective gradients without updating any weights."""
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))
    from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import make_official_loss, oracle_loss_callback
    from train_small_corpus_common_planner import binary_safety_targets
    from navsim.agents.drivoR.layers.losses import drivor_loss as official_loss_module
    from train_lpwm_reduced_particle_stage1 import prepare_records, DrivingClips, ARTIFACT_ROOT
    from train_lpwm_local_stage1_distributed import LOSS_WEIGHTS
    from lpwm_drivor_oracle import DrivoROracleClient
    official_loss_module.three_to_two_classes = binary_safety_targets
    torch.manual_seed(20261010)
    model = load_model('lpwm_joint').train()
    world = model.backbone.world_model
    encoder_parameters = list(world.encoder_module.parameters())
    parameter_names = {id(parameter): name for name, parameter in world.encoder_module.named_parameters()}
    criterion, configuration = make_official_loss('navsim_v1')
    manifest = json.loads((STUDY_ROOT / 'corpus/manifest.json').read_text())
    train = [row for row in manifest['records'] if row['study_split'] == 'train']
    random.Random(4802).shuffle(train)
    # Fixed train samples and SSL clips, never selected by gradient outcome.
    ssl_records, _ = prepare_records(Path(manifest['ssl_manifest']), 32)
    random.Random(4702).shuffle(ssl_records)
    video_dataset = DrivingClips(ssl_records)
    os.environ['TORCH_HOME'] = str(ARTIFACT_ROOT / 'torch')
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction = LossLPIPS(normalized_rgb=False).eval().requires_grad_(False)
    oracle = DrivoROracleClient(STUDY_ROOT / 'corpus/manifest.json', DIAGNOSTIC_ROOT / 'gradients', 0, 2)
    samples = []
    try:
        for sample_index in range(2):
            started = time.time()
            features, targets = scene_inputs(train[sample_index], 'cpu')
            prediction = model(features)
            labels = torch.from_numpy(oracle.score([train[sample_index]['token']], prediction['proposals'].detach().float().numpy()))
            losses = criterion(targets, prediction, configuration, scoring_function=oracle_loss_callback(labels))
            planning = losses['loss']
            planning_gradients = torch.autograd.grad(planning, encoder_parameters, allow_unused=True)
            planning_values = [None if value is None else value.detach().clone() for value in planning_gradients]
            del planning_gradients, labels, losses, prediction, planning
            video = video_dataset[sample_index][None].float() / 255
            result = world(video, deterministic=False, with_loss=True, warmup=False, num_static=1,
                           recon_loss_func=reconstruction, recon_loss_type='vgg', **LOSS_WEIGHTS)
            ssl = result['loss_dict']['loss']
            ssl_gradients = torch.autograd.grad(ssl, encoder_parameters, allow_unused=True)
            groups = {}
            for group in ('encoder', 'geometry_heads', 'appearance', 'interaction'):
                first_squared = second_squared = dot_product = 0.
                parameter_count = 0
                for parameter, first, second in zip(encoder_parameters, planning_values, ssl_gradients):
                    name = parameter_names[id(parameter)]
                    matches = (group == 'encoder' or group == 'geometry_heads' and any(tag in name for tag in ('xy_head', 'scale_xy_head', 'obj_on_head'))
                               or group == 'appearance' and any(tag in name for tag in ('particle_features_enc', 'bg_'))
                               or group == 'interaction' and 'particle_inter_enc' in name)
                    if not matches:
                        continue
                    parameter_count += parameter.numel()
                    if first is not None:
                        first_squared += float(first.float().square().sum())
                    if second is not None:
                        second_squared += float(second.float().square().sum())
                    if first is not None and second is not None:
                        dot_product += float((first.float() * second.float()).sum())
                first_norm, second_norm = first_squared**.5, second_squared**.5
                groups[group] = {'parameter_count': parameter_count, 'planning_gradient_l2': first_norm,
                                 'ssl_gradient_l2': second_norm, 'weighted_ssl_gradient_l2': .1 * second_norm,
                                 'weighted_ssl_over_planning': .1 * second_norm / max(first_norm, 1e-12),
                                 'gradient_cosine': dot_product / max(first_norm * second_norm, 1e-12)}
            samples.append({'planning_token': train[sample_index]['token'], 'ssl_clip_index': sample_index,
                            'ssl_loss': float(ssl.detach()), 'groups': groups, 'seconds': time.time() - started})
            write_json(DIAGNOSTIC_ROOT / 'gradients/separate_gradients.json', {
                       'complete': sample_index == 1, 'checkpoint_updates': 1920, 'samples': samples,
                       'device': 'cpu', 'no_optimizer_step': True, 'unclipped': True,
                       'ssl_weight': .1, 'planning_batch': 1, 'ssl_batch': 1,
                       'scope': 'Fixed two training samples, original resolution/ELBO; CPU execution diagnostic, not historical DDP gradients'})
            print(json.dumps({'gradient_sample_complete': sample_index, 'encoder': groups['encoder']}), flush=True)
            del result, ssl, ssl_gradients, planning_values
    finally:
        oracle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('prepare', 'infer', 'score', 'gradients'), required=True)
    parser.add_argument('--condition', choices=CONDITIONS)
    arguments = parser.parse_args()
    if arguments.mode == 'prepare':
        prepare_panel()
    elif arguments.mode == 'gradients':
        gradients()
    else:
        {'infer': infer, 'score': score}[arguments.mode](arguments.condition)
