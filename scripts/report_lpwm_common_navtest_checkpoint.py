"""Summarize fixed LPWM and preserved baselines on the same NAVTEST subset.

Reuses registered predictions and official selected-trajectory scores. This
script performs no model inference, optimizer step, or candidate-oracle search.
"""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform
import shutil
import sys

import numpy as np

from calibrate_frozen_drivor_scorer import recording_bootstrap
from calibrate_official_toad_navsim_v1_scorer import protected_files_audit
from compare_lpwm_front_history_epoch1 import (
    PROJECT_ROOT, context, read_json, sha256, verify_registration, write_json,
)


def read_aligned_scores(condition_directory, expected_tokens):
    with (condition_directory / 'scores.csv').open() as score_stream:
        records = list(csv.DictReader(score_stream))
    assert [record['token'] for record in records] == expected_tokens
    scores = np.array([float(record['score']) * 100 for record in records])
    assert np.isfinite(scores).all()
    return scores


def report_comparison(configuration_path):
    configuration, output_directory = context(configuration_path)
    registration = verify_registration(configuration_path)
    results_directory = PROJECT_ROOT / configuration['results_directory']
    results_directory.mkdir(parents=True, exist_ok=True)
    panel = read_json(PROJECT_ROOT / configuration['panel'])['records']
    sample_tokens = [record['token'] for record in panel]
    recording_groups = [record['recording_group'] for record in panel]
    ego_status = np.load(PROJECT_ROOT / configuration['inputs'] / 'historical_joint_ego.npy', mmap_mode='r')
    command_indices = ego_status[:, 7:11].argmax(axis=-1)
    evaluations = {}
    score_vectors = {}
    for condition_name, specification in configuration['conditions'].items():
        condition_directory = output_directory / condition_name
        evaluation = read_json(condition_directory / 'evaluation_complete.json')
        assert evaluation['count'] == len(panel) == 1024 and evaluation['failed'] == 0
        assert evaluation['completed_updates'] == specification['completed_updates']
        assert evaluation['checkpoint_sha256'] == sha256(PROJECT_ROOT / specification['checkpoint'])
        predictions = np.load(condition_directory / 'predictions.npz')
        assert predictions['tokens'].tolist() == sample_tokens
        assert predictions['trajectories'].shape == (1024, 8, 3)
        score_vectors[condition_name] = read_aligned_scores(condition_directory, sample_tokens)
        assert abs(score_vectors[condition_name].mean() - evaluation['pdms']) < 1e-8
        evaluations[condition_name] = evaluation
        shutil.copy2(condition_directory / 'scores.csv', results_directory / f'{condition_name}_scores.csv')

    previous_directory = PROJECT_ROOT / configuration['previous_lpwm_same_panel']
    previous_evaluation = read_json(previous_directory / 'evaluation_complete.json')
    previous_scores = read_aligned_scores(previous_directory, sample_tokens)
    comparison_scores = {'lpwm_epoch9': previous_scores,
                         'drivor_epoch5': score_vectors['drivor'],
                         'jepa_epoch5': score_vectors['jepa']}
    paired_differences = {}
    for comparison_name, comparison_vector in comparison_scores.items():
        differences = score_vectors['lpwm'] - comparison_vector
        paired_differences[comparison_name] = {
            'mean_pdms_difference_points': float(differences.mean()),
            'recording_bootstrap_95ci_points': recording_bootstrap(
                differences, recording_groups, configuration['bootstrap_seed'],
                configuration['bootstrap_repetitions']),
        }
    by_current_command = {}
    for command_index in sorted(set(command_indices.tolist())):
        command_mask = command_indices == command_index
        command_name = configuration['command_names'][str(command_index)]
        by_current_command[command_name] = {
            'count': int(command_mask.sum()),
            'pdms_by_model': {condition_name: float(scores[command_mask].mean())
                              for condition_name, scores in score_vectors.items()},
        }
    protection_configuration = read_json(PROJECT_ROOT / 'configs/frozen_planner_postprocessing/official_toad_navsim_v1_scorer_calibration.json')
    protected_training_audit = protected_files_audit(protection_configuration)
    summary = {
        'complete': True, 'benchmark': 'NAVSIM v1', 'full_navtest': False,
        'scope': 'Repeatedly exposed fixed NAVTEST development subset, not independent final evaluation',
        'count': len(panel), 'recording_count': len(set(recording_groups)),
        'planning_epochs_by_condition': {name: specification['planning_epoch']
                                        for name, specification in configuration['conditions'].items()},
        'results': evaluations, 'conditions': configuration['conditions'],
        'previous_lpwm_epoch9': previous_evaluation,
        'lpwm_mean_submetric_change_since_epoch9': {
            metric_name: metric_value - previous_evaluation['mean_metrics'][metric_name]
            for metric_name, metric_value in evaluations['lpwm']['mean_metrics'].items()},
        'paired_pdms_differences': paired_differences,
        'by_current_ego_command': by_current_command,
        'expert_trajectory_turn_strata': {name: evaluation['by_scene_type']
                                          for name, evaluation in evaluations.items()},
        'scene_type_definition': 'Expert terminal yaw >0.25rad left, <-0.25rad right, else straight. Distinct from observed driving command.',
        'current_command_source': 'Cached current ego_status[:,7:11] one-hot driving command; 0 left, 1 straight, 2 right',
        'overlap': registration['overlap'],
        'interpretation': configuration['interpretation'],
        'selection_policy': configuration['selection_policy'],
        'bootstrap_scope': 'Recording resampling of this evaluation panel, not training-seed uncertainty',
        'training_protection_audit': protected_training_audit,
    }
    write_json(results_directory / 'results.json', summary)
    write_json(output_directory / 'comparison_complete.json', summary)
    write_json(results_directory / 'configuration.json', configuration)
    write_json(results_directory / 'subset_manifest.json', {'records': panel})
    with (results_directory / 'summary.csv').open('w', newline='') as result_stream:
        writer = csv.DictWriter(result_stream, fieldnames=['model', 'planning_epoch', 'pdms', 'ade_meters', 'fde_meters', 'scene_count'])
        writer.writeheader()
        for condition_name, evaluation in evaluations.items():
            writer.writerow({'model': condition_name,
                'planning_epoch': configuration['conditions'][condition_name]['planning_epoch'],
                'pdms': evaluation['pdms'], 'ade_meters': evaluation['ade_meters'],
                'fde_meters': evaluation['fde_meters'], 'scene_count': len(panel)})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    condition_names = list(evaluations)
    labels = [f"{name}\nepoch {configuration['conditions'][name]['planning_epoch']}" for name in condition_names]
    values = [evaluations[name]['pdms'] for name in condition_names]
    model_colors = ['#e69f00', '#0072b2', '#009e73']
    bars = axes[0].bar(labels, values, color=model_colors)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel('Official selected-trajectory PDMS')
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 1, f'{value:.2f}', ha='center')
    command_names = list(by_current_command)
    command_offsets = np.arange(len(command_names))
    for model_index, condition_name in enumerate(condition_names):
        axes[1].bar(command_offsets + (model_index - 1) * .25,
                    [by_current_command[name]['pdms_by_model'][condition_name] for name in command_names],
                    width=.25, label=labels[model_index].replace('\n', ' '),
                    color=model_colors[model_index])
    axes[1].set_xticks(command_offsets,
                       [f"{name}\ncount={by_current_command[name]['count']}" for name in command_names])
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel('PDMS by observed driving command')
    axes[1].legend(fontsize=8, loc='lower right')
    figure.suptitle('Shared NAVTEST subset: 1,024 scenes / 44 recordings\nUnequal training budgets; preserved small-corpus baselines')
    figure.tight_layout()
    figure.savefig(results_directory / 'common_navtest_results.png', dpi=160)
    plt.close(figure)

    lines = [f"# LPWM epoch {configuration['planning_epoch']}: common NAVTEST results", '',
        'Same fixed NAVTEST1,024 / 44 recordings, official NAVSIM v1 evaluator. This is a repeatedly exposed development subset; not full NAVTEST or independent final evaluation.', '',
        '| Model | Planning epoch | PDMS | ADE m | FDE m |', '|---|---:|---:|---:|---:|']
    for name, evaluation in evaluations.items():
        lines.append(f"| {name} | {configuration['conditions'][name]['planning_epoch']} | {evaluation['pdms']:.4f} | {evaluation['ade_meters']:.4f} | {evaluation['fde_meters']:.4f} |")
    lines += ['', '## Same-scene changes', '']
    for comparison_name, differences in paired_differences.items():
        lines.append(f"- LPWM minus {comparison_name}: {differences['mean_pdms_difference_points']:+.4f} points; recording bootstrap 95% CI {differences['recording_bootstrap_95ci_points']}.")
    lines += ['', 'Against LPWM epoch9, ADE/FDE decrease from '
              f"{previous_evaluation['ade_meters']:.4f}/{previous_evaluation['fde_meters']:.4f} to "
              f"{evaluations['lpwm']['ade_meters']:.4f}/{evaluations['lpwm']['fde_meters']:.4f} m, "
              'while mean DAC decreases from '
              f"{previous_evaluation['mean_metrics']['drivable_area_compliance']:.6f} to "
              f"{evaluations['lpwm']['mean_metrics']['drivable_area_compliance']:.6f}. "
              'This is an observed submetric change, not an isolated causal attribution of the whole PDMS difference.']
    lines += ['', '## Current ego command', '',
              '| Command | Scenes | LPWM | DrivoR | Drive-JEPA-style |', '|---|---:|---:|---:|---:|']
    for name, values in by_current_command.items():
        scores = values['pdms_by_model']
        lines.append(f"| {name} | {values['count']} | {scores['lpwm']:.4f} | {scores['drivor']:.4f} | {scores['jepa']:.4f} |")
    lines += ['', '## Official mean submetrics', '', '| Metric | LPWM | DrivoR | Drive-JEPA-style |', '|---|---:|---:|---:|']
    for metric_name in evaluations['lpwm']['mean_metrics']:
        if metric_name != 'score':
            metric_values = [evaluations[name]['mean_metrics'][metric_name] for name in condition_names]
            lines.append('| ' + metric_name + ' | ' + ' | '.join(f'{value:.6f}' for value in metric_values) + ' |')
    lines += ['', '## Controls and limits', '', configuration['interpretation'], '',
        'LPWM planning: 75,297 scenes ×14 passes, 16,478 updates, effective64, seed2; front1/history4/128×128; Stage1 23,126 clips×20 epochs. Preserved DrivoR/Drive-JEPA-style: 10,240 scenes×5 passes, 3,200 updates, effective16, seed47; front1/history2/512×256. These are study baselines, not public original full-model checkpoints.', '',
        'Baseline predictions and official scores were copied byte-for-byte. LPWM inference uses the existing strict-loading registered code, original precision and batch4. No original training setting or queue was modified. All three predictions are selected top1 trajectories; these numbers are not candidate-average scores or oracle candidate-best scores.', '',
        'The panel scene_type uses expert terminal yaw and is separate from current ego command. Earlier epoch9 visualization mislabeled these scene_type strata as current command. Historical numeric results are preserved; this report explicitly calculates current-command strata separately.', '',
        'Bootstrap reflects recording-level variation on this panel and does not quantify training-seed variability. Repeated NAVTEST exposure makes this a development diagnosis. A higher score with unequal training budgets does not isolate particle representation or establish superiority over the original papers.', '',
        '![Shared NAVTEST results](common_navtest_results.png)', '', '## Replay', '', '```bash',
        'cd /rhome/junseong/PlanningAwareFuturePrediction',
        'runtime/environments/kjs-lpwm-drivor-joint/bin/python scripts/report_lpwm_common_navtest_checkpoint.py --config configs/lpwm_front_history_stage1_lora/epoch14_common_navtest.json',
        '```', '']
    (results_directory / 'report.md').write_text('\n'.join(lines))
    write_json(results_directory / 'provenance.json', {
        'reported_at_utc': datetime.now(timezone.utc).isoformat(),
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'numpy_version': np.__version__,
        'configuration_sha256': sha256(configuration_path),
        'report_source_sha256': sha256(Path(__file__)),
        'checkpoint_sha256_by_condition': {name: result['checkpoint_sha256'] for name, result in evaluations.items()},
        'registration_path': str(output_directory / 'registration.json'),
        'training_protection_audit': protected_training_audit,
        'prediction_checks': 'token alignment, shape, finite official scores, strict state/checkpoint hashes',
        'inference_cost': evaluations['lpwm'],
        'official_scoring_timing': read_json(output_directory / 'official_scoring_timing.json'),
    })
    print({name: evaluation['pdms'] for name, evaluation in evaluations.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    report_comparison(parser.parse_args().config.resolve())
