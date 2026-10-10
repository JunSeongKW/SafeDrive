"""Evaluate a saved LPWM epoch and reuse verified baseline scores on the same panel."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from compare_planning_epoch_checkpoints import prepare_comparison
from compare_lpwm_front_history_epoch1 import (
    PROJECT_ROOT, context, read_json, sha256, verify_registration, write_json,
)


def prepare_evaluation(arguments):
    prepare_comparison(arguments)
    configuration, output = context(arguments.config)
    registration = read_json(output / 'registration.json')
    registration['sources'][str(Path(__file__).relative_to(PROJECT_ROOT))] = sha256(Path(__file__))
    registration['planning_epochs_by_condition'] = {
        name: specification['planning_epoch']
        for name, specification in configuration['conditions'].items()
    }
    write_json(output / 'registration.json', registration)


def reuse_baseline_results(configuration_path):
    configuration, output = context(configuration_path)
    registration = verify_registration(configuration_path)
    previous_configuration_path = PROJECT_ROOT / configuration['baseline_comparison_configuration']
    previous_configuration, previous_output = context(previous_configuration_path)
    previous_registration = verify_registration(previous_configuration_path)
    assert configuration['panel'] == previous_configuration['panel']
    assert configuration['inputs'] == previous_configuration['inputs']
    for source in (configuration['panel'],
                   'scripts/compare_lpwm_front_history_epoch1.py',
                   'scripts/score_lpwm_drivor_navtest.py'):
        assert registration['sources'][source] == previous_registration['sources'][source], source
    for source, expected in previous_registration['sources'].items():
        if source.startswith(configuration['inputs'] + '/'):
            assert registration['sources'][source] == expected, source
    records = read_json(PROJECT_ROOT / configuration['panel'])['records']
    reused = {}
    for condition in ('drivor', 'jepa'):
        specification = configuration['conditions'][condition]
        previous_specification = previous_configuration['conditions'][condition]
        assert specification['checkpoint'] == previous_specification['checkpoint']
        assert specification['completed_updates'] == previous_specification['completed_updates']
        source_folder = previous_output / condition
        evaluation = read_json(source_folder / 'evaluation_complete.json')
        assert evaluation['checkpoint_sha256'] == registration['saved_baseline_checkpoint_sha256'][condition]
        assert evaluation['count'] == 1024 and evaluation['failed'] == 0
        with np.load(source_folder / 'predictions.npz') as predictions:
            assert predictions['tokens'].tolist() == [record['token'] for record in records]
        destination = output / condition
        destination.mkdir(parents=True, exist_ok=True)
        files = {}
        for filename in ('predictions.npz', 'prediction_complete.json', 'scores.csv', 'evaluation_complete.json'):
            source = source_folder / filename
            target = destination / filename
            expected = sha256(source)
            if target.exists():
                assert sha256(target) == expected
            else:
                shutil.copyfile(source, target)
            files[filename] = expected
        reused[condition] = {'source_directory': str(source_folder.relative_to(PROJECT_ROOT)),
                             'planning_epoch': specification['planning_epoch'], 'file_sha256': files}
    write_json(output / 'baseline_result_reuse.json', {'verified': True, 'models': reused,
        'same_checkpoint_panel_inputs_and_official_scorer': True})


def report_evaluation(configuration_path):
    configuration, output = context(configuration_path)
    registration = verify_registration(configuration_path)
    results = {condition: read_json(output / condition / 'evaluation_complete.json')
               for condition in configuration['conditions']}
    for condition, result in results.items():
        assert result['count'] == 1024 and result['failed'] == 0
        assert result['completed_updates'] == configuration['conditions'][condition]['completed_updates']
        assert result['checkpoint_sha256'] == registration['saved_baseline_checkpoint_sha256'][condition]
    previous = read_json(PROJECT_ROOT / configuration['previous_lpwm_comparison'])
    summary = {'complete': True, 'count': 1024, 'full_navtest': False,
        'recording_count': registration['recording_count'], 'results': results,
        'planning_epochs_by_condition': registration['planning_epochs_by_condition'],
        'same_planning_epoch_comparison': len(set(registration['planning_epochs_by_condition'].values())) == 1,
        'conditions': configuration['conditions'], 'overlap': registration['overlap'],
        'baseline_result_reuse': read_json(output / 'baseline_result_reuse.json'),
        'previous_lpwm_pdms_same_panel': previous['results']['lpwm']['pdms'],
        'lpwm_pdms_change_since_previous_saved_epoch': results['lpwm']['pdms'] - previous['results']['lpwm']['pdms'],
        'lpwm_minus_baseline_pdms': {condition: results['lpwm']['pdms'] - results[condition]['pdms']
                                     for condition in ('drivor', 'jepa')},
        'interpretation': configuration['interpretation'], 'selection_policy': configuration['selection_policy']}
    write_json(output / 'comparison_complete.json', summary)
    lines = ['# LPWM 저장 epoch 평가와 기존 대조군 비교', '',
             '동일 NAVTEST 1,024장면·44주행 기록·공식 NAVSIM v1 PDMS. 전체 NAVTEST 평가는 아니다.', '',
             '| 모델 | Planning epoch | Update | PDMS | ADE (m) | FDE (m) |',
             '|---|---:|---:|---:|---:|---:|']
    for condition, result in results.items():
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
            condition, configuration['conditions'][condition]['planning_epoch'], result['completed_updates'],
            result['pdms'], result['ade_meters'], result['fde_meters']))
    lines.extend(['', configuration['interpretation'], '', configuration['selection_policy'], ''])
    (output / 'comparison.md').write_text('\n'.join(lines))
    print({'complete': True, 'planning_epochs': summary['planning_epochs_by_condition'],
           'pdms': {condition: result['pdms'] for condition, result in results.items()}}, flush=True)


def run_evaluation(arguments):
    configuration, output = context(arguments.config)
    reuse_baseline_results(arguments.config)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}
    gpu_python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    oracle_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    for mode, completion in (('infer', 'prediction_complete.json'), ('score', 'evaluation_complete.json')):
        if (output / 'lpwm' / completion).exists():
            continue
        execution_environment = environment if mode == 'infer' else environment | {
            'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'LD_LIBRARY_PATH': str(oracle_root / 'lib'),
            'PYTHONPATH': str(PROJECT_ROOT / 'reference_repositories/DrivoR')}
        python = gpu_python if mode == 'infer' else oracle_root / 'bin/python'
        command = [str(python), '-u', str(PROJECT_ROOT / 'scripts/compare_planning_epoch_checkpoints.py'),
                   '--config', str(arguments.config), '--mode', mode, '--condition', 'lpwm']
        with (output / f'{mode}_lpwm.log').open('a') as stream:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment,
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            write_json(output / 'controller_state.json', {'status': mode, 'pid': os.getpid(),
                       'active_pid': process.pid, 'command': command, 'time': time.time()})
            assert process.wait() == 0, f'{mode} failed; original training remains intact'
    report_evaluation(arguments.config)
    write_json(output / 'controller_state.json', {'status': 'complete', 'pid': os.getpid(), 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'run', 'report'), required=True)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    try:
        if arguments.mode == 'prepare':
            prepare_evaluation(arguments)
        elif arguments.mode == 'run':
            run_evaluation(arguments)
        else:
            report_evaluation(arguments.config)
    except Exception as error:
        _, output = context(arguments.config)
        write_json(output / 'failed.json', {'mode': arguments.mode, 'error': repr(error), 'time': time.time()})
        raise
