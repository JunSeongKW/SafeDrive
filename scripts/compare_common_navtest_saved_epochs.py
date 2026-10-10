"""Reuse registered inference/scoring and report the actual saved planning epochs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from compare_planning_epoch_checkpoints import prepare_comparison
from compare_lpwm_front_history_epoch1 import (
    PROJECT_ROOT, context, read_json, sha256, verify_registration, write_json,
)


def prepare(arguments):
    prepare_comparison(arguments)
    configuration, output = context(arguments.config)
    registration = read_json(output / 'registration.json')
    registration['sources'][str(Path(__file__).relative_to(PROJECT_ROOT))] = sha256(Path(__file__))
    registration['planning_epochs_by_condition'] = {
        condition: specification['planning_epoch']
        for condition, specification in configuration['conditions'].items()
    }
    write_json(output / 'registration.json', registration)


def report(arguments):
    configuration, output = context(arguments.config)
    registration = verify_registration(arguments.config)
    results = {condition: read_json(output / condition / 'evaluation_complete.json')
               for condition in configuration['conditions']}
    for condition, result in results.items():
        assert result['count'] == 1024 and result['failed'] == 0
        assert result['completed_updates'] == configuration['conditions'][condition]['completed_updates']
        assert result['checkpoint_sha256'] == registration['saved_baseline_checkpoint_sha256'][condition]
    epochs = registration['planning_epochs_by_condition']
    previous = read_json(PROJECT_ROOT / configuration['previous_common_comparison'])
    previous_lpwm = read_json(PROJECT_ROOT / configuration['previous_lpwm_comparison'])
    summary = {
        'complete': True, 'planning_epochs_by_condition': epochs,
        'same_planning_epoch_comparison': len(set(epochs.values())) == 1,
        'count': 1024, 'recording_count': registration['recording_count'], 'full_navtest': False,
        'results': results, 'conditions': configuration['conditions'], 'overlap': registration['overlap'],
        'previous_common_pdms_same_panel': {condition: value['pdms']
                                            for condition, value in previous['results'].items()},
        'pdms_change_since_previous_common_comparison': {
            condition: value['pdms'] - previous['results'][condition]['pdms']
            for condition, value in results.items()},
        'previous_lpwm_pdms_same_panel': previous_lpwm['results']['lpwm']['pdms'],
        'lpwm_pdms_change_since_previous_saved_epoch': results['lpwm']['pdms'] - previous_lpwm['results']['lpwm']['pdms'],
        'lpwm_minus_baseline_pdms': {condition: results['lpwm']['pdms'] - results[condition]['pdms']
                                     for condition in ('drivor', 'jepa')},
        'interpretation': configuration['interpretation'], 'selection_policy': configuration['selection_policy'],
    }
    lines = ['# 공통 NAVTEST 저장 epoch 비교', '',
             '동일 NAVTEST 1,024장면·44주행 기록·공식 NAVSIM v1 PDMS. 전체 NAVTEST 평가는 아니다.', '',
             '| 모델 | Planning epoch | Update | PDMS | ADE (m) | FDE (m) |',
             '|---|---:|---:|---:|---:|---:|']
    for condition, result in results.items():
        lines.append('| {} | {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
            condition, epochs[condition], result['completed_updates'], result['pdms'],
            result['ade_meters'], result['fde_meters']))
    lines.extend(['', '| 상황 | 장면 수 | LPWM | DrivoR | Drive-JEPA |', '|---|---:|---:|---:|---:|'])
    for scene_type, result in results['lpwm']['by_scene_type'].items():
        lines.append('| {} | {} | {:.4f} | {:.4f} | {:.4f} |'.format(
            scene_type, result['count'], result['pdms'], results['drivor']['by_scene_type'][scene_type]['pdms'],
            results['jepa']['by_scene_type'][scene_type]['pdms']))
    lines.extend(['', configuration['interpretation'], '', configuration['selection_policy'], ''])
    (output / 'comparison.md').write_text('\n'.join(lines))
    write_json(output / 'comparison_complete.json', summary)
    print(json.dumps({'complete': True, 'pdms': {condition: result['pdms']
                                                for condition, result in results.items()}}), flush=True)


def run(arguments):
    configuration, output = context(arguments.config)
    verify_registration(arguments.config)
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}
    gpu_python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    oracle_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    for condition in configuration['conditions']:
        for mode, completion in (('infer', 'prediction_complete.json'), ('score', 'evaluation_complete.json')):
            if (output / condition / completion).exists():
                continue
            execution_environment = environment if mode == 'infer' else environment | {
                'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'LD_LIBRARY_PATH': str(oracle_root / 'lib'),
                'PYTHONPATH': str(PROJECT_ROOT / 'reference_repositories/DrivoR')}
            python = gpu_python if mode == 'infer' else oracle_root / 'bin/python'
            command = [str(python), '-u', str(PROJECT_ROOT / 'scripts/compare_planning_epoch_checkpoints.py'),
                       '--config', str(arguments.config), '--mode', mode, '--condition', condition]
            with (output / f'{mode}_{condition}.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                write_json(output / 'controller_state.json', {'status': f'{mode}_{condition}', 'pid': os.getpid(),
                           'active_pid': process.pid, 'command': command, 'time': time.time()})
                assert process.wait() == 0, f'{mode} {condition} failed; original training remains intact'
    report(arguments)
    write_json(output / 'controller_state.json', {'status': 'complete', 'pid': os.getpid(), 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--mode', choices=('prepare', 'run', 'report'), required=True)
    arguments = parser.parse_args()
    arguments.config = arguments.config.resolve()
    try:
        {'prepare': prepare, 'run': run, 'report': report}[arguments.mode](arguments)
    except Exception as error:
        _, output = context(arguments.config)
        write_json(output / 'failed.json', {'mode': arguments.mode, 'error': repr(error), 'time': time.time()})
        raise
