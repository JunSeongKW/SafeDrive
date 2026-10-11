"""Evaluate the latest fixed LPWM checkpoint and render native NAVSIM plots."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

from compare_lpwm_front_history_epoch1 import PROJECT_ROOT, read_json, sha256, write_json
from compare_planning_epoch_checkpoints import prepare_comparison


def run(configuration_path):
    configuration_path = configuration_path.resolve()
    configuration = read_json(configuration_path)
    output = PROJECT_ROOT / configuration['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'registration.json').exists():
        prepare_comparison(SimpleNamespace(config=configuration_path))
        write_json(output / 'configuration.json', configuration)
    # This is exactly the saved final DrivoR evaluation on the same panel.
    source = PROJECT_ROOT / configuration['reuse_drivor_evaluation']
    assert read_json(source / 'evaluation_complete.json')['checkpoint_sha256'] == sha256(
        PROJECT_ROOT / configuration['conditions']['drivor']['checkpoint'])
    destination = output / 'drivor'
    destination.mkdir(exist_ok=True)
    for name in ('predictions.npz', 'prediction_complete.json', 'evaluation_complete.json', 'scores.csv'):
        if not (destination / name).exists():
            shutil.copy2(source / name, destination / name)
        assert sha256(source / name) == sha256(destination / name)
    gpu_python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    evaluation_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}
    stages = [('infer_lpwm', str(gpu_python), 'infer', 'prediction_complete.json'),
              ('score_lpwm', str(evaluation_root / 'bin/python'), 'score', 'evaluation_complete.json'),
              ('render_navsim', str(evaluation_root / 'bin/python'), None, None)]
    started = time.time()
    for stage, interpreter, mode, completion in stages:
        if completion and (output / 'lpwm' / completion).exists():
            continue
        execution_environment = environment.copy()
        if mode == 'infer':
            command = [interpreter, '-u', str(PROJECT_ROOT / 'scripts/compare_planning_epoch_checkpoints.py'),
                '--config', str(configuration_path), '--mode', mode, '--condition', 'lpwm']
        elif mode == 'score':
            execution_environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                LD_LIBRARY_PATH=str(evaluation_root / 'lib'))
            command = [interpreter, '-u', str(PROJECT_ROOT / 'scripts/compare_planning_epoch_checkpoints.py'),
                '--config', str(configuration_path), '--mode', mode, '--condition', 'lpwm']
        else:
            execution_environment.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                LD_LIBRARY_PATH=str(evaluation_root / 'lib'),
                NUPLAN_MAPS_ROOT=str(PROJECT_ROOT / 'dataset/maps'))
            command = [interpreter, '-u', str(PROJECT_ROOT / 'scripts/visualize_lpwm_drivor_navsim.py'),
                '--config', str(configuration_path)]
        with (output / (stage+'.log')).open('a') as stream:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
            write_json(output / 'visualization_queue_state.json', {'stage':stage, 'status':'running',
                'pid':os.getpid(), 'child_pid':process.pid, 'started_unix':started})
            return_code = process.wait()
        if return_code:
            write_json(output / 'visualization_queue_state.json', {'stage':stage,'status':'failed',
                'return_code':return_code,'log':stage+'.log'})
            raise RuntimeError('Visualization stage failed: '+stage)
    write_json(output / 'visualization_queue_state.json', {'status':'complete',
        'wall_seconds':time.time()-started,'pid':os.getpid()})


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
