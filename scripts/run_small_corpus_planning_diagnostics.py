"""Candidate evaluation queue; historical studies and live training stay intact."""
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs/small_corpus_planning_diagnosis_20261010'


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    script = PROJECT_ROOT / 'scripts/diagnose_small_corpus_planner.py'
    gpu_python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    cpu_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    base_environment = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
                        'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}
    for condition in ('drivor', 'jepa', 'lpwm_sequential', 'lpwm_joint'):
        for mode, marker in (('infer', 'inference_complete.json'), ('score', 'candidate_analysis.json')):
            if (OUTPUT_ROOT / condition / marker).exists():
                continue
            environment = base_environment | {'CUDA_VISIBLE_DEVICES': '0' if mode == 'infer' else ''}
            python = gpu_python
            if mode == 'score':
                python = cpu_root / 'bin/python'
                environment['LD_LIBRARY_PATH'] = str(cpu_root / 'lib')
            command = [str(python), '-u', str(script), '--mode', mode, '--condition', condition]
            with (OUTPUT_ROOT / (condition + '_' + mode + '.log')).open('a') as log:
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                (OUTPUT_ROOT / 'queue_state.json').write_text(json.dumps(
                    {'status': 'running', 'condition': condition, 'mode': mode,
                     'pid': child.pid, 'time': time.time()}, indent=2))
                code = child.wait()
            if code or not (OUTPUT_ROOT / condition / marker).exists():
                raise RuntimeError(f'{condition} {mode} failed; inspect log')
    (OUTPUT_ROOT / 'queue_state.json').write_text(json.dumps({'status': 'complete', 'time': time.time()}))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (OUTPUT_ROOT / 'failed.json').write_text(json.dumps({'error': repr(error), 'time': time.time()}))
        raise
