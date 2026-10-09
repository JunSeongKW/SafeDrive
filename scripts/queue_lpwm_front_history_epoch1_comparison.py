"""Adopt the live first epoch, evaluate three checkpoints, then restore the epoch queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from compare_lpwm_front_history_epoch1 import context, read_json, verify_registration, write_json
from queue_lpwm_front_history_stage1_lora import GPU_PYTHON, ORACLE_ENVIRONMENT, PROJECT_ROOT, run as run_epoch_queue


def process_identity(pid):
    directory = Path('/proc') / str(pid)
    try:
        fields = (directory / 'stat').read_text().split(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(fields[19]), 'uid': directory.stat().st_uid,
                'command': (directory / 'cmdline').read_bytes().replace(b'\0', b' ').decode().strip()}
    except FileNotFoundError:
        return None


def run(arguments):
    configuration, output = context(arguments.comparison_config)
    handover = read_json(output / 'handover.json')
    assert handover['comparison_configuration'] == str(arguments.comparison_config)
    verify_registration(arguments.comparison_config)
    assert read_json(output / 'baseline_cpu_load_audit.json')['passed']
    training_configuration = PROJECT_ROOT / configuration['training_configuration']
    execution = PROJECT_ROOT / configuration['training_execution']
    training = read_json(training_configuration)
    root = PROJECT_ROOT / training['output_directory']
    lock = (root / 'queue.lock').open('a')
    # Ready before lock acquisition lets the launcher verify the replacement
    # controller before terminating only the old controller (never torchrun).
    write_json(output / 'controller_ready.json', {'pid': os.getpid(), 'ready': True, 'adopted_training': handover['training']})
    for attempt in range(60):
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(1)
    else:
        raise RuntimeError('Previous controller did not release queue lock')
    (root / 'queue.pid').write_text(str(os.getpid()) + '\n')
    identity = process_identity(handover['training']['pid'])
    assert identity == handover['training'], (identity, handover['training'])
    write_json(root / 'queue_state.json', {'status': 'training_epoch_01', 'queue_pid': os.getpid(),
        'active_pid': identity['pid'], 'command': identity['command'], 'adopted_without_training_restart': True,
        'epoch1_common_comparison_pending': True})
    write_json(output / 'controller_state.json', {'status': 'waiting_for_first_epoch', 'pid': os.getpid(),
        'adopted_training': identity, 'first_epoch_updates': 1177})
    while True:
        identity = process_identity(handover['training']['pid'])
        if identity is None:
            break
        assert identity == handover['training'], 'Training PID was reused or its command changed'
        time.sleep(5)
    if (root / 'pause.requested').exists() or (root.parent / 'pause.requested').exists():
        write_json(output / 'controller_state.json', {'status': 'paused_by_user', 'pid': os.getpid()})
        return
    assert (root / 'epoch_01.pt').is_file(), 'First epoch did not finish; do not score or restart automatically'
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'PYTHONPATH': str(PROJECT_ROOT / 'src')}

    def execute(label, command, execution_environment):
        with (output / (label + '.log')).open('a') as stream:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=execution_environment, stdout=stream, stderr=subprocess.STDOUT)
            state = {'status': label, 'pid': os.getpid(), 'active_pid': process.pid, 'command': command}
            write_json(output / 'controller_state.json', state)
            write_json(root / 'queue_state.json', state | {'queue_pid': os.getpid()})
            result = process.wait()
        assert result == 0, '{} failed: exit {}'.format(label, result)

    for condition in configuration['conditions']:
        folder = output / condition
        if not (folder / 'prediction_complete.json').is_file():
            execute('predict_' + condition, [str(GPU_PYTHON), '-u', 'scripts/compare_lpwm_front_history_epoch1.py',
                '--config', str(arguments.comparison_config), '--mode', 'infer', '--condition', condition], environment)
        if not (folder / 'evaluation_complete.json').is_file():
            execute('score_' + condition, [str(ORACLE_ENVIRONMENT / 'bin/python'), '-u', 'scripts/compare_lpwm_front_history_epoch1.py',
                '--config', str(arguments.comparison_config), '--mode', 'score', '--condition', condition], environment | {
                    'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'LD_LIBRARY_PATH': str(ORACLE_ENVIRONMENT / 'lib'),
                    'PYTHONPATH': str(PROJECT_ROOT / 'reference_repositories/DrivoR')})
    execute('comparison_report', [str(GPU_PYTHON), '-u', 'scripts/compare_lpwm_front_history_epoch1.py',
        '--config', str(arguments.comparison_config), '--mode', 'report'], environment | {'CUDA_VISIBLE_DEVICES': ''})
    assert read_json(output / 'comparison_complete.json')['complete']
    write_json(output / 'controller_state.json', {'status': 'comparison_complete_returning_to_original_queue', 'pid': os.getpid()})
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    # The unchanged registered queue runs epoch1 internal validation, then epochs
    # 2..25 with the original config, optimizer, scheduler and regular gates.
    run_epoch_queue(argparse.Namespace(config=training_configuration, execution=execution))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-config', type=Path, required=True)
    arguments = parser.parse_args()
    try:
        run(arguments)
    except Exception as error:
        configuration, output = context(arguments.comparison_config)
        write_json(output / 'controller_failed.json', {'pid': os.getpid(), 'error': repr(error), 'time': time.time()})
        raise
