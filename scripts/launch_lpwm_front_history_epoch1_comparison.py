"""Replace only the queue controller while preserving its running training child."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time

from compare_lpwm_front_history_epoch1 import context, read_json, verify_registration, write_json
from queue_lpwm_front_history_epoch1_comparison import process_identity
from queue_lpwm_front_history_stage1_lora import GPU_PYTHON, PROJECT_ROOT


def run(arguments):
    configuration, output = context(arguments.comparison_config)
    assert not (output / 'handover.json').exists(), 'This is a one-time live controller handoff'
    verify_registration(arguments.comparison_config)
    assert read_json(output / 'baseline_cpu_load_audit.json')['passed']
    training_configuration = read_json(PROJECT_ROOT / configuration['training_configuration'])
    root = PROJECT_ROOT / training_configuration['output_directory']
    state = read_json(root / 'queue_state.json')
    assert state['status'] == 'training_epoch_01'
    assert not (root / 'pause.requested').exists() and not (root.parent / 'pause.requested').exists()
    queue_identity = process_identity(state['queue_pid'])
    training_identity = process_identity(state['active_pid'])
    assert queue_identity and training_identity
    assert queue_identity['uid'] == training_identity['uid'] == os.getuid()
    assert 'scripts/queue_lpwm_front_history_stage1_lora.py' in queue_identity['command']
    assert 'scripts/train_lpwm_front_history_stage1_lora.py' in training_identity['command']
    assert '--stop-after-epoch' in training_identity['command']
    progress = read_json(root / 'progress.json')
    assert progress['completed_updates'] < 1177
    write_json(output / 'handover.json', {'previous_controller': queue_identity, 'training': training_identity,
        'completed_updates_at_handoff': progress['completed_updates'], 'comparison_configuration': str(arguments.comparison_config),
        'original_training_configuration_sha256': progress['configuration_sha256'],
        'original_training_execution_sha256': progress['execution_sha256'], 'time': time.time()})
    command = [str(GPU_PYTHON), '-u', 'scripts/queue_lpwm_front_history_epoch1_comparison.py',
               '--comparison-config', str(arguments.comparison_config)]
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    with (output / 'controller.log').open('a') as stream:
        controller = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    for attempt in range(30):
        assert controller.poll() is None, 'Replacement controller exited before ready'
        if (output / 'controller_ready.json').is_file():
            assert read_json(output / 'controller_ready.json')['pid'] == controller.pid
            break
        time.sleep(1)
    else:
        raise RuntimeError('Replacement controller readiness timed out; old controller remains intact')
    assert process_identity(queue_identity['pid']) == queue_identity
    assert process_identity(training_identity['pid']) == training_identity
    # SIGTERM the verified owned parent only. Do not signal its process group.
    os.kill(queue_identity['pid'], signal.SIGTERM)
    for attempt in range(30):
        assert controller.poll() is None, 'Replacement controller failed during adoption'
        if (output / 'controller_state.json').is_file():
            assert read_json(output / 'controller_state.json')['status'] == 'waiting_for_first_epoch'
            break
        time.sleep(1)
    else:
        raise RuntimeError('Replacement did not acquire the queue lock')
    assert process_identity(training_identity['pid']) == training_identity
    write_json(output / 'handover_verified.json', {'passed': True, 'queue_pid': controller.pid,
        'training_identity_unchanged': True, 'training': training_identity, 'previous_controller': queue_identity,
        'progress_before': progress['completed_updates'], 'progress_after': read_json(root / 'progress.json')['completed_updates'],
        'training_restart': False, 'first_epoch_comparison_queued': True, 'time': time.time()})
    print((output / 'handover_verified.json').read_text(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-config', type=Path, required=True)
    run(parser.parse_args())
