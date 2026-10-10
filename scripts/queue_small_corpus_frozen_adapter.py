"""Matched control queue, ready to launch after resource scheduling is decided."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(arguments):
    specification = json.loads(arguments.config.read_text())
    output = PROJECT_ROOT / specification['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'pause.requested').exists()
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1', 'OMP_NUM_THREADS': '2',
                   'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'NCCL_P2P_DISABLE': '1',
                   'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True'}
    python = PROJECT_ROOT / 'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
    cpu_root = PROJECT_ROOT / 'runtime/environments/drive_jepa_official_evaluation'
    while True:
        usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                         '--format=csv,noheader,nounits'], text=True)
        used = {int(row.split(',')[0]): int(row.split(',')[1]) * 1024**2 for row in usage.splitlines()}
        if all(used[gpu] + 12_000_000_000 < 47_000_000_000 for gpu in (0, 1)):
            break
        if (output / 'pause.requested').exists():
            return
        (output / 'queue_state.json').write_text(json.dumps({'status': 'waiting_for_gpu_headroom',
                                                          'used_bytes': used, 'time': time.time()}))
        time.sleep(20)
    for label, microbatch, boundary, validation_epoch in [
        ('epoch1', 2, 640, 1), ('epoch3', 2, 1920, 3),
        ('same_original_microbatch_boundary', 2, 1937, None), ('epoch5', 4, 0, 5)]:
        checkpoint = output / 'latest.pt'
        training_complete = False
        if checkpoint.exists():
            import torch
            completed = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)['completed_updates']
            if completed >= (boundary or 3200):
                training_complete = True
        command = [str(python), '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                   str(PROJECT_ROOT / 'scripts/train_small_corpus_frozen_adapter.py'),
                   '--config', str(arguments.config), '--execution-microbatch', str(microbatch),
                   '--stop-after-update', str(boundary)]
        if not training_complete:
            with (output / (label + '.log')).open('a') as log:
                code = subprocess.call(command, cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
            assert code == 0, label
        saved = __import__('torch').load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        assert saved['completed_updates'] >= (boundary or 3200), 'Paused before planned boundary'
        del saved
        if validation_epoch is None:
            continue
        predictions = output / 'validation' / f'pass{validation_epoch}.npz'
        if predictions.with_suffix('.pdms.json').exists():
            continue
        scoring_environment = environment | {'CUDA_VISIBLE_DEVICES': '', 'LD_LIBRARY_PATH': str(cpu_root / 'lib')}
        with (output / f'score_pass{validation_epoch}.log').open('a') as log:
            code = subprocess.call([str(cpu_root / 'bin/python'), '-u', str(PROJECT_ROOT / 'scripts/score_four_model_small_corpus.py'),
                                    '--predictions', str(predictions), '--workers', '4'], cwd=PROJECT_ROOT,
                                   env=scoring_environment, stdout=log, stderr=subprocess.STDOUT)
        assert code == 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    main(parser.parse_args())
