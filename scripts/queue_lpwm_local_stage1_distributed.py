"""Profile fixed-effective-batch configurations, then train and validate local SSL."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / 'scripts/train_lpwm_local_stage1_distributed.py'


def write_json(path, content):
    pending = path.with_suffix(path.suffix + '.pending')
    pending.write_text(json.dumps(content, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def run(root, micro_batch, profile):
    output = root / 'profiles' / f'batch{micro_batch}' if profile else root / 'training'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'complete.json').exists():
        return output
    if (root / 'pause.requested').exists() or (output / 'paused.json').exists():
        raise RuntimeError('User or resource pause requires inspection')
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(TRAINER), '--micro-batch', str(micro_batch), '--accumulation', str(8 // micro_batch),
        '--workers', '4', '--output', str(output)]
    if profile:
        command += ['--profile', '--updates', '4']
    with (output / 'console.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / 'queue_state.json', dict(status='profiling' if profile else 'training',
            micro_batch=micro_batch, child_pid=process.pid, command=command, output=str(output),
            started_unix=time.time()))
        returncode = process.wait()
    if returncode != 0 or not (output / 'complete.json').exists():
        if profile:
            return None
        raise RuntimeError(f'Training incomplete: returncode {returncode}, inspect {output}')
    return output


def measurements(output):
    ranks = [[json.loads(line) for line in (output / f'training_rank{rank}.jsonl').read_text().splitlines()]
             for rank in (0, 1)]
    assert all(len(rows) == 4 for rows in ranks)
    return dict(median_update_seconds=max(statistics.median(row['seconds'] for row in rows[1:]) for rows in ranks),
        max_card_used_bytes=max(row['card_used_bytes'] for rows in ranks for row in rows),
        max_reserved_bytes=max(row['peak_reserved_bytes'] for rows in ranks for row in rows),
        max_allocated_bytes=max(row['peak_allocated_bytes'] for rows in ranks for row in rows))


def main(arguments):
    root = arguments.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                      NCCL_P2P_DISABLE='1', TORCH_NCCL_ASYNC_ERROR_HANDLING='1')
    registration = dict(source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (Path(__file__), TRAINER)},
        effective_batch_clips=16, particle_count=16, decision='fastest measured safe configuration',
        full_330h_training=False, original_capacity_gate_result_preserved=True)
    if (root / 'queue_registration.json').exists():
        assert json.loads((root / 'queue_registration.json').read_text()) == registration
    else:
        write_json(root / 'queue_registration.json', registration)
    candidates = {}
    for micro_batch in (2, 4):
        output = run(root, micro_batch, profile=True)
        if output is not None:
            candidates[micro_batch] = measurements(output)
    # Avoid a predictable OOM: require a conservative full doubling to fit with headroom.
    batch8_decision = 'not attempted: conservative doubled peak exceeds memory budget'
    if 4 in candidates and candidates[4]['max_reserved_bytes'] * 2 < 43_000_000_000:
        output = run(root, 8, profile=True)
        batch8_decision = 'profiled' if output else 'profile failed; discarded'
        if output is not None:
            candidates[8] = measurements(output)
    safe = {batch: stats for batch, stats in candidates.items()
            if stats['max_card_used_bytes'] < 45_500_000_000}
    assert safe, 'No safe measured batch configuration'
    selected = min(safe, key=lambda batch: safe[batch]['median_update_seconds'])
    write_json(root / 'batch_selection.json', dict(candidates=candidates, selected_micro_batch=selected,
        accumulation=8 // selected, effective_batch_clips=16, batch8_decision=batch8_decision,
        profile_weights_discarded=True, compared_to_screen='effective batch changed 4 to 16; new registered condition',
        estimated_training_seconds=655 * safe[selected]['median_update_seconds'],
        estimate_excludes_checkpoints_validation_and_shared_load=True))
    output = run(root, selected, profile=False)
    write_json(root / 'queue_state.json', dict(status='local_epoch_and_validation_complete',
        output=str(output), full_330h_training=False, stage2_started=False,
        next='admit expanded verified corpus and register next segment after validation review',
        finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/lpwm_driving_video_512x256_v1/local_stage1_distributed')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / 'queue_failed.json', dict(error=repr(error), time=time.time()))
        raise
