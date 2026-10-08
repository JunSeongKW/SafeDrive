"""Two-GPU native 512x256 LPWM SSL, with a fixed local corpus and resumable state.

This is a provisional 16-particle adaptation, not the completed 330-hour corpus.
Profiling uses disposable models. Training starts from the public initialization.
"""
import argparse
from contextlib import nullcontext
import ctypes
from datetime import timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import cv2
import numpy as np
import torch
from torch import distributed as distributed
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from train_lpwm_reduced_particle_stage1 import (
    ROOT, ARTIFACT_ROOT, DrivingClips, digest, evaluate, load_rectangular_lpwm,
    prepare_records, write_json,
)

STOP_REQUESTED = False
LOSS_WEIGHTS = dict(beta_kl=.08, beta_obj=.08, beta_dyn=.2, beta_rec=.125,
                    beta_dyn_rec=1., kl_balance=.01)


def request_stop(*_arguments):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def card_memory(physical_gpu):
    return int(subprocess.check_output([
        'nvidia-smi', '--id=' + str(physical_gpu), '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()) * 1024**2


class WorldModelLoss(torch.nn.Module):
    """Return only the loss graph to DDP so unused-parameter detection is correct."""
    def __init__(self, world_model, reconstruction):
        super().__init__()
        self.world_model = world_model
        self.reconstruction = reconstruction
        self.last_components = {}

    def forward(self, video):
        prediction = self.world_model(video, deterministic=False, with_loss=True,
            warmup=False, num_static=1, recon_loss_func=self.reconstruction,
            recon_loss_type='vgg', **LOSS_WEIGHTS)
        components = prediction['loss_dict']
        self.last_components = {name: value.detach().mean() for name, value in components.items()
                                if torch.is_tensor(value)}
        return components['loss']


def random_state():
    return dict(torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore_random_state(saved):
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])


def save_checkpoint(output, model, optimizer, completed, registration, rank):
    rank_states = [None] * distributed.get_world_size()
    distributed.all_gather_object(rank_states, random_state())
    if rank == 0:
        pending = output / 'latest.pending.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            completed_updates=completed, registration=registration, rank_rng_states=rank_states,
            consumed_training_clips=completed * registration['effective_batch_clips']), pending)
        pending.replace(output / 'latest.pt')
    distributed.barrier()


def validate(model, validation, reconstruction, output, label, rank):
    # Evaluation must not change either rank's subsequent stochastic training stream.
    saved_random = random_state()
    distributed.barrier()
    report = None
    if rank == 0:
        report = evaluate(model, validation, reconstruction, output, label)
        from publish_lpwm_reduced_particle_comparison import publish
        publish(output.parent, output / 'particle_overlays')
    distributed.barrier()
    restore_random_state(saved_random)
    return report


def main(arguments):
    global STOP_REQUESTED
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    physical_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert physical_gpus == ['0', '1']
    physical_gpu = int(physical_gpus[rank])
    ctypes.CDLL(None).prctl(15, b'kjs-lpwm-stage1', 0, 0, 0)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.cuda.set_device(rank)
    distributed.init_process_group('nccl', timeout=timedelta(minutes=20))
    assert distributed.get_world_size() == 2
    lock = (output / f'rank{rank}.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    initial_card_memory = card_memory(physical_gpu)
    allocator_budget = min(44_000_000_000, 48_000_000_000 - initial_card_memory - 2_000_000_000)
    assert allocator_budget > 20_000_000_000
    torch.cuda.set_per_process_memory_fraction(
        allocator_budget / torch.cuda.get_device_properties(rank).total_memory)
    torch.manual_seed(47)
    np.random.seed(47)
    random.seed(47)
    training, validation = prepare_records(arguments.manifest, 32)
    effective_batch = arguments.micro_batch * arguments.accumulation * 2
    assert effective_batch == 16, 'This registered continuation uses a fixed effective batch of 16'
    assert len(training) % effective_batch == 0
    target_updates = arguments.updates if arguments.profile else len(training) // effective_batch
    source_paths = [Path(__file__), ROOT / 'scripts/train_lpwm_reduced_particle_stage1.py',
        ROOT / 'scripts/publish_lpwm_reduced_particle_comparison.py',
        ROOT / 'src/planning_aware_future_prediction/object_centric/lpwm_rectangular.py',
        ROOT / 'src/planning_aware_future_prediction/object_centric/lpwm_bridge.py']
    source_paths += sorted((ROOT / 'reference_repositories/LPWM').rglob('*.py'))
    registration = dict(foreground_particles=16, background_particles=1, image_width=512,
        image_height=256, frames=8, fps=2, observed_frames_for_validation=2,
        train_clip_count=len(training), validation_recordings=32, train_validation_recording_overlap=0,
        training_order_sha256=hashlib.sha256(json.dumps(training).encode()).hexdigest(),
        manifest=str(arguments.manifest), manifest_sha256=digest(arguments.manifest),
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_paths},
        micro_batch_per_gpu=arguments.micro_batch, accumulation=arguments.accumulation,
        effective_batch_clips=effective_batch, gpus=[0, 1], workers_per_gpu=arguments.workers,
        prefetch_factor=2, precision='float32', optimizer='Adam', learning_rate=8e-5,
        optimizer_eps=1e-6, gradient_clip_norm=100., loss=LOSS_WEIGHTS, seed=47,
        target_updates=target_updates, profile_only=arguments.profile,
        all_native_world_model_weights_trainable=True, object_gt_supervision=False,
        planning_loss=False, ego_intent_input=False, full_330h_training=False,
        reduced_particle_quality_gate_passed=False,
        intent='longer provisional reduced-particle adaptation during dataset download',
        initial_weights='public Sketchy checkpoint, rectangular spatial adapter, no screening updates reused',
        sample_order='fixed shuffled clip list, rank striding, no repeats or dropped clips',
        whole_card_cap_bytes=48_000_000_000)
    if rank == 0:
        if (output / 'registration.json').exists():
            assert json.loads((output / 'registration.json').read_text()) == registration
        else:
            write_json(output / 'registration.json', registration)
    distributed.barrier()
    model, _ = load_rectangular_lpwm('cuda', foreground_particles=16)
    assert all(parameter.requires_grad for parameter in model.parameters())
    os.environ['TORCH_HOME'] = str(ARTIFACT_ROOT / 'torch')
    os.chdir(ARTIFACT_ROOT)
    from utils.loss_functions import LossLPIPS
    reconstruction = LossLPIPS(normalized_rgb=False).to('cuda').eval().requires_grad_(False)
    objective = WorldModelLoss(model, reconstruction)
    parallel = DistributedDataParallel(objective, device_ids=[rank], output_device=rank,
        find_unused_parameters=True, broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-5, eps=1e-6)
    completed = 0
    if (output / 'latest.pt').exists():
        saved = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        assert saved['registration'] == registration
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        completed = saved['completed_updates']
        restore_random_state(saved['rank_rng_states'][rank])
        del saved
    else:
        # Identical initialized model, independent per-rank stochastic draws.
        torch.manual_seed(47 + rank)
        np.random.seed(47 + rank)
        random.seed(47 + rank)
        if not arguments.profile:
            validate(model, validation, reconstruction, output, 'before_training', rank)
    rank_records = training[completed * effective_batch:target_updates * effective_batch][rank::2]
    loader_options = dict(batch_size=arguments.micro_batch, num_workers=arguments.workers,
        shuffle=False, pin_memory=True, generator=torch.Generator().manual_seed(1047 + rank))
    if arguments.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(DrivingClips(rank_records), **loader_options)
    iterator = iter(loader)
    started = time.time()
    while completed < target_updates:
        should_stop = torch.tensor(int(STOP_REQUESTED or (output / 'pause.requested').exists()
            or card_memory(physical_gpu) > 46_500_000_000), device='cuda')
        distributed.all_reduce(should_stop, op=distributed.ReduceOp.MAX)
        if should_stop.item():
            if not arguments.profile:
                save_checkpoint(output, model, optimizer, completed, registration, rank)
            if rank == 0:
                write_json(output / 'paused.json', dict(completed_updates=completed,
                    reason='user signal, pause marker or shared card pressure', saved=not arguments.profile))
            distributed.destroy_process_group()
            return
        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_started = time.time()
        total_loss = torch.zeros((), device='cuda')
        component_sums = {}
        for micro_step in range(arguments.accumulation):
            context = parallel.no_sync() if micro_step + 1 < arguments.accumulation else nullcontext()
            with context:
                video = next(iterator).to('cuda', non_blocking=True).float() / 255
                loss = parallel(video) / arguments.accumulation
                assert torch.isfinite(loss), 'Nonfinite loss'
                total_loss += loss.detach()
                for name, value in objective.last_components.items():
                    component_sums[name] = component_sums.get(name, 0.) + value / arguments.accumulation
                loss.backward()
                del video, loss
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
        gradient_groups = {}
        if completed == 0 or (completed + 1) % 50 == 0 or arguments.profile:
            for name, module in [('encoder', model.encoder_module), ('context', model.ctx_module),
                                 ('dynamics', model.dyn_module), ('decoder', model.decoder_module)]:
                norm = torch.stack([parameter.grad.detach().square().sum()
                    for parameter in module.parameters() if parameter.grad is not None]).sum().sqrt()
                assert torch.isfinite(norm) and norm > 0, name
                gradient_groups[name] = float(norm)
        optimizer.step()
        distributed.all_reduce(total_loss)
        torch.cuda.synchronize()
        completed += 1
        record = dict(completed_updates=completed, target_updates=target_updates, rank=rank,
            loss=float(total_loss / 2), loss_components_local={name: float(value) for name, value in component_sums.items()},
            seconds=time.time() - update_started, elapsed_seconds=time.time() - started,
            card_used_bytes=card_memory(physical_gpu), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), gradient_norm=float(gradient_norm),
            gradient_groups=gradient_groups, consumed_training_clips=completed * effective_batch)
        with (output / f'training_rank{rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        write_json(output / f'progress_rank{rank}.json', record)
        if rank == 0 and (completed == 1 or completed % 10 == 0 or arguments.profile):
            print(json.dumps(record), flush=True)
        if not arguments.profile and (completed == 1 or completed % 50 == 0):
            save_checkpoint(output, model, optimizer, completed, registration, rank)
        if not arguments.profile and completed % 100 == 0 and completed < target_updates:
            validate(model, validation, reconstruction, output, f'update{completed}', rank)
    if arguments.profile:
        if rank == 0:
            write_json(output / 'complete.json', dict(profile_passed=True, weights_discarded=True,
                completed_updates=completed, effective_batch_clips=effective_batch))
    else:
        save_checkpoint(output, model, optimizer, completed, registration, rank)
        report = validate(model, validation, reconstruction, output, 'after_training', rank)
        if rank == 0:
            write_json(output / 'complete.json', dict(completed_updates=completed,
                consumed_training_clips=len(training), local_epoch_complete=True,
                validation=report['mean'], planning_performance_verified=False,
                reduced_particle_quality_gate_passed=False, full_330h_training=False,
                stage2_started=False, next='admit verified expanded corpus into a new registered segment'))
    distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl')
    parser.add_argument('--micro-batch', type=int, required=True)
    parser.add_argument('--accumulation', type=int, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--updates', type=int, default=4)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / ('failed_rank' + os.environ.get('LOCAL_RANK', '0') + '.json'),
            dict(error=repr(error), time=time.time()))
        raise
