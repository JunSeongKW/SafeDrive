"""Fresh driving adaptation of public V-JEPA2 on the shared small clip order."""
import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

import numpy as np
import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.nn import functional as functional
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from planning_aware_future_prediction.object_centric.small_corpus_models import (
    make_jepa_networks, initialize_jepa_from_public, normalize_imagenet)
from train_lpwm_reduced_particle_stage1 import prepare_records, DrivingClips, write_json, digest
from train_lpwm_local_stage1_distributed import random_state, restore_random_state, card_memory


class JEPAObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder, self.predictor = make_jepa_networks()
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.initialization = initialize_jepa_from_public(self.encoder, self.predictor, self.target_encoder)

    def forward(self, video, encoder_mask, target_mask):
        from src.masks.utils import apply_masks
        video = normalize_imagenet(video).permute(0, 2, 1, 3, 4)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                target = self.target_encoder([video])[0]
                target = functional.layer_norm(target, (target.shape[-1],))
                target = apply_masks(target, [target_mask], concat=False)[0]
            current = self.encoder([video], [[encoder_mask]])
            predicted = self.predictor(current, [[encoder_mask]], [[target_mask]])[0][0]
            return functional.l1_loss(predicted.float(), target.float())


def save(output, objective, optimizer, scheduler, completed, registration, rank):
    states = [None, None]
    dist.all_gather_object(states, random_state())
    if rank == 0:
        pending = output / 'latest.pending.pt'
        torch.save(dict(encoder=objective.encoder.state_dict(), predictor=objective.predictor.state_dict(),
            target_encoder=objective.target_encoder.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), completed_updates=completed, registration=registration,
            rank_rng_states=states), pending)
        pending.replace(output / 'latest.pt')
    dist.barrier()


@torch.no_grad()
def validate(objective, validation, output, label, rank):
    saved_random = random_state()
    dist.barrier()
    if rank == 0:
        from src.masks.multiseq_multiblock3d import _MaskGenerator
        torch.manual_seed(9047)
        generator = _MaskGenerator(crop_size=(256,512), num_frames=8, spatial_patch_size=(16,16),
            temporal_patch_size=2, spatial_pred_mask_scale=(.15,.15), temporal_pred_mask_scale=(1.,1.),
            aspect_ratio=(.75,1.5), npred=8)
        objective.eval()
        losses, feature_std = [], []
        for video in DataLoader(DrivingClips(validation), batch_size=1):
            video = video.to('cuda').float()/255
            observed, target = generator(1)
            losses.append(float(objective(video, observed.cuda(), target.cuda())))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                features = objective.encoder([normalize_imagenet(video).permute(0,2,1,3,4)])[0]
            feature_std.append(float(features.float().std(dim=1).mean()))
        write_json(output / (label+'.json'), dict(masked_latent_l1=float(np.mean(losses)),
            spatial_feature_std=float(np.mean(feature_std)), held_out_recordings=len(validation),
            comparable_to_lpwm_reconstruction_loss=False))
    dist.barrier()
    restore_random_state(saved_random)


def main(arguments):
    rank = int(os.environ['LOCAL_RANK'])
    assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    import ctypes
    ctypes.CDLL(None).prctl(15, b'kjs-jepa-stage1',0,0,0)
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    output = arguments.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    free_budget = 48_000_000_000-card_memory(rank)-2_000_000_000
    torch.cuda.set_per_process_memory_fraction(min(free_budget,44_000_000_000)/torch.cuda.get_device_properties(rank).total_memory)
    torch.manual_seed(47); np.random.seed(47); random.seed(47)
    unique, validation = prepare_records(ROOT/'outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl',32)
    records=[]
    for corpus_pass in range(5):
        selected=list(unique); random.Random(4700+corpus_pass).shuffle(selected); records.extend(selected)
    accumulation=8//arguments.micro_batch
    assert accumulation*arguments.micro_batch*2==16
    total_updates=len(records)//16 if not arguments.profile else arguments.updates
    registration=dict(unique_clips=len(unique), clip_presentations=len(records), effective_batch=16,
        target_updates=total_updates, micro_batch=arguments.micro_batch, seed=47,
        starting_checkpoint='public general-video V-JEPA2 ViT-L', warmup_fraction=.05,
        optimizer='AdamW', lr=1e-4, final_lr=1e-5, weight_decay=.04, ema=.99925,
        objective='official masked-latent L1, EMA stop-gradient target', precision='bf16',
        profile=arguments.profile, source_sha256=digest(Path(__file__)))
    objective=JEPAObjective().cuda()
    if rank==0:
        write_json(output/'initialization.json', objective.initialization)
        if (output/'registration.json').exists():
            assert json.loads((output/'registration.json').read_text())==registration
        else: write_json(output/'registration.json',registration)
    optimizer=torch.optim.AdamW([p for p in objective.parameters() if p.requires_grad],lr=1e-4,weight_decay=.04)
    warmup=max(1,int(total_updates*.05))
    def schedule(step):
        if step<warmup: return .1+.9*step/warmup
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_updates-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    completed=0
    if (output/'latest.pt').exists():
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['registration']==registration
        for name in ('encoder','predictor','target_encoder'): getattr(objective,name).load_state_dict(saved[name],strict=True)
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        completed=saved['completed_updates']; restore_random_state(saved['rank_rng_states'][rank]); del saved
    elif not arguments.profile: validate(objective,validation,output,'before_training',rank)
    else: validate(objective,validation[:3],output,'profile_before',rank)
    wrapper=DistributedDataParallel(objective,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False)
    from src.masks.multiseq_multiblock3d import _MaskGenerator
    masks=_MaskGenerator(crop_size=(256,512),num_frames=8,spatial_patch_size=(16,16),
        temporal_patch_size=2,spatial_pred_mask_scale=(.15,.15),temporal_pred_mask_scale=(1.,1.),
        aspect_ratio=(.75,1.5),npred=8)
    loader=iter(DataLoader(DrivingClips(records[completed*16:total_updates*16][rank::2]),
        batch_size=arguments.micro_batch,num_workers=4,pin_memory=True,shuffle=False,
        generator=torch.Generator().manual_seed(1047+rank)))
    stop=[False]
    def request_stop(*_): stop[0]=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    started=time.time()
    while completed<total_updates:
        objective.train(); objective.target_encoder.eval(); optimizer.zero_grad(set_to_none=True)
        start=time.time(); total=0.
        for micro in range(accumulation):
            # Per-update masking seeds make resume independent of DataLoader prefetch.
            masks._itr_counter.value=(completed*accumulation+micro)*2+rank-1
            torch.manual_seed(900000+(completed*accumulation+micro)*2+rank)
            observed,target=masks(arguments.micro_batch)
            video=next(loader).to('cuda',non_blocking=True).float()/255
            with wrapper.no_sync() if micro+1<accumulation else nullcontext():
                loss=wrapper(video,observed.cuda(),target.cuda())/accumulation
                assert torch.isfinite(loss); loss.backward(); total+=float(loss.detach())
        norm=torch.nn.utils.clip_grad_norm_([p for p in objective.parameters() if p.requires_grad],100.,error_if_nonfinite=True)
        gradient_groups={name:float(torch.stack([parameter.grad.detach().float().square().sum()
            for parameter in module.parameters() if parameter.grad is not None]).sum().sqrt())
            for name,module in [('encoder',objective.encoder),('predictor',objective.predictor)]}
        assert all(np.isfinite(value) and value>0 for value in gradient_groups.values()),gradient_groups
        optimizer.step(); scheduler.step()
        with torch.no_grad():
            for online,target in zip(objective.encoder.parameters(),objective.target_encoder.parameters()):
                target.lerp_(online,1-.99925)
        completed+=1; torch.cuda.synchronize()
        status=dict(completed_updates=completed,target_updates=total_updates,loss=total,
            gradient_norm=float(norm),gradient_groups=gradient_groups,
            seconds=time.time()-start,card_used_bytes=card_memory(rank),
            clip_presentations=completed*16,elapsed_seconds=time.time()-started)
        with (output/f'training_rank{rank}.jsonl').open('a') as stream: stream.write(json.dumps(status)+'\n')
        if rank==0: write_json(output/'progress.json',status)
        stopping=torch.tensor(int(stop[0] or (output/'pause.requested').exists() or card_memory(rank)>46_500_000_000),device='cuda')
        dist.all_reduce(stopping,op=dist.ReduceOp.MAX)
        if not arguments.profile and (completed%100==0 or completed%655==0 or completed==total_updates or stopping.item()):
            save(output,objective,optimizer,scheduler,completed,registration,rank)
        if stopping.item():
            if rank==0: write_json(output/'paused.json',status)
            dist.destroy_process_group(); return
        if not arguments.profile and completed%655==0:
            validate(objective,validation,output,f'pass{completed//655}',rank)
    if arguments.profile: validate(objective,validation[:3],output,'profile_after',rank)
    if rank==0: write_json(output/'complete.json',dict(completed_updates=completed,profile=arguments.profile,weights_discarded=arguments.profile))
    dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--micro-batch',type=int,default=2); parser.add_argument('--profile',action='store_true')
    parser.add_argument('--updates',type=int,default=2)
    main(parser.parse_args())
