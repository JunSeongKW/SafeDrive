"""Matched DrivoR planning loops for the four small-corpus representations."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import hashlib
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
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from planning_aware_future_prediction.object_centric.small_corpus_models import CommonPlannerModel
from planning_aware_future_prediction.object_centric.lpwm_drivor_joint import make_official_loss, oracle_loss_callback
from train_lpwm_drivor_joint import OfficialSceneDataset, device_inputs
from train_lpwm_reduced_particle_stage1 import prepare_records, DrivingClips, write_json, digest, ARTIFACT_ROOT
from train_lpwm_local_stage1_distributed import random_state, restore_random_state, card_memory, LOSS_WEIGHTS
from lpwm_drivor_oracle import DrivoROracleClient
from navsim.agents.drivoR.layers.losses import drivor_loss as official_loss_module


def binary_safety_targets(values):
    # Upstream edits a shared target view in place; inFP32 this invalidates
    # earlier BCE saved targets. The same0.5->0 mapping must be out of place.
    return torch.where(values == .5, torch.zeros_like(values), values)


official_loss_module.three_to_two_classes = binary_safety_targets

CORPUS=ROOT/'outputs/four_model_small_corpus_v1/corpus/manifest.json'


class PlanningObjective(torch.nn.Module):
    def __init__(self, model, oracle, joint):
        super().__init__(); self.model=model; self.oracle=oracle; self.joint=joint
        self.criterion,self.official_config=make_official_loss('navsim_v1')
        self.last={}
        if joint:
            os.environ['TORCH_HOME']=str(ARTIFACT_ROOT/'torch')
            os.chdir(ARTIFACT_ROOT)
            from utils.loss_functions import LossLPIPS
            self.reconstruction=LossLPIPS(normalized_rgb=False).cuda().eval().requires_grad_(False)

    def forward(self, features, targets, tokens, ssl_video=None):
        prediction=self.model(features)
        scores=torch.from_numpy(self.oracle.score(tokens,prediction['proposals'].detach().float().cpu().numpy())).cuda()
        losses=self.criterion(targets,prediction,self.official_config,scoring_function=oracle_loss_callback(scores))
        planning=losses['loss']
        ssl=planning.new_zeros(())
        if self.joint:
            assert ssl_video is not None
            result=self.model.backbone.world_model(ssl_video,deterministic=False,with_loss=True,
                warmup=False,num_static=1,recon_loss_func=self.reconstruction,recon_loss_type='vgg',**LOSS_WEIGHTS)
            ssl=result['loss_dict']['loss']
        self.last={name:float(losses[name].detach()) for name in
                   ('loss','trajectory_loss','final_score_loss','score','best_score')}
        self.last.update(planning_loss=float(planning.detach()),ssl_loss=float(ssl.detach()),
                         total_objective=float((planning+.1*ssl).detach()))
        return planning+.1*ssl


def gradient_groups(model):
    groups={'planner':list(model.planner.trajectory_decoder.parameters())+list(model.planner.traj_head.parameters()),
            'scorer':list(model.planner.scorer_attention.parameters())+list(model.planner.scorer.parameters())}
    if model.kind.startswith('lpwm'):
        world=model.backbone.world_model
        heads=world.encoder_module.particle_enc.particle_attribute_enc
        groups.update(encoder=list(world.encoder_module.parameters()),context=list(world.ctx_module.parameters()),
            dynamics=list(world.dyn_module.parameters()),xy=list(heads.xy_head.parameters()),
            scale=list(heads.scale_xy_head.parameters()),presence=list(heads.obj_on_head.parameters()),
            command=list(model.backbone.command_modulation.parameters()))
        if model.kind=='lpwm_joint': groups['decoder']=list(world.decoder_module.parameters())
    else: groups['encoder']=list(model.backbone.encoder.parameters())
    return {name:float(torch.stack([p.grad.detach().float().square().sum() for p in parameters
                                   if p.grad is not None]).sum().sqrt()) for name,parameters in groups.items()}


def save(output, model, optimizer, scheduler, completed, registration, rank, epoch_end=False):
    states=[None,None]; dist.all_gather_object(states,random_state())
    if rank==0:
        pending=output/'latest.pending.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            completed_updates=completed,registration=registration,rank_rng_states=states),pending)
        pending.replace(output/'latest.pt')
        if epoch_end and completed//640 in (1,3,5):
            torch.save(dict(model=model.state_dict(),completed_updates=completed,registration=registration),
                       output/f'pass{completed//640}.pt')
    dist.barrier()


def render_particles(output,label,visuals):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plot
    from matplotlib.patches import Rectangle
    destination=output/'particles'; destination.mkdir(exist_ok=True)
    np.savez_compressed(destination/(label+'.npz'),**visuals)
    before=np.load(destination/'before.npz') if (destination/'before.npz').exists() else None
    figure,axes=plot.subplots(3,3,figsize=(15,8))
    for scene in range(3):
        image=visuals[f'scene{scene}_image']
        current=visuals[f'scene{scene}_attributes']
        initial=before[f'scene{scene}_attributes'] if before is not None else current
        for column in range(3):
            axis=axes[scene,column]; axis.imshow(image); axis.axis('off')
            choices=[(initial,'cyan')] if column==0 else [(current,'orange')] if column==1 else [(initial,'cyan'),(current,'orange')]
            for attributes,color in choices:
                centers=(attributes[:,:2][:,::-1]+1)*np.array([256,128])-.5
                sizes=attributes[:,2:4][:,::-1]*np.array([512,256])
                axis.scatter(centers[:,0],centers[:,1],s=10,color=color)
                for center,size in zip(centers,sizes): axis.add_patch(Rectangle(center-size/2,*size,fill=False,color=color,lw=.7))
            axis.set_title(('Before planning','Current','Overlay')[column])
    figure.suptitle(label+': fixed16 particle centers and glimpse sizes; not detections')
    figure.tight_layout(); figure.savefig(destination/(label+'.png'),dpi=130); plot.close(figure)


@torch.no_grad()
def evaluate(model,records,output,label,rank):
    saved_random=random_state(); dist.barrier()
    if rank==0:
        model.eval(); trajectories=[]; tokens=[]; errors=[]; visuals={}
        is_particle=model.kind.startswith('lpwm')
        loader=DataLoader(OfficialSceneDataset(records),batch_size=1,num_workers=2,shuffle=False,
                          generator=torch.Generator().manual_seed(20047))
        for index,examples in enumerate(loader):
            features,targets=device_inputs(examples,0,1,'cuda','navsim_v1')
            if is_particle: model.backbone.record_particles=index<3
            prediction=model(features)['trajectory'].float()
            errors.append((prediction[...,:2]-targets['trajectory'][...,:2]).norm(dim=-1).cpu().numpy())
            trajectories.append(prediction.cpu().numpy()); tokens.extend(examples['token'])
            if is_particle and index<3:
                visuals[f'scene{index}_image']=examples['images'][0,-1].numpy()
                visuals[f'scene{index}_attributes']=model.backbone.latest_attributes[0,0].numpy()
        if is_particle:
            model.backbone.record_particles=False; render_particles(output,label,visuals)
        predictions=output/'validation'; predictions.mkdir(exist_ok=True)
        np.savez_compressed(predictions/(label+'.npz'),tokens=np.asarray(tokens),trajectories=np.concatenate(trajectories))
        distances=np.concatenate(errors)
        write_json(predictions/(label+'.json'),dict(count=len(tokens),ade=float(distances.mean()),
            fde=float(distances[:,-1].mean()),official_pdms_pending=True,full_navtest=False))
    dist.barrier(); restore_random_state(saved_random)


def main(arguments):
    rank=int(os.environ['LOCAL_RANK']); assert os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    import ctypes
    ctypes.CDLL(None).prctl(15,b'kjs-small-plan',0,0,0)
    torch.set_num_threads(2); torch.cuda.set_device(rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=45))
    output=arguments.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    lock=(output/f'rank{rank}.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    allowance=min(44_000_000_000,48_000_000_000-card_memory(rank)-2_000_000_000)
    assert allowance>10_000_000_000
    torch.cuda.set_per_process_memory_fraction(allowance/torch.cuda.get_device_properties(rank).total_memory)
    torch.manual_seed(47); np.random.seed(47); random.seed(47)
    manifest=json.loads(CORPUS.read_text()); unique=[r for r in manifest['records'] if r['study_split']=='train']
    validation=[r for r in manifest['records'] if r['study_split']=='dev']
    assert len(unique)==10240 and len(validation)==1024
    records=[]
    for epoch in range(5):
        shuffled=list(unique); random.Random(4800+epoch).shuffle(shuffled); records.extend(shuffled)
    total_updates=3200 if not arguments.profile_updates else arguments.profile_updates
    micro=arguments.micro_batch; accumulation=8//micro; assert micro*accumulation*2==16
    joint=arguments.kind=='lpwm_joint'
    source_paths=[Path(__file__),ROOT/'src/planning_aware_future_prediction/object_centric/small_corpus_models.py',
                  ROOT/'scripts/lpwm_drivor_oracle.py']
    registration=dict(kind=arguments.kind,seed=47,planner_seed=4701,microbatch_per_gpu=micro,
        effective_batch=16,total_updates=total_updates,training_scenes=10240,dev_scenes=1024,
        manifest_sha256=digest(CORPUS),source_sha256={str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        stage1_checkpoint=str(arguments.stage1_checkpoint),ssl_weight=.1 if joint else 0.,
        ssl_presentations_target=52400 if joint else 0,planning_presentations=51200,
        camera_count=1,observed_frames=2,width=512,height=256,scene_tokens=16,
        native_drivor_temporal_input_modified=True,profile_updates=arguments.profile_updates,
        learning_rates={'native_backbone':1e-5,'drivor_lora':1e-4,'planner':1e-4},
        checkpoint_selection='fixedpass5; devcurves at1/3/5; not fullnavtest')
    if rank==0:
        if (output/'registration.json').exists(): assert json.loads((output/'registration.json').read_text())==registration
        else: write_json(output/'registration.json',registration)
    model=CommonPlannerModel(arguments.kind,arguments.stage1_checkpoint).cuda()
    optimizer=torch.optim.AdamW(model.optimizer_groups(),weight_decay=.01)
    warmup=160
    def schedule(step):
        if step<warmup: return .1+.9*step/warmup
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,(step-warmup)/(3200-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    completed=0
    if (output/'latest.pt').exists():
        saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=False)
        assert saved['registration']==registration
        model.load_state_dict(saved['model'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); completed=saved['completed_updates']
        restore_random_state(saved['rank_rng_states'][rank]); del saved
    elif rank==0:
        planner_state={name:value for name,value in model.planner.state_dict().items()
                       if not name.startswith('image_backbone.') and name!='scene_embeds'}
        torch.save(planner_state,output/'initial_common_planner.pt')
        write_json(output/'parameter_counts.json',dict(total=sum(p.numel() for p in model.parameters()),
            trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if completed==0 and not arguments.profile_updates:
        evaluate(model,validation[:3],output,'before',rank)
    oracle=DrivoROracleClient(CORPUS,output,rank,4)
    objective=PlanningObjective(model,oracle,joint)
    wrapper=DistributedDataParallel(objective,device_ids=[rank],find_unused_parameters=True,
                                    broadcast_buffers=False,gradient_as_bucket_view=True)
    loader=iter(DataLoader(OfficialSceneDataset(records[completed*16:total_updates*16][rank::2]),
        batch_size=micro,num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(1047+rank)))
    if joint:
        unique_ssl,_=prepare_records(ROOT/'outputs/lpwm_driving_video_512x256_v1/corpus/local_openscene_episodes.jsonl',32)
        ssl_records=[]
        for epoch in range(5):
            shuffled=list(unique_ssl); random.Random(4700+epoch).shuffle(shuffled); ssl_records.extend(shuffled)
        ssl_cursor=(completed*3275//3200)*16
        ssl_loader=iter(DataLoader(DrivingClips(ssl_records[ssl_cursor:][rank::2]),batch_size=micro,
            num_workers=2,pin_memory=True,generator=torch.Generator().manual_seed(2047+rank)))
    stopping_requested=[False]
    def request_stop(*_): stopping_requested[0]=True
    signal.signal(signal.SIGTERM,request_stop); signal.signal(signal.SIGINT,request_stop)
    started=time.time()
    profile_ssl_presentations=0
    try:
        while completed<total_updates:
            update_started=time.time(); model.train(); optimizer.zero_grad(set_to_none=True); totals={}
            ssl_batches=(completed+1)*3275//3200-completed*3275//3200 if joint else 0
            if joint and arguments.profile_updates and completed+1==total_updates:
                ssl_batches=2  # Exercise the maximum SSL load used by75real updates.
            profile_ssl_presentations+=ssl_batches*16
            for index in range(accumulation):
                examples=next(loader); features,targets=device_inputs(examples,0,micro,'cuda','navsim_v1')
                ssl_video=torch.cat([next(ssl_loader) for _ in range(ssl_batches)]).cuda().float()/255 if joint else None
                with wrapper.no_sync() if index+1<accumulation else nullcontext():
                    loss=wrapper(features,targets,examples['token'],ssl_video)/accumulation
                    assert torch.isfinite(loss); loss.backward()
                for name,value in objective.last.items(): totals[name]=totals.get(name,0.)+value/accumulation
                del features,targets,ssl_video,examples,loss
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),100.,error_if_nonfinite=True)
            groups=gradient_groups(model) if completed<2 or (completed+1)%100==0 or arguments.profile_updates else {}
            assert all(np.isfinite(value) and value>0 for value in groups.values()),groups
            optimizer.step(); scheduler.step(); completed+=1; torch.cuda.synchronize()
            status=dict(completed_updates=completed,target_updates=total_updates,corpus_pass=completed/640,
                seconds=time.time()-update_started,elapsed_seconds=time.time()-started,
                card_used_bytes=card_memory(rank),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gradient_norm=float(norm),gradient_groups=groups,planning_presentations=completed*16,
                ssl_presentations=profile_ssl_presentations if arguments.profile_updates else
                    (completed*3275//3200)*16 if joint else 0,**totals)
            with (output/f'training_rank{rank}.jsonl').open('a') as stream: stream.write(json.dumps(status)+'\n')
            if rank==0:
                write_json(output/'progress.json',status)
                if completed%10==0 or completed==1: print(json.dumps(status),flush=True)
            stop=torch.tensor(int(stopping_requested[0] or (output/'pause.requested').exists()
                or card_memory(rank)>46_500_000_000),device='cuda')
            dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if not arguments.profile_updates and (completed%100==0 or completed%640==0 or stop.item()):
                save(output,model,optimizer,scheduler,completed,registration,rank,completed%640==0)
            if stop.item():
                if rank==0: write_json(output/'paused.json',status)
                return
            if not arguments.profile_updates and completed in (640,1920,3200):
                evaluate(model,validation,output,f'pass{completed//640}',rank)
        if arguments.profile_updates:
            evaluate(model,validation[:3],output,'profile_after',rank)
        if rank==0:
            write_json(output/'complete.json',dict(completed_updates=completed,profile=bool(arguments.profile_updates),
                profile_weights_discarded=bool(arguments.profile_updates),actual_ssl_presentations=status['ssl_presentations'],
                official_pdms_pending=not arguments.profile_updates,last_status=status))
    finally:
        oracle.close(); dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--kind',choices=['jepa','drivor','lpwm_sequential','lpwm_joint'],required=True)
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--stage1-checkpoint',type=Path)
    parser.add_argument('--micro-batch',type=int,default=1); parser.add_argument('--profile-updates',type=int,default=0)
    main(parser.parse_args())
