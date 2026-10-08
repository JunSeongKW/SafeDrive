"""Durable train/validate queue for the authorized four-model pilot."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/four_model_small_corpus_v1'
GPU_PYTHON=ROOT/'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
SSL_PYTHON=Path('/rhome/junseong/envs/kjs-lpwm-navsim/bin/python')
CPU_PYTHON=ROOT/'runtime/environments/drive_jepa_official_evaluation/bin/python'


def write(name,value):
    path=OUTPUT/name; pending=path.with_suffix(path.suffix+'.pending')
    pending.write_text(json.dumps(value,indent=2)+'\n'); pending.replace(path)


def run(label,command,expected,cpu=False):
    if expected.exists(): return
    if (OUTPUT/'pause.requested').exists(): raise RuntimeError('Study pause requested')
    environment=os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES='' if cpu else '0,1',OMP_NUM_THREADS='2',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NCCL_P2P_DISABLE='1',PYTHONUNBUFFERED='1')
    if cpu: environment['LD_LIBRARY_PATH']=str(CPU_PYTHON.parent.parent/'lib')
    with (OUTPUT/(label+'.log')).open('a') as log:
        child=subprocess.Popen([str(item) for item in command],cwd=ROOT,env=environment,
            stdout=log,stderr=subprocess.STDOUT)
        write('queue_state.json',dict(status='running',stage=label,pid=child.pid,
            command=[str(item) for item in command],started_unix=time.time()))
        result=child.wait()
    if result!=0 or not expected.exists():
        raise RuntimeError(f'{label} failed or paused: returncode={result}; inspect {label}.log')


def distributed_command(script,arguments,python=GPU_PYTHON):
    return [python,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',ROOT/'scripts'/script,*arguments]


def gate_lpwm():
    root=OUTPUT/'lpwm_ssl'
    before=json.loads((root/'before_training.json').read_text())['mean']
    after=json.loads((root/'after_training.json').read_text())['mean']
    checks=dict(future_error_improved=after['forecast_mse']<before['forecast_mse'],
        perceptual_error_not_worse=after['forecast_lpips']<=before['forecast_lpips'],
        positions_not_collapsed=after['position_std']>.01,
        appearance_not_collapsed=after['appearance_std']>.001)
    write('lpwm_stage1_gate.json',dict(checks=checks,passed=all(checks.values()),before=before,after=after,
        meaning='admission to small planning pilot; not proof of full NAVSIM adaptation or16particle adequacy'))
    return all(checks.values())


def gate_jepa():
    root=OUTPUT/'jepa_ssl'
    before=json.loads((root/'before_training.json').read_text())
    after=json.loads((root/'pass5.json').read_text())
    checks=dict(masked_loss_not_worse=after['masked_latent_l1']<=before['masked_latent_l1']*1.05,
                spatial_features_not_collapsed=after['spatial_feature_std']>.01)
    write('jepa_stage1_gate.json',dict(checks=checks,passed=all(checks.values()),before=before,after=after))
    return all(checks.values())


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    lock=(OUTPUT/'queue.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sources=[ROOT/'scripts'/name for name in ('queue_four_model_small_corpus.py','train_small_corpus_lpwm_ssl.py',
        'train_small_corpus_jepa_ssl.py','train_small_corpus_common_planner.py','score_four_model_small_corpus.py')]
    sources += [ROOT/'configs/four_model_small_corpus/experiment.json',
                ROOT/'scripts/prepare_four_model_small_corpus.py',
                ROOT/'scripts/download_small_corpus_backbones.py',
                ROOT/'scripts/lpwm_drivor_oracle.py',ROOT/'scripts/train_lpwm_drivor_joint.py',
                ROOT/'scripts/train_lpwm_local_stage1_distributed.py',
                ROOT/'scripts/train_lpwm_reduced_particle_stage1.py']
    sources+=list((ROOT/'reference_repositories/LPWM').rglob('*.py'))
    sources+=list((ROOT/'reference_repositories/DrivoR/navsim/agents/drivoR').rglob('*.py'))
    sources+=list((ROOT/'reference_repositories/Drive-JEPA/navsim_v1/vjepa2/src').rglob('*.py'))
    sources+=list((ROOT/'src/planning_aware_future_prediction/object_centric').glob('*.py'))
    hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    registration=OUTPUT/'queue_registration.json'
    if registration.exists(): assert json.loads(registration.read_text())['source_sha256']==hashes
    else: write('queue_registration.json',dict(source_sha256=hashes,planner='commonDrivoR',
        user_authorized=True,planning_scenes=10240,dev_scenes=1024,passes=5,ssl_clips=10480,
        ssl_presentations=52400,planning_presentations=51200,effective_batch=16,
        camera_count=1,width=512,height=256,observations=2,ssl_weight_joint=.1,
        root_seed=47,planner_seed=4701,full_navtest=False,card_limit_bytes=48000000000))
    # An externally started firstStage1 can be paused by setup profiling. The
    # setup owner must remove its own marker before this queue is launched.
    lpwm=OUTPUT/'lpwm_ssl'
    assert not (lpwm/'pause.requested').exists()
    run('lpwm_ssl',distributed_command('train_small_corpus_lpwm_ssl.py',[
        '--output',lpwm,'--micro-batch','4','--accumulation','2','--epochs','5','--workers','4'],SSL_PYTHON),lpwm/'complete.json')
    lpwm_ready=gate_lpwm()
    weights=ROOT/'runtime/checkpoints/four_model_small_corpus'
    assert (weights/'vjepa2_vitl.pt.json').exists() and (weights/'dinov2_vits_reg4.safetensors.json').exists()
    jepa_profile=OUTPUT/'profiles/jepa_ssl_verified'
    run('profile_jepa_ssl',distributed_command('train_small_corpus_jepa_ssl.py',[
        '--output',jepa_profile,'--micro-batch','2','--profile','--updates','2']),jepa_profile/'complete.json')
    jepa=OUTPUT/'jepa_ssl'
    run('jepa_ssl',distributed_command('train_small_corpus_jepa_ssl.py',[
        '--output',jepa,'--micro-batch','2']),jepa/'complete.json')
    jepa_ready=gate_jepa()
    completed_kinds=[]
    held_kinds={}
    for kind,checkpoint_path,micro in [('drivor',None,2),('lpwm_sequential',lpwm/'latest.pt',2),
                                     ('lpwm_joint',None,2),('jepa',jepa/'latest.pt',2)]:
        if (kind=='lpwm_sequential' and not lpwm_ready) or (kind=='jepa' and not jepa_ready):
            held_kinds[kind]='Stage1 quality gate failed; dependent planning held, independent conditions continue'
            write('held_conditions.json',held_kinds)
            continue
        arguments=['--kind',kind,'--micro-batch',str(micro)]
        if checkpoint_path is not None: arguments+=['--stage1-checkpoint',checkpoint_path]
        profile=OUTPUT/'profiles'/(kind+'_verified')
        run('profile_'+kind,distributed_command('train_small_corpus_common_planner.py',[
            *arguments,'--output',profile,'--profile-updates','2']),profile/'complete.json')
        destination=OUTPUT/kind
        run('train_'+kind,distributed_command('train_small_corpus_common_planner.py',[
            *arguments,'--output',destination]),destination/'complete.json')
        for epoch in (1,3,5):
            prediction=destination/'validation'/f'pass{epoch}.npz'
            run(f'score_{kind}_pass{epoch}',[CPU_PYTHON,'-u',ROOT/'scripts/score_four_model_small_corpus.py',
                '--predictions',prediction,'--workers','8'],prediction.with_suffix('.pdms.json'),cpu=True)
        completed_kinds.append(kind)
    results={kind:json.loads((OUTPUT/kind/'validation/pass5.pdms.json').read_text())
             for kind in completed_kinds}
    write('comparison_partial.json' if held_kinds else 'comparison_complete.json',dict(
        results=results,held_conditions=held_kinds,full_navtest=False,seed_count=1,
        conclusion_scope='small-data system trends; upstream pretraining differs'))
    write('queue_state.json',dict(status='held_for_stage1_review' if held_kinds else 'complete',finished_unix=time.time()))


if __name__=='__main__':
    try: main()
    except Exception as error:
        write('queue_failed.json',dict(error=repr(error),time=time.time())); raise
