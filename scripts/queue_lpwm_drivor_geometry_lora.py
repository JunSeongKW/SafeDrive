"""Chain the geometry-LoRA condition to final official evaluations and v2."""
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

PROJECT_ROOT=Path(__file__).resolve().parents[1]
OUTPUT=PROJECT_ROOT/'outputs/lpwm_drivor_geometry_lora_v1'
SHARED_INPUTS=PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/evaluation_inputs'
TRAINING_PYTHON=PROJECT_ROOT/'runtime/environments/kjs-lpwm-drivor-joint/bin/python'
EXECUTION_CONFIGURATION=json.loads((OUTPUT/'parallelism_selection.json').read_text())['execution_configuration']
CPU_PYTHON=PROJECT_ROOT/'runtime/environments/drive_jepa_official_evaluation/bin/python'


def write_status(content):
    pending=OUTPUT/'queue_status.pending.json'
    pending.write_text(json.dumps(content,indent=2))
    pending.replace(OUTPUT/'queue_status.json')


def check_registration():
    registered=json.loads((OUTPUT/'queue_registration.json').read_text())
    registered['sources'].update(json.loads((OUTPUT/'parallelism_registration.json').read_text())['sources'])
    for name,expected in registered['sources'].items():
        assert hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest()==expected,name


def run_command(label,command,cpu=False):
    assert not (OUTPUT/'pause.requested').exists(),'Queue pause requested'
    check_registration()
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'' if cpu else '0,1',
        'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'1' if cpu else '2',
        'MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONUNBUFFERED':'1'}
    if cpu:
        environment['LD_LIBRARY_PATH']=str(CPU_PYTHON.parent.parent/'lib')
    with (OUTPUT/(label+'.log')).open('a') as log:
        process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT)
        write_status({'stage':label,'pid':process.pid,'command':list(map(str,command)),'started_unix':time.time()})
        while process.poll() is None:
            if (OUTPUT/'pause.requested').exists():
                # Trainer checkpoints at an update boundary; CPU/eval children
                # finish their current bounded command before the queue stops.
                write_status({'stage':label,'pid':process.pid,'pause_pending':True})
            time.sleep(10)
        assert process.returncode==0,f'{label} failed with {process.returncode}; see its log'
    assert not (OUTPUT/'pause.requested').exists(),'Queue paused after command completion'


def evaluate(benchmark,split):
    root=OUTPUT/benchmark/'evaluation'/split
    if (root/'evaluation_complete.json').exists():
        assert json.loads((root/'evaluation_complete.json').read_text())['complete']
        return
    if not (SHARED_INPUTS/split/'manifest.json').exists():
        run_command('prepare_'+split,[str(CPU_PYTHON),'-u','scripts/prepare_lpwm_drivor_evaluation.py','--split',split],cpu=True)
    if not (root/'inference_complete.json').exists():
        run_command('infer_'+split,[str(TRAINING_PYTHON),'-u','scripts/infer_lpwm_drivor_geometry_lora.py',
            '--config','configs/lpwm_drivor_geometry_lora/'+benchmark+'.json','--split',split])
    command=[str(CPU_PYTHON),'-u','scripts/score_lpwm_drivor_navtest.py' if split=='navtest' else 'scripts/run_lpwm_drivor_v2_evaluation.py',
        '--predictions',str(root/'predictions.npz')]
    if split!='navtest':
        command+=['--split',split]
    run_command('score_'+split,command,cpu=True)
    assert json.loads((root/'evaluation_complete.json').read_text())['complete']


def main():
    ctypes.CDLL(None).prctl(15,b'kjs-geom-queue',0,0,0)
    lock=(OUTPUT/'queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    check_registration()
    v1=OUTPUT/'navsim_v1'
    launch=json.loads((v1/'launch.json').read_text())
    while not (v1/'training_complete.json').exists():
        assert not (OUTPUT/'pause.requested').exists() and not (v1/'paused.json').exists(),'Training or queue paused'
        process_path=Path('/proc')/str(launch['pid'])
        assert process_path.exists() and process_path.joinpath('stat').read_text().split()[2]!='Z','v1 training stopped before completion'
        write_status({'stage':'navsim_v1_training','adopted_pid':launch['pid'],'updated_unix':time.time()})
        time.sleep(10)
    evaluate('navsim_v1','navtest')
    v2=OUTPUT/'navsim_v2'
    if not (v2/'training_complete.json').exists():
        run_command('navsim_v2_training',[str(TRAINING_PYTHON),'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            'scripts/train_lpwm_drivor_geometry_lora.py','--config','configs/lpwm_drivor_geometry_lora/navsim_v2.json','--execution',EXECUTION_CONFIGURATION])
        assert (v2/'training_complete.json').exists(),'v2 training exited without completion (possibly paused)'
    evaluate('navsim_v2','warmup_two_stage')
    evaluate('navsim_v2','navhard_two_stage')
    write_status({'stage':'complete','completed_unix':time.time(),'all_evaluations_complete':True})


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        write_status({'stage':'stopped','error':repr(error),'updated_unix':time.time()})
        raise
