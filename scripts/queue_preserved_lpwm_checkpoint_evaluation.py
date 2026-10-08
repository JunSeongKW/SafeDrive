"""GPU0/1 inference and visualization, followed by official CPU scoring."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'outputs/lpwm_preserved_checkpoint_same_panels_20261009'
RESULTS=ROOT/'results/lpwm_preserved_checkpoint_same_panels_20261009'
SCRIPT=ROOT/'scripts/evaluate_preserved_lpwm_same_panels.py'


def write_status(value):
    value.update(updated_at=datetime.now().astimezone().isoformat(),controller_pid=os.getpid(),no_training_updates=True)
    pending=OUTPUT/'queue_state.pending.json'
    pending.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    pending.replace(OUTPUT/'queue_state.json')
    print(json.dumps(value,ensure_ascii=False),flush=True)


def run_phase(phase,condition=None,gpu=None,panel=None):
    assert not (OUTPUT/'pause.requested').exists(), 'Evaluation pause requested'
    arguments=[sys.executable,'-u',str(SCRIPT),'--phase',phase]
    if condition:
        arguments += ['--condition',condition]
    if panel:
        arguments += ['--panel',panel]
    if phase=='infer':
        arguments += ['--batch-size','4']
    environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     PYTHONPATH=str(ROOT/'src'),PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if gpu is not None:
        environment['CUDA_VISIBLE_DEVICES']=str(gpu)
    label='_'.join(str(value) for value in (phase,condition,panel) if value)
    with (OUTPUT/(label+'.log')).open('a') as stream:
        started=time.time()
        child=subprocess.Popen(arguments,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        (OUTPUT/(label+'.launch.json')).write_text(json.dumps(dict(pid=child.pid,arguments=arguments,gpu=gpu,started_unix=started))+'\n')
        returncode=child.wait()
        assert returncode==0, f'{label} failed: see {OUTPUT/(label+".log")}'


def execute_gpu_condition(item):
    condition,gpu=item
    run_phase('infer',condition,gpu)
    run_phase('capture',condition,gpu)
    return condition


def main():
    assert (OUTPUT/'registration.json').exists()
    assert (ROOT/'outputs/four_model_small_corpus_v1/pause.requested').exists()
    write_status(dict(status='running',stage='historical_128_inference_and_visualization'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute_gpu_condition,[('historical_adapter',0),('historical_joint',1)]))
    write_status(dict(status='running',stage='historical_128_official_pdms'))
    for panel in ('reduced_dev','shared_navtest'):
        for condition in ('historical_adapter','historical_joint'):
            run_phase('score',condition,panel=panel)
    write_status(dict(status='running',stage='rectangular_512_heldout_inference_and_visualization'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute_gpu_condition,[('rectangular_sequential',0),('rectangular_joint',1)]))
    write_status(dict(status='running',stage='rectangular_512_heldout_official_pdms'))
    for condition in ('rectangular_sequential','rectangular_joint'):
        run_phase('score',condition,panel='shared_navtest')
    run_phase('render')
    run_phase('summarize')
    summaries={panel:{condition:json.loads((OUTPUT/panel/condition/'pdms.json').read_text())
        for condition in ('historical_adapter','historical_joint','rectangular_sequential','rectangular_joint')
        if (OUTPUT/panel/condition/'pdms.json').exists()} for panel in ('reduced_dev','shared_navtest')}
    result=dict(complete=True,summaries=summaries,registration=json.loads((OUTPUT/'registration.json').read_text()),
        visualization=json.loads((RESULTS/'visualization_report.json').read_text()),
        current_training_remains_paused=True,finished_unix=time.time(),
        interpretation='System comparison with preserved original inputs and budgets; not a resolution or particle-count causal ablation')
    for destination in (OUTPUT,RESULTS):
        (destination/'evaluation_complete.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    write_status(dict(status='complete',stage='all_requested_evaluation_and_visualization_complete'))


if __name__=='__main__':
    try:
        if len(sys.argv)>1:
            import argparse
            parser=argparse.ArgumentParser()
            parser.add_argument('--condition',choices=['rectangular_sequential','rectangular_joint'],required=True)
            parser.add_argument('--gpu',type=int,choices=[0,1],required=True)
            arguments=parser.parse_args()
            execute_gpu_condition((arguments.condition,arguments.gpu))
        else:
            main()
    except BaseException as error:
        write_status(dict(status='failed',error=repr(error)))
        raise
