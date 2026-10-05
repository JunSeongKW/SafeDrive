"""Run the pinned official NAVSIM v2.2 two-stage evaluator and require coverage."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT=Path(__file__).resolve().parents[1]
REFERENCE=PROJECT_ROOT/'reference_repositories/NAVSIMOfficialV2LPWMDrivoR'


def run(arguments):
    output=arguments.predictions.parent.resolve()
    inputs_root=PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/evaluation_inputs'/arguments.split
    expected=set(json.loads((inputs_root/'manifest.json').read_text())['tokens'])
    with np.load(arguments.predictions) as predictions:
        assert expected==set(predictions['tokens'].tolist())
    cache=PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/v2_metric_cache'/arguments.split
    environment={**os.environ,'PYTHONPATH':str(REFERENCE)+':'+str(PROJECT_ROOT/'scripts'),
        'NUPLAN_MAPS_ROOT':str(PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/maps'),
        'NUPLAN_MAP_VERSION':'nuplan-maps-v1.0','OPENSCENE_DATA_ROOT':str(PROJECT_ROOT/'dataset'),
        'NAVSIM_EXP_ROOT':str(PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1'),'CUDA_VISIBLE_DEVICES':'',
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    shared=[f'train_test_split={arguments.split}',f'metric_cache_path={cache}',
        f'synthetic_sensor_path={PROJECT_ROOT}/dataset/{arguments.split}/sensor_blobs',
        f'synthetic_scenes_path={PROJECT_ROOT}/dataset/{arguments.split}/synthetic_scene_pickles',
        'worker=single_machine_thread_pool','worker.use_process_pool=true','worker.max_workers=8']
    subprocess.run([sys.executable,str(REFERENCE/'navsim/planning/script/run_metric_caching.py'),
        *shared,'force_feature_computation=false'],env=environment,cwd=REFERENCE,check=True)
    sys.path.insert(0,str(REFERENCE))
    from navsim.common.dataloader import MetricCacheLoader
    available=set(MetricCacheLoader(cache).tokens)
    assert not expected-available,f'Missing {len(expected-available)} official v2 metric caches'
    subprocess.run([sys.executable,str(REFERENCE/'navsim/planning/script/run_pdm_score.py'),*shared,
        'agent=constant_velocity_agent','agent._target_=lpwm_drivor_cached_agent.CachedTrajectoryAgent',
        f'+agent.predictions_path={arguments.predictions.resolve()}', 'experiment_name=lpwm_drivor_'+arguments.split,
        f'output_dir={output}/official_v2'],env=environment,cwd=REFERENCE,check=True)
    files=sorted((output/'official_v2').glob('*.csv'))
    assert len(files)==1,files
    scores=pd.read_csv(files[0])
    scene_rows=scores[scores['token'].isin(expected)]
    assert set(scene_rows['token'])==expected and len(scene_rows)==len(expected)
    assert scene_rows['valid'].all() and np.isfinite(scene_rows['score']).all()
    combined=scores[scores['token']=='extended_pdm_score_combined']
    assert len(combined)==1 and bool(combined['valid'].iloc[0])
    summary={'complete':True,'benchmark':'NAVSIM v2.2 '+arguments.split,'count':len(expected),
        'failed':0,'epdms':float(combined['score'].iloc[0])*100,'source_commit':'359c7f72304bfa8273e754224a213d3751bd2340',
        'official_csv':str(files[0])}
    (output/'evaluation_complete.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--split',choices=['warmup_two_stage','navhard_two_stage'],required=True)
    run(parser.parse_args())
