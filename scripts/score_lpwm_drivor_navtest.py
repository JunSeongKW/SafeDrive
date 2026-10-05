"""Full official NAVSIM v1 PDMS. No oracle vocabulary or training-score shortcut."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import lzma
import multiprocessing
from pathlib import Path
import pickle
import sys
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/'reference_repositories/DrivoR'))


def score_scene(job):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    token,poses,cache_path=job
    sampling=TrajectorySampling(num_poses=40,interval_length=.1)
    with lzma.open(cache_path,'rb') as stream:
        metric_cache=pickle.load(stream)
    result=pdm_score(metric_cache,Trajectory(poses),sampling,PDMSimulator(sampling),PDMScorer(sampling))
    row={'token':token,'log_name':Path(cache_path).parts[-4],'valid':True,**{name:float(value) for name,value in asdict(result).items()}}
    assert all(np.isfinite(value) for name,value in row.items() if name not in ('token','log_name','valid'))
    return row


def run(arguments):
    import pandas as pd
    import yaml
    predictions=np.load(arguments.predictions)
    expected=set(yaml.safe_load((PROJECT_ROOT/'reference_repositories/DrivoR/navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml').read_text())['tokens'])
    assert set(predictions['tokens'].tolist())==expected and len(expected)==12146
    cache_root=PROJECT_ROOT/'outputs/official_drive_jepa_reproduction/downloaded_metric_cache/metric_cache'
    by_token={path.parent.name:path for path in cache_root.glob('*/*/*/metric_cache.pkl')}
    assert not expected-set(by_token), f'Missing {len(expected-set(by_token))} official evaluation caches'
    jobs=[(str(token),poses,by_token[str(token)]) for token,poses in zip(predictions['tokens'],predictions['trajectories'])]
    with ProcessPoolExecutor(max_workers=arguments.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        rows=list(pool.map(score_scene,jobs,chunksize=16))
    frame=pd.DataFrame(rows)
    output=arguments.predictions.parent
    ego=np.load(PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/evaluation_inputs/navtest/ego.npy',mmap_mode='r')
    input_tokens=json.loads((PROJECT_ROOT/'outputs/lpwm_drivor_joint_v1/evaluation_inputs/navtest/manifest.json').read_text())['tokens']
    intent_by_token=dict(zip(input_tokens,ego[:,7:11].argmax(-1).tolist()))
    frame['ego_command_index']=frame['token'].map(intent_by_token)
    frame.to_csv(output/'official_navtest_scores.csv',index=False)
    summary={'complete':True,'benchmark':'NAVSIM v1 full navtest','count':len(rows),'failed':0,
             'pdms':float(frame['score'].mean())*100,
             'mean_metrics':frame.drop(columns=['ego_command_index']).select_dtypes('number').mean().to_dict(),
             'by_command':frame.groupby('ego_command_index')['score'].agg(['count','mean']).to_dict('index')}
    (output/'evaluation_complete.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    run(parser.parse_args())
