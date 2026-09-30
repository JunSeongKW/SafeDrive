"""E6 — proposal set 자체의 상한: 256개 anchor 중 최선을 고르면 PDMS 가 얼마인가.

어떤 scoring 함수도 이 값을 넘을 수 없다.  실제 성능과의 격차가 크면 병목은
'무엇을 월드에 담는가'가 아니라 '담긴 것으로 어떻게 고르는가'이고, 격차가 작으면
병목은 proposal 생성 자체다.
"""
import os, sys, lzma, pickle, csv, warnings, random
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/kaist5/data/junseong/SafeDrive')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/home/kaist5/data/junseong/SafeDrive/dataset/maps')
os.environ.setdefault('OPENSCENE_DATA_ROOT','/home/kaist5/data/junseong/SafeDrive/dataset')
import numpy as np
from multiprocessing import Pool

ROOT='/home/kaist5/data/junseong/SafeDrive/exp/metric_cache_navtest'
ANCH='/home/kaist5/data/junseong/SafeDrive/trajectory_anchors/trajectory_anchors_256_GTRS.npy'
IDX=[4,9,14,19,24,29,34,39]        # 0.5s 간격 8-pose: 모델 출력과 동일한 해상도
_G={}

def _init():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps=TrajectorySampling(num_poses=40, interval_length=0.1)
    _G['ps']=ps
    _G['ps8']=TrajectorySampling(time_horizon=4, interval_length=0.5)
    _G['sim']=PDMSimulator(proposal_sampling=ps)
    _G['sc']=PDMScorer(proposal_sampling=ps, config=PDMScorerConfig(
        progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0, driving_direction_weight=0.0,
        driving_direction_horizon=1.0, driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0, stopped_speed_threshold=5e-3,
        progress_distance_threshold=5.0))
    _G['anchors']=np.load(ANCH)[:, IDX, :].astype(np.float64)    # (256, 8, 3)
    _G['paths']={}
    for dp,_,fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn: _G['paths'][os.path.basename(dp)]=os.path.join(dp,'metric_cache.pkl')

def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    try:
        with lzma.open(_G['paths'][token],'rb') as f: mc=pickle.load(f)
        ego=mc.ego_state; A=_G['anchors']
        states=[get_trajectory_as_array(mc.trajectory, _G['ps'], ego.time_point)]
        for i in range(A.shape[0]):
            tr=transform_trajectory(Trajectory(A[i], _G['ps8']), ego)
            states.append(get_trajectory_as_array(tr, _G['ps'], ego.time_point))
        arr=np.stack(states,axis=0)
        sim=_G['sim'].simulate_proposals(arr, ego)
        scores=_G['sc'].score_proposals(sim, mc.observation, mc.centerline,
                                        mc.route_lane_ids, mc.drivable_area_map)
        prop=np.asarray(scores[1:])                  # 256개 anchor 의 PDMS
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import WeightedMetricIndex
        ep=_G['sc']._weighted_metrics[WeightedMetricIndex.PROGRESS, 1:]
        best=int(prop.argmax())
        return dict(token=token, valid=1,
                    oracle=round(float(prop.max()),4),
                    median=round(float(np.median(prop)),4),
                    top10=round(float(np.sort(prop)[-10:].mean()),4),
                    oracle_ep=round(float(ep.max()),4),
                    best_idx=best,
                    best_reach=round(float(A[best,-1,0]),2))
    except Exception as ex:
        return dict(token=token, valid=0, oracle=-1, median=-1, top10=-1,
                    oracle_ep=-1, best_idx=-1, best_reach=type(ex).__name__)

if __name__=='__main__':
    out=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 2000
    toks=sorted(pickle.load(open(
        '/home/kaist5/data/junseong/SafeDrive/exp/training/test_baseline/'
        'traj_IMI0.3_NC16.0_DAC48.0_EP0.75_TTC15.0_W1.0_C0.0_PDM1.0_DDC1.0_TLC1.0_LK1.0_HC0.0_'
        'PwNC5.0_TwDAC1.5_BEVSegPred_TwDACMargin1.4_1.6.pkl','rb')))
    random.Random(0).shuffle(toks); toks=toks[:n]
    print(f'{len(toks)} tokens (무작위 표본)',flush=True)
    cols=['token','valid','oracle','median','top10','oracle_ep','best_idx','best_reach']
    with Pool(32,initializer=_init) as p, open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for i,r in enumerate(p.imap_unordered(work,toks,chunksize=4)):
            w.writerow(r)
            if (i+1)%500==0: print(f'  {i+1}/{len(toks)}',flush=True)
    print('DONE ->',out,flush=True)
