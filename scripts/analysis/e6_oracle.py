"""E6 — 고정 anchor 집합의 천장: 256개 중 '가장 좋은 것'을 신이 골라준다면 몇 점인가?

모델을 전혀 쓰지 않는다. 사람이 미리 클러스터링해 박아둔 256개 anchor 를 그대로 PDM
시뮬레이터에 넣어 각각 채점하고, 장면마다 최고점을 취한다.  이것이 현재 행동 공간이
낼 수 있는 상한이다.

  oracle 이 높다  -> anchor 는 충분하고, 고르는 쪽(월드/점수화)이 병목
  oracle 이 낮다  -> anchor 집합 자체가 천장이고, 월드를 아무리 키워도 못 넘는다
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
ANCHORS='/home/kaist5/data/junseong/SafeDrive/trajectory_anchors/trajectory_anchors_256_GTRS.npy'
_G={}

def _init():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps=TrajectorySampling(num_poses=40, interval_length=0.1)
    _G['ps']=ps
    _G['sim']=PDMSimulator(proposal_sampling=ps)
    _G['scorer']=PDMScorer(proposal_sampling=ps, config=PDMScorerConfig(
        progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0, driving_direction_weight=0.0,
        driving_direction_horizon=1.0, driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0, stopped_speed_threshold=5e-3,
        progress_distance_threshold=5.0))
    _G['anchors']=np.load(ANCHORS)                       # (256, 40, 3) ego frame
    _G['paths']={}
    for dp,_,fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn: _G['paths'][os.path.basename(dp)]=os.path.join(dp,'metric_cache.pkl')

def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    try:
        with lzma.open(_G['paths'][token],'rb') as f: mc=pickle.load(f)
        ego=mc.ego_state; ps=_G['ps']
        states=[]
        for a in _G['anchors']:
            tr=transform_trajectory(Trajectory(a.astype(np.float64), ps), ego)
            states.append(get_trajectory_as_array(tr, ps, ego.time_point))
        arr=np.stack(states, axis=0)                      # (256, T, S)
        sim=_G['sim'].simulate_proposals(arr, ego)
        sc=_G['scorer'].score_proposals(sim, mc.observation, mc.centerline,
                                        mc.route_lane_ids, mc.drivable_area_map)
        sc=np.asarray(sc, dtype=np.float64)
        best=int(sc.argmax())
        fwd=float(_G['anchors'][best,-1,0])
        return dict(token=token, valid=1, oracle=round(float(sc.max()),4),
                    mean256=round(float(sc.mean()),4), top10=round(float(np.sort(sc)[-10:].mean()),4),
                    n_above_90=int((sc>=0.90).sum()), best_idx=best, best_fwd=round(fwd,2))
    except Exception as ex:
        return dict(token=token, valid=0, oracle=-1, mean256=-1, top10=-1,
                    n_above_90=-1, best_idx=-1, best_fwd=-1)

if __name__=='__main__':
    out=sys.argv[1]; n=int(sys.argv[2]) if len(sys.argv)>2 else 3000
    toks=sorted(pickle.load(open(sys.argv[3],'rb')))
    random.Random(0).shuffle(toks); toks=toks[:n]
    print(len(toks),'tokens (무작위 표본)',flush=True)
    cols=['token','valid','oracle','mean256','top10','n_above_90','best_idx','best_fwd']
    with Pool(64, initializer=_init) as p, open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for i,r in enumerate(p.imap_unordered(work,toks,chunksize=4)):
            w.writerow(r)
            if (i+1)%500==0: print(f'  {i+1}/{len(toks)}',flush=True)
    print('DONE ->',out,flush=True)
