"""E7 — 고를 수 있는 문제인가, 아니면 장면 이해가 필요한 문제인가?

E6 에서 256 개 anchor 안에 +5.18 PDMS 가 남아 있음을 확인했다.  그렇다면 그 정답을
무엇으로 고를 수 있는지가 남는다.

  - oracle        : 256 개 중 최고점 (상한)
  - nearest_ref   : PDM 기준 궤적에 기하적으로 가장 가까운 anchor 를 고르면?
  - nearest_pred  : 모델 출력에 가장 가까운 anchor (모델의 실질 선택 위치)
  - rank_*        : 그 anchor 가 점수 순위 몇 등인가

nearest_ref 가 oracle 에 가까우면 "기준 궤적만 맞히면 된다"는 뜻이고, 크게 낮으면
어떤 anchor 가 좋은지는 기하로 알 수 없고 장면 이해가 필요하다는 뜻이다.
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

def _init(traj_path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps=TrajectorySampling(num_poses=40, interval_length=0.1)
    _G['ps']=ps; _G['sim']=PDMSimulator(proposal_sampling=ps)
    _G['scorer']=PDMScorer(proposal_sampling=ps, config=PDMScorerConfig(
        progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0, driving_direction_weight=0.0,
        driving_direction_horizon=1.0, driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0, stopped_speed_threshold=5e-3,
        progress_distance_threshold=5.0))
    _G['anchors']=np.load(ANCHORS)
    _G['pred']=pickle.load(open(traj_path,'rb'))
    _G['paths']={}
    for dp,_,fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn: _G['paths'][os.path.basename(dp)]=os.path.join(dp,'metric_cache.pkl')

def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    try:
        with lzma.open(_G['paths'][token],'rb') as f: mc=pickle.load(f)
        ego=mc.ego_state; ps=_G['ps']; A=_G['anchors']
        arr=np.stack([get_trajectory_as_array(transform_trajectory(Trajectory(a.astype(np.float64),ps),ego),ps,ego.time_point) for a in A],axis=0)
        sc=np.asarray(_G['scorer'].score_proposals(_G['sim'].simulate_proposals(arr,ego),
              mc.observation, mc.centerline, mc.route_lane_ids, mc.drivable_area_map), dtype=np.float64)
        order=np.argsort(-sc); rank_of=np.empty(256,dtype=int); rank_of[order]=np.arange(256)

        # PDM 기준 궤적을 ego frame 으로 (anchor 와 같은 좌표계)
        rx,ry,rh=ego.rear_axle.x,ego.rear_axle.y,ego.rear_axle.heading
        ch,sh=np.cos(rh),np.sin(rh)
        ref=np.array([[p.rear_axle.x-rx, p.rear_axle.y-ry] for p in mc.trajectory.get_sampled_trajectory()[:41]])
        ref=np.stack([ref[:,0]*ch+ref[:,1]*sh, -ref[:,0]*sh+ref[:,1]*ch],axis=-1)
        n=min(len(ref),A.shape[1]); ref=ref[:n]
        d_ref=np.linalg.norm(A[:,:n,:2]-ref[None],axis=-1).mean(-1)
        i_ref=int(d_ref.argmin())

        # 모델 출력(8 pose, 0.5 s) 을 anchor 의 대응 시점과 비교
        p=np.asarray(_G['pred'][token],dtype=np.float64)          # (8,3) @0.5s
        idx=[min(int((k+1)*5)-1, A.shape[1]-1) for k in range(8)]  # 0.1s 격자에서 0.5s 간격
        d_pred=np.linalg.norm(A[:,idx,:2]-p[None,:,:2],axis=-1).mean(-1)
        i_pred=int(d_pred.argmin())
        return dict(token=token, valid=1, oracle=round(float(sc.max()),4),
                    near_ref=round(float(sc[i_ref]),4), rank_ref=int(rank_of[i_ref]), dist_ref=round(float(d_ref[i_ref]),2),
                    near_pred=round(float(sc[i_pred]),4), rank_pred=int(rank_of[i_pred]), dist_pred=round(float(d_pred[i_pred]),2))
    except Exception:
        return dict(token=token, valid=0, oracle=-1, near_ref=-1, rank_ref=-1, dist_ref=-1,
                    near_pred=-1, rank_pred=-1, dist_pred=-1)

if __name__=='__main__':
    out,n,tp=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    toks=sorted(pickle.load(open(tp,'rb'))); random.Random(0).shuffle(toks); toks=toks[:n]
    print(len(toks),'tokens',flush=True)
    cols=['token','valid','oracle','near_ref','rank_ref','dist_ref','near_pred','rank_pred','dist_pred']
    with Pool(64, initializer=_init, initargs=(tp,)) as p, open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for i,r in enumerate(p.imap_unordered(work,toks,chunksize=4)):
            w.writerow(r)
            if (i+1)%500==0: print(f'  {i+1}/{n}',flush=True)
    print('DONE ->',out,flush=True)
