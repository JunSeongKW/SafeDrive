"""E8 — E6/E7 을 공식 정규화 방식으로 다시 계산한다.

PDMScorer._aggregate_scores 는 EP 를 '같은 배치 안의 최대 진행량' 으로 나눈다.
공식 평가의 배치는 {PDM 기준, 후보} 2 개뿐이므로, 256 개를 한 배치에 넣으면 분모가
커져 모든 후보의 EP 가 부당하게 낮아진다.

여기서는 256 개를 한 번에 시뮬레이션해 원자료(_progress_raw, _multi_metrics,
_weighted_metrics)를 뽑은 뒤, 후보마다 {기준, 후보} 2 개짜리 정규화를 손으로 다시
적용한다.  결과는 공식 pdm_score() 와 동일한 정의가 된다.
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
THRESH=5.0            # progress_distance_threshold
_G={}

def _init(traj_path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps=TrajectorySampling(num_poses=40, interval_length=0.1)
    cfg=PDMScorerConfig(progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0,
        driving_direction_weight=0.0, driving_direction_horizon=1.0,
        driving_direction_compliance_threshold=2.0, driving_direction_violation_threshold=6.0,
        stopped_speed_threshold=5e-3, progress_distance_threshold=THRESH)
    _G.update(ps=ps, sim=PDMSimulator(proposal_sampling=ps),
              scorer=PDMScorer(proposal_sampling=ps, config=cfg),
              W=cfg.weighted_metrics_array, anchors=np.load(ANCHORS),
              pred=pickle.load(open(traj_path,'rb')), paths={})
    for dp,_,fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn: _G['paths'][os.path.basename(dp)]=os.path.join(dp,'metric_cache.pkl')

def pair_scores(multi, wm, prog_raw):
    """후보 i 를 {기준(0), 후보 i} 2 개 배치로 정규화했을 때의 최종 점수 (벡터화)."""
    rp = prog_raw * multi                     # (N,)
    ref = rp[0]
    mx  = np.maximum(rp, ref)
    norm = np.where(mx > THRESH, rp / np.where(mx > 0, mx, 1.0),
                    np.where(multi == 0.0, 0.0, 1.0))
    w = wm.copy(); w[0] = norm               # PROGRESS 자리 교체
    return multi * (w * _G['W'][:, None]).sum(0) / _G['W'].sum()

def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
    try:
        with lzma.open(_G['paths'][token],'rb') as f: mc=pickle.load(f)
        ego=mc.ego_state; ps=_G['ps']; A=_G['anchors']; sc=_G['scorer']
        ref_states=get_trajectory_as_array(mc.trajectory, ps, ego.time_point)
        arr=np.stack([ref_states]+[get_trajectory_as_array(
            transform_trajectory(Trajectory(a.astype(np.float64),ps),ego),ps,ego.time_point) for a in A],axis=0)
        sc.score_proposals(_G['sim'].simulate_proposals(arr,ego), mc.observation,
                           mc.centerline, mc.route_lane_ids, mc.drivable_area_map)
        multi=sc._multi_metrics.prod(axis=0)          # (257,)
        final=pair_scores(multi, sc._weighted_metrics.copy(), sc._progress_raw)
        anc=final[1:]                                  # 기준 제외
        # 기하 최근접 anchor
        rx,ry,rh=ego.rear_axle.x,ego.rear_axle.y,ego.rear_axle.heading
        ch,sh=np.cos(rh),np.sin(rh)
        r=np.array([[p.rear_axle.x-rx,p.rear_axle.y-ry] for p in mc.trajectory.get_sampled_trajectory()[:41]])
        r=np.stack([r[:,0]*ch+r[:,1]*sh,-r[:,0]*sh+r[:,1]*ch],axis=-1)
        n=min(len(r),A.shape[1])
        d=np.linalg.norm(A[:,:n,:2]-r[None,:n],axis=-1).mean(-1); i_ref=int(d.argmin())
        p=np.asarray(_G['pred'][token],dtype=np.float64)
        idx=[min(int((k+1)*5)-1,A.shape[1]-1) for k in range(8)]
        dp=np.linalg.norm(A[:,idx,:2]-p[None,:,:2],axis=-1).mean(-1); i_pred=int(dp.argmin())
        order=np.argsort(-anc); rk=np.empty(256,dtype=int); rk[order]=np.arange(256)
        return dict(token=token, valid=1,
                    ref_self=round(float(final[0]),4), oracle=round(float(anc.max()),4),
                    mean256=round(float(anc.mean()),4), n_above_90=int((anc>=0.90).sum()),
                    near_ref=round(float(anc[i_ref]),4), rank_ref=int(rk[i_ref]), dist_ref=round(float(d[i_ref]),2),
                    near_pred=round(float(anc[i_pred]),4), rank_pred=int(rk[i_pred]), dist_pred=round(float(dp[i_pred]),2),
                    best_fwd=round(float(A[int(anc.argmax()),-1,0]),2))
    except Exception:
        return dict(token=token, valid=0, **{k:-1 for k in
            ['ref_self','oracle','mean256','n_above_90','near_ref','rank_ref','dist_ref',
             'near_pred','rank_pred','dist_pred','best_fwd']})

if __name__=='__main__':
    out,n,tp=sys.argv[1],int(sys.argv[2]),sys.argv[3]
    toks=sorted(pickle.load(open(tp,'rb'))); random.Random(0).shuffle(toks); toks=toks[:n]
    print(len(toks),'tokens',flush=True)
    cols=['token','valid','ref_self','oracle','mean256','n_above_90','near_ref','rank_ref',
          'dist_ref','near_pred','rank_pred','dist_pred','best_fwd']
    with Pool(64, initializer=_init, initargs=(tp,)) as pool, open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for i,r in enumerate(pool.imap_unordered(work,toks,chunksize=4)):
            w.writerow(r)
            if (i+1)%500==0: print(f'  {i+1}/{n}',flush=True)
    print('DONE ->',out,flush=True)
