"""EP 손실이 월드의 공간 범위(전방 64 m)에 묶여 있는지 확인한다.

TGDA 는 궤적 waypoint 를 grid_config x:[0,64], y:[-32,32] 로 정규화해 BEV 를 샘플하므로
그 밖은 참조할 수 없다.  PDM 기준 궤적이 4 초 동안 64 m 넘게 나가는 장면이라면 planner 는
목표 지점을 볼 수 없는 상태로 계획하는 셈이다.
"""
import os, sys, lzma, pickle, csv, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/kaist5/data/junseong/SafeDrive')
os.environ.setdefault('NUPLAN_MAP_VERSION','nuplan-maps-v1.0')
os.environ.setdefault('NUPLAN_MAPS_ROOT','/home/kaist5/data/junseong/SafeDrive/dataset/maps')
os.environ.setdefault('OPENSCENE_DATA_ROOT','/home/kaist5/data/junseong/SafeDrive/dataset')
import numpy as np
from multiprocessing import Pool

ROOT='/home/kaist5/data/junseong/SafeDrive/exp/metric_cache_navtest'
_G={}

def _init(traj_path):
    _G['pred']=pickle.load(open(traj_path,'rb'))
    _G['paths']={}
    for dp,_,fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn: _G['paths'][os.path.basename(dp)]=os.path.join(dp,'metric_cache.pkl')

def work(token):
    try:
        with lzma.open(_G['paths'][token],'rb') as f: mc=pickle.load(f)
        e=mc.ego_state
        v=float(np.hypot(e.dynamic_car_state.rear_axle_velocity_2d.x,
                         e.dynamic_car_state.rear_axle_velocity_2d.y))
        # PDM 기준 궤적(사람이 아니라 시뮬레이터 기준)의 4초 변위
        rx, ry, rh = e.rear_axle.x, e.rear_axle.y, e.rear_axle.heading
        ch, sh = math.cos(rh), math.sin(rh)
        pdm = mc.trajectory.get_sampled_trajectory()
        last = pdm[-1].rear_axle
        dx, dy = last.x-rx, last.y-ry
        ref_x = dx*ch + dy*sh                       # 전방 변위 (ego frame)
        ref_d = float(np.hypot(dx,dy))
        p = np.asarray(_G['pred'][token],dtype=np.float64)
        pred_x = float(p[-1,0]); pred_d=float(np.hypot(p[-1,0],p[-1,1]))
        return dict(token=token, valid=1, ego_speed=round(v,2),
                    ref_fwd=round(ref_x,2), ref_dist=round(ref_d,2),
                    pred_fwd=round(pred_x,2), pred_dist=round(pred_d,2),
                    ref_beyond64=int(ref_x>64), ref_beyond32=int(ref_x>32))
    except Exception as ex:
        return dict(token=token, valid=0, ego_speed=-1, ref_fwd=-1, ref_dist=-1,
                    pred_fwd=-1, pred_dist=-1, ref_beyond64=-1, ref_beyond32=-1)

if __name__=='__main__':
    tp=sys.argv[1]; out=sys.argv[2]
    toks=list(pickle.load(open(tp,'rb')))
    print(len(toks),'tokens',flush=True)
    cols=['token','valid','ego_speed','ref_fwd','ref_dist','pred_fwd','pred_dist','ref_beyond64','ref_beyond32']
    with Pool(48,initializer=_init,initargs=(tp,)) as pool, open(out,'w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=cols); w.writeheader()
        for i,r in enumerate(pool.imap_unordered(work,toks,chunksize=8)):
            w.writerow(r)
            if (i+1)%4000==0: print(f'  {i+1}/{len(toks)}',flush=True)
    print('DONE ->',out,flush=True)
