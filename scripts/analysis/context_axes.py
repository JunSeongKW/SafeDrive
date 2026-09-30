"""주행 맥락 축을 metric cache 에서 추출한다 (방향 3번: "상황" 정의 사전 확정).

원칙 네 가지를 지킨다.
 1) 입력에서만 정의한다 — 모델 성능으로 계층을 나누면 평균 회귀로 무효가 된다
    (E2 분석에서 baseline 점수 3분위가 +4.93 이라는 허상을 만든 적이 있다).
 2) 버킷이 균형 있어야 한다 — 95/5 로 쪼개지면 비교가 무의미하다.
 3) 리뷰어가 알아볼 수 있는 범주여야 한다.
 4) 축끼리 상관이 낮아야 한다 — 같은 축을 이름만 바꿔 부르는 것이면 독립 증거가 아니다.
"""
import os, sys, lzma, pickle, csv, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/kaist5/data/junseong/SafeDrive')
os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/home/kaist5/data/junseong/SafeDrive/dataset/maps')
os.environ.setdefault('OPENSCENE_DATA_ROOT', '/home/kaist5/data/junseong/SafeDrive/dataset')
import numpy as np
from multiprocessing import Pool

ROOT = '/home/kaist5/data/junseong/SafeDrive/exp/metric_cache_navtest'
_G = {}

def _init():
    _G['paths'] = {}
    for dp, _, fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn:
            _G['paths'][os.path.basename(dp)] = os.path.join(dp, 'metric_cache.pkl')

def work(token):
    try:
        with lzma.open(_G['paths'][token], 'rb') as f:
            mc = pickle.load(f)
        e = mc.ego_state
        rx, ry, rh = e.rear_axle.x, e.rear_axle.y, e.rear_axle.heading
        ch, sh = math.cos(rh), math.sin(rh)
        speed = float(np.hypot(e.dynamic_car_state.rear_axle_velocity_2d.x,
                               e.dynamic_car_state.rear_axle_velocity_2d.y))
        # PDM 기준 궤적: 4초 변위와 방향 전환량
        pdm = mc.trajectory.get_sampled_trajectory()
        last = pdm[-1].rear_axle
        dx, dy = last.x - rx, last.y - ry
        fwd = dx * ch + dy * sh
        lat = -dx * sh + dy * ch
        dheading = math.degrees(math.atan2(math.sin(last.heading - rh),
                                           math.cos(last.heading - rh)))
        # 경로 곡률: 중간 지점이 시작-끝 직선에서 얼마나 벗어나는가
        mid = pdm[len(pdm) // 2].rear_axle
        mx, my = mid.x - rx, mid.y - ry
        L = math.hypot(dx, dy)
        bow = abs(mx * dy - my * dx) / L if L > 1e-3 else 0.0
        # 주변 객체
        n_veh = n_ped = n_static = 0
        for o in mc.observation.unique_objects.values():
            d = math.hypot(o.box.center.x - rx, o.box.center.y - ry)
            if d > 20: continue
            t = str(o.tracked_object_type).split('.')[-1]
            if t == 'VEHICLE': n_veh += 1
            elif t == 'PEDESTRIAN': n_ped += 1
            else: n_static += 1
        return dict(token=token, valid=1,
                    speed=round(speed, 2), fwd=round(fwd, 2), lat=round(lat, 2),
                    dheading=round(dheading, 1), bow=round(bow, 2),
                    n_veh=n_veh, n_ped=n_ped, n_static=n_static,
                    n_agent=n_veh + n_ped)
    except Exception as ex:
        return dict(token=token, valid=0, speed=-1, fwd=-1, lat=-1, dheading=-999,
                    bow=-1, n_veh=-1, n_ped=-1, n_static=-1, n_agent=-1)

if __name__ == '__main__':
    out = sys.argv[1]
    toks = sorted(_ for _ in os.listdir(ROOT))  # placeholder, replaced below
    # navtest 평가에 쓰인 토큰 목록을 그대로 쓴다
    import glob
    ev = glob.glob('/home/kaist5/data/junseong/SafeDrive/exp/safedrive/eval_test_baseline/traj_*.csv')[0]
    toks = [r['token'] for r in csv.DictReader(open(ev)) if r['valid'] == 'True']
    print(f'{len(toks)} tokens', flush=True)
    cols = ['token','valid','speed','fwd','lat','dheading','bow',
            'n_veh','n_ped','n_static','n_agent']
    with Pool(32, initializer=_init) as p, open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for i, r in enumerate(p.imap_unordered(work, toks, chunksize=8)):
            w.writerow(r)
            if (i + 1) % 3000 == 0: print(f'  {i+1}/{len(toks)}', flush=True)
    print('DONE ->', out, flush=True)
