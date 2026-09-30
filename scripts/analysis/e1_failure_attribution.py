"""E1 — 실패 귀인: 충돌을 일으킨 agent 가 sparse world 안에 있었는가?

SafeDrive 의 sparse world 는 `select_topk` 가 plan anchor 궤적까지의 최소 L2 거리로
가장 가까운 25 명만 고른다. PDM 시뮬레이터는 at-fault collision 을 일으킨 agent 의
track token 을 알려주므로, 그 agent 가 25 슬롯 안에 들었을지를 같은 규칙으로 재현해
판정할 수 있다.

"실패를 만든 대상이 애초에 월드에 없었다" 가 흔하다면, 병목은 planner 의 추론이
아니라 사람이 정한 선택 규칙이다.
"""
import os, sys, lzma, pickle, csv, math, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/kaist5/data/junseong/SafeDrive')
os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
os.environ.setdefault('NUPLAN_MAPS_ROOT', '/home/kaist5/data/junseong/SafeDrive/dataset/maps')
os.environ.setdefault('OPENSCENE_DATA_ROOT', '/home/kaist5/data/junseong/SafeDrive/dataset')

import numpy as np
from multiprocessing import Pool

MC_ROOT = '/home/kaist5/data/junseong/SafeDrive/exp/metric_cache_navtest'
OUT     = '/home/kaist5/data/junseong/SafeDrive/analysis/e1_attribution.csv'
K_SLOTS = 25                     # num_filtering_instance in the Phase 2 config

_G = {}

def _init(traj_path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer, PDMScorerConfig
    ps = TrajectorySampling(num_poses=40, interval_length=0.1)
    _G['sim'] = PDMSimulator(proposal_sampling=ps)
    _G['scorer'] = PDMScorer(proposal_sampling=ps, config=PDMScorerConfig(
        progress_weight=5.0, ttc_weight=5.0, comfortable_weight=2.0, driving_direction_weight=0.0,
        driving_direction_horizon=1.0, driving_direction_compliance_threshold=2.0,
        driving_direction_violation_threshold=6.0, stopped_speed_threshold=5e-3,
        progress_distance_threshold=5.0))
    _G['ps'] = ps
    _G['traj'] = pickle.load(open(traj_path, 'rb'))
    _G['paths'] = {}
    for dp, _, fn in os.walk(MC_ROOT):
        if 'metric_cache.pkl' in fn:
            _G['paths'][os.path.basename(os.path.dirname(os.path.join(dp, 'x')))] = \
                os.path.join(dp, 'metric_cache.pkl')

def rank_agents(metric_cache, poses):
    """Replicate select_topk: rank every observed agent by min L2 to the plan waypoints.

    Agents are taken at t=0 (the model only ever sees the current frame) and expressed
    in the ego rear-axle frame, which is the frame the planned trajectory lives in.
    """
    e = metric_cache.ego_state.rear_axle
    ch, sh = math.cos(e.heading), math.sin(e.heading)
    wp = poses[:, :2]                                        # (T, 2) ego frame
    ranked = []
    for tok, o in metric_cache.observation.unique_objects.items():
        dx, dy = o.box.center.x - e.x, o.box.center.y - e.y
        ax, ay = dx * ch + dy * sh, -dx * sh + dy * ch        # global -> ego
        d = float(np.min(np.hypot(wp[:, 0] - ax, wp[:, 1] - ay)))
        ranked.append((d, tok, str(o.tracked_object_type).split('.')[-1]))
    ranked.sort()
    return ranked

def work(token):
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score_pairwise
    try:
        poses = np.asarray(_G['traj'][token], dtype=np.float64)
        with lzma.open(_G['paths'][token], 'rb') as f:
            mc = pickle.load(f)
        r = pdm_score_pairwise(metric_cache=mc,
                               model_trajectory=Trajectory(poses, _G['ps_traj']),
                               future_sampling=_G['ps'], simulator=_G['sim'], scorer=_G['scorer'])
        ranked = rank_agents(mc, poses)
        rank_of = {tok: i for i, (_, tok, _) in enumerate(ranked)}
        type_of = {tok: t for _, tok, t in ranked}
        dist_of = {tok: d for d, tok, _ in ranked}

        hits = [t for t in (r.collision_tokens or []) if t in rank_of]
        n_out = sum(1 for t in hits if rank_of[t] >= K_SLOTS)
        worst = max((rank_of[t] for t in hits), default=-1)
        n_raw = len(r.collision_tokens or [])
        return dict(token=token, valid=1, NC=float(r.no_at_fault_collisions), n_raw=n_raw,
                    TTC=float(r.time_to_collision_within_bound), DAC=float(r.drivable_area_compliance),
                    score=float(r.score), n_agents=len(ranked), n_collide=len(hits),
                    n_collide_outside=n_out, worst_rank=worst,
                    collide_types='|'.join(type_of.get(t, '?') for t in hits),
                    collide_ranks='|'.join(str(rank_of[t]) for t in hits),
                    collide_dists='|'.join(f'{dist_of[t]:.1f}' for t in hits))
    except Exception as ex:
        return dict(token=token, valid=0, NC=-1, n_raw=-1, TTC=-1, DAC=-1, score=-1, n_agents=-1,
                    n_collide=-1, n_collide_outside=-1, worst_rank=-1,
                    collide_types=type(ex).__name__, collide_ranks='', collide_dists='')

def _init2(traj_path):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    _init(traj_path)
    _G['ps_traj'] = TrajectorySampling(time_horizon=4, interval_length=0.5)

if __name__ == '__main__':
    traj_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    tokens = list(pickle.load(open(traj_path, 'rb')))
    print(f'{len(tokens)} tokens', flush=True)
    cols = ['token','valid','NC','n_raw','TTC','DAC','score','n_agents','n_collide',
            'n_collide_outside','worst_rank','collide_types','collide_ranks','collide_dists']
    with Pool(48, initializer=_init2, initargs=(traj_path,)) as p, open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for i, r in enumerate(p.imap_unordered(work, tokens, chunksize=8)):
            w.writerow(r)
            if (i + 1) % 2000 == 0: print(f'  {i+1}/{len(tokens)}', flush=True)
    print('DONE ->', out, flush=True)
