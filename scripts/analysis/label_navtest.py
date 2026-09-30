"""Scenario stratification labels for navtest, read from the metric cache.

For every scenario we record how many pedestrians / vehicles sit within a few
radii of the ego rear axle at t=0, plus the nearest pedestrian distance and the
scenario_type folder name.  This lets the baseline-vs-alpha comparison be split
by "does this scene actually contain a pedestrian" without re-running anything.
"""
import sys, os, lzma, pickle, csv, math
sys.path.insert(0, '/home/kaist5/data/junseong/SafeDrive')
from multiprocessing import Pool

ROOT = '/home/kaist5/data/junseong/SafeDrive/exp/metric_cache_navtest'
OUT  = '/home/kaist5/data/junseong/SafeDrive/analysis/navtest_labels.csv'

def scan(path):
    parts = path.split(os.sep)
    token, stype, log = parts[-2], parts[-3], parts[-4]
    try:
        with lzma.open(path, 'rb') as f:
            mc = pickle.load(f)
        ex, ey = mc.ego_state.rear_axle.x, mc.ego_state.rear_axle.y
        ped, veh, bic = [], [], []
        for o in mc.observation.unique_objects.values():
            d = math.hypot(o.box.center.x - ex, o.box.center.y - ey)
            t = str(o.tracked_object_type)
            if t == 'TrackedObjectType.PEDESTRIAN':  ped.append(d)
            elif t == 'TrackedObjectType.VEHICLE':   veh.append(d)
            elif t == 'TrackedObjectType.BICYCLE':   bic.append(d)
        return (token, log, stype,
                sum(d <= 20 for d in ped), sum(d <= 30 for d in ped), sum(d <= 50 for d in ped),
                sum(d <= 20 for d in veh), sum(d <= 50 for d in veh),
                sum(d <= 20 for d in bic),
                round(min(ped), 2) if ped else -1.0,
                len(mc.observation.unique_objects))
    except Exception as e:
        return (token, log, stype, -1, -1, -1, -1, -1, -1, -1.0, -1)

if __name__ == '__main__':
    files = []
    for dp, dn, fn in os.walk(ROOT):
        if 'metric_cache.pkl' in fn:
            files.append(os.path.join(dp, 'metric_cache.pkl'))
    print(f'{len(files)} metric caches', flush=True)
    with Pool(48) as p, open(OUT, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['token','log','scenario_type','ped20','ped30','ped50',
                    'veh20','veh50','bic20','ped_min_dist','n_objects'])
        for i, r in enumerate(p.imap_unordered(scan, files, chunksize=16)):
            w.writerow(r)
            if (i + 1) % 2000 == 0:
                print(f'  {i+1}/{len(files)}', flush=True)
    print('DONE ->', OUT, flush=True)
