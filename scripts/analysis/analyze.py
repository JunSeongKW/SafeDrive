"""Consolidated baseline-vs-alpha comparison on navtest, with scenario stratification.

Joins the two EPDMS result CSVs against the pedestrian labels built from the
metric cache, so the same table answers both "did alpha win overall" and
"did it win where pedestrians actually are".
"""
import csv, sys, statistics as st

S = os.environ.get('SD_ANALYSIS', '/home/kaist5/data/junseong/SafeDrive/analysis')
COLS = [('no_at_fault_collisions','NC'), ('drivable_area_compliance','DAC'),
        ('time_to_collision_within_bound','TTC'), ('ego_progress','EP'),
        ('comfort','Comf'), ('driving_direction_compliance','DDC'),
        ('traffic_light_compliance','TLC'), ('lane_keeping','LK'), ('score','PDMS')]

def load(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r.get('valid','True') not in ('True','true','1'):
            continue
        out[r['token']] = r
    return out

def agg(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None,'','nan')]
    return st.mean(v)*100 if v else float('nan')

def table(name, base, alpha, keys):
    rows_b = [base[k] for k in keys if k in base]
    rows_a = [alpha[k] for k in keys if k in alpha]
    n = min(len(rows_b), len(rows_a))
    print(f'\n### {name}  (n={n})')
    print(f'{"지표":<6}{"baseline":>10}{"alpha":>10}{"Δ":>9}')
    for col, lab in COLS:
        if col not in rows_b[0]:
            continue
        b, a = agg(rows_b, col), agg(rows_a, col)
        print(f'{lab:<6}{b:>10.2f}{a:>10.2f}{a-b:>+9.2f}')
    return n

if __name__ == '__main__':
    base, alpha = load(sys.argv[1]), load(sys.argv[2])
    lab = {r['token']: r for r in csv.DictReader(open(f'{S}/navtest_labels.csv'))}
    common = [t for t in base if t in alpha]
    print(f'baseline {len(base)} / alpha {len(alpha)} / 공통 {len(common)} / 라벨 {len(lab)}')

    table('전체 navtest', base, alpha, common)
    for thr, name in [(20,'20m'), (30,'30m')]:
        col = f'ped{thr}'
        yes = [t for t in common if t in lab and int(lab[t][col]) > 0]
        no  = [t for t in common if t in lab and int(lab[t][col]) == 0]
        table(f'보행자 있음 (≤{name})', base, alpha, yes)
        table(f'보행자 없음 (>{name})', base, alpha, no)
    # density buckets on the 20 m radius
    for lo, hi, name in [(1,1,'보행자 1명'), (2,3,'보행자 2-3명'), (4,999,'보행자 4명 이상')]:
        sel = [t for t in common if t in lab and lo <= int(lab[t]['ped20']) <= hi]
        if sel: table(name, base, alpha, sel)
