"""완료된 ablation 으로 "상황별로 필요한 정보가 다른가" 를 예비 검증한다.

핵심 질문은 절대 성능이 아니라 **맥락에 따라 어느 정보를 빼는 게 더 아픈지가 바뀌는가** 다.
따라서 각 맥락 버킷에서 ablation 별 ΔPDMS 를 구하고, 버킷 간에 그 순서/크기가
바뀌는지를 본다. 버킷이 작으면 노이즈가 크므로 paired bootstrap 신뢰구간을 함께 낸다.

맥락 정의는 결과를 보기 전에 context_axes.py 로 확정해 둔 입력 기반 축만 쓴다.
모델 성능으로 계층을 나누면 평균 회귀로 허상이 생긴다(과거 실제로 겪었다).
"""
import csv, glob, random, statistics as st, collections

S = os.environ.get('SD_ANALYSIS', '/home/kaist5/data/junseong/SafeDrive/analysis')
B = '/home/kaist5/data/junseong/SafeDrive/exp/safedrive'

ABL = [('F1 미래BEV 제거',      'eval_f1_nofutbev_ev'),
       ('F2 agent궤적 제거',    'eval_f2_nomotionsup_ev'),
       ('E3 월드 25→5',        'eval_e3_final'),
       ('E2 perception 동결',   'eval_e2_final')]

def load(d):
    f = glob.glob(f'{B}/{d}/traj_*.csv')
    if not f: return None
    r = {x['token']: float(x['score']) for x in csv.DictReader(open(f[0])) if x['valid'] == 'True'}
    return r if len(r) > 12000 else None

base = load('eval_test_baseline')
abl = {n: load(d) for n, d in ABL}
abl = {k: v for k, v in abl.items() if v}
ctx = {r['token']: r for r in csv.DictReader(open(f'{S}/context_axes.csv')) if r['valid'] == '1'}

T = [t for t in base if t in ctx and all(t in v for v in abl.values())]

# --- 맥락 축 정의 (입력만 사용, 사전 확정) ---
def f(t, k): return float(ctx[t][k])
def i(t, k): return int(ctx[t][k])
AXES = {
 '회전':     lambda t: '좌회전' if f(t,'dheading') > 20 else ('우회전' if f(t,'dheading') < -20 else '직진'),
 'agent밀도': lambda t: '밀집(8+)' if i(t,'n_agent') >= 8 else ('보통(3-7)' if i(t,'n_agent') >= 3 else '한산(0-2)'),
 '보행자':    lambda t: '있음' if i(t,'n_ped') > 0 else '없음',
 'ego속도':   lambda t: '정지(<1)' if f(t,'speed') < 1 else ('저속(1-5)' if f(t,'speed') < 5 else '주행(5+)'),
 '요구진행량': lambda t: '긺(35+)' if f(t,'fwd') > 35 else ('중간(15-35)' if f(t,'fwd') > 15 else '짧음(<15)'),
 '곡률':     lambda t: '급곡(2+)' if f(t,'bow') >= 2 else ('완곡(0.5-2)' if f(t,'bow') >= 0.5 else '직선(<0.5)'),
 '정적물':    lambda t: '있음' if i(t,'n_static') > 0 else '없음',
}
ORDER = {
 '회전': ['직진','좌회전','우회전'], 'agent밀도': ['한산(0-2)','보통(3-7)','밀집(8+)'],
 '보행자': ['없음','있음'], 'ego속도': ['정지(<1)','저속(1-5)','주행(5+)'],
 '요구진행량': ['짧음(<15)','중간(15-35)','긺(35+)'],
 '곡률': ['직선(<0.5)','완곡(0.5-2)','급곡(2+)'], '정적물': ['없음','있음'],
}

def dboot(toks, model, n=1500):
    d = [abl[model][t] - base[t] for t in toks]
    m = st.mean(d) * 100
    rng = random.Random(0); N = len(d)
    ms = sorted(st.mean(rng.choices(d, k=N)) * 100 for _ in range(n))
    return m, ms[int(.025*n)], ms[int(.975*n)]

print(f'공통 시나리오 {len(T)}\n')
print('값 = ΔPDMS (ablation − baseline). 음수면 그 정보를 빼서 손해.')
print('★ 는 95% CI 가 0 을 포함하지 않는 것.\n')

for ax, fn in AXES.items():
    g = collections.defaultdict(list)
    for t in T: g[fn(t)].append(t)
    buckets = [b for b in ORDER[ax] if len(g[b]) >= 200]
    if len(buckets) < 2: continue
    print(f'── {ax} ' + '─'*(64-len(ax)))
    hdr = f"{'':<20}" + ''.join(f"{b+f'(n={len(g[b])})':>21}" for b in buckets)
    print(hdr)
    for model in abl:
        row = f"{model:<20}"
        for b in buckets:
            m, lo, hi = dboot(g[b], model)
            star = '★' if (lo > 0 or hi < 0) else ' '
            row += f"{m:>+8.2f}{star}[{lo:+5.1f},{hi:+5.1f}]".rjust(21)
        print(row)
    print()
