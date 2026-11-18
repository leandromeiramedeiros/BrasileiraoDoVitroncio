"""Paired comparison of prediction files (same arithmetic and bootstrap as shared/score.py).
usage: python compare.py CANDIDATE.csv REF1.csv [REF2.csv ...]   -> dRPS = RPS(candidate) - RPS(ref), DEV / HOLDOUT / TOTAL"""
import sys, json, numpy as np, pandas as pd
S = '/home/user/bdv-lab/shared/'
M = pd.DataFrame(json.load(open(S + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
M = M[(M.lg == 'A') & (M.season >= 2013) & (M.status == 'post')].copy(); M['id'] = M['id'].astype(str)
M['k'] = np.where(M.hg > M.ag, 0, np.where(M.hg == M.ag, 1, 2))

def metrics(f):
    d = pd.read_csv(f, dtype={'id': str}).drop_duplicates('id').set_index('id').reindex(M['id'])
    p = d[['pH', 'pD', 'pA']].to_numpy(float); assert not np.isnan(p).any(), f + ': missing matches'
    p = np.clip(p, 1e-9, 1); p = p / p.sum(1, keepdims=True); k = M['k'].to_numpy(); y = np.eye(3)[k]
    return ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2, -np.log(p[np.arange(len(k)), k])

cand = sys.argv[1]; rc, lc = metrics(cand); seas = M['season'].to_numpy()
for ref in sys.argv[2:]:
    rr, lr = metrics(ref); rng = np.random.default_rng(7)
    print(f'{cand.split("/")[-1]}  vs  {ref.split("/")[-1]}')
    for lab, mk in [('DEV 2013-2021', seas <= 2021), ('HOLDOUT 2022-2026', seas >= 2022), ('TOTAL', seas > 0)]:
        d = rc[mk] - rr[mk]; bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
        print(f'  {lab:18s} n={mk.sum():4d}  RPS {rc[mk].mean():.5f} vs {rr[mk].mean():.5f} | dRPS {d.mean():+.5f} [IC95 {np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}]  dLL {(lc[mk] - lr[mk]).mean():+.5f}')
