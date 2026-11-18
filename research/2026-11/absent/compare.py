"""Paired dRPS (pred − ref) with bootstrap 95% CI on DEV 2013–2021 and HOLDOUT 2022–2026 (the single final consultation).
   python compare.py pred.csv ref.csv [pred2.csv ref2.csv ...]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/bdv-lab/context/code')
import fitlib as F
M, _ = F._truth()
for i in range(1, len(sys.argv), 2):
    p, r = (pd.read_csv(x, dtype={'id': str}) for x in sys.argv[i:i + 2])
    d = M.merge(p, on='id').merge(r, on='id', suffixes=('', '_r'))
    a, _ = F.rps_ll(d); b, _ = F.rps_ll(d[['k', 'pH_r', 'pD_r', 'pA_r']].rename(columns=lambda c: c.replace('_r', '')))
    print(f'{sys.argv[i]} vs {sys.argv[i + 1]}')
    for lab, m in [('DEV 2013-2021', d.season <= 2021), ('HOLDOUT 2022-2026', d.season >= 2022), ('TOTAL', d.season > 0)]:
        x = (a - b)[m.to_numpy()]; rng = np.random.default_rng(7)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        by = pd.Series(x).groupby(d.season[m].to_numpy()).mean()
        print(f'  {lab:18s} n={len(x)} RPS {a[m.to_numpy()].mean():.5f} ref {b[m.to_numpy()].mean():.5f} dRPS {x.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] seasons better {(by < 0).sum()}/{len(by)}')
