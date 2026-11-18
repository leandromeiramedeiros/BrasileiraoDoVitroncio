"""Paired dRPS of preds_coach.csv vs context/preds_ctx_core.csv on DEV and HOLDOUT (final, single consultation)
   plus early/late-season split. Usage: python code/compare.py [preds.csv]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/bdv-lab/context/code'); import fitlib as F
p = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else '/home/user/bdv-lab/coach/preds_coach.csv', dtype={'id': str})
c = pd.read_csv('/home/user/bdv-lab/context/preds_ctx_core.csv', dtype={'id': str})
X = pd.read_csv('/home/user/bdv-lab/coach/features_coach.csv', dtype={'id': str})[['id', 'h_mseason', 'a_mseason', 'h_new1', 'a_new1', 'h_interim', 'a_interim']]
M, _ = F._truth()
d = M.merge(p, on='id').merge(X, on='id'); b = M.merge(c, on='id').set_index('id').loc[d['id']].reset_index()
r, l = F.rps_ll(d); rb, lb = F.rps_ll(b); diff = r - rb
rng = np.random.default_rng(7)
aff = ((d.h_new1.fillna(0) + d.a_new1.fillna(0) + d.h_interim.fillna(0) + d.a_interim.fillna(0)) > 0).to_numpy()
late = (np.minimum(d.h_mseason, d.a_mseason) >= 12).to_numpy()
for lab, m0 in [('DEV 2013-2021', (d.season <= 2021).to_numpy()), ('HOLDOUT 2022-2026', (d.season >= 2022).to_numpy())]:
    for sub, m1 in [('all', np.ones(len(d), bool)), ('early (<12 played)', ~late), ('late (>=12 played)', late), ('affected matches', aff)]:
        m = m0 & m1; x = diff[m]
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
        print(f'{lab:18s} {sub:20s} n={m.sum():4d} RPS {r[m].mean():.5f} ctx_core {rb[m].mean():.5f} dRPS {x.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] dLL {(l[m] - lb[m]).mean():+.5f}')
