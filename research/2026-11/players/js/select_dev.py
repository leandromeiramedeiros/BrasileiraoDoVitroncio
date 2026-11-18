#!/usr/bin/env python
"""DEV-only (2013-2021) selection for the JS lineup features: level vs centred, and which features to keep.
Shift on top of the ctx_core expected goals, coefficients walk-forward (Série A seasons 2009..Y-1, Poisson with offset).
No holdout season is fitted, predicted or scored here."""
import json, subprocess, sys, numpy as np, pandas as pd
from scipy.optimize import minimize
HERE = '/home/user/bdv-lab/players/js/'; S = '/home/user/bdv-lab/shared/'; CTX = '/home/user/bdv-lab/context/'
M = pd.DataFrame(json.load(open(S + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
M = M[(M.lg == 'A') & (M.status == 'post') & (M.season >= 2009) & (M.season <= 2021)].copy(); M['id'] = M['id'].astype(str)
F = pd.read_csv(HERE + 'features_js.csv', dtype={'id': str})
H = F[F.side == 'h'].set_index('id').drop(columns='side'); A = F[F.side == 'a'].set_index('id').drop(columns='side')
lc = pd.read_csv(CTX + 'coef/lambda_ctx_core.csv', dtype={'id': str}).set_index('id')
lv = pd.read_csv(S + 'baseline_features.csv', dtype={'id': str}).set_index('id')[['lh', 'la']]
M = M[M.id.isin(H.index)].reset_index(drop=True)
lam = lv.loc[M.id].to_numpy(float); inc = M.id.isin(lc.index).to_numpy(); lam[inc] = lc.loc[M.id[inc]].to_numpy(float)
ctx = pd.read_csv(CTX + 'preds_ctx_core.csv', dtype={'id': str}).set_index('id')
hg, ag, season = M.hg.to_numpy(float), M.ag.to_numpy(float), M.season.to_numpy()
k = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
dev = (season >= 2013)
P0 = ctx.loc[M.id[dev]].to_numpy(float)


def rps(p, kk):
    p = np.clip(p, 1e-9, 1); p = p / p.sum(1, keepdims=True); y = np.eye(3)[kk]
    return ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2


def nll(b, D, lh, la, hg, ag):
    e = D @ b; mh = lh * np.exp(e); ma = la * np.exp(-e)
    return (mh - hg * np.log(mh)).sum() + (ma - ag * np.log(ma)).sum()


def run(name, feats, apply=None):
    mk = lambda ff: np.column_stack([(H.loc[M.id, f].to_numpy(float) - A.loc[M.id, f].to_numpy(float)) for f in ff])
    mk0 = mk; mk = lambda ff: np.nan_to_num(mk0(ff), nan=0.0)      # a feature missing for either team contributes 0
    D = mk(feats); Da = D if apply is None else mk(apply)      # apply: columns the fitted coefficients are applied to
    lh, la = lam[:, 0].copy(), lam[:, 1].copy(); last = None
    for Y in range(2013, 2022):
        tr = season < Y; te = season == Y
        b = minimize(nll, np.zeros(len(feats)), args=(D[tr], lam[tr, 0], lam[tr, 1], hg[tr], ag[tr]), method='BFGS').x
        e = Da[te] @ b; lh[te] = lam[te, 0] * np.exp(e); la[te] = lam[te, 1] * np.exp(-e); last = b
    pd.DataFrame({'id': M.id[dev], 'lh': lh[dev], 'la': la[dev]}).to_csv(HERE + '_sel_in.csv', index=False)
    subprocess.run(['node', CTX + 'code/probs.mjs', HERE + '_sel_in.csv', HERE + '_sel_out.csv'], check=True)
    P = pd.read_csv(HERE + '_sel_out.csv', dtype={'id': str}).set_index('id').loc[M.id[dev]].to_numpy(float)
    d = rps(P, k[dev]) - rps(P0, k[dev]); rng = np.random.default_rng(7)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    print(f'{name:34s} DEV n={dev.sum()} RPS {rps(P, k[dev]).mean():.5f} vs ctx {rps(P0, k[dev]).mean():.5f}  dRPS {d.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}]  b(2021)=' + ' '.join(f'{f}:{v:.3f}' for f, v in zip(feats, last)))
    return d.mean()


L = ['usage', 'ga', 'exp', 'bench']; C = ['c_usage', 'c_ga', 'c_exp', 'c_bench']
if len(sys.argv) > 1 and sys.argv[1] == 'pre':
    L3 = ['usage', 'ga', 'bench']; E3 = ['e_usage', 'e_ga', 'e_bench']
    run('LEVEL 3 (announced lineup)', L3)
    run('LEVEL 3 coefs on EXPECTED values', L3, apply=E3)       # what the site shows before the lineup is announced
    run('EXPECTED values, own coefs', E3)
    run('centred usage+bench + level ga', ['c_usage', 'c_bench', 'ga'])
    run('centred usage+bench + expected ga', ['c_usage', 'c_bench', 'e_ga'])
    sys.exit(0)
run('LEVEL  4 features', L)
run('CENTRED 4 features', C)
for f in C: run('CENTRED minus ' + f, [x for x in C if x != f])
for f in L: run('LEVEL  minus ' + f, [x for x in L if x != f])
run('CENTRED usage+bench', ['c_usage', 'c_bench'])
run('CENTRED ga+bench', ['c_ga', 'c_bench'])
