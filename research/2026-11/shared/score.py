#!/usr/bin/env python
"""Placar oficial da bancada. Uso:
     /home/user/bdv-lab/venv/bin/python /home/user/bdv-lab/shared/score.py preds_x.csv [preds_y.csv ...]
   Cada CSV: colunas id,pH,pD,pA (uma linha por partida da Série A, 2013+).
   Mostra RPS / log-loss em DEV (2013-2021), HOLDOUT (2022-2026) e TOTAL,
   a diferença pareada contra o baseline (v1) com IC bootstrap de 95%,
   e a régua de mercado nos mesmos jogos."""
import sys, json, numpy as np, pandas as pd
S = '/home/user/bdv-lab/shared/'
M = pd.DataFrame(json.load(open(S + 'matches.json')))[['id', 'lg', 'season', 'date', 'status', 'hg', 'ag']]
M = M[(M.lg == 'A') & (M.season >= 2013) & (M.status == 'post')].copy()
M['id'] = M['id'].astype(str)
M['k'] = np.where(M.hg > M.ag, 0, np.where(M.hg == M.ag, 1, 2))
base = pd.read_csv(S + 'preds_baseline.csv', dtype={'id': str})
odds = pd.read_csv(S + 'odds_benchmark.csv', dtype={'id': str}).rename(columns={'mH': 'pH', 'mD': 'pD', 'mA': 'pA'})

def metrics(df):
    p = df[['pH', 'pD', 'pA']].to_numpy(float); p = np.clip(p, 1e-9, 1); p = p / p.sum(1, keepdims=True)
    k = df['k'].to_numpy(); y = np.eye(3)[k]
    rps = ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2
    ll = -np.log(p[np.arange(len(k)), k])
    return rps, ll

def report(name, pred):
    d = M.merge(pred, on='id', how='inner')
    miss = len(M) - len(d)
    b = M.merge(base, on='id').set_index('id').loc[d['id']].reset_index()
    rps, ll = metrics(d); rb, lb = metrics(b)
    o = d[['id', 'k']].merge(odds, on='id', how='left')
    print(f'\n== {name}  (jogos: {len(d)}, faltando: {miss})')
    rng = np.random.default_rng(7)
    for lab, mask in [('DEV 2013-2021', (d.season <= 2021).to_numpy()), ('HOLDOUT 2022-2026', (d.season >= 2022).to_numpy()), ('TOTAL', np.ones(len(d), bool))]:
        if mask.sum() == 0: continue
        diff = rps[mask] - rb[mask]
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
        om = mask & o['pH'].notna().to_numpy()
        ro, lo = metrics(o[om]) if om.sum() else (np.array([np.nan]), np.array([np.nan]))
        print(f'  {lab:18s} n={mask.sum():4d}  RPS {rps[mask].mean():.5f}  LL {ll[mask].mean():.5f} | baseline RPS {rb[mask].mean():.5f} LL {lb[mask].mean():.5f} | dRPS {diff.mean():+.5f} [IC95 {np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}]  dLL {(ll[mask] - lb[mask]).mean():+.5f} | mercado RPS {ro.mean():.5f} (modelo nos mesmos jogos {rps[om].mean():.5f})')

if __name__ == '__main__':
    if len(sys.argv) < 2: report('baseline', base)
    for f in sys.argv[1:]: report(f, pd.read_csv(f, dtype={'id': str}))
