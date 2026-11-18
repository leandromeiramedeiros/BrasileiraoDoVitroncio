"""Seed variance and shift-consistency check for the frozen final runs.  usage: python post.py deepsets [gru]"""
import sys, numpy as np, pandas as pd
from nn_lib import Data, load_run, score, rps_np
from chain import grid_lookup_np
data = Data()
sc = data.post & (data.lgB == 0) & (data.season >= 2013); dev = sc & (data.season <= 2021); hold = sc & (data.season >= 2022)
for name in sys.argv[1:]:
    P, LAM = load_run(f'final_{name}', data)
    per = [score(data, P[s], quiet=True) for s in range(len(P))]
    for k in ('DEV', 'HOLD'):
        ds = np.array([x[k]['d'] for x in per])
        print(f'{name:9s} {k:4s} single-seed dRPS vs v1: mean {ds.mean():+.5f} sd {ds.std(ddof=1):.5f} min {ds.min():+.5f} max {ds.max():+.5f}  (5 seeds; seed-avg is what the preds file holds)')
    dl = np.log(LAM / np.stack([data.lh, data.la], 1)[None]).mean(0)
    A = (data.lgB == 0) & (data.season >= 2013)
    ps = grid_lookup_np(data.grid, np.log(data.lh[A]) + dl[A, 0], np.log(data.la[A]) + dl[A, 1]); pm = P.mean(0)[A]
    y = data.y[sc]; i = sc[A]
    print(f'{name:9s} shift file: mean d_log_lh {dl[A, 0].mean():+.4f} (sd {dl[A, 0].std():.4f}), mean d_log_la {dl[A, 1].mean():+.4f} (sd {dl[A, 1].std():.4f}); '
          f'seed-to-seed sd of the shift {np.log(LAM / np.stack([data.lh, data.la], 1)[None])[:, A].std(0).mean():.4f}')
    print(f'{name:9s} v1 chain at shifted lambdas vs seed-averaged probabilities: max |dp| {np.abs(ps - pm).max():.5f}; RPS {rps_np(ps[i], y).mean():.5f} vs {rps_np(pm[i], y).mean():.5f} (all scored)')
    fut = A & ~data.post & (data.season == 2026)
    print(f'{name:9s} unplayed 2026 Série A fixtures with a shift: {int(fut.sum())}')
