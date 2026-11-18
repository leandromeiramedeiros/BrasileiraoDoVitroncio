#!/usr/bin/env python
"""Fifth round (DEV only): tune the observation with the Poisson likelihood of GOAL COUNTS (Serie A 2014-2021 + Serie B
2017-2021, ~2x the matches and two counts per match) instead of the noisier 1X2 RPS of Serie A."""
import numpy as np, pandas as pd, engine as E, lab as LB
L = LB.Lab(); D = L.D; PG = E.PG
A1 = E.STACK['a1']
def gfilt(y, phi): return L.filt(y, phi)
dev = D.post & (D.season <= 2021) & (((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017)))
devA = dev & (D.lg == 0); devB = dev & (D.lg == 1)
def dev_ll(lam, m):   # mean Poisson deviance-like loss per team row (lower = better), relative to v1
    def nll(l): return (l[m] - np.column_stack([D.hg[m], D.ag[m]]) * np.log(l[m])).mean()
    return (nll(lam) - nll(L.lam_v1)) * 1e4
rows = []
for kind in ['tot', 'np']:
    for delta in [0.0, 0.2, 0.35, 0.5]:
        xo = L.xg_obs(kind, delta=delta)
        for wg in [0.0, 0.15, 0.3, 0.45, 0.6, 0.8]:
            for phi in [0.5, 0.7, 1.0]:
                g = gfilt(L.obs(wg, xo), phi); dg = np.log(g[:, :2] / L.g_v1[:, :2])
                for c in [0.5, 0.75, A1]:
                    lam = L.lam_v1 * np.exp(c * dg)
                    rows.append({'kind': kind, 'delta': delta, 'wg': wg, 'phi': phi, 'c': round(c, 2), 'nllA': dev_ll(lam, devA), 'nllB': dev_ll(lam, devB), 'nllAB': dev_ll(lam, dev),
                                 'rps': L.score(lam)['DEV'] * 1e4})
R = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print('best 12 by goal log-likelihood (A+B):'); print(R.sort_values('nllAB').head(12).round(2).to_string(index=False))
print('best 8 by 1X2 RPS (A):'); print(R.sort_values('rps').head(8).round(2).to_string(index=False))
print('corr(nllAB, rps) over the grid:', round(float(np.corrcoef(R.nllAB, R.rps)[0, 1]), 3), ' corr(nllA, nllB):', round(float(np.corrcoef(R.nllA, R.nllB)[0, 1]), 3))
for col in ['kind', 'delta', 'wg', 'phi', 'c']:
    print(col, R.groupby(col)[['nllAB', 'nllA', 'nllB', 'rps']].min().round(2).to_dict('index'))
R.to_csv('work/explore5_grid.csv', index=False)
