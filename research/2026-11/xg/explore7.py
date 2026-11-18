#!/usr/bin/env python
"""DEV only: 're-stack' = refit the slope of the new goal rating (and the weight of the v1 shot-count term) walk-forward
by Poisson regression, ridge prior centred on the v1 stack. Fixes the over-dispersion of the xG-based expected goals."""
import numpy as np, pandas as pd, engine as E, lab as LB
from lab import wf_stack
L = LB.Lab(); D = L.D; PG = E.PG; A1 = E.STACK['a1']; A0 = E.STACK['a0']; LV = np.log(L.lam_v1)
HIST = ((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017))
R = LV - A0 - A1 * np.log(L.g_v1[:, :2])                 # v1 shot-count term a2*rel_sot + a3*rel_sh
def gfilt(y, phi): return L.filt(y, phi)
devA = D.post & (D.lg == 0) & (D.season >= 2015) & (D.season <= 2021)
def nll(lam, m): return (lam[m] - np.column_stack([D.hg[m], D.ag[m]]) * np.log(lam[m])).mean()
def rep(name, lam):
    s = L.score(lam); print(f'{name:58s} dRPS DEV {s["DEV"]*1e4:+7.2f}  DEV15 {s["DEV15"]*1e4:+7.2f} | goal NLL vs v1 {(nll(lam, devA)-nll(L.lam_v1, devA))*1e4:+7.2f}')
for nm, kind, delta, wg, phi in [('np  d.35 wg.15 phi.7', 'np', 0.35, 0.15, 0.7), ('tot d.35 wg.15 phi.7', 'tot', 0.35, 0.15, 0.7), ('tot d0   wg.30 phi.6', 'tot', 0.0, 0.3, 0.6), ('np  d.35 wg.30 phi.7', 'np', 0.35, 0.3, 0.7)]:
    g = gfilt(L.obs(wg, L.xg_obs(kind, delta=delta)), phi); lg = np.log(g[:, :2])
    rep(f'{nm}: swap', np.exp(A0 + A1 * lg + R))
    for l2 in [3, 10, 30]:
        lam, co = wf_stack(L, [lg, R], l2=np.array([l2, l2, l2]), b0=[A0, A1, 1.0], train_mask=HIST, return_coefs=True)
        rep(f'{nm}: restack ridge {l2}', lam)
        if l2 == 10: print('      coefs [a0, slope, shot-term]', {Y: b.round(2).tolist() for Y, b in co.items() if Y in (2015, 2017, 2019, 2021)})
    lam = wf_stack(L, [lg, R], l2=np.array([10, 10, 10]), b0=[A0, A1, 1.0], train_mask=HIST, leagues=(0,))
    rep(f'{nm}: restack ridge 10, Serie A rows only', lam)
