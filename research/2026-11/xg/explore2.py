#!/usr/bin/env python
"""Second round of DEV-only tuning: score-effect correction, penalties, concavity, re-tune of wg/phi."""
import numpy as np, pandas as pd, engine as E, lab as LB
L = LB.Lab(); D = L.D; PG = E.PG; X = L.X
def gfilt(y, phi): return L.filt(y, phi)
def sc(lam): s = L.score(lam); return round(s['DEV'] * 1e4, 2), round(s['DEV15'] * 1e4, 2)
def xobs(dL=0.0, dT=0.0, pen=1.0, base='npxg', gam=1.0, wUp=PG['wRed'], lead2=0.0):
    raw = X[base] - (1 - wUp) * X['xg_manup'] + dL * X['xg_lead'] - dT * X['xg_trail'] + lead2 * X['xg_lead2'] + pen * (X['xg'] - X['npxg'])
    raw = np.clip(raw, 0, None)
    if gam != 1.0: raw = raw ** gam
    key = ('x2', dL, dT, pen, base, gam, wUp, lead2)
    if key not in L.kcache: L.kcache[key] = L.kappa(raw)
    return raw * L.kcache[key][:, None]
def run(wg=0.3, phi=0.6, **k): return sc(L.lam_swap(gfilt(L.obs(wg, xobs(**k)), phi)))
print('delta (lead=trail), penalties ignored / included')
for d in [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]: print(f'  delta={d:4.2f}  np {run(dL=d, dT=d, pen=0.0)}   tot {run(dL=d, dT=d, pen=1.0)}   half-pen {run(dL=d, dT=d, pen=0.5)}')
print('separate lead / trail corrections (np)')
for dL in [0, 0.3, 0.6]:
    print('  dL=%.1f ' % dL, {dT: run(dL=dL, dT=dT, pen=0.0) for dT in [0, 0.3, 0.6]})
print('shot model without state features + delta (np)')
for d in [0, 0.3, 0.6, 0.9]: print(f'  xg_ns delta={d}', run(dL=d, dT=d, pen=0.0, base='xg_ns'))
print('extra weight for xG when 2+ goals up (np, delta=.45)'); print({l2: run(dL=.45, dT=.45, pen=0.0, lead2=l2) for l2 in [-0.3, 0, 0.3]})
print('concavity xg^gamma (np, delta=.45)'); print({g: run(dL=.45, dT=.45, pen=0.0, gam=g) for g in [0.6, 0.8, 1.0, 1.2]})
print('re-tune wg x phi at np, delta=.45')
res = {(wg, phi): run(wg, phi, dL=.45, dT=.45, pen=0.0)[0] for wg in [0.0, 0.15, 0.3, 0.45, 0.6] for phi in [0.4, 0.5, 0.6, 0.8, 1.0]}
print(pd.Series(res).unstack().to_string())
