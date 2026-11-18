#!/usr/bin/env python
"""Third round (DEV only): filter hyper-parameters, stack weights and walk-forward coefficient at the adjusted-xG observation."""
import numpy as np, pandas as pd, engine as E, lab as LB
from lab import wf_stack
L = LB.Lab(); D = L.D; PG = E.PG; X = L.X
A1 = E.STACK['a1']; LV = np.log(L.lam_v1)
HIST = ((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017))
def gfilt(y, **kw): return L.filt(y, kw.pop('phi', None), **kw)
def sc(lam): s = L.score(lam); return round(float(s['DEV'] * 1e4), 2), round(float(s['DEV15'] * 1e4), 2)
WG, PHI, DEL = 0.15, 0.7, 0.35
xo = L.xg_obs('np', delta=DEL)
y = L.obs(WG, xo)
g = gfilt(y, phi=PHI)
print('candidate np delta=.35 wg=.15 phi=.7:', sc(L.lam_swap(g)))
print('neighbours:', {(wg, phi, d): sc(L.lam_swap(gfilt(L.obs(wg, L.xg_obs('np', delta=d)), phi=phi)))[0] for wg in [0.05, 0.15, 0.25] for phi in [0.6, 0.7, 0.8] for d in [0.25, 0.35, 0.45]})
print('P0 x', {m: sc(L.lam_swap(gfilt(y, phi=PHI, P0=PG['P0'] * m))) for m in [0.6, 0.8, 1.0, 1.3, 1.7]})
print('rho', {r: sc(L.lam_swap(gfilt(y, phi=PHI, rho=r))) for r in [0.75, 0.82, 0.866, 0.92, 0.97]})
print('q x', {m: sc(L.lam_swap(gfilt(y, phi=PHI, q=PG['q'] * m))) for m in [0.2, 1, 5, 20]})
print('dPromo x', {m: sc(L.lam_swap(gfilt(y, phi=PHI, dPromoO=PG['dPromoO'] * m, dPromoD=PG['dPromoD'] * m))) for m in [0.8, 1.0, 1.2]})
dg = np.log(g[:, :2] / L.g_v1[:, :2])
print('fixed coefficient c on log(g_new/g_v1):', {c: sc(L.lam_v1 * np.exp(c * dg)) for c in [0.6, 0.8, A1, 1.1, 1.3]})
R = LV - E.STACK['a0'] - A1 * np.log(L.g_v1[:, :2])
print('v1 shot-channel weight x:', {k: sc(np.exp(E.STACK['a0'] + A1 * np.log(g[:, :2]) + k * R)) for k in [0.0, 0.5, 0.75, 1.0, 1.25]})
for l2 in [3, 10, 30, 100]:
    lam, co = wf_stack(L, [LV, dg], l2=np.array([1e4, 1e4, l2]), b0=[0, 1, A1], train_mask=HIST, return_coefs=True)
    print(f'WF coefficient (ridge {l2}, prior at v1 a1):', sc(lam), {Y: round(float(b[2]), 2) for Y, b in co.items() if Y in (2015, 2016, 2018, 2020, 2022, 2026)})
# the v1 shot-count proxy adds nothing once xG is in?
print('proxy share', {wp: sc(L.lam_swap(gfilt(L.obs(WG, xo, wp), phi=PHI))) for wp in [0.0, 0.1, 0.2]})
