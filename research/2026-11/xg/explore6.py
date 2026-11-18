#!/usr/bin/env python
"""DEV-only check of the likelihood-tuned candidates before the second (and last) hold-out consultation,
plus calibration of expected goals (why goal likelihood wants a smaller weight on the rating change than 1X2 RPS)."""
import numpy as np, pandas as pd, engine as E, lab as LB
from lab import wf_stack
L = LB.Lab(); D = L.D; PG = E.PG; A1 = E.STACK['a1']; LV = np.log(L.lam_v1)
HIST = ((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017))
def gfilt(y, phi): return L.filt(y, phi)
devA = D.post & (D.lg == 0) & (D.season >= 2015) & (D.season <= 2021)
def nll(lam, m): return (lam[m] - np.column_stack([D.hg[m], D.ag[m]]) * np.log(lam[m])).mean()
def rep(name, lam):
    s = L.score(lam); print(f'{name:44s} dRPS DEV {s["DEV"]*1e4:+7.2f}  DEV15 {s["DEV15"]*1e4:+7.2f} | goal NLL (A 2015-21) vs v1 {(nll(lam, devA)-nll(L.lam_v1, devA))*1e4:+7.2f}')
def st(dg): s = (dg[:, 0] - dg[:, 1]) / 2; t = (dg[:, 0] + dg[:, 1]) / 2; return np.column_stack([s, -s]), np.column_stack([t, t])
g_ll = gfilt(L.obs(0.15, L.xg_obs('tot', delta=0.35)), 0.7); d_ll = np.log(g_ll[:, :2] / L.g_v1[:, :2])
g_ad = gfilt(L.obs(0.15, L.xg_obs('np', delta=0.35)), 0.7); d_ad = np.log(g_ad[:, :2] / L.g_v1[:, :2])
g_sw = gfilt(L.obs(0.30, L.xg_obs('tot', delta=0.0)), 0.6); d_sw = np.log(g_sw[:, :2] / L.g_v1[:, :2])
rep('xg_swap (c=.95)', L.lam_v1 * np.exp(A1 * d_sw)); rep('xg_adj (c=.95)', L.lam_v1 * np.exp(A1 * d_ad))
fs, ft = st(d_ad); rep('xg_adj_wf (ridge 30 @ .95)', wf_stack(L, [LV, fs, ft], l2=np.array([1e4, 1e4, 30, 30]), b0=[0, 1, A1, A1], train_mask=HIST))
rep('xg_ll (tot, delta .35, c=.5)', L.lam_v1 * np.exp(0.5 * d_ll))
fs, ft = st(d_ll); lam, co = wf_stack(L, [LV, fs, ft], l2=np.array([1e4, 1e4, 10, 10]), b0=[0, 1, 0.5, 0.5], train_mask=HIST, return_coefs=True)
rep('xg_ll_wf (ridge 10 @ .5)', lam); print('   coefs', {Y: b[2:].round(2).tolist() for Y, b in co.items() if Y in (2015, 2017, 2019, 2021)})
print('\ncalibration of expected goals on DEV A 2015-2021 (team rows, deciles of lambda): predicted vs actual')
y = np.concatenate([D.hg[devA], D.ag[devA]])
for name, lam in [('v1', L.lam_v1), ('xg_adj c=.95', L.lam_v1 * np.exp(A1 * d_ad)), ('xg_ll c=.5', L.lam_v1 * np.exp(0.5 * d_ll))]:
    l = np.concatenate([lam[devA, 0], lam[devA, 1]]); q = pd.qcut(l, 10, labels=False)
    t = pd.DataFrame({'q': q, 'l': l, 'y': y}).groupby('q').mean()
    slope = np.polyfit(np.log(t.l), np.log(t.y), 1)[0]
    print(f'  {name:14s} sd(log lam) {np.log(l).std():.3f}  slope log(actual) on log(pred) {slope:.3f}  bottom decile {t.l.iloc[0]:.2f}->{t.y.iloc[0]:.2f}  top decile {t.l.iloc[-1]:.2f}->{t.y.iloc[-1]:.2f}')
tot = D.hg[devA] + D.ag[devA]
for name, lam in [('v1', L.lam_v1), ('xg_adj c=.95', L.lam_v1 * np.exp(A1 * d_ad)), ('xg_ll c=.5', L.lam_v1 * np.exp(0.5 * d_ll))]:
    lt = lam[devA].sum(1); sup = lam[devA, 0] - lam[devA, 1]
    print(f'  {name:14s} corr(pred total, actual total) {np.corrcoef(lt, tot)[0,1]:.3f}   corr(pred supremacy, actual goal diff) {np.corrcoef(sup, D.hg[devA]-D.ag[devA])[0,1]:.3f}  sd pred total {lt.std():.3f}')
