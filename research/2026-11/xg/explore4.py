#!/usr/bin/env python
"""Fourth round (DEV only): small ideas on top of the adjusted-xG observation (np, delta=.35, wg=.15, phi=.7)."""
import numpy as np, pandas as pd, engine as E, lab as LB
L = LB.Lab(); D = L.D; PG = E.PG; X = L.X
def gfilt(y, **kw): return L.filt(y, kw.pop('phi', None), **kw)
def sc(lam): s = L.score(lam); return round(float(s['DEV'] * 1e4), 2), round(float(s['DEV15'] * 1e4), 2)
WG, PHI, DEL = 0.15, 0.7, 0.35
def cal(raw, k0=1.03): return raw * L.kappa(raw, k0=k0)[:, None]
base_raw = X['npxg'] - (1 - PG['wRed']) * X['xg_manup'] + DEL * (X['xg_lead'] - X['xg_trail'])
ref = cal(base_raw)
print('reference', sc(L.lam_swap(gfilt(L.obs(WG, ref), phi=PHI))))
# 1. xG only for Serie A (Serie B keeps the v1 observation)
has = L.has.copy(); L.has = has & (D.lg == 0)
print('xG for Serie A only (Serie B on v1 proxy)', sc(L.lam_swap(gfilt(L.obs(WG, ref), phi=PHI)))); L.has = has
# 2. deserved points mixed in
dp = np.column_stack([3 * L.des[:, 0] + L.des[:, 1], 3 * L.des[:, 2] + L.des[:, 1]]); dpc = cal(dp, 0.9)
for m in [0.25, 0.5, 1.0]: print(f'deserved-points share {m}', sc(L.lam_swap(gfilt(L.obs(WG, (1 - m) * ref + m * dpc), phi=PHI))))
# 3. late xG / first-half xG weights
for nm, adj in [('late (75+) x0.7', -0.3 * X['xg_late']), ('late (75+) x1.3', 0.3 * X['xg_late']), ('first half x1.2', 0.2 * X['xg_1h']), ('first half x0.8', -0.2 * X['xg_1h']),
                ('rebounds (<=5s) x0.5', -0.5 * X['xg_reb']), ('rebounds x0', -1.0 * X['xg_reb']), ('headers x0.7', -0.3 * X['xg_head']), ('headers x1.3', 0.3 * X['xg_head']),
                ('big chances x1.2', 0.2 * X['xg_big']), ('big chances x0.8', -0.2 * X['xg_big']), ('small chances (<0.07) x0.6', -0.4 * X['xg_small']), ('small chances x1.4', 0.4 * X['xg_small']),
                ('flagged set-piece xG x0.7', -0.3 * X['xg_setp']), ('fast-break xG x1.3', 0.3 * X['xg_fast'])]:
    print(f'{nm:32s}', sc(L.lam_swap(gfilt(L.obs(WG, cal(np.clip(base_raw + adj, 0, None))), phi=PHI))))
# 4. score-effect correction applied to the goal part too (goals when leading count more?) -> via blend only, skip.
# 5. xG level calibration window
for tau in [150, 400, 1000]:
    k = L.kappa(base_raw, tau=tau); print(f'calibration window tau={tau}', sc(L.lam_swap(gfilt(L.obs(WG, base_raw * k[:, None]), phi=PHI))))
