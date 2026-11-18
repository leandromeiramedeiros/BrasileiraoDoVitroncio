#!/usr/bin/env python
"""Steps 4-5: DEV experiments. Only DEV numbers are printed (hold-out is consulted once, in make_final.py).
Usage: python run_experiments.py [block ...]   blocks: base grid coef state xgvar shots skill elo
Printed: delta RPS vs v1 (x1e-4) on DEV 2013-2021 and on DEV 2015-2021 (seasons with a full year of xG history)."""
import sys, json, os
import numpy as np, pandas as pd
import engine as E, lab as LB
from lab import wf_stack, rel_of, skill_ratings

L = LB.Lab(); D = L.D; PG = E.PG
LOG = os.path.join(LB.HERE, 'work', 'experiments.jsonl')
WG, PHI = 0.3, 0.6          # chosen on DEV in block `grid` (centre of a flat optimum)
A1 = E.STACK['a1']
HIST = ((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017))   # league-seasons with shot data
LV = np.log(L.lam_v1)


def show(name, lam, extra=None):
    s = L.score(lam)
    print(f'{name:66s} dDEV {s["DEV"]*1e4:+7.2f}   dDEV15 {s["DEV15"]*1e4:+7.2f}   (RPS {s["DEVrps"]:.5f})', flush=True)
    rec = {'name': name, **{k: float(v) for k, v in s.items()}}
    if extra: rec.update(extra)
    open(LOG, 'a').write(json.dumps(rec) + '\n')
    return s['DEV'] * 1e4


def gfilt(y, phi=None, q=None):
    return L.filt(y, phi, **({} if q is None else {'q': q}))


def sup_tot(dg):
    s = (dg[:, 0] - dg[:, 1]) / 2; t = (dg[:, 0] + dg[:, 1]) / 2
    return np.column_stack([s, -s]), np.column_stack([t, t])


blocks = sys.argv[1:] or ['base', 'grid', 'coef', 'state', 'xgvar', 'shots', 'skill']

if 'base' in blocks:
    print('--- baseline checks')
    show('v1 (grid chain)', L.lam_v1)
    show('v1 with goals-only observation (no shot proxy), v1 stack', L.lam_swap(gfilt(D.v1_obs(w=1.0))))
    feats = [np.log(L.g_v1[:, :2]), np.where(HIST[:, None], L.rel['sot'], 0), np.where(HIST[:, None], L.rel['sh'], 0)]
    show('v1 features, stack refitted walk-forward (clean)', wf_stack(L, feats))

if 'grid' in blocks:
    print('--- observation swap: y = wg*adj.goals + (1-wg)*xG, v1 stack kept; rows wg, cols phi')
    xo = L.xg_obs('tot')
    res = {}
    for wg in [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2, 0.1, 0.0]:
        for phi in [0.4, 0.5, 0.6, 0.7, 1.0, 1.4]:
            res[(wg, phi)] = L.score(L.lam_swap(gfilt(L.obs(wg, xo), phi)))['DEV'] * 1e4
    print(pd.Series(res).unstack().round(2).to_string())
    show(f'swap wg={WG} phi={PHI}  <- chosen', L.lam_swap(gfilt(L.obs(WG, xo), PHI)))
    print('--- three-way: goals / v1 shot-count proxy / xG (phi=0.6)')
    for wg, wp in [(0.3, 0.1), (0.3, 0.2), (0.2, 0.2), (0.4, 0.3)]:
        show(f'swap wg={wg} proxy={wp} xg={1-wg-wp:.1f}', L.lam_swap(gfilt(L.obs(wg, xo, wp), PHI)))
    print('--- process noise q (multiple of v1)')
    for qm in [0.3, 1.0, 2.5, 4.0]:
        show(f'swap wg={WG} phi={PHI} q x{qm}', L.lam_swap(gfilt(L.obs(WG, xo), PHI, PG['q'] * qm)))

if 'coef' in blocks:
    print('--- how much weight on the change of rating?  lam = lam_v1 * exp(c * log(g_swap/g_v1))')
    g = gfilt(L.obs(WG, L.xg_obs('tot')), PHI); dg = np.log(g[:, :2] / L.g_v1[:, :2]); fs, ft = sup_tot(dg)
    for c in [0.5, 0.7, A1, 1.2, 1.5]: show(f'fixed c={c:.2f}', L.lam_v1 * np.exp(c * dg))
    for cs, ct in [(A1, 0.0), (A1, 0.5), (1.2, 0.5)]: show(f'supremacy c={cs:.2f}, total c={ct}', L.lam_v1 * np.exp(cs * fs + ct * ft))
    print('--- coefficients fitted walk-forward (Poisson regression on earlier seasons, ridge prior at the swap)')
    for l2 in [1, 10, 50]:
        lam, co = wf_stack(L, [LV, dg], l2=np.array([1e4, 1e4, l2]), b0=[0, 1, A1], train_mask=HIST, return_coefs=True)
        show(f'WF single coef, ridge {l2}', lam)
        lam, co = wf_stack(L, [LV, fs, ft], l2=np.array([1e4, 1e4, l2, l2]), b0=[0, 1, A1, A1], train_mask=HIST, return_coefs=True)
        show(f'WF supremacy/total coefs, ridge {l2}', lam)
        if l2 == 10: print('     coefs', {Y: b[2:].round(2).tolist() for Y, b in co.items() if Y in (2015, 2017, 2019, 2021)})
    print('--- separate pure-xG channel next to the v1 goal channel (walk-forward coefficient)')
    for phi in [0.4, 0.6]:
        gx = gfilt(L.obs(0.0, L.xg_obs('tot')), phi); dx = np.log(gx[:, :2] / L.g_v1[:, :2])
        show(f'WF [lam_v1, log(g_xgonly(phi={phi})/g_v1)] ridge 10', wf_stack(L, [LV, dx], l2=np.array([1e4, 1e4, 10]), b0=[0, 1, 0.5], train_mask=HIST))

if 'state' in blocks:
    print('--- what goes into the xG observation (swap, wg/phi fixed)')
    for kind in ['tot', 'np']:
        for wUp in [1.0, PG['wRed'], 0.3]:
            for delta in [0.0, 0.15, 0.3]:
                show(f'{kind:3s} man-up weight={wUp:.2f} score-effect delta={delta}', L.lam_swap(gfilt(L.obs(WG, L.xg_obs(kind, wUp, delta)), PHI)))
    X = L.X
    for name, adj in [('garbage time: xG at |score diff|>=2 x0.5', X['npxg'] - 0.5 * (X['xg_lead2'] + X['xg_trail2'])),
                      ('big chances x1.3 (xG>=0.3)', X['npxg'] + 0.3 * X['xg_big']),
                      ('big chances x0.7 (xG>=0.3)', X['npxg'] - 0.3 * X['xg_big']),
                      ('small chances x0.5 (xG<0.07)', X['npxg'] - 0.5 * X['xg_small'])]:
        raw = adj - (1 - PG['wRed']) * X['xg_manup'] + (X['xg'] - X['npxg'])
        xo = raw * L.kappa(raw)[:, None]
        show(name, L.lam_swap(gfilt(L.obs(WG, xo), PHI)))
    k = L.kappa(X['xg'])
    show('no online calibration of xG level (kappa=1)', L.lam_swap(gfilt(L.obs(WG, L.xg_obs('tot') / k[:, None]), PHI)))

if 'xgvar' in blocks:
    print('--- which shot model feeds the rating (swap, wg/phi fixed)')
    for base in ['xg', 'xg_ns', 'xg_gb', 'xg_sit']:
        show(f'shot model = {base}', L.lam_swap(gfilt(L.obs(WG, L.xg_obs('tot', base=base)), PHI)))
    X = L.X
    ps = X['psxg'] * L.kappa(X['psxg'])[:, None]
    show('post-shot xG instead of xG', L.lam_swap(gfilt(L.obs(WG, ps), PHI)))
    for mix in [0.25, 0.5]:
        show(f'xG with {mix} post-shot xG mixed in', L.lam_swap(gfilt(L.obs(WG, (1 - mix) * L.xg_obs('tot') + mix * ps), PHI)))
    # crude alternatives: every shot worth the same / shots on target worth the same (calibrated online)
    sh = X['shots'] * L.kappa(X['shots'], k0=0.11)[:, None]; so = X['sot'] * L.kappa(X['sot'], k0=0.31)[:, None]
    show('all shots equal (count) instead of xG', L.lam_swap(gfilt(L.obs(WG, sh), PHI)))
    show('shots on target (count) instead of xG', L.lam_swap(gfilt(L.obs(WG, so), PHI)))
    dp = np.column_stack([3 * L.des[:, 0] + L.des[:, 1], 3 * L.des[:, 2] + L.des[:, 1]])
    show('deserved points (shot simulation) rescaled to goals', L.lam_swap(gfilt(L.obs(WG, dp * L.kappa(dp, k0=0.9)[:, None]), PHI)))

if 'shots' in blocks:
    print('--- weight of the v1 shot-count channels once xG is in the goal channel (k multiplies a2*rel_sot + a3*rel_sh)')
    g = gfilt(L.obs(WG, L.xg_obs('tot')), PHI)
    R = LV - E.STACK['a0'] - A1 * np.log(L.g_v1[:, :2])
    for k in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]:
        show(f'shot-channel weight x{k}', np.exp(E.STACK['a0'] + A1 * np.log(g[:, :2]) + k * R))
    lam, co = wf_stack(L, [np.log(L.lam_swap(g)), np.where(HIST[:, None], R, 0)], l2=np.array([1e4, 1e4, 10]), b0=[0, 1, 0], train_mask=HIST, return_coefs=True)
    show('WF extra coefficient on the shot-channel term', lam); print('     coefs', {Y: round(float(b[2]), 2) for Y, b in co.items() if Y >= 2015})

if 'skill' in blocks:
    print('--- finishing / goalkeeping ratings on top of the swap (walk-forward coefficients, main term pinned)')
    X = L.X
    ls = np.log(L.lam_swap(gfilt(L.obs(WG, L.xg_obs('tot')), PHI)))
    def add(name, feats, l2=10.0):
        lam, co = wf_stack(L, [ls] + feats, l2=np.array([1e4, 1e4] + [l2] * len(feats)), b0=[0, 1] + [0] * len(feats), train_mask=HIST, return_coefs=True)
        show(name, lam); print('     coefs', {Y: b[2:].round(2).tolist() for Y, b in co.items() if Y in (2016, 2018, 2020, 2022)})
    for tau, n0 in [(40, 20), (80, 40), (150, 60)]:
        F, A = skill_ratings(L, X['np_goals'], X['npxg'], tau, n0)
        fin = np.column_stack([F[:, 0], F[:, 1]]); con = np.column_stack([A[:, 1], A[:, 0]])   # row team finishing; opponent conceding
        add(f'+ finishing (goals/xG for)       tau={tau} n0={n0}', [fin])
        add(f'+ conceding (goals/xG against)   tau={tau} n0={n0}', [con])
    F, A = skill_ratings(L, X['goals'] - X['og_for'], X['psxg'], 80, 40)
    add('+ keeper (goals conceded / post-shot xG faced) tau=80', [np.column_stack([A[:, 1], A[:, 0]])])
    F, A = skill_ratings(L, X['psxg'], X['xg'], 80, 40)
    add('+ shot placement (post-shot xG / xG for) tau=80', [np.column_stack([F[:, 0], F[:, 1]])])
