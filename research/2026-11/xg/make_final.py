#!/usr/bin/env python
"""Final run: writes preds_*.csv (exact production Markov chain), features_xg_prematch.csv and work/final_scores.json.
This is the only script that looks at the hold-out (2022-2026).

Variants
  xg_swap    v1 with the shot-count proxy replaced by shot-level xG:  y = 0.30*adj.goals + 0.70*k*xG_total,  phi 0.6
  xg_adj     main: y = 0.15*adj.goals + 0.85*k*[npxG - (1-wRed)*npxG_man_up + 0.35*(npxG_leading - npxG_trailing)], phi 0.7
  xg_adj_wf  xg_adj rating; weight of the rating change (supremacy / total) fitted walk-forward by Poisson regression
  xg_nested  observation settings (penalties in/out, delta, wg, phi) re-selected before every season on earlier seasons only
  xg_ll, xg_restack   second consultation, see below
k = online ratio goals/xG of the league (trailing ~400 matches).  Everything else (EKF hyper-parameters, stack, chain) is v1."""
import json, os, subprocess, sys, itertools
import numpy as np, pandas as pd
import engine as E, lab as LB
from lab import wf_stack, skill_ratings, boot_ci

HERE = LB.HERE; WORK = os.path.join(HERE, 'work'); SHARED = E.SHARED
L = LB.Lab(); D = L.D; PG = E.PG; X = L.X
A1 = E.STACK['a1']; LV = np.log(L.lam_v1)
HIST = ((D.lg == 0) & (D.season >= 2014)) | ((D.lg == 1) & (D.season >= 2017))
CFG = {'xg_swap': dict(kind='tot', delta=0.0, wg=0.30, phi=0.6), 'xg_adj': dict(kind='np', delta=0.35, wg=0.15, phi=0.7)}


def gfilt(y, phi):
    return L.filt(y, phi)      # phi only where the observation contains xG; v1 phi elsewhere


def rating(kind, delta, wg, phi):
    return gfilt(L.obs(wg, L.xg_obs(kind, delta=delta)), phi)


lam = {}; G = {}
for nm, c in CFG.items():
    G[nm] = rating(**c); lam[nm] = L.lam_swap(G[nm])
# --- walk-forward coefficients on the rating change
dg = np.log(G['xg_adj'][:, :2] / L.g_v1[:, :2])
s_ = (dg[:, 0] - dg[:, 1]) / 2; t_ = (dg[:, 0] + dg[:, 1]) / 2
lw, coefs = wf_stack(L, [LV, np.column_stack([s_, -s_]), np.column_stack([t_, t_])], l2=np.array([1e4, 1e4, 30, 30]), b0=[0, 1, A1, A1], train_mask=HIST, return_coefs=True)
lam['xg_adj_wf'] = np.where(np.isfinite(lw), lw, L.lam_v1)
# --- second (last) hold-out consultation; both defined on DEV only (explore5-7):
#  xg_ll      settings preferred by the Poisson likelihood of goal counts on DEV (A+B): penalties kept, delta .35, wg .15, phi .7,
#             and only half of the rating change applied (c = 0.5)
#  xg_restack xg_adj rating; slope of the new goal rating and weight of the v1 shot-count term refitted walk-forward
#             (Poisson regression on earlier seasons, ridge 10 centred on the v1 stack)
g_ll = rating('tot', 0.35, 0.15, 0.7); G['xg_ll'] = g_ll
lam['xg_ll'] = L.lam_v1 * np.exp(0.5 * np.log(g_ll[:, :2] / L.g_v1[:, :2]))
A0 = E.STACK['a0']; Rsh = LV - A0 - A1 * np.log(L.g_v1[:, :2])
lr, coefs_rs = wf_stack(L, [np.log(G['xg_adj'][:, :2]), Rsh], l2=np.array([10.0, 10.0, 10.0]), b0=[A0, A1, 1.0], train_mask=HIST, return_coefs=True)
lam['xg_restack'] = np.where(np.isfinite(lr), lr, lam['xg_adj'])
# --- nested selection of the observation settings
grid = [dict(kind=k, delta=d, wg=w, phi=p) for k in ('tot', 'np') for d in (0.0, 0.2, 0.35, 0.5) for w in (0.0, 0.15, 0.3, 0.5, 0.7, 1.0) for p in (0.5, 0.7, 1.0)]
s = D.scored; se = D.season[s]
R = np.empty((len(grid), s.sum())); LG = []
for i, c in enumerate(grid):
    lg_ = L.lam_swap(rating(**c)); LG.append(lg_[s]); R[i] = E.rps(L.probs(lg_[s]), D.hg[s], D.ag[s])
prior = grid.index(dict(kind='tot', delta=0.0, wg=0.5, phi=1.0))     # a-priori default: half goals, half xG, Poisson noise
ln = L.lam_v1.copy(); chosen = {}
for Y in range(2014, 2027):
    past = (se >= 2014) & (se < Y)
    k = prior if past.sum() < 300 else int(np.argmin(R[:, past].mean(1)))
    chosen[Y] = grid[k]
    idx = np.where(s)[0][se == Y]; ln[idx] = LG[k][se == Y]
lam['xg_nested'] = ln
print('nested choices:', {Y: (c['kind'], c['delta'], c['wg'], c['phi']) for Y, c in chosen.items()})

# ---------------- exact probabilities + files ----------------
M = pd.DataFrame({'id': D.id[s], 'season': se, 'hg': D.hg[s], 'ag': D.ag[s]})
base = pd.read_csv(SHARED + 'preds_baseline.csv', dtype={'id': str}).set_index('id').loc[M.id].to_numpy()
odds = pd.read_csv(SHARED + 'odds_benchmark.csv', dtype={'id': str}).set_index('id').reindex(M.id).to_numpy()   # benchmark only
rb = E.rps(base, M.hg.to_numpy(), M.ag.to_numpy()); lb = E.logloss(base, M.hg.to_numpy(), M.ag.to_numpy())
ro = E.rps(np.nan_to_num(odds, nan=1 / 3), M.hg.to_numpy(), M.ag.to_numpy()); hasodds = np.isfinite(odds[:, 0])
subsets = {'DEV 2013-2021': se <= 2021, 'HOLDOUT 2022-2026': se >= 2022, 'TOTAL 2013-2026': se >= 2013,
           'DEV 2015-2021': (se >= 2015) & (se <= 2021), 'TOTAL 2015-2026': se >= 2015}
final = {'nested_choices': {str(Y): c for Y, c in chosen.items()}, 'wf_coefs': {str(Y): [round(float(v), 3) for v in b[2:]] for Y, b in coefs.items()},
         'restack_coefs': {str(Y): [round(float(v), 3) for v in b] for Y, b in coefs_rs.items()}, 'variants': {}}
gy = np.column_stack([M.hg.to_numpy(), M.ag.to_numpy()])
def gnll(lm): return (lm[s] - gy * np.log(lm[s])).mean(1)       # Poisson negative log-likelihood of the two goal counts (up to a constant)
n0 = gnll(L.lam_v1)
for nm, lm in lam.items():
    tmp = os.path.join(WORK, f'lam_{nm}.csv')
    pd.DataFrame({'id': D.id[s], 'lh': lm[s, 0], 'la': lm[s, 1]}).to_csv(tmp, index=False)
    out = os.path.join(HERE, f'preds_{nm}.csv')
    subprocess.run(['node', os.path.join(HERE, 'grid_chain.mjs'), tmp, out], check=True)
    p = pd.read_csv(out, dtype={'id': str}).set_index('id').loc[M.id].to_numpy()
    r = E.rps(p, M.hg.to_numpy(), M.ag.to_numpy()); ll = E.logloss(p, M.hg.to_numpy(), M.ag.to_numpy())
    final['variants'][nm] = {}
    print(f'\n== {nm}')
    for lab_, m in subsets.items():
        d = r[m] - rb[m]; lo, hi = boot_ci(d)
        mo = m & hasodds
        rec = {'n': int(m.sum()), 'rps': float(r[m].mean()), 'll': float(ll[m].mean()), 'base_rps': float(rb[m].mean()), 'drps': float(d.mean()), 'ci': [lo, hi],
               'dll': float((ll[m] - lb[m]).mean()), 'market_rps': float(ro[mo].mean()), 'model_rps_same_matches': float(r[mo].mean()), 'base_rps_same_matches': float(rb[mo].mean()),
               'by_season': {int(y): round(float((r[m & (se == y)] - rb[m & (se == y)]).mean() * 1e4), 2) for y in sorted(set(se[m]))} if lab_.startswith('TOTAL 2013') else None}
        rec['d_goal_nll'] = float((gnll(lm)[m] - n0[m]).mean())
        final['variants'][nm][lab_] = rec
        print(f'  {lab_:18s} n={rec["n"]:4d} RPS {rec["rps"]:.5f} LL {rec["ll"]:.5f} | v1 {rec["base_rps"]:.5f} | dRPS {rec["drps"]:+.5f} [{lo:+.5f},{hi:+.5f}] dLL {rec["dll"]:+.5f} | goal-count NLL vs v1 {rec["d_goal_nll"]*1e4:+.1f}e-4 | market {rec["market_rps"]:.5f} (model, same matches {rec["model_rps_same_matches"]:.5f})')
    print('  dRPS x1e4 by season:', final['variants'][nm]['TOTAL 2013-2026']['by_season'])
# robustness: same expected goals through a plain independent-Poisson score matrix instead of the production chain
from scipy.stats import poisson
def indep(lm):
    k = np.arange(0, 13); ph = poisson.pmf(k[None, :], lm[:, 0:1]); pa_ = poisson.pmf(k[None, :], lm[:, 1:2])
    J = ph[:, :, None] * pa_[:, None, :]
    pH = np.tril(J, -1).sum((1, 2)); pD = np.trace(J, axis1=1, axis2=2); p = np.column_stack([pH, pD, 1 - pH - pD]); return p
r0 = E.rps(indep(L.lam_v1[s]), M.hg.to_numpy(), M.ag.to_numpy())
final['independent_poisson_check'] = {}
for nm, lm in lam.items():
    r1 = E.rps(indep(lm[s]), M.hg.to_numpy(), M.ag.to_numpy())
    final['independent_poisson_check'][nm] = {lab_: float((r1[m] - r0[m]).mean()) for lab_, m in subsets.items()}
print('\nindependent-Poisson link, dRPS x1e4 vs v1 lambdas through the same link:', {nm: {k: round(v * 1e4, 2) for k, v in d.items()} for nm, d in final['independent_poisson_check'].items()})
json.dump(final, open(os.path.join(WORK, 'final_scores.json'), 'w'), indent=1)

# ---------------- pre-match feature file (all Serie A and B matches, including fixtures not yet played) ----------------
ga, gs = G['xg_adj'], G['xg_swap']
gx = rating('np', 0.35, 0.0, 0.7)                 # pure adjusted-xG rating (no goals at all)
n = len(D.M)
F = pd.DataFrame({'id': D.id, 'lg': np.where(D.lg == 0, 'A', 'B'), 'season': D.season, 'date': D.date,
                  'h': [m['h'] for m in D.M], 'a': [m['a'] for m in D.M], 'status': [m['status'] for m in D.M]})
F['xga_lh'] = lam['xg_adj'][:, 0]; F['xga_la'] = lam['xg_adj'][:, 1]            # final expected goals, main variant
F['xgs_lh'] = lam['xg_swap'][:, 0]; F['xgs_la'] = lam['xg_swap'][:, 1]
F['xgw_lh'] = lam['xg_adj_wf'][:, 0]; F['xgw_la'] = lam['xg_adj_wf'][:, 1]      # walk-forward weighted variants (= v1 / swap before 2013)
F['xgr_lh'] = lam['xg_restack'][:, 0]; F['xgr_la'] = lam['xg_restack'][:, 1]
F['xgl_lh'] = lam['xg_ll'][:, 0]; F['xgl_la'] = lam['xg_ll'][:, 1]
F['g_xga_lh'] = ga[:, 0]; F['g_xga_la'] = ga[:, 1]                              # goal-channel expectation with the adjusted-xG observation
F['g_xgonly_lh'] = gx[:, 0]; F['g_xgonly_la'] = gx[:, 1]                        # same with 100% xG
F['d_xga_h'] = dg[:, 0]; F['d_xga_a'] = dg[:, 1]                                # log(g_xga / g_v1): what xG changes relative to v1
F['d_xgonly_h'] = np.log(gx[:, 0] / L.g_v1[:, 0]); F['d_xgonly_a'] = np.log(gx[:, 1] / L.g_v1[:, 1])
F['att_h'] = ga[:, 6]; F['def_h'] = ga[:, 7]; F['att_a'] = ga[:, 8]; F['def_a'] = ga[:, 9]; F['homefac_h'] = ga[:, 10]
F['S_h'] = ga[:, 2]; F['S_a'] = ga[:, 3]
Ff, Aa = skill_ratings(L, X['np_goals'], X['npxg'], 80, 40)
F['fin_h'] = Ff[:, 0]; F['fin_a'] = Ff[:, 1]; F['con_h'] = Aa[:, 0]; F['con_a'] = Aa[:, 1]     # log shrunk goals/xG for and against (non-penalty)
Fk, Ak = skill_ratings(L, X['goals'] - X['og_for'], X['psxg'], 80, 40)
F['gk_h'] = Ak[:, 0]; F['gk_a'] = Ak[:, 1]                                      # log shrunk goals conceded / post-shot xG faced (<0 = good keeper)
Fp, Ap = skill_ratings(L, X['psxg'], X['xg'], 80, 40)
F['place_h'] = Fp[:, 0]; F['place_a'] = Fp[:, 1]                                # log shrunk post-shot xG / xG (shot placement)
# rolling form: exponentially weighted means over the team's previous matches with shot data (decay 0.85 per match)
T = len(D.tids); dec = 0.85
acc = {k: [0.0] * T for k in ('w', 'xf', 'xa', 'xp', 'gf', 'ga', 'n')}
cols = {k: np.full((n, 2), np.nan) for k in ('xf', 'xa', 'xp', 'gdx')}; cnt = np.zeros((n, 2))
xpts = np.column_stack([3 * L.des[:, 0] + L.des[:, 1], 3 * L.des[:, 2] + L.des[:, 1]])
for i in range(n):
    for j, t in enumerate((D.h[i], D.a[i])):
        if acc['w'][t] > 0:
            cols['xf'][i, j] = acc['xf'][t] / acc['w'][t]; cols['xa'][i, j] = acc['xa'][t] / acc['w'][t]; cols['xp'][i, j] = acc['xp'][t] / acc['w'][t]
            cols['gdx'][i, j] = (acc['gf'][t] - acc['ga'][t] - acc['xf'][t] + acc['xa'][t]) / acc['w'][t]
        cnt[i, j] = acc['n'][t]
    if L.has[i]:
        for j, t in enumerate((D.h[i], D.a[i])):
            o = 1 - j
            acc['w'][t] = acc['w'][t] * dec + 1; acc['xf'][t] = acc['xf'][t] * dec + X['npxg'][i, j]; acc['xa'][t] = acc['xa'][t] * dec + X['npxg'][i, o]
            acc['xp'][t] = acc['xp'][t] * dec + xpts[i, j]; acc['gf'][t] = acc['gf'][t] * dec + X['np_goals'][i, j]; acc['ga'][t] = acc['ga'][t] * dec + X['np_goals'][i, o]
            acc['n'][t] += 1
for j, sd in enumerate('ha'):
    F[f'form_npxgf_{sd}'] = cols['xf'][:, j]; F[f'form_npxga_{sd}'] = cols['xa'][:, j]; F[f'form_xpts_{sd}'] = cols['xp'][:, j]
    F[f'form_gd_minus_xgd_{sd}'] = cols['gdx'][:, j]; F[f'n_xg_matches_{sd}'] = cnt[:, j].astype(int)
pa = L.probs(lam['xg_adj'])
F['p_xga_H'] = pa[:, 0]; F['p_xga_D'] = pa[:, 1]; F['p_xga_A'] = pa[:, 2]
pl = L.probs(lam['xg_ll'])                         # recommended variant (grid-interpolated chain, |error| < 5e-5)
F['p_xgl_H'] = pl[:, 0]; F['p_xgl_D'] = pl[:, 1]; F['p_xgl_A'] = pl[:, 2]
num = F.select_dtypes('float').columns; F[num] = F[num].round(5)
F.to_csv(os.path.join(HERE, 'features_xg_prematch.csv'), index=False)
print('\nfeatures_xg_prematch.csv', F.shape, 'rows by status', F.status.value_counts().to_dict())
