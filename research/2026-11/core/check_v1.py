"""Sanity check: the numba port reproduces shared/baseline_features.csv and preds_baseline.csv exactly
when fed the production (leaky) parameters from site/data/model.json."""
import json
import numpy as np, pandas as pd
import lib

d = lib.load()
model = json.load(open('/home/user/brasileiraodovitroncio/site/data/model.json'))
P = model['params']; xg = model['xg']; stack = model['stack']; hz = model['hazard']


def obs_g(d, wRed, w, xg):
    gh = np.where(d.evOk, d.g_all[:, 0] - (1 - wRed) * d.g_adv[:, 0], d.hg).astype(float)
    ga = np.where(d.evOk, d.g_all[:, 1] - (1 - wRed) * d.g_adv[:, 1], d.ag).astype(float)
    if xg is not None:
        xh = np.maximum(0.05, xg['b0'] + xg['b1'] * d.st_sot[:, 0] + xg['b2'] * np.maximum(0, d.st_sh[:, 0] - d.st_sot[:, 0]))
        xa = np.maximum(0.05, xg['b0'] + xg['b1'] * d.st_sot[:, 1] + xg['b2'] * np.maximum(0, d.st_sh[:, 1] - d.st_sot[:, 1]))
        gh = np.where(d.statsOk, w * gh + (1 - w) * xh, gh); ga = np.where(d.statsOk, w * ga + (1 - w) * xa, ga)
    return gh, ga


pv = lambda c: np.array([P[c].get(k, 0.0) for k in lib.V1_KEYS])
gh, ga = obs_g(d, P['g']['wRed'], P['g']['w'], xg)
og = lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, gh, ga, d.post, pv('g'), True, d.nteams)
osh = lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, np.nan_to_num(d.st_sh[:, 0]), np.nan_to_num(d.st_sh[:, 1]), d.statsOk, pv('sh'), False, d.nteams)
oso = lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, np.nan_to_num(d.st_sot[:, 0]), np.nan_to_num(d.st_sot[:, 1]), d.statsOk, pv('sot'), False, d.nteams)
B = pd.read_csv(lib.S + 'baseline_features.csv', dtype={'id': str})
assert (B.id.values == d.ids).all()
for nm_, mine, col in [('g_lh', og[:, 0], 'g_lh'), ('g_la', og[:, 1], 'g_la'), ('g_Sh', og[:, 2], 'g_Sh'), ('sh_lh', osh[:, 0], 'sh_lh'), ('sot_la', oso[:, 1], 'sot_la'), ('mu', og[:, 4], 'mu'), ('hfa', og[:, 5], 'hfa')]:
    print(nm_, 'max abs diff', np.abs(mine - B[col].values).max())
lh = np.exp(stack['a0'] + stack['a1'] * np.log(og[:, 0]) + stack['a2'] * oso[:, 6] + stack['a3'] * osh[:, 6])
la = np.exp(stack['a0'] + stack['a1'] * np.log(og[:, 1]) + stack['a2'] * oso[:, 7] + stack['a3'] * osh[:, 7])
print('lh', np.abs(lh - B.lh.values).max(), 'la', np.abs(la - B.la.values).max())
H = dict(goal_bin=hz['goal']['bin'], goal_diff=hz['goal']['diff'], goal_men=hz['goal']['men'], lateTrail=hz['goal']['lateTrail'], norm=hz['goal']['norm'],
         red_bin=hz['red']['bin'], red_diff=hz['red']['diff'], red_away=hz['red']['away'])
goalF, redH, redA = lib.compile_hazard(H)
sc = d.scored
pr = lib.chain_probs(lib.timeline_bins(), goalF, redH, redA, lh[sc], la[sc])
pb = pd.read_csv(lib.S + 'preds_baseline.csv', dtype={'id': str}).set_index('id').loc[d.ids[sc]].to_numpy()
print('preds max abs diff', np.abs(pr[:, :3] - pb).max())
tab = lib.ChainTable(H)
pt = tab(lh[sc], la[sc])
print('table lookup max abs diff', np.abs(pt - pb).max())
lib.dev_report('v1 port (exact chain)', d, pr[:, :3])
lib.dev_report('v1 port (table)', d, pt)
lib.dev_report('v1 lambdas + indep Poisson', d, lib.pois_probs(lh[sc], la[sc], 0.0))
lib.dev_report('v1 lambdas + DC rho=-0.08', d, lib.pois_probs(lh[sc], la[sc], -0.08))
