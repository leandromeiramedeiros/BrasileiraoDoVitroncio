#!/usr/bin/env python
"""Final variants (frozen after DEV exploration). Writes preds_<name>.csv for all 5,279 scored matches.

  python final_models.py            -> writes prediction files, prints DEV numbers only
  python final_models.py holdout    -> also prints HOLDOUT / ALL (use sparingly)

Variants
  known_xi          lineup-known, starting XI only          : k_usage_in, k_gashare_in, k_exp
  known_squad       lineup-known, XI + announced bench      : + k_bench_usage
  known_squad_plus  same + XI's prior shots per 90          : + k_sh90
  unknown           lineup-unknown, two-stage               : history-only predictions of the known_squad inputs,
                                                              plugged into the known_squad coefficients
  unknown_plus      same for known_squad_plus
  unknown_direct    lineup-unknown, coefficients fitted directly on history-only columns
Each "known"/"unknown" model is the average of an ordered-logit shift of the v1 probabilities and (for known_*) a Poisson
regression with the v1 expected goals as offset; all coefficients come from seasons strictly before the predicted one.
Fallbacks: known_* -> unknown when the match has no roster; unknown -> v1 baseline when the teams have no usable history.
"""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab

D = E.D; K = E.K_AVAIL; U = E.U_AVAIL; d = E.d
HOLD = len(sys.argv) > 1 and sys.argv[1] == 'holdout'

XI = ['k_usage_in', 'k_gashare_in', 'k_exp']
SQ = XI + ['k_bench_usage']
SQP = SQ + ['k_sh90']

# ---- stage 1 for the lineup-unknown variants: history-only predictions of today's lineup features ------------------------
S1 = ['u_avail', 'u_lag1_squad_usage', 'u_lag3_squad_usage', 'u_lag1_usage', 'u_lag3_usage', 'u_lag1_bench_usage', 'u_lag3_bench_usage',
      'u_susp', 'u_absent', 'u_e_usage', 'u_usage10', 'u_early_off', 'n_season']
r2 = {}
r2['k_usage'] = lab.stage1(D, 'k_usage', S1)
r2['k_bench_usage'] = lab.stage1(D, 'k_bench_usage', S1)
r2['k_gashare'] = lab.stage1(D, 'k_gashare', S1 + ['u_e_gashare', 'u_lag1_gashare', 'u_lag3_gashare', 'u_susp_ga', 'u_absent_ga'])
r2['k_exp'] = lab.stage1(D, 'k_exp', ['u_exp', 'u_lag1_exp', 'u_lag3_exp', 'u_susp', 'u_absent'])
r2['k_sh90'] = lab.stage1(D, 'k_sh90', ['u_sh90', 'u_lag1_sh90', 'u_lag3_sh90', 'u_susp', 'u_absent'])
for s in 'ha':
    early = D[f'n_season_{s}'].to_numpy(float) < 3
    D[f'k_usage_in_hat_{s}'] = np.where(~early, D[f'k_usage_hat_{s}'], np.nan)
    D[f'k_gashare_in_hat_{s}'] = np.where(~early, D[f'k_gashare_hat_{s}'], np.nan)
UA = U & D.k_usage_hat_h.notna().to_numpy() & D.k_usage_hat_a.notna().to_numpy()
hat = lambda feats: d(*[f + '_hat' for f in feats])

out = {}
coef_log = {}


def known(name, feats, fallback):
    Po, co = lab.walk_forward(D, d(*feats), K, fallback=fallback)
    Pp, cp, _, _ = lab.walk_forward_poisson(D, feats, K)
    P = fallback.copy()
    ch = K & (D.season >= 2013).to_numpy()
    P[ch] = (Po[ch] + Pp[ch]) / 2
    out[name] = P; coef_log[name] = (co, cp)
    return P


def unknown2(name, feats):
    P, co = lab.walk_forward(D, d(*feats), K, apply_spec=hat(feats), apply_avail=UA)
    out[name] = P; coef_log[name] = (co, None)
    return P


Pu = unknown2('unknown', SQ)
Pup = unknown2('unknown_plus', SQP)
Pd, cod = lab.walk_forward(D, d('u_lag3_squad_usage_in', 'u_susp_in', 'u_e_gashare_in', 'u_exp'), U)
out['unknown_direct'] = Pd
known('known_xi', XI, Pu)
known('known_squad', SQ, Pu)
known('known_squad_plus', SQP, Pup)

print('stage-1 out-of-sample R2 (mean over 2013-2021):', {k: round(float(np.mean([v for y, v in r.items() if 2013 <= y <= 2021])), 3) for k, r in r2.items()})
for name, P in out.items():
    sub = UA if name.startswith('unknown') else K
    lab.score(D, P, name, holdout=HOLD, sub=sub)
    lab.write_preds(D, P, name)
for name in ('known_squad', 'known_squad_plus', 'unknown'):
    co, cp = coef_log[name]
    print(name, 'ordered-logit coefs (per SD of home-away diff), model for 2026:', {k: float(v) for k, v in co[2026].items()})
    if cp: print(name, 'Poisson coefs (per SD), model for 2026:', {k: float(v) for k, v in cp[2026].items()})

# compact file with exactly the model inputs (per side) for a downstream stacker
cols = ['id', 'lg', 'season', 'date', 'status']
keep = []
for f in SQP + ['k_usage', 'k_gashare', 'k_squad_usage']:
    keep += [f + '_h', f + '_a']
hk = []
for f in ['k_usage', 'k_usage_in', 'k_bench_usage', 'k_gashare', 'k_gashare_in', 'k_exp', 'k_sh90']:
    hk += [f + '_hat_h', f + '_hat_a']
M = D[cols + keep + hk].copy()
M.insert(5, 'has_known', K.astype(int)); M.insert(6, 'has_unknown', UA.astype(int))
M.to_csv(f'{lab.ROOT}/features_model_inputs.csv', index=False, float_format='%.5f')
print('wrote features_model_inputs.csv', M.shape)
