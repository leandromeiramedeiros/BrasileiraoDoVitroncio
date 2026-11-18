#!/usr/bin/env python
"""Scored variants (walk-forward ordered-logit shift on top of v1).

  python experiments.py dev            -> DEV table for every variant (no holdout numbers are computed or shown)
  python experiments.py final A B ...  -> writes preds_<variant>.csv for the named variants and prints DEV+HOLDOUT

Selection of variants / hyper-parameters is done on DEV only.
"""
import sys
import numpy as np, pandas as pd
import lab

import warnings; warnings.filterwarnings('ignore')
D = lab.load()
n = len(D)


def col(c):
    return D[c].to_numpy(float)


# ---------- derived per-side columns (all built from pre-match columns only) ----------
for s in 'ha':
    ns = col(f'n_season_{s}')
    early = ns < 3
    D[f'k_usage_in_{s}'] = np.where(~early, col(f'k_usage_{s}'), np.nan)
    D[f'k_usage_early_{s}'] = np.where(early, col(f'k_usage_{s}'), np.nan)
    D[f'k_usage_rel_{s}'] = col(f'k_usage_{s}') - col(f'u_usage10_{s}')
    D[f'k_usage_rel_in_{s}'] = np.where(~early, D[f'k_usage_rel_{s}'], np.nan)
    D[f'k_lowusage_{s}'] = np.clip(0.7 - col(f'k_usage_{s}'), 0, None)
    D[f'k_lowusage_in_{s}'] = np.where(~early, D[f'k_lowusage_{s}'], np.nan)
    D[f'k_gashare_in_{s}'] = np.where(~early, col(f'k_gashare_{s}'), np.nan)
    D[f'k_shshare_in_{s}'] = np.where(~early, col(f'k_shshare_{s}'), np.nan)
    D[f'k_garel_{s}'] = col(f'k_gashare_{s}') - col(f'k_gashare_usual_{s}')
    D[f'k_garel_in_{s}'] = np.where(~early, D[f'k_garel_{s}'], np.nan)
    for r in ('rapm0g', 'rapm1g', 'rapm0x', 'rapm1x', 'ga90', 'sh90', 'ped', 'exp'):
        D[f'k_{r}_rel_{s}'] = col(f'k_{r}_{s}') - col(f'u_{r}_{s}')
    D[f'u_prev_in_{s}'] = np.where(~early, col(f'u_prev_usage_{s}'), np.nan)
    D[f'u_usage3_in_{s}'] = np.where(~early, col(f'u_usage3_{s}'), np.nan)
    D[f'u_usage3_early_{s}'] = np.where(early, col(f'u_usage3_{s}'), np.nan)
    D[f'u_ret_early_{s}'] = np.where(ns < 6, col(f'u_ret_{s}'), np.nan)
    D[f'k_ret_early_{s}'] = np.where(ns < 6, col(f'k_ret_squad_{s}'), np.nan)
    D[f'k_retxi_early_{s}'] = np.where(ns < 6, col(f'k_ret_xi_{s}'), np.nan)
    for c in ('u_avail', 'u_lag3_squad_usage', 'u_e_usage', 'u_e_gashare', 'u_susp', 'u_absent'):
        D[f'{c}_in_{s}'] = np.where(~early, col(f'{c}_{s}'), np.nan)

K_AVAIL = ((D.k_ok_h == 1) & (D.k_ok_a == 1) & D.k_usage_h.notna() & D.k_usage_a.notna()).to_numpy()
U_AVAIL = (D.u_usage3_h.notna() & D.u_usage3_a.notna()).to_numpy()

d = lambda *names: [(x, 'diff') for x in names]

VARIANTS = {
    # ---- lineup-known ----
    'k_usage': dict(spec=d('k_usage'), avail=K_AVAIL),
    'k_usage_split': dict(spec=d('k_usage_in', 'k_usage_early'), avail=K_AVAIL),
    'k_usage_in': dict(spec=d('k_usage_in'), avail=K_AVAIL),
    'k_usage_s': dict(spec=d('k_usage_s'), avail=K_AVAIL),
    'k_usage_long': dict(spec=d('k_usage_long'), avail=K_AVAIL),
    'k_start': dict(spec=d('k_start'), avail=K_AVAIL),
    'k_usage_xl': dict(spec=d('k_usage_xl'), avail=K_AVAIL),
    'k_usage+xl': dict(spec=d('k_usage', 'k_usage_xl'), avail=K_AVAIL),
    'k_usage_rel': dict(spec=d('k_usage_rel'), avail=K_AVAIL),
    'k_usage_ha': dict(spec=[('k_usage', 'h'), ('k_usage', 'a')], avail=K_AVAIL),
    'k_usage+low': dict(spec=d('k_usage', 'k_lowusage'), avail=K_AVAIL),
    'k_chg': dict(spec=d('k_chg'), avail=K_AVAIL),
    'k_gashare': dict(spec=d('k_gashare'), avail=K_AVAIL),
    'k_shshare': dict(spec=d('k_shshare'), avail=K_AVAIL),
    'k_garel': dict(spec=d('k_garel'), avail=K_AVAIL),
    'k_top+gk': dict(spec=d('k_top', 'k_gk'), avail=K_AVAIL),
    'k_miss_pos': dict(spec=d('k_miss_G', 'k_miss_D', 'k_miss_M', 'k_miss_F'), avail=K_AVAIL),
    'k_rapm0g_rel': dict(spec=d('k_rapm0g_rel'), avail=K_AVAIL),
    'k_rapm1g_rel': dict(spec=d('k_rapm1g_rel'), avail=K_AVAIL),
    'k_rapm0x_rel': dict(spec=d('k_rapm0x_rel'), avail=K_AVAIL),
    'k_rapm1x_rel': dict(spec=d('k_rapm1x_rel'), avail=K_AVAIL),
    'k_rapm1g_abs': dict(spec=d('k_rapm1g'), avail=K_AVAIL),
    'k_rapm1x_abs': dict(spec=d('k_rapm1x'), avail=K_AVAIL),
    'k_ped': dict(spec=d('k_ped'), avail=K_AVAIL),
    'k_ga90_rel': dict(spec=d('k_ga90_rel'), avail=K_AVAIL),
    'k_ga90': dict(spec=d('k_ga90'), avail=K_AVAIL),
    'k_exp': dict(spec=d('k_exp'), avail=K_AVAIL),
    'k_debut+new': dict(spec=d('k_debut', 'k_new'), avail=K_AVAIL),
    'k_ret': dict(spec=d('k_ret_early', 'k_retxi_early'), avail=K_AVAIL),
    'k_usage+ga': dict(spec=d('k_usage', 'k_gashare'), avail=K_AVAIL),
    'k_usage+ga+sh': dict(spec=d('k_usage', 'k_gashare', 'k_shshare'), avail=K_AVAIL),
    'k_usage+rapm1x': dict(spec=d('k_usage', 'k_rapm1x_rel'), avail=K_AVAIL),
    'k_usage+ga+rapm1x+ped': dict(spec=d('k_usage', 'k_gashare', 'k_rapm1x_rel', 'k_ped'), avail=K_AVAIL),
    'k_all': dict(spec=d('k_usage_in', 'k_usage_early', 'k_gashare', 'k_shshare', 'k_top', 'k_gk', 'k_rapm1x_rel', 'k_ped', 'k_ga90_rel', 'k_exp', 'k_debut'), avail=K_AVAIL, l2=20.0),
    # ---- lineup-unknown ----
    'u_usage3': dict(spec=d('u_usage3'), avail=U_AVAIL),
    'u_prev': dict(spec=d('u_prev_usage'), avail=U_AVAIL),
    'u_usage10': dict(spec=d('u_usage10'), avail=U_AVAIL),
    'u_susp': dict(spec=d('u_susp'), avail=U_AVAIL),
    'u_absent': dict(spec=d('u_absent'), avail=U_AVAIL),
    'u_susp+absent': dict(spec=d('u_susp', 'u_absent'), avail=U_AVAIL),
    'u_susp+prev': dict(spec=d('u_susp', 'u_prev_usage'), avail=U_AVAIL),
    'u_susp+absent+usage3': dict(spec=d('u_susp', 'u_absent', 'u_usage3'), avail=U_AVAIL),
    'u_rapm1x': dict(spec=d('u_rapm1x'), avail=U_AVAIL),
    'u_ped': dict(spec=d('u_ped'), avail=U_AVAIL),
    'u_ga90': dict(spec=d('u_ga90'), avail=U_AVAIL),
    'u_ret': dict(spec=d('u_ret_early'), avail=U_AVAIL),
    'u_exp': dict(spec=d('u_exp'), avail=U_AVAIL),
    'u_all': dict(spec=d('u_susp', 'u_absent', 'u_usage3', 'u_rapm1x', 'u_ped', 'u_ga90', 'u_ret_early', 'u_exp'), avail=U_AVAIL, l2=20.0),
}


def run(name, holdout=False, write=False, quiet=False):
    v = VARIANTS[name]
    P, coefs = lab.walk_forward(D, v['spec'], v['avail'], l2=v.get('l2', 5.0), train_lg=v.get('train_lg', ('A', 'B')))
    out = lab.score(D, P, name, holdout=holdout, sub=v['avail'], quiet=quiet)
    if write:
        lab.write_preds(D, P, name)
    return P, coefs, out


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'dev':
        names = sys.argv[2:] or list(VARIANTS)
        rows = []
        for nm in names:
            P, coefs, out = run(nm)
            last = coefs[max(coefs)] if coefs else {}
            rows.append((nm, out['DEV'][3], out['DEV'][4], out['DEV'][5], out['DEV'][6], out['DEVsub'][0], out['DEVsub'][3], str(last)))
        pd.DataFrame(rows, columns=['variant', 'dRPS_dev', 'lo', 'hi', 'dLL_dev', 'n_sub', 'dRPS_sub', 'coefs_last']).to_csv(f'{lab.ROOT}/work/results_dev.csv', index=False)
    elif mode == 'final':
        for nm in sys.argv[2:]:
            P, coefs, out = run(nm, holdout=True, write=True)
            print('   coefs by season:', {y: c for y, c in coefs.items() if y in (2013, 2017, 2022, 2026)})
