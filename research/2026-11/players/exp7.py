import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
from exp3 import ol, pois
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL; d=E.d
for s in 'ha':
    early = D[f'n_season_{s}'].to_numpy(float) < 3
    for c in ('u_avail','u_lag1_squad_usage','u_lag3_squad_usage','u_e_usage','u_e_gashare','u_lag1_usage','u_lag3_usage','u_lag3_gashare','u_susp','u_absent','u_lag3_bench_usage'):
        D[f'{c}_in_{s}'] = np.where(~early, D[f'{c}_{s}'], np.nan)
what=sys.argv[1]
if what=='single':
    for f in ('u_avail','u_avail_in','u_lag1_squad_usage','u_lag3_squad_usage','u_lag3_squad_usage_in','u_lag3_bench_usage','u_e_usage','u_e_usage_in','u_e_gashare','u_e_gashare_in','u_lag3_usage_in','u_lag3_gashare','u_lag3_gashare_in','u_susp_ga','u_absent_ga','u_susp_in','u_absent_in','u_lag3_exp','u_lag3_sh90','u_sh90','u_exp','u_usage10'):
        ol(f, d(f), avail=U)
if what=='combo':
    ol('avail_in+exp', d('u_avail_in','u_exp'), avail=U)
    ol('avail_in+exp+sh90', d('u_avail_in','u_exp','u_sh90'), avail=U)
    ol('avail+susp+absent', d('u_avail','u_susp','u_absent'), avail=U)
    ol('lag3squad+susp', d('u_lag3_squad_usage','u_susp'), avail=U)
    ol('lag3squad_in+susp_in+e_gashare_in+exp', d('u_lag3_squad_usage_in','u_susp_in','u_e_gashare_in','u_exp'), avail=U)
    ol('lag3squad+susp+absent+usage10+exp', d('u_lag3_squad_usage','u_susp','u_absent','u_usage10','u_exp'), avail=U)
    ol('e_usage_in+e_gashare_in+exp+avail_in', d('u_e_usage_in','u_e_gashare_in','u_exp','u_avail_in'), avail=U)
if what=='two':
    # stage 1: predict today's lineup features from history; stage 2: coefficients of the lineup-known model
    preds=['u_avail','u_lag1_squad_usage','u_lag3_squad_usage','u_lag1_usage','u_lag3_usage','u_lag1_bench_usage','u_lag3_bench_usage','u_susp','u_absent','u_e_usage','u_usage10','u_early_off','n_season']
    for tgt in ('k_usage','k_bench_usage','k_squad_usage'):
        r2=lab.stage1(D, tgt, preds); print(tgt, 'stage-1 R2 by season', {y: round(v,2) for y,v in r2.items() if y>=2013})
    r2=lab.stage1(D, 'k_gashare', preds+['u_e_gashare','u_lag1_gashare','u_lag3_gashare','u_susp_ga','u_absent_ga']); print('k_gashare R2', {y: round(v,2) for y,v in r2.items() if y>=2013})
    r2=lab.stage1(D, 'k_exp', ['u_exp','u_lag1_exp','u_lag3_exp','u_susp','u_absent']); print('k_exp R2', {y: round(v,2) for y,v in r2.items() if y>=2013})
    r2=lab.stage1(D, 'k_sh90', ['u_sh90','u_lag1_sh90','u_lag3_sh90','u_susp','u_absent']); print('k_sh90 R2', {y: round(v,2) for y,v in r2.items() if y>=2013})
    for s in 'ha':
        early = D[f'n_season_{s}'].to_numpy(float) < 3
        D[f'k_usage_hat_in_{s}'] = np.where(~early, D[f'k_usage_hat_{s}'], np.nan)
        D[f'k_gashare_hat_in_{s}'] = np.where(~early, D[f'k_gashare_hat_{s}'], np.nan)
    UA = U & D.k_usage_hat_h.notna().to_numpy() & D.k_usage_hat_a.notna().to_numpy()
    def two(name, spec, aspec):
        P,co=lab.walk_forward(D, spec, K, apply_spec=aspec, apply_avail=UA)
        lab.score(D,P,'2STAGE '+name, sub=UA); return P
    two('base3', d('k_usage_in','k_gashare_in','k_exp'), d('k_usage_hat_in','k_gashare_hat_in','k_exp_hat'))
    two('base4', d('k_usage_in','k_gashare_in','k_exp','k_bench_usage'), d('k_usage_hat_in','k_gashare_hat_in','k_exp_hat','k_bench_usage_hat'))
    two('base4+sh90', d('k_usage_in','k_gashare_in','k_exp','k_bench_usage','k_sh90'), d('k_usage_hat_in','k_gashare_hat_in','k_exp_hat','k_bench_usage_hat','k_sh90_hat'))
    two('squad+ga+exp', d('k_squad_usage','k_gashare_in','k_exp'), d('k_squad_usage_hat','k_gashare_hat_in','k_exp_hat'))
    ol('direct on hats: squad_hat+ga_hat+exp_hat', d('k_squad_usage_hat','k_gashare_hat_in','k_exp_hat'), avail=UA)
    ol('direct on hats: base4 hats', d('k_usage_hat_in','k_gashare_hat_in','k_exp_hat','k_bench_usage_hat'), avail=UA)
