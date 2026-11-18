import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
from exp3 import ol, pois
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL; d=E.d
for s in 'ha':
    early = D[f'n_season_{s}'].to_numpy(float) < 3
    D[f'k_bench_unused_{s}'] = D[f'k_bench_usage_{s}'] - D[f'diag_bench_used_{s}']
    D[f'k_bench_gashare_in_{s}'] = np.where(~early, D[f'k_bench_gashare_{s}'], np.nan)
    D[f'k_sq_gashare_in_{s}'] = np.where(~early, D[f'k_bench_gashare_{s}'] + D[f'k_gashare_{s}'], np.nan)
base=['k_usage_in','k_gashare_in','k_exp']
what=sys.argv[1]
if what=='diag':
    ol('base3+bench', d(*base,'k_bench_usage'))
    ol('base3+bench_unused (diag)', d(*base,'k_bench_unused'))
    ol('base3+bench_used (diag, post-match info)', d(*base,'diag_bench_used'))
    ol('base3+bench_unused+used (diag)', d(*base,'k_bench_unused','diag_bench_used'))
    ol('base3+bench_n', d(*base,'k_bench_n'))
    ol('base3+bench+bench_n', d(*base,'k_bench_usage','k_bench_n'))
    ol('bench_n', d('k_bench_n'))
if what=='known':
    b4=base+['k_bench_usage']
    ol('base4', d(*b4))
    for extra in ('k_bench_n','k_bench_start','k_bench_exp','k_bench_ga90','k_bench_sh90','k_bench_reg','k_bench_gashare','k_bench_gashare_in','k_bench_shshare','k_sh90','k_ga90','k_expA','k_rapm1x','k_gk','k_top','k_shshare_in','u_usage10','u_susp','u_absent','k_usage_early','k_ped','k_exp_club','u_lag3_squad_usage','u_lag3_bench_usage'):
        ol('base4 + '+extra, d(*b4,extra))
    ol('usage_in + sq_gashare_in + exp + bench', d('k_usage_in','k_sq_gashare_in','k_exp','k_bench_usage'))
    ol('squad_usage + gashare_in + exp', d('k_squad_usage','k_gashare_in','k_exp'))
