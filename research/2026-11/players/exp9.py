import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
from exp3 import ol, pois
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL; d=E.d
b4=['k_usage_in','k_gashare_in','k_exp','k_bench_usage']
what=sys.argv[1]
if what=='box':
    ol('base4', d(*b4))
    for extra in ('k_sh90','k_sot90','k_fs90','k_g90','k_ga90','k_startrate','k_years'):
        ol('base4 + '+extra, d(*b4,extra))
    ol('base4 + sh90 + fs90', d(*b4,'k_sh90','k_fs90'))
    ol('base4 + sh90 + startrate', d(*b4,'k_sh90','k_startrate'))
    ol('base4 - exp + years', d('k_usage_in','k_gashare_in','k_years','k_bench_usage'))
    ol('base4 - exp + startrate', d('k_usage_in','k_gashare_in','k_startrate','k_bench_usage'))
    # leave-one-out of the base model
    for i,f in enumerate(b4):
        ol('base4 minus '+f, d(*[x for x in b4 if x!=f]))
if what=='xg':
    ol('base4', d(*b4))
    for extra in ('k_xg90','k_sh90'):
        ol('base4 + '+extra, d(*b4,extra))
    ol('base4 + sh90 + xg90', d(*b4,'k_sh90','k_xg90'))
    for f in ('u_t_gres_s','u_t_gres_l','u_t_xgres_s','u_t_xgres_l','u_t_luck_s','u_t_luck_l','u_xg90','u_sh90','u_sot90','u_fs90'):
        ol(f, d(f), avail=U)
    ol('u_t_xgres_l + u_t_luck_l', d('u_t_xgres_l','u_t_luck_l'), avail=U)
    ol('u_t_gres_l + u_t_xgres_l', d('u_t_gres_l','u_t_xgres_l'), avail=U)
    ol('base4 + u_t_xgres_l', d(*b4,'u_t_xgres_l'))
    ol('base4 + u_t_gres_l', d(*b4,'u_t_gres_l'))
    ol('base4 + sh90 + u_t_xgres_l', d(*b4,'k_sh90','u_t_xgres_l'))
