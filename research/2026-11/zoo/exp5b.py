import numpy as np, pandas as pd
from models import *
F = load()
mm = (F.ev & (F.season <= 2021)).values; kk = F.k.values[mm]; ss = F.season.values[mm]
def dev(P): return rps_ll(P[mm], kk)[0]
rv1 = dev(F[['v_pH', 'v_pD', 'v_pA']].values)
S1 = ['f_tot_r_40', 'v_slc', 'f_dom_r_4', 'f_dom_r_40', 'f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40', 'f_pos_raw_40', 'st_deadh_d', 'mn_susp_d']
co = walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False)); r0 = dev(co)
parts = np.load('cache_pool_parts.npy'); zoo_all = parts[5]; zoo_goal = parts[4]
print('CO-S1', r0.mean().round(5), (r0 - rv1).mean().round(5))
for w in (0.9, 0.8, 0.7, 0.6):
    for nm, Z_ in (('zoo_all', zoo_all), ('zoo_goal', zoo_goal)):
        r = dev(w * co + (1 - w) * Z_); e = r - r0; lo, hi = boot_ci(e, 1000)
        print(f'pool {w:.1f} CO-S1 + {1 - w:.1f} {nm:8s}: RPS {r.mean():.5f} vs CO-S1 {e.mean():+.5f} [{lo:+.5f},{hi:+.5f}] [13-17 {e[ss <= 2017].mean():+.5f} | 18-21 {e[ss >= 2018].mean():+.5f}]')
