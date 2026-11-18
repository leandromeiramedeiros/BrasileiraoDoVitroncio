"""Stage 3: greedy forward selection of feature groups on DEV (CO learner), then ridge strength."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
ref = walk(F, fp_co(loss='cum', free_scale=False))
mm = (F.ev & (F.season <= 2021)).values; kk = F.k.values[mm]; ss = F.season.values[mm]
def dev(P): return rps_ll(P[mm], kk)[0]
rref = dev(ref); rv1 = dev(F[['v_pH', 'v_pD', 'v_pA']].values)
print('v1', rv1.mean().round(5), 'recal', rref.mean().round(5))
G = {
 'v_sl': (['v_sl'], []),
 'dead': (['st_deadh_d'], []),
 'dom': (['f_dom_r_4', 'f_dom_r_40'], []),
 'gfc': (['f_gfc_r_4'], []),
 'statform': (['f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40'], []),
 'poss': (['f_pos_raw_40'], []),
 'venue': (['ven_neutral'], []),
 'susp': (['mn_susp_d'], []),
 'ratings_goal': (['elo20_d', 'pi035_d', 'ber05_d'], []),
 'ratings_stat': (['eloxg_d', 'pixg_d'], []),
 'gap': (['gapsh_d', 'gapsot_d', 'gapco_d'], []),
 'v1chan': (['v_gdl', 'v_shd', 'v_sotd', 'v_cod'], []),
 'rawform': (['f_pts_raw_4', 'f_gd_raw_4'], []),
 'resform': (['f_gd_r_4', 'f_gd_r_40', 'f_xgd_r_12'], []),
 'aatt': (['f_aatt_40', 'f_hatt_40'], []),
 'draw': ([], ['lgd_tot', 'n_exp_d', 'a_absdl']),
}
F['a_absdl'] = (F.v_dl - 0.36).abs() - 0.27
tau = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
sel = []; cur = rref; curA, curD = [], []
while True:
    best = None
    for g, (a, d) in G.items():
        if g in sel: continue
        P = walk(F, fp_co(feats=curA + a, dfeats=curD + d, free_scale=False, tau=tau))
        r = dev(P)
        dd = (r - cur).mean()
        print(f'   try +{g:13s} {dd:+.5f}  [13-17 {(r - cur)[ss <= 2017].mean():+.5f} | 18-21 {(r - cur)[ss >= 2018].mean():+.5f}]', flush=True)
        if best is None or dd < best[1]: best = (g, dd, r)
    if best is None or best[1] > -0.00005: break
    g = best[0]; sel.append(g); curA += G[g][0]; curD += G[g][1]; cur = best[2]
    lo, hi = boot_ci(cur - rv1, 1000)
    print(f'ADD {g}: RPS {cur.mean():.5f}  vs v1 {(cur - rv1).mean():+.5f} [{lo:+.5f},{hi:+.5f}]  vs recal {(cur - rref).mean():+.5f}', flush=True)
print('selected', sel, curA, curD)
