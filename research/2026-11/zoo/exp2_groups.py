"""Stage 2: feature groups on DEV (cumulative-offset logit and Poisson-offset learners)."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
ref = walk(F, fp_co(loss='cum', free_scale=False))
score(F, ref, name='REF recal (2 intercepts)')

def halves(P, ref):
    mm = (F.ev & (F.season <= 2021)).values; s = F.season.values[mm]
    r, _ = rps_ll(P[mm], F.k.values[mm]); rr, _ = rps_ll(ref[mm], F.k.values[mm]); d = r - rr
    return f'[13-17 {d[s <= 2017].mean():+.5f} | 18-21 {d[s >= 2018].mean():+.5f} | seasons<0: {int((pd.Series(d).groupby(s).mean() < 0).sum())}/9]'

G = {
 'isA': (['isA'], ['isA']),
 'v_sl': (['v_sl'], []),
 'dead': (['st_deadh_d'], []),
 'dead_ha': (['st_deadh_h', 'st_deadh_a'], []),
 'dom': (['f_dom_r_4', 'f_dom_r_40'], []),
 'dom12': (['f_dom_r_12'], []),
 'gfc': (['f_gfc_r_4'], []),
 'statform': (['f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40'], []),
 'poss': (['f_pos_raw_40', 'pos_d'], []),
 'venue': (['ven_neutral'], []),
 'susp': (['mn_susp_d'], []),
 'ratings_goal': (['elo20_d', 'pi035_d', 'ber05_d'], []),
 'ratings_stat': (['eloxg_d', 'pixg_d', 'gapsh_d', 'gapsot_d', 'gapco_d'], []),
 'v1chan': (['v_gdl', 'v_shd', 'v_sotd', 'v_cod'], []),
 'novel': (['n_promoe_d', 'n_newe_d', 'n_rest_d', 'n_short_d', 'n_exp_d'], ['n_early']),
 'h2h': (['h2h_r', 'h2h_home_r', 'ven_team_r', 'ven_long_d'], []),
 'geo': (['ven_dist', 'ven_derby'], ['ven_derby']),
 'drift': (['lgd_hfa'], ['lgd_draw']),
}
lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
print('--- single groups on top of recal, lam', lam)
res = {}
for g, (a, d) in G.items():
    P = walk(F, fp_co(feats=a, dfeats=d, free_scale=False, lam=lam))
    r, dd = score(F, P, name=f'CO +{g}', ref=ref, verbose=False)
    mm = (F.ev & (F.season <= 2021)).values
    rr, _ = rps_ll(ref[mm], F.k.values[mm]); r2, _ = rps_ll(P[mm], F.k.values[mm]); lo, hi = boot_ci(r2 - rr, 1000)
    print(f'CO +{g:14s} d vs ref {(r2 - rr).mean():+.5f} [{lo:+.5f},{hi:+.5f}] {halves(P, ref)}', flush=True)
    res[g] = (r2 - rr).mean()
