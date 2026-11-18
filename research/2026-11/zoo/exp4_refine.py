"""Stage 4: refine the selected set on DEV: leave-one-group-out, additions, ridge strength, loss, B weight."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
mm = (F.ev & (F.season <= 2021)).values; kk = F.k.values[mm]; ss = F.season.values[mm]
def dev(P): return rps_ll(P[mm], kk)[0]
rv1 = dev(F[['v_pH', 'v_pD', 'v_pA']].values)
GR = {
 'open': ['f_tot_r_40', 'v_slc'],
 'dom': ['f_dom_r_4', 'f_dom_r_40'],
 'statform': ['f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40'],
 'poss': ['f_pos_raw_40'],
 'dead': ['st_deadh_d'],
 'susp': ['mn_susp_d'],
}
ALL = sum(GR.values(), [])
def run(feats, name, base=None, **kw):
    kw.setdefault('tau', 0.05); kw.setdefault('free_scale', False)
    P = walk(F, fp_co(feats=feats, **kw)); r = dev(P)
    d = r - rv1; lo, hi = boot_ci(d, 1000)
    s = f'{name:34s} RPS {r.mean():.5f} vs v1 {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]'
    if base is not None:
        e = r - base; s += f' | vs full {e.mean():+.5f} [13-17 {e[ss <= 2017].mean():+.5f} | 18-21 {e[ss >= 2018].mean():+.5f}]'
    print(s, flush=True); return r
full = run(ALL, 'FULL (S1)')
what = sys.argv[1] if len(sys.argv) > 1 else 'loo'
if what == 'loo':
    for g in GR: run([f for f in ALL if f not in GR[g]], f'  minus {g}', full)
    for nm, add in [('ht', ['f_ht_r_12']), ('first', ['f_first_r_12']), ('ht4,40', ['f_ht_r_4', 'f_ht_r_40']), ('gfc', ['f_gfc_r_4']), ('venue', ['ven_neutral']), ('tot4,12', ['f_tot_r_4', 'f_tot_r_12']),
                    ('dom12', ['f_dom_r_12']), ('xgd_r40', ['f_xgd_r_40']), ('gap', ['gapsh_d', 'gapsot_d', 'gapco_d']), ('v1chan', ['v_shd', 'v_sotd', 'v_cod']), ('pos_d', ['pos_d']),
                    ('elo20', ['elo20_d']), ('pi+ber', ['pi035_d', 'ber05_d']), ('eloxg,pixg', ['eloxg_d', 'pixg_d']), ('susp_g', ['mn_suspg_d']), ('red', ['mn_red_d']), ('dead_s', ['st_deads_d']),
                    ('rel/top', ['st_relh_d', 'st_toph_d']), ('gd_r', ['f_gd_r_4', 'f_gd_r_40']), ('fv', ['fv_gd_r_40', 'fv_xgd_r_40']), ('lgd', ['lgd_hfa']), ('ber_s', ['ber05_s']),
                    ('stat12', ['f_shr_r_12', 'f_sotr_r_12', 'f_cor_r_12']), ('rest', ['n_rest_d', 'n_short_d'])]:
        run(ALL + add, f'  plus {nm}', full)
if what == 'hyper':
    for tau in (0.02, 0.035, 0.07, 0.1, 0.2): run(ALL, f'  tau={tau}', full, tau=tau)
    for loss in ('ll', 'rps'): run(ALL, f'  loss={loss}', full, loss=loss)
    for wB in (0.0, 0.25, 1.0): run(ALL, f'  wB={wB}', full, wB=wB)
    for dec in (0.9, 0.95): run(ALL, f'  decay={dec}', full, decay=dec)
    run(ALL, '  free_scale', full, free_scale=True)
