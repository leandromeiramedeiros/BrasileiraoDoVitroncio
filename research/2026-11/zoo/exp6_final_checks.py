"""Stage 6: last DEV checks around the chosen model (core subset, ridge flatness, nested-CV ridge)."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
mm = (F.ev & (F.season <= 2021)).values; kk = F.k.values[mm]; ss = F.season.values[mm]
def dev(P): return rps_ll(P[mm], kk)[0]
rv1 = dev(F[['v_pH', 'v_pD', 'v_pA']].values)
S1 = ['f_tot_r_40', 'v_slc', 'f_dom_r_4', 'f_dom_r_40', 'f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40', 'f_pos_raw_40', 'st_deadh_d', 'mn_susp_d']
CORE = S1[:7]
def rep(P, name, base=None):
    r = dev(P); d = r - rv1; lo, hi = boot_ci(d, 1000)
    s = f'{name:40s} RPS {r.mean():.5f} vs v1 {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]'
    if base is not None:
        e = r - dev(base); s += f' | vs CO-S1 {e.mean():+.5f} [13-17 {e[ss <= 2017].mean():+.5f} | 18-21 {e[ss >= 2018].mean():+.5f}]'
    print(s, flush=True); return r
co, C = walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False), keep_coef=True); rep(co, 'CO-S1 tau=.1 wB=1')
print(pd.DataFrame(C).T.drop(columns=['a1', 'a2']).round(3).to_string())
print('dRPS vs v1 by DEV season:', by_season(F, co).to_dict())
rep(walk(F, fp_co(feats=CORE, tau=0.1, wB=1.0, free_scale=False)), 'CO-core (open+dom+statform)', co)
for tau in (0.05, 0.2, 0.5): rep(walk(F, fp_co(feats=S1, tau=tau, wB=1.0, free_scale=False)), f'CO-S1 tau={tau}', co)
rep(walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False), train_from=2011), 'CO-S1 train_from=2011', co)
rep(walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False), train_from=2007), 'CO-S1 train_from=2007', co)

# nested choice of tau: for season Y pick tau by leave-one-season-out CV inside the training seasons (rule-1 clean)
def fp_cv(feats, taus=(0.02, 0.05, 0.1, 0.2, 0.5)):
    def fp(tr, te, Y):
        seas = sorted(tr.season.unique()); best = None
        for tau in taus:
            L = 0.0
            for s in seas[-6:]:
                a = tr[tr.season != s]; b = tr[(tr.season == s)]
                P = fp_co(feats=feats, tau=tau, wB=1.0, free_scale=False)(a, b, Y)
                L += rps_ll(P, b.k.values)[0].sum()
            if best is None or L < best[0]: best = (L, tau)
        fp.tau = best[1]
        return fp_co(feats=feats, tau=best[1], wB=1.0, free_scale=False)(tr, te, Y)
    return fp
f = fp_cv(S1); taus = {}
P = np.full((len(F), 3), np.nan)
for Y in EVAL_SEASONS:
    tr = F[(F.season < Y) & (F.season >= 2009)]; te = F[(F.season == Y) & F.ev]
    P[te.index.values] = f(tr, te, Y); taus[Y] = f.tau
rep(P, 'CO-S1 nested-CV tau', co); print('   chosen tau by season:', taus)
np.save('cache_cvtau.npy', P)
