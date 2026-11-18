"""Stage 1b: univariate screening with the Poisson-offset learner (goals as target), DEV only."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
c = F.v_dl - 0.36
F['i_early_dl'] = F.n_early * c; F['i_S_dl'] = F.v_S * 100 * c; F['i_phase_dl'] = (F.phase - 0.5) * c
F['a_absdl'] = c.abs() - 0.27
meta = {'id', 'lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag', 'k', 'isA', 'ev', 'v_lh', 'v_la', 'v_pH', 'v_pD', 'v_pA', 'n_h', 'n_a'}
feats = [c for c in F.columns if c not in meta]
ref = walk(F, fp_po(calib=True))
score(F, ref, name='REF PO intercept + calib')
rows = []
t = time.time()
for f in feats:
    P, C = walk(F, fp_po(feats=[f], lam=0.01), keep_coef=True)
    mm = (F.ev & (F.season <= 2021)).values
    r, _ = rps_ll(P[mm], F.k.values[mm]); rr, _ = rps_ll(ref[mm], F.k.values[mm]); d = r - rr
    s = F.season.values[mm]
    lo, hi = boot_ci(d, 500)
    rows.append(dict(f=f, d=d.mean(), lo=lo, hi=hi, d1317=d[s <= 2017].mean(), d1821=d[s >= 2018].mean(), coef21=C[2021][f], coef16=C[2016][f],
                     nseas_neg=int((pd.Series(d).groupby(s).mean() < 0).sum())))
R = pd.DataFrame(rows).sort_values('d')
R.to_csv('screen_po.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(5).to_string())
print(time.time() - t)
