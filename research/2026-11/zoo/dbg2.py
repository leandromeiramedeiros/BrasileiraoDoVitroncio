import numpy as np, pandas as pd
from models import *
F = load()
F['T40'] = F.f_hatt_40 + F.f_aatt_40
D = F[(F.season >= 2009) & (F.season <= 2021)].copy()
for col in ['T40', 'v_sl']:
    for lg in ['A', 'B']:
        x = D[D.lg == lg]
        b = pd.qcut(x[col], 5)
        print(col, lg)
        print(x.groupby(b, observed=True).apply(lambda z: pd.Series(dict(n=len(z), hg_res=(z.hg - z.v_lh).mean(), ag_res=(z.ag - z.v_la).mean(), lh=z.v_lh.mean(), la=z.v_la.mean(),
               H=(z.k == 0).mean() - z.v_pH.mean(), D=(z.k == 1).mean() - z.v_pD.mean(), A=(z.k == 2).mean() - z.v_pA.mean())), include_groups=False).round(3).to_string())
print(D.groupby('season')[['T40', 'v_sl']].mean().round(3).T.to_string())
print(np.corrcoef(D.T40, D.v_sl)[0, 1])
# PO forms
ref = walk(F, fp_co(loss='cum', free_scale=False))
for nm, kw in [('PO anti T40', dict(feats=['T40'])), ('PO sym T40', dict(sfeats=['T40'])), ('PO anti hatt,aatt', dict(feats=['f_hatt_40', 'f_aatt_40'])),
               ('PO anti v_sl', dict(feats=['v_sl'])), ('PO sym v_sl', dict(sfeats=['v_sl'])), ('PO anti+sym T40', dict(feats=['T40'], sfeats=['T40']))]:
    P, C = walk(F, fp_po(tau=0.05, **kw), keep_coef=True); score(F, P, name=nm, ref=ref); print('     coef 2021', {k: round(v, 4) for k, v in C[2021].items()})
