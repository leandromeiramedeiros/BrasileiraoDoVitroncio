import numpy as np, pandas as pd
from models import *
F = load()
B = pd.read_csv(S + 'baseline_features.csv', dtype={'id': str})[['id', 'mu', 'hfa']]
F = F.merge(B, on='id', how='left')
F['v_mu'] = F.mu - 0.15; F['v_slc'] = F.v_sl - F.mu
F['T40'] = F.f_hatt_40 + F.f_aatt_40
print(F[F.season <= 2021].groupby(['lg', 'season'])[['mu', 'hfa', 'v_sl', 'v_slc']].mean().round(3).T.to_string())
ref = walk(F, fp_co(loss='cum', free_scale=False))
for nm, a in [('v_sl', ['v_sl']), ('v_mu', ['v_mu']), ('v_slc', ['v_slc']), ('v_mu+v_slc', ['v_mu', 'v_slc']), ('T40', ['T40']), ('T40+v_mu', ['T40', 'v_mu']), ('T40+v_slc', ['T40', 'v_slc']),
              ('season trend', ['trend'])]:
    if nm == 'season trend': F['trend'] = (F.season - 2009) / 10.0
    P, C = walk(F, fp_co(feats=a, free_scale=False, tau=0.05), keep_coef=True); score(F, P, name=nm, ref=ref); print('     coef 2017/2021', {k: round(v, 4) for k, v in C[2017].items()}, {k: round(v, 4) for k, v in C[2021].items()})
    print('     ', by_season(F, P, ref=ref).values)
