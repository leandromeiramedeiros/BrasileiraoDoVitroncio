"""Compact stacking file with the surviving context features + the walk-forward log-rate shifts of ctx_core / ctx_core_r.
Output: features_core.csv (id-keyed; all Série A/B matches; shifts only where a walk-forward fit exists = Série A 2013+)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, fitlib as F
D = F.load()
L0 = pd.read_csv(os.path.join(F.HERE, 'coef', 'lambda_ctx_core.csv'), dtype={'id': str}).set_index('id')
L1 = pd.read_csv(os.path.join(F.HERE, 'coef', 'lambda_ctx_core_r.csv'), dtype={'id': str}).set_index('id')
K = pd.DataFrame(index=D.index); K['lg'] = D.lg; K['season'] = D.season
for s in 'ha':
    K[f'releg_{s}'] = D[f'releg_{s}']; K[f'clinched_{s}'] = D[f'clinched_{s}']; K[f'next_cont4_{s}'] = D[f'next_cont4_{s}']
K['closed'] = D.closed; K['home_disp'] = (D.home_disp_km > 100).astype(float)
K['home_disp_km'] = D.home_disp_km.round(1); K['dist_km'] = D.dist_km.round(1)
K['r_dh_all'] = D.r_dh_all.round(5); K['r_da_all'] = D.r_da_all.round(5)
K['ctx_dlh'] = (np.log(L0.lh.reindex(D.index)) - np.log(D.lh)).round(5); K['ctx_dla'] = (np.log(L0.la.reindex(D.index)) - np.log(D.la)).round(5)
K['ctxr_dlh'] = (np.log(L1.lh.reindex(D.index)) - np.log(D.lh)).round(5); K['ctxr_dla'] = (np.log(L1.la.reindex(D.index)) - np.log(D.la)).round(5)
K.to_csv(os.path.join(F.HERE, 'features_core.csv')); print(K.shape)
