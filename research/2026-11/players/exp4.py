import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
from exp3 import ol, pois
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL; d=E.d
for s in 'ha':
    early = D[f'n_season_{s}'].to_numpy(float) < 3
    D[f'k_bench_in_{s}'] = np.where(~early, D[f'k_bench_usage_{s}'], np.nan)
base=['k_usage_in','k_gashare_in','k_exp']
ol('u_sh90', d('u_sh90'), avail=U)
ol('k_sh90', d('k_sh90'))
ol('k_sh90_rel', d('k_sh90_rel'))
ol('base3+k_sh90_rel', d(*base,'k_sh90_rel'))
ol('base3+u_sh90', d(*base,'u_sh90'))
ol('base3+bench', d(*base,'k_bench_usage'))
ol('base3+bench_in', d(*base,'k_bench_in'))
ol('base3+bench+sh90', d(*base,'k_bench_usage','k_sh90'))
ol('usage_in+bench', d('k_usage_in','k_bench_usage'))
ol('usage_in+bench_in', d('k_usage_in','k_bench_in'))
P=ol('base3+bench A-only', d(*base,'k_bench_usage'), train_lg=('A',))
P=ol('base3+bench', d(*base,'k_bench_usage'))
# per-season stability on DEV
P0=D[['pH','pD','pA']].to_numpy(float); k=D.k.to_numpy()
r1,_=lab.rps_ll(P,k); r0,_=lab.rps_ll(P0,k)
m=(D.scored&(D.season<=2021)).to_numpy()
print(pd.Series((r1-r0)[m]).groupby(D.season[m].to_numpy()).mean().round(5).to_dict())
