import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
D=E.D; K=E.K_AVAIL; d=E.d
base=['k_usage_in','k_gashare_in','k_exp']
P0=D[['pH','pD','pA']].to_numpy(float); k=D.k.to_numpy()
m=(D.scored&(D.season<=2021)).to_numpy()
def per_season(spec):
    P,_=lab.walk_forward(D, spec, K)
    r1,_=lab.rps_ll(P,k); r0,_=lab.rps_ll(P0,k)
    return pd.Series((r1-r0)[m]).groupby(D.season[m].to_numpy()).mean()
a=per_season(d(*base)); b=per_season(d(*base,'k_bench_usage')); c=per_season(d(*base,'k_bench_n'))
print(pd.DataFrame({'base3':a,'base3+bench':b,'incr_bench':b-a,'incr_bench_n':c-a}).round(5).to_string())
# completeness of the bench list: unused bench players per team-match
for s in 'ha':
    D[f'unused_{s}']=D[f'k_bench_n_{s}']-D[f'diag_n_used_{s}']
A=D[(D.lg=='A')&(D.season>=2010)&D.post]
t=pd.concat([A[['season','unused_h','k_bench_n_h']].rename(columns={'unused_h':'unused','k_bench_n_h':'n'}),A[['season','unused_a','k_bench_n_a']].rename(columns={'unused_a':'unused','k_bench_n_a':'n'})])
print(t.groupby('season').agg(n_mean=('n','mean'),unused_mean=('unused','mean'),unused0=('unused',lambda x:(x<=0).mean()),unused_le2=('unused',lambda x:(x<=2).mean()),unused_le4=('unused',lambda x:(x<=4).mean())).round(3).to_string())
