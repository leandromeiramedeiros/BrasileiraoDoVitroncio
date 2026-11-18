import numpy as np, pandas as pd
from models import *
F = load()
D = F[(F.season <= 2021) & (F.season >= 2009)]
g = D.groupby(['lg', 'season']).apply(lambda x: pd.Series(dict(n=len(x), H=(x.k == 0).mean(), pH=x.v_pH.mean(), D=(x.k == 1).mean(), pD=x.v_pD.mean(), A=(x.k == 2).mean(), pA=x.v_pA.mean(),
                                                                  hg=x.hg.mean(), lh=x.v_lh.mean(), ag=x.ag.mean(), la=x.v_la.mean())), include_groups=False).round(3)
print(g.to_string())
print(D.groupby('lg')[['hg', 'v_lh', 'ag', 'v_la']].mean().round(3))
P, C = walk(F, fp_co(loss='cum', free_scale=False), keep_coef=True)
print(pd.DataFrame(C).T.round(3).to_string())
print(by_season(F, P).to_string())
# reliability of v1 in DEV A by pH bucket
E = F[F.ev & (F.season <= 2021)]
b = pd.cut(E.v_pH, [0, .3, .4, .5, .6, 1])
print(E.groupby(b, observed=True).apply(lambda x: pd.Series(dict(n=len(x), pH=x.v_pH.mean(), H=(x.k == 0).mean(), pD=x.v_pD.mean(), D=(x.k == 1).mean(), pA=x.v_pA.mean(), A=(x.k == 2).mean())), include_groups=False).round(3))
