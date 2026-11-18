import numpy as np, pandas as pd
from models import *
F = load()
ref = walk(F, fp_co(loss='cum', free_scale=False))
for f in ['gapsot4_d', 'v_shd']:
    P, C = walk(F, fp_co(feats=[f], free_scale=False, lam=0.01), keep_coef=True)
    print(f); print(pd.DataFrame(C).T.round(4).T.to_string())
    print(by_season(F, P, ref=ref).to_string())
    print(F[F.season <= 2021].groupby(['lg', 'season'])[f].agg(['std', 'min', 'max']).round(3).T.to_string())
