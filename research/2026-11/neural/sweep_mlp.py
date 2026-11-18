"""Pilot for the context-only residual MLP (DEV seasons only): what is the correction made of?
   lg/noextras = one constant shift of (log lh, log la) per league (4 numbers) fitted by the same machinery."""
import sys, json, itertools
from nn_lib import *
d = Data()
grid = [dict(ctx='lg', extras=False), dict(ctx='v1lam', extras=False), dict(ctx='v1lam'), dict(ctx='full'), dict(ctx='full', tau=4.0), dict(ctx='lg', extras=False, tau=4.0),
        dict(ctx='full', delta_pen=0.0), dict(ctx='full', w_rps=0.0, w_ce=1.0), dict(ctx='full', w_rps=0.0, w_pois=1.0)]
for re in (3, 1):
    for g in grid:
        cfg = {**DEFAULT, 'enc': 'none', **g}
        P, _ = run_walk(d, cfg, [0, 1, 2, 3, 4], seasons=range(2013, 2022), refit_every=re, verbose=False)
        summarize(d, P, f'mlp refit{re} ' + json.dumps(g))
