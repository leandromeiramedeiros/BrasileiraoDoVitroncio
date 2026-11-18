"""Coefficients to use for fixtures of the NEXT season / remaining fixtures: same ridge fit as the walk-forward,
on every finished Série A season from 2009 to date. Output: coef/production_ctx_core.json"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, fitlib as F, variants as V
D = F.load()
out = {}
for name in ('ctx_core', 'ctx_core_r'):
    # a pseudo "season 2027" with no rows: walk_forward fits on everything before it
    L, path = F.walk_forward(D, V.spec(*V.GROUPS[name]), tau=0.05, seasons=[int(D.season.max()) + 1])
    out[name] = {k: round(float(v), 5) for k, v in path.iloc[0].items()}
out['_how_to_apply'] = ('log lh\' = log lh_v1 + c_tot + c_hfa + decided*(dec_h - dec_a) + next_cont4*(nc_h - nc_a) + closed*closed + home_disp*disp [+ r_all*r_dh_all]; '
                        'log la\' = log la_v1 + c_tot - c_hfa - decided*(dec_h - dec_a) - next_cont4*(nc_h - nc_a) - closed*closed - home_disp*disp [+ r_all*r_da_all]')
json.dump(out, open(os.path.join(F.HERE, 'coef', 'production_ctx_core.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
