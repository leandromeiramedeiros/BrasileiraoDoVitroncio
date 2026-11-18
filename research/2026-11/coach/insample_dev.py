"""In-sample (DEV-era, Série A 2009-2021) single-feature coefficients with SEs; early vs late season split."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/bdv-lab/coach/code'); sys.path.insert(0, '/home/user/bdv-lab/context/code')
import fit as FT
D = FT.load()
s1 = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
out = []
for n in ['new1', 'new3', 'new5', 'interim', 'sacked7', 'chg7', 'logm_coach', 'logm_vac', 'n_chg', 'multi', 'new5_bad', 'new5_late']:
    t = FT.insample_noconst(D, [n], 2009, s1); out.append(t)
print(pd.concat(out).round(4).to_string())
# early vs late: split new3 / new5 by team's match number this season
for c in ['new3', 'new5', 'interim']:
    for lab, lo, hi in [('early(<12)', 0, 12), ('late(>=12)', 12, 99)]:
        FT.FEATS['tmp'] = ('tmp', 'diff', lambda D, c=c, lo=lo, hi=hi: tuple(D[s + c].to_numpy(float) * ((D[s + 'mseason'] >= lo) & (D[s + 'mseason'] < hi)).to_numpy() for s in ('h_', 'a_')))
        t = FT.insample_noconst(D, ['tmp'], 2009, s1); t.index = [f'{c} {lab}']; print(t.round(4).to_string(header=False))
