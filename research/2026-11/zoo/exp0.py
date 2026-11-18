"""Stage 0: sanity + plain recalibration of v1 (DEV only)."""
import numpy as np, pandas as pd, time
from models import *
F = load()
print(len(F), F.ev.sum())
Pv = F[['v_pH', 'v_pD', 'v_pA']].values
score(F, Pv, name='v1 (chain port)')
t = time.time()
for loss in ('cum', 'll', 'rps'):
    for fs in (False, True):
        P = walk(F, fp_co(loss=loss, free_scale=fs)); score(F, P, name=f'recal CO loss={loss} free_scale={fs}')
P = walk(F, fp_co(loss='cum', free_scale=True), useB=False); score(F, P, name='recal CO cum, A only')
for wB in (0.25, 1.0):
    P = walk(F, fp_co(loss='cum', free_scale=True, wB=wB)); score(F, P, name=f'recal CO cum, wB={wB}')
P = walk(F, fp_po(calib=False)); score(F, P, name='PO intercepts only, no calib')
P = walk(F, fp_po(calib=True)); score(F, P, name='PO intercepts only + calib')
P = walk(F, fp_po(feats=['v_dl'], sfeats=['v_sl'], calib=True)); score(F, P, name='PO v_dl,v_sl + calib')
print(time.time() - t)
