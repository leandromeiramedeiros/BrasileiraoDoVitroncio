"""Scores variants on DEV only (2013-2021).  usage: python code/run_dev.py name=block+block[:tau[:leagues]] ..."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import fitlib as F, variants as V
D = F.load()
os.makedirs(os.path.join(F.HERE, 'work'), exist_ok=True)
for arg in sys.argv[1:]:
    name, rhs = arg.split('=')
    parts = rhs.split(':'); blocks = [b for b in parts[0].split('+') if b]
    tau = float(parts[1]) if len(parts) > 1 and parts[1] else 0.02
    leagues = tuple(parts[2]) if len(parts) > 2 else ('A',)
    L, path = F.walk_forward(D, V.spec(*blocks), tau=tau, leagues=leagues)
    P = F.to_probs(L, os.path.join(F.HERE, 'work', f'preds_{name}.csv'))
    F.score_dev(P, f'{name} (tau={tau},{"".join(leagues)})')
    path.round(4).to_csv(os.path.join(F.HERE, 'work', f'coef_{name}.csv'))
