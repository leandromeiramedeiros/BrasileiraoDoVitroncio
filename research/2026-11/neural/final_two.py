"""Frozen final runs (no sweeps).  usage: python final_two.py {deepsets|gru}
Per-season expanding window: for every season Y in 2013..2026 the 5 networks are fitted from scratch on finished
matches of seasons < Y (Série A + B, fixed 15-epoch budget, no early stopping) and then predict all of season Y.
Writes preds_neural_<name>.csv (seed-averaged probabilities, the 5,279 scored matches) and
shift_neural_<name>.csv (seed-mean shift of v1's log expected goals, every Série A match 2013+, incl. unplayed 2026)."""
import sys, numpy as np, pandas as pd
from nn_lib import Data, run_walk, save_run, score, D
BASE = dict(d=32, dh=32, N=20, drop=0.2, head='lam_resid', groups=('base', 'v1', 'ev', 'st'), ctx='full', w_rps=1.0, w_ce=0.0, w_pois=0.0, w_aux=0.0,
            wd=0.05, epochs=15, bs=512, wB=1.0, first_season=2007, extras=True, tau=0.0)
FROZEN = {'deepsets': dict(enc='deepsets', lr=3e-3, delta_pen=0.3), 'gru': dict(enc='gru', lr=1e-3, delta_pen=0.3)}
name = sys.argv[1]; cfg = {**BASE, **FROZEN[name]}
data = Data()
assert 'v1_loglf' in data.tok_names and len(data.ids) == 15203
P, LAM = run_walk(data, cfg, [0, 1, 2, 3, 4], seasons=range(2013, 2027), refit_every=1, tag=name)
save_run(f'final_{name}', data, P, LAM, cfg)
p = P.mean(0); p = p / p.sum(1, keepdims=True)
sc = data.post & (data.lgB == 0) & (data.season >= 2013)
assert sc.sum() == 5279 and np.isfinite(p[sc]).all()
pd.DataFrame({'id': data.ids[sc], 'pH': p[sc, 0], 'pD': p[sc, 1], 'pA': p[sc, 2]}).to_csv(D + f'preds_neural_{name}.csv', index=False, float_format='%.6f')
dl = np.log(LAM / np.stack([data.lh, data.la], 1)[None]).mean(0)
A = (data.lgB == 0) & (data.season >= 2013)
assert np.isfinite(dl[A]).all()
pd.DataFrame({'id': data.ids[A], 'd_log_lh': dl[A, 0], 'd_log_la': dl[A, 1]}).to_csv(D + f'shift_neural_{name}.csv', index=False, float_format='%.5f')
print('written', name, 'scored rows', int(sc.sum()), 'shift rows', int(A.sum()), 'unplayed A 2026 rows', int((A & ~data.post & (data.season == 2026)).sum()), flush=True)
