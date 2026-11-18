"""Regularisation pilot for sequence residual models (DEV seasons only, screening refits 2013/2016/2019).
usage: python sweep_seq.py ENC 'json-overrides' ['json-overrides' ...]"""
import sys, json
from nn_lib import *
d = Data()
enc = sys.argv[1]
for g in sys.argv[2:]:
    g = json.loads(g)
    cfg = {**DEFAULT, 'enc': enc, **g}
    t0 = time.time()
    P, _ = run_walk(d, cfg, [0, 1, 2, 3, 4], seasons=range(2013, 2022), refit_every=3, verbose=False)
    summarize(d, P, f'{enc} {json.dumps(g)} [{time.time() - t0:.0f}s]')
