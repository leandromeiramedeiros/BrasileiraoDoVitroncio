"""Experiment driver.
  python run.py NAME [NAME ...] [--final] [--dev] [--seeds K]
    default   : screening protocol - weights refitted every 3 seasons (2013, 2016, 2019, 2022, 2025), 5 seeds
    --final   : final protocol - weights refitted before EVERY season 2013..2026, 10 seeds (saved as NAME_final)
    --dev     : only DEV seasons (2013-2021); used for hyper-parameter pilots (never touches HOLDOUT)
Every variant is a dict of overrides of nn_lib.DEFAULT.  Results are appended to results.jsonl."""
import sys, json, time
import numpy as np
from nn_lib import Data, DEFAULT, run_walk, summarize, save_run, D

B = ('base',); V = ('base', 'v1'); E = ('base', 'v1', 'ev'); ALL = ('base', 'v1', 'ev', 'st')
VARIANTS = {
    # ---- 3. residual on v1's lambdas (zero-init, default loss = RPS) ---------------------------------
    'mlp_resid':        dict(enc='none'),                                   # no history: context-only correction
    'ds_resid':         dict(enc='deepsets'),
    'gru_resid':        dict(enc='gru'),
    'cnn_resid':        dict(enc='cnn'),
    'tf_resid':         dict(enc='transformer'),
    # ---- 4. losses (same GRU residual net) ------------------------------------------------------------
    'gru_resid_ce':     dict(enc='gru', w_rps=0.0, w_ce=1.0),
    'gru_resid_pois':   dict(enc='gru', w_rps=0.0, w_pois=1.0),
    'gru_resid_multi':  dict(enc='gru', w_rps=1.0, w_pois=0.15, w_aux=0.15),
    'gru_resid_logit':  dict(enc='gru', head='logit_resid'),
    'gru_resid_logit_ce': dict(enc='gru', head='logit_resid', w_rps=0.0, w_ce=1.0),
    # ---- regularisation / capacity / history length -----------------------------------------------------
    'gru_resid_pen':    dict(enc='gru', delta_pen=0.3),
    'gru_resid_N10':    dict(enc='gru', N=10),
    'gru_resid_N40':    dict(enc='gru', N=40),
    'gru_resid_noev':   dict(enc='gru', groups=('base', 'v1', 'st')),
    'gru_resid_nost':   dict(enc='gru', groups=E),
    'gru_resid_v1lamctx': dict(enc='gru', ctx='v1lam'),
    # ---- 2. sequence encoders that are NOT anchored on v1 (free Poisson head) ----------------------------
    'gru_pure':         dict(enc='gru', head='free', groups=('base', 'ev', 'st'), ctx='pure', w_rps=1.0, w_pois=0.3),   # nothing from v1
    'gru_opp':          dict(enc='gru', head='free', groups=ALL, ctx='pure', w_rps=1.0, w_pois=0.3),                    # v1 only as opponent strength in tokens
    'gru_free':         dict(enc='gru', head='free', groups=ALL, ctx='full', w_rps=1.0, w_pois=0.3),                    # + v1 lambdas in the head
    'gru_pure_N40':     dict(enc='gru', head='free', groups=('base', 'ev', 'st'), ctx='pure', w_rps=1.0, w_pois=0.3, N=40),
    'gru_opp_N40':      dict(enc='gru', head='free', groups=ALL, ctx='pure', w_rps=1.0, w_pois=0.3, N=40),
}

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    final = '--final' in sys.argv; dev = '--dev' in sys.argv
    K = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else (10 if final else 5)
    if '--seeds' in sys.argv: args = [a for a in args if a != str(K)]
    extra = {}
    for a in list(args):                       # inline overrides, e.g.  lr=0.001  epochs=25  (pilot use)
        if '=' in a:
            k, v = a.split('='); extra[k] = type(DEFAULT[k])(v) if not isinstance(DEFAULT[k], tuple) else tuple(v.split(',')); args.remove(a)
    data = Data()
    for name in args:
        cfg = {**DEFAULT, **VARIANTS[name], **extra}
        tag = name + ''.join(f'_{k}{v}' for k, v in extra.items()) + ('_final' if final else '') + ('_dev' if dev else '')
        t0 = time.time()
        P, LAM = run_walk(data, cfg, list(range(K)), seasons=range(2013, 2022 if dev else 2027), refit_every=1 if final else 3, tag=tag)
        per, avg = summarize(data, P, tag, dev_only=dev)
        save_run(tag, data, P, LAM, cfg)
        rec = dict(tag=tag, cfg=cfg, secs=round(time.time() - t0), seeds=K, final=final,
                   per_seed={k: [float(x[k]['d']) for x in per] for k in avg}, avg={k: {q: float(v) for q, v in avg[k].items()} for k in avg})
        open(D + 'results.jsonl', 'a').write(json.dumps(rec) + '\n')
