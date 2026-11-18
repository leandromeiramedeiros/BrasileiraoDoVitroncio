"""Walk-forward Poisson GLM on top of v1 (log lh / log la as offsets) + scoring helpers.

Feature kinds (one coefficient each):
  'diff' : team condition x (home value xh, away value xa):  log lam_h += b (xh - xa),  log lam_a -= b (xh - xa)
  'hfa'  : home-advantage modifier z:                         log lam_h += b z,          log lam_a -= b z
  'tot'  : total-goals modifier z:                            log lam_h += b z,          log lam_a += b z
  'pair' : side-specific log-rate shifts (vh, va):            log lam_h += b vh,         log lam_a += b va
Coefficients for season Y are estimated on seasons < Y only (expanding window), ridge-penalised.
"""
import os, sys, json, subprocess, tempfile
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = '/home/user/bdv-lab/shared/'

def load():
    B = pd.read_csv(SHARED + 'baseline_features.csv', dtype={'id': str}).set_index('id')
    X = pd.read_csv(os.path.join(HERE, 'features_context.csv'), dtype={'id': str}).set_index('id')
    R = pd.read_csv(os.path.join(HERE, 'features_ratings.csv'), dtype={'id': str}).set_index('id')
    D = B.join(X.drop(columns=['comp', 'season']), how='left').join(R, how='left')
    D['played'] = (D.status == 'post') & D.hg.notna()
    return D

def design(D, spec):
    """spec: list of (name, kind, fn) with fn(D) -> array (hfa/tot) or (arr_h, arr_a) (diff/pair). Returns Xh, Xa, names."""
    Xh, Xa, names = [], [], []
    for name, kind, fn in spec:
        v = fn(D)
        if kind == 'diff': d = np.asarray(v[0], float) - np.asarray(v[1], float); xh, xa = d, -d
        elif kind == 'hfa': z = np.asarray(v, float); xh, xa = z, -z
        elif kind == 'tot': z = np.asarray(v, float); xh, xa = z, z
        elif kind == 'pair': xh, xa = np.asarray(v[0], float), np.asarray(v[1], float)
        else: raise ValueError(kind)
        Xh.append(xh); Xa.append(xa); names.append(name)
    return np.column_stack(Xh), np.column_stack(Xa), names

def fit_poisson(X, y, off, pen, iters=50):
    """Ridge-penalised Poisson regression with offset (Newton). pen: per-coefficient L2 penalty."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = np.exp(off + X @ b)
        g = X.T @ (y - mu) - pen * b
        H = (X * mu[:, None]).T @ X + np.diag(pen)
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9: break
    mu = np.exp(off + X @ b)
    cov = np.linalg.inv((X * mu[:, None]).T @ X + np.diag(pen))
    return b, np.sqrt(np.diag(cov))

def prep(D, spec, train_mask):
    """Stack home/away rows for the training mask; standardise non-constant columns (train stats)."""
    Xh, Xa, names = design(D, spec)
    med = np.nanmedian(np.vstack([Xh[train_mask], Xa[train_mask]]), axis=0) if train_mask.sum() else np.zeros(Xh.shape[1])
    med = np.where(np.isnan(med), 0, med)
    Xh = np.where(np.isnan(Xh), med, Xh); Xa = np.where(np.isnan(Xa), med, Xa)
    return Xh, Xa, names

def walk_forward(D, spec, tau=0.02, start=2009, leagues=('A',), seasons=range(2013, 2027), consts=True, tau_const=1.0):
    """Returns DataFrame (id, lh, la) with adjusted expected goals for Série A matches of `seasons`, and the
    coefficient path (season -> coefficients in ORIGINAL feature units)."""
    spec = list(spec)
    if consts: spec = [('c_tot', 'tot', lambda d: np.ones(len(d))), ('c_hfa', 'hfa', lambda d: np.ones(len(d)))] + spec
    lg = D.lg.to_numpy(); season = D.season.to_numpy(); played = D.played.to_numpy()
    offh, offa = np.log(D.lh.to_numpy()), np.log(D.la.to_numpy())
    yh, ya = D.hg.to_numpy(float), D.ag.to_numpy(float)
    out = []; path = {}
    for Y in seasons:
        tr = played & np.isin(lg, leagues) & (season < Y) & (season >= start)
        te = (lg == 'A') & (season == Y)
        Xh, Xa, names = prep(D, spec, tr)
        Xtr = np.vstack([Xh[tr], Xa[tr]]); ytr = np.concatenate([yh[tr], ya[tr]]); otr = np.concatenate([offh[tr], offa[tr]])
        sd = Xtr.std(axis=0); isconst = np.array([n in ('c_tot', 'c_hfa') for n in names])
        sd = np.where(isconst | (sd < 1e-12), 1.0, sd)
        pen = np.where(isconst, 1 / tau_const ** 2, 1 / tau ** 2)
        if tr.sum() >= 300:
            b, se = fit_poisson(Xtr / sd, ytr, otr, pen)
            b = b / sd; se = se / sd
        else:
            b = np.zeros(len(names)); se = np.full(len(names), np.nan)
        path[Y] = dict(zip(names, b))
        out.append(pd.DataFrame({'id': D.index[te], 'lh': np.exp(offh[te] + Xh[te] @ b), 'la': np.exp(offa[te] + Xa[te] @ b)}))
    return pd.concat(out, ignore_index=True), pd.DataFrame(path).T

def insample(D, spec, max_season=2021, start=2009, leagues=('A',), tau=1.0):
    """Coefficient table (original units, with z-scores) fitted on seasons start..max_season - evidence of effect size."""
    spec = [('c_tot', 'tot', lambda d: np.ones(len(d))), ('c_hfa', 'hfa', lambda d: np.ones(len(d)))] + list(spec)
    tr = (D.played & D.lg.isin(leagues) & (D.season <= max_season) & (D.season >= start)).to_numpy()
    Xh, Xa, names = prep(D, spec, tr)
    Xtr = np.vstack([Xh[tr], Xa[tr]]); ytr = np.concatenate([D.hg.to_numpy(float)[tr], D.ag.to_numpy(float)[tr]])
    otr = np.concatenate([np.log(D.lh.to_numpy())[tr], np.log(D.la.to_numpy())[tr]])
    sd = Xtr.std(axis=0); sd = np.where(sd < 1e-12, 1.0, sd)
    b, se = fit_poisson(Xtr / sd, ytr, otr, np.full(len(names), 1 / tau ** 2))
    return pd.DataFrame({'coef': b / sd, 'se': se / sd, 'z': b / se, 'effect_per_sd': b, 'sd': sd}, index=names)

def to_probs(L, out_csv):
    """(id, lh, la) -> id,pH,pD,pA through the v1 Markov-chain match model (node)."""
    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        L[['id', 'lh', 'la']].to_csv(f, index=False); tmp = f.name
    subprocess.run(['node', os.path.join(HERE, 'code', 'probs.mjs'), tmp, out_csv], check=True)
    os.unlink(tmp)
    return pd.read_csv(out_csv, dtype={'id': str})

# ---------------------------------------------------------------- scoring (DEV only while iterating)
_M = None
def _truth():
    global _M
    if _M is None:
        M = pd.DataFrame(json.load(open(SHARED + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
        M = M[(M.lg == 'A') & (M.season >= 2013) & (M.status == 'post')].copy(); M['id'] = M['id'].astype(str)
        M['k'] = np.where(M.hg > M.ag, 0, np.where(M.hg == M.ag, 1, 2))
        base = pd.read_csv(SHARED + 'preds_baseline.csv', dtype={'id': str})
        _M = (M, base)
    return _M

def rps_ll(df):
    p = np.clip(df[['pH', 'pD', 'pA']].to_numpy(float), 1e-9, 1); p = p / p.sum(1, keepdims=True)
    k = df['k'].to_numpy(); y = np.eye(3)[k]
    rps = ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2
    return rps, -np.log(p[np.arange(len(k)), k])

def score_dev(pred, label='', ref=None, extra_masks=None, quiet=False):
    """DEV-only (2013-2021) score against v1 (or against another prediction frame `ref`). Never touches HOLDOUT."""
    M, base = _truth()
    d = M.merge(pred, on='id'); d = d[d.season <= 2021]
    b = M.merge(base if ref is None else ref, on='id').set_index('id').loc[d['id']].reset_index()
    r, l = rps_ll(d); rb, lb = rps_ll(b)
    rng = np.random.default_rng(7); diff = r - rb
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    res = {'label': label, 'n': len(d), 'rps': r.mean(), 'll': l.mean(), 'drps': diff.mean(), 'lo': np.percentile(bs, 2.5), 'hi': np.percentile(bs, 97.5), 'dll': (l - lb).mean()}
    # by season sign consistency
    by = pd.Series(diff).groupby(d.season.to_numpy()).mean()
    res['seasons_better'] = int((by < 0).sum()); res['seasons'] = len(by)
    if not quiet:
        print(f"{label:34s} DEV n={res['n']} RPS {res['rps']:.5f} LL {res['ll']:.5f} dRPS {res['drps']:+.5f} [{res['lo']:+.5f},{res['hi']:+.5f}] dLL {res['dll']:+.5f}  seasons better {res['seasons_better']}/{res['seasons']}")
    return res
