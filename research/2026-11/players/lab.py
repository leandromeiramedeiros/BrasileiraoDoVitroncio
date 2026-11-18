"""Shared helpers: load features, walk-forward ordered-logit shift on top of v1, scoring (DEV only by default)."""
import json, os
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED = '/home/user/bdv-lab/shared/'
EPS = 1e-9


def load():
    M = pd.DataFrame(json.load(open(SHARED + 'matches.json')))[['id', 'lg', 'season', 'date', 'status', 'h', 'a', 'hg', 'ag']]
    M['id'] = M['id'].astype(str)
    base = pd.read_csv(f'{ROOT}/work/base_all.csv', dtype={'id': str})
    off = pd.read_csv(SHARED + 'preds_baseline.csv', dtype={'id': str}).set_index('id')
    M = M.merge(base, on='id', how='inner')
    # use the official baseline numbers wherever they exist (identical up to rounding)
    ix = M['id'].isin(off.index)
    M.loc[ix, ['pH', 'pD', 'pA']] = off.loc[M.loc[ix, 'id']].to_numpy()
    bf = pd.read_csv(SHARED + 'baseline_features.csv', dtype={'id': str})[['id', 'lh', 'la', 'nh', 'na']]
    M = M.merge(bf, on='id', how='left')
    F = pd.read_csv(f'{ROOT}/work/side_features.csv', dtype={'id': str, 'tid': str})
    drop = ['tid', 'lg', 'season', 'date', 'status', 'side']
    H = F[F.side == 'h'].drop(columns=drop).set_index('id').add_suffix('_h')
    A = F[F.side == 'a'].drop(columns=drop).set_index('id').add_suffix('_a')
    D = pd.concat([M.set_index('id'), H.reindex(M['id']), A.reindex(M['id'])], axis=1).reset_index().copy()
    D['k'] = np.where(D.hg > D.ag, 0, np.where(D.hg == D.ag, 1, 2))
    D['post'] = D.status == 'post'
    D['scored'] = (D.lg == 'A') & (D.season >= 2013) & D.post
    return D


def shift_probs(P, delta):
    """Ordered-logit shift of base probabilities P (n,3 = H,D,A) by delta (positive = towards home)."""
    P = np.clip(P, 1e-6, 1); P = P / P.sum(1, keepdims=True)
    c1 = logit(P[:, 2]); c2 = logit(P[:, 2] + P[:, 1])
    pa = expit(c1 - delta); pad = expit(c2 - delta)
    return np.column_stack([1 - pad, pad - pa, pa])


def nll(beta, X, P, k, l2):
    Q = shift_probs(P, X @ beta)
    return -np.log(np.clip(Q[np.arange(len(k)), k], EPS, 1)).sum() + l2 * (beta ** 2).sum()


def fit_shift(X, P, k, l2):
    res = minimize(nll, np.zeros(X.shape[1]), args=(X, P, k, l2), method='L-BFGS-B')
    return res.x


def rps_ll(P, k):
    P = np.clip(P, 1e-9, 1); P = P / P.sum(1, keepdims=True)
    y = np.eye(3)[k]
    rps = ((P[:, 0] - y[:, 0]) ** 2 + (P[:, 0] + P[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2
    return rps, -np.log(P[np.arange(len(k)), k])


def make_X(D, spec, train_mask, apply_spec=None):
    """spec: list of (name, kind) with kind in {'diff','h','a','raw'}; NaN -> training mean (per side), then z-scored on training rows.
    apply_spec (optional, same length): columns to read the VALUES from (centred/scaled with the statistics of `spec` on the
    training rows) - used to plug history-based predictions of a lineup feature into coefficients fitted on the real one."""
    cols, names = [], []
    for i, (name, kind) in enumerate(spec):
        def build(nm, mu=None, sd=None):
            if kind == 'raw':
                v = D[nm].to_numpy(float)
                if mu is None: mu = np.nanmean(v[train_mask])
                x = np.where(np.isnan(v), mu, v) - mu
            else:
                h = D[nm + '_h'].to_numpy(float); a = D[nm + '_a'].to_numpy(float)
                if mu is None: mu = np.nanmean(np.concatenate([h[train_mask], a[train_mask]]))
                h = np.where(np.isnan(h), mu, h); a = np.where(np.isnan(a), mu, a)
                x = h - a if kind == 'diff' else (h - mu if kind == 'h' else a - mu)
            if sd is None: sd = x[train_mask].std() or 1.0
            return x / sd, mu, sd
        x, mu, sd = build(name)
        if apply_spec is not None:
            x, _, _ = build(apply_spec[i][0], mu, sd)
        cols.append(x); names.append(f'{name}:{kind}')
    return np.column_stack(cols), names


def walk_forward(D, spec, avail, l2=5.0, first_train=2008, train_lg=('A', 'B'), fallback=None, verbose=False, min_train=600,
                 apply_spec=None, apply_avail=None):
    """Expanding-window fit by season. `avail`: boolean array (rows where the features exist).
    Returns (P_new for all rows, dict season->coefs). Rows not available keep `fallback` (default: v1 baseline).
    With apply_spec/apply_avail the coefficients are fitted on `spec` but applied to the columns of `apply_spec`."""
    P0 = D[['pH', 'pD', 'pA']].to_numpy(float)
    Pn = P0.copy() if fallback is None else fallback.copy()
    k = D['k'].to_numpy()
    season = D['season'].to_numpy()
    coefs = {}
    te_av = avail if apply_avail is None else apply_avail
    for Y in range(2013, 2027):
        tr = avail & D.post.to_numpy() & (season < Y) & (season >= first_train) & D.lg.isin(train_lg).to_numpy()
        te = te_av & (season == Y)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        X, names = make_X(D, spec, tr)
        beta = fit_shift(X[tr], P0[tr], k[tr], l2)
        Xa = X if apply_spec is None else make_X(D, spec, tr, apply_spec)[0]
        Pn[te] = shift_probs(P0[te], Xa[te] @ beta)
        coefs[Y] = dict(zip(names, np.round(beta, 4)))
        if verbose: print(Y, tr.sum(), coefs[Y])
    return Pn, coefs


def stage1(D, target, preds, first_train=2008, l2=1.0, suffix='_hat'):
    """Walk-forward ridge regression (side level) of a lineup-known feature on history-only predictors.
    Adds columns <target><suffix>_h/_a holding out-of-sample predictions (NaN where predictors are all missing)."""
    season = D['season'].to_numpy(); post = D.post.to_numpy()
    out = {s: np.full(len(D), np.nan) for s in 'ha'}
    r2 = {}
    for Y in range(2011, 2027):
        Xs, ys, ms = {}, {}, {}
        tr_rows = post & (season < Y) & (season >= first_train)
        allX = np.vstack([D.loc[tr_rows, [p + '_' + s for p in preds]].to_numpy(float) for s in 'ha'])
        ally = np.concatenate([D.loc[tr_rows, target + '_' + s].to_numpy(float) for s in 'ha'])
        ok = ~np.isnan(ally) & ~np.isnan(allX).all(1)
        if ok.sum() < 500: continue
        mu = np.nanmean(allX[ok], 0)
        Xtr = np.where(np.isnan(allX[ok]), mu, allX[ok]) - mu
        sd = Xtr.std(0); sd[sd == 0] = 1
        Xtr = Xtr / sd; ytr = ally[ok]; ym = ytr.mean()
        b = np.linalg.solve(Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1]), Xtr.T @ (ytr - ym))
        te = season == Y
        for s in 'ha':
            Xte = D.loc[te, [p + '_' + s for p in preds]].to_numpy(float)
            miss = np.isnan(Xte).all(1)
            Xte = (np.where(np.isnan(Xte), mu, Xte) - mu) / sd
            pr = ym + Xte @ b
            pr[miss] = np.nan
            out[s][te] = pr
        yt = np.concatenate([D.loc[te, target + '_' + s].to_numpy(float) for s in 'ha']); pt = np.concatenate([out[s][te] for s in 'ha'])
        okk = ~np.isnan(yt) & ~np.isnan(pt)
        if okk.sum() > 50: r2[Y] = 1 - ((yt[okk] - pt[okk]) ** 2).sum() / ((yt[okk] - yt[okk].mean()) ** 2).sum()
    for s in 'ha':
        D[target + suffix + '_' + s] = out[s]
    return r2


def score(D, Pn, label='', holdout=False, sub=None, quiet=False):
    """Paired RPS difference vs baseline. DEV only unless holdout=True. `sub`: optional row mask for a subset report."""
    P0 = D[['pH', 'pD', 'pA']].to_numpy(float); k = D['k'].to_numpy()
    out = {}
    rng = np.random.default_rng(7)
    parts = [('DEV', D.scored & (D.season <= 2021))]
    if holdout: parts += [('HOLD', D.scored & (D.season >= 2022)), ('ALL', D.scored)]
    for lab, mask in parts:
        mask = mask.to_numpy()
        for sl, mm in (('', mask), ('sub', mask & sub if sub is not None else None)):
            if mm is None: continue
            r1, l1 = rps_ll(Pn[mm], k[mm]); r0, l0 = rps_ll(P0[mm], k[mm])
            d = r1 - r0
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
            out[lab + sl] = (mm.sum(), r1.mean(), l1.mean(), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (l1 - l0).mean())
    if not quiet:
        s = ' | '.join(f'{kk} n={v[0]} RPS {v[1]:.5f} d {v[3]:+.5f} [{v[4]:+.5f},{v[5]:+.5f}] dLL {v[6]:+.5f}' for kk, v in out.items())
        print(f'{label:38s} {s}')
    return out


def write_preds(D, Pn, name):
    S = D[D.scored]
    out = pd.DataFrame({'id': S['id'], 'pH': Pn[S.index, 0], 'pD': Pn[S.index, 1], 'pA': Pn[S.index, 2]})
    out.to_csv(f'{ROOT}/preds_{name}.csv', index=False, float_format='%.6f')
    return f'{ROOT}/preds_{name}.csv'


# ----------------------------------------------------------------------------------------------------------------------
# Poisson regression with the v1 expected goals as offset: lambda' = lambda_v1 * exp(b_own * x_own + b_opp * x_opp)
# ----------------------------------------------------------------------------------------------------------------------
def _pois_nll(beta, Zh, Za, lh, la, hg, ag, l2):
    eh = lh * np.exp(Zh @ beta); ea = la * np.exp(Za @ beta)
    return (eh - hg * np.log(eh)).sum() + (ea - ag * np.log(ea)).sum() + l2 * (beta ** 2).sum()


def lam2prob(ids, lh, la, tag='tmp'):
    import subprocess
    fi, fo = f'{ROOT}/work/_l2p_{tag}_in.csv', f'{ROOT}/work/_l2p_{tag}_out.csv'
    pd.DataFrame({'id': ids, 'lh': lh, 'la': la}).to_csv(fi, index=False)
    subprocess.run(['node', f'{ROOT}/lam2prob.mjs', fi, fo], check=True)
    return pd.read_csv(fo, dtype={'id': str})[['pH', 'pD', 'pA']].to_numpy(float)


def walk_forward_poisson(D, feats, avail, l2=5.0, first_train=2008, train_lg=('A', 'B'), min_train=600, intercept=False):
    """feats: list of per-side feature names (columns <name>_h / <name>_a). Each gets an 'own' and an 'opp' coefficient."""
    lh0, la0 = D.lh.to_numpy(float), D.la.to_numpy(float)
    lh, la = lh0.copy(), la0.copy()
    hg, ag = D.hg.to_numpy(float), D.ag.to_numpy(float)
    season = D['season'].to_numpy(); post = D.post.to_numpy(); lgok = D.lg.isin(train_lg).to_numpy()
    coefs = {}
    for Y in range(2013, 2027):
        tr = avail & post & (season < Y) & (season >= first_train) & lgok
        te = avail & (season == Y)
        if tr.sum() < min_train or te.sum() == 0:
            continue
        Zh, Za, names = [], [], []
        for name in feats:
            h = D[name + '_h'].to_numpy(float); a = D[name + '_a'].to_numpy(float)
            mu = np.nanmean(np.concatenate([h[tr], a[tr]]))
            h = np.where(np.isnan(h), mu, h) - mu; a = np.where(np.isnan(a), mu, a) - mu
            sd = np.concatenate([h[tr], a[tr]]).std() or 1.0
            h, a = h / sd, a / sd
            Zh += [h, a]; Za += [a, h]; names += [name + ':own', name + ':opp']
        if intercept:
            Zh.append(np.ones(len(D))); Za.append(np.ones(len(D))); names.append('const')
        Zh, Za = np.column_stack(Zh), np.column_stack(Za)
        beta = minimize(_pois_nll, np.zeros(Zh.shape[1]), args=(Zh[tr], Za[tr], lh0[tr], la0[tr], hg[tr], ag[tr], l2), method='L-BFGS-B').x
        lh[te] = lh0[te] * np.exp(Zh[te] @ beta); la[te] = la0[te] * np.exp(Za[te] @ beta)
        coefs[Y] = dict(zip(names, np.round(beta, 4)))
    Pn = D[['pH', 'pD', 'pA']].to_numpy(float).copy()
    ch = (lh != lh0) | (la != la0)
    if ch.any():
        Pn[ch] = lam2prob(D['id'].to_numpy()[ch], lh[ch], la[ch])
    return Pn, coefs, lh, la
