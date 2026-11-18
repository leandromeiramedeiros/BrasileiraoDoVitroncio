"""Meta-learners (all fitted walk-forward by season) and the DEV-only evaluation harness."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from lib import ChainGrid, rps_ll, boot_ci, Z, S

GRID = ChainGrid()
EVAL_SEASONS = list(range(2013, 2027))


def load():
    F = pd.read_pickle(Z + 'features_all.pkl')
    F = F[F.status == 'post'].reset_index(drop=True)
    F['k'] = np.where(F.hg > F.ag, 0, np.where(F.hg == F.ag, 1, 2))
    F['isA'] = (F.lg == 'A').astype(float)
    F['ev'] = (F.lg == 'A') & (F.season >= 2013)
    return F


# --------------------------------------------------------------------------- cumulative-offset logit
def _co_unpack(th, p, q, free_scale):
    d1, d2 = th[0], th[1]
    a1, a2 = (th[2], th[3]) if free_scale else (0.0, 0.0)
    b = th[4:4 + p]; g = th[4 + p:4 + p + q]
    return d1, d2, a1, a2, b, g


def co_fit(o1, o2, X, D, k, w, lam=0.02, lam_d=None, loss='cum', free_scale=True, lam_s=0.0):
    """logit P(H) = (1+a1) o1 + d1 + X b - D g ;  logit P(H or D) = (1+a2) o2 + d2 + X b + D g.
    o1/o2 = v1's own cumulative logits (the model reproduces v1 exactly at theta = 0).
    X: antisymmetric (strength) features, D: draw-propensity features. Ridge on standardised b, g."""
    n, p = X.shape; q = D.shape[1]
    lam_d = lam if lam_d is None else lam_d
    y1 = (k == 0).astype(float); y2 = (k <= 1).astype(float)
    w = w / w.sum()
    pen = np.r_[0, 0, lam_s, lam_s, np.full(p, lam), np.full(q, lam_d)]

    def f(th):
        d1, d2, a1, a2, b, g = _co_unpack(th, p, q, free_scale)
        xb = X @ b if p else 0.0; dg = D @ g if q else 0.0
        l1 = (1 + a1) * o1 + d1 + xb - dg; l2 = (1 + a2) * o2 + d2 + xb + dg
        s1 = expit(l1); s2 = expit(l2)
        if loss == 'cum':
            L = -(w * (y1 * np.log(s1 + 1e-12) + (1 - y1) * np.log(1 - s1 + 1e-12) + y2 * np.log(s2 + 1e-12) + (1 - y2) * np.log(1 - s2 + 1e-12))).sum()
            g1 = w * (s1 - y1); g2 = w * (s2 - y2)
        elif loss == 'rps':
            L = (w * ((s1 - y1) ** 2 + (s2 - y2) ** 2)).sum() / 2
            g1 = w * (s1 - y1) * s1 * (1 - s1); g2 = w * (s2 - y2) * s2 * (1 - s2)
        else:  # multinomial log-likelihood
            pd_ = np.maximum(s2 - s1, 1e-6); yd = y2 - y1; ya = 1 - y2
            L = -(w * (y1 * np.log(s1 + 1e-12) + yd * np.log(pd_) + ya * np.log(1 - s2 + 1e-12))).sum()
            g1 = -w * (y1 * (1 - s1) - yd * s1 * (1 - s1) / pd_); g2 = -w * (yd * s2 * (1 - s2) / pd_ - ya * s2)
        gr = np.zeros_like(th)
        gr[0] = g1.sum(); gr[1] = g2.sum()
        if free_scale: gr[2] = (g1 * o1).sum(); gr[3] = (g2 * o2).sum()
        if p: gr[4:4 + p] = X.T @ (g1 + g2)
        if q: gr[4 + p:] = D.T @ (g2 - g1)
        return L + 0.5 * (pen * th * th).sum(), gr + pen * th

    r = minimize(f, np.zeros(4 + p + q), jac=True, method='L-BFGS-B', options=dict(maxiter=500))
    return r.x


def co_predict(th, o1, o2, X, D, free_scale=True):
    p = X.shape[1]; q = D.shape[1]
    d1, d2, a1, a2, b, g = _co_unpack(th, p, q, free_scale)
    xb = X @ b if p else 0.0; dg = D @ g if q else 0.0
    l1 = (1 + a1) * o1 + d1 + xb - dg; l2 = (1 + a2) * o2 + d2 + xb + dg
    l2 = np.maximum(l2, l1 + 1e-4)
    s1 = expit(l1); s2 = expit(l2)
    return np.stack([s1, s2 - s1, 1 - s2], 1)


def offsets(P):
    P = np.clip(P, 1e-6, 1)
    P = P / P.sum(1, keepdims=True)
    return logit(P[:, 0]), logit(np.clip(P[:, 0] + P[:, 1], 1e-6, 1 - 1e-6))


class Std:
    """Scale columns by the training std (no centring: 0 means 'no information' for every zoo feature)."""

    def fit(self, X):
        # RMS over the rows where the feature is available (non-zero): a feature that only exists for part of the
        # training window keeps its natural scale and is automatically shrunk harder by the ridge.
        if X.shape[1] == 0: self.s = np.zeros(0); return self
        nz = (X != 0); cnt = np.maximum(nz.sum(0), 1)
        s = np.sqrt((X ** 2).sum(0) / cnt)
        self.s = np.where((s > 1e-9) & (nz.sum(0) >= 30), s, 1e9 if True else 1.0); return self

    def __call__(self, X):
        return X / self.s if X.shape[1] else X


def weights(tr, wB=0.5, decay=None, Y=None):
    w = np.where(tr.lg.values == 'A', 1.0, wB)
    if decay: w = w * decay ** (Y - 1 - tr.season.values)
    return w


def fp_co(feats=(), dfeats=(), lam=0.02, lam_d=None, loss='cum', wB=0.5, free_scale=True, decay=None, base=('v_pH', 'v_pD', 'v_pA'), lam_s=0.0, tau=None):
    """fit/predict closure: cumulative-offset logit on top of base probabilities."""
    feats, dfeats = list(feats), list(dfeats)

    def fp(tr, te, Y):
        o1, o2 = offsets(tr[list(base)].values); t1, t2 = offsets(te[list(base)].values)
        sx = Std().fit(tr[feats].values); sd = Std().fit(tr[dfeats].values)
        w = weights(tr, wB, decay, Y); l = lam if tau is None else 1.0 / (tau * tau * w.sum())
        th = co_fit(o1, o2, sx(tr[feats].values), sd(tr[dfeats].values), tr.k.values, w, l, l if tau is not None else lam_d, loss, free_scale, lam_s)
        fp.coef = dict(zip(['d1', 'd2', 'a1', 'a2'] + feats + dfeats, th))
        return co_predict(th, t1, t2, sx(te[feats].values), sd(te[dfeats].values), free_scale)
    return fp


# --------------------------------------------------------------------------- Poisson offset GLM on goals
def po_fit(lh, la, Xa, Xs, hg, ag, w, lam=0.02):
    """log mu_h = log lh + c_h + Xa b + Xs z ; log mu_a = log la + c_a - Xa b + Xs z  (ridge on b, z)."""
    p = Xa.shape[1]; q = Xs.shape[1]
    w = w / w.sum() / 2
    pen = np.r_[0, 0, np.full(p + q, lam)]
    llh, lla = np.log(lh), np.log(la)

    def f(th):
        b = th[2:2 + p]; z = th[2 + p:]
        xa = Xa @ b if p else 0.0; xs = Xs @ z if q else 0.0
        eh = llh + th[0] + xa + xs; ea = lla + th[1] - xa + xs
        mh = np.exp(eh); ma = np.exp(ea)
        L = (w * (mh - hg * eh + ma - ag * ea)).sum()
        rh = w * (mh - hg); ra = w * (ma - ag)
        gr = np.zeros_like(th); gr[0] = rh.sum(); gr[1] = ra.sum()
        if p: gr[2:2 + p] = Xa.T @ (rh - ra)
        if q: gr[2 + p:] = Xs.T @ (rh + ra)
        return L + 0.5 * (pen * th * th).sum(), gr + pen * th

    r = minimize(f, np.zeros(2 + p + q), jac=True, method='L-BFGS-B', options=dict(maxiter=500))
    return r.x


def po_lambda(th, lh, la, Xa, Xs):
    p = Xa.shape[1]
    b = th[2:2 + p]; z = th[2 + p:]
    xa = Xa @ b if p else 0.0; xs = Xs @ z if Xs.shape[1] else 0.0
    return lh * np.exp(th[0] + xa + xs), la * np.exp(th[1] - xa + xs)


def fp_po(feats=(), sfeats=(), lam=0.02, wB=0.5, calib=True, decay=None, cal_loss='cum', free_scale=False, tau=None):
    """Poisson offset GLM on goals -> v1 Markov chain -> (optional) cumulative-logit recalibration fitted in-sample."""
    feats, sfeats = list(feats), list(sfeats)

    def fp(tr, te, Y):
        sa = Std().fit(tr[feats].values); ss = Std().fit(tr[sfeats].values)
        w = weights(tr, wB, decay, Y)
        l = lam if tau is None else 1.0 / (tau * tau * w.sum() * 2)
        th = po_fit(tr.v_lh.values, tr.v_la.values, sa(tr[feats].values), ss(tr[sfeats].values), tr.hg.values, tr.ag.values, w, l)
        fp.coef = dict(zip(['c_h', 'c_a'] + feats + sfeats, th))
        lh, la = po_lambda(th, te.v_lh.values, te.v_la.values, sa(te[feats].values), ss(te[sfeats].values))
        fp.lam = (lh, la)
        P = GRID(lh, la)
        if calib:
            lht, lat = po_lambda(th, tr.v_lh.values, tr.v_la.values, sa(tr[feats].values), ss(tr[sfeats].values))
            Pt = GRID(lht, lat); o1, o2 = offsets(Pt); E = np.zeros((len(tr), 0))
            thc = co_fit(o1, o2, E, E, tr.k.values, w, 0, 0, cal_loss, free_scale)
            t1, t2 = offsets(P); E = np.zeros((len(te), 0))
            P = co_predict(thc, t1, t2, E, E, free_scale)
        return P
    return fp


# --------------------------------------------------------------------------- harness
def walk(F, fp, seasons=EVAL_SEASONS, train_from=2009, useB=True, keep_coef=False, ev=None):
    """Expanding window by season: the model that predicts season Y is fitted on seasons train_from..Y-1 only."""
    out = np.full((len(F), 3), np.nan); coefs = {}
    for Y in seasons:
        tr = F[(F.season < Y) & (F.season >= train_from)]
        if not useB: tr = tr[tr.lg == 'A']
        te = F[(F.season == Y) & (F.ev if ev is None else ev)]
        if len(te) == 0: continue
        out[te.index.values] = fp(tr, te, Y)
        if keep_coef and hasattr(fp, 'coef'): coefs[Y] = dict(fp.coef)
    return (out, coefs) if keep_coef else out


def score(F, P, base=None, name='', seasons=(2013, 2021), verbose=True, ref=None):
    """DEV-only scoring (HOLDOUT is deliberately not shown here). Returns (rps, drps_vs_v1)."""
    m = (F.ev & (F.season >= seasons[0]) & (F.season <= seasons[1])).values
    k = F.k.values[m]
    r, l = rps_ll(P[m], k)
    rb, lb = rps_ll(F[['v_pH', 'v_pD', 'v_pA']].values[m], k)
    d = r - rb; lo, hi = boot_ci(d, 1000)
    s = f'{name:44s} RPS {r.mean():.5f} LL {l.mean():.5f} | d vs v1 {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]'
    if ref is not None:
        rr, _ = rps_ll(ref[m], k); d2 = r - rr; lo2, hi2 = boot_ci(d2, 1000)
        s += f' | d vs ref {d2.mean():+.5f} [{lo2:+.5f},{hi2:+.5f}]'
    if verbose: print(s, flush=True)
    return r.mean(), d.mean()


def by_season(F, P, ref=None, seasons=(2013, 2021)):
    m = (F.ev & (F.season >= seasons[0]) & (F.season <= seasons[1])).values
    k = F.k.values[m]; r, _ = rps_ll(P[m], k)
    rb, _ = rps_ll((F[['v_pH', 'v_pD', 'v_pA']].values if ref is None else ref)[m], k)
    return pd.Series(r - rb).groupby(F.season.values[m]).mean().round(5)


def save_preds(F, P, name):
    m = F.ev.values
    d = pd.DataFrame(dict(id=F.id.values[m], pH=P[m, 0], pD=P[m, 1], pA=P[m, 2]))
    assert d[['pH', 'pD', 'pA']].notna().all().all() and len(d) == 5279
    d.to_csv(Z + f'preds_{name}.csv', index=False, float_format='%.6f')
    return Z + f'preds_{name}.csv'


# --------------------------------------------------------------------------- multinomial logit with offsets
def mn_fit(O, X, D, k, w, lam=0.02):
    """eta_H = O_H + c_H + X b ; eta_D = O_D + c_D + D g ; eta_A = O_A - X b   (softmax)."""
    n, p = X.shape; q = D.shape[1]
    w = w / w.sum(); Y = np.eye(3)[k]
    pen = np.r_[0, 0, np.full(p + q, lam)]

    def f(th):
        b = th[2:2 + p]; g = th[2 + p:]
        xb = X @ b if p else np.zeros(n); dg = D @ g if q else np.zeros(n)
        E = O + np.stack([th[0] + xb, th[1] + dg, -xb], 1)
        E = E - E.max(1, keepdims=True); P = np.exp(E); P /= P.sum(1, keepdims=True)
        L = -(w * np.log(P[np.arange(n), k] + 1e-12)).sum()
        G = w[:, None] * (P - Y)
        gr = np.zeros_like(th); gr[0] = G[:, 0].sum(); gr[1] = G[:, 1].sum()
        if p: gr[2:2 + p] = X.T @ (G[:, 0] - G[:, 2])
        if q: gr[2 + p:] = D.T @ G[:, 1]
        return L + 0.5 * (pen * th * th).sum(), gr + pen * th
    return minimize(f, np.zeros(2 + p + q), jac=True, method='L-BFGS-B', options=dict(maxiter=500)).x


def fp_mn(feats=(), dfeats=(), lam=0.02, wB=0.5, base=('v_pH', 'v_pD', 'v_pA')):
    feats, dfeats = list(feats), list(dfeats)

    def fp(tr, te, Y):
        sx = Std().fit(tr[feats].values); sd = Std().fit(tr[dfeats].values)
        th = mn_fit(np.log(np.clip(tr[list(base)].values, 1e-6, 1)), sx(tr[feats].values), sd(tr[dfeats].values), tr.k.values, weights(tr, wB), lam)
        p = len(feats); b = th[2:2 + p]; g = th[2 + p:]
        X = sx(te[feats].values); D = sd(te[dfeats].values)
        xb = X @ b if p else np.zeros(len(te)); dg = D @ g if len(dfeats) else np.zeros(len(te))
        E = np.log(np.clip(te[list(base)].values, 1e-6, 1)) + np.stack([th[0] + xb, th[1] + dg, -xb], 1)
        E = E - E.max(1, keepdims=True); P = np.exp(E)
        fp.coef = dict(zip(['cH', 'cD'] + feats + dfeats, th))
        return P / P.sum(1, keepdims=True)
    return fp


# --------------------------------------------------------------------------- LightGBM learners
def fp_lgb_mc(feats, rounds=150, lr=0.02, leaves=4, depth=2, min_leaf=200, l2=50.0, ff=0.5, seeds=(1, 2, 3), wB=0.5, base=('v_pH', 'v_pD', 'v_pA'), recal=True):
    """Multiclass GBDT on the 1X2 outcome, boosted from v1's log-probabilities (init_score)."""
    import lightgbm as lgb
    feats = list(feats)

    def fp(tr, te, Y):
        w = weights(tr, wB)
        i_tr = np.log(np.clip(tr[list(base)].values, 1e-6, 1)); i_te = np.log(np.clip(te[list(base)].values, 1e-6, 1))
        raw_te = np.zeros((len(te), 3)); raw_tr = np.zeros((len(tr), 3))
        for s in seeds:
            prm = dict(objective='multiclass', num_class=3, learning_rate=lr, num_leaves=leaves, max_depth=depth, min_data_in_leaf=min_leaf,
                       lambda_l2=l2, feature_fraction=ff, bagging_fraction=0.7, bagging_freq=1, seed=s, num_threads=2, verbose=-1)
            bst = lgb.train(prm, lgb.Dataset(tr[feats].values, label=tr.k.values, weight=w, init_score=i_tr), rounds)
            raw_te += bst.predict(te[feats].values, raw_score=True) / len(seeds)
            raw_tr += bst.predict(tr[feats].values, raw_score=True) / len(seeds)
        fp.imp = dict(zip(feats, bst.feature_importance('gain')))

        def sm(E):
            E = E - E.max(1, keepdims=True); P = np.exp(E); return P / P.sum(1, keepdims=True)
        P = sm(i_te + raw_te)
        if recal:  # in-sample recalibration of the boosted scores (2 intercepts + 2 slopes)
            o1, o2 = offsets(sm(i_tr + raw_tr)); E0 = np.zeros((len(tr), 0))
            th = co_fit(o1, o2, E0, E0, tr.k.values, w, 0, 0, 'cum', False)
            t1, t2 = offsets(P); E1 = np.zeros((len(te), 0))
            P = co_predict(th, t1, t2, E1, E1, False)
        return P
    return fp


def fp_lgb_pois(feats, rounds=150, lr=0.02, leaves=4, depth=2, min_leaf=300, l2=50.0, ff=0.5, seeds=(1, 2, 3), wB=0.5, mono=None, calib=True):
    """GBDT with Poisson objective on goals. Two rows per match (scoring side's perspective: antisymmetric features
    change sign for the away row, plus a 'side' flag), boosted from log(lambda_v1). Then v1 chain + recalibration."""
    import lightgbm as lgb
    feats = list(feats)

    def rows(d):
        X = d[feats].values
        Xh = np.c_[X, np.ones(len(d))]; Xa = np.c_[-X, np.zeros(len(d))]
        return np.r_[Xh, Xa], np.r_[np.log(d.v_lh.values), np.log(d.v_la.values)]

    def fp(tr, te, Y):
        w = weights(tr, wB); w2 = np.r_[w, w]
        Xtr, itr = rows(tr); Xte, ite = rows(te); y = np.r_[tr.hg.values, tr.ag.values]
        rte = np.zeros(len(Xte)); rtr = np.zeros(len(Xtr))
        for s in seeds:
            prm = dict(objective='poisson', learning_rate=lr, num_leaves=leaves, max_depth=depth, min_data_in_leaf=min_leaf, lambda_l2=l2,
                       feature_fraction=ff, bagging_fraction=0.7, bagging_freq=1, seed=s, num_threads=2, verbose=-1, boost_from_average=False)
            if mono is not None: prm['monotone_constraints'] = list(mono) + [0]
            bst = lgb.train(prm, lgb.Dataset(Xtr, label=y, weight=w2, init_score=itr), rounds)
            rte += bst.predict(Xte, raw_score=True) / len(seeds); rtr += bst.predict(Xtr, raw_score=True) / len(seeds)
        fp.imp = dict(zip(feats + ['side'], bst.feature_importance('gain')))
        n = len(te); P = GRID(np.exp(ite[:n] + rte[:n]), np.exp(ite[n:] + rte[n:]))
        if calib:
            m = len(tr); Pt = GRID(np.exp(itr[:m] + rtr[:m]), np.exp(itr[m:] + rtr[m:]))
            o1, o2 = offsets(Pt); E0 = np.zeros((m, 0))
            th = co_fit(o1, o2, E0, E0, tr.k.values, w, 0, 0, 'cum', False)
            t1, t2 = offsets(P); E1 = np.zeros((n, 0))
            P = co_predict(th, t1, t2, E1, E1, False)
        return P
    return fp


# --------------------------------------------------------------------------- pooling of probability forecasts
def pool_fit(Ps, k, w, kind='lin', lam=0.0):
    """Weights (softmax-parametrised, sum to 1) of a linear or log-linear opinion pool, fitted by RPS."""
    m = len(Ps); Y = np.eye(3)[k]; w = w / w.sum()
    L = [np.log(np.clip(P, 1e-9, 1)) for P in Ps]

    def mix(a):
        ww = np.exp(a - a.max()); ww /= ww.sum()
        if kind == 'lin': P = sum(wi * Pi for wi, Pi in zip(ww, Ps))
        else:
            E = sum(wi * Li for wi, Li in zip(ww, L)); E = E - E.max(1, keepdims=True); P = np.exp(E); P = P / P.sum(1, keepdims=True)
        return P

    def f(a):
        P = mix(a)
        r = ((P[:, 0] - Y[:, 0]) ** 2 + (P[:, 0] + P[:, 1] - Y[:, 0] - Y[:, 1]) ** 2) / 2
        return (w * r).sum() + lam * ((a - a.mean()) ** 2).sum()
    a = minimize(f, np.zeros(m), method='Nelder-Mead' if m <= 4 else 'Powell', options=dict(xatol=1e-3, fatol=1e-8, maxiter=4000) if m <= 4 else dict(xtol=1e-3, ftol=1e-8)).x
    ww = np.exp(a - a.max()); return ww / ww.sum(), mix
