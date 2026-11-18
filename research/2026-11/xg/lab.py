#!/usr/bin/env python
"""Experiment library for the xG track (steps 4-5): observations, online calibration, walk-forward Poisson stack,
finishing / goalkeeping ratings, evaluation.  Imported by run_experiments.py and make_final.py."""
import json, os, math
import numpy as np, pandas as pd
import engine as E

HERE = os.path.dirname(os.path.abspath(__file__))
PG = E.PG


class Lab:
    def __init__(self):
        self.D = D = E.Data()
        self.C = E.Chain()
        n = len(D.M)
        F = pd.read_csv(os.path.join(HERE, 'features_xg_match.csv'), dtype={'id': str}).set_index('id')
        self.F = F
        has = np.array([i in F.index for i in D.id]) & D.post
        self.has = has
        idx = D.id[has]
        self.X = {}
        cols = ['xg', 'npxg', 'pens', 'pen_goals', 'goals', 'np_goals', 'og_for', 'shots', 'sot', 'psxg', 'xg_ns', 'xg_gb', 'xg_sit', 'xg_level', 'xg_lead', 'xg_trail', 'xg_lead2', 'xg_trail2',
                'xg_11v11', 'xg_manup', 'xg_mandn', 'bigch', 'xg_big', 'xg_small', 'n_six', 'n_boxc', 'n_boxs', 'n_out', 'xg_setp', 'xg_fast', 'xg_head', 'xg_late', 'xg_1h', 'xg_adj', 'xg_reb', 'xg_cross', 'xg_through']
        for c in cols:
            a = np.full((n, 2), np.nan)
            a[has, 0] = F.loc[idx, 'h_' + c].to_numpy(); a[has, 1] = F.loc[idx, 'a_' + c].to_numpy()
            self.X[c] = a
        self.des = np.full((n, 3), np.nan); self.des[has] = F.loc[idx, ['des_pH', 'des_pD', 'des_pA']].to_numpy()
        # v1 shot-channel relative strengths (re-run of the v1 filters; identical to baseline_features)
        self.rel = {}
        for c in ['sh', 'sot']:
            p = dict(E.MODEL['params'][c]); p.update(Pg0=0.0, qg=0.0, rhoG=1.0)
            y = np.full((n, 2), np.nan)
            for i, m in enumerate(D.M):
                if m['status'] == 'post' and m['statsOk']: y[i] = [m['stats']['h'][c], m['stats']['a'][c]]
            out = E.run_filter(D, y, p)
            self.rel[c] = np.column_stack([np.log(out[:, 0]) - (out[:, 4] + out[:, 5] / 2), np.log(out[:, 1]) - (out[:, 4] - out[:, 5] / 2)])
        self.g_v1 = E.run_filter(D, D.v1_obs())
        B = D.B
        self.lam_v1 = np.column_stack([B.lh.to_numpy(), B.la.to_numpy()])
        self.goals = np.column_stack([D.hg, D.ag])
        self.kcache = {}

    # ---- online calibration of xG to goal level: trailing ratio goals / xG per league (strictly earlier matches) ----
    def kappa(self, x, tau=400.0, n0=100.0, k0=1.03):
        """x[n,2]: xG-type quantity. Returns k[n] = (prior + decayed goals) / (prior + decayed x) before each match."""
        D = self.D
        dec = math.exp(-1.0 / tau)
        num = [n0 * 2.5 * k0] * 2; den = [n0 * 2.5] * 2
        k = np.ones(len(D.M))
        tot = np.nansum(x, 1); g = np.nansum(self.goals, 1)
        for i in range(len(D.M)):
            L = D.lg[i]
            k[i] = num[L] / den[L]
            if self.has[i]:
                num[L] = num[L] * dec + g[i]; den[L] = den[L] * dec + tot[i]
        return k

    def xg_obs(self, kind='tot', wUp=None, delta=0.0, base='xg'):
        """xG-type observation on goal scale. kind: 'tot' (non-penalty xG + penalties) or 'np' (penalties ignored).
        wUp: weight of xG created with a man advantage (v1 uses wRed for goals); delta: score-effect correction
        (xG while leading * (1+delta), while trailing * (1-delta))."""
        X = self.X
        wUp = PG['wRed'] if wUp is None else wUp
        npx = X['npxg'] if base == 'xg' else X[base]
        adj = npx - (1 - wUp) * X['xg_manup'] + delta * (X['xg_lead'] - X['xg_trail'])
        raw = adj + (X['xg'] - X['npxg']) if kind == 'tot' else adj
        key = (kind, wUp, delta, base)
        if key not in self.kcache: self.kcache[key] = self.kappa(raw)
        return raw * self.kcache[key][:, None]

    def obs(self, wg, xo, wp=0.0, wRed=None):
        """Blend: wg * red-adjusted goals + wp * v1 shot proxy + (1-wg-wp) * xG observation, where shots exist; v1 observation elsewhere."""
        D = self.D
        y = D.v1_obs(wRed=wRed)
        g = D.gadj(PG['wRed'] if wRed is None else wRed)
        prox = np.where(np.isnan(D.prox), xo, D.prox)
        yy = wg * g + wp * prox + (1 - wg - wp) * xo
        return np.where(self.has[:, None], yy, y)

    def filt(self, y, phi=None, **kw):
        """Goal-channel EKF with v1 hyper-parameters; `phi` (observation noise) is applied only to matches whose observation
        contains xG, so the rating stream is exactly v1 until the first match with shot data."""
        p = dict(PG); p.update(kw)
        return E.run_filter(self.D, y, p, phi_arr=None if phi is None else np.where(self.has, phi, p['phi']))

    def lam_swap(self, gnew):
        """Final expected goals when the goal channel is replaced and the v1 stack kept: lam' = lam * (g'/g)^a1."""
        a1 = E.STACK['a1']
        return self.lam_v1 * (gnew[:, :2] / self.g_v1[:, :2]) ** a1

    # ---- evaluation ----
    def probs(self, lam):
        return self.C.probs(lam[:, 0], lam[:, 1])

    def score(self, lam, holdout=False, p=None):
        D = self.D; s = D.scored
        p = self.probs(lam[s]) if p is None else p
        r = E.rps(p, D.hg[s], D.ag[s]); se = D.season[s]
        pb = self.probs(self.lam_v1[s]); rb = E.rps(pb, D.hg[s], D.ag[s])
        out = {'DEV': r[se <= 2021].mean() - rb[se <= 2021].mean(), 'DEV15': r[(se <= 2021) & (se >= 2015)].mean() - rb[(se <= 2021) & (se >= 2015)].mean(),
               'DEVrps': r[se <= 2021].mean()}
        if holdout:
            out.update({'HOLD': r[se >= 2022].mean() - rb[se >= 2022].mean(), 'HOLDrps': r[se >= 2022].mean()})
        return out


def pois_fit(X, y, l2=1e-6, w=None, iters=40, b0=None):
    """Poisson regression (log link) by Newton steps with ridge penalty l2*|b-b0|^2 (l2 scalar or vector)."""
    k = X.shape[1]
    b0 = np.zeros(k) if b0 is None else np.asarray(b0, float)
    l2 = np.full(k, l2) if np.isscalar(l2) else np.asarray(l2, float)
    b = b0.copy()
    if not b.any() and len(y): b[0] = math.log(max(1e-6, y.mean()))
    if len(y) == 0: return b0.copy()
    w = np.ones(len(y)) if w is None else w
    for _ in range(iters):
        mu = np.exp(np.clip(X @ b, -8, 4))
        g = X.T @ (w * (y - mu)) - l2 * (b - b0)
        Hm = (X * (w * mu)[:, None]).T @ X + np.diag(l2)
        step = np.linalg.solve(Hm, g)
        b += step
        if np.abs(step).max() < 1e-9: break
    return b


def wf_stack(L, feats, first=2009, seasons=range(2013, 2027), leagues=(0, 1), l2=1e-6, half_life=None, return_coefs=False, b0=None, train_mask=None):
    """Walk-forward Poisson regression of goals on features. feats: list of arrays [n,2] (home col, away col).
    For season Y coefficients are fitted on finished matches of seasons first..Y-1 (optionally restricted by train_mask),
    with a ridge prior centred on b0. Returns lam[n,2] (nan outside `seasons`)."""
    D = L.D; n = len(D.M)
    Xh = np.column_stack([np.ones(n)] + [f[:, 0] for f in feats]); Xa = np.column_stack([np.ones(n)] + [f[:, 1] for f in feats])
    ok = D.post & np.isin(D.lg, leagues) & np.isfinite(Xh).all(1) & np.isfinite(Xa).all(1)
    if train_mask is not None: ok = ok & train_mask
    lam = np.full((n, 2), np.nan); coefs = {}
    for Y in seasons:
        tr = ok & (D.season >= first) & (D.season < Y)
        X = np.vstack([Xh[tr], Xa[tr]]); y = np.concatenate([D.hg[tr], D.ag[tr]])
        w = None
        if half_life: w = np.tile(np.power(0.5, (Y - 1 - D.season[tr]) / half_life), 2)
        b = pois_fit(X, y, l2, w, b0=b0); coefs[Y] = b
        te = D.season == Y
        lam[te, 0] = np.exp(Xh[te] @ b); lam[te, 1] = np.exp(Xa[te] @ b)
    return (lam, coefs) if return_coefs else lam


def rel_of(out):
    """log expected rate relative to league level/home advantage for a filter output."""
    return np.column_stack([np.log(out[:, 0]) - (out[:, 4] + out[:, 5] / 2), np.log(out[:, 1]) - (out[:, 4] - out[:, 5] / 2)])


def skill_ratings(L, num, den, tau=60.0, n0=30.0, carry=0.6):
    """Online shrunk ratio ratings per team. For 'for' side: sum(num)/sum(den) with exponential decay (tau team-matches),
    prior n0 units of `den` at ratio 1, regression to 1 by `carry` at each season change.
    num/den are [n,2] arrays in team perspective (home col = home team). Returns (for[n,2], against[n,2]) pre-match log-ratios:
      for[i,0]  = log ratio of the home team's own num/den;  against[i,0] = log ratio of what the home team concedes (opponents' num/den against it)."""
    D = L.D; T = len(D.tids); dec = math.exp(-1.0 / tau)
    fn = [0.0] * T; fd = [0.0] * T; an = [0.0] * T; ad = [0.0] * T; ts = [0] * T
    F = np.zeros((len(D.M), 2)); A = np.zeros((len(D.M), 2))
    for i in range(len(D.M)):
        h, a, se = D.h[i], D.a[i], D.season[i]
        for t in (h, a):
            if ts[t] and se > ts[t]:
                c = carry ** (se - ts[t]); fn[t] *= c; fd[t] *= c; an[t] *= c; ad[t] *= c
            ts[t] = se
        F[i, 0] = math.log((fn[h] + n0) / (fd[h] + n0)); F[i, 1] = math.log((fn[a] + n0) / (fd[a] + n0))
        A[i, 0] = math.log((an[h] + n0) / (ad[h] + n0)); A[i, 1] = math.log((an[a] + n0) / (ad[a] + n0))
        if L.has[i] and np.isfinite(num[i]).all() and np.isfinite(den[i]).all():
            fn[h] = fn[h] * dec + num[i, 0]; fd[h] = fd[h] * dec + den[i, 0]
            fn[a] = fn[a] * dec + num[i, 1]; fd[a] = fd[a] * dec + den[i, 1]
            an[h] = an[h] * dec + num[i, 1]; ad[h] = ad[h] * dec + den[i, 1]
            an[a] = an[a] * dec + num[i, 0]; ad[a] = ad[a] * dec + den[i, 0]
    return F, A


def boot_ci(diff, seed=7, B=2000):
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
