"""Shared building blocks for the 'core' research track.

- load():            arrays from data.npz
- ekf_v1():          exact numba port of site/js/engine/ratings.js (one channel)
- hazard fit + exact Markov-chain score distribution (port of match-model.js / refit.mjs step 5)
- metrics, optimiser helpers, CSV writers
"""
import math
import numpy as np
from numba import njit
from scipy.optimize import minimize

D = '/home/user/bdv-lab/core/'
S = '/home/user/bdv-lab/shared/'


class Data:
    pass


def load():
    z = np.load(D + 'data.npz', allow_pickle=False)
    d = Data()
    for k in z.files:
        setattr(d, k, z[k])
    d.n = len(d.h)
    d.nteams = int(max(d.h.max(), d.a.max()) + 1)
    d.k = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    d.scored = (d.lg == 0) & (d.season >= 2013) & d.post
    d.isA = d.lg == 0
    return d


# ───────────────────────── v1 EKF (exact port) ─────────────────────────────────
# p = [q, P0, rho, sSeason, dPromoO, dPromoD, dNew, phi, kMu, kH, Pg0, qg, rhoG, mu0, hfa0]
V1_KEYS = ['q', 'P0', 'rho', 'sSeason', 'dPromoO', 'dPromoD', 'dNew', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG', 'mu0', 'hfa0']


@njit(cache=True, error_model='numpy')
def ekf_v1(h, a, lg, season, post, yh, ya, hasobs, p, isG, nteams):
    q, P0, rho, sSeason, dPO, dPD, dNew, phi, kMu, kH, Pg0, qg, rhoG, mu0, hfa0 = p
    n = len(h)
    o = np.zeros(nteams); d = np.zeros(nteams); g = np.zeros(nteams)
    Po = np.zeros(nteams); Pd = np.zeros(nteams); Pg = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams, np.int64)
    mu = np.array([mu0, mu0]); hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    # outputs: lh, la, Sh, Sa, mu, hfa, relH (o_h - d_a), relA (o_a - d_h), g_h, nh, na
    out = np.zeros((n, 11))
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                fresh = (L == 0 and se <= 2007)
                off = 0.0 if fresh else -dNew
                o[t] = off; d[t] = off; Po[t] = P0; Pd[t] = P0; g[t] = 0.0; Pg[t] = Pg0
                if L == 0 and se > 2007:
                    o[t] -= dPO - dNew; d[t] -= dPD - dNew
                tse[t] = se; tlg[t] = L; tn[t] = 0
            elif se > tse[t]:
                gap = se - tse[t]
                r = rho ** gap
                o[t] *= r; d[t] *= r
                Po[t] = min(P0, r * r * Po[t] + sSeason * gap)
                Pd[t] = min(P0, r * r * Pd[t] + sSeason * gap)
                if isG:
                    rg = rhoG ** gap
                    g[t] *= rg; Pg[t] = min(Pg0, rg * rg * Pg[t] + qg * 20 * gap)
                if gap > 1:
                    o[t] = min(o[t], -dNew); d[t] = min(d[t], -dNew); Po[t] = P0; Pd[t] = P0
                elif L != tlg[t]:
                    sign = -1.0 if L == 0 else 1.0
                    o[t] += sign * dPO; d[t] += sign * dPD
                    Po[t] = min(P0, Po[t] + sSeason); Pd[t] = min(P0, Pd[t] + sSeason)
                tse[t] = se; tlg[t] = L; tn[t] = 0
        H = h[i]; A = a[i]
        gg = g[H] if isG else 0.0
        etaH = mu[L] + hfa[L] / 2 + gg + o[H] - d[A]
        etaA = mu[L] - hfa[L] / 2 - gg + o[A] - d[H]
        out[i, 0] = math.exp(etaH); out[i, 1] = math.exp(etaA)
        out[i, 2] = Po[H] + Pd[A] + (Pg[H] if isG else 0.0)
        out[i, 3] = Po[A] + Pd[H] + (Pg[H] if isG else 0.0)
        out[i, 4] = mu[L]; out[i, 5] = hfa[L]
        out[i, 6] = o[H] - d[A]; out[i, 7] = o[A] - d[H]; out[i, 8] = gg
        out[i, 9] = tn[H]; out[i, 10] = tn[A]
        if not post[i]:
            continue
        if hasobs[i]:
            Po[H] += q; Pd[H] += q; Po[A] += q; Pd[A] += q
            if isG:
                Pg[H] += qg
            lh = math.exp(mu[L] + hfa[L] / 2 + gg + o[H] - d[A])
            la = math.exp(mu[L] - hfa[L] / 2 - gg + o[A] - d[H])
            eh = yh[i] - lh; ea = ya[i] - la
            Sx = Po[H] + Pd[A] + (Pg[H] if isG else 0.0)
            den = lh * Sx + phi
            o[H] += Po[H] * eh / den; d[A] -= Pd[A] * eh / den
            if isG:
                g[H] += Pg[H] * eh / den; Pg[H] -= Pg[H] * Pg[H] * lh / den
            Po[H] -= Po[H] * Po[H] * lh / den; Pd[A] -= Pd[A] * Pd[A] * lh / den
            Sx = Po[A] + Pd[H] + (Pg[H] if isG else 0.0)
            den = la * Sx + phi
            o[A] += Po[A] * ea / den; d[H] -= Pd[H] * ea / den
            if isG:
                g[H] -= Pg[H] * ea / den; Pg[H] -= Pg[H] * Pg[H] * la / den
            Po[A] -= Po[A] * Po[A] * la / den; Pd[H] -= Pd[H] * Pd[H] * la / den
            kM = max(kMu, 1.0 / (Ln[L] + 30)); kHH = max(kH, 1.0 / (Ln[L] + 30))
            mu[L] += kM * (eh / lh + ea / la) / 2
            hfa[L] += kHH * (eh / lh - ea / la) / 2
            Ln[L] += 1
        tn[H] += 1; tn[A] += 1
    return out


# ───────────────────────── Markov chain (exact port) ───────────────────────────
MAXG = 10
G1 = MAXG + 1


def timeline_bins(S1=3, S2=7):
    b = []
    for m in range(1, 46): b.append(0 if m <= 15 else 1 if m <= 30 else 2)
    b += [3] * S1
    for m in range(46, 91): b.append(4 if m <= 60 else 5 if m <= 75 else 6)
    b += [7] * S2
    return np.array(b, np.int64)


@njit(cache=True)
def _diff_idx(d):
    return 0 if d <= -2 else 4 if d >= 2 else d + 2


@njit(cache=True)
def _men_idx(own, opp):
    return 2 if opp > own else 0 if own > opp else 1


@njit(cache=True)
def step_of(S1, S2, mn, add):
    if mn <= 45:
        if mn < 45 or add == 0:
            return max(0, min(44, mn - 1))
        return 45 + max(0, min(S1 - 1, add - 1))
    if mn < 90 or (mn == 90 and add == 0):
        return S1 + max(45, min(89, mn - 1))
    ad = add if add > 0 else mn - 90
    return 90 + S1 + max(0, min(S2 - 1, ad - 1))


def compile_hazard(hz):
    """hz: dict(goal_bin[8], goal_diff[5], goal_men[3], lateTrail, norm, red_bin[8], red_diff[5], red_away)."""
    goalF = np.zeros((8, 5, 3)); redH = np.zeros((8, 5)); redA = np.zeros((8, 5))
    for b in range(8):
        for d in range(5):
            for m in range(3):
                e = hz['goal_bin'][b] + hz['goal_diff'][d] + hz['goal_men'][m]
                if b >= 6 and d < 2: e += hz.get('lateTrail', 0.0)
                goalF[b, d, m] = math.exp(e) / 90 * hz.get('norm', 1.0)
            redH[b, d] = math.exp(hz['red_bin'][b] + hz['red_diff'][d])
            redA[b, d] = math.exp(hz['red_bin'][b] + hz['red_diff'][d] + hz['red_away'])
    return goalF, redH, redA


@njit(cache=True)
def chain_matrix(bins, goalF, redH, redA, lh, la):
    N = G1 * G1 * 9
    cur = np.zeros(N); nxt = np.zeros(N)
    cur[0] = 1.0
    maxH = 0; maxA = 0
    for t in range(len(bins)):
        b = bins[t]
        nxt[:] = 0.0
        for hg in range(maxH + 1):
            for ag in range(maxA + 1):
                dH = _diff_idx(hg - ag); dA = _diff_idx(ag - hg)
                for rh in range(3):
                    for ra in range(3):
                        i = ((hg * G1 + ag) * 3 + rh) * 3 + ra
                        p = cur[i]
                        if p < 1e-15:
                            continue
                        pgh = lh * goalF[b, dH, _men_idx(rh, ra)] if hg < MAXG else 0.0
                        pga = la * goalF[b, dA, _men_idx(ra, rh)] if ag < MAXG else 0.0
                        prh = redH[b, dH] if rh < 2 else 0.0
                        pra = redA[b, dA] if ra < 2 else 0.0
                        nxt[i] += p * (1 - pgh - pga - prh - pra)
                        if pgh > 0: nxt[(((hg + 1) * G1 + ag) * 3 + rh) * 3 + ra] += p * pgh
                        if pga > 0: nxt[((hg * G1 + ag + 1) * 3 + rh) * 3 + ra] += p * pga
                        if prh > 0: nxt[((hg * G1 + ag) * 3 + rh + 1) * 3 + ra] += p * prh
                        if pra > 0: nxt[((hg * G1 + ag) * 3 + rh) * 3 + ra + 1] += p * pra
        if maxH < MAXG: maxH += 1
        if maxA < MAXG: maxA += 1
        tmp = cur; cur = nxt; nxt = tmp
    mat = np.zeros((G1, G1))
    for hg in range(G1):
        for ag in range(G1):
            s = 0.0
            for r in range(9):
                s += cur[(hg * G1 + ag) * 9 + r]
            mat[hg, ag] = s
    return mat


@njit(cache=True)
def chain_probs(bins, goalF, redH, redA, lh, la):
    """returns pH,pD,pA,xgH,xgA for arrays lh, la"""
    n = len(lh)
    out = np.zeros((n, 5))
    for k in range(n):
        m = chain_matrix(bins, goalF, redH, redA, lh[k], la[k])
        for i in range(G1):
            for j in range(G1):
                p = m[i, j]
                if i > j: out[k, 0] += p
                elif i == j: out[k, 1] += p
                else: out[k, 2] += p
                out[k, 3] += i * p; out[k, 4] += j * p
    return out


class ChainTable:
    """Bilinear lookup (in log-lambda) of the Markov-chain 1X2 probabilities for one hazard set."""

    def __init__(self, hz, lo=0.08, hi=6.0, npts=90):
        self.gl = np.linspace(math.log(lo), math.log(hi), npts)
        self.hz = hz
        goalF, redH, redA = compile_hazard(hz)
        bins = timeline_bins(hz.get('S1', 3), hz.get('S2', 7))
        A, B = np.meshgrid(self.gl, self.gl, indexing='ij')
        r = chain_probs(bins, goalF, redH, redA, np.exp(A.ravel()), np.exp(B.ravel()))
        self.tab = r.reshape(npts, npts, 5)

    def __call__(self, lh, la):
        return table_lookup(self.gl, self.tab, np.asarray(lh, float), np.asarray(la, float))


@njit(cache=True)
def table_lookup(gl, tab, lh, la):
    n = len(lh); out = np.zeros((n, 3))
    step = gl[1] - gl[0]; m = len(gl)
    for k in range(n):
        x = (math.log(lh[k]) - gl[0]) / step; y = (math.log(la[k]) - gl[0]) / step
        x = min(max(x, 0.0), m - 1.000001); y = min(max(y, 0.0), m - 1.000001)
        i = int(x); j = int(y); fx = x - i; fy = y - j
        for c in range(3):
            out[k, c] = (tab[i, j, c] * (1 - fx) * (1 - fy) + tab[i + 1, j, c] * fx * (1 - fy)
                         + tab[i, j + 1, c] * (1 - fx) * fy + tab[i + 1, j + 1, c] * fx * fy)
        s = out[k, 0] + out[k, 1] + out[k, 2]
        out[k, 0] /= s; out[k, 1] /= s; out[k, 2] /= s
    return out


# ───────────────────────── independent-Poisson family (fast) ───────────────────
@njit(cache=True)
def pois_probs(lh, la, rho=0.0, maxg=12):
    """1X2 from independent Poisson with optional Dixon-Coles tau(rho)."""
    n = len(lh); out = np.zeros((n, 3))
    ph = np.zeros(maxg + 1); pa = np.zeros(maxg + 1)
    for k in range(n):
        ph[0] = math.exp(-lh[k]); pa[0] = math.exp(-la[k])
        for i in range(1, maxg + 1):
            ph[i] = ph[i - 1] * lh[k] / i; pa[i] = pa[i - 1] * la[k] / i
        H = 0.0; Dr = 0.0; A = 0.0
        for i in range(maxg + 1):
            for j in range(maxg + 1):
                p = ph[i] * pa[j]
                if rho != 0.0:
                    if i == 0 and j == 0: p *= 1 - lh[k] * la[k] * rho
                    elif i == 0 and j == 1: p *= 1 + lh[k] * rho
                    elif i == 1 and j == 0: p *= 1 + la[k] * rho
                    elif i == 1 and j == 1: p *= 1 - rho
                if i > j: H += p
                elif i == j: Dr += p
                else: A += p
        s = H + Dr + A
        out[k, 0] = H / s; out[k, 1] = Dr / s; out[k, 2] = A / s
    return out


# ───────────────────────── metrics ─────────────────────────────────────────────
def rps_ll(p, k):
    p = np.clip(p, 1e-9, 1); p = p / p.sum(1, keepdims=True)
    y = np.eye(3)[k]
    rps = ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2
    ll = -np.log(p[np.arange(len(k)), k])
    return rps, ll


_LF = np.cumsum(np.log(np.maximum(1, np.arange(60))))


def pois_ll(k, lam):
    k = np.minimum(np.round(k).astype(int), 59)
    return k * np.log(lam) - lam - _LF[k]


def nm(f, x0, steps=None, maxiter=400, rounds=1, xatol=1e-4, fatol=1e-8):
    """Nelder-Mead with an explicit initial simplex and optional restarts."""
    x = np.array(x0, float); best = None
    for r in range(rounds):
        st = np.full(len(x), 0.3 if r == 0 else 0.15) if steps is None else np.asarray(steps) * (1.0 if r == 0 else 0.5)
        sim = np.vstack([x] + [x + np.eye(len(x))[i] * st[i] for i in range(len(x))])
        res = minimize(f, x, method='Nelder-Mead', options=dict(initial_simplex=sim, maxiter=maxiter, maxfev=maxiter * 3, xatol=xatol, fatol=fatol))
        x = res.x; best = res.fun
    return x, best


def write_preds(name, ids, p):
    import pandas as pd
    df = pd.DataFrame({'id': ids, 'pH': np.round(p[:, 0], 6), 'pD': np.round(p[:, 1], 6), 'pA': np.round(p[:, 2], 6)})
    df.to_csv(D + f'preds_{name}.csv', index=False)
    return D + f'preds_{name}.csv'


_BASE = None


def dev_report(name, d, p_scored, base=None, verbose=True):
    """DEV-only paired comparison against the shared v1 baseline (HOLDOUT is deliberately not printed)."""
    import pandas as pd
    global _BASE
    if _BASE is None:
        b = pd.read_csv(S + 'preds_baseline.csv', dtype={'id': str}).set_index('id')
        _BASE = b.loc[d.ids[d.scored]].to_numpy()
    k = d.k[d.scored]; se = d.season[d.scored]
    r, l = rps_ll(p_scored, k); rb, lb = rps_ll(_BASE if base is None else base, k)
    m = se <= 2021
    diff = (r - rb)[m]
    rng = np.random.default_rng(7)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
    if verbose:
        print(f'{name:28s} DEV RPS {r[m].mean():.5f} LL {l[m].mean():.5f} | dRPS {diff.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] dLL {(l - lb)[m].mean():+.5f}')
    return r[m].mean(), l[m].mean()
