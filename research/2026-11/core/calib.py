"""Question 3 (+6): score-distribution / draw / calibration layers on top of fixed walk-forward expected goals.

Every layer for season Y is fitted on Serie A matches of seasons [2009, Y) whose lambdas come from the pass
run with hyper-parameters tuned on seasons < Y (so nothing from season Y or later is used).

usage: calib.py out_<name>.pkl [engine:ekf2|ekf3]
"""
import sys, math, pickle
import numpy as np
from numba import njit
from scipy.optimize import minimize
import lib, v1h

MAXK = 12


@njit(cache=True)
def score_matrix(lh, la, rho, nu, out):
    """out[(MAXK+1)^2]: joint score probabilities. nu = CMP dispersion (1 = Poisson) with mean matched to lambda; rho = DC."""
    ph = np.zeros(MAXK + 1); pa = np.zeros(MAXK + 1)
    for side in range(2):
        lam = lh if side == 0 else la
        p = ph if side == 0 else pa
        if nu == 1.0:
            p[0] = math.exp(-lam)
            for i in range(1, MAXK + 1): p[i] = p[i - 1] * lam / i
        else:
            # Conway-Maxwell-Poisson: p(k) ~ l^k / (k!)^nu ; solve l so that the mean equals lam (bisection on log l)
            lo = -8.0; hi = 6.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                s = 0.0; m = 0.0; lw = 0.0
                for k in range(MAXK + 1):
                    if k > 0: lw += mid - nu * math.log(k)
                    w = math.exp(lw); s += w; m += k * w
                if m / s < lam: lo = mid
                else: hi = mid
            mid = 0.5 * (lo + hi); s = 0.0; lw = 0.0
            for k in range(MAXK + 1):
                if k > 0: lw += mid - nu * math.log(k)
                p[k] = math.exp(lw); s += p[k]
            for k in range(MAXK + 1): p[k] /= s
    tot = 0.0
    for i in range(MAXK + 1):
        for j in range(MAXK + 1):
            q = ph[i] * pa[j]
            if rho != 0.0:
                if i == 0 and j == 0: q *= 1 - lh * la * rho
                elif i == 0 and j == 1: q *= 1 + lh * rho
                elif i == 1 and j == 0: q *= 1 + la * rho
                elif i == 1 and j == 1: q *= 1 - rho
            q = max(q, 1e-12)
            out[i * (MAXK + 1) + j] = q; tot += q
    for i in range((MAXK + 1) ** 2): out[i] /= tot


@njit(cache=True)
def probs_gen(lh, la, rho, nu):
    n = len(lh); P = np.zeros((n, 3)); m = np.zeros((MAXK + 1) ** 2)
    for k in range(n):
        score_matrix(lh[k], la[k], rho, nu, m)
        for i in range(MAXK + 1):
            for j in range(MAXK + 1):
                q = m[i * (MAXK + 1) + j]
                if i > j: P[k, 0] += q
                elif i == j: P[k, 1] += q
                else: P[k, 2] += q
    return P


@njit(cache=True)
def score_ll(lh, la, hg, ag, rho, nu):
    n = len(lh); m = np.zeros((MAXK + 1) ** 2); s = 0.0
    for k in range(n):
        score_matrix(lh[k], la[k], rho, nu, m)
        s += math.log(m[min(hg[k], MAXK) * (MAXK + 1) + min(ag[k], MAXK)])
    return s / n


@njit(cache=True)
def probs_mix(lh, la, Sh, Sa, Sha, kappa, meanfix):
    """1X2 integrating over log-normal uncertainty of the two log-rates (3x3-point Gauss-Hermite on a correlated normal)."""
    xs = np.array([-1.7320508075688772, 0.0, 1.7320508075688772]); ws = np.array([1.0 / 6, 2.0 / 3, 1.0 / 6])
    n = len(lh); P = np.zeros((n, 3)); one = np.zeros(1); two = np.zeros(1)
    for k in range(n):
        vh = max(kappa * Sh[k], 0.0); va = max(kappa * Sa[k], 0.0); c = kappa * Sha[k]
        sh_ = math.sqrt(vh); sa_ = math.sqrt(va)
        r = c / (sh_ * sa_) if sh_ > 0 and sa_ > 0 else 0.0
        r = max(-0.95, min(0.95, r))
        for i in range(3):
            for j in range(3):
                z1 = xs[i]; z2 = r * xs[i] + math.sqrt(1 - r * r) * xs[j]
                a = lh[k] * math.exp(sh_ * z1 - (0.5 * vh if meanfix else 0.0))
                b = la[k] * math.exp(sa_ * z2 - (0.5 * va if meanfix else 0.0))
                one[0] = a; two[0] = b
                q = lib.pois_probs(one, two)
                w = ws[i] * ws[j]
                P[k, 0] += w * q[0, 0]; P[k, 1] += w * q[0, 1]; P[k, 2] += w * q[0, 2]
    return P


def nll(P, k):
    return -np.log(np.clip(P[np.arange(len(k)), k], 1e-12, 1)).mean()


def fit1(f, x0, bounds=None):
    r = minimize(f, x0, method='Nelder-Mead', options=dict(xatol=1e-5, fatol=1e-9, maxiter=2000))
    return r.x


def draw_adj(P, delta):
    """multiply draw odds by exp(delta), rescale H/A proportionally"""
    pd_ = P[:, 1]; od = pd_ / (1 - pd_) * np.exp(delta); nd = od / (1 + od)
    Q = P.copy(); s = (1 - nd) / (1 - pd_)
    Q[:, 0] *= s; Q[:, 2] *= s; Q[:, 1] = nd
    return Q


def layers(tr, te, hz_tab=None):
    """tr / te: dicts with lh, la, hg, ag, k, Sh, Sa, Sha (te has no outcomes used). Returns dict name -> test probs."""
    out = {}
    lh, la, k = tr['lh'], tr['la'], tr['k']
    P0tr = lib.pois_probs(lh, la); P0te = lib.pois_probs(te['lh'], te['la'])
    out['indep'] = P0te
    # DC rho (score likelihood)
    r = fit1(lambda x: -score_ll(lh, la, tr['hg'], tr['ag'], float(np.clip(x[0], -0.3, 0.3)), 1.0), [0.0])
    out['dc_rho'] = probs_gen(te['lh'], te['la'], float(np.clip(r[0], -0.3, 0.3)), 1.0); out['_rho'] = float(r[0])
    # diagonal inflation (constant factor on all draws)
    x = fit1(lambda x: nll(draw_adj(P0tr, x[0]), k), [0.0])
    out['diag_infl'] = draw_adj(P0te, x[0]); out['_diag'] = float(x[0])
    # draw correction depending on expected total and closeness
    ft = lambda L1, L2: (np.log(L1 + L2) - math.log(2.5), np.abs(np.log(L1 / L2)))
    a_, b_ = ft(lh, la); ta, tb = ft(te['lh'], te['la'])
    x = fit1(lambda x: nll(draw_adj(P0tr, x[0] + x[1] * a_ + x[2] * b_), k) + 1e-3 * (x[1] ** 2 + x[2] ** 2), [0.0, 0.0, 0.0])
    out['draw_fn'] = draw_adj(P0te, x[0] + x[1] * ta + x[2] * tb); out['_drawfn'] = [float(v) for v in x]
    # CMP dispersion (score likelihood)
    x = fit1(lambda x: -score_ll(lh, la, tr['hg'], tr['ag'], 0.0, float(np.clip(math.exp(x[0]), 0.5, 2.0))), [0.0])
    nu = float(np.clip(math.exp(x[0]), 0.5, 2.0))
    out['cmp'] = probs_gen(te['lh'], te['la'], 0.0, nu); out['_nu'] = nu
    # temperature
    def temp(P, T):
        Q = np.clip(P, 1e-9, 1) ** (1 / T); return Q / Q.sum(1, keepdims=True)
    x = fit1(lambda x: nll(temp(P0tr, math.exp(x[0])), k), [0.0])
    out['temperature'] = temp(P0te, math.exp(x[0])); out['_T'] = math.exp(x[0])
    # per-class recalibration: log p' = b log p + a_k
    def pc(P, x):
        Lg = math.exp(x[0]) * np.log(np.clip(P, 1e-9, 1)); Lg[:, 1] += x[1]; Lg[:, 2] += x[2]
        Q = np.exp(Lg - Lg.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)
    x = fit1(lambda x: nll(pc(P0tr, x), k), [0.0, 0.0, 0.0])
    out['per_class'] = pc(P0te, x); out['_pc'] = [float(v) for v in x]
    # Dirichlet calibration (full matrix, ridge towards identity)
    def dc(P, x):
        W = np.eye(3) + x[:9].reshape(3, 3); Lg = np.log(np.clip(P, 1e-9, 1)) @ W.T; Lg[:, 1] += x[9]; Lg[:, 2] += x[10]
        Q = np.exp(Lg - Lg.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)
    r = minimize(lambda x: nll(dc(P0tr, x), k) + 5e-3 * (x[:9] ** 2).sum(), np.zeros(11), method='BFGS')
    out['dirichlet'] = dc(P0te, r.x)
    # lambda recalibration: log lam' = a + b log lam   (Poisson likelihood, both sides)
    def lamcal(x):
        Lh = np.exp(x[0] + x[1] * np.log(lh)); La = np.exp(x[0] + x[1] * np.log(la))
        return -(lib.pois_ll(tr['hg'], Lh) + lib.pois_ll(tr['ag'], La)).mean()
    x = fit1(lamcal, [0.0, 1.0])
    out['lam_recal'] = lib.pois_probs(np.exp(x[0] + x[1] * np.log(te['lh'])), np.exp(x[0] + x[1] * np.log(te['la']))); out['_lamcal'] = [float(v) for v in x]
    # rating-uncertainty mixture (kappa = 1: the filter's own variances; kappa fitted on 1X2 log-loss)
    if 'Sh' in tr:
        out['mix_k1'] = probs_mix(te['lh'], te['la'], te['Sh'], te['Sa'], te['Sha'], 1.0, True)
        out['mix_k1_nofix'] = probs_mix(te['lh'], te['la'], te['Sh'], te['Sa'], te['Sha'], 1.0, False)
        ks = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
        ll = [nll(probs_mix(lh, la, tr['Sh'], tr['Sa'], tr['Sha'], kk, True), k) for kk in ks]
        kb = ks[int(np.argmin(ll))]
        out['mix_kfit'] = probs_mix(te['lh'], te['la'], te['Sh'], te['Sa'], te['Sha'], kb, True); out['_kappa'] = kb
    if hz_tab is not None:
        out['chain'] = hz_tab(te['lh'], te['la'])
    return out


def run(resname, engine='ekf3', with_chain=True, names=None):
    d = lib.load()
    res = pickle.load(open(lib.D + resname, 'rb'))
    FULL = res['FULL']; seasons = res['seasons']
    cS = (2, 3, 4) if engine == 'ekf3' else (2, 3, None)
    allP = {}; extras = {}
    for yi, Y in enumerate(seasons):
        F = FULL[yi].astype(float)
        mtr = d.post & (d.lg == 0) & (d.season >= 2009) & (d.season < Y)
        mte = d.post & (d.lg == 0) & (d.season == Y)
        def pack(m):
            return dict(lh=F[m, 0], la=F[m, 1], hg=d.hg[m], ag=d.ag[m], k=d.k[m], Sh=F[m, cS[0]], Sa=F[m, cS[1]],
                        Sha=F[m, cS[2]] if cS[2] is not None else np.zeros(m.sum()))
        tab = None
        if with_chain:
            before = d.season < Y
            hz = v1h.fit_hazard(d, F[:, 0], F[:, 1], before & (((d.lg == 0) & (d.season >= 2007)) | ((d.lg == 1) & (d.season >= 2014))), before & (d.lg == 0) & (d.season >= Y - 3))
            tab = lib.ChainTable(hz, npts=60)
        o = layers(pack(mtr), pack(mte), tab)
        for kname, v in o.items():
            if kname.startswith('_'):
                extras.setdefault(kname, {})[Y] = v
            else:
                allP.setdefault(kname, np.zeros((d.n, 3)))[mte] = v
        print(Y, {kk: (np.round(v, 3).tolist() if isinstance(v, (list, np.ndarray)) else round(v, 3)) for kk, v in o.items() if kk.startswith('_')}, flush=True)
    sc = d.scored
    base = allP['indep'][sc]
    print('--- DEV, paired vs independent Poisson on the same lambdas')
    for kname, P in allP.items():
        r, l = lib.rps_ll(P[sc], d.k[sc]); rb, lb = lib.rps_ll(base, d.k[sc]); m = d.season[sc] <= 2021
        diff = (r - rb)[m]; rng = np.random.default_rng(7)
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        print(f'{kname:14s} DEV RPS {r[m].mean():.5f} LL {l[m].mean():.5f} | vs indep dRPS {diff.mean():+.5f} [{np.percentile(bs, 2.5):+.5f},{np.percentile(bs, 97.5):+.5f}] dLL {(l - lb)[m].mean():+.5f}')
    pickle.dump(dict(P=allP, extras=extras), open(lib.D + 'calib_' + resname, 'wb'))
    return allP, extras


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'ekf3')
