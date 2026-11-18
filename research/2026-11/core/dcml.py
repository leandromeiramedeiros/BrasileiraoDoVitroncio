"""Question 2(a): time-decayed Dixon-Coles style maximum likelihood with ridge (hierarchical) shrinkage,
refit weekly on all Serie A + B matches played before the refit date.

   log lam_h = mu_L + gam_L/2 + alpha_H - beta_A        alpha_i = dA * fB_i + a_i ,  a_i ~ N(0, 1/ridge)
   log lam_a = mu_L - gam_L/2 + alpha_A - beta_H        beta_i  = dD * fB_i + b_i ,  b_i ~ N(0, 1/ridge)
   weights    w = 2^(-age/hl),  age = days + brk * (number of season boundaries crossed)
   fB_i = decay-weighted share of the team's matches that were played in Serie B (prior mean of a B-calibre side)

Hyper-parameters (hl, ridge, brk) are chosen per season on a grid using only earlier seasons.
usage: dcml.py
"""
import math, pickle, itertools, time
import numpy as np
from numba import njit
import lib


@njit(cache=True, error_model='numpy')
def dcml_pass(h, a, lg, season, post, day, hg, ag, hl, ridge, brk, nteams, refit_days, upto):
    n = len(h)
    npar = 6 + 2 * nteams
    th = np.zeros(npar); th[0] = math.log(1.25); th[1] = math.log(1.25); th[2] = 0.35; th[3] = 0.35
    out = np.zeros((n, 2))
    fB = np.zeros(nteams); wsum = np.zeros(nteams)
    next_fit = -1e9
    first = True
    g = np.zeros(npar); Hs = np.zeros((npar, npar))
    idx = np.zeros(6, np.int64); val = np.zeros(6)
    lnk = math.log(2.0) / hl
    start = 0
    for i in range(n):
        if season[i] > upto:
            break
        if day[i] >= next_fit and i > 200:
            T = math.floor(day[i]); seT = season[i]
            next_fit = T + refit_days
            # window start: drop matches with negligible weight
            while start < i and (T - day[start]) + brk * (seT - season[start]) > 7.0 * hl:
                start += 1
            # team B-share
            fB[:] = 0; wsum[:] = 0
            for j in range(start, i):
                if not post[j] or day[j] >= T: continue
                w = math.exp(-lnk * ((T - day[j]) + brk * (seT - season[j])))
                wsum[h[j]] += w; wsum[a[j]] += w
                if lg[j] == 1:
                    fB[h[j]] += w; fB[a[j]] += w
            for t in range(nteams):
                fB[t] = fB[t] / wsum[t] if wsum[t] > 0 else 0.0
            for it in range(12 if first else 3):
                g[:] = 0; Hs[:, :] = 0
                for j in range(start, i):
                    if not post[j] or day[j] >= T: continue
                    w = math.exp(-lnk * ((T - day[j]) + brk * (seT - season[j])))
                    L = lg[j]; H_ = h[j]; A_ = a[j]
                    for side in range(2):
                        att = H_ if side == 0 else A_; dfn = A_ if side == 0 else H_
                        y = hg[j] if side == 0 else ag[j]
                        idx[0] = L; val[0] = 1.0
                        idx[1] = 2 + L; val[1] = 0.5 if side == 0 else -0.5
                        idx[2] = 4; val[2] = fB[att]
                        idx[3] = 5; val[3] = -fB[dfn]
                        idx[4] = 6 + att; val[4] = 1.0
                        idx[5] = 6 + nteams + dfn; val[5] = -1.0
                        eta = 0.0
                        for q in range(6): eta += val[q] * th[idx[q]]
                        lam = math.exp(eta)
                        for q in range(6):
                            g[idx[q]] += w * (y - lam) * val[q]
                            for r in range(6):
                                Hs[idx[q], idx[r]] += w * lam * val[q] * val[r]
                for q in range(npar):
                    rd = ridge if q >= 6 else (0.5 if q >= 4 else 1e-3)
                    g[q] -= rd * (th[q] if q >= 4 else 0.0)
                    Hs[q, q] += rd
                th += np.linalg.solve(Hs, g)
            first = False
        L = lg[i]; H_ = h[i]; A_ = a[i]
        # a team never seen in the window: B-type prior if it now plays in B, promoted prior if it shows up in A
        fh = fB[H_] if wsum[H_] > 0 else 1.0
        fa = fB[A_] if wsum[A_] > 0 else 1.0
        out[i, 0] = math.exp(th[L] + th[2 + L] / 2 + th[4] * fh - th[5] * fa + th[6 + H_] - th[6 + nteams + A_])
        out[i, 1] = math.exp(th[L] - th[2 + L] / 2 + th[4] * fa - th[5] * fh + th[6 + A_] - th[6 + nteams + H_])
    return out


if __name__ == '__main__':
    d = lib.load()
    hgf = d.hg.astype(float); agf = d.ag.astype(float)
    grid = list(itertools.product([150.0, 250.0, 400.0, 650.0], [4.0, 8.0, 16.0, 32.0], [0.0, 90.0]))
    res = {}
    t0 = time.time()
    for gi, (hl, rd, brk) in enumerate(grid):
        res[(hl, rd, brk)] = dcml_pass(d.h, d.a, d.lg, d.season, d.post, d.day, hgf, agf, hl, rd, brk, d.nteams, 7.0, 2026)
        print(gi, hl, rd, brk, f'{time.time() - t0:.0f}s', flush=True)
    pickle.dump(res, open(lib.D + 'out_dcml_grid.pkl', 'wb'))
    lam = np.zeros((d.n, 2)); choice = {}
    for Y in range(2013, 2027):
        mA = d.post & (d.season >= 2009) & (d.season < Y) & (d.lg == 0)
        mB = d.post & (d.season >= 2009) & (d.season < Y) & (d.lg == 1)
        best = None
        for k, o in res.items():
            v = lib.pois_ll(hgf, np.maximum(o[:, 0], 1e-6)) + lib.pois_ll(agf, np.maximum(o[:, 1], 1e-6))
            f = -v[mA].mean() - 0.3 * v[mB].mean()
            if best is None or f < best[0]: best = (f, k)
        choice[Y] = best[1]
        m = d.season == Y
        lam[m] = res[best[1]][m]
        print(Y, best)
    sc = d.scored
    lib.dev_report('alt_dcml', d, lib.pois_probs(lam[sc, 0], lam[sc, 1]))
    pickle.dump(dict(name='alt_dcml', lam=lam, choice=choice), open(lib.D + 'out_alt_dcml.pkl', 'wb'))
