"""Combining the Poisson EKF (expected goals) with the ordinal rating, walk-forward.

For season Y every combiner is fitted on Serie A matches of seasons [2009, Y) using the passes run with the
hyper-parameters tuned on seasons < Y.

usage: combo.py out_<poisson>.pkl out_<ordinal>.pkl [tag]
"""
import sys, math, pickle
import numpy as np
from scipy.optimize import minimize
import lib, v1h, calib


def nll(P, k):
    return -np.log(np.clip(P[np.arange(len(k)), k], 1e-12, 1)).mean()


def F(x):
    return 1 / (1 + np.exp(-x))


def ord3(eta, c):
    pA = F(-c - eta); pH = 1 - F(c - eta)
    return np.c_[pH, 1 - pA - pH, pA]


def combine(d, pres, ores, with_chain=False, verbose=True, lo=2009, trainB=False):
    seasons = pres['seasons']
    names = ['pois', 'ord', 'avg50', 'wavg', 'logpool', 'stack_lam', 'stack_lam_side', 'meta_ord', 'meta_ord_tot']
    if with_chain: names.append('stack_lam_chain')
    P = {k: np.zeros((d.n, 3)) for k in names}
    LAM = np.zeros((d.n, 2)); COEF = {}
    for yi, Y in enumerate(seasons):
        Fp = pres['FULL'][yi].astype(float); Fo = ores['FULL'][yi].astype(float)
        mtr = d.post & ((d.lg == 0) | trainB) & (d.season >= lo) & (d.season < Y)
        mte = (d.season == Y)
        lh, la = Fp[:, 0], Fp[:, 1]; eta = Fo[:, 3]; cdr = Fo[:, 4]
        Pp = lib.pois_probs(np.maximum(lh, 1e-6), np.maximum(la, 1e-6)); Po = np.clip(Fo[:, :3], 1e-9, 1)
        k = d.k
        P['pois'][mte] = Pp[mte]; P['ord'][mte] = Po[mte]
        P['avg50'][mte] = 0.5 * Pp[mte] + 0.5 * Po[mte]
        # weighted average
        ws = np.linspace(0, 1, 21); ll = [nll(w * Pp[mtr] + (1 - w) * Po[mtr], k[mtr]) for w in ws]
        w = ws[int(np.argmin(ll))]
        P['wavg'][mte] = w * Pp[mte] + (1 - w) * Po[mte]
        # log-linear pool
        def lp(x, m):
            L = x[0] * np.log(Pp[m]) + x[1] * np.log(Po[m]); Q = np.exp(L - L.max(1, keepdims=True)); return Q / Q.sum(1, keepdims=True)
        r = minimize(lambda x: nll(lp(x, mtr), k[mtr]), [0.5, 0.5], method='Nelder-Mead')
        P['logpool'][mte] = lp(r.x, mte)
        # lambda stack: log lam = a0 + a1 log lam_pois +- a2 eta   (Poisson GLM, both sides)
        n = mtr.sum()
        X = np.r_[np.c_[np.ones(n), np.log(lh[mtr]), eta[mtr]], np.c_[np.ones(n), np.log(la[mtr]), -eta[mtr]]]
        y = np.r_[d.hg[mtr], d.ag[mtr]].astype(float)
        b = v1h.pois_glm(X, y, ridge=1e-3)
        L1 = np.exp(b[0] + b[1] * np.log(lh) + b[2] * eta); L2 = np.exp(b[0] + b[1] * np.log(la) - b[2] * eta)
        P['stack_lam'][mte] = lib.pois_probs(L1[mte], L2[mte]); LAM[mte, 0] = L1[mte]; LAM[mte, 1] = L2[mte]
        # with a side dummy (home advantage re-estimated)
        Xs = np.c_[X, np.r_[np.ones(n), np.zeros(n)]]
        bs = v1h.pois_glm(Xs, y, ridge=1e-3)
        L1s = np.exp(bs[0] + bs[3] + bs[1] * np.log(lh) + bs[2] * eta); L2s = np.exp(bs[0] + bs[1] * np.log(la) - bs[2] * eta)
        P['stack_lam_side'][mte] = lib.pois_probs(L1s[mte], L2s[mte])
        # meta ordered logit on [eta, log supremacy]
        sup = np.log(lh / la); tot = np.log(lh + la) - math.log(2.5)
        def mo(x, m, use_tot):
            c = cdr[m] * np.exp(x[2] + (x[3] * tot[m] if use_tot else 0.0))
            return ord3(x[0] * eta[m] + x[1] * sup[m] + x[4], c)
        r = minimize(lambda x: nll(mo(x, mtr, False), k[mtr]), [0.5, 0.5, 0.0, 0.0, 0.0], method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-5, fatol=1e-9))
        P['meta_ord'][mte] = mo(r.x, mte, False)
        r2 = minimize(lambda x: nll(mo(x, mtr, True), k[mtr]), r.x, method='Nelder-Mead', options=dict(maxiter=3000, xatol=1e-5, fatol=1e-9))
        P['meta_ord_tot'][mte] = mo(r2.x, mte, True)
        COEF[Y] = dict(w=float(w), logpool=r.x.tolist() if False else None, stack=b.tolist(), stack_side=bs.tolist(), meta=r.x.tolist(), meta_tot=r2.x.tolist())
        if with_chain:
            before = d.season < Y
            hz = v1h.fit_hazard(d, L1, L2, before & (((d.lg == 0) & (d.season >= 2007)) | ((d.lg == 1) & (d.season >= 2014))), before & (d.lg == 0) & (d.season >= Y - 3))
            tab = lib.ChainTable(hz, npts=60)
            mm = mte & (d.lg == 0)
            P['stack_lam_chain'][mm] = tab(L1[mm], L2[mm])
            COEF[Y]['hz'] = hz
        if verbose:
            print(Y, f'w={w:.2f} stack={np.round(b, 3)} side={np.round(bs, 3)} meta={np.round(r.x, 3)} meta_tot={np.round(r2.x, 3)}', flush=True)
    sc = d.scored
    for kname in names:
        lib.dev_report(kname, d, P[kname][sc])
    return P, LAM, COEF


if __name__ == '__main__':
    d = lib.load()
    pres = pickle.load(open(lib.D + sys.argv[1], 'rb')); ores = pickle.load(open(lib.D + sys.argv[2], 'rb'))
    tag = sys.argv[3] if len(sys.argv) > 3 else 'combo'
    P, LAM, COEF = combine(d, pres, ores, with_chain='chain' in sys.argv)
    pickle.dump(dict(P=P, LAM=LAM, COEF=COEF), open(lib.D + f'out_{tag}.pkl', 'wb'))
