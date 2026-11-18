"""Honest (walk-forward) re-run of the v1 pipeline.

For every season Y in 2013..2026 EVERYTHING fitted is estimated on seasons < Y only:
  xG proxy (OLS), goal-channel hyper-parameters (Nelder-Mead), sh/sot channel hyper-parameters,
  stacking layer (Poisson GLM), minute-by-minute hazard (Poisson GLMs + norm).
Objectives for the goal-channel tuning: 'pois' (v1: Poisson goal log-lik), 'rps', 'll' (1X2 log-loss).

Run:  .venv/bin/python v1h.py [pois|rps|ll]
"""
import sys, math, json, pickle
import numpy as np
from numba import njit
import lib

DEF = {
    'g': dict(q=0.0012, P0=0.06, rho=0.85, sSeason=0.01, dPromoO=0.18, dPromoD=0.18, dNew=0.15, w=0.6, phi=1.0, kMu=0.004, kH=0.004, Pg0=0.01, qg=0.0001, rhoG=0.9, wRed=0.8, mu0=math.log(1.25), hfa0=0.35),
    'sh': dict(q=0.0006, P0=0.03, rho=0.8, sSeason=0.004, dPromoO=0.05, dPromoD=0.05, dNew=0.05, phi=2.0, kMu=0.004, kH=0.004, Pg0=0, qg=0, rhoG=0, mu0=math.log(12.5), hfa0=0.2),
    'sot': dict(q=0.0006, P0=0.03, rho=0.8, sSeason=0.004, dPromoO=0.08, dPromoD=0.08, dNew=0.08, phi=1.5, kMu=0.004, kH=0.004, Pg0=0, qg=0, rhoG=0, mu0=math.log(4.3), hfa0=0.22),
}
LOGK = {'q', 'P0', 'sSeason', 'phi', 'kMu', 'kH', 'Pg0', 'qg'}
LGT = {'rho', 'w', 'rhoG', 'wRed'}
GKEYS = ['q', 'P0', 'rho', 'sSeason', 'dPromoO', 'dPromoD', 'dNew', 'w', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG', 'wRed']
OKEYS = ['q', 'P0', 'rho', 'sSeason', 'phi']


def enc(p, keys):
    out = []
    for k in keys:
        v = p[k]
        if k in LOGK: out.append(math.log(max(v, 1e-7)))
        elif k in LGT:
            v = min(0.9999, max(1e-4, v)); out.append(math.log(v / (1 - v)))
        else: out.append(v)
    return np.array(out)


def dec(x, keys, base):
    p = dict(base)
    for k, v in zip(keys, x):
        p[k] = math.exp(v) if k in LOGK else 1 / (1 + math.exp(-v)) if k in LGT else float(v)
    return p


def pvec(p):
    return np.array([p.get(k, 0.0) for k in lib.V1_KEYS])


def fit_xg(d, mask):
    m = mask & d.statsOk & d.post
    if m.sum() < 300:
        return None
    X = np.concatenate([np.c_[np.ones(m.sum()), d.st_sot[m, s], np.maximum(0, d.st_sh[m, s] - d.st_sot[m, s])] for s in (0, 1)])
    y = np.concatenate([d.hg[m], d.ag[m]]).astype(float)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(b0=b[0], b1=b[1], b2=b[2])


def obs_g(d, wRed, w, xg):
    gh = np.where(d.evOk, d.g_all[:, 0] - (1 - wRed) * d.g_adv[:, 0], d.hg).astype(float)
    ga = np.where(d.evOk, d.g_all[:, 1] - (1 - wRed) * d.g_adv[:, 1], d.ag).astype(float)
    if xg is not None:
        xh = np.maximum(0.05, xg['b0'] + xg['b1'] * d.st_sot[:, 0] + xg['b2'] * np.maximum(0, d.st_sh[:, 0] - d.st_sot[:, 0]))
        xa = np.maximum(0.05, xg['b0'] + xg['b1'] * d.st_sot[:, 1] + xg['b2'] * np.maximum(0, d.st_sh[:, 1] - d.st_sot[:, 1]))
        gh = np.where(d.statsOk, w * gh + (1 - w) * xh, gh); ga = np.where(d.statsOk, w * ga + (1 - w) * xa, ga)
    return gh, ga


def run_g(d, p, xg):
    gh, ga = obs_g(d, p['wRed'], p['w'], xg)
    return lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, gh, ga, d.post, pvec(p), True, d.nteams)


def run_o(d, c, p):
    y = getattr(d, 'st_' + c)
    return lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, np.nan_to_num(y[:, 0]), np.nan_to_num(y[:, 1]), d.statsOk, pvec(p), False, d.nteams)


def make_loss(d, objective, lo, hi, yh, ya, obsmask):
    """loss over seasons [lo, hi): -(mean over A) - 0.3 (mean over B), like refit.mjs channelLoss."""
    mA = d.post & obsmask & (d.season >= lo) & (d.season < hi) & (d.lg == 0)
    mB = d.post & obsmask & (d.season >= lo) & (d.season < hi) & (d.lg == 1)
    kA = d.k[mA]; kB = d.k[mB]

    def loss(out):
        lh, la = out[:, 0], out[:, 1]
        if not np.all(np.isfinite(lh)) or not np.all(np.isfinite(la)):
            return 1e9
        if objective == 'pois':
            v = lib.pois_ll(yh, lh) + lib.pois_ll(ya, la)
            f = -v[mA].mean() - (0.3 * v[mB].mean() if mB.sum() else 0.0)
        else:
            j = 0 if objective == 'rps' else 1
            f = lib.rps_ll(lib.pois_probs(lh[mA], la[mA]), kA)[j].mean()
            if mB.sum():
                f += 0.3 * lib.rps_ll(lib.pois_probs(lh[mB], la[mB]), kB)[j].mean()
        return f if np.isfinite(f) else 1e9
    return loss


def tune_g(d, Y, p0, xg, objective, lo=2009, rounds=2, maxiter=450):
    L = make_loss(d, objective, lo, Y, d.hg, d.ag, d.post)
    f = lambda x: L(run_g(d, dec(x, GKEYS, p0), xg))
    x, best = lib.nm(f, enc(p0, GKEYS), rounds=rounds, maxiter=maxiter)
    return dec(x, GKEYS, p0), best


def tune_o(d, c, Y, p0, lo):
    y = getattr(d, 'st_' + c)
    L = make_loss(d, 'pois', lo, Y, np.nan_to_num(y[:, 0]), np.nan_to_num(y[:, 1]), d.statsOk)
    f = lambda x: L(run_o(d, c, dec(x, OKEYS, p0)))
    x, best = lib.nm(f, enc(p0, OKEYS), rounds=1, maxiter=160)
    return dec(x, OKEYS, p0), best


def pois_glm(X, y, offset=None, ridge=1e-6, iters=40):
    n, k = X.shape; b = np.zeros(k)
    off = np.zeros(n) if offset is None else offset
    for _ in range(iters):
        mu = np.exp(off + X @ b)
        g = X.T @ (y - mu) - ridge * b
        A = (X * mu[:, None]).T @ X + ridge * np.eye(k)
        dlt = np.linalg.solve(A, g)
        b += dlt
        if np.abs(dlt).max() < 1e-9:
            break
    return b


def fit_stack(d, og, osh, oso, mask):
    m = mask & d.post
    if 2 * m.sum() < 3000:
        return None
    X = np.concatenate([np.c_[np.ones(m.sum()), np.log(og[m, 0]), oso[m, 6], osh[m, 6]], np.c_[np.ones(m.sum()), np.log(og[m, 1]), oso[m, 7], osh[m, 7]]])
    y = np.concatenate([d.hg[m], d.ag[m]]).astype(float)
    return pois_glm(X, y, ridge=1e-3)


def apply_stack(st, og, osh, oso):
    if st is None:
        return og[:, 0].copy(), og[:, 1].copy()
    lh = np.exp(st[0] + st[1] * np.log(og[:, 0]) + st[2] * oso[:, 6] + st[3] * osh[:, 6])
    la = np.exp(st[0] + st[1] * np.log(og[:, 1]) + st[2] * oso[:, 7] + st[3] * osh[:, 7])
    return lh, la


@njit(cache=True)
def hazard_cells(idx, lh, la, gm, gx, gs, rm, rx, rs, bins, S1, S2):
    # cells[side, bin, diff, men, (eG, g, eM, r)]
    C = np.zeros((2, 8, 5, 3, 4))
    nst = len(bins)
    evg = np.zeros((nst, 2), np.int64); evr = np.zeros((nst, 2), np.int64)
    for ii in range(len(idx)):
        i = idx[ii]
        evg[:] = 0; evr[:] = 0
        for j in range(gm.shape[1]):
            if gm[i, j] < 0: break
            evg[lib.step_of(S1, S2, gm[i, j], gx[i, j]), gs[i, j]] += 1
        for j in range(rm.shape[1]):
            if rm[i, j] < 0: break
            evr[lib.step_of(S1, S2, rm[i, j], rx[i, j]), rs[i, j]] += 1
        hg = 0; ag = 0; rh = 0; ra = 0
        for t in range(nst):
            b = bins[t]
            dh = lib._diff_idx(hg - ag); da = lib._diff_idx(ag - hg)
            mh = lib._men_idx(rh, ra); ma = lib._men_idx(ra, rh)
            C[0, b, dh, mh, 0] += lh[i] / 90; C[1, b, da, ma, 0] += la[i] / 90
            C[0, b, dh, mh, 2] += 1; C[1, b, da, ma, 2] += 1
            C[0, b, dh, mh, 1] += evg[t, 0]; C[1, b, da, ma, 1] += evg[t, 1]
            C[0, b, dh, mh, 3] += evr[t, 0]; C[1, b, da, ma, 3] += evr[t, 1]
            hg += evg[t, 0]; ag += evg[t, 1]; rh += evr[t, 0]; ra += evr[t, 1]
    return C


def fit_hazard(d, lh, la, mask, norm_mask):
    S1, S2 = 3, 7
    bins = lib.timeline_bins(S1, S2)
    idx = np.where(mask & d.post & d.evOk)[0]
    C = hazard_cells(idx, lh, la, d.gm, d.gx, d.gs, d.rm, d.rx, d.rs, bins, S1, S2)
    XG, yG, oG, XR, yR, oR = [], [], [], [], [], []
    for side in range(2):
        for b in range(8):
            for dd in range(5):
                for mn in range(3):
                    c = C[side, b, dd, mn]
                    if c[2] < 1: continue
                    x = np.zeros(16); x[b] = 1
                    if dd != 2: x[8 + (dd - 1 if dd > 2 else dd)] = 1
                    xg_ = x.copy()
                    if mn != 1: xg_[12 + (1 if mn == 2 else 0)] = 1
                    if b >= 6 and dd < 2: xg_[14] = 1
                    xr = x.copy()
                    if side == 1: xr[15] = 1
                    XG.append(xg_); yG.append(c[1]); oG.append(math.log(c[0]))
                    XR.append(xr); yR.append(c[3]); oR.append(math.log(c[2]))
    bG = pois_glm(np.array(XG), np.array(yG), np.array(oG), ridge=1e-4)
    bR = pois_glm(np.array(XR), np.array(yR), np.array(oR), ridge=1e-4)
    df = lambda b: [b[8], b[9], 0.0, b[10], b[11]]
    hz = dict(S1=S1, S2=S2, goal_bin=list(bG[:8]), goal_diff=df(bG), goal_men=[bG[12], 0.0, bG[13]], lateTrail=bG[14], norm=1.0,
              red_bin=list(bR[:8]), red_diff=df(bR), red_away=bR[15])
    # normalisation: chain-implied expected totals must match lambda on real games
    nm_ = np.where(norm_mask & d.post)[0][::4]
    goalF, redH, redA = lib.compile_hazard(hz)
    r = lib.chain_probs(bins, goalF, redH, redA, lh[nm_], la[nm_])
    hz['norm'] = float((lh[nm_] + la[nm_]).sum() / (r[:, 3] + r[:, 4]).sum())
    return hz


def main(objective='pois', tag=None, seasons=range(2013, 2027)):
    d = lib.load()
    tag = tag or objective
    pg = dict(DEF['g']); psh = dict(DEF['sh']); pso = dict(DEF['sot'])
    P = np.zeros((d.n, 3)); LAM = np.zeros((d.n, 2)); LAMG = np.zeros((d.n, 2))
    log = {}
    for Y in seasons:
        before = d.season < Y
        xg = fit_xg(d, before)
        first = (Y == seasons[0])
        pg, fg = tune_g(d, Y, pg, xg, objective, rounds=3 if first else 2, maxiter=450 if first else 300)
        nst = (before & d.statsOk & d.post).sum()
        if nst >= 700:
            lo_o = 2014 if (before & d.statsOk & (d.season >= 2015)).sum() < 300 else 2015
            psh, _ = tune_o(d, 'sh', Y, psh, lo_o); pso, _ = tune_o(d, 'sot', Y, pso, lo_o)
        og = run_g(d, pg, xg); osh = run_o(d, 'sh', psh); oso = run_o(d, 'sot', pso)
        st = fit_stack(d, og, osh, oso, before & (d.season >= 2015))
        lh, la = apply_stack(st, og, osh, oso)
        hz = fit_hazard(d, lh, la, before & (((d.lg == 0) & (d.season >= 2007)) | ((d.lg == 1) & (d.season >= 2014))), before & (d.lg == 0) & (d.season >= Y - 3))
        tab = lib.ChainTable(hz)
        m = d.season == Y
        P[m] = tab(lh[m], la[m]); LAM[m, 0] = lh[m]; LAM[m, 1] = la[m]; LAMG[m] = og[m, :2]
        log[Y] = dict(pg=pg, psh=psh, pso=pso, xg=xg, stack=None if st is None else list(st), hz=hz, loss=fg)
        sc = m & d.scored
        r, l = lib.rps_ll(P[sc], d.k[sc])
        print(Y, f'loss {fg:.5f}', f'RPS {r.mean():.4f} LL {l.mean():.4f}' if Y <= 2021 else '(holdout season: metrics hidden)', {k: round(v, 5) for k, v in pg.items() if k in GKEYS}, 'stack', None if st is None else np.round(st, 3), flush=True)
    sc = d.scored
    lib.write_preds('v1honest_' + tag, d.ids[sc], P[sc])
    pickle.dump(dict(log=log, P=P, LAM=LAM, LAMG=LAMG), open(lib.D + f'out_v1honest_{tag}.pkl', 'wb'))
    lib.dev_report('v1 honest / ' + tag, d, P[sc])


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'pois')
