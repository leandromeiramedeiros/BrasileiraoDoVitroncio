"""Question 2: alternative estimators on goals only, walk-forward tuned, compared with the goals-only EKF (g_pure).

usage: exp_alt.py gas | ord3 | ord5 | full
"""
import sys, math, pickle, time
import numpy as np
import lib, alt

d = lib.load()
SEASONS = list(range(2013, 2027))
hgf = d.hg.astype(float); agf = d.ag.astype(float)


def walk(name, pass_fn, x0, loss_kind, tune_every=1, first=(3, 500), nxt=(2, 250), prob_fn=None, lo=2009, with_yi=False):
    """pass_fn(x) -> out; loss on seasons [lo, Y)."""
    t0 = time.time()
    x = np.array(x0, float); xinit = x.copy()
    lam = np.zeros((d.n, 2)); P = np.zeros((d.n, 3)); PAR = {}; EX = np.zeros((d.n, 4))
    for yi, Y in enumerate(SEASONS):
        mA = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 0)
        mB = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 1)

        def f(z, yi=yi):
            out = pass_fn(z, yi) if with_yi else pass_fn(z)
            if not np.all(np.isfinite(out[:, :2])):
                return 1e9
            if loss_kind == 'pois':
                if out[:, :2].min() <= 0: return 1e9
                v = lib.pois_ll(hgf, out[:, 0]) + lib.pois_ll(agf, out[:, 1])
                r = -v[mA].mean() - 0.3 * v[mB].mean()
            else:
                pr = np.clip(out[:, :3], 1e-9, 1)
                l = -np.log(pr[np.arange(d.n), np.where(d.post, d.k, 0)])
                r = l[mA].mean() + 0.3 * l[mB].mean()
            r += 1e-6 * float(((z - xinit) ** 2).sum())
            return r if np.isfinite(r) else 1e9
        if yi % tune_every == 0:
            r_, it = first if yi == 0 else nxt
            x, best = lib.nm(f, x, rounds=r_, maxiter=it)
        out = pass_fn(x, yi) if with_yi else pass_fn(x)
        m = d.season == Y
        if loss_kind == 'pois':
            lam[m] = out[m, :2]
            EX[m, :out.shape[1] - 2] = out[m, 2:6] if out.shape[1] > 2 else 0
        else:
            P[m] = out[m, :3]; EX[m, 0] = out[m, 3]
        PAR[Y] = x.copy()
        print(f'  {name} {Y} loss {best:.5f} x {np.round(x, 4)}', flush=True)
    sc = d.scored
    if loss_kind == 'pois':
        P[sc] = lib.pois_probs(lam[sc, 0], lam[sc, 1])
    lib.dev_report(name, d, P[sc])
    pickle.dump(dict(name=name, lam=lam, P=P, PAR=PAR, EX=EX), open(lib.D + f'out_{name}.pkl', 'wb'))
    print(f'  ({time.time() - t0:.0f}s)')
    return P


def lgt(x): return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
    for which in sys.argv[1:]:
        if which == 'gas':
            # x: log Ao, log Ad, logit B, logit rho, dPO, dPD, log kMu, log kH, log Ag, logit rhoG, log boost, log nb
            def pf(x):
                p = np.array([math.exp(x[0]), math.exp(x[1]), lgt(x[2]), lgt(x[3]), x[4], x[5], math.exp(x[6]), math.exp(x[7]), math.exp(x[8]), lgt(x[9]),
                              math.log(1.25), 0.35, math.exp(x[10]), math.exp(x[11])])
                return alt.gas_pois(d.h, d.a, d.lg, d.season, d.post, hgf, agf, p, d.nteams)
            x0 = [math.log(0.012), math.log(0.012), 6.0, 1.7, 0.25, 0.2, math.log(0.002), math.log(0.0005), math.log(0.004), -1.5, math.log(0.5), math.log(8.0)]
            walk('alt_gas', pf, x0, 'pois')
        elif which in ('ord3', 'ord5'):
            nc = 3 if which == 'ord3' else 5
            # x: log K, logit rho, dP, log kH, log c1, log c2, log boost, log nb, log Kg, logit rhoG, log kC
            def pf(x, nc=nc):
                p = np.array([math.exp(x[0]), lgt(x[1]), x[2], math.exp(x[3]), math.exp(x[4]), math.exp(x[5]), math.exp(x[6]), math.exp(x[7]), 0.45,
                              math.exp(x[8]), lgt(x[9]), math.exp(x[10])])
                return alt.gas_ord(d.h, d.a, d.lg, d.season, d.post, d.hg, d.ag, p, d.nteams, nc, np.zeros(d.n))
            x0 = [math.log(0.04), 1.7, 0.45, math.log(0.001), math.log(0.6), math.log(1.0), math.log(0.5), math.log(8.0), math.log(0.01), -1.5, math.log(1e-4)]
            walk('alt_' + which, pf, x0, 'll')
        elif which in ('ord5s', 'ord7s', 'ord5s_sot'):
            import core3
            CHR = core3.channel_preds(d)
            RS = [{c: core3.rel_surprise(d, CHR['CH'], c, yi) for c in ('sot', 'sh')} for yi in range(len(SEASONS))]
            DS = [{c: (R[c][:, 0] - R[c][:, 1]) for c in R} for R in RS]
            # x: log K, logit rho, dP, log kH, log c1, log c2, log boost, log nb, log Kg, logit rhoG, log kC, c_sot, c_sh
            def pf(x, yi, which=which):
                p = np.array([math.exp(x[0]), lgt(x[1]), x[2], math.exp(x[3]), math.exp(x[4]), math.exp(x[5]), math.exp(x[6]), math.exp(x[7]), 0.45,
                              math.exp(x[8]), lgt(x[9]), math.exp(x[10])])
                xr = x[11] * DS[yi]['sot'] + (x[12] * DS[yi]['sh'] if which != 'ord5s_sot' else 0.0)
                return alt.gas_ord(d.h, d.a, d.lg, d.season, d.post, d.hg, d.ag, p, d.nteams, 5, xr)
            x0 = [math.log(0.04), 1.7, 0.45, math.log(0.001), math.log(0.6), math.log(1.0), math.log(0.5), math.log(8.0), math.log(0.01), -1.5, math.log(1e-4), 0.3, 0.3]
            if which == 'ord5s_sot': x0 = x0[:12]
            walk('alt_' + which, pf, x0, 'll', with_yi=True)
        elif which in ('full', 'fullc', 'fullbd', 'fullbd0'):
            Yobs = np.c_[hgf, agf]
            bd = which in ('fullbd', 'fullbd0'); usec = which in ('fullc', 'fullbd')
            # x: log qm, log qd, log P0, logit rho, log sS, dPO, dPD, log phi, log qmu, log qh, log Pg0, log qg, logit rhoG, log sG, [atanh cod]
            def pf(x, bd=bd, usec=usec):
                cod = math.tanh(x[14]) if usec else 0.0
                p = np.array([math.exp(x[0]), math.exp(x[1]), math.exp(x[2]), lgt(x[3]), math.exp(x[4]), x[5], x[6], math.exp(x[7]), math.exp(x[8]), math.exp(x[9]),
                              math.exp(x[10]), math.exp(x[11]), lgt(x[12]), math.exp(x[13]), math.log(1.25), 0.35, 0.0, 1.0, cod])
                return alt.ekf_full(d.h, d.a, d.lg, d.season, d.post, d.day, Yobs, p, d.nteams, bd)
            x0 = [math.log(1e-4), math.log(2e-5), math.log(0.04), 1.7, math.log(0.01), 0.27, 0.2, math.log(2.0), math.log(1e-5), math.log(1e-5), math.log(0.02), math.log(0.001), -1.5, math.log(0.01)]
            if usec: x0 = x0 + [0.3]
            walk('alt_' + which, pf, x0, 'pois', tune_every=2, first=(2, 350), nxt=(1, 200))
