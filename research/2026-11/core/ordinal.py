"""Ordinal (ordered-logit) rating on the clipped goal difference, score-driven updates, optional shots augmentation.

   eta = r_H - r_A + hfa_L + g_H
   P(GD category <= j) = F(theta_j - eta),  F = logistic,  categories of the goal difference clipped at +-(ncat-1)/2
   thresholds (5 cats): -(c+c2), -c, +c, +(c+c2)     c = league-specific draw half-width (slowly adapted online)
   1X2:  P(A) = F(-c - eta),  P(D) = F(c - eta) - F(-c - eta),  P(H) = 1 - F(c - eta)

   update after the match (s = d log P(observed category) / d eta):
       s' = s + sum_k c_k * (relative stat surprise home - away)          (shots on target, shots)
       r_H += K * k_H * s'      r_A -= K * k_A * s'      k_i = 1 + boost * exp(-n_i / nb)   (n_i = matches this season)
       g_H += Kg * s'           hfa_L += max(kH, 1/(n_L+30)) * s
   season start: r *= rho, g *= rhoG; promoted to A: r -= dP; relegated to B: r += dP; new club in A: r = -dP

usage: ordinal.py name [name ...]
"""
import sys, math, pickle, time
import numpy as np
from numba import njit
import lib, core3

PK = ['K', 'rho', 'dP', 'kH', 'c1', 'c2', 'c3', 'boost', 'nb', 'Kg', 'rhoG', 'kC', 'hfa0', 'qd', 'ncat', 'Kt', 'tauK', 'cT', 'Ka', 'rhoA']
DEF = dict(K=0.035, rho=0.9, dP=0.6, kH=0.001, c1=0.62, c2=1.05, c3=1.0, boost=1.5, nb=1.0, Kg=0.01, rhoG=0.3, kC=1e-4, hfa0=0.45, qd=0.0, ncat=5, Kt=0.0, tauK=30.0, cT=0.0, Ka=0.0, rhoA=0.3)
LOGK = {'K', 'kH', 'c1', 'c2', 'c3', 'boost', 'nb', 'Kg', 'kC', 'qd', 'Kt', 'tauK', 'Ka'}
LGT = {'rho', 'rhoG', 'rhoA'}
NOUT = 8
SEASONS = core3.SEASONS


@njit(cache=True)
def _F(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True, error_model='numpy')
def ord_pass(h, a, lg, season, post, day, hg, ag, xr, ltot, p, nteams):
    K, rho, dP, kH, c1, c2, c3, boost, nb, Kg, rhoG, kC, hfa0, qd, ncat_f, Kt, tauK, cT, Ka, rhoA = p
    ncat = int(ncat_f)
    n = len(h)
    r = np.zeros(nteams); g = np.zeros(nteams); ga = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams); tlast = np.zeros(nteams)
    hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    cc = np.array([c1, c1])
    out = np.zeros((n, NOUT))
    th = np.zeros(6)
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                if L == 0 and se > 2007:
                    r[t] = -dP
                tse[t] = se; tlg[t] = L; tlast[t] = day[i]
            elif se > tse[t]:
                gap = se - tse[t]
                r[t] *= rho ** gap; g[t] *= rhoG ** gap; ga[t] *= rhoA ** gap
                if gap > 1:
                    r[t] = min(r[t], 0.0)
                elif L != tlg[t]:
                    r[t] += -dP if L == 0 else dP
                tse[t] = se; tlg[t] = L; tn[t] = 0
        H = h[i]; A = a[i]
        eta = r[H] - r[A] + hfa[L] + g[H] - ga[A]
        c = cc[L] * math.exp(-cT * ltot[i])          # optional: narrower draw band when many goals are expected
        pA = _F(-c - eta); pH = 1.0 - _F(c - eta); pD = 1.0 - pA - pH
        out[i, 0] = pH; out[i, 1] = pD; out[i, 2] = pA; out[i, 3] = eta; out[i, 4] = c
        out[i, 5] = r[H]; out[i, 6] = r[A]; out[i, 7] = g[H]
        if not post[i]:
            continue
        gd = hg[i] - ag[i]
        if ncat == 3:
            nth = 2; th[0] = -c; th[1] = c
            y = 0 if gd < 0 else 1 if gd == 0 else 2
        elif ncat == 5:
            nth = 4; th[0] = -c - c2; th[1] = -c; th[2] = c; th[3] = c + c2
            y = min(max(gd, -2), 2) + 2
        else:
            nth = 6; th[0] = -c - c2 - c3; th[1] = -c - c2; th[2] = -c; th[3] = c; th[4] = c + c2; th[5] = c + c2 + c3
            y = min(max(gd, -3), 3) + 3
        lo = 0.0 if y == 0 else _F(th[y - 1] - eta)
        hi = 1.0 if y == nth else _F(th[y] - eta)
        flo = 0.0 if y == 0 else lo * (1 - lo)
        fhi = 0.0 if y == nth else hi * (1 - hi)
        py = max(hi - lo, 1e-12)
        s = -(fhi - flo) / py
        sx = s + xr[i]
        kh = 1.0 + boost * math.exp(-tn[H] / nb); ka = 1.0 + boost * math.exp(-tn[A] / nb)
        if Kt > 0:      # extra gain after a long break (time-based uncertainty)
            kh += Kt * (1 - math.exp(-max(day[i] - tlast[H], 0.0) / tauK)); ka += Kt * (1 - math.exp(-max(day[i] - tlast[A], 0.0) / tauK))
        r[H] += K * kh * sx; r[A] -= K * ka * sx
        g[H] += Kg * sx
        if Ka > 0: ga[A] -= Ka * sx
        hfa[L] += max(kH, 1.0 / (Ln[L] + 30)) * s
        if kC > 0:
            a_ = _F(-c - eta); b_ = _F(c - eta)
            if gd == 0: sc = (b_ * (1 - b_) + a_ * (1 - a_)) / max(b_ - a_, 1e-12)
            elif gd > 0: sc = -b_
            else: sc = -(1 - a_)
            cc[L] += kC * sc
        Ln[L] += 1; tn[H] += 1; tn[A] += 1; tlast[H] = day[i]; tlast[A] = day[i]
    return out


def enc(k, v):
    if k in LOGK: return math.log(max(v, 1e-9))
    if k in LGT:
        v = min(0.9999, max(1e-4, v)); return math.log(v / (1 - v))
    return float(v)


def dec(k, x):
    if k in LOGK: return math.exp(min(x, 30))
    if k in LGT: return 1 / (1 + math.exp(-x))
    return float(x)


class OSpec:
    def __init__(self, name, free, fixed=None, lo=2009):
        self.name = name; self.free = list(free); self.fixed = dict(fixed or {}); self.lo = lo

    def params(self, x):
        p = dict(DEF); hp = {}
        for k, v in self.fixed.items(): (p if k in p else hp)[k] = v
        for k, xv in zip(self.free, x): (p if k in p else hp)[k] = dec(k, xv)
        return p, hp

    def x0(self):
        p, hp = self.params([]); allp = dict(p); allp.update(hp)
        return np.array([enc(k, allp.get(k, 0.0)) for k in self.free])


def run(d, spec, x, DS, ltot=None):
    p, hp = spec.params(x)
    xr = np.zeros(d.n)
    for c in ('sot', 'sh', 'co', 'soff'):
        ck = hp.get('c_' + c, 0.0)
        if ck != 0.0: xr = xr + ck * DS[c]
    if hp.get('c_red', 0.0) != 0.0:
        xr = xr + hp['c_red'] * np.where(d.evOk, d.fadv[:, 1] - d.fadv[:, 0], 0.0)   # home side handicapped while the away side had an extra man
    if ltot is None: ltot = np.zeros(d.n)
    return ord_pass(d.h, d.a, d.lg, d.season, d.post, d.day, d.hg, d.ag, xr, ltot, np.array([p[k] for k in PK], float), d.nteams)


def walk(d, spec, CHR, seasons=SEASONS, first=(3, 500), nxt=(2, 250), verbose=True, save=True, ltot_full=None, rw=0.0):
    t0 = time.time()
    x = spec.x0(); xref = x.copy()
    FULL = np.zeros((len(seasons), d.n, NOUT), np.float32); PAR = {}
    kk = np.where(d.post, d.k, 0)
    for yi, Y in enumerate(seasons):
        DS = {c: (lambda R: R[:, 0] - R[:, 1])(core3.rel_surprise(d, CHR['CH'], c, yi)) for c in CHR['CH']}
        mA = d.post & (d.season >= spec.lo) & (d.season < Y) & (d.lg == 0)
        mB = d.post & (d.season >= spec.lo) & (d.season < Y) & (d.lg == 1)
        lt = None if ltot_full is None else ltot_full[yi]
        xprev = x.copy()

        def f(z):
            out = run(d, spec, z, DS, lt)
            if not np.all(np.isfinite(out[:, :3])): return 1e9
            l = -np.log(np.clip(out[np.arange(d.n), kk], 1e-9, 1))
            v = l[mA].mean() + 0.3 * l[mB].mean() + 1e-6 * float(((z - xref) ** 2).sum()) + rw * float(((z - xprev) ** 2).sum())
            return v if np.isfinite(v) else 1e9
        r_, it = first if yi == 0 else nxt
        x, best = lib.nm(f, x, rounds=r_, maxiter=it)
        FULL[yi] = run(d, spec, x, DS, lt)
        p, hp = spec.params(x); allp = dict(p); allp.update(hp); PAR[Y] = dict(allp, _loss=best)
        if verbose:
            print(f'  {spec.name} {Y} loss {best:.5f} ' + str({k: round(dec(k, xv), 4) for k, xv in zip(spec.free, x)}), flush=True)
    P = np.zeros((d.n, 3))
    for yi, Y in enumerate(seasons):
        m = d.season == Y; P[m] = FULL[yi][m, :3]
    res = dict(name=spec.name, seasons=list(seasons), FULL=FULL, PAR=PAR, P=P, free=spec.free)
    res['dev'] = lib.dev_report(spec.name, d, P[d.scored])
    if verbose: print(f'  ({time.time() - t0:.0f}s)', flush=True)
    if save: pickle.dump(res, open(lib.D + f'out_{spec.name}.pkl', 'wb'))
    return res


OB = ['K', 'rho', 'dP', 'kH', 'c1', 'c2', 'boost', 'nb', 'Kg', 'rhoG', 'kC']
SPECS = {
    'O5': OSpec('O5', OB),
    'O3': OSpec('O3', [k for k in OB if k != 'c2'], dict(ncat=3)),
    'O5s': OSpec('O5s', OB + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3)),
    'O7s': OSpec('O7s', OB + ['c3', 'c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, ncat=7)),
    'O3s': OSpec('O3s', [k for k in OB if k != 'c2'] + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, ncat=3)),
    'O5s_co': OSpec('O5s_co', OB + ['c_sot', 'c_sh', 'c_co'], dict(c_sot=0.3, c_sh=0.3, c_co=0.05)),
    'O5s_nohome': OSpec('O5s_nohome', [k for k in OB if k not in ('Kg', 'rhoG')] + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, Kg=0.0)),
    'O5s_noboost': OSpec('O5s_noboost', [k for k in OB if k not in ('boost', 'nb')] + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, boost=0.0)),
    'O5s_Kt': OSpec('O5s_Kt', OB + ['c_sot', 'c_sh', 'Kt'], dict(c_sot=0.3, c_sh=0.3, Kt=0.3)),
    'O5s_nokC': OSpec('O5s_nokC', [k for k in OB if k != 'kC'] + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, kC=0.0)),
    'O5s_lean': OSpec('O5s_lean', ['K', 'rho', 'dP', 'c1', 'c2', 'boost', 'Kg', 'c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3, kH=0.001, nb=1.0, rhoG=0.3, kC=1e-4)),
    'O5s_rw3': OSpec('O5s_rw3', OB + ['c_sot', 'c_sh'], dict(c_sot=0.3, c_sh=0.3)),
}
OL = ['K', 'rho', 'dP', 'kH', 'c1', 'c2', 'Kg', 'rhoG']          # lean base: no early-season boost, no draw-width drift
LF = dict(boost=0.0, kC=0.0, c_sot=0.3, c_sh=0.3, c_co=0.2)
SPECS.update({
    'OL': OSpec('OL', OL + ['c_sot', 'c_sh', 'c_co'], LF),
    'OL_away': OSpec('OL_away', OL + ['c_sot', 'c_sh', 'c_co', 'Ka', 'rhoA'], dict(LF, Ka=0.01)),
    'OL_red': OSpec('OL_red', OL + ['c_sot', 'c_sh', 'c_co', 'c_red'], dict(LF, c_red=0.3)),
    'OL_rw3': OSpec('OL_rw3', OL + ['c_sot', 'c_sh', 'c_co'], LF),
    'OL_rw2': OSpec('OL_rw2', OL + ['c_sot', 'c_sh', 'c_co'], LF),
    'OL_rw3_nosh': OSpec('OL_rw3_nosh', OL + ['c_sot', 'c_co'], dict(LF, c_sh=0.0)),
    'OL_rw3_soff': OSpec('OL_rw3_soff', OL + ['c_sot', 'c_soff', 'c_co'], dict(LF, c_sh=0.0, c_soff=0.3)),
    'OL7_rw3': OSpec('OL7_rw3', OL + ['c3', 'c_sot', 'c_sh', 'c_co'], dict(LF, ncat=7)),
})
RW = {'O5s_rw3': 1e-3, 'OL_rw3': 1e-3, 'OL_rw2': 1e-2, 'OL_rw3_nosh': 1e-3, 'OL_rw3_soff': 1e-3, 'OL7_rw3': 1e-3}

if __name__ == '__main__':
    d = lib.load()
    CHR = core3.channel_preds(d)
    for k in sys.argv[1:]:
        walk(d, SPECS[k], CHR, rw=RW.get(k, 0.0))
