"""Engine v3 (the candidate production model) + walk-forward driver.

Goal channel EKF, per team a 2x2 covariance block for (attack o, defence d) with prior correlation `cod`
(cod = 0 reproduces v1 exactly), per-team home effect g, league level mu and home advantage hfa.

   log lam_h = mu_L + hfa_L/2 + g_H + o_H - d_A  (+ bcov if empty stadium)
   log lam_a = mu_L - hfa_L/2 - g_H + o_A - d_H  (- bcov)

Innovation ("adjusted performance"):
   e_h = y_h - lam_h * E_h + lam_h * X_h
   y   = goals with category weights (decided-game goals, penalties, ...), E = optional exposure,
   X_h = sum_k c_k * (stat_k,h / expected stat_k,h - 1)   relative surprise in shots on target / shots / corners,
         where the expected stats come from the (v1) per-channel EKFs run on their own likelihood.
"""
import math, pickle, time
import numpy as np
from numba import njit
import lib, v1h

CORE = ['qm', 'qd', 'P0', 'rho', 'sS', 'cap', 'dPO', 'dPD', 'dNew', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG', 'kappa', 'Ppromo', 'cod',
        'mu0', 'hfa0', 'kCov', 'sG']
DEFAULT = dict(qm=1e-4, qd=0.0, P0=0.015, rho=0.85, sS=1.0, cap=1.0, dPO=0.2, dPD=0.2, dNew=0.0, phi=1.0, kMu=0.002, kH=0.0005,
               Pg0=0.01, qg=0.001, rhoG=0.2, kappa=1.0, Ppromo=0.0, cod=0.0, mu0=math.log(1.25), hfa0=0.35, kCov=0.0, sG=-1.0)
LOGK = {'qm', 'qd', 'P0', 'sS', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'Ppromo', 'kCov', 'sG'}
LGT = {'rho', 'rhoG', 'w', 'wRed', 'wPen', 'wOwn', 'wLead', 'wTrail', 'wLead2', 'wTrail2'}
TANH = {'cod'}
NOUT = 20
SEASONS = list(range(2013, 2027))


@njit(cache=True, error_model='numpy')
def ekf3(h, a, lg, season, post, day, Y, E, XR, covid, p, nteams):
    (qm, qd, P0, rho, sS, cap, dPO, dPD, dNew, phi, kMu, kH, Pg0, qg, rhoG, kappa, Ppromo, cod, mu0, hfa0, kCov, sG) = p
    n = len(h)
    if sG < 0: sG = qg * 20
    o = np.zeros(nteams); d = np.zeros(nteams); g = np.zeros(nteams)
    Poo = np.zeros(nteams); Pdd = np.zeros(nteams); Pod = np.zeros(nteams); Pg = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams, np.int64); tlast = np.zeros(nteams)
    mu = np.array([mu0, mu0]); hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    bcov = 0.0
    out = np.zeros((n, NOUT))
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                fresh = (L == 0 and se <= 2007)
                off = 0.0 if fresh else -dNew
                o[t] = off; d[t] = off; g[t] = 0.0
                Poo[t] = P0; Pdd[t] = P0; Pod[t] = cod * P0; Pg[t] = Pg0
                if L == 0 and se > 2007:
                    o[t] = -dPO; d[t] = -dPD
                tse[t] = se; tlg[t] = L; tn[t] = 0; tlast[t] = day[i]
            elif se > tse[t]:
                gap = se - tse[t]
                r = rho ** gap
                o[t] *= r; d[t] *= r
                Poo[t] = r * r * Poo[t] + sS * gap; Pdd[t] = r * r * Pdd[t] + sS * gap; Pod[t] = r * r * Pod[t] + cod * sS * gap
                if cap > 0.5:
                    if Poo[t] >= P0 and Pdd[t] >= P0:
                        Poo[t] = P0; Pdd[t] = P0; Pod[t] = cod * P0
                    else:
                        Poo[t] = min(P0, Poo[t]); Pdd[t] = min(P0, Pdd[t])
                        lim = 0.98 * math.sqrt(Poo[t] * Pdd[t])
                        Pod[t] = max(-lim, min(lim, Pod[t]))
                rg = rhoG ** gap
                g[t] *= rg; Pg[t] = min(Pg0, rg * rg * Pg[t] + sG * gap)
                if gap > 1:
                    o[t] = min(o[t], -dNew); d[t] = min(d[t], -dNew); Poo[t] = P0; Pdd[t] = P0; Pod[t] = cod * P0
                elif L != tlg[t]:
                    if L == 0:
                        o[t] = kappa * o[t] - dPO; d[t] = kappa * d[t] - dPD
                        Poo[t] += Ppromo; Pdd[t] += Ppromo; Pod[t] += cod * Ppromo
                    else:
                        o[t] += dPO; d[t] += dPD
                tse[t] = se; tlg[t] = L; tn[t] = 0
        H = h[i]; A = a[i]
        dtH = day[i] - tlast[H]; dtA = day[i] - tlast[A]
        if qd > 0:
            aH = qd * min(max(dtH, 0.0), 400.0); aA = qd * min(max(dtA, 0.0), 400.0)
            Poo[H] += aH; Pdd[H] += aH; Pod[H] += cod * aH
            Poo[A] += aA; Pdd[A] += aA; Pod[A] += cod * aA
            tlast[H] = day[i]; tlast[A] = day[i]
        hx = g[H] + (bcov if covid[i] else 0.0)
        lh = math.exp(mu[L] + hfa[L] / 2 + hx + o[H] - d[A])
        la = math.exp(mu[L] - hfa[L] / 2 - hx + o[A] - d[H])
        out[i, 0] = lh; out[i, 1] = la
        out[i, 2] = Poo[H] + Pdd[A] + Pg[H]; out[i, 3] = Poo[A] + Pdd[H] + Pg[H]; out[i, 4] = -Pod[H] - Pod[A] - Pg[H]
        out[i, 5] = mu[L]; out[i, 6] = hfa[L]
        out[i, 7] = o[H]; out[i, 8] = d[H]; out[i, 9] = o[A]; out[i, 10] = d[A]; out[i, 11] = hx
        out[i, 12] = tn[H]; out[i, 13] = tn[A]; out[i, 14] = dtH; out[i, 15] = dtA
        out[i, 16] = Poo[H]; out[i, 17] = Pdd[H]; out[i, 18] = Poo[A]; out[i, 19] = Pdd[A]
        if not post[i]:
            continue
        Poo[H] += qm; Pdd[H] += qm; Pod[H] += cod * qm; Poo[A] += qm; Pdd[A] += qm; Pod[A] += cod * qm
        Pg[H] += qg
        mh = lh * E[i, 0]; ma = la * E[i, 1]
        eh = Y[i, 0] - mh + lh * XR[i, 0]; ea = Y[i, 1] - ma + la * XR[i, 1]
        # --- home goals: loads +o_H, -d_A, +g_H
        den = mh * (Poo[H] + Pdd[A] + Pg[H]) + phi
        k = eh / den; f = mh / den
        co = Poo[H]; cd = Pod[H]
        o[H] += co * k; d[H] += cd * k
        Poo[H] -= co * co * f; Pod[H] -= co * cd * f; Pdd[H] -= cd * cd * f
        co = -Pod[A]; cd = -Pdd[A]
        o[A] += co * k; d[A] += cd * k
        Poo[A] -= co * co * f; Pod[A] -= co * cd * f; Pdd[A] -= cd * cd * f
        g[H] += Pg[H] * k; Pg[H] -= Pg[H] * Pg[H] * f
        # --- away goals: loads +o_A, -d_H, -g_H
        den = ma * (Poo[A] + Pdd[H] + Pg[H]) + phi
        k = ea / den; f = ma / den
        co = Poo[A]; cd = Pod[A]
        o[A] += co * k; d[A] += cd * k
        Poo[A] -= co * co * f; Pod[A] -= co * cd * f; Pdd[A] -= cd * cd * f
        co = -Pod[H]; cd = -Pdd[H]
        o[H] += co * k; d[H] += cd * k
        Poo[H] -= co * co * f; Pod[H] -= co * cd * f; Pdd[H] -= cd * cd * f
        g[H] -= Pg[H] * k; Pg[H] -= Pg[H] * Pg[H] * f
        # --- league level / home advantage
        kM = max(kMu, 1.0 / (Ln[L] + 30)); kHH = max(kH, 1.0 / (Ln[L] + 30))
        rh = eh / lh; ra = ea / la
        mu[L] += kM * (rh + ra) / 2
        hfa[L] += kHH * (rh - ra) / 2
        Ln[L] += 1
        if kCov > 0 and covid[i]:
            bcov += kCov * (rh - ra) / 2
        if qd <= 0:
            tlast[H] = day[i]; tlast[A] = day[i]
        tn[H] += 1; tn[A] += 1
    return out


def enc(k, v):
    if k in LOGK: return math.log(max(v, 1e-9))
    if k in LGT or k.startswith('w'):
        v = min(0.9999, max(1e-4, v)); return math.log(v / (1 - v))
    if k in TANH: return math.atanh(max(-0.999, min(0.999, v)))
    return float(v)


def dec(k, x):
    if k in LOGK: return math.exp(min(x, 50))
    if k in LGT or k.startswith('w'): return 1 / (1 + math.exp(-x))
    if k in TANH: return math.tanh(x)
    return float(x)


# ───────────────────────── stat channels (v1 per-channel EKFs, own-likelihood tuned, walk-forward) ─────────────
CHDEF = dict(v1h.DEF)
CHDEF['co'] = dict(q=0.0006, P0=0.03, rho=0.8, sSeason=0.004, dPromoO=0.04, dPromoD=0.04, dNew=0.04, phi=1.6, kMu=0.004, kH=0.004, Pg0=0, qg=0, rhoG=0, mu0=math.log(5.1), hfa0=0.2)
CHDEF['soff'] = dict(q=0.0006, P0=0.03, rho=0.8, sSeason=0.004, dPromoO=0.03, dPromoD=0.03, dNew=0.03, phi=2.0, kMu=0.004, kH=0.004, Pg0=0, qg=0, rhoG=0, mu0=math.log(8.2), hfa0=0.2)


def stat_obs(d, c):
    v = d.st_sh - d.st_sot if c == 'soff' else getattr(d, 'st_' + c)
    return np.nan_to_num(np.maximum(v, 0))


CHKEYS = ['q', 'P0', 'rho']
CHCLIP = np.array([4.0, 1.6, 4.0])


def channel_preds(d, chans=('sot', 'sh', 'co', 'soff'), cache=True):
    """CH[c][yi] = (n,2) expected stat under channel params tuned on seasons < SEASONS[yi].
    Only (q, P0, rho) are tuned (own Poisson likelihood); phi is fixed and the variance is reset to P0 every season
    (sSeason = 1), which removes the flat P0/phi and sSeason directions of the v1 tuning."""
    fn = lib.D + 'cache_channels.pkl'
    if cache:
        try:
            return pickle.load(open(fn, 'rb'))
        except Exception:
            pass
    CH = {}; PAR = {}
    for c in chans:
        y = stat_obs(d, c); p = dict(CHDEF[c]); p['sSeason'] = 1.0
        CH[c] = np.zeros((len(SEASONS), d.n, 2), np.float32); PAR[c] = {}
        has = d.statsOk & d.post & np.isfinite(getattr(d, 'st_co' if c == 'co' else 'st_sh')[:, 0])
        x0 = v1h.enc(p, CHKEYS)
        for yi, Y in enumerate(SEASONS):
            before = (d.season < Y) & has
            if before.sum() >= 700:
                L = v1h.make_loss(d, 'pois', 2014, Y, y[:, 0], y[:, 1], has)
                f = lambda x: L(lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, y[:, 0], y[:, 1], has, v1h.pvec(v1h.dec(np.clip(x, x0 - CHCLIP, x0 + CHCLIP), CHKEYS, p)), False, d.nteams)) + 1e-6 * float(((x - x0) ** 2).sum())
                x, best = lib.nm(f, v1h.enc(p, CHKEYS), rounds=2, maxiter=150)
                p = v1h.dec(np.clip(x, x0 - CHCLIP, x0 + CHCLIP), CHKEYS, p)
            out = lib.ekf_v1(d.h, d.a, d.lg, d.season, d.post, y[:, 0], y[:, 1], has, v1h.pvec(p), False, d.nteams)
            CH[c][yi] = out[:, :2]; PAR[c][Y] = dict(p)
        print('channel', c, {Y: {k: round(PAR[c][Y][k], 5) for k in CHKEYS} for Y in (2016, 2019, 2022, 2026)}, flush=True)
    res = dict(CH=CH, PAR=PAR)
    pickle.dump(res, open(fn, 'wb'))
    return res


def rel_surprise(d, CH, c, yi):
    y = stat_obs(d, c)
    ok = (d.statsOk & d.post)[:, None] & np.isfinite(getattr(d, 'st_co' if c == 'co' else 'st_sh'))
    r = np.where(ok, y / np.maximum(CH[c][yi], 1e-6) - 1.0, 0.0)
    return r


# ───────────────────────── observation builder ─────────────────────────────────
GCATS = (('wRed', 'adv'), ('wPen', 'pen'), ('wOwn', 'own'), ('wLead', 'late_lead2'), ('wTrail', 'late_trail2'), ('wLead2', 'lead2'), ('wTrail2', 'trail2'))


def build_obs(d, hp, R):
    """hp: dict of observation hyper-parameters; R: dict channel -> relative surprise (n,2) for the current tuning season."""
    g = d.g_all.astype(float).copy()
    for key, cat in GCATS:
        w = hp.get(key, 1.0)
        if w < 0.99999:
            g = g - (1 - w) * getattr(d, 'g_' + cat)
    g = np.where(d.evOk[:, None], g, np.c_[d.hg, d.ag]).astype(float)
    ex = hp.get('expo', 0.0)
    E = np.ones((d.n, 2))
    if ex > 0:
        E = np.where(d.evOk[:, None], np.exp(ex * (0.40 * d.fadv - 0.45 * d.fadv[:, ::-1])), 1.0)
    X = np.zeros((d.n, 2))
    for c in ('sot', 'sh', 'co', 'soff'):
        ck = hp.get('c_' + c, 0.0)
        if ck != 0.0:
            X = X + ck * R[c]
    return g, E, X


class Spec:
    def __init__(self, name, free, fixed=None, covid=None, prior=1e-6, lo=2009):
        self.name = name; self.free = list(free); self.fixed = dict(fixed or {}); self.covid = covid; self.prior = prior; self.lo = lo

    def params(self, x):
        core = dict(DEFAULT); hp = {}
        for k, v in self.fixed.items():
            (core if k in core else hp)[k] = v
        for k, xv in zip(self.free, x):
            (core if k in core else hp)[k] = dec(k, xv)
        return core, hp

    def x0(self, start=None):
        core, hp = self.params([])
        allp = dict(core); allp.update(hp)
        if start: allp.update({k: v for k, v in start.items() if k in self.free})
        return np.array([enc(k, allp.get(k, 0.0)) for k in self.free])


_ZERO = {}


def run(d, spec, x, R):
    core, hp = spec.params(x)
    g, E, X = build_obs(d, hp, R)
    if d.n not in _ZERO: _ZERO[d.n] = np.zeros(d.n, np.bool_)
    cv = _ZERO[d.n] if spec.covid is None else spec.covid
    return ekf3(d.h, d.a, d.lg, d.season, d.post, d.day, g, E, X, cv, np.array([core[k] for k in CORE], float), d.nteams)


def goal_loss(d, lo, Y, objective='pois'):
    mA = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 0)
    mB = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 1)
    hg = d.hg.astype(float); ag = d.ag.astype(float)
    if objective != 'pois':
        kA = d.k[mA]; kB = d.k[mB]; j = 0 if objective == 'rps' else 1

        def f2(out):
            lh, la = out[:, 0], out[:, 1]
            if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(la)) and lh.min() > 0 and la.min() > 0):
                return 1e9
            r = lib.rps_ll(lib.pois_probs(lh[mA], la[mA]), kA)[j].mean() + 0.3 * lib.rps_ll(lib.pois_probs(lh[mB], la[mB]), kB)[j].mean()
            if objective == 'mix':      # outcome log-loss plus a small weight on the goal likelihood (keeps the totals honest)
                v = lib.pois_ll(hg, lh) + lib.pois_ll(ag, la)
                r += 0.25 * (-v[mA].mean() - 0.3 * v[mB].mean())
            return r if np.isfinite(r) else 1e9
        return f2

    def f(out):
        lh, la = out[:, 0], out[:, 1]
        if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(la)) and lh.min() > 0 and la.min() > 0):
            return 1e9
        v = lib.pois_ll(hg, lh) + lib.pois_ll(ag, la)
        r = -v[mA].mean() - 0.3 * v[mB].mean()
        return r if np.isfinite(r) else 1e9
    return f


def walk(d, spec, CHR, seasons=SEASONS, verbose=True, first_iter=(3, 500), next_iter=(2, 250), save=True, init=None, quiet=False, rw=0.0, steps=None, objective='pois'):
    """Walk-forward: for season Y tune spec.free on seasons [lo, Y), then one full pass. init: dict Y -> {param: value} warm starts."""
    t0 = time.time()
    CH = CHR['CH']
    x = spec.x0(init[seasons[0]] if init else None); xref = x.copy()
    FULL = np.zeros((len(seasons), d.n, NOUT), np.float32); PAR = {}
    for yi, Y in enumerate(seasons):
        R = {c: rel_surprise(d, CH, c, yi) for c in CH}
        L = goal_loss(d, spec.lo, Y, objective)
        xprev = x.copy()
        f = lambda z: L(run(d, spec, z, R)) + spec.prior * float(((z - xref) ** 2).sum()) + rw * float(((z - xprev) ** 2).sum())
        if init is not None:
            xs = spec.x0(init[Y])
            if f(xs) < f(x): x = xs
        if len(x):
            r, it = first_iter if yi == 0 else next_iter
            x, best = lib.nm(f, x, rounds=r, maxiter=it, steps=steps)
        else:
            best = f(x)
        out = run(d, spec, x, R)
        FULL[yi] = out
        core, hp = spec.params(x)
        allp = dict(core); allp.update(hp)
        PAR[Y] = dict(allp, _loss=best)
        if verbose:
            print(f'  {spec.name} {Y} loss {best:.5f} ' + str({k: round(dec(k, xv), 5) for k, xv in zip(spec.free, x)}), flush=True)
    res = dict(name=spec.name, seasons=list(seasons), FULL=FULL, PAR=PAR, free=spec.free)
    lam = np.zeros((d.n, 2))
    for yi, Y in enumerate(seasons):
        m = d.season == Y
        lam[m] = FULL[yi][m, :2]
    res['lam'] = lam
    sc = d.scored
    res['dev'] = lib.dev_report(spec.name + ' [indep Poisson]', d, lib.pois_probs(lam[sc, 0], lam[sc, 1]), verbose=not quiet)
    res['loss2021'] = PAR[2021]['_loss'] if 2021 in PAR else None
    if verbose: print(f'  ({time.time() - t0:.0f}s)', flush=True)
    if save:
        pickle.dump(res, open(lib.D + f'out_{spec.name}.pkl', 'wb'))
    return res
