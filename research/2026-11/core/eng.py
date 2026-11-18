"""Engine v2: generalised diagonal EKF for the goal channel with optional extras.

log lam_h = mu + hfa/2 + g_h + ga_a + turf + o_h - d_a          (times exposure E_h)
log lam_a = mu - hfa/2 - g_h - ga_a - turf + o_a - d_h          (times exposure E_a)

Optional secondary ("factor") channels c (e.g. shots on target):
log m_h = mu_c + hfa_c/2 + beta_c (o_h - d_a) + u_h - v_a
so a shots surprise updates the COMMON strength (o, d) and the channel-specific deviation (u, v).

All switches are driven by the parameter vector so that every variant is tuned with the same code.
"""
import math
import numpy as np
from numba import njit

# core parameter names (order matters)
CORE = ['qm', 'qd', 'P0', 'rho', 'sS', 'dPO', 'dPD', 'dNew', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG',
        'kappa', 'Ppromo', 'cap', 'Pa0', 'qa', 'rhoA', 'shG', 'shT', 'mu0', 'hfa0', 'kTurf', 'kCov', 'rhoB', 'dRel']
# per-secondary-channel parameter names
SEC = ['beta', 'P0', 'q', 'rho', 'sS', 'phi', 'mu0', 'hfa0', 'kMu', 'kH']
NOUT = 22

DEFAULT = dict(qm=1e-4, qd=0.0, P0=0.04, rho=0.85, sS=0.02, dPO=0.25, dPD=0.2, dNew=0.0, phi=2.0, kMu=0.002, kH=0.0005,
               Pg0=0.02, qg=0.002, rhoG=0.2, kappa=1.0, Ppromo=0.0, cap=1.0, Pa0=0.0, qa=0.0, rhoA=0.0, shG=0.0, shT=2.0,
               mu0=math.log(1.25), hfa0=0.35, kTurf=0.0, kCov=0.0, rhoB=-1.0, dRel=-1.0)
LOGK = {'qm', 'qd', 'P0', 'sS', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'Ppromo', 'Pa0', 'qa', 'shG', 'kTurf', 'kCov', 'q'}
LGT = {'rho', 'rhoG', 'rhoA', 'rhoB'}


@njit(cache=True, error_model='numpy')
def ekf2(h, a, lg, season, post, day, Y, E, hasobs, SO, hasS, turf, covid, p, ps, nteams):
    (qm, qd, P0, rho, sS, dPO, dPD, dNew, phi, kMu, kH, Pg0, qg, rhoG, kappa, Ppromo, cap, Pa0, qa, rhoA,
     shG, shT, mu0, hfa0, kTurf, kCov, rhoB, dRel) = p
    n = len(h)
    nc = ps.shape[0]
    if rhoB < 0: rhoB = rho          # separate between-season persistence for teams whose previous season was in B
    if dRel < 0: dRel = 1.0          # multiplier on the offsets for relegated teams (A -> B)
    o = np.zeros(nteams); d = np.zeros(nteams); g = np.zeros(nteams); ga = np.zeros(nteams)
    Po = np.zeros(nteams); Pd = np.zeros(nteams); Pg = np.zeros(nteams); Pa = np.zeros(nteams)
    u = np.zeros((nteams, max(nc, 1))); v = np.zeros((nteams, max(nc, 1)))
    Pu = np.zeros((nteams, max(nc, 1))); Pv = np.zeros((nteams, max(nc, 1)))
    cus = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams, np.int64); tlast = np.zeros(nteams)
    mu = np.array([mu0, mu0]); hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    muc = np.zeros((2, max(nc, 1))); hfac = np.zeros((2, max(nc, 1))); Lnc = np.zeros((2, max(nc, 1)))
    for c in range(nc):
        muc[0, c] = ps[c, 6]; muc[1, c] = ps[c, 6]; hfac[0, c] = ps[c, 7]; hfac[1, c] = ps[c, 7]
    bturf = 0.0; Pturf = 0.01; bcov = 0.0; Pcov = 0.02
    out = np.zeros((n, NOUT))
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                fresh = (L == 0 and se <= 2007)
                off = 0.0 if fresh else -dNew
                o[t] = off; d[t] = off; Po[t] = P0; Pd[t] = P0; g[t] = 0.0; Pg[t] = Pg0; ga[t] = 0.0; Pa[t] = Pa0
                if L == 0 and se > 2007:
                    o[t] = -dPO; d[t] = -dPD
                for c in range(nc):
                    u[t, c] = 0.0; v[t, c] = 0.0; Pu[t, c] = ps[c, 1]; Pv[t, c] = ps[c, 1]
                tse[t] = se; tlg[t] = L; tn[t] = 0; tlast[t] = day[i]
            elif se > tse[t]:
                gap = se - tse[t]
                r = (rho if tlg[t] == 0 else rhoB) ** gap
                o[t] *= r; d[t] *= r
                Po[t] = r * r * Po[t] + sS * gap; Pd[t] = r * r * Pd[t] + sS * gap
                if cap > 0.5:
                    Po[t] = min(P0, Po[t]); Pd[t] = min(P0, Pd[t])
                rg = rhoG ** gap
                g[t] *= rg; Pg[t] = min(Pg0, rg * rg * Pg[t] + qg * 20 * gap)
                if Pa0 > 0:
                    rgA = rhoA ** gap
                    ga[t] *= rgA; Pa[t] = min(Pa0, rgA * rgA * Pa[t] + qa * 20 * gap)
                for c in range(nc):
                    rc = ps[c, 3] ** gap
                    u[t, c] *= rc; v[t, c] *= rc
                    Pu[t, c] = min(ps[c, 1], rc * rc * Pu[t, c] + ps[c, 4] * gap)
                    Pv[t, c] = min(ps[c, 1], rc * rc * Pv[t, c] + ps[c, 4] * gap)
                if gap > 1:
                    o[t] = min(o[t], -dNew); d[t] = min(d[t], -dNew); Po[t] = P0; Pd[t] = P0
                elif L != tlg[t]:
                    if L == 0:      # promoted
                        o[t] = kappa * o[t] - dPO; d[t] = kappa * d[t] - dPD
                    else:           # relegated
                        o[t] += dPO * dRel; d[t] += dPD * dRel
                    if cap > 0.5:
                        Po[t] = min(P0, Po[t] + sS); Pd[t] = min(P0, Pd[t] + sS)
                    if L == 0:
                        Po[t] += Ppromo; Pd[t] += Ppromo
                tse[t] = se; tlg[t] = L; tn[t] = 0; cus[t] = 0.0
        H = h[i]; A = a[i]
        dtH = day[i] - tlast[H]; dtA = day[i] - tlast[A]
        # time-based process noise is added BEFORE predicting (uncertainty grows over breaks)
        if qd > 0:
            addH = qd * min(max(dtH, 0.0), 400.0); addA = qd * min(max(dtA, 0.0), 400.0)
            Po[H] += addH; Pd[H] += addH; Po[A] += addA; Pd[A] += addA
            tlast[H] = day[i]; tlast[A] = day[i]   # so repeated predictions do not double count
        hx = g[H] + ga[A] + (bturf if turf[i] else 0.0) + (bcov if covid[i] else 0.0)
        etaH = mu[L] + hfa[L] / 2 + hx + o[H] - d[A]
        etaA = mu[L] - hfa[L] / 2 - hx + o[A] - d[H]
        lh = math.exp(etaH); la = math.exp(etaA)
        Pex = Pg[H] + Pa[A]
        out[i, 0] = lh; out[i, 1] = la
        out[i, 2] = Po[H] + Pd[A] + Pex; out[i, 3] = Po[A] + Pd[H] + Pex
        out[i, 4] = mu[L]; out[i, 5] = hfa[L]
        out[i, 6] = o[H]; out[i, 7] = d[H]; out[i, 8] = o[A]; out[i, 9] = d[A]; out[i, 10] = g[H]; out[i, 11] = ga[A]
        out[i, 12] = tn[H]; out[i, 13] = tn[A]; out[i, 14] = dtH; out[i, 15] = dtA
        out[i, 16] = Po[H]; out[i, 17] = Pd[H]; out[i, 18] = Po[A]; out[i, 19] = Pd[A]
        if nc > 0:
            out[i, 20] = u[H, 0] - v[A, 0]; out[i, 21] = u[A, 0] - v[H, 0]
        if not post[i]:
            continue
        if hasobs[i]:
            Po[H] += qm; Pd[H] += qm; Po[A] += qm; Pd[A] += qm
            Pg[H] += qg
            if Pa0 > 0: Pa[A] += qa
            mh = lh * E[i, 0]; ma = la * E[i, 1]
            eh = Y[i, 0] - mh; ea = Y[i, 1] - ma
            # --- home goals
            Sx = Po[H] + Pd[A] + Pg[H] + Pa[A]
            den = mh * Sx + phi
            o[H] += Po[H] * eh / den; d[A] -= Pd[A] * eh / den
            g[H] += Pg[H] * eh / den; Pg[H] -= Pg[H] * Pg[H] * mh / den
            if Pa0 > 0:
                ga[A] += Pa[A] * eh / den; Pa[A] -= Pa[A] * Pa[A] * mh / den
            Po[H] -= Po[H] * Po[H] * mh / den; Pd[A] -= Pd[A] * Pd[A] * mh / den
            # --- away goals
            Sx = Po[A] + Pd[H] + Pg[H] + Pa[A]
            den = ma * Sx + phi
            o[A] += Po[A] * ea / den; d[H] -= Pd[H] * ea / den
            g[H] -= Pg[H] * ea / den; Pg[H] -= Pg[H] * Pg[H] * ma / den
            if Pa0 > 0:
                ga[A] -= Pa[A] * ea / den; Pa[A] -= Pa[A] * Pa[A] * ma / den
            Po[A] -= Po[A] * Po[A] * ma / den; Pd[H] -= Pd[H] * Pd[H] * ma / den
            # --- league level / home advantage
            kM = max(kMu, 1.0 / (Ln[L] + 30)); kHH = max(kH, 1.0 / (Ln[L] + 30))
            mu[L] += kM * (eh / mh + ea / ma) / 2
            hfa[L] += kHH * (eh / mh - ea / ma) / 2
            Ln[L] += 1
            if kTurf > 0 and turf[i]:
                bturf += kTurf * (eh / mh - ea / ma) / 2
            if kCov > 0 and covid[i]:
                bcov += kCov * (eh / mh - ea / ma) / 2
            # --- squad-shock detector (CUSUM of standardised net surprise)
            if shG > 0:
                z = eh / math.sqrt(mh) - ea / math.sqrt(ma)
                cus[H] = 0.8 * cus[H] + z; cus[A] = 0.8 * cus[A] - z
                exH = abs(cus[H]) - shT; exA = abs(cus[A]) - shT
                if exH > 0:
                    Po[H] += shG * exH * exH; Pd[H] += shG * exH * exH
                if exA > 0:
                    Po[A] += shG * exA * exA; Pd[A] += shG * exA * exA
        if hasS[i]:
            for c in range(nc):
                beta = ps[c, 0]; qc = ps[c, 2]; phc = ps[c, 5]
                Pu[H, c] += qc; Pv[H, c] += qc; Pu[A, c] += qc; Pv[A, c] += qc
                # innovations use the PRE-match common strength (etaH/etaA components) -> no double counting of goals
                sh_ = math.exp(muc[L, c] + hfac[L, c] / 2 + beta * (out[i, 6] - out[i, 9]) + u[H, c] - v[A, c])
                sa_ = math.exp(muc[L, c] - hfac[L, c] / 2 + beta * (out[i, 8] - out[i, 7]) + u[A, c] - v[H, c])
                e1 = SO[i, 0, c] - sh_; e2 = SO[i, 1, c] - sa_
                Sx = beta * beta * (Po[H] + Pd[A]) + Pu[H, c] + Pv[A, c]
                den = sh_ * Sx + phc
                o[H] += beta * Po[H] * e1 / den; d[A] -= beta * Pd[A] * e1 / den
                u[H, c] += Pu[H, c] * e1 / den; v[A, c] -= Pv[A, c] * e1 / den
                Po[H] -= beta * beta * Po[H] * Po[H] * sh_ / den; Pd[A] -= beta * beta * Pd[A] * Pd[A] * sh_ / den
                Pu[H, c] -= Pu[H, c] * Pu[H, c] * sh_ / den; Pv[A, c] -= Pv[A, c] * Pv[A, c] * sh_ / den
                Sx = beta * beta * (Po[A] + Pd[H]) + Pu[A, c] + Pv[H, c]
                den = sa_ * Sx + phc
                o[A] += beta * Po[A] * e2 / den; d[H] -= beta * Pd[H] * e2 / den
                u[A, c] += Pu[A, c] * e2 / den; v[H, c] -= Pv[H, c] * e2 / den
                Po[A] -= beta * beta * Po[A] * Po[A] * sa_ / den; Pd[H] -= beta * beta * Pd[H] * Pd[H] * sa_ / den
                Pu[A, c] -= Pu[A, c] * Pu[A, c] * sa_ / den; Pv[H, c] -= Pv[H, c] * Pv[H, c] * sa_ / den
                kM = max(ps[c, 8], 1.0 / (Lnc[L, c] + 30)); kHH = max(ps[c, 9], 1.0 / (Lnc[L, c] + 30))
                muc[L, c] += kM * (e1 / sh_ + e2 / sa_) / 2
                hfac[L, c] += kHH * (e1 / sh_ - e2 / sa_) / 2
                Lnc[L, c] += 1
        if qd <= 0:
            tlast[H] = day[i]; tlast[A] = day[i]
        tn[H] += 1; tn[A] += 1
    return out


def pvec(p):
    return np.array([p[k] for k in CORE], float)


def psvec(plist):
    if not plist:
        return np.zeros((0, len(SEC)))
    return np.array([[q[k] for k in SEC] for q in plist], float)


def enc1(k, v):
    k = k.split(':')[-1]
    if k in LOGK: return math.log(max(v, 1e-9))
    if k in LGT:
        v = min(0.9999, max(1e-4, v)); return math.log(v / (1 - v))
    return v


def dec1(k, x):
    k = k.split(':')[-1]
    if k in LOGK: return math.exp(min(x, 50))
    if k in LGT: return 1 / (1 + math.exp(-x))
    return float(x)
