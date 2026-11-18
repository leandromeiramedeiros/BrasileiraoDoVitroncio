"""Alternative dynamic strength estimators (question 2).

gas_pois   : score-driven (GAS) bivariate Poisson with constant gains (Koopman & Lit 2015 style)
gas_ord    : score-driven ordered logit on a rating difference (3 = W/D/L or 5 = clipped goal difference categories)
ekf_full   : EKF with a FULL covariance matrix over (league level, league home advantage, attack, defence, home effect)
             and time-based (per day) process noise
"""
import math
import numpy as np
from numba import njit


@njit(cache=True, error_model='numpy')
def gas_pois(h, a, lg, season, post, hg, ag, p, nteams):
    Ao, Ad, B, rho, dPO, dPD, kMu, kH, Ag, rhoG, mu0, hfa0, boost, nb = p
    n = len(h)
    o = np.zeros(nteams); d = np.zeros(nteams); g = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams)
    mu = np.array([mu0, mu0]); hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    out = np.zeros((n, 2))
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                if L == 0 and se > 2007:
                    o[t] = -dPO; d[t] = -dPD
                tse[t] = se; tlg[t] = L
            elif se > tse[t]:
                gap = se - tse[t]
                r = rho ** gap
                o[t] *= r; d[t] *= r; g[t] *= rhoG ** gap
                if gap > 1:
                    o[t] = min(o[t], 0.0); d[t] = min(d[t], 0.0)
                elif L != tlg[t]:
                    sg = -1.0 if L == 0 else 1.0
                    o[t] += sg * dPO; d[t] += sg * dPD
                tse[t] = se; tlg[t] = L; tn[t] = 0
        H = h[i]; A = a[i]
        lh = math.exp(mu[L] + hfa[L] / 2 + g[H] + o[H] - d[A])
        la = math.exp(mu[L] - hfa[L] / 2 - g[H] + o[A] - d[H])
        out[i, 0] = lh; out[i, 1] = la
        if not post[i]:
            continue
        eh = hg[i] - lh; ea = ag[i] - la
        kh = 1.0 + boost * math.exp(-tn[H] / nb); ka = 1.0 + boost * math.exp(-tn[A] / nb)
        o[H] = B * o[H] + Ao * kh * eh; d[A] = B * d[A] - Ad * ka * eh
        o[A] = B * o[A] + Ao * ka * ea; d[H] = B * d[H] - Ad * kh * ea
        g[H] += Ag * (eh - ea)
        kM = max(kMu, 1.0 / (Ln[L] + 30)); kHH = max(kH, 1.0 / (Ln[L] + 30))
        mu[L] += kM * (eh / lh + ea / la) / 2
        hfa[L] += kHH * (eh / lh - ea / la) / 2
        Ln[L] += 1; tn[H] += 1; tn[A] += 1
    return out


@njit(cache=True)
def _F(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit(cache=True, error_model='numpy')
def gas_ord(h, a, lg, season, post, hg, ag, p, nteams, ncat, xr):
    """Ordered logit on eta = r_h - r_a + hfa_L + g_h.  ncat=3: A/D/H;  ncat=5: gd<=-2,-1,0,1,>=2.
    p = K, rho, dP, kH, c1, c2, boost, nb, hfa0, Kg, rhoG, kC"""
    K, rho, dP, kH, c1, c2, boost, nb, hfa0, Kg, rhoG, kC = p
    n = len(h)
    r = np.zeros(nteams); g = np.zeros(nteams)
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tn = np.zeros(nteams)
    hfa = np.array([hfa0, hfa0]); Ln = np.zeros(2)
    cc = np.array([c1, c1])      # league-specific draw half-width (drifts with gain kC)
    out = np.zeros((n, 4))
    th = np.zeros(6); cum = np.zeros(6); dens = np.zeros(6)
    for i in range(n):
        L = lg[i]; se = season[i]
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            if tse[t] < 0:
                if L == 0 and se > 2007:
                    r[t] = -dP
                tse[t] = se; tlg[t] = L
            elif se > tse[t]:
                gap = se - tse[t]
                r[t] *= rho ** gap; g[t] *= rhoG ** gap
                if gap > 1:
                    r[t] = min(r[t], 0.0)
                elif L != tlg[t]:
                    r[t] += -dP if L == 0 else dP
                tse[t] = se; tlg[t] = L; tn[t] = 0
        H = h[i]; A = a[i]
        eta = r[H] - r[A] + hfa[L] + g[H]
        c = cc[L]
        pA = _F(-c - eta); pH = 1.0 - _F(c - eta); pD = 1.0 - pA - pH
        out[i, 0] = pH; out[i, 1] = pD; out[i, 2] = pA; out[i, 3] = eta
        if not post[i]:
            continue
        gd = hg[i] - ag[i]
        if ncat == 3:
            nth = 2; th[0] = -c; th[1] = c
            y = 0 if gd < 0 else 1 if gd == 0 else 2
        else:
            nth = 4; th[0] = -c - c2; th[1] = -c; th[2] = c; th[3] = c + c2
            y = 0 if gd <= -2 else 1 if gd == -1 else 2 if gd == 0 else 3 if gd == 1 else 4
        # cumulative probs P(Y<=j) = F(th_j - eta)
        lo = 0.0 if y == 0 else _F(th[y - 1] - eta)
        hi = 1.0 if y == nth else _F(th[y] - eta)
        flo = 0.0 if y == 0 else lo * (1 - lo)
        fhi = 0.0 if y == nth else hi * (1 - hi)
        py = max(hi - lo, 1e-12)
        s = -(fhi - flo) / py                  # d log p / d eta
        s_out = s
        s = s + xr[i]                          # optional shots-surprise augmentation of the score
        kh = 1.0 + boost * math.exp(-tn[H] / nb); ka = 1.0 + boost * math.exp(-tn[A] / nb)
        r[H] += K * kh * s; r[A] -= K * ka * s
        g[H] += Kg * s
        kHH = max(kH, 1.0 / (Ln[L] + 30))
        hfa[L] += kHH * s_out
        if kC > 0:
            # d log p / d c for the 3-way outcome (draw wider <-> more draws)
            a_ = _F(-c - eta); b_ = _F(c - eta)
            if gd == 0: sc = (b_ * (1 - b_) + a_ * (1 - a_)) / max(b_ - a_, 1e-12)
            elif gd > 0: sc = -b_          # d log(1-F(c-eta))/dc = -f/(1-F) = -F
            else: sc = -(1 - a_)           # d log F(-c-eta)/dc = -(1-F)
            cc[L] += kC * sc
        Ln[L] += 1; tn[H] += 1; tn[A] += 1
    return out


@njit(cache=True, error_model='numpy')
def ekf_full(h, a, lg, season, post, day, Y, p, nteams, blockdiag=False):
    """Full-covariance EKF.  state: [mu_A, mu_B, hfa_A, hfa_B, (o, d, g) per team].
    p = qm, qd, P0, rho, sS, dPO, dPD, phi, qmu, qh, Pg0, qg, rhoG, sG, mu0, hfa0, Ppromo, kappa, cod
    cod = prior correlation between a team's attack and defence (applied to P0, sS and the process noise).
    blockdiag=True discards all cross-team covariances (keeps the 3x3 block of each team)."""
    qm, qd, P0, rho, sS, dPO, dPD, phi, qmu, qh, Pg0, qg, rhoG, sG, mu0, hfa0, Ppromo, kappa, cod = p
    n = len(h)
    ns = 4 + 3 * nteams
    x = np.zeros(ns); P = np.zeros((ns, ns))
    x[0] = mu0; x[1] = mu0; x[2] = hfa0; x[3] = hfa0
    P[0, 0] = 0.02; P[1, 1] = 0.02; P[2, 2] = 0.02; P[3, 3] = 0.02
    tse = np.full(nteams, -1); tlg = np.full(nteams, -1); tlast = np.zeros(nteams)
    act = np.zeros(ns, np.int64); nact = 4
    act[0] = 0; act[1] = 1; act[2] = 2; act[3] = 3
    isact = np.zeros(nteams, np.bool_)
    out = np.zeros((n, 10))
    v1 = np.zeros(ns); v2 = np.zeros(ns)
    cur_season = -1
    for i in range(n):
        L = lg[i]; se = season[i]
        if se != cur_season:
            # new season: rebuild the active set lazily (teams join as they appear); drop teams idle for a full season
            cur_season = se
            nact = 4
            for t in range(nteams):
                if tse[t] >= se - 1:
                    isact[t] = True
                    act[nact] = 4 + 3 * t; act[nact + 1] = 5 + 3 * t; act[nact + 2] = 6 + 3 * t; nact += 3
                else:
                    isact[t] = False
        for side in range(2):
            t = h[i] if side == 0 else a[i]
            io = 4 + 3 * t; idd = io + 1; ig = io + 2
            if tse[t] < 0 or se - tse[t] > 1:
                # brand new (or back after >1 season away): fresh prior, no correlations
                for k in range(ns):
                    P[io, k] = 0; P[k, io] = 0; P[idd, k] = 0; P[k, idd] = 0; P[ig, k] = 0; P[k, ig] = 0
                P[io, io] = P0; P[idd, idd] = P0; P[ig, ig] = Pg0
                P[io, idd] = cod * P0; P[idd, io] = cod * P0
                x[io] = 0.0; x[idd] = 0.0; x[ig] = 0.0
                if L == 0 and se > 2007:
                    x[io] = -dPO; x[idd] = -dPD
                if not isact[t]:
                    isact[t] = True
                    act[nact] = io; act[nact + 1] = idd; act[nact + 2] = ig; nact += 3
                tse[t] = se; tlg[t] = L; tlast[t] = day[i]
            elif se > tse[t]:
                for k in range(ns):
                    P[io, k] *= rho; P[k, io] *= rho; P[idd, k] *= rho; P[k, idd] *= rho; P[ig, k] *= rhoG; P[k, ig] *= rhoG
                x[io] *= rho; x[idd] *= rho; x[ig] *= rhoG
                P[io, io] += sS; P[idd, idd] += sS; P[ig, ig] += sG
                P[io, idd] += cod * sS; P[idd, io] += cod * sS
                if L != tlg[t]:
                    if L == 0:
                        x[io] = kappa * x[io] - dPO; x[idd] = kappa * x[idd] - dPD
                        P[io, io] += Ppromo; P[idd, idd] += Ppromo
                    else:
                        x[io] += dPO; x[idd] += dPD
                tse[t] = se; tlg[t] = L
        H = h[i]; A = a[i]
        oH = 4 + 3 * H; dH = oH + 1; gH = oH + 2; oA = 4 + 3 * A; dA = oA + 1
        iM = L; iF = 2 + L
        # time-based process noise (added before predicting)
        dtH = min(max(day[i] - tlast[H], 0.0), 400.0); dtA = min(max(day[i] - tlast[A], 0.0), 400.0)
        P[oH, oH] += qd * dtH; P[dH, dH] += qd * dtH; P[oA, oA] += qd * dtA; P[dA, dA] += qd * dtA
        P[oH, dH] += cod * qd * dtH; P[dH, oH] += cod * qd * dtH; P[oA, dA] += cod * qd * dtA; P[dA, oA] += cod * qd * dtA
        tlast[H] = day[i]; tlast[A] = day[i]
        etaH = x[iM] + x[iF] / 2 + x[gH] + x[oH] - x[dA]
        etaA = x[iM] - x[iF] / 2 - x[gH] + x[oA] - x[dH]
        lh = math.exp(etaH); la = math.exp(etaA)
        # v1 = P h1, v2 = P h2 over active states
        for kk in range(nact):
            k = act[kk]
            v1[k] = P[k, iM] + 0.5 * P[k, iF] + P[k, gH] + P[k, oH] - P[k, dA]
            v2[k] = P[k, iM] - 0.5 * P[k, iF] - P[k, gH] + P[k, oA] - P[k, dH]
        S11 = v1[iM] + 0.5 * v1[iF] + v1[gH] + v1[oH] - v1[dA]
        S22 = v2[iM] - 0.5 * v2[iF] - v2[gH] + v2[oA] - v2[dH]
        S12 = v2[iM] + 0.5 * v2[iF] + v2[gH] + v2[oH] - v2[dA]
        out[i, 0] = lh; out[i, 1] = la; out[i, 2] = S11; out[i, 3] = S22; out[i, 4] = S12
        if not post[i]:
            continue
        # per-match noise + update
        P[oH, oH] += qm; P[dH, dH] += qm; P[oA, oA] += qm; P[dA, dA] += qm; P[gH, gH] += qg
        P[oH, dH] += cod * qm; P[dH, oH] += cod * qm; P[oA, dA] += cod * qm; P[dA, oA] += cod * qm
        P[iM, iM] += qmu; P[iF, iF] += qh
        for kk in range(nact):
            k = act[kk]
            v1[k] = P[k, iM] + 0.5 * P[k, iF] + P[k, gH] + P[k, oH] - P[k, dA]
            v2[k] = P[k, iM] - 0.5 * P[k, iF] - P[k, gH] + P[k, oA] - P[k, dH]
        S11 = v1[iM] + 0.5 * v1[iF] + v1[gH] + v1[oH] - v1[dA] + phi / lh
        S22 = v2[iM] - 0.5 * v2[iF] - v2[gH] + v2[oA] - v2[dH] + phi / la
        S12 = v2[iM] + 0.5 * v2[iF] + v2[gH] + v2[oH] - v2[dA]
        det = S11 * S22 - S12 * S12
        i11 = S22 / det; i22 = S11 / det; i12 = -S12 / det
        z1 = (Y[i, 0] - lh) / lh; z2 = (Y[i, 1] - la) / la
        w1 = i11 * z1 + i12 * z2; w2 = i12 * z1 + i22 * z2
        for kk in range(nact):
            k = act[kk]
            x[k] += v1[k] * w1 + v2[k] * w2
        for kk in range(nact):
            k = act[kk]
            a1 = v1[k] * i11 + v2[k] * i12; a2 = v1[k] * i12 + v2[k] * i22
            if blockdiag:
                if k < 4:
                    for j in range(4):
                        P[k, j] -= a1 * v1[j] + a2 * v2[j]
                else:
                    b0 = 4 + 3 * ((k - 4) // 3)
                    for j in range(b0, b0 + 3):
                        P[k, j] -= a1 * v1[j] + a2 * v2[j]
            else:
                for jj in range(nact):
                    j = act[jj]
                    P[k, j] -= a1 * v1[j] + a2 * v2[j]
        out[i, 5] = x[oH]; out[i, 6] = x[dH]; out[i, 7] = x[oA]; out[i, 8] = x[dA]; out[i, 9] = x[gH]
    return out
