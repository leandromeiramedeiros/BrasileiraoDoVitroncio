#!/usr/bin/env python
"""Python re-implementation of the v1 goal-channel rating (extended Kalman filter on a bivariate Poisson state-space model,
site/js/engine/ratings.js) with a pluggable observation, plus helpers (production Markov chain via an interpolation grid,
RPS scoring identical to shared/score.py).

`python engine.py` checks that the re-implementation reproduces g_lh / g_la of shared/baseline_features.csv.
"""
import json, os, math
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'work')
SHARED = '/home/user/bdv-lab/shared/'
MODEL = json.load(open(os.path.join(HERE, 'v1_snapshot.json')))   # frozen v1 parameters (the live model.json may be refitted at any time)
PG = dict(MODEL['params']['g'])          # v1 hyper-parameters of the goal channel
XGP = MODEL['xg']                        # v1 shot-count proxy  b0 + b1*sot + b2*(sh-sot)
STACK = MODEL['stack']


class Data:
    def __init__(self):
        M = json.load(open(SHARED + 'matches.json'))
        self.M = [m for m in M if m['status'] != 'off']
        n = len(self.M)
        tids = sorted({m['h'] for m in self.M} | {m['a'] for m in self.M})
        self.tix = {t: i for i, t in enumerate(tids)}; self.tids = tids
        self.id = np.array([m['id'] for m in self.M])
        self.h = np.array([self.tix[m['h']] for m in self.M]); self.a = np.array([self.tix[m['a']] for m in self.M])
        self.lg = np.array([0 if m['lg'] == 'A' else 1 for m in self.M]); self.season = np.array([m['season'] for m in self.M])
        self.post = np.array([m['status'] == 'post' for m in self.M])
        self.hg = np.array([m['hg'] if m['status'] == 'post' else np.nan for m in self.M], float)
        self.ag = np.array([m['ag'] if m['status'] == 'post' else np.nan for m in self.M], float)
        self.date = np.array([m['date'] for m in self.M])
        # v1 observation pieces: red-card adjusted goals (weight wRed applied later), shot proxy
        g1 = np.zeros((n, 2)); gR = np.zeros((n, 2)); prox = np.full((n, 2), np.nan)
        for i, m in enumerate(self.M):
            if m['status'] != 'post': continue
            if m['evOk'] and m['ev']:
                rh = ra = 0
                for e in m['ev']:
                    t = e['t']
                    if t == 'R':
                        if e['s'] == 'h': rh += 1
                        else: ra += 1
                    elif t in ('G', 'P', 'O'):
                        if e['s'] == 'h':
                            if ra > rh: gR[i, 0] += 1
                            else: g1[i, 0] += 1
                        else:
                            if rh > ra: gR[i, 1] += 1
                            else: g1[i, 1] += 1
            else:
                g1[i, 0] = m['hg']; g1[i, 1] = m['ag']
            if m['statsOk']:
                s = m['stats']
                for j, sd in enumerate('ha'):
                    prox[i, j] = max(0.05, XGP['b0'] + XGP['b1'] * s[sd]['sot'] + XGP['b2'] * max(0, s[sd]['sh'] - s[sd]['sot']))
        self.g1, self.gR, self.prox = g1, gR, prox
        B = pd.read_csv(SHARED + 'baseline_features.csv', dtype={'id': str}).set_index('id').loc[self.id].reset_index()
        self.B = B
        self.scored = ((self.lg == 0) & (self.season >= 2013) & self.post)

    def gadj(self, wRed):
        return self.g1 + wRed * self.gR

    def v1_obs(self, w=None, wRed=None):
        w = PG['w'] if w is None else w; wRed = PG['wRed'] if wRed is None else wRed
        g = self.gadj(wRed)
        return np.where(np.isnan(self.prox), g, w * g + (1 - w) * self.prox)


def run_filter(D, y, p=PG, return_state=False, phi_arr=None):
    """EKF goal-type channel. y[n,2] = observation (home, away) for finished matches. Returns pre-match (lh, la, Sh, Sa, mu, hfa, ...).
    phi_arr: optional per-match observation-noise parameter (default p['phi'] for every match)."""
    n = len(D.M); T = len(D.tids)
    o = [0.0] * T; d = [0.0] * T; g = [0.0] * T; Po = [0.0] * T; Pd = [0.0] * T; Pg = [0.0] * T
    tl = [-1] * T; ts = [0] * T
    mu = [p['mu0'], p['mu0']]; hfa = [p['hfa0'], p['hfa0']]; Ln = [0, 0]
    q, P0, rho, sS, dPO, dPD, dNew, phi = p['q'], p['P0'], p['rho'], p['sSeason'], p['dPromoO'], p['dPromoD'], p['dNew'], p['phi']
    kMu, kH, Pg0, qg, rhoG = p['kMu'], p['kH'], p['Pg0'], p['qg'], p['rhoG']
    out = np.empty((n, 11))   # lh, la, Sh, Sa, mu, hfa, o_h, d_h, o_a, d_a, homefactor_h
    H, A, LG, SE, POST = D.h.tolist(), D.a.tolist(), D.lg.tolist(), D.season.tolist(), D.post.tolist()
    yl = y.tolist()
    phil = None if phi_arr is None else np.asarray(phi_arr, float).tolist()
    exp = math.exp
    for i in range(n):
        lg, se = LG[i], SE[i]
        for t in (H[i], A[i]):
            if tl[t] < 0:
                fresh = (lg == 0 and se <= 2007)
                off = 0.0 if fresh else -dNew
                o[t] = off; d[t] = off; Po[t] = P0; Pd[t] = P0; g[t] = 0.0; Pg[t] = Pg0
                if lg == 0 and se > 2007:
                    o[t] -= dPO - dNew; d[t] -= dPD - dNew
                tl[t] = lg; ts[t] = se
            elif se > ts[t]:
                gap = se - ts[t]; r = rho ** gap
                o[t] *= r; d[t] *= r
                Po[t] = min(P0, r * r * Po[t] + sS * gap); Pd[t] = min(P0, r * r * Pd[t] + sS * gap)
                rg = rhoG ** gap; g[t] *= rg; Pg[t] = min(Pg0, rg * rg * Pg[t] + qg * 20 * gap)
                if gap > 1:
                    o[t] = min(o[t], -dNew); d[t] = min(d[t], -dNew); Po[t] = P0; Pd[t] = P0
                elif lg != tl[t]:
                    sign = -1.0 if lg == 0 else 1.0
                    o[t] += sign * dPO; d[t] += sign * dPD
                    Po[t] = min(P0, Po[t] + sS); Pd[t] = min(P0, Pd[t] + sS)
                ts[t] = se; tl[t] = lg
        h, a = H[i], A[i]
        m_, hf = mu[lg], hfa[lg]
        row = out[i]
        row[0] = exp(m_ + hf / 2 + g[h] + o[h] - d[a]); row[1] = exp(m_ - hf / 2 - g[h] + o[a] - d[h])
        row[2] = Po[h] + Pd[a] + Pg[h]; row[3] = Po[a] + Pd[h] + Pg[h]; row[4] = m_; row[5] = hf
        row[6] = o[h]; row[7] = d[h]; row[8] = o[a]; row[9] = d[a]; row[10] = g[h]
        if not POST[i]: continue
        yh, ya = yl[i]
        if yh != yh: continue          # no observation for this channel (v1: `if (!y) continue`)
        if phil is not None: phi = phil[i]
        Po[h] += q; Pd[h] += q; Po[a] += q; Pd[a] += q; Pg[h] += qg
        lh = exp(m_ + hf / 2 + g[h] + o[h] - d[a]); la = exp(m_ - hf / 2 - g[h] + o[a] - d[h])
        eh = yh - lh; ea = ya - la
        S = Po[h] + Pd[a] + Pg[h]; den = lh * S + phi
        o[h] += Po[h] * eh / den; d[a] -= Pd[a] * eh / den
        g[h] += Pg[h] * eh / den; Pg[h] -= Pg[h] * Pg[h] * lh / den
        Po[h] -= Po[h] * Po[h] * lh / den; Pd[a] -= Pd[a] * Pd[a] * lh / den
        S = Po[a] + Pd[h] + Pg[h]; den = la * S + phi
        o[a] += Po[a] * ea / den; d[h] -= Pd[h] * ea / den
        g[h] -= Pg[h] * ea / den; Pg[h] -= Pg[h] * Pg[h] * la / den
        Po[a] -= Po[a] * Po[a] * la / den; Pd[h] -= Pd[h] * Pd[h] * la / den
        k = 1.0 / (Ln[lg] + 30)
        kM = max(kMu, k); kHH = max(kH, k)
        mu[lg] += kM * (eh / lh + ea / la) / 2; hfa[lg] += kHH * (eh / lh - ea / la) / 2
        Ln[lg] += 1
    if return_state:
        return out, {'o': o, 'd': d, 'g': g, 'Po': Po, 'Pd': Pd, 'mu': mu, 'hfa': hfa}
    return out


# ---------------- production Markov chain through an interpolation grid ----------------
class Chain:
    def __init__(self):
        G = json.load(open(os.path.join(WORK, 'chain_grid.json')))
        self.g = np.array(G['g']); self.T = np.stack([np.array(G['pH']), np.array(G['pD']), np.array(G['pA'])], -1)
        self.step = self.g[1] - self.g[0]
    def probs(self, lh, la):
        x = (np.log(np.clip(lh, 0.121, 5.99)) - self.g[0]) / self.step; yv = (np.log(np.clip(la, 0.121, 5.99)) - self.g[0]) / self.step
        i = np.clip(np.floor(x).astype(int), 0, len(self.g) - 2); j = np.clip(np.floor(yv).astype(int), 0, len(self.g) - 2)
        fx = (x - i)[:, None]; fy = (yv - j)[:, None]
        T = self.T
        p = T[i, j] * (1 - fx) * (1 - fy) + T[i + 1, j] * fx * (1 - fy) + T[i, j + 1] * (1 - fx) * fy + T[i + 1, j + 1] * fx * fy
        return p / p.sum(1, keepdims=True)


def rps(p, hg, ag):
    k = np.where(hg > ag, 0, np.where(hg == ag, 1, 2)); yv = np.eye(3)[k]
    return ((p[:, 0] - yv[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - yv[:, 0] - yv[:, 1]) ** 2) / 2


def logloss(p, hg, ag):
    k = np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    return -np.log(np.clip(p[np.arange(len(k)), k], 1e-9, 1))


def write_preds(D, p, name):
    """p: probabilities for D.scored rows (in order)."""
    df = pd.DataFrame({'id': D.id[D.scored], 'pH': p[:, 0], 'pD': p[:, 1], 'pA': p[:, 2]})
    f = os.path.join(HERE, f'preds_{name}.csv'); df.round(6).to_csv(f, index=False); return f


if __name__ == '__main__':
    D = Data()
    out = run_filter(D, D.v1_obs())
    B = D.B
    print('n', len(D.M), 'max |d g_lh|', np.abs(out[:, 0] - B.g_lh).max(), 'max |d g_la|', np.abs(out[:, 1] - B.g_la).max(),
          'Sh', np.abs(out[:, 2] - B.g_Sh).max(), 'mu', np.abs(out[:, 4] - B.mu).max(), 'hfa', np.abs(out[:, 5] - B.hfa).max())
    C = Chain()
    s = D.scored
    p = C.probs(B.lh.to_numpy()[s], B.la.to_numpy()[s])
    pb = pd.read_csv(SHARED + 'preds_baseline.csv', dtype={'id': str}).set_index('id').loc[D.id[s]].to_numpy()
    print('grid vs exact chain: max abs prob error', np.abs(p - pb).max(), ' RPS grid', rps(p, D.hg[s], D.ag[s]).mean(), ' RPS exact', rps(pb, D.hg[s], D.ag[s]).mean())
    import time; t0 = time.time(); run_filter(D, D.v1_obs()); print('filter time', round(time.time() - t0, 3), 's')
