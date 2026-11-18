"""Shared helpers for the ratings-zoo track: data loading, v1 score-chain port, metrics."""
import json
import os
import numpy as np
import pandas as pd

S = '/home/user/bdv-lab/shared/'
Z = '/home/user/bdv-lab/zoo/'

# ESPN changed the id of a few clubs over the years; v1 treats them as different teams.
ID_MERGE = {'5266': '3395',   # Goiás (2006-2013 -> 2014+)
            '6274': '4929',   # Santa Cruz
            '10982': '6154',  # América-MG
            '6080': '3448', '7385': '3448'}  # Guarani


def load_matches():
    M = json.load(open(S + 'matches.json'))
    for m in M:
        m['h0'], m['a0'] = m['h'], m['a']
        m['h'] = ID_MERGE.get(m['h'], m['h'])
        m['a'] = ID_MERGE.get(m['a'], m['a'])
    return M


def load_base():
    b = pd.read_csv(S + 'baseline_features.csv', dtype={'id': str, 'h': str, 'a': str})
    return b


# Frozen copy of the v1 hazard table (model.hazard of site/data/model.json as exported for the shared baseline),
# so results do not move if the repository's model.json is refitted while the study runs.
_HZ_FILE = Z + 'v1_hazard.json'
if not os.path.exists(_HZ_FILE):
    json.dump(json.load(open('/home/user/brasileiraodovitroncio/site/data/model.json'))['hazard'], open(_HZ_FILE, 'w'))
HZ = json.load(open(_HZ_FILE))

MAXG = 10


def chain_probs(lh, la, hz=HZ, return_matrix=False):
    """Vectorised port of v1 scoreDistribution (match-model.js): exact DP of the minute-level Markov chain.
    lh, la: arrays of pre-match expected goals. Returns [n,3] (pH,pD,pA)."""
    lh = np.atleast_1d(np.asarray(lh, float)); la = np.atleast_1d(np.asarray(la, float))
    n = len(lh); G1 = MAXG + 1
    bins = []
    for m in range(1, 46): bins.append(0 if m <= 15 else 1 if m <= 30 else 2)
    bins += [3] * hz['S1']
    for m in range(46, 91): bins.append(4 if m <= 60 else 5 if m <= 75 else 6)
    bins += [7] * hz['S2']
    g = hz['goal']; r = hz['red']
    goalF = np.zeros((8, 5, 3)); redH = np.zeros((8, 5)); redA = np.zeros((8, 5))
    for b in range(8):
        for d in range(5):
            for mi in range(3):
                e = g['bin'][b] + g['diff'][d] + g['men'][mi]
                if b >= 6 and d < 2: e += g.get('lateTrail', 0) or 0
                goalF[b, d, mi] = np.exp(e) / 90 * (g.get('norm', 1) or 1)
            redH[b, d] = np.exp(r['bin'][b] + r['diff'][d])
            redA[b, d] = np.exp(r['bin'][b] + r['diff'][d] + r['away'])
    hg = np.arange(G1)[:, None]; ag = np.arange(G1)[None, :]
    dH = np.clip(hg - ag, -2, 2) + 2  # [G1,G1]
    dA = np.clip(ag - hg, -2, 2) + 2
    rh = np.arange(3)[:, None]; ra = np.arange(3)[None, :]
    menH = np.where(ra > rh, 2, np.where(rh > ra, 0, 1))  # [3,3] home has one more = 2
    menA = menH.T
    cur = np.zeros((n, G1, G1, 3, 3)); cur[:, 0, 0, 0, 0] = 1.0
    LH = lh[:, None, None, None, None]; LA = la[:, None, None, None, None]
    for b in bins:
        gH = goalF[b][dH[:, :, None, None], menH[None, None, :, :]]  # [G1,G1,3,3]
        gA = goalF[b][dA[:, :, None, None], menA[None, None, :, :]]
        gH = np.broadcast_to(gH, (G1, G1, 3, 3)).copy(); gA = np.broadcast_to(gA, (G1, G1, 3, 3)).copy()
        gH[MAXG, :, :, :] = 0; gA[:, MAXG, :, :] = 0
        pRH = np.broadcast_to(redH[b][dH][:, :, None, None], (G1, G1, 3, 3)).copy(); pRH[:, :, 2, :] = 0
        pRA = np.broadcast_to(redA[b][dA][:, :, None, None], (G1, G1, 3, 3)).copy(); pRA[:, :, :, 2] = 0
        pgh = LH * gH[None]; pga = LA * gA[None]
        nxt = cur * (1 - pgh - pga - pRH[None] - pRA[None])
        nxt[:, 1:, :, :, :] += (cur * pgh)[:, :-1]
        nxt[:, :, 1:, :, :] += (cur * pga)[:, :, :-1]
        nxt[:, :, :, 1:, :] += (cur * pRH[None])[:, :, :, :-1]
        nxt[:, :, :, :, 1:] += (cur * pRA[None])[:, :, :, :, :-1]
        cur = nxt
    mat = cur.sum(axis=(3, 4))
    if return_matrix:
        return mat
    iu = np.triu_indices(G1, 1); il = np.tril_indices(G1, -1)
    pA = mat[:, iu[0], iu[1]].sum(1); pH = mat[:, il[0], il[1]].sum(1); pD = np.trace(mat, axis1=1, axis2=2)
    return np.stack([pH, pD, pA], 1)


class ChainGrid:
    """Bilinear lookup of the v1 chain on a (log lh, log la) grid -- fast and differentiable enough for fitting."""

    def __init__(self, lo=0.08, hi=6.0, k=121):
        f = Z + 'cache_chain_grid.npz'
        self.x = np.linspace(np.log(lo), np.log(hi), k)
        if os.path.exists(f):
            self.P = np.load(f)['P']
        else:
            A, B = np.meshgrid(self.x, self.x, indexing='ij')
            P = np.zeros((k * k, 3))
            a = np.exp(A.ravel()); b = np.exp(B.ravel())
            for i in range(0, k * k, 2000):
                P[i:i + 2000] = chain_probs(a[i:i + 2000], b[i:i + 2000])
            self.P = P.reshape(k, k, 3)
            np.savez_compressed(f, P=self.P)
        self.k = k

    def __call__(self, lh, la):
        x = self.x
        u = (np.clip(np.log(lh), x[0], x[-1] - 1e-9) - x[0]) / (x[1] - x[0])
        v = (np.clip(np.log(la), x[0], x[-1] - 1e-9) - x[0]) / (x[1] - x[0])
        i = np.floor(u).astype(int); j = np.floor(v).astype(int); fu = (u - i)[:, None]; fv = (v - j)[:, None]
        P = self.P
        p = (P[i, j] * (1 - fu) * (1 - fv) + P[i + 1, j] * fu * (1 - fv) + P[i, j + 1] * (1 - fu) * fv + P[i + 1, j + 1] * fu * fv)
        return p / p.sum(1, keepdims=True)


def rps_ll(p, k):
    p = np.clip(np.asarray(p, float), 1e-9, 1); p = p / p.sum(1, keepdims=True)
    y = np.eye(3)[k]
    rps = ((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2
    ll = -np.log(p[np.arange(len(k)), k])
    return rps, ll


def boot_ci(diff, B=2000, seed=7):
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(B)]
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)
