"""Numpy port of v1's exact minute-by-minute Markov chain (site/js/engine/match-model.js,
scoreDistribution) vectorised over many (lh, la) pairs, plus a lookup grid in log-lambda
space used as a differentiable lambda -> (pH, pD, pA) map inside the networks.

The hazard parameters are v1's own (model.json); they are part of the baseline, exactly as
lh/la in baseline_features.csv are.  With delta = 0 the residual networks reproduce v1."""
import json
import numpy as np

MODEL_JSON = '/home/user/brasileiraodovitroncio/site/data/model.json'
MAXG = 10
G1 = MAXG + 1


def _diff_idx(d):
    return np.clip(d, -2, 2) + 2


def chain_probs(lh, la, hz=None, return_matrix=False):
    """lh, la: arrays (n,). Returns (n,3) [pH,pD,pA] (and optionally the (n,11,11) score matrix)."""
    if hz is None:
        hz = json.load(open(MODEL_JSON))['hazard']
    lh = np.asarray(lh, float).ravel(); la = np.asarray(la, float).ravel()
    n = len(lh)
    # timeline bins
    bins = []
    for m in range(1, 46): bins.append(0 if m <= 15 else 1 if m <= 30 else 2)
    bins += [3] * hz['S1']
    for m in range(46, 91): bins.append(4 if m <= 60 else 5 if m <= 75 else 6)
    bins += [7] * hz['S2']
    g = hz['goal']; r = hz['red']
    goalF = np.zeros((8, 5, 3)); redH = np.zeros((8, 5)); redA = np.zeros((8, 5))
    for b in range(8):
        for d in range(5):
            for mm in range(3):
                e = g['bin'][b] + g['diff'][d] + g['men'][mm]
                if b >= 6 and d < 2: e += g.get('lateTrail', 0) or 0
                goalF[b, d, mm] = np.exp(e) / 90 * (g.get('norm', 1) or 1)
            redH[b, d] = np.exp(r['bin'][b] + r['diff'][d])
            redA[b, d] = np.exp(r['bin'][b] + r['diff'][d] + r['away'])
    hg = np.arange(G1)[:, None, None, None]; ag = np.arange(G1)[None, :, None, None]
    rh = np.arange(3)[None, None, :, None]; ra = np.arange(3)[None, None, None, :]
    dH = np.broadcast_to(_diff_idx(hg - ag), (G1, G1, 3, 3)); dA = np.broadcast_to(_diff_idx(ag - hg), (G1, G1, 3, 3))
    # menIdx(own, opp): 2 if opp>own (has one more), 0 if own>opp, else 1
    menH = np.broadcast_to(np.where(ra > rh, 2, np.where(rh > ra, 0, 1)), (G1, G1, 3, 3))
    menA = np.broadcast_to(np.where(rh > ra, 2, np.where(ra > rh, 0, 1)), (G1, G1, 3, 3))
    okH = np.broadcast_to(hg < MAXG, (G1, G1, 3, 3)).astype(float); okA = np.broadcast_to(ag < MAXG, (G1, G1, 3, 3)).astype(float)
    okRh = np.broadcast_to(rh < 2, (G1, G1, 3, 3)).astype(float); okRa = np.broadcast_to(ra < 2, (G1, G1, 3, 3)).astype(float)
    cur = np.zeros((n, G1, G1, 3, 3)); cur[:, 0, 0, 0, 0] = 1
    LH = lh[:, None, None, None, None]; LA = la[:, None, None, None, None]
    for b in bins:
        pgh = LH * (goalF[b][dH, menH] * okH)[None]
        pga = LA * (goalF[b][dA, menA] * okA)[None]
        prh = (redH[b][dH] * okRh)[None]; pra = (redA[b][dA] * okRa)[None]
        stay = 1 - pgh - pga - prh - pra
        nxt = cur * stay
        nxt[:, 1:, :, :, :] += (cur * pgh)[:, :-1]
        nxt[:, :, 1:, :, :] += (cur * pga)[:, :, :-1]
        nxt[:, :, :, 1:, :] += (cur * prh)[:, :, :, :-1]
        nxt[:, :, :, :, 1:] += (cur * pra)[:, :, :, :, :-1]
        cur = nxt
    mat = cur.sum((3, 4))
    iu = np.triu_indices(G1, 1); il = np.tril_indices(G1, -1)
    pA = mat[:, iu[0], iu[1]].sum(1); pH = mat[:, il[0], il[1]].sum(1); pD = np.trace(mat, axis1=1, axis2=2)
    out = np.stack([pH, pD, pA], 1)
    return (out, mat) if return_matrix else out


# ---- lookup grid ---------------------------------------------------------------------------
LOG_MIN, LOG_MAX, NG = np.log(0.08), np.log(6.0), 121


def build_grid():
    ax = np.linspace(LOG_MIN, LOG_MAX, NG)
    A, B = np.meshgrid(ax, ax, indexing='ij')
    out = np.zeros((NG * NG, 3))
    lh = np.exp(A.ravel()); la = np.exp(B.ravel())
    for s in range(0, NG * NG, 2048):
        out[s:s + 2048] = chain_probs(lh[s:s + 2048], la[s:s + 2048])
    return out.reshape(NG, NG, 3)


def grid_lookup_np(grid, loglh, logla):
    u = (np.clip(loglh, LOG_MIN, LOG_MAX - 1e-9) - LOG_MIN) / (LOG_MAX - LOG_MIN) * (NG - 1)
    v = (np.clip(logla, LOG_MIN, LOG_MAX - 1e-9) - LOG_MIN) / (LOG_MAX - LOG_MIN) * (NG - 1)
    i = np.floor(u).astype(int); j = np.floor(v).astype(int); fu = (u - i)[:, None]; fv = (v - j)[:, None]
    return (grid[i, j] * (1 - fu) * (1 - fv) + grid[i + 1, j] * fu * (1 - fv) + grid[i, j + 1] * (1 - fu) * fv + grid[i + 1, j + 1] * fu * fv)


if __name__ == '__main__':
    import pandas as pd
    S = '/home/user/bdv-lab/shared/'
    f = pd.read_csv(S + 'baseline_features.csv', dtype={'id': str}); b = pd.read_csv(S + 'preds_baseline.csv', dtype={'id': str})
    d = b.merge(f[['id', 'lh', 'la']], on='id')
    p = chain_probs(d.lh.values, d.la.values)
    print('exact chain max abs err vs preds_baseline:', np.abs(p - d[['pH', 'pD', 'pA']].values).max())
    grid = build_grid(); np.save('/home/user/bdv-lab/neural/chain_grid.npy', grid)
    q = grid_lookup_np(grid, np.log(d.lh.values), np.log(d.la.values))
    print('grid interp max abs err:', np.abs(q - d[['pH', 'pD', 'pA']].values).max())
    print('lambda range', f.lh.min(), f.lh.max(), f.la.min(), f.la.max())
