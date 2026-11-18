#!/usr/bin/env python
"""Empilhamento final (walk-forward) de arquivos de previsão id,pH,pD,pA.
Uso: python stack.py saida.csv modelo1.csv modelo2.csv ...
Para a temporada Y, os pesos são estimados só com temporadas < Y.
Métodos comparados: média linear, pool log-linear com pesos + temperatura
(otimizados em log-loss com penalização para pesos iguais), e regressão
logística multinomial nos logitos. O método é escolhido por DEV."""
import sys, json, numpy as np, pandas as pd
from scipy.optimize import minimize
S = '/home/user/bdv-lab/shared/'
M = pd.DataFrame(json.load(open(S + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
M = M[(M.lg == 'A') & (M.season >= 2013) & (M.status == 'post')].copy(); M['id'] = M['id'].astype(str)
M['k'] = np.where(M.hg > M.ag, 0, np.where(M.hg == M.ag, 1, 2))

def load(files):
    P = []
    for f in files:
        d = pd.read_csv(f, dtype={'id': str}).drop_duplicates('id').set_index('id').reindex(M['id'])
        p = d[['pH', 'pD', 'pA']].to_numpy(float)
        if np.isnan(p).any(): raise SystemExit(f'{f}: faltam {int(np.isnan(p).any(1).sum())} jogos')
        p = np.clip(p, 1e-6, 1); P.append(p / p.sum(1, keepdims=True))
    return np.stack(P, 1)  # n x m x 3

def rps(p, k):
    y = np.eye(3)[k]; return (((p[:, 0] - y[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - y[:, 0] - y[:, 1]) ** 2) / 2)

def pool(P, w, t=1.0, bias=None):
    z = (np.log(P) * w[None, :, None]).sum(1) * t
    if bias is not None: z = z + bias[None, :]
    z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

def fit_pool(P, k, lam=2.0, use_bias=True):
    m = P.shape[1]
    def unpack(x):
        w = np.exp(x[:m]); w = w / w.sum(); t = np.exp(x[m]); b = np.concatenate([x[m + 1:m + 3], [0.0]]) if use_bias else None
        return w, t, b
    def loss(x):
        w, t, b = unpack(x); p = pool(P, w, t, b)
        return -np.log(p[np.arange(len(k)), k]).mean() + lam * ((w - 1 / m) ** 2).sum() / len(k) * 50 + (0.5 * (x[m + 1:] ** 2).sum() / len(k) * 20 if use_bias else 0)
    x0 = np.zeros(m + 1 + (2 if use_bias else 0))
    r = minimize(loss, x0, method='L-BFGS-B')
    return unpack(r.x)

def main(out, files):
    P = load(files); k = M['k'].to_numpy(); seas = M['season'].to_numpy()
    names = [f.split('/')[-1].replace('preds_', '').replace('.csv', '') for f in files]
    res = {'media': np.zeros((len(k), 3)), 'pool': np.zeros((len(k), 3)), 'pool_sb': np.zeros((len(k), 3))}
    last_w = None
    for Y in sorted(set(seas)):
        te = seas == Y; tr = seas < Y
        res['media'][te] = P[te].mean(1)
        if tr.sum() < 700:
            res['pool'][te] = res['pool_sb'][te] = pool(P[te], np.ones(P.shape[1]) / P.shape[1])
            continue
        w, t, b = fit_pool(P[tr], k[tr]); res['pool'][te] = pool(P[te], w, t, b); last_w = (w, t, b)
        w2, t2, _ = fit_pool(P[tr], k[tr], use_bias=False); res['pool_sb'][te] = pool(P[te], w2, t2)
    dev = seas <= 2021; hold = seas >= 2022
    print('modelos individuais:')
    for i, n in enumerate(names): print(f'  {n:28s} DEV {rps(P[dev, i], k[dev]).mean():.5f}  HOLD {rps(P[hold, i], k[hold]).mean():.5f}')
    best = None
    for n, p in res.items():
        d, h = rps(p[dev], k[dev]).mean(), rps(p[hold], k[hold]).mean()
        print(f'  >> {n:10s} DEV {d:.5f}  HOLD {h:.5f}  TOTAL {rps(p, k).mean():.5f}')
        if best is None or d < best[1]: best = (n, d)
    print('escolhido por DEV:', best[0])
    if last_w: print('pesos finais (última temporada):', dict(zip(names, np.round(last_w[0], 3))), 'temperatura', round(float(last_w[1]), 3), 'viés', None if last_w[2] is None else np.round(last_w[2], 3))
    o = M[['id']].copy(); o[['pH', 'pD', 'pA']] = res[best[0]]; o.to_csv(out, index=False)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
