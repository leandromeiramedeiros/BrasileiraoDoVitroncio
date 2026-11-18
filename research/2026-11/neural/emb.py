"""Model 1 - learned team embeddings with time dynamics (no v1 input at all).

  e[team, season] in R^d;  prior  e[t, s] ~ N(rho * e[t, s_prev] + b[transition], 1/kappa)
  transition in {stay, promoted B->A, relegated A->B, new/returning in A, new/returning in B}
  head 'pois'   : (log lh, log la) = league constants + linear(e_h, e_a) + small MLP(e_h, e_a); 1X2 from an
                  independent-Poisson grid with learnable draw inflation; loss = Poisson NLL + RPS
  head 'ologit' : scalar s = linear + MLP; ordered logit with two learnable cut points; loss = cross-entropy

Walk-forward for season Y:
  1. fit everything (embeddings of all team-seasons < Y, rho, b, head) on matches of seasons < Y;
  2. freeze the network; the season-Y embeddings start at their prior mean and are re-estimated (MAP, a few
     warm-started Adam steps) before every match day of Y using only season-Y matches already finished.
usage: python emb.py [pois|ologit] [--dev] [d=6 kappa=3 ...]"""
import sys, json, time
import numpy as np
import torch, torch.nn as nn
from nn_lib import Data, PoissonGrid, rps_loss, ce_loss, pois_nll, summarize, save_run, D

torch.set_num_threads(1)
CFG = dict(head='pois', d=6, dh=16, kappa=3.0, lr=0.02, steps=400, wd=1e-3, K=8, lr_on=0.03, w_rps=1.0, w_pois=0.5)


class EmbNet(nn.Module):
    def __init__(self, cfg, n_ts):
        super().__init__()
        d = cfg['d']; self.cfg = cfg
        self.E = nn.Parameter(torch.randn(n_ts, d) * 0.05)
        self.rho_raw = nn.Parameter(torch.tensor(1.5)); self.b = nn.Parameter(torch.zeros(5, d))
        nout = 2 if cfg['head'] == 'pois' else 1
        self.lin = nn.Linear(2 * d, nout); self.mlp = nn.Sequential(nn.Linear(2 * d, cfg['dh']), nn.Tanh(), nn.Linear(cfg['dh'], nout))
        nn.init.zeros_(self.mlp[2].weight); nn.init.zeros_(self.mlp[2].bias)
        self.lgc = nn.Parameter(torch.zeros(2, nout))
        with torch.no_grad():
            if cfg['head'] == 'pois': self.lin.bias.copy_(torch.tensor([np.log(1.45), np.log(0.99)]))
        self.pg = PoissonGrid(1); self.cut = nn.Parameter(torch.tensor([-0.6, 0.6]))

    def prior_mean(self, E, prev, trans):
        rho = torch.sigmoid(self.rho_raw)
        return torch.where((prev >= 0)[:, None], rho * E[prev.clamp(min=0)], torch.zeros(1)) + self.b[trans]

    def predict(self, eh, ea, lgB):
        z = torch.cat([eh, ea], 1); o = self.lin(z) + self.mlp(z) + self.lgc[lgB]
        if self.cfg['head'] == 'pois':
            p = self.pg(o[None, :, 0], o[None, :, 1])[0]
            return p, o
        s = o[:, 0]; c1 = self.cut[0]; c2 = c1 + torch.nn.functional.softplus(self.cut[1] - self.cut[0])
        pa = torch.sigmoid(c1 - s); pd_ = torch.sigmoid(c2 - s) - pa
        return torch.stack([1 - pa - pd_, pd_, pa], 1), None

    def match_loss(self, p, o, y, hg, ag):
        if self.cfg['head'] == 'pois':
            return self.cfg['w_rps'] * rps_loss(p, y) + self.cfg['w_pois'] * (pois_nll(o[:, 0], hg) + pois_nll(o[:, 1], ag))
        return ce_loss(p, y)


def build_index(data):
    """team-season ids, previous team-season and transition type (all known before the season starts)."""
    key = {}; ts_h = np.zeros(data.n, int); ts_a = np.zeros(data.n, int); info = []
    last = {}                                                   # team -> (season, lgB, ts index)
    order = np.lexsort((np.arange(data.n), data.season))
    for i in order:
        for side, t in ((0, data.h[i]), (1, data.a[i])):
            k = (t, data.season[i])
            if k not in key:
                key[k] = len(info); lg = int(data.lgB[i])
                if t in last and last[t][0] == data.season[i] - 1:
                    pl = last[t][1]; tr = 0 if pl == lg else (1 if lg == 0 else 2); prev = last[t][2]
                else:
                    tr = 3 + lg; prev = -1
                info.append((data.season[i], prev, tr)); last[t] = (data.season[i], lg, key[k])
            (ts_h if side == 0 else ts_a)[i] = key[k]
    info = np.array(info)
    return ts_h, ts_a, info[:, 0], info[:, 1], info[:, 2]


def run(data, cfg, seeds, seasons, tag=''):
    ts_h, ts_a, ts_season, ts_prev, ts_tr = build_index(data)
    T = lambda x, dt=torch.long: torch.tensor(np.asarray(x), dtype=dt)
    th, ta, lgB, y = T(ts_h), T(ts_a), T(data.lgB.astype(int)), T(np.maximum(data.y, 0))
    hg, ag = T(np.maximum(data.hg, 0), torch.float32), T(np.maximum(data.ag, 0), torch.float32)
    prev, trans = T(ts_prev), T(ts_tr)
    dayi = np.floor(data.day).astype(int)
    P = np.full((len(seeds), data.n, 3), np.nan, np.float32); LAM = np.full((len(seeds), data.n, 2), np.nan, np.float32)
    t0 = time.time()
    for Y in seasons:
        tr_idx = T(np.where(data.post & (data.season < Y))[0]); old = T(np.where(ts_season < Y)[0])
        cur = np.where(ts_season == Y)[0]; cur_t = T(cur); loc = -np.ones(len(ts_season), int); loc[cur] = np.arange(len(cur)); loc_t = T(loc)
        te_all = np.where(data.season == Y)[0]
        for si, sd in enumerate(seeds):
            torch.manual_seed(1000 * sd + Y)
            net = EmbNet(cfg, len(ts_season))
            opt = torch.optim.Adam([{'params': [net.E, net.b, net.rho_raw, net.lgc, net.cut] + list(net.pg.parameters()), 'weight_decay': 0.0},
                                    {'params': list(net.lin.parameters()) + list(net.mlp.parameters()), 'weight_decay': cfg['wd']}], lr=cfg['lr'])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg['steps'])
            for step in range(cfg['steps']):                                   # 1. full-batch fit on seasons < Y
                p, o = net.predict(net.E[th[tr_idx]], net.E[ta[tr_idx]], lgB[tr_idx])
                pri = ((net.E[old] - net.prior_mean(net.E, prev[old], trans[old])) ** 2).sum()
                loss = (net.match_loss(p, o, y[tr_idx], hg[tr_idx], ag[tr_idx]).sum() + cfg['kappa'] * pri) / len(tr_idx)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            for q in net.parameters(): q.requires_grad_(False)                 # 2. frozen net, online team states
            mu0 = net.prior_mean(net.E, prev[cur_t], trans[cur_t]).clone()
            EY = mu0.clone().requires_grad_(True)
            oo = torch.optim.Adam([EY], lr=cfg['lr_on'])
            done = []                                                          # finished season-Y matches so far
            for dday in np.unique(dayi[te_all]):
                blk = te_all[dayi[te_all] == dday]
                if done:
                    di = T(np.array(done))
                    for _ in range(cfg['K']):
                        p, o = net.predict(EY[loc_t[th[di]]], EY[loc_t[ta[di]]], lgB[di])
                        loss = (net.match_loss(p, o, y[di], hg[di], ag[di]).sum() + cfg['kappa'] * ((EY - mu0) ** 2).sum()) / max(len(di), 50)
                        oo.zero_grad(); loss.backward(); oo.step()
                with torch.no_grad():
                    bi = T(blk); p, o = net.predict(EY[loc_t[th[bi]]], EY[loc_t[ta[bi]]], lgB[bi])
                    P[si, blk] = p.numpy()
                    if o is not None: LAM[si, blk] = o.exp().numpy()
                done += [int(i) for i in blk if data.post[i]]                  # only now do the block's results become visible
        print(f'  [{tag}] season {Y} done, elapsed {time.time() - t0:.0f}s', flush=True)
    return P, LAM


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]; dev = '--dev' in sys.argv
    cfg = dict(CFG); cfg['head'] = args[0] if args and '=' not in args[0] else 'pois'
    for a in args:
        if '=' in a: k, v = a.split('='); cfg[k] = type(CFG[k])(v)
    data = Data()
    tag = 'emb_' + cfg['head'] + ''.join(f'_{a}' for a in args if '=' in a) + ('_dev' if dev else '')
    P, LAM = run(data, cfg, [0, 1, 2, 3, 4], range(2013, 2022 if dev else 2027), tag)
    per, avg = summarize(data, P, tag, dev_only=dev)
    save_run(tag, data, P, LAM, cfg)
    open(D + 'results.jsonl', 'a').write(json.dumps(dict(tag=tag, cfg=cfg, seeds=5, final=True, per_seed={k: [float(x[k]['d']) for x in per] for k in avg},
                                                         avg={k: {q: float(v) for q, v in avg[k].items()} for k in avg})) + '\n')
