"""Neural forecasters for the Vitrôncio bench (track: neural).

Walk-forward protocol implemented here (see run_walk):
  for every season Y in 2013..2026:
      weights are fitted from scratch on finished matches with season < Y (Série A + B),
      with a FIXED epoch budget (no early stopping, nothing from season >= Y is looked at);
      feature standardisation uses the same training rows only;
      then every match of season Y is predicted from the frozen network.  The network input for a
      season-Y match is the team's own history of matches finished before kick-off (this is how the
      model "updates online" during Y) plus v1's pre-match numbers for that match.
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

D = '/home/user/bdv-lab/neural/'
torch.set_num_threads(int(os.environ.get('NN_THREADS', '1')))


# --------------------------------------------------------------------------------------------- data
class Data:
    def __init__(self):
        z = np.load(D + 'data.npz', allow_pickle=True)
        for k in z.files: setattr(self, k, z[k])
        self.n = len(self.ids)
        self.tok_names = [str(x) for x in self.tok_names]; self.ctx_names = [str(x) for x in self.ctx_names]
        self.NMAX = self.hist.shape[2]
        self.grid = np.load(D + 'chain_grid.npy')
        # rest days (pre-match): days since the team's previous finished match
        last = self.hist[:, :, -1]
        rest = np.where(last >= 0, self.day[:, None] - self.tok_day[np.maximum(last, 0)], 200.0)
        self.rest = np.clip(np.log1p(rest) / 3.0, 0, 2).astype(np.float32)           # (n,2)
        self.nhist = ((self.hist >= 0).sum(2) / self.NMAX).astype(np.float32)        # (n,2)

    def tok_cols(self, groups):
        """groups: subset of {'base','opp','v1','ev','st'} -> column indices"""
        out = []
        for i, nm in enumerate(self.tok_names):
            g = 'v1' if nm.startswith(('v1_', 'sur_')) else 'ev' if nm.startswith('ev_') else 'st' if nm.startswith('st_') else 'opp' if nm.startswith('opp_') else 'base'
            if nm == 'ev_sur_yc': g = 'v1ev'
            if g in groups or (g == 'v1ev' and 'v1' in groups and 'ev' in groups): out.append(i)
        return out


# ------------------------------------------------------------------------------------------- pieces
class ChainGrid(nn.Module):
    """Differentiable (bilinear) lookup of v1's exact Markov-chain map (lh, la) -> (pH, pD, pA)."""
    def __init__(self, grid):
        super().__init__()
        from chain import LOG_MIN, LOG_MAX, NG
        self.register_buffer('g', torch.tensor(grid, dtype=torch.float32)); self.lo, self.hi, self.ng = LOG_MIN, LOG_MAX, NG

    def forward(self, llh, lla):
        u = (llh.clamp(self.lo, self.hi - 1e-4) - self.lo) / (self.hi - self.lo) * (self.ng - 1)
        v = (lla.clamp(self.lo, self.hi - 1e-4) - self.lo) / (self.hi - self.lo) * (self.ng - 1)
        i = u.floor().long().clamp(0, self.ng - 2); j = v.floor().long().clamp(0, self.ng - 2)
        fu = (u - i).unsqueeze(1); fv = (v - j).unsqueeze(1); g = self.g
        return g[i, j] * (1 - fu) * (1 - fv) + g[i + 1, j] * fu * (1 - fv) + g[i, j + 1] * (1 - fu) * fv + g[i + 1, j + 1] * fu * fv


class PoissonGrid(nn.Module):
    """Independent Poisson score grid with a learnable draw inflation (one per seed; fitted on training data only)."""
    def __init__(self, S, maxg=10):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(S, 1))
        k = torch.arange(maxg + 1, dtype=torch.float32)
        self.register_buffer('k', k); self.register_buffer('lf', torch.lgamma(k + 1))

    def forward(self, llh, lla):                            # (S, B)
        ph = torch.exp(self.k * llh.unsqueeze(-1) - torch.exp(llh).unsqueeze(-1) - self.lf)
        pa = torch.exp(self.k * lla.unsqueeze(-1) - torch.exp(lla).unsqueeze(-1) - self.lf)
        m = ph.unsqueeze(-1) * pa.unsqueeze(-2)             # (S, B, G, G): rows = home goals
        d = torch.diagonal(m, dim1=-2, dim2=-1).sum(-1) * torch.exp(self.gamma)
        h = torch.tril(m, -1).sum((-2, -1)); a = torch.triu(m, 1).sum((-2, -1))
        p = torch.stack([h, d, a], -1)
        return p / p.sum(-1, keepdim=True)


class ELinear(nn.Module):
    """S independent linear layers evaluated in one bmm (one per ensemble member / seed)."""
    def __init__(self, S, fin, fout, zero=False, bias=None, std=None):
        super().__init__()
        bound = 1 / np.sqrt(fin)
        self.W = nn.Parameter(torch.empty(S, fin, fout).uniform_(-bound, bound)); self.b = nn.Parameter(torch.empty(S, 1, fout).uniform_(-bound, bound))
        with torch.no_grad():
            if zero: self.W.zero_(); self.b.zero_()
            if std is not None: self.W.normal_(std=std)
            if bias is not None: self.b.copy_(torch.tensor(bias, dtype=torch.float32).expand(S, 1, fout))

    def forward(self, x):                                   # (S, ..., fin)
        sh = x.shape
        return torch.baddbmm(self.b, x.reshape(sh[0], -1, sh[-1]), self.W).reshape(*sh[:-1], -1)


class ELayerNorm(nn.Module):
    def __init__(self, S, d):
        super().__init__(); self.g = nn.Parameter(torch.ones(S, 1, 1, d)); self.b = nn.Parameter(torch.zeros(S, 1, 1, d)); self.d = d

    def forward(self, x): return Fn.layer_norm(x, (self.d,)) * self.g + self.b


class Encoder(nn.Module):
    """History encoder.  x: (S, B, N, F) tokens, oldest -> most recent (right-aligned); mask (S, B, N), 1 = real match."""
    def __init__(self, S, kind, fin, d, n, drop):
        super().__init__()
        self.S, self.kind, self.n, self.d, self.p = S, kind, n, d, drop
        self.inp = ELinear(S, fin, d)
        if kind == 'gru':
            self.Wi = ELinear(S, d, 3 * d); self.Wh = ELinear(S, d, 3 * d)
        elif kind == 'transformer':
            self.H = 4; self.pos = nn.Parameter(torch.randn(S, 1, n, d) * 0.02)
            self.blocks = nn.ModuleList([nn.ModuleDict(dict(ln1=ELayerNorm(S, d), qkv=ELinear(S, d, 3 * d), proj=ELinear(S, d, d), ln2=ELayerNorm(S, d),
                                                            ff1=ELinear(S, d, 2 * d), ff2=ELinear(S, 2 * d, d))) for _ in range(2)])
            self.lnf = ELayerNorm(S, d); self.att = ELinear(S, d, 1)
        elif kind == 'cnn':
            bound = 1 / np.sqrt(3 * d)
            self.cw = nn.ParameterList([nn.Parameter(torch.empty(S * d, d, 3).uniform_(-bound, bound)) for _ in range(2)])
            self.cb = nn.ParameterList([nn.Parameter(torch.zeros(S * d)) for _ in range(2)])
            self.out = ELinear(S, 3 * d, d)
        elif kind == 'deepsets':
            self.phi = ELinear(S, d, d); self.att = ELinear(S, d, 2); self.out = ELinear(S, 2 * d, d)
        else:
            raise ValueError(kind)

    def forward(self, x, mask):
        S, B, N, _ = x.shape; d = self.d
        z = torch.relu(self.inp(x))
        if self.kind == 'gru':
            gi = self.Wi(z).unbind(2); ms = mask.unsqueeze(-1).unbind(2); h = torch.zeros(S, B, d)
            for t in range(N):                                     # (unbind/chunk keep the backward pass cheap)
                ir, iu, ic = gi[t].chunk(3, -1); hr, hu, hc = self.Wh(h).chunk(3, -1)
                r = torch.sigmoid(ir + hr); u = torch.sigmoid(iu + hu); c = torch.tanh(ic + r * hc)
                h = torch.where(ms[t] > 0.5, (1 - u) * c + u * h, h)   # padded steps leave the state untouched
            return h
        if self.kind == 'transformer':
            m = mask.clone(); m[:, :, -1] = 1                      # never mask every key
            key = (m > 0.5).reshape(S * B, 1, 1, N)
            z = z + self.pos
            for blk in self.blocks:
                q, k, v = blk['qkv'](blk['ln1'](z)).reshape(S * B, N, 3, self.H, d // self.H).permute(2, 0, 3, 1, 4)
                a = Fn.scaled_dot_product_attention(q, k, v, attn_mask=key, dropout_p=self.p if self.training else 0.0)
                z = z + blk['proj'](a.transpose(1, 2).reshape(S, B, N, d))
                z = z + blk['ff2'](Fn.dropout(torch.relu(blk['ff1'](blk['ln2'](z))), self.p, self.training))
            z = self.lnf(z)
            w = self.att(z).squeeze(-1).masked_fill(m < 0.5, -1e9)
            return (torch.softmax(w, 2).unsqueeze(-1) * z).sum(2)
        if self.kind == 'cnn':
            mk = mask.unsqueeze(-1)
            c = (z * mk).permute(1, 0, 3, 2).reshape(B, S * d, N)
            for w, b in zip(self.cw, self.cb): c = torch.relu(Fn.conv1d(c, w, b, padding=1, groups=S))
            c = c.reshape(B, S, d, N).permute(1, 0, 3, 2) * mk
            mean = c.sum(2) / mask.sum(2, keepdim=True).clamp(min=1)
            return torch.relu(self.out(torch.cat([mean, c.max(2).values, c[:, :, -1]], -1)))
        if self.kind == 'deepsets':
            f = torch.relu(self.phi(z))
            m = mask.clone(); m[:, :, -1] = 1
            w = torch.softmax(self.att(z).masked_fill(m.unsqueeze(-1) < 0.5, -1e9), 2)       # two learned (recency-aware) attention pools
            return torch.relu(self.out(torch.einsum('sbnk,sbnd->sbkd', w, f).reshape(S, B, 2 * d)))


class Net(nn.Module):
    """S independent copies (seeds) of: siamese history encoder (shared by the two sides) + match head.

    head modes
      lam_resid   : (dh, da) = head(.) with zero-initialised last layer; lambda = lambda_v1 * exp(d);
                    probabilities = v1's Markov chain evaluated at the adjusted lambdas (=> exactly v1 at d = 0)
      logit_resid : logits = log p_v1 + head(.) (zero-init); separate zero-init lambda correction for the goal loss
      free        : head outputs log-lambdas directly; probabilities from an independent-Poisson grid with a
                    learnable draw inflation.  No v1 number enters the head unless cfg['ctx'] says so.
    """
    def __init__(self, S, cfg, fin, cin, grid):
        super().__init__()
        self.cfg = cfg; self.S = S; d = cfg['d']; self.p = cfg['drop']
        self.use_seq = cfg['enc'] != 'none'
        if self.use_seq: self.enc = Encoder(S, cfg['enc'], fin, d, cfg['N'], cfg['drop'])
        hin = (2 * d if self.use_seq else 0) + cin
        self.trunk = ELinear(S, hin, cfg['dh'])
        self.mode = cfg['head']
        if self.mode == 'lam_resid': self.out = ELinear(S, cfg['dh'], 2, zero=True)
        elif self.mode == 'logit_resid': self.out = ELinear(S, cfg['dh'], 5, zero=True)
        else: self.out = ELinear(S, cfg['dh'], 2, std=0.01, bias=[np.log(1.45), np.log(0.99)])
        self.chain = ChainGrid(grid); self.pg = PoissonGrid(S)
        self.aux = ELinear(S, cfg['dh'], 4, std=0.01, bias=[np.log(14.0), np.log(11.0), np.log(4.8), np.log(3.7)]) if cfg.get('w_aux', 0) > 0 else None

    def forward(self, xh, mh, xa, ma, ctx, llh0, lla0, logp0):
        S, B = llh0.shape; parts = []
        if self.use_seq:
            e = self.enc(torch.cat([xh, xa], 1), torch.cat([mh, ma], 1))
            e = Fn.dropout(e, self.p, self.training)
            parts += [e[:, :B], e[:, B:]]
        if ctx.shape[-1]: parts.append(ctx)
        t = Fn.dropout(torch.relu(self.trunk(torch.cat(parts, -1))), self.p, self.training); o = self.out(t)
        if self.mode == 'lam_resid':
            llh, lla = llh0 + o[..., 0], lla0 + o[..., 1]
            p = self.chain(llh.reshape(-1), lla.reshape(-1)).reshape(S, B, 3); delta = o
        elif self.mode == 'logit_resid':
            p = torch.softmax(logp0 + o[..., :3], -1)
            llh, lla = llh0 + o[..., 3], lla0 + o[..., 4]; delta = o
        else:
            llh, lla = o[..., 0], o[..., 1]
            p = self.pg(llh, lla); delta = None
        aux = self.aux(t) if self.aux is not None else None
        return p, llh, lla, delta, aux


# ------------------------------------------------------------------------------------------- losses
def rps_loss(p, y):
    oh = Fn.one_hot(y, 3).float()
    c1 = p[..., 0] - oh[..., 0]; c2 = p[..., 0] + p[..., 1] - oh[..., 0] - oh[..., 1]
    return (c1 * c1 + c2 * c2) / 2


def ce_loss(p, y):
    return -torch.log(p.gather(-1, y.unsqueeze(-1)).squeeze(-1).clamp(min=1e-7))


def pois_nll(loglam, k):
    return torch.exp(loglam) - k * loglam


DEFAULT = dict(enc='gru', d=32, dh=32, N=20, drop=0.2, head='lam_resid', groups=('base', 'v1', 'ev', 'st'), ctx='full',
               w_rps=1.0, w_ce=0.0, w_pois=0.0, w_aux=0.0, delta_pen=0.3, lr=1e-3, wd=0.05, epochs=15, bs=512, wB=1.0, first_season=2007, extras=True, tau=0.0)

CTX_SETS = {
    'none': [],
    'lg': ['lgB'],
    'v1lam': ['log_lh', 'log_la', 'lgB'],
    'full': ['log_lh', 'log_la', 'log_glh', 'log_gla', 'g_Sh', 'g_Sa', 'rel_sh_h', 'rel_sh_a', 'rel_sot_h', 'rel_sot_a', 'rel_co_h', 'rel_co_a',
             'rel_fo_h', 'rel_fo_a', 'nh', 'na', 'lgB', 'mu', 'hfa'],
    'pure': ['nh', 'na', 'lgB'],            # nothing from v1: games played this season + league flag
}


class Fold:
    """Tensors for one configuration (token columns / history length / context set)."""
    def __init__(self, data, cfg):
        self.data, self.cfg = data, cfg
        cols = data.tok_cols(set(cfg['groups'])); N = cfg['N']
        tok = np.vstack([data.TOK[:, cols], np.zeros((1, len(cols)), np.float32)])           # last row = padding
        self.tok = torch.tensor(tok)
        self.tok_day = torch.tensor(np.append(data.tok_day, 0).astype(np.float32)); self.tok_season = torch.tensor(np.append(data.tok_season, 0).astype(np.float32))
        h = data.hist[:, :, data.NMAX - N:].copy(); self.mask = torch.tensor((h >= 0).astype(np.float32)); h[h < 0] = len(tok) - 1
        self.hist = torch.tensor(h.astype(np.int64))
        cn = CTX_SETS[cfg['ctx']]; ci = [data.ctx_names.index(c) for c in cn]
        extra = np.concatenate([data.rest, data.nhist], 1)                                    # rest days, history length (not v1)
        if not cfg['extras']: extra = extra[:, :0]
        self.ctx_raw = np.concatenate([data.CTX[:, ci], extra], 1).astype(np.float32)
        self.ctx_floor = np.array([0.3 if c.startswith('rel_') else 1e-6 for c in cn] + [1e-6] * extra.shape[1], np.float32)
        self.day = torch.tensor(data.day.astype(np.float32)); self.season = torch.tensor(data.season.astype(np.float32))
        self.llh0 = torch.tensor(np.log(data.lh).astype(np.float32)); self.lla0 = torch.tensor(np.log(data.la).astype(np.float32))
        self.logp0 = torch.tensor(np.log(data.pv1).astype(np.float32))
        self.y = torch.tensor(np.maximum(data.y, 0).astype(np.int64)); self.hg = torch.tensor(np.maximum(data.hg, 0).astype(np.float32)); self.ag = torch.tensor(np.maximum(data.ag, 0).astype(np.float32))
        self.shots = torch.tensor(np.concatenate([data.st[:, :, 0], data.st[:, :, 1]], 1).astype(np.float32)); self.stok = torch.tensor(data.stok.astype(np.float32))
        self.w = torch.tensor(np.where(data.lgB > 0, cfg['wB'], 1.0).astype(np.float32))
        self.fin = len(cols) + 3; self.cin = self.ctx_raw.shape[1]

    def set_scaler(self, train_idx):
        """Per-fold input scaling, from TRAINING rows only.  Columns that are constant in the training rows (e.g. the
        shot channels before ESPN published box scores: Série A < 2014, Série B < 2017) are switched off for the fold,
        so that never-trained input weights cannot inject noise at prediction time; the standard deviation used for
        the v1 shot/corner/foul expectations has an a-priori floor for the same reason."""
        x = self.ctx_raw[train_idx]; mu = x.mean(0); sd = x.std(0)
        live = sd > 1e-3
        sd = np.maximum(sd, self.ctx_floor)
        z = np.clip((self.ctx_raw - mu) / sd, -5, 5) * live
        self.ctx = torch.tensor(z.astype(np.float32))
        trow = np.concatenate([2 * train_idx, 2 * train_idx + 1])
        tlive = self.tok[trow].std(0) > 1e-6
        self.tokmask = torch.cat([tlive.float(), torch.ones(3)])           # (+ days-ago, season-gap, pad flag)
        rw = np.ones(self.data.n, np.float32)                              # optional recency weighting of training matches
        if self.cfg['tau'] > 0: rw = np.exp(-(self.data.season[train_idx].max() - self.data.season) / self.cfg['tau']).astype(np.float32)
        self.wfold = self.w * torch.tensor(rw)

    def batch(self, idx):
        out = []
        for s in (0, 1):
            hi = self.hist[idx, s]; m = self.mask[idx, s]
            x = self.tok[hi]
            ago = torch.log1p((self.day[idx].unsqueeze(-1) - self.tok_day[hi]).clamp(min=0)) / 6.0 * m
            sgap = (self.season[idx].unsqueeze(-1) - self.tok_season[hi]).clamp(0, 3) / 3.0 * m
            out += [torch.cat([x, ago.unsqueeze(-1), sgap.unsqueeze(-1), m.unsqueeze(-1)], -1) * self.tokmask, m]
        return out + [self.ctx[idx], self.llh0[idx], self.lla0[idx], self.logp0[idx]]


def fit_predict(fold, cfg, train_idx, test_idx, seeds, seed_base):
    """Trains len(seeds) independent networks (different initialisation, mini-batch order and dropout masks) in one
    batched pass and returns P (S, n_test, 3), LAM (S, n_test, 2)."""
    S = len(seeds)
    torch.manual_seed(seed_base + 7919 * seeds[0]); rngs = [np.random.default_rng(seed_base + 7919 * s) for s in seeds]
    fold.set_scaler(train_idx)
    net = Net(S, cfg, fold.fin, fold.cin, fold.data.grid)
    dec = [p for nme, p in net.named_parameters() if nme.endswith('.W') or '.cw.' in nme]
    no_dec = [p for nme, p in net.named_parameters() if not (nme.endswith('.W') or '.cw.' in nme)]
    opt = torch.optim.AdamW([{'params': dec, 'weight_decay': cfg['wd']}, {'params': no_dec, 'weight_decay': 0.0}], lr=cfg['lr'])
    nb = int(np.ceil(len(train_idx) / cfg['bs'])); steps = cfg['epochs'] * nb
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=steps, pct_start=0.15)
    aux_scale = torch.tensor([14.0, 11.0, 4.8, 3.7])
    net.train()
    for ep in range(cfg['epochs']):
        perm = np.stack([r.permutation(train_idx) for r in rngs])          # every seed has its own order
        for s in range(0, perm.shape[1], cfg['bs']):
            idx = torch.tensor(perm[:, s:s + cfg['bs']])
            p, llh, lla, delta, aux = net(*fold.batch(idx))
            y = fold.y[idx]; w = fold.wfold[idx]
            loss = 0
            if cfg['w_rps']: loss = loss + cfg['w_rps'] * rps_loss(p, y)
            if cfg['w_ce']: loss = loss + cfg['w_ce'] * ce_loss(p, y)
            if cfg['w_pois']: loss = loss + cfg['w_pois'] * (pois_nll(llh, fold.hg[idx]) + pois_nll(lla, fold.ag[idx]))
            loss = ((loss * w).sum(1) / w.sum(1)).sum()                    # sum over seeds of each seed's mean loss
            if cfg['w_aux'] and aux is not None:
                so = fold.stok[idx]
                la_ = (pois_nll(aux, fold.shots[idx]) / aux_scale).sum(-1)
                loss = loss + cfg['w_aux'] * ((la_ * so).sum(1) / so.shape[1]).sum()
            if cfg['delta_pen'] and delta is not None: loss = loss + cfg['delta_pen'] * (delta ** 2).sum(-1).mean(1).sum()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); P, L = [], []
    with torch.no_grad():
        for s in range(0, len(test_idx), 1024):
            idx = torch.tensor(np.tile(test_idx[s:s + 1024], (S, 1)))
            p, llh, lla, _, _ = net(*fold.batch(idx))
            P.append(p.numpy()); L.append(torch.stack([llh, lla], -1).exp().numpy())
    return np.concatenate(P, 1), np.concatenate(L, 1)


# ------------------------------------------------------------------------------------- walk-forward
def run_walk(data, cfg, seeds, seasons=range(2013, 2027), refit_every=1, verbose=True, tag=''):
    """Returns P (S, n, 3), LAM (S, n, 2) filled for matches with season in `seasons` (NaN elsewhere).
    refit_every=1: weights refitted before every season (final protocol).
    refit_every=3: screening protocol, weights refitted before 2013, 2016, 2019, 2022, 2025 and used for the
                   next three seasons (still leak-free: weights for season Y only saw seasons < Y)."""
    cfg = {**DEFAULT, **cfg}
    fold = Fold(data, cfg)
    P = np.full((len(seeds), data.n, 3), np.nan, np.float32); LAM = np.full((len(seeds), data.n, 2), np.nan, np.float32)
    t0 = time.time()
    seasons = list(seasons)
    for Y in seasons[::refit_every]:
        tr = np.where(data.post & (data.season < Y) & (data.season >= cfg['first_season']))[0]
        te = np.where((data.season >= Y) & (data.season < Y + refit_every) & (data.season <= seasons[-1]))[0]
        assert data.season[tr].max() < Y <= data.season[te].min()
        P[:, te], LAM[:, te] = fit_predict(fold, cfg, tr, te, list(seeds), Y)
        if verbose: print(f'  [{tag}] season {Y} done, n_train {len(tr)}, elapsed {time.time() - t0:.0f}s', flush=True)
    return P, LAM


# ------------------------------------------------------------------------------------------ scoring
def rps_np(p, y):
    oh = np.eye(3)[y]
    return ((p[:, 0] - oh[:, 0]) ** 2 + (p[:, 0] + p[:, 1] - oh[:, 0] - oh[:, 1]) ** 2) / 2


def score(data, p, label='', quiet=False, lg='A'):
    """Same arithmetic as shared/score.py (Série A, 2013+, finished).  p: (n,3) aligned with data rows."""
    m = data.post & (data.lgB == (0 if lg == 'A' else 1)) & (data.season >= 2013) & np.isfinite(p[:, 0])
    out = {}
    for nm, mk in [('DEV', m & (data.season <= 2021)), ('HOLD', m & (data.season >= 2022))]:
        if mk.sum() == 0: continue
        q = np.clip(p[mk], 1e-9, 1); q = q / q.sum(1, keepdims=True); y = data.y[mk]
        r = rps_np(q, y); rb = rps_np(data.pv1[mk], y)
        ll = -np.log(q[np.arange(len(y)), y]); lb = -np.log(data.pv1[mk][np.arange(len(y)), y])
        out[nm] = dict(rps=r.mean(), ll=ll.mean(), d=(r - rb).mean(), dll=(ll - lb).mean(), n=int(mk.sum()))
    if not quiet:
        print(f'{label:34s} ' + ' | '.join(f"{k} rps {v['rps']:.5f} ll {v['ll']:.5f} d {v['d']:+.5f} dll {v['dll']:+.5f}" for k, v in out.items()), flush=True)
    return out


def summarize(data, P, label, dev_only=True):
    """Per-seed DEV dRPS mean/sd and seed-average score."""
    per = [score(data, P[s], quiet=True) for s in range(len(P))]
    avg = score(data, np.nanmean(P, 0), quiet=True)
    keys = ['DEV'] if dev_only else ['DEV', 'HOLD']
    txt = []
    for k in keys:
        ds = np.array([x[k]['d'] for x in per])
        txt.append(f"{k}: single-seed dRPS {ds.mean():+.5f} (sd {ds.std(ddof=1) if len(ds) > 1 else 0:.5f}, min {ds.min():+.5f}, max {ds.max():+.5f}) | seed-avg RPS {avg[k]['rps']:.5f} dRPS {avg[k]['d']:+.5f} dLL {avg[k]['dll']:+.5f}")
    print(f'{label:28s} ' + ' || '.join(txt), flush=True)
    return per, avg


def save_run(name, data, P, LAM, cfg):
    keep = data.season >= 2013
    np.savez_compressed(D + f'out/{name}.npz', P=P[:, keep], LAM=LAM[:, keep], ids=data.ids[keep], cfg=json.dumps(cfg, default=str))


def load_run(name, data):
    z = np.load(D + f'out/{name}.npz', allow_pickle=True); keep = data.season >= 2013
    P = np.full((z['P'].shape[0], data.n, 3), np.nan, np.float32); LAM = np.full((z['P'].shape[0], data.n, 2), np.nan, np.float32)
    P[:, keep] = z['P']; LAM[:, keep] = z['LAM']
    return P, LAM
