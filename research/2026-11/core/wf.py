"""Walk-forward driver for engine v2 (eng.ekf2).

A *spec* describes one model variant:
  free   : list of tuned hyper-parameter names ('qm', 'sot:beta', ...)
  fixed  : dict of overrides of eng.DEFAULT
  obs    : function(d, hp, Y) -> (Yobs (n,2), E (n,2))   builds the goal observation (only data < Y may be used for fits)
  sec    : list of secondary channel names (keys of SECDEF) or []
For each season Y the free parameters are tuned on seasons [lo, Y) (Poisson goal log-lik, A + 0.3 B, like v1),
warm-started from the previous season, and one full pass with those parameters is stored.
"""
import math, pickle, time
import numpy as np
import lib, eng

SECDEF = {
    'sot': dict(beta=1.0, P0=0.02, q=2e-4, rho=0.6, sS=0.01, phi=2.0, mu0=math.log(4.3), hfa0=0.22, kMu=0.004, kH=0.004),
    'sh': dict(beta=0.8, P0=0.02, q=4e-4, rho=0.5, sS=0.01, phi=4.0, mu0=math.log(12.5), hfa0=0.2, kMu=0.004, kH=0.004),
    'soff': dict(beta=0.5, P0=0.02, q=4e-4, rho=0.5, sS=0.01, phi=3.0, mu0=math.log(8.2), hfa0=0.2, kMu=0.004, kH=0.004),
    'co': dict(beta=0.6, P0=0.02, q=4e-4, rho=0.5, sS=0.01, phi=2.5, mu0=math.log(5.1), hfa0=0.2, kMu=0.004, kH=0.004),
    'sv': dict(beta=1.0, P0=0.02, q=2e-4, rho=0.6, sS=0.01, phi=2.0, mu0=math.log(3.1), hfa0=0.2, kMu=0.004, kH=0.004),
}
SEASONS = list(range(2013, 2027))
ZERO = None


def sec_obs(d, names):
    SO = np.zeros((d.n, 2, max(1, len(names))))
    for c, nmx in enumerate(names):
        if nmx == 'soff': v = d.st_sh - d.st_sot
        elif nmx == 'sv': v = d.st_sot - np.c_[d.hg, d.ag]      # shots on target that were not goals ("saves")
        else: v = getattr(d, 'st_' + nmx)
        SO[:, :, c] = np.nan_to_num(np.maximum(v, 0))
    return SO


def default_obs(d, hp, Y):
    return np.c_[d.hg, d.ag].astype(float), np.ones((d.n, 2))


class Spec:
    def __init__(self, name, free, fixed=None, obs=default_obs, sec=(), hp0=None, lo=2009, turf=None, covid=None, prior=1e-6):
        self.name = name; self.free = list(free); self.fixed = dict(fixed or {}); self.obs = obs; self.sec = list(sec)
        self.hp0 = dict(hp0 or {}); self.lo = lo; self.turf = turf; self.covid = covid; self.prior = prior


def full_params(spec, x):
    """x (transformed) -> (core dict, list of sec dicts, extra dict of obs hyper-parameters)"""
    core = dict(eng.DEFAULT); core.update({k: v for k, v in spec.fixed.items() if ':' not in k and k in core})
    secs = [dict(SECDEF[s]) for s in spec.sec]
    for k, v in spec.fixed.items():
        if ':' in k and k.split(':')[0] in spec.sec:
            secs[spec.sec.index(k.split(':')[0])][k.split(':')[1]] = v
    extra = {k: v for k, v in spec.fixed.items() if ':' not in k and k not in core}
    for k, xv in zip(spec.free, x):
        v = dec_extra(k, xv)
        if ':' in k: secs[spec.sec.index(k.split(':')[0])][k.split(':')[1]] = v
        elif k in core: core[k] = v
        else: extra[k] = v
    return core, secs, extra


EXTRA_LGT = {'w', 'wRed', 'wPen', 'wOwn', 'wLate', 'wTrail', 'wLead'}


def enc_extra(k, v):
    if k in EXTRA_LGT:
        v = min(0.9999, max(1e-4, v)); return math.log(v / (1 - v))
    return eng.enc1(k, v)


def dec_extra(k, x):
    if k in EXTRA_LGT: return 1 / (1 + math.exp(-x))
    return eng.dec1(k, x)


def start_vector(spec):
    core = dict(eng.DEFAULT); core.update(spec.fixed); 
    x = []
    for k in spec.free:
        if k in spec.hp0: v = spec.hp0[k]
        elif ':' in k: v = SECDEF[k.split(':')[0]][k.split(':')[1]]
        else: v = core[k]
        x.append(enc_extra(k, v))
    return np.array(x)


def run(d, spec, x, Y, SO=None, aux=None):
    core, secs, extra = full_params(spec, x)
    Yo, E = spec.obs(d, extra, Y) if aux is None else spec.obs(d, extra, Y, aux)
    global ZERO
    if ZERO is None or len(ZERO) != d.n: ZERO = np.zeros(d.n, np.bool_)
    turf = ZERO if spec.turf is None else spec.turf
    covid = ZERO if spec.covid is None else spec.covid
    hasS = d.statsOk & d.post if spec.sec else ZERO
    if SO is None: SO = sec_obs(d, spec.sec)
    return eng.ekf2(d.h, d.a, d.lg, d.season, d.post, d.day, Yo, E, d.post, SO, hasS, turf, covid, eng.pvec(core), eng.psvec(secs), d.nteams)


def goal_loss(d, lo, Y):
    mA = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 0)
    mB = d.post & (d.season >= lo) & (d.season < Y) & (d.lg == 1)
    hg = d.hg.astype(float); ag = d.ag.astype(float)

    def f(out):
        lh, la = out[:, 0], out[:, 1]
        if not (np.all(np.isfinite(lh)) and np.all(np.isfinite(la)) and lh.min() > 0 and la.min() > 0):
            return 1e9
        v = lib.pois_ll(hg, lh) + lib.pois_ll(ag, la)
        r = -v[mA].mean() - 0.3 * v[mB].mean()
        return r if np.isfinite(r) else 1e9
    return f


def walk(d, spec, seasons=SEASONS, verbose=True, first_iter=(3, 500), next_iter=(2, 250), save=True, aux_fn=None):
    """returns dict with FULL[Yi] = pass output (n, NOUT) under params tuned on < Y, and per-season params."""
    t0 = time.time()
    x = start_vector(spec); x0 = x.copy()
    SO = sec_obs(d, spec.sec)
    FULL = np.zeros((len(seasons), d.n, eng.NOUT), np.float32); PAR = {}
    for yi, Y in enumerate(seasons):
        L = goal_loss(d, spec.lo, Y)
        aux = aux_fn(d, Y) if aux_fn else None
        f = lambda z: L(run(d, spec, z, Y, SO, aux)) + spec.prior * float(((z - x0) ** 2).sum())
        if len(x):
            r, it = first_iter if yi == 0 else next_iter
            x, best = lib.nm(f, x, rounds=r, maxiter=it)
        else:
            best = f(x)
        out = run(d, spec, x, Y, SO, aux)
        FULL[yi] = out
        core, secs, extra = full_params(spec, x)
        PAR[Y] = dict(core=core, secs=secs, extra=extra, loss=best, x=x.copy())
        if verbose:
            fr = {k: round(dec_extra(k, xv), 5) for k, xv in zip(spec.free, x)}
            print(f'  {spec.name} {Y} loss {best:.5f} {fr}', flush=True)
    res = dict(name=spec.name, seasons=list(seasons), FULL=FULL, PAR=PAR)
    lam = np.zeros((d.n, 2))
    for yi, Y in enumerate(seasons):
        m = d.season == Y
        lam[m] = FULL[yi][m, :2]
    res['lam'] = lam
    sc = d.scored
    p = lib.pois_probs(lam[sc, 0], lam[sc, 1])
    res['dev'] = lib.dev_report(spec.name + ' [indep Poisson]', d, p)
    if verbose: print(f'  ({time.time() - t0:.0f}s)')
    if save:
        pickle.dump(res, open(lib.D + f'out_{spec.name}.pkl', 'wb'))
    return res


def wf_eval(d, res_or_lam, name='', probs=lib.pois_probs):
    lam = res_or_lam['lam'] if isinstance(res_or_lam, dict) else res_or_lam
    sc = d.scored
    return lib.dev_report(name, d, probs(lam[sc, 0], lam[sc, 1]))
