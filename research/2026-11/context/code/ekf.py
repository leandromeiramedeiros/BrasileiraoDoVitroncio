"""All-competitions Poisson EKF attack/defence rating (goals only) on ONE absolute scale.

Re-implementation of the 'g' channel of v1 (site/js/engine/ratings.js) with three changes:
  * ratings live on an absolute scale (tier means: A 0, B -delta, C -2 delta, ...) so that teams from
    different divisions/countries can meet in cups without a per-league re-centering;
  * every competition has its own level (mu) and home advantage (hfa), learned online;
  * every competition has an observation weight w (dispersion phi / w): cup and state matches are noisier.
No team-specific home factor, no xG blending, no red-card adjustment (kept deliberately simple: what is
used downstream is the DIFFERENCE between two runs of this same engine, league-only vs. league+extras).

All hyper-parameters are fixed a priori (v1's published 'g' values rounded; weights 1 or as given).
"""
import math, collections
import numpy as np, pandas as pd

V1 = dict(q=3.9e-5, P0=0.0192, rho=0.866, phi=1.0, kMu=0.0023, kH=0.00038, mu0=math.log(1.25), hfa0=0.35)
DELTA = 0.30                      # tier gap per component (v1: dPromoO .35 / dPromoD .29)
TIER = {'A': 0.0, 'B': -1.0, 'C': -2.0, 'S': -2.5, 'F': -0.5}   # in units of DELTA
FAMILY = {'A': 'LG', 'B': 'LG', 'C': 'C', 'D': 'C', 'SP': 'ST', 'RJ': 'ST', 'RS': 'ST', 'MG': 'ST', 'CNE': 'ST',
          'CDB': 'CUP', 'LIB': 'CONT', 'SUD': 'CONT', 'REC': 'CONT', 'CWC': 'OTH', 'ICC': 'OTH', 'SUP': 'OTH'}
DOMESTIC = {'A', 'B', 'C', 'D', 'SP', 'RJ', 'RS', 'MG', 'CDB', 'CNE', 'SUP'}

def is_neutral(comp, stage, season):
    if comp in ('CWC', 'ICC', 'SUP'): return True
    s = stage or ''
    if comp in ('LIB', 'SUD') and season >= 2019 and 'final' in s and 'semi' not in s and 'quarter' not in s: return True
    return False

def divisions(M):
    """(team, season) -> 'A'|'B'|'C' from league files; set of Brazilian teams."""
    div = {}
    for comp in ('C', 'B', 'A'):
        s = M[M.comp == comp]
        for t, y in set(zip(s.h, s.season)) | set(zip(s.a, s.season)): div[(t, y)] = comp
    dom = M[M.comp.isin(DOMESTIC)]
    br = set(dom.h) | set(dom.a)
    return div, br

def run(M, weights, hp=V1, delta=DELTA, p0_mult_new=3.0, record=('A', 'B'), reset_at_league=False):
    """M: merged stream (DataFrame from common.load_all()), weights: family -> w (0 = ignore the competition).
    Returns DataFrame indexed by match id with the PRE-MATCH prediction for the recorded competitions."""
    div, br = divisions(M)
    q, P0, rho, phi0 = hp['q'], hp['P0'], hp['rho'], hp['phi']
    teams, lev = {}, {}
    out = []
    cols = ['id', 'comp', 'season', 'h', 'a', 'status', 'hg', 'ag', 'stage']
    for id_, comp, season, h, a, status, hg, ag, stage in M[cols].itertuples(index=False, name=None):
        w = weights.get(FAMILY[comp], 0.0)
        if w <= 0: continue
        L = lev.get(comp)
        if L is None: L = lev[comp] = [hp['mu0'], hp['hfa0'] if not is_neutral(comp, stage, season) else 0.0, 0]
        st = []
        for t in (h, a):
            T = teams.get(t)
            tier_now = div.get((t, season)) or ('S' if t in br else 'F')
            if T is None:
                if comp == 'A' and season <= 2007: m0 = 0.0
                elif comp in ('A', 'B'): m0 = -delta
                else: m0 = TIER[tier_now] * delta
                p = P0 if comp in ('A', 'B') else P0 * p0_mult_new
                T = teams[t] = {'o': m0, 'd': m0, 'Po': p, 'Pd': p, 'season': season, 'tier': tier_now, 'n': 0, 'nlg': 0}
            elif season > T['season']:
                gap = season - T['season']
                m_old = TIER[T['tier']] * delta
                r = rho ** gap
                T['o'] = m_old + r * (T['o'] - m_old); T['d'] = m_old + r * (T['d'] - m_old)
                T['Po'] = T['Pd'] = P0 if (T['tier'] in ('A', 'B') or tier_now in ('A', 'B')) else P0 * p0_mult_new
                if gap > 1 and comp in ('A', 'B'):
                    T['o'] = min(T['o'], -delta); T['d'] = min(T['d'], -delta)
                T['season'] = season; T['tier'] = tier_now; T['n'] = 0; T['nlg'] = 0
            st.append(T)
        H, A = st
        if reset_at_league and comp in ('A', 'B'):
            # optional: forget the precision gained in pre-league matches (keep the mean shift) at the first league match
            for T in st:
                if T['nlg'] == 0: T['Po'] = max(T['Po'], P0); T['Pd'] = max(T['Pd'], P0)
        neutral = is_neutral(comp, stage, season)
        hfa = 0.0 if neutral else L[1]
        lh = math.exp(L[0] + hfa / 2 + H['o'] - A['d']); la = math.exp(L[0] - hfa / 2 + A['o'] - H['d'])
        if comp in record:
            out.append((id_, lh, la, H['o'], H['d'], A['o'], A['d'], H['Po'] + A['Pd'], A['Po'] + H['Pd'], H['n'], A['n'], H['nlg'], A['nlg'], L[0], L[1]))
        if status != 'post' or hg is None or ag is None or hg != hg: continue
        phi = phi0 / w
        H['Po'] += q; H['Pd'] += q; A['Po'] += q; A['Pd'] += q
        eh, ea = hg - lh, ag - la
        den = lh * (H['Po'] + A['Pd']) + phi
        H['o'] += H['Po'] * eh / den; A['d'] -= A['Pd'] * eh / den
        H['Po'] -= H['Po'] ** 2 * lh / den; A['Pd'] -= A['Pd'] ** 2 * lh / den
        den = la * (A['Po'] + H['Pd']) + phi
        A['o'] += A['Po'] * ea / den; H['d'] -= H['Pd'] * ea / den
        A['Po'] -= A['Po'] ** 2 * la / den; H['Pd'] -= H['Pd'] ** 2 * la / den
        kM = max(hp['kMu'], 1 / (L[2] + 30)); kH = max(hp['kH'], 1 / (L[2] + 30))
        L[0] += kM * (eh / lh + ea / la) / 2
        if not neutral: L[1] += kH * (eh / lh - ea / la) / 2
        L[2] += 1
        H['n'] += 1; A['n'] += 1
        if comp in ('A', 'B'): H['nlg'] += 1; A['nlg'] += 1
    R = pd.DataFrame(out, columns=['id', 'lh', 'la', 'oh', 'dh', 'oa', 'da', 'Sh', 'Sa', 'nh_all', 'na_all', 'nh_lg', 'na_lg', 'mu', 'hfa']).set_index('id')
    R.attrs['levels'] = {k: tuple(v) for k, v in lev.items()}
    return R

RUNS = {  # name -> family weights
    'lg':   {'LG': 1},
    'st':   {'LG': 1, 'ST': 1},
    'cup':  {'LG': 1, 'C': 1, 'CUP': 1},
    'cont': {'LG': 1, 'CONT': 1, 'OTH': 1},
    'all':  {'LG': 1, 'C': 1, 'ST': 1, 'CUP': 1, 'CONT': 1, 'OTH': 1},
    'allw': {'LG': 1, 'C': 1, 'ST': 0.5, 'CUP': 0.5, 'CONT': 0.5, 'OTH': 0.5},
    'opt':  {'LG': 1, 'C': 1, 'ST': 0.25, 'CUP': 0.5, 'CONT': 0.5, 'OTH': 0.5},   # weights suggested by the DEV coefficients of the weight-1 runs
    'st_reset':  {'LG': 1, 'ST': 1, '_reset': True},
    'opt_reset': {'LG': 1, 'C': 1, 'ST': 0.25, 'CUP': 0.5, 'CONT': 0.5, 'OTH': 0.5, '_reset': True},
}

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import common as C
    M = C.load_all()
    res = {k: run(M, {a: b for a, b in w.items() if a != '_reset'}, reset_at_league=bool(w.get('_reset'))) for k, w in RUNS.items()}
    base = res['lg']
    F = pd.DataFrame(index=base.index)
    F['e_lh_lg'] = np.log(base.lh); F['e_la_lg'] = np.log(base.la)
    for k, R in res.items():
        if k == 'lg': continue
        R = R.reindex(base.index)
        F[f'r_dh_{k}'] = np.log(R.lh) - np.log(base.lh)     # change in log expected home goals when the extra matches are used
        F[f'r_da_{k}'] = np.log(R.la) - np.log(base.la)
        # strength-only part (levels/hfa of the league removed): (o_h - d_a) and (o_a - d_h)
        F[f'r_sh_{k}'] = (R.oh - R.da) - (base.oh - base.da)
        F[f'r_sa_{k}'] = (R.oa - R.dh) - (base.oa - base.dh)
        F[f'r_nh_{k}'] = R.nh_all - R.nh_lg; F[f'r_na_{k}'] = R.na_all - R.na_lg   # extra matches seen this season
    F.to_csv(os.path.join(C.HERE, 'features_ratings.csv'))
    print(F.describe().T[['mean', 'std', 'min', 'max']].round(4))
    print({k: (round(v[0], 3), round(v[1], 3), v[2]) for k, v in res['all'].attrs['levels'].items()})
    # sanity: log-likelihood of the engine's own predictions on Série A, by run
    S = pd.read_csv(C.SHARED + 'baseline_features.csv', dtype={'id': str}).set_index('id')
    S = S[(S.lg == 'A') & (S.season >= 2013) & (S.status == 'post')]
    from scipy.special import gammaln
    def ll(lh, la, d): return -np.mean(d.hg * np.log(lh) - lh + d.ag * np.log(la) - la - gammaln(d.hg + 1) - gammaln(d.ag + 1))
    for lab, msk in [('DEV', S.season <= 2021), ('early DEV (first 8 rounds)', (S.season <= 2021) & (np.maximum(S.nh, S.na) < 8))]:
        d = S[msk]; print(lab, 'n=', len(d), ' v1 poisson NLL', round(ll(d.lh, d.la, d), 5), {k: round(ll(R.lh.reindex(d.index), R.la.reindex(d.index), d), 5) for k, R in res.items()})
