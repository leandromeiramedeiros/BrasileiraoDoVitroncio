"""Walk-forward Poisson adjustment of ctx_core expected goals with coach features.
   /home/user/bdv-lab/venv/bin/python code/fit.py dev            -> DEV table for all variants (never touches HOLDOUT)
   /home/user/bdv-lab/venv/bin/python code/fit.py final NAME     -> preds_coach.csv, coef_coach.json, in-sample table for variant NAME
Offsets: ctx_core lambdas (context/coef/lambda_ctx_core.csv) for Série A 2013-2026; v1 lambdas (shared/baseline_features.csv)
for Série A 2009-2012 (ctx_core is not available there; only used as training rows). Coefficients for season Y: Série A seasons
2009..Y-1, ridge on standardised coefficients (tau), no extra constants (ctx_core is already calibrated)."""
import os, sys, json, numpy as np, pandas as pd
sys.path.insert(0, '/home/user/bdv-lab/context/code')
import fitlib as F
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = '/home/user/bdv-lab/shared/'
TAU = 0.05

def load():
    B = pd.read_csv(SHARED + 'baseline_features.csv', dtype={'id': str}).set_index('id')
    D = B[['lg', 'season', 'status', 'hg', 'ag', 'lh', 'la']].copy()
    L = pd.read_csv('/home/user/bdv-lab/context/coef/lambda_ctx_core.csv', dtype={'id': str}).set_index('id')
    D.loc[L.index, 'lh'] = L.lh; D.loc[L.index, 'la'] = L.la
    X = pd.read_csv(os.path.join(HERE, 'features_coach.csv'), dtype={'id': str}).set_index('id')
    D = D.join(X.drop(columns=['lg', 'season', 'status', 'h', 'a', 'date']))
    D['played'] = (D.status == 'post') & D.hg.notna()
    return D

def d(c, f=lambda x: x):
    return lambda D: (f(D['h_' + c].to_numpy(float)), f(D['a_' + c].to_numpy(float)))
def late(c):      # same feature, only from round ~13 on (team has played >= 12 league matches this season)
    return lambda D: (D['h_' + c].to_numpy(float) * (D.h_mseason.to_numpy() >= 12), D['a_' + c].to_numpy(float) * (D.a_mseason.to_numpy() >= 12))

FEATS = {
    'new1': ('new1', 'diff', d('new1')),
    'new3': ('new3', 'diff', d('new3')),
    'new5': ('new5', 'diff', d('new5')),
    'interim': ('interim', 'diff', d('interim')),
    'sacked7': ('sacked7', 'diff', d('sacked7')),
    'chg7': ('chg7', 'diff', d('chg7')),
    'logm_coach': ('logm_coach', 'diff', d('m_coach', np.log1p)),
    'logm_vac': ('logm_vac', 'diff', d('m_vac', np.log1p)),
    'n_chg': ('n_chg', 'diff', d('n_chg_season')),
    'multi': ('multi', 'diff', d('n_chg_season', lambda x: (x >= 2).astype(float))),
    'new5_bad': ('new5_bad', 'diff', lambda D: tuple(D[s + 'new5'].to_numpy(float) * (D[s + 'ppg5'].to_numpy(float) < 1.0) for s in ('h_', 'a_'))),
    'new5_late': ('new5_late', 'diff', late('new5')),
    'interim_late': ('interim_late', 'diff', late('interim')),
    'turmoil': ('turmoil', 'diff', lambda D: tuple(np.fmax(D[s + 'new1'].to_numpy(float), D[s + 'interim'].to_numpy(float)) for s in ('h_', 'a_'))),
    'turmoil_hfa': ('turmoil_hfa', 'pair', lambda D: (-np.fmax(D['a_new1'].to_numpy(float), D['a_interim'].to_numpy(float)), -np.fmax(D['h_new1'].to_numpy(float), D['h_interim'].to_numpy(float)))),
}
VARIANTS = {
    'new1': ['new1'], 'new3': ['new3'], 'new5': ['new5'], 'interim': ['interim'], 'sacked7': ['sacked7'], 'chg7': ['chg7'],
    'logm_coach': ['logm_coach'], 'logm_vac': ['logm_vac'], 'n_chg': ['n_chg'], 'multi': ['multi'], 'new5_bad': ['new5_bad'],
    'new3+interim': ['new3', 'interim'], 'new5+n_chg': ['new5', 'n_chg'], 'new3+logm_vac': ['new3', 'logm_vac'],
    'new1+new5+interim+n_chg': ['new1', 'new5', 'interim', 'n_chg'],
    'new1+interim': ['new1', 'interim'], 'turmoil': ['turmoil'], 'interim_late': ['interim_late'], 'turmoil+new5': ['turmoil', 'new5'],
    'all': ['new1', 'new3', 'new5', 'interim', 'sacked7', 'logm_coach', 'logm_vac', 'n_chg', 'new5_bad'],
}

def spec_of(names): return [FEATS[n] for n in names]

def run(D, names, tau=TAU):
    L, path = F.walk_forward(D, spec_of(names), tau=tau, start=2009, leagues=('A',), seasons=range(2013, 2027), consts=False)
    return L, path

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dev'
    D = load()
    ctx = pd.read_csv('/home/user/bdv-lab/context/preds_ctx_core.csv', dtype={'id': str})
    os.makedirs(os.path.join(HERE, 'variants'), exist_ok=True)
    if mode == 'dev':
        res = []
        only = sys.argv[2:] or list(VARIANTS)
        for v in only:
            L, path = run(D, VARIANTS[v])
            P = F.to_probs(L, os.path.join(HERE, 'variants', f'preds_{v}.csv'))
            r = F.score_dev(P, v, ref=ctx); r['coef_2021'] = json.dumps({k: round(x, 4) for k, x in path.loc[2021].items()}); res.append(r)
        pd.DataFrame(res).to_csv(os.path.join(HERE, 'variants', 'dev_table.csv'), index=False, mode='a', header=not os.path.exists(os.path.join(HERE, 'variants', 'dev_table.csv')))
    elif mode == 'final':
        v = sys.argv[2]; names = VARIANTS[v]
        L, path = run(D, names)
        L.to_csv(os.path.join(HERE, 'lambda_coach.csv'), index=False)
        P = F.to_probs(L, os.path.join(HERE, 'preds_coach_all.csv'))
        scored = P.merge(F._truth()[0][['id']], on='id')
        scored.to_csv(os.path.join(HERE, 'preds_coach.csv'), index=False)
        print('preds_coach.csv rows:', len(scored))
        ins = F.insample(D, spec_of(names), max_season=2026, start=2009, tau=1.0).drop(['c_tot', 'c_hfa'])
        ins_nc = insample_noconst(D, names, 2009, 2026)
        print('in-sample 2009-2026 (Série A; with free constants):\n', ins.round(4).to_string())
        print('in-sample 2009-2026 (Série A; no constants, as used in production):\n', ins_nc.round(4).to_string())
        prod = path.loc[2026].to_dict()
        # production coefficients for 2027+: fitted on all played Série A 2009-2026
        Lp, pathp = F.walk_forward(D, spec_of(names), tau=TAU, start=2009, leagues=('A',), seasons=[2027], consts=False)
        json.dump({'features': names, 'kind': 'diff: log lh += b*(x_home - x_away); log la -= b*(x_home - x_away)',
                   'coef_for_2026_fit_2009_2025': prod, 'coef_for_2027_fit_2009_2026': pathp.loc[2027].to_dict(),
                   'path': {int(k): v for k, v in path.T.to_dict().items()}, 'tau_std': TAU,
                   'insample_2009_2026': ins_nc.to_dict(orient='index')}, open(os.path.join(HERE, 'coef_coach.json'), 'w'), indent=1)
        print('walk-forward path:\n', path.round(4).to_string())

def insample_noconst(D, names, s0, s1):
    tr = (D.played & (D.lg == 'A') & (D.season >= s0) & (D.season <= s1)).to_numpy()
    Xh, Xa, nm = F.prep(D, spec_of(names), tr)
    X = np.vstack([Xh[tr], Xa[tr]]); y = np.concatenate([D.hg.to_numpy(float)[tr], D.ag.to_numpy(float)[tr]])
    off = np.concatenate([np.log(D.lh.to_numpy())[tr], np.log(D.la.to_numpy())[tr]])
    b, se = F.fit_poisson(X, y, off, np.full(len(nm), 1e-6))
    return pd.DataFrame({'coef': b, 'se': se, 'z': b / se, 'mean_abs_x': np.abs(X).mean(0), 'n_nonzero_rows': (X != 0).sum(0)}, index=nm)

if __name__ == '__main__':
    main()
