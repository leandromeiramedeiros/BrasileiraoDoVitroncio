"""Walk-forward Poisson shift on top of ctx_core with the missing-player features (features_absent.csv, from build.mjs).
  log λ_home += Σ b_j (x_j,home − x_j,away),  log λ_away −= the same   (null for either team → 0)
Coefficients for season Y: ridge Poisson with offsets on Série A seasons START..Y−1 (offset = ctx_core expected goals
2013+, v1 expected goals before). Usage:
  python fit.py dev                 DEV table of variants (never reads HOLDOUT)
  python fit.py final F1,F2,...     preds_absent.csv (+ preds_absent_lineup.csv on top of the announced-lineup model),
                                    coef_absent.json (production fit 2009–2026), future shifts
  python fit.py ref                 referee test (DEV only)
"""
import sys, os, json, gzip, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, '/home/user/bdv-lab/context/code')
import fitlib as F

HERE = os.path.dirname(os.path.abspath(__file__))
SH = '/home/user/bdv-lab/shared/'
CTX = '/home/user/bdv-lab/context/'
LJS = '/home/user/bdv-lab/players/js/'
TAU = 0.05            # ridge prior sd of the per-sd effect (light)


def load(lam_path=CTX + 'coef/lambda_ctx_core.csv'):
    M = pd.DataFrame(json.load(open(SH + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag', 'date']]
    M['id'] = M['id'].astype(str)
    M = M[M.lg == 'A'].set_index('id')
    v1 = pd.read_csv(SH + 'baseline_features.csv', dtype={'id': str}).set_index('id')[['lh', 'la']]
    lc = pd.read_csv(lam_path, dtype={'id': str}).set_index('id')
    D = M.join(v1, how='left')
    D['inctx'] = D.index.isin(lc.index)
    D.loc[lc.index.intersection(D.index), ['lh', 'la']] = lc.loc[lc.index.intersection(D.index), ['lh', 'la']].to_numpy()
    X = pd.read_csv(os.path.join(HERE, 'features_absent.csv'), dtype={'id': str}).set_index('id').drop(columns=['lg', 'season'])
    D = D.join(X, how='left')
    for s in 'ha':   # single "how much of the usual team is unavailable" measures (absent part null → suspensions only)
        for nm, ab in [('unav1', 'abs1_w'), ('unav2', 'abs2_w'), ('unavP', 'absP_w')]:
            D[f'{s}_{nm}'] = D[f'{s}_susp_w'] + D[f'{s}_{ab}'].fillna(0)
        D[f'{s}_unavg1'] = D[f'{s}_susp_g'] + D[f'{s}_abs1_g'].fillna(0)
    D['played'] = (D.status == 'post') & D.hg.notna()
    return D


def dcol(D, f):
    h, a = D['h_' + f].to_numpy(float), D['a_' + f].to_numpy(float)
    d = h - a
    return np.where(np.isnan(d), 0.0, d)


def walk(D, feats, start=2009, tau=TAU, seasons=range(2013, 2027)):
    X = np.column_stack([dcol(D, f) for f in feats])
    offh, offa = np.log(D.lh.to_numpy()), np.log(D.la.to_numpy())
    yh, ya = D.hg.to_numpy(float), D.ag.to_numpy(float)
    season, played, inctx = D.season.to_numpy(), D.played.to_numpy(), D.inctx.to_numpy()
    out, path = [], {}
    for Y in seasons:
        tr = played & (season < Y) & (season >= start) & ~np.isnan(offh)
        te = inctx & (season == Y)
        Xt = np.vstack([X[tr], -X[tr]]); yt = np.concatenate([yh[tr], ya[tr]]); ot = np.concatenate([offh[tr], offa[tr]])
        sd = Xt.std(0); sd = np.where(sd < 1e-12, 1, sd)
        if tr.sum() >= 300:
            b, se = F.fit_poisson(Xt / sd, yt, ot, np.full(len(feats), 1 / tau ** 2)); b = b / sd
        else:
            b = np.zeros(len(feats))
        path[Y] = b
        e = X[te] @ b
        out.append(pd.DataFrame({'id': D.index[te], 'lh': np.exp(offh[te] + e), 'la': np.exp(offa[te] - e)}))
    return pd.concat(out, ignore_index=True), path


def probs(L, path):
    L.to_csv(path + '.lam.csv', index=False)
    subprocess.run(['node', CTX + 'code/probs.mjs', path + '.lam.csv', path], check=True)
    os.unlink(path + '.lam.csv')
    return pd.read_csv(path, dtype={'id': str})


def dev():
    D = load()
    ref = pd.read_csv(CTX + 'preds_ctx_core.csv', dtype={'id': str})
    tmp = os.path.join(HERE, 'work'); os.makedirs(tmp, exist_ok=True)
    # coverage / scale of the features (Série A 2013–2021)
    d = D[(D.season >= 2013) & (D.season <= 2021)]
    for f in ['susp_w', 'susp_g', 'abs1_w', 'abs1_g', 'abs2_w', 'absP_w']:
        v = pd.concat([d['h_' + f], d['a_' + f]])
        print(f'{f:7s} non-null {v.notna().mean():.3f} mean {v.mean():.4f} sd {v.std():.4f} share>0 {(v > 0).mean():.3f}')
    V = [('susp_w', ['susp_w']), ('susp_g', ['susp_g']), ('abs1_w', ['abs1_w']), ('abs1_g', ['abs1_g']), ('abs2_w', ['abs2_w']),
         ('absP_w', ['absP_w']), ('susp_w+abs1_w', ['susp_w', 'abs1_w']), ('susp_w+absP_w', ['susp_w', 'absP_w']),
         ('susp_w+abs2_w', ['susp_w', 'abs2_w']), ('susp_wg+abs1_wg', ['susp_w', 'susp_g', 'abs1_w', 'abs1_g']),
         ('susp_g+abs1_g', ['susp_g', 'abs1_g'])]
    if len(sys.argv) > 2: V = [(s, s.split('+')) for s in sys.argv[2].split(';')]
    rows = []
    for name, feats in V:
        for start in ([2009, 2013] if len(sys.argv) <= 3 else [int(sys.argv[3])]):
            L, path = walk(D, feats, start=start)
            P = probs(L, os.path.join(tmp, 'p.csv'))
            r = F.score_dev(P, f'{name} start{start}', ref=ref)
            r['b2021'] = ' '.join(f'{x:+.3f}' for x in path[2021]); rows.append(r)
            print('    coefs fitted through 2020 (used for 2021):', r['b2021'], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(tmp, 'dev_table.csv'), index=False)


def prod_fit(D, feats, start=2009):
    tr = (D.played & (D.season >= start) & D.lh.notna()).to_numpy()
    X = np.column_stack([dcol(D, f) for f in feats])[tr]
    Xt = np.vstack([X, -X]); yt = np.concatenate([D.hg.to_numpy(float)[tr], D.ag.to_numpy(float)[tr]])
    ot = np.concatenate([np.log(D.lh.to_numpy()[tr]), np.log(D.la.to_numpy()[tr])])
    sd = Xt.std(0)
    b, se = F.fit_poisson(Xt / sd, yt, ot, np.full(len(feats), 1 / TAU ** 2))
    b0, se0 = F.fit_poisson(Xt / sd, yt, ot, np.full(len(feats), 1e-8))   # unpenalised, for the SE / z
    return b / sd, se / sd, b0 / sd, se0 / sd, int(tr.sum()), X.std(0)


def final(feats):
    feats = feats.split(',')
    D = load()
    L, path = walk(D, feats)
    probs(L, os.path.join(HERE, 'preds_absent.csv'))
    L.to_csv(os.path.join(HERE, 'lambda_absent.csv'), index=False)
    # same shift on top of ctx_core + announced lineup (coefficients refitted with that offset)
    DL = load(LJS + 'lambda_lineup_js.csv')
    LL, pathL = walk(DL, feats)
    probs(LL, os.path.join(HERE, 'preds_absent_lineup.csv'))
    b, se, b0, se0, n, sdx = prod_fit(D, feats)
    J = {'features': feats, 'type': 'diff', 'b': [round(x, 5) for x in b], 'se': [round(x, 5) for x in se],
         'b_unpenalised': [round(x, 5) for x in b0], 'se_unpenalised': [round(x, 5) for x in se0], 'sd_diff': [round(x, 4) for x in sdx],
         'ridge_tau_per_sd': TAU, 'fit': 'Serie A 2009-2026 (offset ctx_core 2013+, v1 before)', 'n': n,
         'walk_forward_path': {int(y): [round(x, 4) for x in v] for y, v in path.items()},
         'rule': 'log lambda_home += sum_j b_j*(x_j_home - x_j_away); log lambda_away -= the same; null for either team -> 0. '
                 'x from absent.mjs absentFeatures(teamSnapshot(lineupState, team, date), lineupState.teams[team], cards, team, lg, season, date)'}
    json.dump(J, open(os.path.join(HERE, 'coef_absent.json'), 'w'), indent=1)
    print(json.dumps({k: J[k] for k in ['features', 'b', 'se', 'b_unpenalised', 'se_unpenalised', 'sd_diff', 'n']}))
    # future fixtures: production shift
    Fu = pd.read_csv(os.path.join(HERE, 'future_absent.csv'), dtype={'id': str})
    e = sum(b[j] * (Fu['h_' + f] - Fu['a_' + f]).fillna(0) for j, f in enumerate(feats))
    Fu['shift_absent'] = e.round(5)
    Fu.to_csv(os.path.join(HERE, 'future_absent.csv'), index=False)
    print('future fixtures', len(Fu), 'shift sd', round(float(e.std()), 4))


def referee():
    """Referee tendencies: walk-forward shrunk residuals (goals minus expected) over all earlier A+B matches."""
    refs = {}
    with gzip.open('/home/user/bdv-lab/players/data/rosters.jsonl.gz', 'rt') as fh:
        for line in fh:
            r = json.loads(line); refs[str(r['id'])] = (r.get('ref') or '').strip()
    M = pd.DataFrame(json.load(open(SH + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
    M['id'] = M['id'].astype(str)
    v1 = pd.read_csv(SH + 'baseline_features.csv', dtype={'id': str}).set_index('id')[['lh', 'la']]
    lc = pd.read_csv(CTX + 'coef/lambda_ctx_core.csv', dtype={'id': str}).set_index('id')
    print('matches with a referee name:', sum(1 for m in M.id if refs.get(m)), 'of', len(M))
    K = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    st = {}; rows = []
    for m in M.itertuples():
        ref = refs.get(m.id, '')
        lam = lc.loc[m.id] if m.id in lc.index else (v1.loc[m.id] if m.id in v1.index else None)
        s = st.get(ref, [0.0, 0.0, 0.0, 0.0]) if ref else [0.0, 0.0, 0.0, 0.0]
        rows.append((m.id, s[0] / (s[3] + K), s[1] / (s[3] + K), s[2] / (s[3] + K), s[3]))
        if m.status == 'post' and ref and lam is not None and m.hg == m.hg:
            s = st.setdefault(ref, [0.0, 0.0, 0.0, 0.0])
            rh, ra = m.hg - lam.lh, m.ag - lam.la
            hw = float(m.hg > m.ag) - (0.0)   # raw home-win indicator (centred below)
            s[0] += rh - ra; s[1] += rh + ra; s[2] += hw; s[3] += 1
    R = pd.DataFrame(rows, columns=['id', 'ref_hadv', 'ref_tot', 'ref_hw', 'ref_n']).set_index('id')
    R['ref_hw'] = R['ref_hw'] - R['ref_n'] / (R['ref_n'] + K) * 0.45   # centre the home-win rate (shrunk to ~league mean)
    D = load().join(R, how='left')
    ref = pd.read_csv(CTX + 'preds_ctx_core.csv', dtype={'id': str})
    tmp = os.path.join(HERE, 'work'); os.makedirs(tmp, exist_ok=True)
    R.to_csv(os.path.join(tmp, f'features_referee_K{int(K)}.csv'))
    for name, spec in [('ref_hadv (hfa)', [('ref_hadv', 'hfa')]), ('ref_tot (tot)', [('ref_tot', 'tot')]),
                       ('ref_hadv+ref_tot', [('ref_hadv', 'hfa'), ('ref_tot', 'tot')]), ('ref_hw (hfa)', [('ref_hw', 'hfa')])]:
        sp = [(n, k, (lambda col: (lambda d: d[col].fillna(0).to_numpy()))(n)) for n, k in spec]
        L, path = F.walk_forward(D.assign(lg='A'), sp, tau=TAU, consts=False)
        P = probs(L, os.path.join(tmp, 'pr.csv'))
        F.score_dev(P, f'{name} K{int(K)}', ref=ref)
        print('    coefs used for 2021:', path.loc[2021].round(4).to_dict())


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'dev': dev()
    elif cmd == 'final': final(sys.argv[2])
    elif cmd == 'ref': referee()
