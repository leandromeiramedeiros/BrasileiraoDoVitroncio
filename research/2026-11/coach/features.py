"""Pre-match coach features for every Série A/B match in shared/matches.json.
   /home/user/bdv-lab/venv/bin/python code/features.py     (run after parse.py)
Writes features_coach.csv (id + home/away features, all matches incl. future), future_coach.csv (unplayed 2026).
Leak rule: only changes dated STRICTLY BEFORE the local (UTC-3) match date are used."""
import json, os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARED = '/home/user/bdv-lab/shared/'
CANON = {'5266': '3395', '6080': '3448', '6274': '4929'}
CAP_M, CAP_D = 38, 365

def load_changes():
    C = pd.read_csv(os.path.join(HERE, 'coach_changes.csv'), dtype={'team_id': str})
    C['v'] = pd.to_datetime(C.date_vacancy).dt.date
    C['a'] = pd.to_datetime(C.date_appointment).dt.date
    C['a'] = [a if isinstance(a, type(v)) and not pd.isna(a) and a >= v else v for a, v in zip(C.a, C.v)]
    C = C.dropna(subset=['v']).sort_values(['team_id', 'v', 'a']).reset_index(drop=True)
    # incoming coach is a caretaker if flagged, or if his own departure is later labelled "end of interim spell"
    C['inc_interim'] = C.interim.fillna(0).astype(int)
    for t, g in C.groupby('team_id'):
        idx = list(g.index)
        for i, j in zip(idx[:-1], idx[1:]):
            if (C.at[j, 'manner'] == 'interim end' or C.at[j, 'out_interim'] == 1) and str(C.at[j, 'outgoing']).strip() == str(C.at[i, 'incoming']).strip():
                C.at[i, 'inc_interim'] = 1
    # a "real" departure = a permanent coach leaving (not a caretaker handing over)
    C['real'] = ((C.manner != 'interim end') & (C.out_interim.fillna(0) == 0)).astype(int)
    C['sacked'] = (C.manner.isin(['sacked', 'mutual']) & (C.real == 1)).astype(int)
    return C

def main():
    M = pd.DataFrame(json.load(open(SHARED + 'matches.json')))
    M['id'] = M['id'].astype(str); M['h'] = M.h.astype(str).replace(CANON); M['a'] = M.a.astype(str).replace(CANON)
    M['d'] = (pd.to_datetime(M.date, utc=True) - pd.Timedelta(hours=3)).dt.date
    C = load_changes()
    cov = set(map(tuple, C[['lg', 'season']].drop_duplicates().to_numpy()))        # (lg, season) with a parsed table
    # team-match long table
    L = pd.concat([M[['id', 'lg', 'season', 'd', 'status', 'h', 'hg', 'ag']].rename(columns={'h': 'team', 'hg': 'gf', 'ag': 'ga'}).assign(side='h'),
                   M[['id', 'lg', 'season', 'd', 'status', 'a', 'ag', 'hg']].rename(columns={'a': 'team', 'ag': 'gf', 'hg': 'ga'}).assign(side='a')])
    L = L.sort_values(['team', 'd', 'id']).reset_index(drop=True)
    L['pts'] = np.where(L.gf > L.ga, 3, np.where(L.gf == L.ga, 1, 0)).astype(float); L.loc[L.gf.isna(), 'pts'] = np.nan
    first = L.groupby(['team', 'lg', 'season']).d.transform('min')
    L['season_start'] = first
    rows = []
    for t, g in L.groupby('team', sort=False):
        ev = C[C.team_id == t]
        V = np.array(ev.v.tolist(), dtype='datetime64[D]') if len(ev) else np.array([], dtype='datetime64[D]')
        A = np.array(ev.a.tolist(), dtype='datetime64[D]') if len(ev) else np.array([], dtype='datetime64[D]')
        inc_int = ev.inc_interim.to_numpy(); real = ev.real.to_numpy(); sack = ev.sacked.to_numpy()
        dates = np.array(g.d.tolist(), dtype='datetime64[D]')
        pts = g.pts.to_numpy()
        for k, (i, r) in enumerate(g.iterrows()):
            d = dates[k]; ss = np.datetime64(r.season_start)
            covered = (r.lg, r.season) in cov
            prior = np.where(V < d)[0]
            f = dict(id=r.id, side=r.side, covered=int(covered))
            f['mseason'] = int(((dates < d) & (dates >= ss)).sum())     # league matches already played this season
            # form: points per game over the team's previous 5 league matches (any league), for the "bad run" interaction
            prev = pts[max(0, k - 5):k]; prev = prev[~np.isnan(prev)]
            f['ppg5'] = prev.mean() if len(prev) >= 3 else np.nan
            if len(prior) == 0:
                f.update(days_coach=CAP_D, m_coach=CAP_M, days_vac=CAP_D, m_vac=CAP_M, interim=0, n_chg_season=0, sacked7=0, chg7=0)
            else:
                j = prior[-1]
                start = A[j] if A[j] < d else V[j]
                interim = int(A[j] >= d or inc_int[j] == 1)
                rj = [x for x in prior if real[x] == 1]
                vac = V[rj[-1]] if rj else None
                f['days_coach'] = min(int((d - start).astype(int)), CAP_D)
                f['m_coach'] = min(int(((dates > start) & (dates < d)).sum()), CAP_M)
                f['days_vac'] = min(int((d - vac).astype(int)), CAP_D) if vac is not None else CAP_D
                f['m_vac'] = min(int(((dates > vac) & (dates < d)).sum()), CAP_M) if vac is not None else CAP_M
                f['interim'] = interim
                f['n_chg_season'] = int(sum(1 for x in rj if V[x] >= ss))
                f['sacked7'] = int(any((d - V[x]).astype(int) <= 7 and sack[x] for x in rj))
                f['chg7'] = int(any((d - V[x]).astype(int) <= 7 for x in rj))
            for kk in (1, 3, 5):
                f[f'new{kk}'] = int(f['m_vac'] < kk)            # first kk league matches since a permanent coach left
            if not covered:                                     # no table for this league-season: unknown, not "no change"
                for c in ['days_coach', 'm_coach', 'days_vac', 'm_vac', 'interim', 'n_chg_season', 'sacked7', 'chg7', 'new1', 'new3', 'new5']: f[c] = np.nan
            rows.append(f)
    F = pd.DataFrame(rows)
    W = F[F.side == 'h'].drop(columns='side').set_index('id').add_prefix('h_').join(F[F.side == 'a'].drop(columns='side').set_index('id').add_prefix('a_'))
    W = M[['id', 'lg', 'season', 'date', 'status', 'h', 'a']].set_index('id').join(W)
    W.to_csv(os.path.join(HERE, 'features_coach.csv'))
    fut = W[(W.status != 'post') & (W.season == 2026)]
    fut.to_csv(os.path.join(HERE, 'future_coach.csv'))
    print('features:', W.shape, ' future:', fut.shape)
    print('coverage share by lg:', W.groupby('lg').h_covered.mean().round(3).to_dict())
    print(W[W.lg == 'A'].groupby('season')[['h_new1', 'h_new3', 'h_interim', 'h_sacked7', 'h_n_chg_season']].mean().round(3).to_string())

if __name__ == '__main__':
    main()
