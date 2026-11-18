#!/usr/bin/env python
"""Step 3 + 6a - per-match team xG quantities and per-player-per-match xG.

Input : work/shots_xg.pkl (walk-forward valued shots), work/owngoals.pkl, players/compact rosters, shared/matches.json
Output: features_xg_match.csv   (POST-match quantities of the match itself: inputs for ratings, never pre-match features)
        features_xg_player.csv  (per player per match)
"""
import json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'work')
SHARED = '/home/user/bdv-lab/shared/'
COMPACT = '/home/user/bdv-lab/players/compact/'
BIG = 0.30   # "big chance" = non-penalty shot with xG >= 0.30 (the feed has no big-chance flag)


def pois_binom(p):
    """Distribution of the number of goals from independent shots with probabilities p."""
    d = np.zeros(len(p) + 1); d[0] = 1.0
    for q in p:
        d[1:] = d[1:] * (1 - q) + d[:-1] * q
        d[0] *= (1 - q)
    return d


def deserved(ph, pa):
    dh, da = pois_binom(ph), pois_binom(pa)
    ch = np.cumsum(da)                       # P(away <= k)
    pH = sum(dh[i] * (ch[i - 1] if i - 1 < len(ch) else 1.0) for i in range(1, len(dh)))
    pD = sum(dh[i] * da[i] for i in range(min(len(dh), len(da))))
    return pH, pD, 1 - pH - pD


def main():
    S = pd.read_pickle(os.path.join(WORK, 'shots_xg.pkl'))
    OG = pd.read_pickle(os.path.join(WORK, 'owngoals.pkl'))
    meta = pd.read_pickle(os.path.join(WORK, 'match_meta.pkl'))
    meta = meta[meta.full == 1].set_index('id')
    S['np'] = (S.pen == 0)
    S['adv'] = (S.red_opp - S.red_for)
    zgrp = {'close': 'six', 'six_side': 'six', 'box_c': 'boxc', 'box_side': 'boxs', 'angle': 'boxs', 'angle_long': 'boxs', 'out': 'out', 'long': 'out', 'far35': 'out', 'pen': 'pen'}
    S['zg'] = S.zone.map(zgrp)
    rows = []
    og = OG.dropna(subset=['side']).groupby(['id', 'side']).size() if len(OG) else pd.Series(dtype=float)
    for mid, g in S.groupby('id', sort=False):
        if mid not in meta.index: continue
        m = meta.loc[mid]
        r = {'id': mid, 'lg': m.lg, 'season': m.season, 'date': m.date, 'goals_ok': m.goals_ok}
        xs = {}
        for sd in 'ha':
            q = g[g.side == sd]; n = q[q.np]
            x = n.xg.to_numpy()
            xs[sd] = q.xg.to_numpy()
            lvl, lead, trail = n.dsc == 0, n.dsc > 0, n.dsc < 0
            eq = n.adv == 0
            r.update({
                f'{sd}_xg': q.xg.sum(), f'{sd}_npxg': x.sum(), f'{sd}_pens': int((~q.np).sum()), f'{sd}_pen_goals': int(q.goal[~q.np].sum()),
                f'{sd}_goals': int(q.goal.sum()) + int(og.get((mid, sd), 0)), f'{sd}_og_for': int(og.get((mid, sd), 0)), f'{sd}_np_goals': int(n.goal.sum()),
                f'{sd}_shots': len(n), f'{sd}_sot': int(n.ot.sum()), f'{sd}_blocked': int((n.outcome == 'blocked').sum()),
                f'{sd}_xg_ns': n.xg_ns.sum(), f'{sd}_xg_gb': n.xg_gb.sum(), f'{sd}_xg_sit': n.xg_sit.sum(), f'{sd}_psxg': q.psxg.sum(),
                # "following a corner / set piece" and free-kick wording is printed only for goals and off-target attempts, so
                # xg_setp is a lower bound and xg_open (= npxG - flagged set-piece xG) an upper bound of the true split
                f'{sd}_xg_setp': x[n.sit0.isin(['corner', 'setp', 'dfk']).to_numpy()].sum(), f'{sd}_xg_fast': x[(n.sit0 == 'fast').to_numpy()].sum(),
                f'{sd}_xg_open': x[(~n.sit0.isin(['corner', 'setp', 'dfk'])).to_numpy()].sum(),
                f'{sd}_xg_head': x[(n.body == 'head').to_numpy()].sum(), f'{sd}_xg_cross': x[(n.ast == 'cross').to_numpy()].sum(),
                f'{sd}_xg_through': x[(n.ast == 'through').to_numpy()].sum(), f'{sd}_xg_reb': x[((n.stoppage == 0) & (n.dt_prev <= 5)).to_numpy()].sum(),
                f'{sd}_n_six': int((n.zg == 'six').sum()), f'{sd}_n_boxc': int((n.zg == 'boxc').sum()), f'{sd}_n_boxs': int((n.zg == 'boxs').sum()), f'{sd}_n_out': int((n.zg == 'out').sum()),
                f'{sd}_n_head': int((n.body == 'head').sum()),
                f'{sd}_bigch': int((x >= BIG).sum()), f'{sd}_xg_big': x[x >= BIG].sum(), f'{sd}_xg_small': x[x < 0.07].sum(),
                f'{sd}_xg_level': x[lvl.to_numpy()].sum(), f'{sd}_xg_lead': x[lead.to_numpy()].sum(), f'{sd}_xg_trail': x[trail.to_numpy()].sum(),
                f'{sd}_xg_lead2': x[(n.dsc >= 2).to_numpy()].sum(), f'{sd}_xg_trail2': x[(n.dsc <= -2).to_numpy()].sum(),
                f'{sd}_xg_11v11': x[eq.to_numpy()].sum(), f'{sd}_xg_manup': x[(n.adv > 0).to_numpy()].sum(), f'{sd}_xg_mandn': x[(n.adv < 0).to_numpy()].sum(),
                f'{sd}_xg_1h': x[(n.per == 1).to_numpy()].sum(), f'{sd}_xg_late': x[(n.minute >= 75).to_numpy()].sum(),
            })
            # default state-adjusted xG (a-priori weights): man up x0.7, man down x1.2, two or more goals apart x0.8
            w = np.where(n.adv > 0, 0.7, np.where(n.adv < 0, 1.2, 1.0)) * np.where(n.dsc.abs() >= 2, 0.8, 1.0)
            r[f'{sd}_xg_adj'] = float((x * w).sum())
        pH, pD, pA = deserved(xs['h'], xs['a'])
        r.update({'des_pH': pH, 'des_pD': pD, 'des_pA': pA, 'h_xpts': 3 * pH + pD, 'a_xpts': 3 * pA + pD})
        rows.append(r)
    F = pd.DataFrame(rows).sort_values(['date', 'id'])
    num = F.select_dtypes('float').columns
    F[num] = F[num].round(4)
    F.to_csv(os.path.join(HERE, 'features_xg_match.csv'), index=False)
    print('features_xg_match.csv', F.shape)
    print(F.groupby(['lg', 'season']).agg(n=('id', 'size'), xg=('h_xg', 'mean'), xga=('a_xg', 'mean'), g=('h_goals', 'mean'), ga=('a_goals', 'mean'), pH=('des_pH', 'mean'), pD=('des_pD', 'mean')).round(3).to_string())

    # ---------------- per player per match ----------------
    M = {m['id']: m for m in json.load(open(SHARED + 'matches.json'))}
    sh = S.groupby(['id', 'side', 'shooter_id']).agg(shots=('np', 'sum'), npxg=('xg', lambda v: v[S.loc[v.index, 'np']].sum()), xg=('xg', 'sum'),
                                                     goals=('goal', 'sum'), np_goals=('goal', lambda v: v[S.loc[v.index, 'np']].sum()),
                                                     sot=('ot', 'sum'), psxg=('psxg', 'sum'), pens=('pen', 'sum'), name=('shooter', 'first')).reset_index().rename(columns={'shooter_id': 'pid'})
    A = S[S.assister_id.notna()]
    xa = A.groupby(['id', 'side', 'assister_id']).agg(xa=('xg', 'sum'), key_passes=('xg', 'size'), assists=('goal', 'sum'), name_a=('assister', 'first')).reset_index().rename(columns={'assister_id': 'pid'})
    P = sh.merge(xa, on=['id', 'side', 'pid'], how='outer')
    P['name'] = P['name'].fillna(P['name_a']); P = P.drop(columns=['name_a'])
    # minutes + roster info (every player who appeared, also those without shots)
    ids = set(meta.index)
    ros = []
    for mid in ids:
        try: r = json.load(open(COMPACT + mid + '.json'))
        except Exception: continue
        m = M[mid]
        for t in r.get('teams') or []:
            sd = 'h' if t.get('ha') == 'home' else 'a'
            for p in t.get('pl') or []:
                st = p.get('s'); si = p.get('si'); so = p.get('so')
                if not (st or si): continue
                t_in, t_out = (0, 90)
                for c, typ in p.get('pl') or []:
                    if c is None: continue
                    mn = min(90, c[0])
                    if 'S' in typ:
                        if si and not st and t_in == 0: t_in = mn
                        else: t_out = mn
                    if 'R' in typ: t_out = min(t_out, mn)
                ros.append({'id': mid, 'side': sd, 'pid': p.get('i'), 'rname': p.get('n'), 'pos': p.get('pos'), 'starter': int(bool(st)), 'minutes': max(0, t_out - t_in)})
    R = pd.DataFrame(ros)
    P = R.merge(P, on=['id', 'side', 'pid'], how='outer')
    P['name'] = P['rname'].fillna(P['name']); P = P.drop(columns=['rname'])
    for c in ['shots', 'npxg', 'xg', 'goals', 'np_goals', 'sot', 'psxg', 'pens', 'xa', 'key_passes', 'assists']: P[c] = P[c].fillna(0)
    # goalkeepers: post-shot xG faced and goals conceded (attributed to the starting keeper when he played 90')
    tm = S.groupby(['id', 'side']).agg(psxg=('psxg', 'sum'), goals=('goal', 'sum'), sot=('ot', 'sum'), xg=('xg', 'sum')).reset_index()
    opp = tm.assign(side=tm.side.map({'h': 'a', 'a': 'h'})).rename(columns={'psxg': 'gk_psxg_faced', 'goals': 'gk_goals_conceded', 'sot': 'gk_sot_faced', 'xg': 'gk_xg_faced'})
    gk = P[(P.pos == 'G') & (P.starter == 1) & (P.minutes >= 90)][['id', 'side', 'pid']].merge(opp, on=['id', 'side'], how='left')
    P = P.merge(gk, on=['id', 'side', 'pid'], how='left')
    P['tid'] = [M[i][s] if i in M else None for i, s in zip(P.id, P.side)]
    P['season'] = [M[i]['season'] for i in P.id]; P['lg'] = [M[i]['lg'] for i in P.id]; P['date'] = [M[i]['date'] for i in P.id]
    cols = ['id', 'lg', 'season', 'date', 'side', 'tid', 'pid', 'name', 'pos', 'starter', 'minutes', 'shots', 'sot', 'goals', 'np_goals', 'pens', 'xg', 'npxg', 'psxg', 'xa', 'key_passes', 'assists',
            'gk_psxg_faced', 'gk_xg_faced', 'gk_sot_faced', 'gk_goals_conceded']
    P = P[cols].sort_values(['date', 'id', 'side', 'pid'])
    for c in ['xg', 'npxg', 'psxg', 'xa', 'gk_psxg_faced', 'gk_xg_faced']: P[c] = P[c].round(4)
    P.to_csv(os.path.join(HERE, 'features_xg_player.csv'), index=False)
    print('features_xg_player.csv', P.shape, 'pid missing', P.pid.isna().mean().round(4), 'minutes missing', P.minutes.isna().mean().round(4))
    top = P.groupby(['pid', 'name']).agg(g=('np_goals', 'sum'), x=('npxg', 'sum'), sh=('shots', 'sum'), xa=('xa', 'sum')).sort_values('x', ascending=False).head(10)
    print(top.round(1).to_string())


if __name__ == '__main__':
    main()
