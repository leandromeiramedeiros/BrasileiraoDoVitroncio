#!/usr/bin/env python
"""Shot-level expected goals from the ESPN play-by-play (coordinates + text flags), fitted walk-forward by season.

For shots of season Y the logistic model is fitted on shots of seasons < Y (both divisions). For the first season with
play-by-play (2014) a fixed a-priori table is used. Own goals and VAR bookkeeping events are ignored.

Outputs
  data/match_xg.csv      mid, xg_h, xg_a, shots_h, shots_a          (POST-match quantities, to be used for LATER matches)
  data/player_xg.csv.gz  mid, side, pid, xg, xa, shots              (same remark)
"""
import os, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.abspath(__file__))
S = pd.read_csv(f'{ROOT}/data/shots.csv.gz', dtype={'mid': str, 'pid': str, 'ast': str})
T = pd.read_csv(f'{ROOT}/data/team_match.csv', dtype={'mid': str}).drop_duplicates('mid').set_index('mid')
S['season'] = S.mid.map(T.season)
keep = S.type.str.startswith('Shot') | S.type.str.startswith('Goal') | S.type.str.startswith('Penalty - ')
S = S[keep & S.side.isin(['h', 'a'])].copy()
S['goal'] = (S.type.str.startswith('Goal') | S.type.eq('Penalty - Scored')).astype(int)
S['pen'] = S.type.str.startswith('Penalty - ').astype(int)

# coordinates -> metres from the goal line (dx) and from the pitch axis (dy); two conventions exist in the feed
x, y = S.x.to_numpy(float), S.y.to_numpy(float)
pct = (x > 1) | (y > 1)
dx = np.where(pct, (100 - x) / 100 * 105, x * 52.5)
dy = np.where(pct, (y - 50) / 100 * 68, (y - 0.5) * 68)
miss = (x == 0) & (y == 0) | np.isnan(x)
# missing coordinates: typical location of the text category
dx = np.where(miss, np.where(S.close == 1, 4.5, np.where(S.centre == 1, 11.0, np.where(S.outbox == 1, 25.0, 13.0))), dx)
dy = np.where(miss, np.where((S.close == 1) | (S.centre == 1) | (S.outbox == 1), 0.0, 9.0), dy)
dist = np.sqrt(dx ** 2 + dy ** 2).clip(1, 60)
ang = np.arctan2(7.32 * np.maximum(dx, 0.3), dx ** 2 + dy ** 2 - 3.66 ** 2)
ang = np.where(ang < 0, ang + np.pi, ang)
S['dist'], S['ang'] = dist, ang
# 'fast break' is left out on purpose: the feed attaches it disproportionately to goals (30% conversion), i.e. it is outcome-biased
# so is the free-kick wording (every free-kick goal is tagged, most missed free kicks are not)
X = np.column_stack([np.log(dist), ang, S.header, S.setpiece, S.close, S.outbox, S.cross, S.through, S.diffangle])


def prior_xg(r):
    base = np.where(r.close == 1, 0.30, np.where(r.centre == 1, 0.12, np.where(r.outbox == 1, 0.03, 0.06)))
    return np.where(r.header == 1, base * 0.7, base)


xg = np.zeros(len(S))
seasons = sorted(S.season.unique())
first = seasons[0]
for Y in seasons:
    te = (S.season == Y).to_numpy()
    op = te & (S.pen == 0).to_numpy()
    if Y == first:
        xg[op] = prior_xg(S[op])
        pen_rate = 0.76
    else:
        tr = (S.season < Y).to_numpy() & (S.pen == 0).to_numpy()
        m = LogisticRegression(C=10.0, max_iter=500).fit(X[tr], S.goal.to_numpy()[tr])
        xg[op] = m.predict_proba(X[op])[:, 1]
        ptr = (S.season < Y).to_numpy() & (S.pen == 1).to_numpy()
        pen_rate = (S.goal.to_numpy()[ptr].sum() + 0.76 * 50) / (ptr.sum() + 50)
        if Y == seasons[-1]:
            print('last model coefs [log dist, angle, header, set piece, close, outside box, cross, through ball, difficult angle]:', np.round(m.coef_[0], 3), 'pen', round(pen_rate, 3))
    xg[te & (S.pen == 1).to_numpy()] = pen_rate
S['xg'] = xg
cal = S.groupby('season').agg(shots=('goal', 'size'), goals=('goal', 'sum'), xg=('xg', 'sum'))
cal['ratio'] = cal.goals / cal.xg
print(cal.round(1).to_string())
tm = S.groupby(['mid', 'side']).agg(xg=('xg', 'sum'), shots=('xg', 'size')).unstack('side')
tm.columns = [f'{a}_{b}' for a, b in tm.columns]
tm.reset_index().to_csv(f'{ROOT}/data/match_xg.csv', index=False, float_format='%.4f')
sh = S[S.pid.notna()].groupby(['mid', 'side', 'pid']).agg(xg=('xg', 'sum'), shots=('xg', 'size')).reset_index()
xa = S[S.ast.notna()].groupby(['mid', 'side', 'ast']).agg(xa=('xg', 'sum')).reset_index().rename(columns={'ast': 'pid'})
pl = sh.merge(xa, on=['mid', 'side', 'pid'], how='outer').fillna(0)
pl.to_csv(f'{ROOT}/data/player_xg.csv.gz', index=False, float_format='%.4f')
print('matches with xG:', len(tm), 'player rows:', len(pl))
