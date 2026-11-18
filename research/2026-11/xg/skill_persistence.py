#!/usr/bin/env python
"""Step 4b - do finishing (goals - xG) and goalkeeping/defending (xGA - GA) persist?
Team-season level: split-half (odd vs even matches of the same season) and year-to-year correlations,
compared with the persistence of xG itself. Writes work/skill_persistence.json and prints the table."""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F = pd.read_csv(os.path.join(HERE, 'features_xg_match.csv'), dtype={'id': str})
M = {m['id']: m for m in json.load(open('/home/user/bdv-lab/shared/matches.json'))}
rows = []
for r in F.itertuples():
    m = M[r.id]
    for sd, od in (('h', 'a'), ('a', 'h')):
        g = getattr(r, sd + '_np_goals'); ga = getattr(r, od + '_np_goals')
        rows.append({'id': r.id, 'lg': r.lg, 'season': r.season, 'date': r.date, 'tid': m[sd],
                     'g': g, 'xg': getattr(r, sd + '_npxg'), 'ga': ga, 'xga': getattr(r, od + '_npxg'),
                     'ps': getattr(r, sd + '_psxg'), 'psa': getattr(r, od + '_psxg'),
                     'gall': getattr(r, sd + '_goals') - getattr(r, sd + '_og_for'), 'gaall': getattr(r, od + '_goals') - getattr(r, od + '_og_for'),
                     'xgt': getattr(r, sd + '_xg'), 'xgat': getattr(r, od + '_xg')})
T = pd.DataFrame(rows).sort_values(['tid', 'date'])
T['k'] = T.groupby(['tid', 'season']).cumcount()


def agg(d):
    return pd.Series({'n': len(d), 'g': d.g.mean(), 'xg': d.xg.mean(), 'ga': d.ga.mean(), 'xga': d.xga.mean(),
                      'fin': d.g.sum() / d.xg.sum() - 1, 'con': d.ga.sum() / d.xga.sum() - 1,
                      'gk': d.gaall.sum() / max(1e-9, d.psa.sum()) - 1, 'place': d.ps.sum() / d.xgt.sum() - 1,
                      'gd': d.g.mean() - d.ga.mean(), 'xgd': d.xg.mean() - d.xga.mean()})


out = {}
metrics = ['g', 'xg', 'ga', 'xga', 'gd', 'xgd', 'fin', 'con', 'gk', 'place']
# split-half
a = T[T.k % 2 == 0].groupby(['lg', 'season', 'tid']).apply(agg); b = T[T.k % 2 == 1].groupby(['lg', 'season', 'tid']).apply(agg)
j = a.join(b, lsuffix='_a', rsuffix='_b').dropna()
j = j[(j.n_a >= 15) & (j.n_b >= 15)]
out['split_half'] = {m: round(float(np.corrcoef(j[m + '_a'], j[m + '_b'])[0, 1]), 3) for m in metrics}; out['split_half']['n_team_seasons'] = int(len(j))
# first half of season -> second half (chronological)
a = T[T.k < 19].groupby(['lg', 'season', 'tid']).apply(agg); b = T[T.k >= 19].groupby(['lg', 'season', 'tid']).apply(agg)
j = a.join(b, lsuffix='_a', rsuffix='_b').dropna(); j = j[(j.n_a >= 15) & (j.n_b >= 15)]
out['first_to_second_half'] = {m: round(float(np.corrcoef(j[m + '_a'], j[m + '_b'])[0, 1]), 3) for m in metrics}; out['first_to_second_half']['n_team_seasons'] = int(len(j))
# what predicts second-half goal difference: first-half GD or xGD?
X = np.column_stack([np.ones(len(j)), j.gd_a, j.xgd_a]); bb = np.linalg.lstsq(X, j.gd_b.to_numpy(), rcond=None)[0]
out['second_half_gd_on_first_half'] = {'coef_gd': round(float(bb[1]), 3), 'coef_xgd': round(float(bb[2]), 3),
                                       'r_gd': round(float(np.corrcoef(j.gd_a, j.gd_b)[0, 1]), 3), 'r_xgd': round(float(np.corrcoef(j.xgd_a, j.gd_b)[0, 1]), 3)}
# year to year (same league both years)
s = T.groupby(['lg', 'season', 'tid']).apply(agg).reset_index()
s2 = s.assign(season=s.season - 1)
j = s.merge(s2, on=['lg', 'season', 'tid'], suffixes=('_a', '_b')); j = j[(j.n_a >= 30) & (j.n_b >= 30)]
out['year_to_year'] = {m: round(float(np.corrcoef(j[m + '_a'], j[m + '_b'])[0, 1]), 3) for m in metrics}; out['year_to_year']['n_pairs'] = int(len(j))
json.dump(out, open(os.path.join(HERE, 'work', 'skill_persistence.json'), 'w'), indent=1)
print(pd.DataFrame({k: v for k, v in out.items() if k != 'second_half_gd_on_first_half'}).to_string())
print(out['second_half_gd_on_first_half'])
