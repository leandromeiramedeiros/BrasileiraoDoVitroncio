#!/usr/bin/env python
"""Writes js/rosters.json.gz: every finished match with a usable starting XI on both sides (Série A and B), chronological,
in the shape consumed by lineup.mjs. Only the fields the module uses (no shots).
  { id, date, lg, season, h, a, home: { xi: [ids], bench: [ids], min: {id: minutes}, ga: {id: goals+assists} }, away: {...} }
Team ids are canonical (5266->3395, 6080->3448, 6274 in Série A->4929, 7385 in Série B->3448).
Bench lists before 2010 only contain the substitutes that were used (post-match information) and are therefore emptied."""
import json, gzip, os
import pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open('/home/user/bdv-lab/shared/matches.json'))
P = pd.read_csv(f'{ROOT}/data/player_match.csv.gz', dtype={'mid': str, 'pid': str, 'tid': str})


def canon(tid, lg):
    tid = str(tid)
    if tid == '5266': return '3395'
    if tid == '6080': return '3448'
    if tid == '6274' and lg == 'A': return '4929'
    if tid == '7385' and lg == 'B': return '3448'
    return tid


sides = {}
for (mid, side), g in P.groupby(['mid', 'side'], sort=False):
    st = g[g.starter == 1]
    if len(st) < 10: continue
    season = int(g.season.iloc[0])
    sides[(mid, side)] = {
        'xi': list(st.pid),
        'bench': list(g.pid[g.starter != 1]) if season >= 2010 else [],
        'min': {p: int(round(m)) for p, m in zip(g.pid, g['min']) if m > 0},
        'ga': {p: int(x + y) for p, x, y in zip(g.pid, g.g, g.a) if x + y > 0},
    }
out = []
for m in M:
    if m['status'] != 'post': continue
    h, a = sides.get((m['id'], 'h')), sides.get((m['id'], 'a'))
    if h is None or a is None: continue
    out.append({'id': str(m['id']), 'date': m['date'], 'lg': m['lg'], 'season': m['season'], 'h': canon(m['h'], m['lg']), 'a': canon(m['a'], m['lg']), 'home': h, 'away': a})
path = f'{ROOT}/js/rosters.json.gz'
with gzip.open(path, 'wt', compresslevel=9) as f:
    json.dump(out, f, separators=(',', ':'))
c = pd.Series([(x['lg'], x['season']) for x in out]).value_counts().sort_index()
print('matches:', len(out), 'size MB:', round(os.path.getsize(path) / 1e6, 2))
print({f'{k[0]}{k[1]}': int(v) for k, v in c.items()})
