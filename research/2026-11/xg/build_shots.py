#!/usr/bin/env python
"""Step 1 - shot dataset.

Reads the compact ESPN summaries written by the players track
(/home/user/bdv-lab/players/compact/<id>.json) plus shared/matches.json and writes
  work/shots.pkl   one row per shot (own goals and VAR lines are dropped; see `og` table)
  work/match_meta.pkl  one row per match that has full shot commentary (+ sanity columns)

Nothing here is fitted: it is pure parsing, so it can be re-run for new matches at any time.
"""
import json, os, re, sys, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COMPACT = '/home/user/bdv-lab/players/compact/'
SHARED = '/home/user/bdv-lab/shared/'
WORK = os.path.join(HERE, 'work'); os.makedirs(WORK, exist_ok=True)

# pitch geometry in "ESPN units": x = share of a 96 m half... calibrated on penalties (x=0.23 -> 11 m)
XM, YM = 48.0, 68.0

ZONE_RE = re.compile(
    r'from (the centre of the box|very close range|outside the box|the left side of the box|the right side of the box|'
    r'the left side of the six yard box|the right side of the six yard box|'
    r'a difficult angle and long range on the (?:left|right)|a difficult angle on the (?:left|right)|'
    r'long range on the (?:left|right)|more than \d+ yards(?: on the (?:left|right) wing)?)')
ZONE_CODE = {
    'the centre of the box': 'box_c', 'very close range': 'close', 'outside the box': 'out',
    'the left side of the box': 'box_side', 'the right side of the box': 'box_side',
    'the left side of the six yard box': 'six_side', 'the right side of the six yard box': 'six_side',
}
PLACE_RE = re.compile(r'(?:saved\s+in|to) the (bottom left corner|bottom right corner|top left corner|top right corner|'
                      r'centre of the goal|top centre of the goal|high centre of the goal)')
MISS_RE = re.compile(r'(is close, but misses to the (?:left|right)|misses to the (?:left|right)|is too high|is just a bit too high|'
                     r'is high and wide to the (?:left|right)|hits the (?:bar|left post|right post))')
SCORE_RE = re.compile(r'^Goal!\s+(.*?) (\d+), (.*?) (\d+)\.')


def zone_of(tx):
    m = ZONE_RE.search(tx)
    if not m: return None
    z = m.group(1)
    if z in ZONE_CODE: return ZONE_CODE[z]
    if z.startswith('a difficult angle and long range'): return 'angle_long'
    if z.startswith('a difficult angle'): return 'angle'
    if z.startswith('long range'): return 'long'
    return 'far35'


def zone_from_xy(dx, dy):
    """Fallback zone from coordinates (metres from goal line / from centre line)."""
    ady = abs(dy)
    if dx <= 5.5 and ady <= 3.7: return 'close'
    if dx <= 5.5 and ady <= 9.2: return 'six_side'
    if dx <= 5.5: return 'angle'
    if dx <= 16.5 and ady <= 9.2: return 'box_c'
    if dx <= 16.5 and ady <= 20.2: return 'box_side'
    if dx <= 16.5: return 'angle_long'
    if dx > 32: return 'far35'
    if ady > 20.2: return 'long'
    return 'out'

# typical location (metres) per zone, used only when the coordinates are missing (0,0) -- fixed a priori
ZONE_XY = {'close': (3.5, 0.0), 'six_side': (3.7, 5.0), 'angle': (3.8, 13.0), 'box_c': (11.0, 0.0), 'box_side': (11.0, 13.0),
           'angle_long': (10.5, 22.0), 'out': (23.0, 0.0), 'long': (23.0, 22.0), 'far35': (37.0, 0.0), 'pen': (11.0, 0.0)}


def main():
    M = {m['id']: m for m in json.load(open(SHARED + 'matches.json'))}
    T = json.load(open(SHARED + 'teams.json'))
    files = sorted(f for f in os.listdir(COMPACT) if f.endswith('.json'))
    # ---- pass 1: display name -> team id (vote over matches) -------------------------------------------------
    name2tid = collections.defaultdict(collections.Counter)
    recs = {}
    for f in files:
        try: r = json.load(open(COMPACT + f))
        except Exception: continue
        if not r.get('shots') or r['id'] not in M: continue
        recs[r['id']] = r
        m = M[r['id']]
        hn, an = None, None
        for s in r['shots']:
            g = SCORE_RE.match(s['tx'] or '')
            if g and s['tm']:
                # "Goal!  Home 1, Away 0. Player (Team) ..." -> the team in brackets is the scorer's team as written in text
                pass
        for s in r['shots']:
            if s['tm']:
                name2tid[s['tm']][m['h']] += 1; name2tid[s['tm']][m['a']] += 1
    disp = {}
    for nm, c in name2tid.items():
        (t1, n1), = c.most_common(1)
        disp[nm] = t1
    # a display name always co-occurs with its own team id; the top vote is that id (the opponent ids are spread out)
    bad = []
    rows, ogs, meta = [], [], []
    for mid, r in recs.items():
        m = M[mid]
        if m['status'] != 'post': continue
        side_of = {}
        for nm in {s['tm'] for s in r['shots'] if s['tm']}:
            t = disp.get(nm)
            if t == m['h']: side_of[nm] = 'h'
            elif t == m['a']: side_of[nm] = 'a'
        if len(side_of) < len({s['tm'] for s in r['shots'] if s['tm']}):
            # resolve by elimination
            names = sorted({s['tm'] for s in r['shots'] if s['tm']})
            if len(names) == 2 and len(side_of) == 1:
                other = [n for n in names if n not in side_of][0]
                side_of[other] = 'a' if list(side_of.values())[0] == 'h' else 'h'
            else:
                bad.append((mid, names)); continue
        # player name -> id per side
        pid = {'h': {}, 'a': {}}
        for t in r.get('teams') or []:
            sd = 'h' if t.get('ha') == 'home' else 'a'
            for p in t.get('pl') or []:
                if p.get('n'): pid[sd][p['n']] = p.get('i')
        reds = [(1 if (e['m'] <= 45) else 2, e['m'], e.get('x') or 0, e['s']) for e in m['ev'] if e['t'] == 'R']
        sc = {'h': 0, 'a': 0}
        nshot = {'h': 0, 'a': 0}; ngoal = {'h': 0, 'a': 0}
        shots = [s for s in r['shots']]
        nfull = sum(1 for s in shots if s['ty'].startswith('Shot'))
        prev = {'h': None, 'a': None}  # last shot of the side: (abs time, type)
        seen_scores, seen_shots = set(), set()
        match_rows = []
        for k, s in enumerate(shots):
            ty, tx = s['ty'], (s['tx'] or '')
            if ty.startswith('VAR'):
                if 'No Goal' in tx or ('VAR: Goal' in tx and 'cancelled' in tx):
                    pass  # handled via the running score below
                continue
            sd = side_of.get(s['tm'])
            per = s['per'] or (1 if (s['c'] or 0) <= 2700 else 2)
            c = float(s['c'] or 0)
            minute = c / 60.0
            if ty == 'Own Goal':
                # ESPN puts the BENEFITING team in `team` for own goals (checked against final scores)
                ben = sd
                ogs.append({'id': mid, 'side': ben, 'c': c, 'per': per, 'k': k})
                if ben: sc[ben] += 1; ngoal[ben] += 1
                continue
            if sd is None: continue
            is_pen = ty.startswith('Penalty')
            goal = int(ty.startswith('Goal') or ty == 'Penalty - Scored')
            if goal:
                # the feed sometimes repeats a goal line (same running score) -> keep the first only
                g = SCORE_RE.match(tx)
                key = (g.group(2), g.group(4)) if g else None
                if key is not None and key in seen_scores: continue
                if key is not None: seen_scores.add(key)
            else:
                key = (ty, c, per, tx)
                if c not in (2700.0, 5400.0) and key in seen_shots: continue
                seen_shots.add(key)
            # coordinates -> metres
            x, y = s['x'], s['y']
            miss_xy = (x is None or y is None or (x == 0 and y == 0))
            if not miss_xy:
                if x > 1.5 or y > 1.5:      # 2026 feed: percent of the full pitch, own goal line = 0
                    x, y = (100 - x) / 50.0, 1 - y / 100.0
                dx, dy = x * XM, (y - 0.5) * YM
            body = 'head' if re.search(r'\bheader\b', tx) else 'right' if 'right footed' in tx else 'left' if 'left footed' in tx else 'other'
            z = 'pen' if is_pen else zone_of(tx)
            dfk = int(('from a direct free kick' in tx) or ('from a free kick' in tx) or ty == 'Goal - Free-kick')
            if z is None and not miss_xy: z = zone_from_xy(dx, dy)
            ztxt = z
            if z is None: z = 'out' if dfk else 'box_c'
            if miss_xy:
                dx, dy = ZONE_XY[z]
            sit = 'pen' if is_pen else 'dfk' if dfk else 'corner' if 'following a corner' in tx else \
                'setp' if 'following a set piece' in tx else 'fast' if 'following a fast break' in tx else 'open'
            am = re.search(r'Assisted by (.*?)(?: with a (cross|through ball|headed pass))?\s*(?:following|\.|$)', tx)
            ast = 'none'
            if am: ast = {'cross': 'cross', 'through ball': 'through', 'headed pass': 'headpass'}.get(am.group(2), 'pass')
            pm = PLACE_RE.search(tx); mm = MISS_RE.search(tx)
            outcome = 'goal' if goal else 'saved' if ('saved' in tx and ty not in ('Shot Blocked',)) else \
                'blocked' if (ty == 'Shot Blocked' or 'is blocked' in tx) else 'wood' if (ty.endswith('Woodwork') or 'hits the' in tx) else 'off'
            if ty == 'Shot Hit Woodwork' and not goal: outcome = 'wood' if 'hits the' in tx else outcome
            ot = int(goal or outcome == 'saved')
            # rebound / second ball: previous shot of the same side a few seconds earlier
            tabs = (per - 1) * 10000 + c
            pv = prev[sd]
            dt_prev = (tabs - pv[0]) if pv else 9999.0
            prev_out = pv[1] if pv else 'none'
            # game state BEFORE the shot
            dsc = sc[sd] - sc['a' if sd == 'h' else 'h']
            rf = sum(1 for (rp, rm, rx, rs) in reds if rs == sd and (rp, rm * 60 + rx * 60) < (per, c + (0 if c not in (2700.0, 5400.0) else 1e6)))
            ra = sum(1 for (rp, rm, rx, rs) in reds if rs != sd and (rp, rm * 60 + rx * 60) < (per, c + (0 if c not in (2700.0, 5400.0) else 1e6)))
            who = s.get('who') or []
            shooter = who[0] if who else None
            assister = who[1] if len(who) > 1 else None
            row = {
                'id': mid, 'k': k, 'season': m['season'], 'lg': m['lg'], 'date': m['date'], 'side': sd,
                'tid': m[sd], 'opp': m['a' if sd == 'h' else 'h'], 'per': per, 'c': c, 'minute': minute,
                'stoppage': int(c in (2700.0, 5400.0)), 'ty': ty, 'goal': goal, 'pen': int(is_pen),
                'dx': dx, 'dy': dy, 'miss_xy': int(miss_xy), 'zone': z, 'zone_txt': int(ztxt is not None), 'body': body, 'sit': sit,
                'ast': ast, 'outcome': outcome, 'ot': ot, 'place': pm.group(1) if pm else None, 'miss': mm.group(1) if mm else None,
                'dt_prev': dt_prev, 'prev_out': prev_out, 'dsc': dsc, 'red_for': rf, 'red_opp': ra,
                'shooter': shooter, 'shooter_id': pid[sd].get(shooter), 'assister': assister if am else None,
                'assister_id': pid[sd].get(assister) if am else None, 'newfmt': int(' by ' in tx and 'saved' in tx and tx.count('(') >= 2),
            }
            match_rows.append(row)
            prev[sd] = (tabs, outcome)
            nshot[sd] += 1
            if goal:
                sc[sd] += 1; ngoal[sd] += 1
        # VAR-cancelled goals: if commentary goals exceed the final score, drop the surplus using the running score text
        full = nfull >= 8   # goal-only commentaries (old Serie B) have no "Shot ..." lines
        ok = (ngoal['h'] == m['hg'] and ngoal['a'] == m['ag'])
        st = m['stats'] if m.get('statsOk') else None
        meta.append({'id': mid, 'season': m['season'], 'lg': m['lg'], 'date': m['date'], 'h': m['h'], 'a': m['a'], 'hg': m['hg'], 'ag': m['ag'],
                     'full': int(full), 'goals_ok': int(ok), 'cg_h': ngoal['h'], 'cg_a': ngoal['a'], 'n_h': nshot['h'], 'n_a': nshot['a'],
                     'st_sh_h': st['h']['sh'] if st else np.nan, 'st_sh_a': st['a']['sh'] if st else np.nan,
                     'st_sot_h': st['h']['sot'] if st else np.nan, 'st_sot_a': st['a']['sot'] if st else np.nan,
                     'newfmt': int(any(rw['newfmt'] for rw in match_rows))})
        if full: rows.extend(match_rows)
    S = pd.DataFrame(rows)
    meta = pd.DataFrame(meta)
    OG = pd.DataFrame(ogs)
    S.to_pickle(os.path.join(WORK, 'shots.pkl')); meta.to_pickle(os.path.join(WORK, 'match_meta.pkl')); OG.to_pickle(os.path.join(WORK, 'owngoals.pkl'))
    print('unresolved team names in', len(bad), 'matches', bad[:5])
    print('matches with shots:', len(meta), ' full commentary:', int(meta.full.sum()), ' shots:', len(S))
    return S, meta, OG


if __name__ == '__main__':
    S, meta, OG = main()
    f = meta[meta.full == 1]
    print(f.groupby(['lg', 'season']).agg(n=('id', 'size'), goals_ok=('goals_ok', 'mean'), newfmt=('newfmt', 'sum')).to_string())
