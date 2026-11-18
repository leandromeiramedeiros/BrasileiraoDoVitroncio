#!/usr/bin/env python
"""Turn compact/<id>.json (reduced ESPN summaries) + shared/matches.json into clean tables.

Outputs (in data/):
  player_match.csv.gz  one row per (match, team, listed player): starter, minutes, position, box-score stats
  team_match.csv       one row per (match, side): formation, n starters, data-quality flags
  shots.csv.gz         one row per shot from the play-by-play (2014+ A, 2017+ B): side, minute, shooter id, x/y, type, text flags
  rosters.jsonl.gz     the compact per-match rosters concatenated (the "compact roster dataset")
"""
import json, os, gzip, re, sys, unicodedata
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED = '/home/user/bdv-lab/shared/'
os.makedirs(f'{ROOT}/data', exist_ok=True)

POSG = {'G': 'G', 'D': 'D', 'M': 'M', 'F': 'F'}


def posgroup(p):
    if not p or p == 'SUB':
        return ''
    if p in POSG:
        return p
    b = p.split('-')[0]
    if b == 'G':
        return 'G'
    if b in ('CD', 'LB', 'RB', 'SW', 'LWB', 'RWB', 'D', 'CB'):
        return 'D'
    if b in ('DM', 'CM', 'LM', 'RM', 'AM', 'M'):
        return 'M'
    if b in ('F', 'CF', 'LF', 'RF', 'S', 'ST', 'LW', 'RW'):
        return 'F'
    return '?'


def norm(s):
    s = unicodedata.normalize('NFKD', s or '')
    return ''.join(c for c in s if not unicodedata.combining(c)).lower().strip()


def minute(c):
    """[min, added] -> minute on a 0..90 scale (stoppage time collapsed into 45 / 90)."""
    if c is None:
        return None
    return float(min(c[0], 90))


def main():
    M = json.load(open(SHARED + 'matches.json'))
    rows, trows, srows = [], [], []
    posseen = {}
    jl = gzip.open(f'{ROOT}/data/rosters.jsonl.gz', 'wt')
    nfile = 0
    for m in M:
        f = f'{ROOT}/compact/{m["id"]}.json'
        if not os.path.exists(f):
            continue
        c = json.load(open(f))
        jl.write(json.dumps(c, ensure_ascii=False, separators=(',', ':')) + '\n')
        nfile += 1
        if m['status'] != 'post':
            continue
        # red cards from the shared event list (player id -> minute)
        red = {}
        for e in m['ev']:
            if e['t'] == 'R' and e.get('p'):
                red.setdefault(str(e['p']), float(min(e['m'], 90)))
        name2id = {'h': {}, 'a': {}}
        for t in c['teams']:
            side = 'h' if str(t['tid']) == str(m['h']) else 'a' if str(t['tid']) == str(m['a']) else None
            if side is None:
                continue
            # de-duplicate players (older seasons list some twice: prefer active, with stats, starter)
            best = {}
            for p in t['pl']:
                if not p.get('i'):
                    continue
                key = (p.get('act', 1), 1 if p.get('st') else 0, 1 if p.get('pl') else 0, p['s'])
                if p['i'] not in best or key > best[p['i']][0]:
                    best[p['i']] = (key, p)
            pl = [v[1] for v in best.values()]
            anystats = any(p.get('st') for p in pl)
            st = [p for p in pl if p['s']]
            if len(st) > 11:  # drop inactive / stat-less "starters" first
                def strength(p):
                    return (p.get('act', 1), 1 if (p.get('st') or {}).get('appearances') else 0, 1 if p.get('pl') else 0)
                st = sorted(st, key=strength, reverse=True)
                for p in st[11:]:
                    p['s'] = 0; p['_demoted'] = 1
                st = st[:11]
            nst = len(st)
            n_used = 0
            for p in pl:
                pid = p['i']
                subs = [minute(x[0]) for x in p.get('pl', []) if 'S' in x[1] and x[0] is not None]
                t_in, t_out = None, None
                if p['s']:
                    t_in = 0.0
                    if p['so'] and subs:
                        t_out = subs[-1]
                elif p['si']:
                    t_in = subs[0] if subs else 70.0
                    if p['so'] and len(subs) > 1:
                        t_out = subs[-1]
                if t_in is not None:
                    end = 90.0 if t_out is None else t_out
                    if pid in red:
                        end = min(end, red[pid])
                    mins = max(0.0, end - t_in)
                    if p['si'] and not p['s']:
                        mins = max(mins, 1.0)
                    t_out = end
                    n_used += 1
                else:
                    mins = 0.0
                s = p.get('st') or {}
                ev = ''.join(x[1] for x in p.get('pl', []))
                g_ev = sum(1 for x in p.get('pl', []) if 'G' in x[1] and 'O' not in x[1])
                a_ev = sum(1 for x in p.get('pl', []) if 'A' in x[1])
                pg = posgroup(p.get('pos'))
                rows.append((m['id'], m['lg'], m['season'], m['date'], t['tid'], side, pid, p.get('n'), p['s'],
                             int(t_in is not None), t_in, t_out, mins, p.get('pos'), pg, p.get('fp'),
                             max(g_ev, int(s.get('totalGoals', 0))) if anystats else g_ev,
                             max(a_ev, int(s.get('goalAssists', 0))) if anystats else a_ev,
                             s.get('totalShots', 0) if anystats else np.nan, s.get('shotsOnTarget', 0) if anystats else np.nan,
                             s.get('yellowCards', ev.count('Y')), int(pid in red or 'R' in ev),
                             s.get('foulsCommitted', 0) if anystats else np.nan, s.get('foulsSuffered', 0) if anystats else np.nan,
                             s.get('saves', 0) if anystats else np.nan, s.get('ownGoals', 0), s.get('offsides', 0) if anystats else np.nan))
                name2id[side][norm(p.get('n'))] = pid
            trows.append((m['id'], m['lg'], m['season'], m['date'], t['tid'], side, t.get('form'), nst, len(pl), n_used,
                          int(anystats), int(any(p.get('pos') == 'SUB' or (not p['s'] and not p['si']) for p in pl))))
        # shots from the play-by-play
        if c.get('shots'):
            tn = {}
            for sh in c['shots']:
                who = (sh.get('who') or [None])
                shooter = norm(who[0]) if who and who[0] else ''
                # side: by shooter name in rosters (robust), fallback none
                side = 'h' if shooter in name2id['h'] and shooter not in name2id['a'] else 'a' if shooter in name2id['a'] and shooter not in name2id['h'] else ''
                if side:
                    tn.setdefault(sh.get('tm'), {}).setdefault(side, 0)
                    tn[sh.get('tm')][side] += 1
            tmside = {k: max(v, key=v.get) for k, v in tn.items()}
            for sh in c['shots']:
                who = (sh.get('who') or [None])
                shooter = norm(who[0]) if who and who[0] else ''
                side = tmside.get(sh.get('tm'), '')
                tx = (sh.get('tx') or '').lower()
                pid = name2id.get(side, {}).get(shooter, '') if side else ''
                ast = name2id.get(side, {}).get(norm(who[1]), '') if side and len(who) > 1 and who[1] else ''
                srows.append((m['id'], side, sh.get('c'), sh.get('per'), sh.get('ty'), pid, ast, sh.get('x'), sh.get('y'),
                              int('header' in tx), int('penalty' in tx or 'Penalty' in (sh.get('ty') or '')),
                              int('outside the box' in tx or 'long range' in tx or '35 yards' in tx),
                              int('six yard box' in tx or 'very close range' in tx),
                              int('centre of the box' in tx), int('fast break' in tx), int('free kick' in tx),   # goals say 'from a free kick', misses 'from a direct free kick'
                              # NB: never test for the bare word "corner" - goal texts say "to the bottom left corner" (outcome leak)
                              int('following a corner' in tx or 'set piece' in tx), int('Own Goal' in (sh.get('ty') or '')),
                              int('with a cross' in tx), int('through ball' in tx), int('difficult angle' in tx), int('side of the box' in tx)))
    jl.close()
    cols = ['mid', 'lg', 'season', 'date', 'tid', 'side', 'pid', 'name', 'starter', 'used', 't_in', 't_out', 'min', 'pos', 'posg', 'fp',
            'g', 'a', 'sh', 'sot', 'yc', 'rc', 'fc', 'fs', 'sv', 'og', 'off']
    P = pd.DataFrame(rows, columns=cols)
    P.to_csv(f'{ROOT}/data/player_match.csv.gz', index=False)
    T = pd.DataFrame(trows, columns=['mid', 'lg', 'season', 'date', 'tid', 'side', 'form', 'nst', 'nlisted', 'nused', 'hasstats', 'hasbench'])
    T.to_csv(f'{ROOT}/data/team_match.csv', index=False)
    S = pd.DataFrame(srows, columns=['mid', 'side', 'clock', 'per', 'type', 'pid', 'ast', 'x', 'y', 'header', 'pen', 'outbox', 'close', 'centre', 'fastbreak', 'dfk', 'setpiece', 'og', 'cross', 'through', 'diffangle', 'sidebox'])
    S.to_csv(f'{ROOT}/data/shots.csv.gz', index=False)
    print('compact files:', nfile, 'player rows:', len(P), 'team rows:', len(T), 'shots:', len(S))
    cov = T.groupby(['lg', 'season']).agg(teams=('mid', 'size'), ok11=('nst', lambda x: (x == 11).mean()), stats=('hasstats', 'mean'),
                                          bench=('hasbench', 'mean'), form=('form', lambda x: x.notna().mean()), used=('nused', 'mean'), listed=('nlisted', 'mean'))
    print(cov.round(3).to_string())
    sc = S.groupby(S.mid.map(dict(zip(T.mid, T.lg.astype(str) + T.season.astype(str))))).agg(n=('mid', 'size'), nm=('mid', 'nunique'), pid=('pid', lambda x: (x != '').mean()), xy=('x', lambda x: (x > 0).mean()))
    print(sc.round(3).to_string())


if __name__ == '__main__':
    main()
