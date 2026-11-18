#!/usr/bin/env python
"""Download ESPN match summaries and reduce each one to a compact roster JSON.

Usage:
  python fetch_rosters.py A 2014 2025        # league, first season, last season
  python fetch_rosters.py B 2014 2026
  python fetch_rosters.py cache              # reduce the 2026 Série A summaries already cached by the site repo

Resumable: a match whose compact file already exists is skipped.
Polite: at most 4 curl processes in flight, retries with exponential backoff, global pause when ESPN refuses.
Nothing but the compact JSON is kept (raw summaries are ~370 KB each).
"""
import json, os, re, subprocess, sys, time, threading
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'compact')
SHARED = '/home/user/bdv-lab/shared/'
CACHE = '/home/user/brasileiraodovitroncio/model/cache/sum/'
URL = 'https://site.api.espn.com/apis/site/v2/sports/soccer/{slug}/summary?event={id}'
os.makedirs(OUT, exist_ok=True)

CLK = re.compile(r"^(\d+)'(?:\+(\d+)')?")
SHOT_TYPES = ('shot', 'goal', 'penalty')


def clock(s):
    m = CLK.match(s or '')
    return [int(m.group(1)), int(m.group(2) or 0)] if m else None


def reduce_summary(d, mid):
    hdr = d.get('header') or {}
    comp = (hdr.get('competitions') or [{}])[0]
    gi = d.get('gameInfo') or {}
    out = {
        'id': str(mid),
        'date': comp.get('date'),
        'season': (hdr.get('season') or {}).get('year'),
        'status': ((comp.get('status') or {}).get('type') or {}).get('name'),
        'att': gi.get('attendance'),
        'ref': next((o.get('displayName') for o in gi.get('officials') or []), None),
        'comp': [{'tid': (c.get('team') or {}).get('id'), 'ha': c.get('homeAway'), 'score': c.get('score')}
                 for c in comp.get('competitors') or []],
        'teams': [], 'shots': [],
    }
    for t in d.get('rosters') or []:
        pl = []
        for p in t.get('roster') or []:
            a = p.get('athlete') or {}
            plays = []
            for e in p.get('plays') or []:
                c = clock((e.get('clock') or {}).get('displayValue'))
                typ = ('S' if e.get('substitution') else '') + ('R' if e.get('redCard') else '') + \
                      ('Y' if e.get('yellowCard') else '') + ('G' if e.get('didScore') else '') + \
                      ('A' if e.get('didAssist') else '') + ('P' if e.get('penaltyKick') else '') + \
                      ('O' if e.get('ownGoal') else '')
                plays.append([c, typ])
            rec = {
                'i': a.get('id'), 'n': a.get('displayName'),
                's': int(bool(p.get('starter'))), 'act': int(bool(p.get('active', True))),
                'si': int(bool(p.get('subbedIn'))), 'so': int(bool(p.get('subbedOut'))),
                'pos': (p.get('position') or {}).get('abbreviation'),
                'fp': p.get('formationPlace'), 'j': p.get('jersey'),
            }
            f = ((p.get('subbedInFor') or {}).get('athlete') or {}).get('id')
            if f: rec['inFor'] = f
            f = ((p.get('subbedOutFor') or {}).get('athlete') or {}).get('id')
            if f: rec['outFor'] = f
            if plays: rec['pl'] = plays
            if p.get('stats'):
                rec['st'] = {s['name']: s['value'] for s in p['stats'] if s.get('value')}
            pl.append(rec)
        out['teams'].append({'tid': (t.get('team') or {}).get('id'), 'ha': t.get('homeAway'),
                             'form': t.get('formation'), 'pl': pl})
    com = d.get('commentary') or []
    out['ncom'] = len(com)
    out['lineupLine'] = int(any('Lineups are announced' in (e.get('text') or '') for e in com[:5]))
    for e in com:
        p = e.get('play') or {}
        ty = ((p.get('type') or {}).get('text') or '')
        if not any(k in ty.lower() for k in SHOT_TYPES):
            continue
        parts = [((x.get('athlete') or {}).get('displayName')) for x in p.get('participants') or []]
        out['shots'].append({
            'c': (p.get('clock') or {}).get('value'), 'per': (p.get('period') or {}).get('number'),
            'ty': ty, 'tm': (p.get('team') or {}).get('displayName'), 'who': parts,
            'x': p.get('fieldPositionX'), 'y': p.get('fieldPositionY'), 'tx': e.get('text'),
        })
    return out


def save(rec):
    tmp = os.path.join(OUT, rec['id'] + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f, ensure_ascii=False, separators=(',', ':'))
    os.replace(tmp, os.path.join(OUT, rec['id'] + '.json'))


state = {'fail_streak': 0, 'pauses': 0, 'done': 0, 'failed': []}
lock = threading.Lock()


def fetch(job):
    slug, mid = job
    if os.path.exists(os.path.join(OUT, mid + '.json')):
        return
    for attempt in range(5):
        if state['pauses'] > 6:
            return
        try:
            r = subprocess.run(['curl', '-sS', '--compressed', '--max-time', '60', '-w', '\n%{http_code}', URL.format(slug=slug, id=mid)],
                               capture_output=True, timeout=90)
            body, _, code = r.stdout.rpartition(b'\n')
            code = code.decode().strip()
            if r.returncode == 0 and code == '200':
                rec = reduce_summary(json.loads(body), mid)
                rec['slug'] = slug
                save(rec)
                with lock:
                    state['fail_streak'] = 0; state['done'] += 1
                    if state['done'] % 200 == 0:
                        print(time.strftime('%H:%M:%S'), 'done', state['done'], flush=True)
                return
            err = f'http {code} rc {r.returncode} {r.stderr[:120]!r}'
        except Exception as ex:  # noqa
            err = repr(ex)[:200]
        with lock:
            state['fail_streak'] += 1
            streak = state['fail_streak']
        print(time.strftime('%H:%M:%S'), 'retry', mid, attempt, err, flush=True)
        if streak >= 12:
            with lock:
                if state['fail_streak'] >= 12:
                    state['pauses'] += 1; state['fail_streak'] = 0
                    print('ESPN refusing; global pause 90s, pause #', state['pauses'], flush=True)
            time.sleep(90)
        else:
            time.sleep(2 * 2 ** attempt)
    with lock:
        state['failed'].append(mid)


def main():
    if sys.argv[1] == 'cache':
        n = 0
        for f in sorted(os.listdir(CACHE)):
            mid = f[:-5]
            if os.path.exists(os.path.join(OUT, mid + '.json')):
                continue
            rec = reduce_summary(json.load(open(CACHE + f)), mid); rec['slug'] = 'bra.1'; save(rec); n += 1
        print('reduced from cache:', n)
        return
    lg, y0, y1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    M = json.load(open(SHARED + 'matches.json'))
    slug = 'bra.1' if lg == 'A' else 'bra.2'
    jobs = [(slug, str(m['id'])) for m in M if m['lg'] == lg and y0 <= m['season'] <= y1 and m['status'] != 'pre']
    jobs = [j for j in jobs if not os.path.exists(os.path.join(OUT, j[1] + '.json'))]
    print('to fetch:', len(jobs), flush=True)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(fetch, jobs))
    print('finished; ok', state['done'], 'failed', len(state['failed']), state['failed'][:30], flush=True)


if __name__ == '__main__':
    main()
