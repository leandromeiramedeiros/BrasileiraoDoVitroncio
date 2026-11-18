"""Parse every raw ESPN scoreboard in raw/ (other competitions) and merge with the shared
Série A/B base into one chronologically ordered all-competitions match file.

Output: all_matches.json  (list of dicts, sorted by date) and all_teams.json (id -> name).
Odds blocks present in ESPN payloads are deliberately NOT read (protocol rule 4).

Run with:  python -I code/parse_all.py
"""
import json, glob, os, re, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, 'raw')
SHARED = '/home/user/bdv-lab/shared/'

COMP = {  # slug -> short code
    'bra.copa_do_brazil': 'CDB', 'conmebol.libertadores': 'LIB', 'conmebol.sudamericana': 'SUD',
    'bra.camp.paulista': 'SP', 'bra.camp.carioca': 'RJ', 'bra.camp.gaucho': 'RS', 'bra.camp.mineiro': 'MG',
    'bra.copa_do_nordeste': 'CNE', 'conmebol.recopa': 'REC', 'bra.supercopa_do_brazil': 'SUP',
    'fifa.cwc': 'CWC', 'fifa.intercontinental_cup': 'ICC', 'bra.3': 'C', 'bra.4': 'D',
}

def status_of(st):
    t = (st or {}).get('type') or {}
    name, state = t.get('name', ''), t.get('state', 'pre')
    if name in ('STATUS_POSTPONED', 'STATUS_CANCELED', 'STATUS_ABANDONED', 'STATUS_SUSPENDED'): return 'off'
    if state == 'post': return 'post'
    if state == 'in': return 'in'
    return 'pre'

def stat(lst, name):
    for s in lst or []:
        if s.get('name') == name:
            try: return float(s.get('displayValue'))
            except Exception: return None
    return None

def parse_event(e, comp):
    cs = e.get('competitions') or []
    if not cs: return None, None
    c = cs[0]
    home = next((x for x in c['competitors'] if x.get('homeAway') == 'home'), None)
    away = next((x for x in c['competitors'] if x.get('homeAway') == 'away'), None)
    if not home or not away: return None, None
    st = c.get('status') or e.get('status') or {}
    status = status_of(st)
    reds = {'h': 0, 'a': 0}
    for d in c.get('details') or []:
        t = (d.get('type') or {}).get('text', '')
        if d.get('redCard') or t == 'Red Card':
            tid = (d.get('team') or {}).get('id')
            if tid == home['team']['id']: reds['h'] += 1
            elif tid == away['team']['id']: reds['a'] += 1
    hs, as_ = home.get('statistics'), away.get('statistics')
    sh_h, sh_a = stat(hs, 'totalShots'), stat(as_, 'totalShots')
    venue = c.get('venue') or {}
    addr = venue.get('address') or {}
    series = c.get('series') or {}
    played = status not in ('pre', 'off')
    def sc(x):
        try: return int(x.get('score') or 0)
        except Exception: return 0
    m = {
        'id': str(e['id']), 'comp': comp, 'season': (e.get('season') or {}).get('year') or int(e['date'][:4]),
        'stage': (e.get('season') or {}).get('slug'), 'date': e['date'], 'status': status,
        'statusName': (st.get('type') or {}).get('name', ''),
        'h': str(home['team']['id']), 'a': str(away['team']['id']),
        'hg': sc(home) if played else None, 'ag': sc(away) if played else None,
        'neutral': bool(c.get('neutralSite')),
        'leg': (c.get('leg') or {}).get('value'), 'legs': series.get('totalCompetitions'),
        'series': series.get('title'),
        'venue': venue.get('fullName') or (e.get('venue') or {}).get('displayName'),
        'city': addr.get('city'), 'country': addr.get('country'),
        'att': c.get('attendance') or None,
        'rh': reds['h'], 'ra': reds['a'],
        'sh_h': sh_h, 'sh_a': sh_a, 'sot_h': stat(hs, 'shotsOnTarget'), 'sot_a': stat(as_, 'shotsOnTarget'),
    }
    teams = {}
    for x in (home, away):
        t = x['team']
        teams[str(t['id'])] = {'name': t.get('shortDisplayName') or t.get('displayName') or t.get('name'),
                               'full': t.get('displayName') or t.get('name'), 'abbr': t.get('abbreviation')}
    return m, teams

def main():
    allm, teams, seen = [], {}, set()
    for f in sorted(glob.glob(os.path.join(RAW, '*.json'))):
        slug, y = os.path.basename(f)[:-5].rsplit('_', 1)
        comp = COMP[slug]
        d = json.load(open(f))
        for e in d.get('events') or []:
            m, t = parse_event(e, comp)
            if not m or m['id'] in seen: continue
            seen.add(m['id']); allm.append(m)
            for k, v in t.items(): teams[k] = v
    n_other = len(allm)
    # shared league base (authoritative for Série A/B)
    shared = json.load(open(SHARED + 'matches.json'))
    st = json.load(open(SHARED + 'teams.json'))
    for k, v in st.items(): teams[k] = {'name': v['name'], 'full': v['full'], 'abbr': v.get('abbr')}
    for m in shared:
        rh = sum(1 for e in m['ev'] if e['t'] == 'R' and e['s'] == 'h'); ra = sum(1 for e in m['ev'] if e['t'] == 'R' and e['s'] == 'a')
        s = m['stats'] if m.get('statsOk') else None
        allm.append({'id': str(m['id']), 'comp': m['lg'], 'season': m['season'], 'stage': 'league', 'date': m['date'],
                     'status': m['status'], 'statusName': '', 'h': str(m['h']), 'a': str(m['a']), 'hg': m['hg'], 'ag': m['ag'],
                     'neutral': False, 'leg': None, 'legs': None, 'series': None, 'venue': m['venue'], 'city': m['city'],
                     'country': 'Brazil', 'att': None, 'rh': rh, 'ra': ra,
                     'sh_h': s['h']['sh'] if s else None, 'sh_a': s['a']['sh'] if s else None,
                     'sot_h': s['h']['sot'] if s else None, 'sot_a': s['a']['sot'] if s else None})
    allm.sort(key=lambda m: (m['date'], m['id']))
    json.dump(allm, open(os.path.join(HERE, 'all_matches.json'), 'w'), ensure_ascii=False)
    json.dump(teams, open(os.path.join(HERE, 'all_teams.json'), 'w'), ensure_ascii=False, indent=0)
    print('other competitions:', n_other, ' total merged:', len(allm), ' teams:', len(teams))

if __name__ == '__main__':
    main()
