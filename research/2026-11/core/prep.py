"""Builds flat numpy arrays from shared/matches.json (read-only) -> core/data.npz.

Everything here is a deterministic re-encoding of the raw match records; no fitted quantity.
Run:  .venv/bin/python prep.py
"""
import json
import numpy as np

S = '/home/user/bdv-lab/shared/'
OUT = '/home/user/bdv-lab/core/data.npz'
MAXE = 16  # max goal events stored per match


def main():
    M = json.load(open(S + 'matches.json'))
    M = [m for m in M if m['status'] != 'off']          # same filter as shared/export.mjs
    n = len(M)
    tid = {}
    for m in M:
        for k in (m['h'], m['a']):
            if k not in tid:
                tid[k] = len(tid)
    ids = np.array([str(m['id']) for m in M])
    h = np.array([tid[m['h']] for m in M], np.int64)
    a = np.array([tid[m['a']] for m in M], np.int64)
    lg = np.array([0 if m['lg'] == 'A' else 1 for m in M], np.int64)
    season = np.array([m['season'] for m in M], np.int64)
    post = np.array([m['status'] == 'post' for m in M])
    date = np.array([np.datetime64(m['date'].replace('Z', '')) for m in M])
    day = (date - np.datetime64('2006-01-01T00:00')) / np.timedelta64(1, 'D')
    hg = np.array([m['hg'] if m['status'] == 'post' else -1 for m in M], np.int64)
    ag = np.array([m['ag'] if m['status'] == 'post' else -1 for m in M], np.int64)
    evOk = np.array([bool(m['evOk']) and len(m['ev'] or []) > 0 for m in M])
    hasEv = np.array([len(m['ev'] or []) > 0 for m in M])
    statsOk = np.array([bool(m['statsOk']) for m in M])

    # ---- event-derived counts -------------------------------------------------
    # goal categories per side (index 0 = home, 1 = away)
    cat = ['all', 'adv', 'dis', 'pen', 'own', 'late_lead2', 'late_trail2', 'lead2', 'trail2', 'late80', 'lead1_late']
    G = {c: np.zeros((n, 2)) for c in cat}
    yc = np.zeros((n, 2)); rc = np.zeros((n, 2))
    # fraction of the match each side spent with one man MORE than the opponent
    fadv = np.zeros((n, 2))
    # goal event table for hazard fitting: minute, added, side(0/1), type; reds likewise
    gm = np.full((n, MAXE), -1, np.int64); gx = np.zeros((n, MAXE), np.int64); gs = np.zeros((n, MAXE), np.int64)
    rm = np.full((n, 8), -1, np.int64); rx = np.zeros((n, 8), np.int64); rs = np.zeros((n, 8), np.int64)
    T = 96.0
    for i, m in enumerate(M):
        ev = m['ev'] or []
        sc = [0, 0]; red = [0, 0]
        ng = 0; nr = 0
        last_t = 0.0
        for e in ev:
            s = 0 if e['s'] == 'h' else 1
            t = e['t']
            mm = e['m']; xx = e.get('x', 0) or 0
            tt = min(T, mm + xx + (3 if mm > 45 else 0))  # rough clock incl. ~3' first-half stoppage
            if t == 'Y':
                yc[i, s] += 1
            elif t == 'R':
                # accumulate man-advantage time up to now
                if red[0] != red[1]:
                    fadv[i, 0 if red[1] > red[0] else 1] += max(0.0, tt - last_t)
                last_t = tt
                red[s] += 1; rc[i, s] += 1
                if nr < 8:
                    rm[i, nr] = mm; rx[i, nr] = xx; rs[i, nr] = s; nr += 1
            elif t in 'GPO':
                o = 1 - s
                d = sc[s] - sc[o]
                G['all'][i, s] += 1
                if red[o] > red[s]: G['adv'][i, s] += 1
                if red[s] > red[o]: G['dis'][i, s] += 1
                if t == 'P': G['pen'][i, s] += 1
                if t == 'O': G['own'][i, s] += 1
                if d >= 2: G['lead2'][i, s] += 1
                if d <= -2: G['trail2'][i, s] += 1
                if mm >= 75 and d >= 2: G['late_lead2'][i, s] += 1
                if mm >= 75 and d <= -2: G['late_trail2'][i, s] += 1
                if mm >= 80: G['late80'][i, s] += 1
                if mm >= 80 and d >= 1: G['lead1_late'][i, s] += 1
                sc[s] += 1
                if ng < MAXE:
                    gm[i, ng] = mm; gx[i, ng] = xx; gs[i, ng] = s; ng += 1
        if red[0] != red[1]:
            fadv[i, 0 if red[1] > red[0] else 1] += max(0.0, T - last_t)
    fadv /= T

    st = {k: np.full((n, 2), np.nan) for k in ['sh', 'sot', 'co', 'fo', 'po']}
    for i, m in enumerate(M):
        if m['stats'] is None:
            continue
        for j, s in enumerate('ha'):
            for k in st:
                v = m['stats'][s].get(k)
                if v is not None:
                    st[k][i, j] = v

    venues = sorted({m['venue'] or '' for m in M}); vid = {v: i for i, v in enumerate(venues)}
    cities = sorted({m['city'] or '' for m in M}); cid = {v: i for i, v in enumerate(cities)}
    venue = np.array([vid[m['venue'] or ''] for m in M], np.int64)
    city = np.array([cid[m['city'] or ''] for m in M], np.int64)

    np.savez_compressed(
        OUT, ids=ids, h=h, a=a, lg=lg, season=season, post=post, day=day, hg=hg, ag=ag, evOk=evOk, hasEv=hasEv,
        statsOk=statsOk, yc=yc, rc=rc, fadv=fadv, gm=gm, gx=gx, gs=gs, rm=rm, rx=rx, rs=rs,
        venue=venue, city=city, venue_names=np.array(venues), city_names=np.array(cities),
        team_ids=np.array(list(tid.keys())), datestr=np.array([m['date'] for m in M]),
        **{'g_' + c: G[c] for c in cat}, **{'st_' + k: v for k, v in st.items()})
    print('matches', n, 'teams', len(tid), 'post', post.sum(), 'venues', len(venues))


if __name__ == '__main__':
    main()
