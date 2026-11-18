"""Pre-match context features for every Série A / Série B match (keyed by ESPN match id).

  schedule (any competition): rest, congestion, proximity of cup/continental ties, cups still alive, travel
  geography: away-team trip, displaced home matches, latitude (climate) shifts, derbies, kick-off hour
  stakes: league table as it stood before kick-off (gaps, mathematical flags, pressure indices)
  season context: round, closed-doors (COVID) period

Everything uses only information available before kick-off (results of earlier matches; the fixture list of
future matches, which is public in advance).  Output: features_context.csv
"""
import os, sys, math, bisect, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

M = C.load_all()
M = M[M.status.isin(['post', 'pre', 'in'])].reset_index(drop=True)   # postponed/cancelled listings are ignored
lat = M.vkey.map(lambda k: C.coords(k)[0]).to_numpy(); lon = M.vkey.map(lambda k: C.coords(k)[1]).to_numpy()
M['vlat'], M['vlon'] = lat, lon
M['vcountry'] = M.vkey.map(lambda k: C.CITIES[k][3] if k in C.CITIES else None)
# foreign venue when city unknown: continental away match against a non-Brazilian club
dom = M[M.comp.isin(C.DOMESTIC)]; BR = set(dom.h) | set(dom.a)
M['abroad'] = [(vc is not None and vc != 'BR') or (vc is None and h not in BR) for vc, h in zip(M.vcountry, M.h)]

def stage_kind(comp, stage):
    s = stage or ''
    if comp in ('A', 'B', 'C', 'D'): return 'lg'
    if comp in ('LIB', 'SUD'):
        if 'group' in s: return 'cont_grp'
        return 'cont_big' if any(k in s for k in ('round-of-16', 'quarter', 'semi', 'final', 'knockout')) else 'cont_pre'
    if comp in ('CWC', 'ICC', 'REC', 'SUP'): return 'cont_big'
    if comp == 'CDB':
        return 'cup_big' if any(k in s for k in ('round-of-16', 'quarter', 'semi', 'final')) else 'cup_early'
    if any(k in s for k in ('semi', 'final')) and 'quarter' not in s: return 'st_big'
    return 'st'
M['kind'] = [stage_kind(c, s) for c, s in zip(M.comp, M.stage)]

# ---------------------------------------------------------------- per-team timelines
TL = collections.defaultdict(list)   # team -> list of (day, row index, is_home)
for i, (h, a, d) in enumerate(zip(M.h, M.a, M.day)):
    TL[h].append((d, i, True)); TL[a].append((d, i, False))
for t in TL: TL[t].sort()
DAYS = {t: [x[0] for x in v] for t, v in TL.items()}
comp_a, kind_a, status_a, season_a, abroad_a, day_a = (M[c].to_numpy() for c in ('comp', 'kind', 'status', 'season', 'abroad', 'day'))
abase_lat = M.abase.map(lambda k: C.coords(k)[0]).to_numpy(); abase_lon = M.abase.map(lambda k: C.coords(k)[1]).to_numpy()
hbase_lat = M.hbase.map(lambda k: C.coords(k)[0]).to_numpy(); hbase_lon = M.hbase.map(lambda k: C.coords(k)[1]).to_numpy()

def hav(a1, o1, a2, o2):
    if any(x != x for x in (a1, o1, a2, o2)): return np.nan
    return float(C.haversine(a1, o1, a2, o2))

def sched(team, i, is_home):
    """Schedule features of `team` for match row i."""
    t = day_a[i]; tl = TL[team]; days = DAYS[team]
    k = bisect.bisect_left(days, t - 1e-6)          # matches strictly before
    prev = [x for x in tl[max(0, k - 14):k] if status_a[x[1]] == 'post']
    nxt = tl[bisect.bisect_right(days, t + 1e-6):][:1]
    base = (hbase_lat[i], hbase_lon[i]) if is_home else (abase_lat[i], abase_lon[i])
    f = {}
    rest = t - prev[-1][0] if prev else 99.0
    f['rest'] = min(rest, 15.0)
    f['fresh'] = float(rest > 20)
    f['n7'] = sum(1 for x in prev if t - x[0] <= 7.5); f['n14'] = sum(1 for x in prev if t - x[0] <= 14.5)
    f['n30'] = sum(1 for x in tl[max(0, k - 20):k] if status_a[x[1]] == 'post' and t - x[0] <= 30.5)
    f['n14_lg'] = sum(1 for x in prev if t - x[0] <= 14.5 and kind_a[x[1]] == 'lg')
    pk = kind_a[prev[-1][1]] if prev and rest <= 4.6 else None
    f['prev_nonlg4'] = float(pk is not None and pk != 'lg')
    f['prev_big4'] = float(pk in ('cont_big', 'cup_big', 'st_big'))
    f['prev_cont4'] = float(pk in ('cont_big', 'cont_grp', 'cont_pre'))
    f['prev_abroad5'] = float(bool(prev) and rest <= 5.6 and bool(abroad_a[prev[-1][1]]))
    f['prev_away4'] = float(bool(prev) and rest <= 4.6 and not prev[-1][2])
    if nxt:
        gap = nxt[0][0] - t; nk = kind_a[nxt[0][1]] if gap <= 4.6 else None
        f['nxt'] = min(gap, 15.0)
    else:
        nk = None; f['nxt'] = 15.0
    f['next_nonlg4'] = float(nk is not None and nk != 'lg')
    f['next_big4'] = float(nk in ('cont_big', 'cup_big', 'st_big'))
    f['next_cont4'] = float(nk in ('cont_big', 'cont_grp', 'cont_pre'))
    f['sandwich'] = float(rest <= 4.6 and f['nxt'] <= 4.6)
    # cups still alive (a later fixture of the same season exists in that competition)
    later = tl[bisect.bisect_right(days, t + 1e-6):]
    yr = season_a[i]
    alive = {c: 0.0 for c in ('CDB', 'LIB', 'SUD')}
    for x in later:
        c = comp_a[x[1]]
        if c in alive and season_a[x[1]] == yr: alive[c] = 1.0
    f['alive_cdb'] = alive['CDB']; f['alive_cont'] = max(alive['LIB'], alive['SUD']); f['alive_lib'] = alive['LIB']
    f['alive_n'] = alive['CDB'] + f['alive_cont']
    # travel: leg from the previous venue (if within 10 days, else from home base) to this venue; km over 14 days
    vl, vo = M.vlat.iat[i], M.vlon.iat[i]
    if prev and rest <= 10: f['trav_prev'] = hav(M.vlat.iat[prev[-1][1]], M.vlon.iat[prev[-1][1]], vl, vo)
    else: f['trav_prev'] = hav(base[0], base[1], vl, vo)
    pts = [(M.vlat.iat[x[1]], M.vlon.iat[x[1]]) for x in prev if t - x[0] <= 14.5] + [(vl, vo)]
    km = 0.0
    for p, q in zip(pts[:-1], pts[1:]):
        d = hav(p[0], p[1], q[0], q[1]); km += 0.0 if d != d else d
    f['km14'] = km
    return f

# ---------------------------------------------------------------- league tables (stakes)
def stakes(L, lag=0.08):
    """L: league matches of one (comp, season), chronological. Yields dict per match (both sides).
    lag: a result enters the table `lag` days after ITS kick-off (0.08 d ~ 2 h = final whistle; 4.0 = what a
    batch job run 4 days before kick-off would know)."""
    teams = sorted(set(L.h) | set(L.a)); N = len(teams)
    total = 2 * (N - 1)
    pts = dict.fromkeys(teams, 0); gp = dict.fromkeys(teams, 0); gd = dict.fromkeys(teams, 0)
    comp = L.comp.iat[0]
    ntop = 4; nrel = 4
    rows = {}
    # process in kick-off order; results become visible only after the match day slot (use 0.08 d ~ 2 h lag)
    pending = []
    def flush(t):
        nonlocal pending
        keep = []
        for (d, h, a, hg, ag) in pending:
            if d + lag <= t:
                gp[h] += 1; gp[a] += 1; gd[h] += hg - ag; gd[a] += ag - hg
                if hg > ag: pts[h] += 3
                elif hg < ag: pts[a] += 3
                else: pts[h] += 1; pts[a] += 1
            else: keep.append((d, h, a, hg, ag))
        pending = keep
    for id_, d, h, a, hg, ag, status in L[['id', 'day', 'h', 'a', 'hg', 'ag', 'status']].itertuples(index=False, name=None):
        flush(d)
        order = sorted(teams, key=lambda x: (-pts[x], -gd[x]))
        P = np.array([pts[x] for x in order]); MX = np.array([pts[x] + 3 * (total - gp[x]) for x in order])
        f = {}
        for side, t in (('h', h), ('a', a)):
            p = pts[t]; rem = total - gp[t]; mx = p + 3 * rem; pos = order.index(t) + 1
            others_P = np.delete(P, pos - 1); others_MX = np.delete(MX, pos - 1)
            def margin(k):   # signed points margin to the cut between place k and k+1 (+ = inside)
                return p - P[k] if pos <= k else p - P[k - 1]
            m_title = p - others_P.max(); m_top = margin(ntop); m_g6 = margin(6); m_rel = margin(N - nrel)
            sd = 1.8 * math.sqrt(max(rem, 0)) + 0.5
            pr = lambda m: math.exp(-0.5 * (m / sd) ** 2)
            can_title = mx >= others_P.max(); clinched = p > others_MX.max()
            can_top = (others_P > mx).sum() < ntop if comp == 'B' else (others_P > mx).sum() < 6
            sure_top = (others_MX >= p).sum() < (ntop if comp == 'B' else 6)
            safe = (others_MX < p).sum() >= nrel
            releg = (others_P > mx).sum() >= N - nrel
            dead = releg or clinched or (safe and not can_top) or (safe and sure_top and not can_title)
            g = gp[t]
            f.update({f'gp_{side}': g, f'pos_{side}': pos, f'ppg_{side}': p / g if g else np.nan,
                      f'm_title_{side}': m_title, f'm_top_{side}': m_top if comp == 'B' else m_g6, f'm_rel_{side}': m_rel,
                      f'press_title_{side}': pr(m_title), f'press_top_{side}': pr(m_top if comp == 'B' else m_g6),
                      f'press_g4_{side}': pr(m_top), f'press_rel_{side}': pr(m_rel),
                      f'safe_{side}': float(safe), f'releg_{side}': float(releg), f'clinched_{side}': float(clinched),
                      f'can_title_{side}': float(can_title), f'can_top_{side}': float(can_top), f'dead_{side}': float(dead)})
        rows[id_] = f
        if status == 'post' and hg == hg and hg is not None: pending.append((d, h, a, int(hg), int(ag)))
    return rows

# ---------------------------------------------------------------- assemble
LG = M[M.comp.isin(['A', 'B'])]
idx_of = {id_: i for i, id_ in zip(LG.index, LG.id)}
out = {}
for i in LG.index:
    h, a = M.h.iat[i], M.a.iat[i]
    f = {}
    for side, team, ish in (('h', h, True), ('a', a, False)):
        for k, v in sched(team, i, ish).items(): f[f'{k}_{side}'] = v
    vl, vo = M.vlat.iat[i], M.vlon.iat[i]
    dist = hav(abase_lat[i], abase_lon[i], vl, vo); hd = hav(hbase_lat[i], hbase_lon[i], vl, vo)
    f['dist_km'] = dist; f['home_disp_km'] = hd
    f['lat_diff'] = vl - abase_lat[i]; f['lon_diff'] = vo - abase_lon[i]
    f['v_lat'] = vl; f['a_lat'] = abase_lat[i]
    vuf = C.CITIES[M.vkey.iat[i]][2] if M.vkey.iat[i] in C.CITIES else None
    auf = C.CITIES[M.abase.iat[i]][2] if M.abase.iat[i] in C.CITIES else None
    huf = C.CITIES[M.hbase.iat[i]][2] if M.hbase.iat[i] in C.CITIES else None
    f['same_city'] = float(dist == dist and dist < 40)
    f['same_state'] = float(auf is not None and auf == huf)
    ts = M.ts.iat[i]
    f['local_hour'] = (ts.hour + ts.minute / 60 - 3) % 24
    f['month'] = ts.month; f['dow'] = ts.day_of_week
    f['closed'] = float(pd.Timestamp('2020-03-15', tz='UTC') <= ts < pd.Timestamp('2021-10-02', tz='UTC'))
    out[M.id.iat[i]] = f
for (comp, season), L in LG.groupby(['comp', 'season']):
    Ls = L.sort_values(['day', 'id'])
    for id_, f in stakes(Ls).items(): out[id_].update(f)
    for id_, f in stakes(Ls, lag=4.0).items():      # stale table (weekly batch run ~4 days before kick-off)
        for side in 'ha':
            out[id_][f'releg_lag4_{side}'] = f[f'releg_{side}']; out[id_][f'clinched_lag4_{side}'] = f[f'clinched_{side}']
F = pd.DataFrame.from_dict(out, orient='index'); F.index.name = 'id'
F.insert(0, 'comp', LG.set_index('id').comp.reindex(F.index)); F.insert(1, 'season', LG.set_index('id').season.reindex(F.index))
F.to_csv(os.path.join(C.HERE, 'features_context.csv'))
print(F.shape)
pd.set_option('display.width', 200); pd.set_option('display.max_rows', 200)
if '-v' in sys.argv: print(F[(F.comp == 'A') & (F.season >= 2013)].describe().T[['count', 'mean', 'std', 'min', 'max']].round(3).to_string())
