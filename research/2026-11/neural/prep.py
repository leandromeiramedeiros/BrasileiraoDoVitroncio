"""Builds data.npz: match table, per-(match, side) history tokens and history index.

Everything in a token describes a match that is ALREADY finished from the point of view of the
team; the pre-match v1 expectations (lh, la, sh_*, ...) of that earlier game come from
baseline_features.csv (walk-forward, pre-match for that game).  The history of a target match
contains only matches whose kick-off is strictly earlier than the target's kick-off."""
import json
import numpy as np
import pandas as pd
from chain import chain_probs

S = '/home/user/bdv-lab/shared/'
OUT = '/home/user/bdv-lab/neural/data.npz'
NMAX = 40

M = json.load(open(S + 'matches.json'))
F = pd.read_csv(S + 'baseline_features.csv', dtype={'id': str}).set_index('id')
M = [m for m in M if m['id'] in F.index]           # drops the 2 'off' matches (no v1 row)
n = len(M)
F = F.loc[[m['id'] for m in M]].reset_index()
date = pd.to_datetime([m['date'] for m in M], utc=True)
day = ((date - pd.Timestamp('2006-01-01', tz='UTC')) / pd.Timedelta(days=1)).to_numpy(float)
assert np.all(np.diff(day) >= 0), 'matches must be chronological'
teams = sorted({m['h'] for m in M} | {m['a'] for m in M}); tix = {t: i for i, t in enumerate(teams)}
post = np.array([m['status'] == 'post' for m in M])
lgB = np.array([m['lg'] == 'B' for m in M], float)
season = np.array([m['season'] for m in M])
h = np.array([tix[m['h']] for m in M]); a = np.array([tix[m['a']] for m in M])
hg = np.array([m['hg'] if m['status'] == 'post' else -1 for m in M], float)
ag = np.array([m['ag'] if m['status'] == 'post' else -1 for m in M], float)
lh = F.lh.to_numpy(); la = F.la.to_numpy()
pv1 = np.vstack([chain_probs(lh[s:s + 2048], la[s:s + 2048]) for s in range(0, n, 2048)])   # v1 1X2 for every match

# ---- per-match raw quantities, home perspective ------------------------------------------------
def minute(e):
    return min(90.0, float(e['m'])) if e['m'] <= 90 else 90.0

ev = np.zeros((n, 2, 14), np.float32)   # per side: see names below
EV_NAMES = ['g_h1', 'g_late', 'g_close', 'g_11v11', 'pen', 'og_for', 'yel', 'red', 'mandown_frac', 'lead_frac', 'first_goal', 'first_goal_min', 'g_first30', 'last_goal_min']
evok = np.zeros(n, np.float32)
for i, m in enumerate(M):
    if not post[i]: continue
    E = sorted(m.get('ev') or [], key=lambda e: (e['m'], e.get('x') or 0))
    ok = bool(m.get('evOk'))
    evok[i] = float(ok)
    sc = [0, 0]; red = [0, 0]; last_t = 0.0; lead = [0.0, 0.0]; first_red = [None, None]; first = None
    for e in E:
        s = 0 if e['s'] == 'h' else 1
        t = minute(e)
        if e['t'] in ('G', 'P', 'O'):
            if not ok: continue
            # time in lead until this goal
            if sc[0] > sc[1]: lead[0] += t - last_t
            elif sc[1] > sc[0]: lead[1] += t - last_t
            last_t = t
            d = sc[s] - sc[1 - s]
            if e['m'] <= 45: ev[i, s, 0] += 1
            if e['m'] >= 76: ev[i, s, 1] += 1
            if abs(d) <= 1: ev[i, s, 2] += 1
            if red[0] == red[1]: ev[i, s, 3] += 1
            if e['t'] == 'P': ev[i, s, 4] += 1
            if e['t'] == 'O': ev[i, s, 5] += 1
            if e['m'] <= 30: ev[i, s, 12] += 1
            ev[i, s, 13] = t / 90
            if first is None:
                first = s; ev[i, s, 10] = 1; ev[i, 1 - s, 10] = -1; ev[i, :, 11] = t / 90
            sc[s] += 1
        elif e['t'] == 'Y': ev[i, s, 6] += 1
        elif e['t'] == 'R':
            ev[i, s, 7] += 1; red[s] += 1
            if first_red[s] is None: first_red[s] = t
    if ok:
        if sc[0] > sc[1]: lead[0] += 90 - last_t
        elif sc[1] > sc[0]: lead[1] += 90 - last_t
        ev[i, 0, 9] = lead[0] / 90; ev[i, 1, 9] = lead[1] / 90
    for s in (0, 1):
        if first_red[s] is not None: ev[i, s, 8] = (90 - first_red[s]) / 90

st = np.zeros((n, 2, 5), np.float32); stok = np.zeros(n, np.float32)   # sh, sot, co, fo, po
for i, m in enumerate(M):
    if post[i] and m.get('statsOk'):
        ok = True
        for s, k in enumerate('ha'):
            v = [m['stats'][k].get(q) for q in ('sh', 'sot', 'co', 'fo', 'po')]
            if v[0] is None or v[1] is None: ok = False; break
            st[i, s] = [x if x is not None else np.nan for x in v]
        stok[i] = float(ok)
st = np.nan_to_num(st, nan=0.0)
co_ok = (stok > 0) & (st[:, :, 2].sum(1) > 0); fo_ok = (stok > 0) & (st[:, :, 3].sum(1) > 0); po_ok = (stok > 0) & (st[:, :, 4].sum(1) > 50)

# ---- history index: last NMAX finished matches of the team with kick-off strictly before the target
hist = -np.ones((n, 2, NMAX), np.int32)
per_team = {t: [] for t in range(len(teams))}
i = 0
while i < n:
    j = i
    while j < n and day[j] == day[i]: j += 1          # block of simultaneous kick-offs
    for k in range(i, j):
        for s, t in enumerate((h[k], a[k])):
            L = per_team[t][-NMAX:]
            if L: hist[k, s, NMAX - len(L):] = L       # right-aligned, most recent last
    for k in range(i, j):
        if post[k]:
            per_team[h[k]].append(2 * k); per_team[a[k]].append(2 * k + 1)
    i = j
tok_team = np.stack([h, a], 1).reshape(-1); tok_day = np.repeat(day, 2); tok_season = np.repeat(season, 2)
# sanity: every history token is strictly earlier
for s in (0, 1):
    idx = hist[:, s, :]; ok = idx >= 0
    assert np.all(tok_day[np.where(ok, idx, 0)][ok] < np.repeat(day[:, None], NMAX, 1)[ok])
    assert np.all(tok_team[np.where(ok, idx, 0)][ok] == np.repeat(np.stack([h, a], 1)[:, s][:, None], NMAX, 1)[ok])


# ---- simple pre-match form of each side (last 20 finished matches; nothing from v1) - used as a crude, v1-free
#      opponent-strength descriptor inside the tokens of the "pure" models
_res = np.sign(np.stack([hg - ag, ag - hg], 1)).reshape(-1); _pts = np.where(_res > 0, 3.0, np.where(_res == 0, 1.0, 0.0))
_gd = np.clip(np.stack([hg - ag, ag - hg], 1).reshape(-1), -4, 4); _lgB = np.repeat(lgB, 2)
w20 = hist[:, :, -20:]; ok20 = w20 >= 0; i20 = np.where(ok20, w20, 0); n20 = ok20.sum(2)
form_ppg = np.where(n20 > 0, (_pts[i20] * ok20).sum(2) / np.maximum(n20, 1), 1.35)          # (n,2)
form_gd = np.where(n20 > 0, (_gd[i20] * ok20).sum(2) / np.maximum(n20, 1), 0.0)
form_B = np.where(n20 > 0, (_lgB[i20] * ok20).sum(2) / np.maximum(n20, 1), np.stack([lgB, lgB], 1))   # share of those games played in Série B
form_n = n20 / 20.0

# ---- tokens: row 2*i + side (side 0 = home team of match i) --------------------------------------
exp = {c: np.stack([F[c + '_lh'].to_numpy(), F[c + '_la'].to_numpy()], 1) for c in ('sh', 'sot', 'co', 'yc', 'fo')}
gl = np.stack([F.g_lh.to_numpy(), F.g_la.to_numpy()], 1)
lam = np.stack([lh, la], 1); goals = np.stack([hg, ag], 1)
names, cols = [], []
def add(name, arr2):   # arr2: (n, 2) value for [home-team token, away-team token]
    names.append(name); cols.append(np.asarray(arr2, np.float32).reshape(n, 2))
sw = lambda x: x[:, ::-1]
gf = goals; ga = sw(goals)
res = np.sign(gf - ga)
exp_pts = np.stack([3 * pv1[:, 0] + pv1[:, 1], 3 * pv1[:, 2] + pv1[:, 1]], 1)
pts = np.where(res > 0, 3, np.where(res == 0, 1, 0))
# --- group "base": always available
add('is_home', np.stack([np.ones(n), np.zeros(n)], 1))
add('lgB', np.stack([lgB, lgB], 1))
add('gf', np.minimum(gf, 6) / 2); add('ga', np.minimum(ga, 6) / 2)
add('win', res > 0); add('draw', res == 0); add('loss', res < 0)
add('gd', np.clip(gf - ga, -4, 4) / 2)
# --- group "opp": v1-free opponent descriptor (opponent's own pre-match form over its last 20 games)
add('opp_ppg', (sw(form_ppg) - 1.35) * 2); add('opp_gd', sw(form_gd) * 2); add('opp_B', sw(form_B)); add('opp_n', sw(form_n))
# --- group "v1": opponent strength / surprise versus v1's pre-match expectation of that game
add('v1_loglf', np.log(lam)); add('v1_logla', np.log(sw(lam)))
add('v1_glf', np.log(gl)); add('v1_gla', np.log(sw(gl)))
add('v1_exp_pts', exp_pts / 3)
add('sur_gf', np.clip(gf - lam, -3, 4) / 1.5); add('sur_ga', np.clip(ga - sw(lam), -3, 4) / 1.5)
add('sur_pts', (pts - exp_pts) / 3)
add('sur_win', (res > 0) - np.stack([pv1[:, 0], pv1[:, 2]], 1))
# --- group "ev": minute-level events (zero + flag when events do not reconcile with the score)
e_ok = np.stack([evok, evok], 1)
add('ev_ok', e_ok)
for k, nm in enumerate(EV_NAMES):
    sc_ = {'yel': 3.0, 'g_h1': 2.0, 'g_late': 2.0, 'g_close': 2.0, 'g_11v11': 2.0, 'g_first30': 2.0}.get(nm, 1.0)
    add('ev_' + nm + '_f', ev[:, :, k] / sc_ * (1 if nm in ('yel',) else e_ok))
    if nm not in ('first_goal', 'first_goal_min', 'last_goal_min'): add('ev_' + nm + '_a', sw(ev[:, :, k]) / sc_ * (1 if nm in ('yel',) else e_ok))
add('ev_sur_yc', (ev[:, :, 6] - exp['yc']) / 2)
# --- group "st": box-score stats (only when statsOk), raw and relative to v1's expectation
s_ok = np.stack([stok, stok], 1)
add('st_ok', s_ok)
for k, (nm, mu_, sd_) in enumerate([('sh', 12.5, 5), ('sot', 4.3, 2.5), ('co', 5.2, 3), ('fo', 15, 5)]):
    okk = s_ok if nm in ('sh', 'sot') else np.stack([co_ok, co_ok], 1) if nm == 'co' else np.stack([fo_ok, fo_ok], 1)
    add(f'st_{nm}_f', (st[:, :, k] - mu_) / sd_ * okk); add(f'st_{nm}_a', (sw(st[:, :, k]) - mu_) / sd_ * okk)
    add(f'st_sur_{nm}_f', np.log((st[:, :, k] + 1) / (exp[nm] + 1)) * okk); add(f'st_sur_{nm}_a', np.log((sw(st[:, :, k]) + 1) / (sw(exp[nm]) + 1)) * okk)
add('st_po', (st[:, :, 4] - 50) / 10 * np.stack([po_ok, po_ok], 1))
add('st_po_ok', np.stack([po_ok, po_ok], 1))
TOK = np.stack(cols, 2).reshape(2 * n, len(names))          # (2n, F)
TOK[~np.repeat(post, 2)] = 0
assert np.isfinite(TOK).all()

# ---- match-level context (all pre-match) ----------------------------------------------------------
ctx_names = ['log_lh', 'log_la', 'log_glh', 'log_gla', 'g_Sh', 'g_Sa', 'rel_sh_h', 'rel_sh_a', 'rel_sot_h', 'rel_sot_a', 'rel_co_h', 'rel_co_a',
             'rel_yc_h', 'rel_yc_a', 'rel_fo_h', 'rel_fo_a', 'nh', 'na', 'lgB', 'mu', 'hfa', 'v1_pH', 'v1_pD', 'v1_pA']
def rel(c):
    x = np.log(np.stack([F[c + '_lh'].to_numpy(), F[c + '_la'].to_numpy()], 1))
    return x - np.log(F[[c + '_lh', c + '_la']].to_numpy()).mean()
C = [np.log(lh), np.log(la), np.log(F.g_lh), np.log(F.g_la), (F.g_Sh - 0.04) * 50, (F.g_Sa - 0.04) * 50]
for c in ('sh', 'sot', 'co', 'yc', 'fo'):
    r = np.clip(rel(c), -1.5, 1.5); C += [r[:, 0] * 3, r[:, 1] * 3]
C += [F.nh / 38, F.na / 38, lgB, (F.mu - 0.235) * 10, (F.hfa - 0.385) * 10, pv1[:, 0], pv1[:, 1], pv1[:, 2]]
CTX = np.stack([np.asarray(c, np.float32) for c in C], 1)
assert np.isfinite(CTX).all()
y = np.where(hg > ag, 0, np.where(hg == ag, 1, 2)); y[~post] = -1

np.savez_compressed(OUT, ids=np.array([m['id'] for m in M]), status=np.array([m['status'] for m in M]), post=post, lgB=lgB, season=season, day=day,
                    h=h, a=a, hg=hg, ag=ag, y=y, lh=lh, la=la, pv1=pv1, TOK=TOK, tok_names=np.array(names), tok_day=tok_day, tok_season=tok_season,
                    hist=hist, CTX=CTX, ctx_names=np.array(ctx_names), st=st, stok=stok, teams=np.array(teams))
print('matches', n, 'post', post.sum(), 'tokens', TOK.shape, 'ctx', CTX.shape)
print('token features:', names)
print('hist coverage (mean #tokens):', (hist >= 0).sum(2).mean())
