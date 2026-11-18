#!/usr/bin/env python
"""Walk-forward lineup / player features for every match (Série A and B).

Everything used for match m comes from matches strictly BEFORE m, plus (for the k_* "lineup-known" columns only)
the identity of the 11 announced starters of m (and which of them is the goalkeeper).
No minutes, substitutions, stats or events of match m itself are touched before the features are written.

Outputs
  work/side_features.csv   one row per (match, side) with all k_* (lineup-known) and u_* (lineup-unknown) columns
  features_known.csv / features_unknown.csv   one row per match: home, away and home-minus-away columns
"""
import json, math, os, sys, time
from collections import defaultdict, Counter
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator

ROOT = os.path.dirname(os.path.abspath(__file__))
SHARED = '/home/user/bdv-lab/shared/'

# ---- fixed a-priori settings (never tuned on results) ---------------------------------------------------------------
K_SHORT, DECAY_SHORT = 10, 0.85      # window for "usual XI" (minutes / start shares)
K_LONG, DECAY_LONG = 20, 0.95        # window for goal / shot shares
K_XL, DECAY_XL = 40, 0.97            # very long window (roughly the memory of the v1 team rating)
MAXDAYS = 400                        # ignore team matches older than this when building windows
LG_OFF = {'A': 0.0, 'B': -0.6}       # prior strength gap between divisions (log-goal scale, o+d)
PED_HL = 30.0                        # pedigree half-life, in full matches played
PED_K = 3.0                          # pedigree shrinkage (matches) towards current team strength
RAPM_HL = 1095.0                     # RAPM time-decay half-life in days
RAPM_LAMS = (150.0, 600.0)           # ridge penalties
XG_SOT, XG_OFF = 0.25, 0.02          # fixed xG proxy: goals ~ 0.25*SoT + 0.02*off-target shots


def day_of(iso):
    return int(pd.Timestamp(iso).value // 86_400_000_000_000)


class RAPM:
    """Ridge-regularised adjusted plus-minus on match residuals (vs v1 expectation), refit before each match day."""

    def __init__(self, lams, hl):
        self.lams, self.hl = lams, hl
        self.idx = {}
        self.r, self.c, self.v, self.y, self.d = [], [], [], [], []
        self.nrow = 0
        self.theta = {lam: np.zeros(0) for lam in lams}
        self.fitted_rows = 0

    def add(self, day, hp, ap, ys):
        for sign, pl in ((1.0, hp), (-1.0, ap)):
            for pid, m in pl.items():
                if m <= 0: continue
                j = self.idx.setdefault(pid, len(self.idx))
                self.r.append(self.nrow); self.c.append(j); self.v.append(sign * m)
        self.y.append(ys); self.d.append(day); self.nrow += 1

    def fit(self, day):
        if self.nrow == self.fitted_rows or self.nrow < 200:
            return
        n = len(self.idx)
        X = sp.csr_matrix((self.v, (self.r, self.c)), shape=(self.nrow, n))
        w = 0.5 ** ((day - np.array(self.d)) / self.hl)
        Y = np.array(self.y)            # (rows, ntargets)
        XtW = X.T.multiply(w).tocsr()
        for lam in self.lams:
            old = self.theta[lam]
            th = np.zeros((n, Y.shape[1]))
            th[:old.shape[0]] = old if old.ndim == 2 else 0
            A = LinearOperator((n, n), matvec=lambda z, lam=lam: XtW @ (X @ z) + lam * z)
            for k in range(Y.shape[1]):
                b = XtW @ Y[:, k]
                sol, _ = cg(A, b, x0=th[:, k], rtol=1e-5, maxiter=200)
                th[:, k] = sol
            self.theta[lam] = th
        self.fitted_rows = self.nrow

    def get(self, lam, pid, k):
        j = self.idx.get(pid)
        th = self.theta[lam]
        if j is None or j >= th.shape[0]: return 0.0
        return th[j, k]


LAG_FEATS = ('k_usage', 'k_bench_usage', 'k_squad_usage', 'k_gashare', 'k_bench_gashare', 'k_exp', 'k_sh90', 'k_bench_reg', 'k_sot90', 'k_fs90', 'k_xg90')
GA_PRIOR, GA_K = 0.20, 5.0           # shrink goals+assists per 90 towards 0.20 with 5 matches of prior weight
SH_PRIOR, SH_K = 1.10, 5.0


def ga90(pl_ga, p):
    nd = pl_ga.get(p)
    return (nd[0] + GA_K * GA_PRIOR) / (nd[1] + GA_K) if nd else GA_PRIOR


def sh90(pl_sh, p):
    nd = pl_sh.get(p)
    return (nd[0] + SH_K * SH_PRIOR) / (nd[1] + SH_K) if nd else SH_PRIOR


def rate(store, p, prior, k=5.0):
    nd = store.get(p)
    return (nd[0] + k * prior) / (nd[1] + k) if nd else prior


def window(hist, day, K, decay):
    """Decay-weighted shares over the team's last K matches. Returns None if no usable history."""
    u, s, ga, g, sh = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    W = 0.0; Wsh = 0.0
    j = 0
    for r in reversed(hist[-K:]):
        if day - r['day'] > MAXDAYS: break
        w = decay ** j; j += 1; W += w
        for pid, m in r['mins'].items(): u[pid] += w * m
        for pid in r['st']: s[pid] += w
        for pid, x in r['ga'].items(): ga[pid] += w * x
        for pid, x in r['g'].items(): g[pid] += w * x
        if r['sh'] is not None:
            Wsh += w
            for pid, x in r['sh'].items(): sh[pid] += w * x
    if W == 0: return None
    for d in (u, s):
        for k in d: d[k] /= W
    return dict(u=u, s=s, ga=ga, g=g, sh=sh, W=W, n=j, Wsh=Wsh)


def main():
    t0 = time.time()
    M = json.load(open(SHARED + 'matches.json'))
    BF = pd.read_csv(SHARED + 'baseline_features.csv', dtype={'id': str}).set_index('id')
    TS = pd.read_csv(f'{ROOT}/work/team_state.csv', dtype={'id': str}).set_index('id')
    PM = pd.read_csv(f'{ROOT}/data/player_match.csv.gz', dtype={'mid': str, 'pid': str, 'tid': str, 'pos': str, 'posg': str})
    PM['posg'] = PM['posg'].fillna('')
    groups = {}
    for (mid, side), g in PM.groupby(['mid', 'side'], sort=False):
        groups[(mid, side)] = g
    MX = pd.read_csv(f'{ROOT}/data/match_xg.csv', dtype={'mid': str}).fillna(0).set_index('mid')
    mxg = {mid: (r.xg_h, r.xg_a) for mid, r in MX.iterrows()}
    PX = pd.read_csv(f'{ROOT}/data/player_xg.csv.gz', dtype={'mid': str, 'pid': str})
    pxg = defaultdict(dict)
    for mid_, side_, pid_, xg_, xa_ in zip(PX.mid, PX.side, PX.pid, PX.xg, PX.xa):
        pxg[(mid_, side_)][pid_] = xg_ + xa_
    print('loaded', len(PM), 'player rows', round(time.time() - t0, 1), 's', flush=True)

    team_hist = defaultdict(list)            # tid -> list of records
    pl_apps = defaultdict(float)             # pid -> career full-match equivalents in DB
    pl_appsA = defaultdict(float)            # same, Série A only
    pl_club = defaultdict(float)             # (pid, tid) -> full-match equivalents for that club
    pl_pos = defaultdict(Counter)            # pid -> position counter (when starting)
    pl_ped = {}                              # pid -> [num, den]
    pl_last_tid = {}
    yc_count = defaultdict(int)              # (pid, season, lg) -> yellows
    susp = defaultdict(set)                  # tid -> players suspended for the team's next league match
    team_kusage = defaultdict(list)          # tid -> list of (day, k_usage) of past matches (for lagged features)
    team_kusage_s = defaultdict(list)        # same for the in-season usage index
    team_lag = defaultdict(list)             # tid -> list of (day, {known features of that past match})
    pl_ga = {}                               # pid -> [decayed goals+assists, decayed full matches]
    pl_sh = {}                               # pid -> [decayed shots, decayed full matches] (seasons with stats)
    pl_sot, pl_fs, pl_g = {}, {}, {}         # same for shots on target, fouls suffered, goals
    pl_xg = {}                               # pid -> [decayed xG + xA, decayed full matches] (matches with play-by-play)
    pl_listed = defaultdict(int)             # pid -> times named in a matchday squad
    pl_starts = defaultdict(int)             # pid -> times in a starting XI
    pl_first = {}                            # pid -> first day seen
    season_n = defaultdict(int)              # (tid, season) -> matches played this season
    prev_season_min = defaultdict(lambda: defaultdict(float))   # (tid, season) -> pid -> minutes
    seen_this_season = defaultdict(set)      # (tid, season) -> players listed (XI + bench + subs) so far
    rapm = RAPM(RAPM_LAMS, RAPM_HL)
    last_fit_day = None
    out = []

    for m in M:
        mid = m['id']
        if mid not in BF.index: continue
        day = day_of(m['date'])
        if last_fit_day != day:
            rapm.fit(day); last_fit_day = day
        bf, ts = BF.loc[mid], TS.loc[mid]
        strength = {'h': ts.oh + ts.dh + LG_OFF[m['lg']], 'a': ts.oa + ts.da + LG_OFF[m['lg']]}
        rosters = {s: groups.get((mid, s)) for s in 'ha'} if m['status'] == 'post' else {'h': None, 'a': None}
        for s in 'ha':   # a roster is usable only if it has a (nearly) complete starting XI
            if rosters[s] is not None and int((rosters[s].starter == 1).sum()) < 10:
                rosters[s] = None
        feats = {}
        for side in 'ha':
            tid = m[side]
            hist = team_hist[tid]
            f = dict(id=mid, side=side, tid=tid, lg=m['lg'], season=m['season'], date=m['date'], status=m['status'])
            ws = window(hist, day, K_SHORT, DECAY_SHORT)
            wss = window([r for r in hist[-K_SHORT:] if r['season'] == m['season']], day, K_SHORT, DECAY_SHORT)
            wl = window(hist, day, K_LONG, DECAY_LONG)
            wx = window(hist, day, K_XL, DECAY_XL)
            sn = season_n[(tid, m['season'])]
            f['n_hist'] = ws['n'] if ws else 0
            f['n_season'] = sn
            sus = susp.get(tid, set())
            # ---------------- lineup-unknown features (history only) ----------------
            ku = [v for d0, v in team_kusage[tid][-10:] if day - d0 <= MAXDAYS]
            f['u_prev_usage'] = ku[-1] if ku else np.nan
            f['u_usage3'] = float(np.mean(ku[-3:])) if ku else np.nan
            f['u_usage10'] = float(np.mean(ku[-10:])) if ku else np.nan
            lg_ = [x for d0, x in team_lag[tid][-5:] if day - d0 <= MAXDAYS]
            for nm in LAG_FEATS:
                vals = [x[nm] for x in lg_ if x.get(nm) is not None and x[nm] == x[nm]]
                f['u_lag1_' + nm[2:]] = vals[-1] if vals else np.nan
                f['u_lag3_' + nm[2:]] = float(np.mean(vals[-3:])) if vals else np.nan
            kus = [v for d0, v, sea in team_kusage_s[tid][-10:] if sea == m['season']]
            f['u_prev_usage_s'] = kus[-1] if kus else np.nan
            f['u_usage3_s'] = float(np.mean(kus[-3:])) if kus else np.nan
            if ws:
                u, s = ws['u'], ws['s']
                top = sorted(u.values(), reverse=True)[:11]
                U11 = sum(top) or 1.0
                S11 = sum(sorted(s.values(), reverse=True)[:11]) or 1.0
                f['u_susp'] = sum(u.get(p, 0.0) for p in sus) / U11
                f['u_susp_n'] = sum(1 for p in sus if s.get(p, 0.0) >= 0.5)
                # players who went off injured-looking (before 35') or were absent from last squad although regular
                last = hist[-1]
                f['u_early_off'] = sum(u.get(p, 0.0) for p in last['early_off']) / U11
                # regulars missing from the whole last matchday squad (injured / gone), not counting the newly suspended
                f['u_absent'] = sum(v for p, v in u.items() if p not in last['listed'] and p not in sus) / U11
                f['u_ga90'] = 11.0 * sum(v * ga90(pl_ga, p) for p, v in s.items() if p not in sus) / (sum(v for p, v in s.items() if p not in sus) or 1.0)
                f['u_sh90'] = 11.0 * sum(v * sh90(pl_sh, p) for p, v in s.items() if p not in sus) / (sum(v for p, v in s.items() if p not in sus) or 1.0)
                f['u_ga90_susp'] = sum(s.get(p, 0.0) * ga90(pl_ga, p) for p in sus)
                # expected XI = start-share weights, suspended players removed
                p_start = {p: v for p, v in s.items() if p not in sus and v > 0}
                tot = sum(p_start.values()) or 1.0
                f['u_e_usage'] = 11.0 * sum(v * u.get(p, 0.0) for p, v in p_start.items()) / tot / U11
                if wl:
                    tga_ = sum(wl['ga'].values())
                    if tga_ > 0:
                        f['u_e_gashare'] = 11.0 * sum(v * wl['ga'].get(p, 0.0) for p, v in p_start.items()) / tot / tga_
                        f['u_susp_ga'] = sum(wl['ga'].get(p, 0.0) for p in sus) / tga_
                        f['u_absent_ga'] = sum(v for p, v in wl['ga'].items() if p not in last['listed'] and p not in sus) / tga_
                # last matchday squad (XI + bench) minus the newly suspended: who is plausibly available today
                f['u_avail'] = sum(u.get(p, 0.0) for p in last['listed'] if p not in sus) / U11
                for li, lam in enumerate(RAPM_LAMS):
                    for k, nm in enumerate(('g', 'x')):
                        f[f'u_rapm{li}{nm}'] = 11.0 * sum(v * rapm.get(lam, p, k) for p, v in p_start.items()) / tot
                peds = []
                for p, v in p_start.items():
                    nd = pl_ped.get(p)
                    pe = ((nd[0] + PED_K * strength[side]) / (nd[1] + PED_K)) if nd else strength[side]
                    peds.append((v, pe))
                f['u_ped'] = sum(v * pe for v, pe in peds) / tot - strength[side]
                f['u_exp'] = sum(v * math.log1p(pl_apps.get(p, 0.0)) for p, v in p_start.items()) / tot
                f['u_sot90'] = 11.0 * sum(v * rate(pl_sot, p, 0.38) for p, v in p_start.items()) / tot
                f['u_xg90'] = 11.0 * sum(v * rate(pl_xg, p, 0.20) for p, v in p_start.items()) / tot
                # team-level form relative to the v1 expectation (goals and shot-location xG), short and long windows
                for tag, K_, dec in (('s', K_SHORT, DECAY_SHORT), ('l', K_LONG, DECAY_LONG)):
                    W_, Wx, gres, xres, luck = 0.0, 0.0, 0.0, 0.0, 0.0
                    j_ = 0
                    for r in reversed(hist[-K_:]):
                        if day - r['day'] > MAXDAYS: break
                        w_ = dec ** j_; j_ += 1
                        W_ += w_; gres += w_ * ((r['tgf'] - r['tga']) - (r['tlf'] - r['tla']))
                        if r['xgf'] is not None:
                            Wx += w_; xres += w_ * ((r['xgf'] - r['xga']) - (r['tlf'] - r['tla'])); luck += w_ * ((r['tgf'] - r['tga']) - (r['xgf'] - r['xga']))
                    if W_ > 0: f['u_t_gres_' + tag] = gres / W_
                    if Wx > 0.5 * W_ and Wx > 0:
                        f['u_t_xgres_' + tag] = xres / Wx; f['u_t_luck_' + tag] = luck / Wx
                f['u_fs90'] = 11.0 * sum(v * rate(pl_fs, p, 1.30) for p, v in p_start.items()) / tot
                f['u_expA'] = sum(v * math.log1p(pl_appsA.get(p, 0.0)) for p, v in p_start.items()) / tot
                # squad retention from previous season (players seen so far this season)
                pm = prev_season_min.get((tid, m['season'] - 1))
                if pm and sn >= 1:
                    totm = sum(pm.values()) or 1.0
                    seen = seen_this_season[(tid, m['season'])]
                    f['u_ret'] = sum(v for p, v in pm.items() if p in seen) / totm
                else:
                    f['u_ret'] = np.nan
            # ---------------- lineup-known features (history + today's announced XI) ----------------
            g = rosters[side]
            f['k_ok'] = 0
            if g is not None:
                st = g[g.starter == 1]
                S = list(st.pid)
                f['k_nst'] = len(S)
                if len(S) >= 10:
                    f['k_ok'] = 1
                    n11 = float(len(S))
                    gk_today = [p for p, pg in zip(st.pid, st.posg) if pg == 'G']
                    f['k_debut'] = sum(1 for p in S if pl_apps.get(p, 0.0) == 0.0 and p not in pl_last_tid)
                    f['k_exp'] = float(np.mean([math.log1p(pl_apps.get(p, 0.0)) for p in S]))
                    f['k_expA'] = float(np.mean([math.log1p(pl_appsA.get(p, 0.0)) for p in S]))
                    f['k_exp_club'] = float(np.mean([math.log1p(pl_club.get((p, tid), 0.0)) for p in S]))
                    f['k_exp_min'] = float(np.min([math.log1p(pl_apps.get(p, 0.0)) for p in S]))
                    for li, lam in enumerate(RAPM_LAMS):
                        for k, nm in enumerate(('g', 'x')):
                            f[f'k_rapm{li}{nm}'] = 11.0 / n11 * sum(rapm.get(lam, p, k) for p in S)
                    pe = []
                    for p in S:
                        nd = pl_ped.get(p)
                        pe.append(((nd[0] + PED_K * strength[side]) / (nd[1] + PED_K)) if nd else strength[side])
                    f['k_ped'] = float(np.mean(pe)) - strength[side]
                    f['k_newclub'] = sum(1 for p in S if p in pl_last_tid and pl_last_tid[p] != tid)
                    f['k_ga90'] = 11.0 / n11 * sum(ga90(pl_ga, p) for p in S)
                    f['k_sh90'] = 11.0 / n11 * sum(sh90(pl_sh, p) for p in S)
                    f['k_sot90'] = 11.0 / n11 * sum(rate(pl_sot, p, 0.38) for p in S)
                    f['k_fs90'] = 11.0 / n11 * sum(rate(pl_fs, p, 1.30) for p in S)
                    f['k_g90'] = 11.0 / n11 * sum(rate(pl_g, p, 0.11) for p in S)
                    f['k_xg90'] = 11.0 / n11 * sum(rate(pl_xg, p, 0.20) for p in S)
                    f['k_startrate'] = float(np.mean([(pl_starts.get(p, 0) + 2.5) / (pl_listed.get(p, 0) + 5.0) for p in S]))
                    f['k_years'] = float(np.mean([(day - pl_first.get(p, day)) / 365.25 for p in S]))
                    if wss:
                        us = wss['u']
                        f['k_usage_s'] = sum(us.get(p, 0.0) for p in S) / (sum(sorted(us.values(), reverse=True)[:11]) or 1.0)
                    if ws:
                        f['k_usage'] = sum(u.get(p, 0.0) for p in S) / U11
                        f['k_start'] = sum(s.get(p, 0.0) for p in S) / S11
                        f['k_new'] = sum(1 for p in S if u.get(p, 0.0) == 0.0)
                        f['k_chg'] = len(set(S) - hist[-1]['st'])
                        if m['season'] >= 2010:   # full bench lists exist from 2010 (before that only used subs are listed)
                            bench = [p for p, s1 in zip(g.pid, g.starter) if s1 != 1]
                            f['k_bench_usage'] = sum(u.get(p, 0.0) for p in bench) / U11
                            f['k_squad_usage'] = f['k_bench_usage'] + sum(u.get(p, 0.0) for p in S) / U11
                            f['k_bench_n'] = len(bench)
                            f['k_bench_start'] = sum(s.get(p, 0.0) for p in bench) / S11
                            f['k_bench_exp'] = float(np.mean([math.log1p(pl_apps.get(p, 0.0)) for p in bench])) if bench else np.nan
                            f['k_bench_ga90'] = sum(ga90(pl_ga, p) for p in bench)
                            f['k_bench_sh90'] = sum(sh90(pl_sh, p) for p in bench)
                            f['k_bench_reg'] = sum(1 for p in bench if s.get(p, 0.0) >= 0.5)      # regular starters left on the bench
                            # DIAGNOSTIC ONLY (uses who actually came on - post-match information, never fed to a model)
                            f['diag_bench_used'] = sum(u.get(p, 0.0) for p, s1, us in zip(g.pid, g.starter, g.used) if s1 != 1 and us == 1) / U11
                            f['diag_n_used'] = int(((g.starter != 1) & (g.used == 1)).sum())
                        usual11 = sorted(u, key=u.get, reverse=True)[:11]
                        for grp in 'GDMF':
                            f['k_miss_' + grp] = sum(u[p] for p in usual11 if p not in S and (pl_pos[p].most_common(1) or [('?', 0)])[0][0] == grp)
                        f['k_susp_start'] = sum(1 for p in S if p in sus)     # diagnostic for the suspension rule
                        # usual goalkeeper starts?
                        gks = {p: v for p, v in u.items() if pl_pos[p].most_common(1) and pl_pos[p].most_common(1)[0][0] == 'G'}
                        if gks and gk_today:
                            f['k_gk'] = float(max(gks, key=gks.get) in gk_today)
                        tga = sum(wl['ga'].values()); tg = sum(wl['g'].values()); tsh = sum(wl['sh'].values())
                        lu = wl['u']
                        Ul = sum(sorted(lu.values(), reverse=True)[:11]) or 1.0
                        f['k_usage_long'] = sum(lu.get(p, 0.0) for p in S) / Ul
                        xu = wx['u']
                        f['k_usage_xl'] = sum(xu.get(p, 0.0) for p in S) / (sum(sorted(xu.values(), reverse=True)[:11]) or 1.0)
                        if tga > 0: f['k_gashare'] = sum(wl['ga'].get(p, 0.0) for p in S) / tga
                        if m['season'] >= 2010:
                            if tga > 0: f['k_bench_gashare'] = sum(wl['ga'].get(p, 0.0) for p in bench) / tga
                            if tsh > 0 and wl['Wsh'] > 0: f['k_bench_shshare'] = sum(wl['sh'].get(p, 0.0) for p in bench) / tsh
                        if tg > 0:
                            f['k_gshare'] = sum(wl['g'].get(p, 0.0) for p in S) / tg
                            top_p = max(wl['g'], key=wl['g'].get)
                            if wl['g'][top_p] >= 2.0: f['k_top'] = float(top_p in S)
                        if tsh > 0 and wl['Wsh'] > 0: f['k_shshare'] = sum(wl['sh'].get(p, 0.0) for p in S) / tsh
                        # goal/shot share relative to what the usual XI carries (so that it measures WHO is missing)
                        if tga > 0:
                            usual = sorted(u, key=u.get, reverse=True)[:11]
                            f['k_gashare_usual'] = sum(wl['ga'].get(p, 0.0) for p in usual) / tga
                    pm = prev_season_min.get((tid, m['season'] - 1))
                    if pm:
                        totm = sum(pm.values()) or 1.0
                        f['k_ret_xi'] = sum(pm.get(p, 0.0) for p in S) / totm
                        listed = set(g.pid) | seen_this_season[(tid, m['season'])]
                        f['k_ret_squad'] = sum(v for p, v in pm.items() if p in listed) / totm
            feats[side] = f
            out.append(f)

        # ------------------------------------------------ state updates (after the match) -----------------------------
        if m['status'] != 'post':
            continue
        both = rosters['h'] is not None and rosters['a'] is not None
        mins90 = {}
        for side in 'ha':
            tid = m[side]
            g = rosters[side]
            susp[tid] = set()   # whatever suspension applied has now been served (or was wrong)
            season_n[(tid, m['season'])] += 1
            if g is None:
                continue
            mins = {p: mm / 90.0 for p, mm in zip(g.pid, g['min']) if mm > 0}
            mins90[side] = mins
            st = set(g.pid[g.starter == 1])
            has_sh = g.sh.notna().any()
            xg_m = mxg.get(mid)
            hside = side == 'h'
            rec = dict(mid=mid, day=day, season=m['season'], lg=m['lg'], mins=mins, st=st, listed=set(g.pid),
                       tgf=m['hg'] if hside else m['ag'], tga=m['ag'] if hside else m['hg'],
                       tlf=bf.lh if hside else bf.la, tla=bf.la if hside else bf.lh,
                       xgf=None if xg_m is None else (xg_m[0] if hside else xg_m[1]),
                       xga=None if xg_m is None else (xg_m[1] if hside else xg_m[0]),
                       ga={p: x + y for p, x, y in zip(g.pid, g.g, g.a) if x + y > 0},
                       g={p: x for p, x in zip(g.pid, g.g) if x > 0},
                       sh={p: x for p, x in zip(g.pid, g.sh) if x > 0} if has_sh else None,
                       early_off={p for p, s1, to in zip(g.pid, g.starter, g.t_out) if s1 == 1 and to is not None and to == to and to < 35 and p not in set(g.pid[g.rc == 1])})
            team_hist[tid].append(rec)
            ku = feats[side].get('k_usage')
            if ku is not None and ku == ku:
                team_kusage[tid].append((day, ku))
            team_lag[tid].append((day, {nm: feats[side].get(nm) for nm in LAG_FEATS}))
            kus = feats[side].get('k_usage_s')
            if kus is not None and kus == kus:
                team_kusage_s[tid].append((day, kus, m['season']))
            pxg_side = pxg.get((mid, side), {})
            for p, s1, mm, pg, yc, rc, gg, aa, shh, sott, fss in zip(g.pid, g.starter, g['min'], g.posg, g.yc, g.rc, g.g, g.a, g.sh, g.sot, g.fs):
                seen_this_season[(tid, m['season'])].add(p)
                pl_listed[p] += 1; pl_starts[p] += int(s1 == 1); pl_first.setdefault(p, day)
                if mm > 0:
                    w = mm / 90.0
                    gam = 0.5 ** (w / PED_HL)
                    nd = pl_g.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + gg; nd[1] = nd[1] * gam + w
                    if xg_m is not None:
                        nd = pl_xg.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + pxg_side.get(p, 0.0); nd[1] = nd[1] * gam + w
                    if sott == sott:
                        nd = pl_sot.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + sott; nd[1] = nd[1] * gam + w
                    if fss == fss:
                        nd = pl_fs.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + fss; nd[1] = nd[1] * gam + w
                    nd = pl_ga.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + gg + aa; nd[1] = nd[1] * gam + w
                    if shh == shh:
                        nd = pl_sh.setdefault(p, [0.0, 0.0]); nd[0] = nd[0] * gam + shh; nd[1] = nd[1] * gam + w
                    pl_apps[p] += w
                    if m['lg'] == 'A': pl_appsA[p] += w
                    pl_club[(p, tid)] += w
                    prev_season_min[(tid, m['season'])][p] += mm
                    nd = pl_ped.setdefault(p, [0.0, 0.0])
                    gam = 0.5 ** (w / PED_HL)
                    nd[0] = nd[0] * gam + w * strength[side]; nd[1] = nd[1] * gam + w
                    pl_last_tid[p] = tid
                    if s1 == 1 and pg: pl_pos[p][pg] += 1
                if rc == 1:
                    susp[tid].add(p)
                elif yc and yc == yc and yc >= 1:
                    key = (p, m['season'], m['lg'])
                    yc_count[key] += 1
                    if yc_count[key] % 3 == 0:
                        susp[tid].add(p)
        if both:
            res_g = (m['hg'] - m['ag']) - (bf.lh - bf.la)
            if m.get('statsOk') and m.get('stats'):
                s = m['stats']
                xh = XG_SOT * s['h']['sot'] + XG_OFF * max(0, s['h']['sh'] - s['h']['sot'])
                xa = XG_SOT * s['a']['sot'] + XG_OFF * max(0, s['a']['sh'] - s['a']['sot'])
                res_x = 0.5 * (m['hg'] - m['ag']) + 0.5 * (xh - xa) - (bf.lh - bf.la)
            else:
                res_x = res_g
            rapm.add(day, mins90['h'], mins90['a'], (res_g, res_x))

    F = pd.DataFrame(out)
    F.to_csv(f'{ROOT}/work/side_features.csv', index=False)
    print('side rows', len(F), 'time', round(time.time() - t0, 1), 's')
    kcols = [c for c in F.columns if c.startswith('k_')]   # diag_* columns are deliberately left out of the deliverables
    ucols = [c for c in F.columns if c.startswith('u_')] + ['n_hist', 'n_season']
    for name, cols in (('known', kcols + ucols), ('unknown', ucols)):
        H = F[F.side == 'h'].set_index('id'); A = F[F.side == 'a'].set_index('id')
        parts = [H[['lg', 'season', 'date', 'status']]]
        for c in cols:
            parts += [H[c].rename(c + '_h'), A[c].rename(c + '_a')]
        pd.concat(parts, axis=1).reset_index().to_csv(f'{ROOT}/features_{name}.csv', index=False)
    cov = F.groupby(['lg', 'season']).agg(n=('id', 'size'), k_ok=('k_ok', 'mean'), usage=('k_usage', lambda x: x.notna().mean()),
                                          usage_mean=('k_usage', 'mean'), u_prev=('u_prev_usage', lambda x: x.notna().mean()))
    print(cov.round(3).to_string())


if __name__ == '__main__':
    main()
