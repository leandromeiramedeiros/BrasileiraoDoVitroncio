#!/usr/bin/env python
"""Step 2 - shot-level xG model, fitted walk-forward.

For the shots of season Y the model is trained on shots of seasons < Y (Serie A and B pooled).
2014 (first season with shots) is valued in monthly blocks: block b uses a model fitted on the 2014 shots of the
earlier blocks; the first block (matches before 2014-06-15, i.e. before the World Cup break) uses a fixed a-priori
zone table (literature conversion rates, not fitted on this data).

Outputs work/shots_xg.pkl = shots + columns
  xg      main pre-shot xG (logistic, no "situation" text, with game state)
  xg_ns   logistic without game state (score / minute / men)
  xg_gb   LightGBM variant
  xg_sit  logistic + "following a corner / set piece / fast break" (inverse-probability weighted, see REPORT)
  psxg    post-shot xG for shots on target (placement), 0 for the others  -> goalkeeping
and work/xg_eval.json with out-of-sample quality numbers.
"""
import json, os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
warnings.filterwarnings('ignore')

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, 'work')
ZONES = ['box_c', 'close', 'six_side', 'box_side', 'angle', 'angle_long', 'long', 'far35']  # baseline: out
PRIOR = {  # a-priori conversion table (foot, head); used only for the first weeks of 2014
    'close': (0.42, 0.35), 'six_side': (0.20, 0.14), 'box_c': (0.14, 0.09), 'box_side': (0.06, 0.04), 'angle': (0.05, 0.04),
    'angle_long': (0.03, 0.02), 'out': (0.03, 0.02), 'long': (0.02, 0.01), 'far35': (0.012, 0.01)}
PEN_PRIOR, PEN_N0 = 0.76, 100.0
HALF_LIFE = 3.0


def base_frame(S):
    d = np.sqrt(S.dx ** 2 + S.dy ** 2).clip(0.5, 60)
    th = np.arctan2(7.32 * S.dx, S.dx ** 2 + S.dy ** 2 - 3.66 ** 2)
    th = np.where(th < 0, th + np.pi, th)
    F = pd.DataFrame({'d': d, 'ld': np.log(d), 'th': th, 'ady': S.dy.abs().clip(0, 34)}, index=S.index)
    head = (S.body == 'head').astype(float)
    F['head'] = head; F['other'] = (S.body == 'other').astype(float)
    F['head_ld'] = head * (F.ld - np.log(10)); F['head_th'] = head * (th - 0.5)
    for z in ZONES: F['z_' + z] = (S.zone == z).astype(float)
    F['head_close'] = head * (S.zone == 'close')
    for a in ['cross', 'headpass', 'pass', 'through']: F['a_' + a] = (S.ast == a).astype(float)
    F['cross_foot'] = ((S.ast == 'cross') & (S.body != 'head')).astype(float)
    quick = (S.stoppage == 0) & (S.dt_prev >= 0) & (S.dt_prev <= 5)
    F['reb_sv'] = (quick & S.prev_out.isin(['saved', 'wood'])).astype(float)
    F['reb_bl'] = (quick & (S.prev_out == 'blocked')).astype(float)
    F['reb_off'] = (quick & (S.prev_out == 'off')).astype(float)
    F['stop_seq'] = ((S.stoppage == 1) & (S.dt_prev == 0) & S.prev_out.isin(['saved', 'wood'])).astype(float)
    F['lgB'] = (S.lg == 'B').astype(float)
    return F


def state_frame(S):
    F = pd.DataFrame(index=S.index)
    F['lead'] = (S.dsc > 0).astype(float); F['lead2'] = (S.dsc >= 2).astype(float); F['trail'] = (S.dsc < 0).astype(float)
    F['min90'] = (S.minute / 90).clip(0, 1); F['late'] = (S.minute >= 75).astype(float)
    adv = (S.red_opp - S.red_for).clip(-1, 1)
    F['man_up'] = (adv > 0).astype(float); F['man_dn'] = (adv < 0).astype(float)
    return F


def sit_frame(S):
    F = pd.DataFrame(index=S.index)
    for s in ['corner', 'setp', 'fast']: F['s_' + s] = (S.sit == s).astype(float)
    return F


class Logit:
    """Logistic regression with natural-spline bases on distance and angle."""
    def __init__(self, C=0.3):
        self.C = C
    def _X(self, F, fit=False):
        cont = F[['ld', 'th', 'ady']].to_numpy()
        if fit:
            self.sp = SplineTransformer(n_knots=5, degree=3, extrapolation='linear', include_bias=False).fit(cont)
        B = self.sp.transform(cont)
        rest = F.drop(columns=['d', 'ld', 'th', 'ady']).to_numpy()
        X = np.hstack([B, rest])
        if fit: self.sc = StandardScaler().fit(X)
        return self.sc.transform(X)
    def fit(self, F, y, w=None):
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(self._X(F, True), y, sample_weight=w)
        return self
    def predict(self, F):
        return self.m.predict_proba(self._X(F))[:, 1]


class GBM:
    def __init__(self):
        import lightgbm as lgb
        self.lgb = lgb
    def fit(self, F, y, w=None):
        self.cols = list(F.columns)
        n = len(F)
        self.m = self.lgb.LGBMClassifier(n_estimators=140 if n > 30000 else 90, learning_rate=0.07, num_leaves=8, min_child_samples=max(40, n // 400),
                                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1, n_jobs=2, random_state=1)
        self.m.fit(F[self.cols], y, sample_weight=w)
        return self
    def predict(self, F):
        return self.m.predict_proba(F[self.cols])[:, 1]


def prior_xg(S):
    return np.array([PRIOR[z][1 if b == 'head' else 0] for z, b in zip(S.zone, S.body)])


def blocks(S):
    """Valuation blocks: (label, mask of shots to value, mask of shots allowed for training)."""
    out = []
    d14 = S.date[S.season == 2014]
    edges = ['2014-06-15', '2014-08-01', '2014-09-01', '2014-10-01', '2014-11-01', '2015-01-01']
    lo = '2014-01-01'
    for e in edges:
        out.append((f'2014<{e}', (S.season == 2014) & (S.date >= lo) & (S.date < e), (S.date < lo) if lo > '2014-01-01' else (S.season < 2014)))
        lo = e
    for y in sorted(S.season.unique()):
        if y > 2014: out.append((str(y), S.season == y, S.season < y))
    return out


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {'n': int(len(y)), 'goals': int(y.sum()), 'xg': float(p.sum()), 'logloss': float(log_loss(y, p)), 'brier': float(brier_score_loss(y, p)), 'auc': float(roc_auc_score(y, p))}


def main():
    S = pd.read_pickle(os.path.join(WORK, 'shots.pkl')).sort_values(['date', 'id', 'k']).reset_index(drop=True)
    S['sit0'] = S.sit
    S.loc[S.sit == 'dfk', 'sit'] = 'open'   # direct-free-kick wording exists only for goals in the old feed -> unusable
    FB, FS, FT = base_frame(S), state_frame(S), sit_frame(S)
    np_ = (S.pen == 0).to_numpy()
    y = S.goal.to_numpy()
    for c in ['xg', 'xg_ns', 'xg_nd', 'xg_gb', 'xg_sit', 'psxg']: S[c] = np.nan
    seas = S.season.to_numpy().astype(float)
    # --update: keep the valuations already stored in work/shots_xg.pkl and only value blocks that contain new shots
    # (the weekly case: a handful of new matches of the current season -> one block, ~1 minute)
    done = np.zeros(len(S), bool)
    old_f = os.path.join(WORK, 'shots_xg.pkl')
    if '--update' in sys.argv and os.path.exists(old_f):
        O = pd.read_pickle(old_f).set_index(['id', 'k'])
        key = pd.MultiIndex.from_frame(S[['id', 'k']])
        hit = key.isin(O.index)
        for c in ['xg', 'xg_ns', 'xg_nd', 'xg_gb', 'xg_sit', 'psxg']:
            S.loc[hit, c] = O[c].reindex(key[hit]).to_numpy()
        done = S.xg.notna().to_numpy()
    # --- penalties: running conversion rate of earlier seasons shrunk to the prior
    for lab, val, trn in blocks(S):
        val = val.to_numpy(); trn = trn.to_numpy()
        if val.sum() == 0 or done[val].all():
            continue
        pt = trn & ~np_
        ppen = (PEN_PRIOR * PEN_N0 + y[pt].sum()) / (PEN_N0 + pt.sum())
        vp = val & ~np_
        for c in ['xg', 'xg_ns', 'xg_nd', 'xg_gb', 'xg_sit']: S.loc[vp, c] = ppen
        S.loc[vp, 'psxg'] = ppen  # penalties: no placement model
        v = val & np_; t = trn & np_
        if t.sum() < 1500:
            p0 = prior_xg(S[v])
            for c in ['xg', 'xg_ns', 'xg_nd', 'xg_gb', 'xg_sit']: S.loc[v, c] = p0
            S.loc[v, 'psxg'] = np.where(S.ot[v] == 1, np.clip(p0 * 3.0, 0.02, 0.95), 0.0)
            print(lab, 'prior table', int(v.sum())); continue
        Fa = pd.concat([FB, FS], axis=1)
        C = 0.05 if t.sum() < 12000 else 0.3
        # recency weights: conversion rates drift (e.g. "very close range" 52% -> 42% over the decade); half-life 3 seasons
        wt = np.power(0.5, (seas[t].max() - seas) / HALF_LIFE)
        mm = Logit(C).fit(Fa[t], y[t], wt[t])
        S.loc[v, 'xg'] = mm.predict(Fa[v])
        S.loc[v, 'xg_ns'] = Logit(C).fit(FB[t], y[t], wt[t]).predict(FB[v])
        S.loc[v, 'xg_nd'] = Logit(C).fit(Fa[t], y[t]).predict(Fa[v])   # no recency weighting (for the report)
        S.loc[v, 'xg_gb'] = GBM().fit(Fa[t], y[t], wt[t]).predict(Fa[v]) if len(lab) == 4 else S.loc[v, 'xg']
        # --- situation variant. The feed prints "following a corner/set piece/fast break" only for goals and for
        # off-target / woodwork attempts (never for saved or blocked ones).  Fit on goals + off/wood non-goals, the
        # latter weighted by 1/P(off or wood | non-goal, x); value saved/blocked shots with the no-situation mixture.
        ng = t & (y == 0)
        obs = S.outcome.isin(['off', 'wood']).to_numpy()
        pi = Logit(C).fit(Fa[ng], obs[ng].astype(int), wt[ng]).predict(Fa)
        tt = t & ((y == 1) | obs)
        w = np.where(y == 1, 1.0, 1.0 / np.clip(pi, 0.15, 1)) * wt
        Fs = pd.concat([FB, FS, FT], axis=1)
        ms = Logit(C).fit(Fs[tt], y[tt], w[tt])
        ps = ms.predict(Fs[v])
        # saved/blocked: expected value over the unknown situation, approximated by the "open" prediction rescaled so
        # that the block's training mean is preserved
        hid = ~((y[v] == 1) | obs[v])
        base = S.loc[v, 'xg'].to_numpy()
        # rescale observed-situation shots so that mean(ps) == mean(base) on the training shots with observed situation
        ratio = (mm.predict(Fa[tt]) * w[tt]).sum() / max(1e-9, (ms.predict(Fs[tt]) * w[tt]).sum())
        S.loc[v, 'xg_sit'] = np.where(hid, base, np.clip(ps * ratio, 0, 0.99))
        # --- post-shot model for on-target attempts
        ot = (S.ot == 1).to_numpy()
        Fp = pd.concat([FB, FS], axis=1).copy()
        pl = S.place.fillna('unk')
        for nm in ['bottom left corner', 'bottom right corner', 'top left corner', 'top right corner', 'centre of the goal', 'top centre of the goal', 'high centre of the goal']:
            Fp['p_' + nm.replace(' ', '_')] = (pl == nm).astype(float)
        tp, vp2 = t & ot, v & ot
        S.loc[v, 'psxg'] = 0.0
        S.loc[vp2, 'psxg'] = Logit(C).fit(Fp[tp], y[tp], wt[tp]).predict(Fp[vp2])
        print(lab, int(t.sum()), '->', int(v.sum()), 'pen', round(ppen, 3))
    # ---------------- evaluation (out-of-sample by construction) ----------------
    ev = {}
    N = S[np_]
    for c in ['xg', 'xg_ns', 'xg_nd', 'xg_gb', 'xg_sit']:
        ev[c] = {'all': metrics(N.goal.to_numpy(), N[c].to_numpy()),
                 'A_2015+': metrics(N.goal[(N.lg == 'A') & (N.season >= 2015)].to_numpy(), N[c][(N.lg == 'A') & (N.season >= 2015)].to_numpy()),
                 'by_season': {int(s): metrics(g.goal.to_numpy(), g[c].to_numpy()) for s, g in N.groupby('season')}}
    base_rate = np.full(len(N), N.goal.mean())
    ev['const'] = {'all': metrics(N.goal.to_numpy(), base_rate)}
    ev['zone_table'] = {'all': metrics(N.goal.to_numpy(), prior_xg(N))}
    q = pd.qcut(N.xg, 10, duplicates='drop')
    ev['calibration_xg'] = [{'bin': str(k), 'n': int(len(g)), 'pred': float(g.xg.mean()), 'obs': float(g.goal.mean())} for k, g in N.groupby(q)]
    hi = pd.cut(N.xg, [0, .02, .05, .1, .2, .3, .5, 1])
    ev['calibration_fixed'] = [{'bin': str(k), 'n': int(len(g)), 'pred': float(g.xg.mean()), 'obs': float(g.goal.mean())} for k, g in N.groupby(hi)]
    O = S[np_ & (S.ot == 1)]
    ev['psxg_on_target'] = metrics(O.goal.to_numpy(), O.psxg.to_numpy()); ev['xg_on_target'] = metrics(O.goal.to_numpy(), O.xg.to_numpy())
    P = S[~np_]
    ev['penalties'] = {'n': int(len(P)), 'conv': float(P.goal.mean()), 'xg_mean': float(P.xg.mean())}
    json.dump(ev, open(os.path.join(WORK, 'xg_eval.json'), 'w'), indent=1)
    S.to_pickle(os.path.join(WORK, 'shots_xg.pkl'))
    for c in ['const', 'zone_table', 'xg_ns', 'xg_nd', 'xg', 'xg_gb', 'xg_sit']:
        print(f'{c:11s}', {k: round(v, 4) for k, v in ev[c]['all'].items()})
    print('A 2015+   ', {c: round(ev[c]['A_2015+']['logloss'], 4) for c in ['xg_ns', 'xg_nd', 'xg', 'xg_gb', 'xg_sit']})
    for r in ev['calibration_xg']: print('  ', r)
    for r in ev['calibration_fixed']: print('  ', r)
    print('on target: psxg', ev['psxg_on_target'], '\n           xg  ', ev['xg_on_target'])
    print('pens', ev['penalties'])
    print(pd.DataFrame({c: {s: round(m['logloss'], 4) for s, m in ev[c]['by_season'].items()} for c in ['xg_ns', 'xg_nd', 'xg', 'xg_gb', 'xg_sit']}).to_string())
    print(pd.DataFrame({'goals': {s: m['goals'] for s, m in ev['xg']['by_season'].items()}, 'xg': {s: round(m['xg'], 1) for s, m in ev['xg']['by_season'].items()}}).T.to_string())


if __name__ == '__main__':
    main()
