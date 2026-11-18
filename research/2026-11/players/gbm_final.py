#!/usr/bin/env python
"""Frozen squad-features GBM (configs exactly as in exp8.py; no tuning, no holdout numbers computed here).

Model: LightGBM multiclass (H/D/A) with init_score = log of the v1 baseline probabilities, i.e. a probability-level
correction on top of v1. The prediction is the average of the two frozen exp8 configs. For season Y the trees are fitted
only on matches of seasons < Y (Série A + B, from 2008) that have the features.

Two variants
  preds_players_gbm.csv         (a) information available a day or more before kickoff only (u_* columns: past lineups,
                                    minutes, bench lists, suspensions implied by cards, career box scores)
  preds_players_gbm_lineup.csv  (b) the announced starting XI and bench of THIS match (k_* columns), exp8 feature list;
                                    matches without a roster fall back to (a)
  future_players_gbm.csv        (a) applied to the unplayed 2026 Série A fixtures (features as of the last played round)
Matches where the teams have no usable history keep the v1 baseline probabilities.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
import experiments as E, lab

D = E.D; K = E.K_AVAIL; U = E.U_AVAIL

CONFIGS = (dict(num_leaves=4, learning_rate=0.02, n_estimators=150, min_child_samples=150, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, reg_lambda=20),
           dict(num_leaves=3, learning_rate=0.02, n_estimators=80, min_child_samples=250, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=50))

# (b) announced lineup of this match - identical to exp8
LINEUP_DIFF = ['k_usage', 'k_usage_s', 'k_gashare', 'k_shshare', 'k_exp', 'k_bench_usage', 'k_squad_usage', 'k_sh90', 'k_ga90', 'k_bench_n', 'k_chg',
               'k_new', 'k_debut', 'k_rapm1x', 'k_bench_gashare', 'u_usage10', 'u_susp', 'n_season']
LINEUP_SIDE = ['k_usage', 'k_squad_usage', 'k_exp', 'n_season']
# (a) history-only analogues of the same quantities (declared a priori, not selected on results)
HIST_DIFF = ['u_e_usage', 'u_lag1_usage', 'u_lag3_usage', 'u_usage10', 'u_e_gashare', 'u_lag3_gashare', 'u_exp', 'u_lag3_bench_usage',
             'u_lag3_squad_usage', 'u_avail', 'u_sh90', 'u_ga90', 'u_susp', 'u_absent', 'u_early_off', 'u_rapm1x', 'u_lag3_bench_gashare', 'n_season']
HIST_SIDE = ['u_lag3_usage', 'u_lag3_squad_usage', 'u_exp', 'n_season']

P0 = D[['pH', 'pD', 'pA']].to_numpy(float)
k = D.k.to_numpy(); season = D.season.to_numpy(); post = D.post.to_numpy()


def design(diff, side):
    names = [f + ':diff' for f in diff] + [f + s for f in side for s in ('_h', '_a')]
    X = np.column_stack([D[f + '_h'].to_numpy(float) - D[f + '_a'].to_numpy(float) for f in diff] +
                        [D[f + s].to_numpy(float) for f in side for s in ('_h', '_a')])
    return X, names


def walk(diff, side, avail, fallback):
    X, names = design(diff, side)
    Pn = fallback.copy()
    imp = np.zeros(len(names))
    for Y in range(2013, 2027):
        tr = avail & post & (season < Y) & (season >= 2008)
        te = avail & (season == Y)
        if te.sum() == 0: continue
        acc = np.zeros((te.sum(), 3))
        for params in CONFIGS:
            m = lgb.LGBMClassifier(objective='multiclass', verbose=-1, random_state=1, n_jobs=2, **params)
            m.fit(X[tr], k[tr], init_score=np.log(P0[tr]))
            raw = m.predict(X[te], raw_score=True) + np.log(P0[te])
            e = np.exp(raw - raw.max(1, keepdims=True)); acc += e / e.sum(1, keepdims=True)
            if Y == 2026: imp += m.booster_.feature_importance('gain') / m.booster_.feature_importance('gain').sum() / len(CONFIGS)
        Pn[te] = acc / len(CONFIGS)
    return Pn, pd.Series(imp, index=names).sort_values(ascending=False)


Pa, imp_a = walk(HIST_DIFF, HIST_SIDE, U, P0)
Pb, imp_b = walk(LINEUP_DIFF, LINEUP_SIDE, K, Pa)

S = D[D.scored]
for name, P in (('players_gbm', Pa), ('players_gbm_lineup', Pb)):
    pd.DataFrame({'id': S['id'], 'pH': P[S.index, 0], 'pD': P[S.index, 1], 'pA': P[S.index, 2]}).to_csv(f'{lab.ROOT}/preds_{name}.csv', index=False, float_format='%.6f')
F = D[(D.lg == 'A') & (D.status == 'pre')]
fut = pd.DataFrame({'id': F['id'], 'date': F['date'], 'pH': Pa[F.index, 0], 'pD': Pa[F.index, 1], 'pA': Pa[F.index, 2],
                    'base_pH': P0[F.index, 0], 'base_pD': P0[F.index, 1], 'base_pA': P0[F.index, 2], 'has_features': U[F.index].astype(int)})
fut.to_csv(f'{lab.ROOT}/future_players_gbm.csv', index=False, float_format='%.6f')
print('scored rows:', len(S), '| with history features:', int(U[S.index].sum()), '| with announced lineup:', int(K[S.index].sum()), '| future fixtures:', len(fut), 'with features:', int(fut.has_features.sum()))
print('\n(a) pre-kickoff-day model, gain share of the 2026 model:'); print(imp_a.head(12).round(3).to_string())
print('\n(b) announced-lineup model, gain share of the 2026 model:'); print(imp_b.head(12).round(3).to_string())
