"""Final walk-forward run: writes preds_<variant>.csv for every variant kept in the report (+ loo/ for the ablation).
Usage: /home/user/bdv-lab/venv/bin/python run_final.py        (needs features_all.pkl from build_features.py)"""
import os, time
import numpy as np, pandas as pd
from models import *

F = load()
os.makedirs(Z + 'loo', exist_ok=True)
GR = {'open': ['f_tot_r_40', 'v_slc'],
      'dom': ['f_dom_r_4', 'f_dom_r_40'],
      'statform': ['f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40'],
      'poss': ['f_pos_raw_40'],
      'dead': ['st_deadh_d'],
      'susp': ['mn_susp_d']}
S1 = sum(GR.values(), [])
CO = dict(tau=0.1, wB=1.0, free_scale=False)
t0 = time.time()


def out(P, name, sub=''):
    m = F.ev.values
    d = pd.DataFrame(dict(id=F.id.values[m], pH=P[m, 0], pD=P[m, 1], pA=P[m, 2]))
    assert len(d) == 5279 and d[['pH', 'pD', 'pA']].notna().all().all()
    d.to_csv(Z + sub + f'preds_{name}.csv', index=False, float_format='%.6f')
    print(f'{time.time() - t0:6.0f}s  wrote {sub}preds_{name}.csv', flush=True)


# 1 recalibration only
rec = walk(F, fp_co(**CO)); out(rec, 'recal')
# 2-5 the zoo without v1, and pooled with v1
F[['u_pH', 'u_pD', 'u_pA']] = np.array([0.50, 0.26, 0.24]); U = ('u_pH', 'u_pD', 'u_pA')
out(walk(F, fp_co(feats=['elo20_d'], tau=1.0, wB=1.0, free_scale=False, base=U)), 'elo20_only')
out(walk(F, fp_co(feats=['elo20_d', 'pi035_d', 'ber05_d', 'ber05_s'], tau=1.0, wB=1.0, free_scale=False, base=U)), 'zoo_goal')
zoo = walk(F, fp_co(feats=['elo20_d', 'pi035_d', 'ber05_d', 'ber05_s', 'eloxg_d', 'pixg_d', 'gapsh_d', 'gapsot_d', 'gapco_d', 'pos_d'], tau=0.1, wB=1.0, free_scale=False, base=U))
out(zoo, 'zoo_all'); out((rec + zoo) / 2, 'pool_recal_zoo')
# 6-10 single groups on top of the recalibration, core
out(walk(F, fp_co(feats=GR['open'], **CO)), 'co_open')
out(walk(F, fp_co(feats=GR['dom'], **CO)), 'co_dom')
out(walk(F, fp_co(feats=GR['statform'] + GR['poss'], **CO)), 'co_stat')
out(walk(F, fp_co(feats=GR['dead'] + GR['susp'], **CO)), 'co_deadsusp')
out(walk(F, fp_co(feats=GR['open'] + GR['dom'] + GR['statform'], **CO)), 'co_core')
# 11 the selected model
co, C = walk(F, fp_co(feats=S1, **CO), keep_coef=True); out(co, 'co_s1')
pd.DataFrame(C).T.drop(columns=['a1', 'a2']).round(4).to_csv(Z + 'coef_co_s1.csv')
# 12 nested-CV ridge
if os.path.exists(Z + 'cache_cvtau.npy'): out(np.load(Z + 'cache_cvtau.npy'), 'co_s1_cvtau')
# 13-14 other GLM learners
po = walk(F, fp_po(feats=S1, tau=0.1, wB=1.0)); out(po, 'po_s1'); out((co + po) / 2, 'mean_co_po')
out(walk(F, fp_mn(feats=S1, lam=1.0 / (0.1 ** 2 * 9000), wB=1.0)), 'mn_s1')
# 15 does the ratings zoo add anything on top of S1?
out(walk(F, fp_co(feats=S1 + ['elo20_d', 'pi035_d', 'ber05_d', 'eloxg_d', 'gapsh_d', 'gapsot_d'], **CO)), 'co_s1_plus_zoo')
out(0.8 * co + 0.2 * zoo, 'pool_co_s1_zoo')
# 16 kitchen sink
meta = {'id', 'lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag', 'k', 'isA', 'ev', 'v_lh', 'v_la', 'v_pH', 'v_pD', 'v_pA', 'n_h', 'n_a', 'v_dl', 'v_gsl', 'st_deadh_h', 'st_deadh_a', 'u_pH', 'u_pD', 'u_pA'}
ALLF = [c for c in F.columns if c not in meta]
out(walk(F, fp_co(feats=ALLF, tau=0.01, wB=1.0, free_scale=False)), 'co_kitchen')
# ablation (leave one group out)
for g in GR: out(walk(F, fp_co(feats=[f for f in S1 if f not in GR[g]], **CO)), f'co_s1_minus_{g}', 'loo/')
# 17-19 gradient boosting
out(walk(F, fp_lgb_mc(S1, rounds=150, wB=1.0)), 'lgb_mc_s1')
out(walk(F, fp_lgb_pois(S1, rounds=150, wB=1.0, mono=[-1, -1, 1, 1, 1, 1, 1, 1, -1, -1])), 'lgb_pois_s1')
metaL = {'id', 'lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag', 'k', 'ev', 'v_pH', 'v_pD', 'v_pA', 'st_deadh_h', 'st_deadh_a', 'u_pH', 'u_pD', 'u_pA'}
out(walk(F, fp_lgb_mc([c for c in F.columns if c not in metaL], rounds=60, wB=1.0, min_leaf=400, l2=200)), 'lgb_mc_all')

# demo: the selected model on the fixtures that have not been played yet (trained on everything finished before 2026)
A = pd.read_pickle(Z + 'features_all.pkl'); fut = A[(A.status == 'pre') & (A.lg == 'A')].reset_index(drop=True)
if len(fut):
    P = fp_co(feats=S1, **CO)(F[(F.season >= 2009) & (F.season < 2026)], fut, 2026)
    pd.DataFrame(dict(id=fut.id, date=fut.date, h=fut.h, a=fut.a, pH=P[:, 0], pD=P[:, 1], pA=P[:, 2])).round(5).to_csv(Z + 'future_co_s1.csv', index=False)
    print('future fixtures predicted:', len(fut))
