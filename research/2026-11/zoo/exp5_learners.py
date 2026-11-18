"""Stage 5: meta-learner comparison on the selected feature set S1 (DEV only)."""
import numpy as np, pandas as pd, time, sys
from models import *
F = load()
mm = (F.ev & (F.season <= 2021)).values; kk = F.k.values[mm]; ss = F.season.values[mm]
def dev(P): return rps_ll(P[mm], kk)[0]
rv1 = dev(F[['v_pH', 'v_pD', 'v_pA']].values)
S1 = ['f_tot_r_40', 'v_slc', 'f_dom_r_4', 'f_dom_r_40', 'f_shr_r_40', 'f_sotr_r_40', 'f_cor_r_40', 'f_pos_raw_40', 'st_deadh_d', 'mn_susp_d']
OUT = {}
def rep(P, name, base=None):
    r = dev(P); d = r - rv1; lo, hi = boot_ci(d, 1000); l = rps_ll(P[mm], kk)[1].mean()
    s = f'{name:40s} RPS {r.mean():.5f} LL {l:.5f} vs v1 {d.mean():+.5f} [{lo:+.5f},{hi:+.5f}]'
    if base is not None:
        e = r - dev(base); lo, hi = boot_ci(e, 1000); s += f' | vs CO-S1 {e.mean():+.5f} [{lo:+.5f},{hi:+.5f}] [13-17 {e[ss <= 2017].mean():+.5f} | 18-21 {e[ss >= 2018].mean():+.5f}]'
    print(s, flush=True); OUT[name] = P; return r
what = sys.argv[1] if len(sys.argv) > 1 else 'glm'
co = walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False)); rep(co, 'CO-S1 tau=.1 wB=1')
if what == 'glm':
    mn = walk(F, fp_mn(feats=S1, lam=1.0 / (0.1 ** 2 * 9000), wB=1.0)); rep(mn, 'MN-S1', co)
    mn = walk(F, fp_mn(feats=S1, dfeats=['v_slc', 'f_tot_r_40', 'lgd_draw'], lam=1.0 / (0.1 ** 2 * 9000), wB=1.0)); rep(mn, 'MN-S1 + draw feats', co)
    for loss in ('ll', 'rps'):
        P = walk(F, fp_co(feats=S1, tau=0.1 if loss == 'll' else 0.25, wB=1.0, free_scale=False, loss=loss)); rep(P, f'CO-S1 loss={loss}', co)
    # kitchen sink
    meta = {'id', 'lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag', 'k', 'isA', 'ev', 'v_lh', 'v_la', 'v_pH', 'v_pD', 'v_pA', 'n_h', 'n_a', 'v_dl', 'v_gsl', 'st_deadh_h', 'st_deadh_a'}
    ALLF = [c for c in F.columns if c not in meta]
    print(len(ALLF), 'features in kitchen sink')
    for tau in (0.01, 0.02, 0.05):
        P = walk(F, fp_co(feats=ALLF, tau=tau, wB=1.0, free_scale=False)); rep(P, f'CO kitchen-sink ({len(ALLF)}) tau={tau}', co)
if what == 'lgb':
    meta = {'id', 'lg', 'season', 'date', 'h', 'a', 'status', 'hg', 'ag', 'k', 'ev', 'v_pH', 'v_pD', 'v_pA', 'st_deadh_h', 'st_deadh_a'}
    ALLF = [c for c in F.columns if c not in meta]
    t = time.time()
    P = walk(F, fp_lgb_mc(S1, rounds=150, wB=1.0)); rep(P, 'LGB-mc S1 150r depth2', co); print(time.time() - t)
    P = walk(F, fp_lgb_mc(S1 + ['v_dl', 'v_sl'], rounds=300, lr=0.01, wB=1.0)); rep(P, 'LGB-mc S1+v 300r lr.01', co)
    P = walk(F, fp_lgb_mc(ALLF, rounds=150, wB=1.0)); rep(P, f'LGB-mc all({len(ALLF)}) 150r', co)
    P = walk(F, fp_lgb_mc(ALLF, rounds=60, wB=1.0, min_leaf=400, l2=200)); rep(P, f'LGB-mc all 60r heavy reg', co)
    mono = [-1, -1, 1, 1, 1, 1, 1, 1, -1, -1]
    P = walk(F, fp_lgb_pois(S1, rounds=150, wB=1.0, mono=mono)); rep(P, 'LGB-pois S1 monotone', co)
    P = walk(F, fp_lgb_pois(S1, rounds=150, wB=1.0)); rep(P, 'LGB-pois S1 free', co)
    # GBDT on top of the GLM (residual boosting from CO-S1 probabilities)
    Fx = F.copy(); coA = walk(F, fp_co(feats=S1, tau=0.1, wB=1.0, free_scale=False), seasons=range(2011, 2027), ev=(F.season >= 2011))
    Fx[['c_pH', 'c_pD', 'c_pA']] = coA
    Fx2 = Fx[Fx.season >= 2011].copy(); Fx2.index = Fx2.index  # keep global index
    P = np.full((len(F), 3), np.nan)
    Pp = walk(Fx, fp_lgb_mc(ALLF, rounds=100, wB=1.0, base=('c_pH', 'c_pD', 'c_pA'), recal=False), train_from=2011); rep(Pp, 'LGB-mc all on top of CO-S1 (nested)', co)
if what == 'pool':
    F[['u_pH', 'u_pD', 'u_pA']] = np.array([0.50, 0.26, 0.24])
    evx = (F.season >= 2011)
    R = {}
    for r in ['elo10_d', 'elo20_d', 'elo35_d', 'pi035_d', 'pi07_d', 'ber05_d', 'ber12_d']:
        R[r] = walk(F, fp_co(feats=[r], dfeats=[], tau=1.0, wB=1.0, free_scale=False, base=('u_pH', 'u_pD', 'u_pA')), seasons=range(2011, 2027), ev=evx)
        rep(R[r], f'single rating model: {r}')
    R['zoo_goal'] = walk(F, fp_co(feats=['elo20_d', 'pi035_d', 'ber05_d', 'ber05_s'], tau=1.0, wB=1.0, free_scale=False, base=('u_pH', 'u_pD', 'u_pA')), seasons=range(2011, 2027), ev=evx)
    rep(R['zoo_goal'], 'zoo model: elo20+pi035+ber05 (no v1)')
    R['zoo_all'] = walk(F, fp_co(feats=['elo20_d', 'pi035_d', 'ber05_d', 'ber05_s', 'eloxg_d', 'pixg_d', 'gapsh_d', 'gapsot_d', 'gapco_d', 'pos_d'], tau=0.1, wB=1.0, free_scale=False, base=('u_pH', 'u_pD', 'u_pA')), seasons=range(2011, 2027), ev=evx)
    rep(R['zoo_all'], 'zoo model: goal + shot ratings (no v1)')
    V = F[['v_pH', 'v_pD', 'v_pA']].values
    rec = walk(F, fp_co(tau=0.1, wB=1.0, free_scale=False), seasons=range(2011, 2027), ev=evx); rep(rec, 'v1 recal (wB=1)')
    for nm, mem in [('v1+elo20', ['elo20_d']), ('v1+elo20+pi035+ber05', ['elo20_d', 'pi035_d', 'ber05_d']), ('v1+zoo_goal', ['zoo_goal']), ('v1+zoo_all', ['zoo_all'])]:
        rep((V + sum(R[x] for x in mem)) / (1 + len(mem)), f'equal linear pool raw {nm}')
        rep((rec + sum(R[x] for x in mem)) / (1 + len(mem)), f'equal linear pool recal {nm}')
    # walk-forward fitted weights (linear and log-linear), fitted on out-of-sample forecasts of earlier seasons
    for kind in ('lin', 'log'):
        for nm, mem in [('recal+elo20+pi035+ber05', ['elo20_d', 'pi035_d', 'ber05_d']), ('recal+zoo_goal+zoo_all', ['zoo_goal', 'zoo_all'])]:
            P = np.full((len(F), 3), np.nan); W = {}
            for Y in EVAL_SEASONS:
                tr = ((F.season >= 2011) & (F.season < Y) & (F.lg == 'A')).values; te = ((F.season == Y) & F.ev).values
                Ps = [rec] + [R[x] for x in mem]
                w, mix = pool_fit([p[tr] for p in Ps], F.k.values[tr], np.ones(tr.sum()), kind)
                ww = np.array(w)
                if kind == 'lin': P[te] = sum(wi * p[te] for wi, p in zip(ww, Ps))
                else:
                    E = sum(wi * np.log(np.clip(p[te], 1e-9, 1)) for wi, p in zip(ww, Ps)); E = np.exp(E); P[te] = E / E.sum(1, keepdims=True)
                W[Y] = ww.round(2)
            rep(P, f'fitted {kind} pool {nm}'); print('      weights 2015/2021:', W[2015], W[2021])
    np.save('cache_pool_parts.npy', np.stack([rec] + [R[x] for x in ['elo20_d', 'pi035_d', 'ber05_d', 'zoo_goal', 'zoo_all']]))
