"""Reporting tables for REPORT.md (run AFTER model selection; reads the preds_*.csv written by run_final.py)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import fitlib as F, variants as V
from run_final import FINAL, pred_path
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
D = F.load(); M, base = F._truth()
rnd = (np.maximum(D.gp_h, D.gp_a) + 1).rename('rnd')
def rps_of(P):
    d = M.merge(P, on='id').merge(rnd, left_on='id', right_index=True); r, l = F.rps_ll(d); d['rps'] = r; d['ll'] = l
    return d.set_index('id')
B = rps_of(base); R = rps_of(pd.read_csv(pred_path('recal'), dtype={'id': str}))
def ci(x, seed=7):
    rng = np.random.default_rng(seed); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(2000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
rows = []; per = {}
for name, blocks, tau, lgs in FINAL:
    P = rps_of(pd.read_csv(pred_path(name), dtype={'id': str}))
    row = {'variant': name, 'k': len(V.spec(*blocks))}
    for lab, m in (('DEV', P.season <= 2021), ('HOLD', P.season >= 2022)):
        m = m.to_numpy()
        a = ci((P.rps - B.rps.loc[P.index]).to_numpy()[m]); b = ci((P.rps - R.rps.loc[P.index]).to_numpy()[m])
        row.update({f'{lab}_rps': P.rps[m].mean(), f'{lab}_ll': P.ll[m].mean(), f'{lab}_d': a[0], f'{lab}_lo': a[1], f'{lab}_hi': a[2],
                    f'{lab}_dr': b[0], f'{lab}_dr_lo': b[1], f'{lab}_dr_hi': b[2]})
    rows.append(row); per[name] = P
T = pd.DataFrame(rows); T.to_csv(os.path.join(F.HERE, 'coef', 'final_table.csv'), index=False)
print(T.round(5).to_string(index=False))
print('\n-- per-season dRPS vs v1 (ctx_core, ctx_core_r, recal)')
S = pd.DataFrame({n: (per[n].rps - B.rps.loc[per[n].index]).groupby(per[n].season).mean() for n in ('recal', 'ctx_core', 'ctx_core_r')})
S['n'] = B.groupby('season').size(); print(S.round(5).to_string())
print('\n-- by round block, dRPS vs recal (DEV | HOLDOUT)')
for n in ('r_all', 'r_opt', 'r_fam', 'r_cupcont', 'r_fam_early', 'ctx_core', 'ctx_core_r'):
    P = per[n]; line = f'{n:12s}'
    for lab, sm in (('DEV', P.season <= 2021), ('HOLD', P.season >= 2022)):
        for rl, rm in (('r1-10', P.rnd <= 10), ('r11-28', (P.rnd > 10) & (P.rnd <= 28)), ('r29-38', P.rnd > 28)):
            m = (sm & rm).to_numpy(); a = ci((P.rps - R.rps.loc[P.index]).to_numpy()[m])
            line += f' | {lab} {rl} n={m.sum()} {a[0]:+.5f} [{a[1]:+.5f},{a[2]:+.5f}]'
    print(line)
print('\n-- coefficient path ctx_core_r'); print(pd.read_csv(os.path.join(F.HERE, 'coef', 'ctx_core_r.csv'), index_col=0).round(4).to_string())
print('\n-- effect sizes, fit on Serie A 2009-2021 (DEV era) and 2009-2026 (all), weak ridge (tau=1)')
for ms in (2021, 2026):
    print(ms); print(F.insample(D, V.spec(*V.GROUPS['ctx_core_r']), max_season=ms, start=2009).round(4).to_string())
print('\n-- holdout-era only (2022-2026) fit, for sign replication'); print(F.insample(D, V.spec('decided2', 'nextcont', 'homedisp', 'r_all'), max_season=2026, start=2022).round(4).to_string())
