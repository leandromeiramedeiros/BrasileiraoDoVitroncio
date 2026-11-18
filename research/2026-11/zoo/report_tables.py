"""Builds tables.md: official score.py numbers for every variant + paired comparisons between variants (DEV/HOLDOUT).
This is the ONE final look at the HOLDOUT; nothing was selected with it."""
import glob, os, re, subprocess, sys, json
import numpy as np, pandas as pd
from lib import S, Z, rps_ll

PY = '/home/user/bdv-lab/venv/bin/python'
ORDER = ['recal', 'elo20_only', 'zoo_goal', 'zoo_all', 'pool_recal_zoo', 'co_open', 'co_dom', 'co_stat', 'co_deadsusp', 'co_core', 'co_s1', 'co_s1_cvtau',
         'po_s1', 'mean_co_po', 'mn_s1', 'co_s1_plus_zoo', 'pool_co_s1_zoo', 'co_kitchen', 'lgb_mc_s1', 'lgb_pois_s1', 'lgb_mc_all']
files = [Z + f'preds_{n}.csv' for n in ORDER if os.path.exists(Z + f'preds_{n}.csv')]
raw = subprocess.run([PY, S + 'score.py'] + files, capture_output=True, text=True).stdout
open(Z + 'score_official.txt', 'w').write(raw)
rows = []; cur = None
for line in raw.splitlines():
    m = re.match(r'== (.*?)\s+\(jogos: (\d+), faltando: (\d+)\)', line)
    if m: cur = os.path.basename(m.group(1)).replace('preds_', '').replace('.csv', ''); continue
    m = re.match(r'\s+(DEV|HOLDOUT|TOTAL)\S*\s.*?n=\s*(\d+)\s+RPS ([\d.]+)\s+LL ([\d.]+) \| baseline RPS ([\d.]+) LL ([\d.]+) \| dRPS ([+-][\d.]+) \[IC95 ([+-][\d.]+),([+-][\d.]+)\]\s+dLL ([+-][\d.]+) \| mercado RPS ([\d.]+)', line)
    if m: rows.append(dict(variant=cur, split=m.group(1), n=int(m.group(2)), RPS=m.group(3), LL=m.group(4), dRPS=m.group(7), lo=m.group(8), hi=m.group(9), dLL=m.group(10), mkt=m.group(11), bRPS=m.group(5), bLL=m.group(6)))
R = pd.DataFrame(rows)
md = ['## Official scores (shared/score.py) -- dRPS is the paired difference vs the v1 baseline, bootstrap 95% CI\n']
b = R[R.variant == R.variant.iloc[0]].set_index('split')
md.append(f"Baseline v1: DEV RPS {b.loc['DEV', 'bRPS']} LL {b.loc['DEV', 'bLL']} | HOLDOUT RPS {b.loc['HOLDOUT', 'bRPS']} LL {b.loc['HOLDOUT', 'bLL']} | market (closing odds, ruler only): DEV {b.loc['DEV', 'mkt']}, HOLDOUT {b.loc['HOLDOUT', 'mkt']}\n")
md.append('| variant | DEV RPS | DEV LL | DEV dRPS [95% CI] | HOLD RPS | HOLD LL | HOLD dRPS [95% CI] | TOTAL dRPS [95% CI] |')
md.append('|---|---|---|---|---|---|---|---|')
for v in R.variant.unique():
    x = R[R.variant == v].set_index('split'); d, h, t = x.loc['DEV'], x.loc['HOLDOUT'], x.loc['TOTAL']
    md.append(f"| {v} | {d.RPS} | {d.LL} | {d.dRPS} [{d.lo}, {d.hi}] | {h.RPS} | {h.LL} | {h.dRPS} [{h.lo}, {h.hi}] | {t.dRPS} [{t.lo}, {t.hi}] |")

# ---- paired comparisons between variants
M = pd.DataFrame(json.load(open(S + 'matches.json')))[['id', 'lg', 'season', 'status', 'hg', 'ag']]
M = M[(M.lg == 'A') & (M.season >= 2013) & (M.status == 'post')].copy(); M['id'] = M['id'].astype(str)
M['k'] = np.where(M.hg > M.ag, 0, np.where(M.hg == M.ag, 1, 2))
def rps_of(path):
    d = M.merge(pd.read_csv(path, dtype={'id': str}), on='id'); assert len(d) == len(M)
    return rps_ll(d[['pH', 'pD', 'pA']].values, d.k.values)[0]
base = rps_of(S + 'preds_baseline.csv')
dev = (M.season <= 2021).values; hold = ~dev; seas = M.season.values
def ci(d):
    rng = np.random.default_rng(7); bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    return f'{d.mean():+.5f} [{np.percentile(bs, 2.5):+.5f}, {np.percentile(bs, 97.5):+.5f}]'
def pair(a, b): return f'| {ci((a - b)[dev])} | {ci((a - b)[hold])} |'
P = {os.path.basename(f)[6:-4]: rps_of(f) for f in files}
md += ['\n## Increment of each feature group on top of the walk-forward recalibration (paired, variant minus `recal`)\n', '| variant | DEV | HOLDOUT |', '|---|---|---|']
for v in ['co_open', 'co_dom', 'co_stat', 'co_deadsusp', 'co_core', 'co_s1']:
    if v in P: md.append(f'| {v} − recal {pair(P[v], P["recal"])}')
md += ['\n## Ablation: leave one group out of the selected model (paired, `co_s1` without the group minus `co_s1`; positive = the group helps)\n', '| group removed | DEV | HOLDOUT |', '|---|---|---|']
for f in sorted(glob.glob(Z + 'loo/preds_co_s1_minus_*.csv')):
    g = os.path.basename(f)[len('preds_co_s1_minus_'):-4]; md.append(f'| {g} {pair(rps_of(f), P["co_s1"])}')
md += ['\n## Other learners / combinations against the selected model (paired, variant minus `co_s1`; positive = worse than co_s1)\n', '| variant | DEV | HOLDOUT |', '|---|---|---|']
for v in ['co_s1_cvtau', 'po_s1', 'mean_co_po', 'mn_s1', 'co_s1_plus_zoo', 'pool_co_s1_zoo', 'co_kitchen', 'lgb_mc_s1', 'lgb_pois_s1', 'lgb_mc_all', 'co_core']:
    if v in P: md.append(f'| {v} {pair(P[v], P["co_s1"])}')
md += ['\n## `co_s1` minus v1 by season (paired dRPS)\n', '| season | n | v1 RPS | co_s1 RPS | dRPS | recal dRPS |', '|---|---|---|---|---|---|']
for s in sorted(set(seas)):
    m = seas == s; md.append(f'| {s}{" (holdout)" if s >= 2022 else ""} | {m.sum()} | {base[m].mean():.5f} | {P["co_s1"][m].mean():.5f} | {(P["co_s1"] - base)[m].mean():+.5f} | {(P["recal"] - base)[m].mean():+.5f} |')
open(Z + 'tables.md', 'w').write('\n'.join(md) + '\n')
print('\n'.join(md))
