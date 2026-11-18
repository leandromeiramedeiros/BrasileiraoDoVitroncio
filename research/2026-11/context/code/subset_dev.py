"""DEV-only subset comparison of work/preds_<name>.csv against a reference variant (default: recal)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, fitlib as F
D = F.load()
M, base = F._truth()
ref_name = 'recal'
names = sys.argv[1:]
ref = pd.read_csv(os.path.join(F.HERE, 'work', f'preds_{ref_name}.csv'), dtype={'id': str})
rnd = (np.maximum(D.gp_h, D.gp_a) + 1).rename('rnd')
for n in names:
    P = pd.read_csv(os.path.join(F.HERE, 'work', f'preds_{n}.csv'), dtype={'id': str})
    d = M.merge(P, on='id'); d = d[d.season <= 2021].merge(rnd, left_on='id', right_index=True)
    r, _ = F.rps_ll(d); rb, _ = F.rps_ll(M.merge(ref, on='id').set_index('id').loc[d.id].reset_index()); rv, _ = F.rps_ll(M.merge(base, on='id').set_index('id').loc[d.id].reset_index())
    line = f'{n:16s}'
    for lab, m in [('all', d.rnd > 0), ('r1-5', d.rnd <= 5), ('r1-10', d.rnd <= 10), ('r11-28', (d.rnd > 10) & (d.rnd <= 28)), ('r29-38', d.rnd > 28)]:
        m = m.to_numpy(); diff = (r - rb)[m]; rng = np.random.default_rng(3)
        bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        line += f' | {lab} n={m.sum()} vs{ref_name} {diff.mean():+.5f} [{np.percentile(bs,2.5):+.5f},{np.percentile(bs,97.5):+.5f}] vs v1 {(r-rv)[m].mean():+.5f}'
    print(line)
