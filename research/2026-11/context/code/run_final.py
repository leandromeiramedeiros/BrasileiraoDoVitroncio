"""Final variant list of the context track: clean walk-forward predictions + scores.

  python code/run_final.py            -> fit every variant, write preds_<name>.csv / coef/<name>.csv, print DEV scores
  python code/run_final.py --holdout  -> additionally run the official scorer (DEV + HOLDOUT) on every preds file
"""
import sys, os, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import fitlib as F, variants as V

G = V.GROUPS
FINAL = [  # name, blocks, tau, training leagues
    ('recal',        [], 0.02, 'A'),
    # (a) extra matches as rating evidence
    ('r_all',        ['r_all'], 0.02, 'A'),
    ('r_opt',        ['r_opt'], 0.02, 'A'),
    ('r_fam',        ['r_fam'], 0.02, 'A'),
    ('r_cupcont',    ['r_cupcont'], 0.02, 'A'),
    ('r_fam_early',  ['r_fam_early'], 0.02, 'A'),
    # (b) schedule
    ('rest',         ['rest'], 0.02, 'A'),
    ('sched_all',    G['sched_all'], 0.02, 'A'),
    ('nextcont',     ['nextcont'], 0.05, 'A'),
    ('travel',       ['travel'], 0.02, 'A'),
    # geography
    ('dist',         ['dist'], 0.02, 'A'),
    ('geo_all',      G['geo_all'], 0.02, 'A'),
    ('climate',      ['climate'], 0.02, 'A'),
    ('derby',        ['derby'], 0.02, 'A'),
    ('homedisp',     ['homedisp'], 0.05, 'A'),
    # stakes / season context
    ('covid',        ['covid'], 0.02, 'A'),
    ('late',         ['late'], 0.02, 'A'),
    ('stakes_all',   G['stakes_all'], 0.02, 'A'),
    ('decided',      ['decided'], 0.05, 'A'),
    # combinations
    ('ctx_core',     G['ctx_core'], 0.05, 'A'),
    ('ctx_core_r',   G['ctx_core_r'], 0.05, 'A'),
    ('sink',         G['sink'], 0.02, 'A'),
    # production robustness: same as ctx_core but the 'decided' flags come from a league table that is 4 days stale
    ('ctx_core_lag4', G['ctx_core_lag4'], 0.05, 'A'),
]
KEEP = {'recal', 'ctx_core', 'ctx_core_r', 'ctx_core_lag4', 'nextcont', 'decided', 'homedisp', 'r_all'}   # written to the track root; the rest to variants/
def pred_path(name): return os.path.join(F.HERE, f'preds_{name}.csv') if name in KEEP else os.path.join(F.HERE, 'variants', f'preds_{name}.csv')

if __name__ == '__main__':
    D = F.load()
    for d in ('coef', 'variants', 'future'): os.makedirs(os.path.join(F.HERE, d), exist_ok=True)
    rows = []; files = []
    for name, blocks, tau, lgs in FINAL:
        L, path = F.walk_forward(D, V.spec(*blocks), tau=tau, leagues=tuple(lgs))
        out = pred_path(name); files.append(out)
        P = F.to_probs(L, out)
        # protocol: preds files hold exactly the scored matches (Série A 2013+, finished); fixtures still to be played go to future/
        ok = P.id.isin(F._truth()[0].id)
        if name in KEEP and (~ok).any(): P[~ok].to_csv(os.path.join(F.HERE, 'future', f'future_{name}.csv'), index=False)
        P = P[ok]; P.to_csv(out, index=False)
        path.round(5).to_csv(os.path.join(F.HERE, 'coef', f'{name}.csv'))
        L.to_csv(os.path.join(F.HERE, 'coef', f'lambda_{name}.csv'), index=False)
        r = F.score_dev(P, name)
        if name == 'recal': ref = P
        r2 = F.score_dev(P, name, ref=ref, quiet=True)
        r.update({'vs_recal': r2['drps'], 'vs_recal_lo': r2['lo'], 'vs_recal_hi': r2['hi'], 'k': len(V.spec(*blocks)), 'tau': tau})
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(F.HERE, 'coef', 'dev_table.csv'), index=False)
    pd.set_option('display.width', 250)
    print(T[['label', 'k', 'tau', 'rps', 'll', 'drps', 'lo', 'hi', 'vs_recal', 'vs_recal_lo', 'vs_recal_hi', 'seasons_better']].round(5).to_string(index=False))
    if '--holdout' in sys.argv:
        subprocess.run(['/home/user/bdv-lab/venv/bin/python', F.SHARED + 'score.py'] + files, check=True)
