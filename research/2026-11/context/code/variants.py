"""Feature specifications (building blocks) for the context-track variants."""
import numpy as np
c = lambda name: (lambda d: d[name].to_numpy(float))
p = lambda a, b: (lambda d: (d[a].to_numpy(float), d[b].to_numpy(float)))
def pair_early(a, b, n=10):   # rating delta x early-season weight (1 at round 1 -> 0 from round n+1)
    return lambda d: (d[a].to_numpy(float) * np.clip(1 - d.gp_h.to_numpy(float) / n, 0, 1), d[b].to_numpy(float) * np.clip(1 - d.gp_a.to_numpy(float) / n, 0, 1))
def late(d): return np.clip((np.maximum(d.gp_h, d.gp_a).to_numpy(float) - 27) / 10, 0, 1)   # 0 until round 28, 1 at round 38
def short(col): return lambda d: (np.clip(5 - d[col + '_h'].to_numpy(float), 0, 3), np.clip(5 - d[col + '_a'].to_numpy(float), 0, 3))
def latex(col): return lambda d: (d[col + '_h'].to_numpy(float) * late(d), d[col + '_a'].to_numpy(float) * late(d))
def one_minus_late(col): return lambda d: ((1 - d[col + '_h'].to_numpy(float)) * late(d), (1 - d[col + '_a'].to_numpy(float)) * late(d))
ldist = lambda d: np.log1p(d.dist_km.to_numpy(float) / 100.0)

BLOCKS = {
  # ---- (a) extra matches as rating evidence
  'r_all':   [('r_all', 'pair', p('r_dh_all', 'r_da_all'))],
  'r_allw':  [('r_allw', 'pair', p('r_dh_allw', 'r_da_allw'))],
  'r_fam':   [('r_st', 'pair', p('r_dh_st', 'r_da_st')), ('r_cup', 'pair', p('r_dh_cup', 'r_da_cup')), ('r_cont', 'pair', p('r_dh_cont', 'r_da_cont'))],
  'r_cupcont': [('r_cup', 'pair', p('r_dh_cup', 'r_da_cup')), ('r_cont', 'pair', p('r_dh_cont', 'r_da_cont'))],
  'r_fam_early': [('r_st_early', 'pair', pair_early('r_dh_st', 'r_da_st')), ('r_cup_early', 'pair', pair_early('r_dh_cup', 'r_da_cup')), ('r_cont_early', 'pair', pair_early('r_dh_cont', 'r_da_cont'))],
  'r_allw_early': [('r_allw_early', 'pair', pair_early('r_dh_allw', 'r_da_allw'))],
  # ---- (b) schedule
  'rest':    [('rest_short', 'diff', short('rest'))],
  'cong':    [('n14', 'diff', p('n14_h', 'n14_a')), ('n30', 'diff', p('n30_h', 'n30_a'))],
  'cupprox': [('next_big4', 'diff', p('next_big4_h', 'next_big4_a')), ('prev_big4', 'diff', p('prev_big4_h', 'prev_big4_a'))],
  'cupprox2': [('next_nonlg4', 'diff', p('next_nonlg4_h', 'next_nonlg4_a')), ('prev_nonlg4', 'diff', p('prev_nonlg4_h', 'prev_nonlg4_a'))],
  'contprox': [('next_cont4', 'diff', p('next_cont4_h', 'next_cont4_a')), ('prev_cont4', 'diff', p('prev_cont4_h', 'prev_cont4_a')), ('prev_abroad5', 'diff', p('prev_abroad5_h', 'prev_abroad5_a'))],
  'alive':   [('alive_cdb', 'diff', p('alive_cdb_h', 'alive_cdb_a')), ('alive_cont', 'diff', p('alive_cont_h', 'alive_cont_a'))],
  'travel':  [('trav_prev', 'diff', lambda d: (np.log1p(d.trav_prev_h.to_numpy(float) / 100), np.log1p(d.trav_prev_a.to_numpy(float) / 100))),
              ('km14', 'diff', lambda d: (d.km14_h.to_numpy(float) / 1000, d.km14_a.to_numpy(float) / 1000))],
  # ---- geography / home-advantage modifiers
  'dist':    [('ldist', 'hfa', ldist)],
  'geo':     [('ldist', 'hfa', ldist), ('same_city', 'hfa', c('same_city')), ('home_disp', 'hfa', lambda d: (d.home_disp_km.to_numpy(float) > 100).astype(float))],
  'climate': [('north_trip', 'hfa', lambda d: np.clip(d.lat_diff.to_numpy(float), 0, None) / 10), ('south_trip', 'hfa', lambda d: np.clip(-d.lat_diff.to_numpy(float), 0, None) / 10),
              ('lon_abs', 'hfa', lambda d: np.abs(d.lon_diff.to_numpy(float)) / 10)],
  'derby':   [('same_city', 'hfa', c('same_city')), ('same_state', 'hfa', c('same_state'))],
  # ---- stakes / season context
  'covid':   [('closed', 'hfa', c('closed'))],
  'late':    [('late_hfa', 'hfa', late), ('late_tot', 'tot', late)],
  'stakes':  [('press_rel_late', 'diff', latex('press_rel')), ('press_top_late', 'diff', latex('press_top')), ('press_title_late', 'diff', latex('press_title'))],
  'dead':    [('dead', 'diff', p('dead_h', 'dead_a'))],
  'dead2':   [('dead', 'diff', p('dead_h', 'dead_a')), ('releg', 'diff', p('releg_h', 'releg_a')), ('safe', 'diff', p('safe_h', 'safe_a'))],
}
def spec(*names):
    out = []
    for n in names: out += BLOCKS[n]
    return out

# ---- refined blocks chosen after the first DEV pass (see REPORT.md)
def decided(d): return (d.releg_h.to_numpy(float) + d.clinched_h.to_numpy(float), d.releg_a.to_numpy(float) + d.clinched_a.to_numpy(float))
BLOCKS.update({
  'decided':  [('decided', 'diff', decided)],
  'decided2': [('releg', 'diff', p('releg_h', 'releg_a')), ('clinched', 'diff', p('clinched_h', 'clinched_a'))],
  'nextcont': [('next_cont4', 'diff', p('next_cont4_h', 'next_cont4_a'))],
  'alivecont': [('alive_cont', 'diff', p('alive_cont_h', 'alive_cont_a'))],
  'homedisp': [('home_disp', 'hfa', lambda d: (d.home_disp_km.to_numpy(float) > 100).astype(float))],
  'relfight': [('press_rel_late', 'diff', latex('press_rel'))],
  'samecity': [('same_city', 'hfa', c('same_city'))],
})

for _k in ('opt', 'st_reset', 'opt_reset'):
    BLOCKS['r_' + _k] = [('r_' + _k, 'pair', p(f'r_dh_{_k}', f'r_da_{_k}'))]
    BLOCKS['r_' + _k + '_early'] = [('r_' + _k + '_early', 'pair', pair_early(f'r_dh_{_k}', f'r_da_{_k}'))]
BLOCKS['same_state'] = [('same_state', 'hfa', c('same_state'))]
GROUPS = {
  'sched_all':  ['rest', 'cong', 'cupprox', 'contprox', 'alive', 'travel'],
  'geo_all':    ['geo', 'climate', 'same_state'],
  'stakes_all': ['late', 'stakes', 'dead2'],
  'sink':       ['rest', 'cong', 'cupprox', 'contprox', 'alive', 'travel', 'geo', 'climate', 'same_state', 'late', 'stakes', 'dead2', 'covid', 'r_fam'],
  'ctx_core':   ['decided', 'nextcont', 'covid', 'homedisp'],
  'ctx_core_r': ['decided', 'nextcont', 'covid', 'homedisp', 'r_all'],
}

def decided_lag(d): return (d.releg_lag4_h.to_numpy(float) + d.clinched_lag4_h.to_numpy(float), d.releg_lag4_a.to_numpy(float) + d.clinched_lag4_a.to_numpy(float))
BLOCKS['decided_lag4'] = [('decided', 'diff', decided_lag)]
GROUPS['ctx_core_lag4'] = ['decided_lag4', 'nextcont', 'covid', 'homedisp']
