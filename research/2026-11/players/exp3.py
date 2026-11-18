"""DEV exploration round 3 (run after features include Série B): training pool, extra features, two-stage unknown model."""
import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
D = E.D; K = E.K_AVAIL; U = E.U_AVAIL
d = E.d


def ol(name, spec, avail=K, l2=5.0, **kw):
    P, coefs = lab.walk_forward(D, spec, avail, l2=l2, **kw)
    lab.score(D, P, 'OLOG ' + name, sub=avail)
    print('      ', {k: float(v) for k, v in coefs[max(coefs)].items()})
    return P


def pois(name, feats, avail=K, l2=5.0, **kw):
    P, coefs, lh, la = lab.walk_forward_poisson(D, feats, avail, l2=l2, **kw)
    lab.score(D, P, 'POIS ' + name, sub=avail)
    print('      ', {k: float(v) for k, v in coefs[max(coefs)].items()})
    return P


if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'pool':
        for lg in (('A',), ('A', 'B')):
            ol(f'usage_in+ga_in+exp train={lg}', d('k_usage_in', 'k_gashare_in', 'k_exp'), train_lg=lg)
            ol(f'usage_in train={lg}', d('k_usage_in'), train_lg=lg)
            ol(f'usage_in+ga_in+expA train={lg}', d('k_usage_in', 'k_gashare_in', 'k_expA'), train_lg=lg)
            pois(f'usage_in+ga_in+exp train={lg}', ['k_usage_in', 'k_gashare_in', 'k_exp'], train_lg=lg)
    if what == 'feat':
        base = ['k_usage_in', 'k_gashare_in', 'k_exp']
        ol('base3', d(*base))
        for extra in ('k_expA', 'k_exp_club', 'k_exp_min', 'k_bench_usage', 'k_squad_usage', 'k_shshare_in', 'k_top', 'k_gk', 'k_rapm1x_rel', 'k_rapm1g_rel',
                      'k_rapm1x', 'k_ped', 'k_ga90', 'k_ga90_rel', 'k_sh90', 'k_debut', 'k_new', 'k_chg', 'k_usage_early', 'k_usage_xl', 'k_newclub',
                      'u_usage10', 'u_susp', 'u_absent', 'k_ret_early', 'k_retxi_early', 'k_lowusage_in', 'k_miss_D', 'k_miss_G'):
            ol('base3 + ' + extra, d(*base, extra))
    if what == 'single':
        for f in ('k_usage_in', 'k_usage_s', 'k_gashare_in', 'k_shshare_in', 'k_exp', 'k_expA', 'k_exp_club', 'k_bench_usage', 'k_squad_usage', 'k_rapm1x_rel', 'k_rapm1x', 'k_ped', 'k_ga90'):
            ol(f, d(f))
        for f in ('u_usage10', 'u_usage3', 'u_susp', 'u_absent', 'u_exp', 'u_expA', 'u_rapm1x', 'u_ped', 'u_ga90', 'u_ret_early', 'u_early_off'):
            ol(f, d(f), avail=U)
