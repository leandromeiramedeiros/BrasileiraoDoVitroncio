"""Engine experiments, batch 3: variations on top of F1 = goals + shots-on-target factor channel."""
import sys, numpy as np, lib, eng, wf, obsfn
d = lib.load()
G = ['qm', 'P0', 'rho', 'sS', 'dPO', 'dPD', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG']
ch = lambda c: [f'{c}:{k}' for k in ('beta', 'P0', 'q', 'rho', 'phi')]
F1 = G + ch('sot')
base = dict(dNew=0.0)
# a-priori artificial-turf home venues (club id, first date): Athletico-PR 2016, Palmeiras 2020, Botafogo 2023, Atletico-MG 2025
import datetime as dt
day0 = dt.date(2006, 1, 1)
tid = {t: i for i, t in enumerate(d.team_ids)}
turf = np.zeros(d.n, np.bool_)
for club, since in (('3458', dt.date(2016, 3, 1)), ('2029', dt.date(2020, 2, 1)), ('6086', dt.date(2023, 4, 1)), ('7632', dt.date(2025, 5, 1))):
    turf |= (d.h == tid[club]) & (d.day >= (since - day0).days)
# empty-stadium period: 2020 season restart (Aug 2020) until crowds returned (Oct 2021)
covid = (d.day >= (dt.date(2020, 3, 15) - day0).days) & (d.day < (dt.date(2021, 10, 1) - day0).days)
specs = {
    'F1': wf.Spec('F1', F1, base, obs=obsfn.obs_adj, sec=['sot']),
    'F1_nocap': wf.Spec('F1_nocap', F1, dict(base, cap=0.0, sS=0.01), obs=obsfn.obs_adj, sec=['sot']),
    'F1_qd': wf.Spec('F1_qd', [k if k != 'qm' else 'qd' for k in F1], dict(base, qm=0.0, qd=2e-5), obs=obsfn.obs_adj, sec=['sot']),
    'F1_qd_nocap': wf.Spec('F1_qd_nocap', [k if k != 'qm' else 'qd' for k in F1], dict(base, qm=0.0, qd=2e-5, cap=0.0, sS=0.01), obs=obsfn.obs_adj, sec=['sot']),
    'F1_promo': wf.Spec('F1_promo', F1 + ['kappa', 'Ppromo'], dict(base, Ppromo=0.005), obs=obsfn.obs_adj, sec=['sot']),
    'F1_rhoB': wf.Spec('F1_rhoB', F1 + ['rhoB'], dict(base, rhoB=0.85), obs=obsfn.obs_adj, sec=['sot']),
    'F1_away': wf.Spec('F1_away', F1 + ['Pa0', 'qa', 'rhoA'], dict(base, Pa0=0.01, qa=0.001, rhoA=0.2), obs=obsfn.obs_adj, sec=['sot']),
    'F1_turf': wf.Spec('F1_turf', F1 + ['kTurf'], dict(base, kTurf=0.01), obs=obsfn.obs_adj, sec=['sot'], turf=turf),
    'F1_covid': wf.Spec('F1_covid', F1, dict(base, kCov=0.02), obs=obsfn.obs_adj, sec=['sot'], covid=covid),
    'F1_shock': wf.Spec('F1_shock', F1 + ['shG', 'shT'], dict(base, shG=0.001, shT=2.0), obs=obsfn.obs_adj, sec=['sot']),
    'F1_wred': wf.Spec('F1_wred', F1 + ['wRed'], dict(base, wRed=0.8), obs=obsfn.obs_adj, sec=['sot']),
    'F1_expo': wf.Spec('F1_expo', F1, dict(base, expo=1.0), obs=obsfn.obs_adj, sec=['sot']),
    'F1_wpen': wf.Spec('F1_wpen', F1 + ['wPen', 'wOwn'], dict(base, wPen=0.8, wOwn=0.8), obs=obsfn.obs_adj, sec=['sot']),
    'F1_wlate': wf.Spec('F1_wlate', F1 + ['wLead', 'wTrail'], dict(base, wLead=0.8, wTrail=0.8), obs=obsfn.obs_adj, sec=['sot']),
    'F1_nohome': wf.Spec('F1_nohome', [k for k in F1 if k not in ('Pg0', 'qg', 'rhoG')], dict(base, Pg0=1e-12, qg=0.0), obs=obsfn.obs_adj, sec=['sot']),
    'F1_kH': wf.Spec('F1_kH', [k for k in F1 if k != 'kH'], dict(base, kH=0.004), obs=obsfn.obs_adj, sec=['sot']),
}
if __name__ == '__main__':
    for k in sys.argv[1:]:
        wf.walk(d, specs[k], first_iter=(3, 700), next_iter=(2, 300))
