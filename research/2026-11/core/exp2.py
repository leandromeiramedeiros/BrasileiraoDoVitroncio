"""Engine experiments, batch 2: factor channels (shots etc. informing the common strength)."""
import sys, numpy as np, lib, eng, wf, obsfn
d = lib.load()
G15 = ['qm', 'P0', 'rho', 'sS', 'dPO', 'dPD', 'dNew', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG']
ch = lambda c: [f'{c}:{k}' for k in ('beta', 'P0', 'q', 'rho', 'phi')]
pure = dict(w=1.0, wRed=1.0)
specs = {
    'f_sot': wf.Spec('f_sot', G15 + ch('sot'), pure, obs=obsfn.obs_v1, sec=['sot']),
    'f_sh': wf.Spec('f_sh', G15 + ch('sh'), pure, obs=obsfn.obs_v1, sec=['sh']),
    'f_sot_soff': wf.Spec('f_sot_soff', G15 + ch('sot') + ch('soff'), pure, obs=obsfn.obs_v1, sec=['sot', 'soff']),
    'f_sv_soff': wf.Spec('f_sv_soff', G15 + ch('sv') + ch('soff'), pure, obs=obsfn.obs_v1, sec=['sv', 'soff']),
    'f_sot_soff_co': wf.Spec('f_sot_soff_co', G15 + ch('sot') + ch('soff') + ch('co'), pure, obs=obsfn.obs_v1, sec=['sot', 'soff', 'co']),
    'f_sot_sh': wf.Spec('f_sot_sh', G15 + ch('sot') + ch('sh'), pure, obs=obsfn.obs_v1, sec=['sot', 'sh']),
}
for k in sys.argv[1:]:
    wf.walk(d, specs[k], first_iter=(3, 700), next_iter=(2, 300))
