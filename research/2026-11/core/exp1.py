"""Engine experiments, batch 1: baseline goal channel and simple ablations."""
import sys, numpy as np, lib, eng, wf, obsfn
d = lib.load()
G15 = ['qm', 'P0', 'rho', 'sS', 'dPO', 'dPD', 'dNew', 'phi', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG']
xgaux = lambda d, Y: obsfn.fit_xg_linear(d, Y)
which = sys.argv[1:]
specs = {
    'g_v1': wf.Spec('g_v1', G15 + ['w', 'wRed'], dict(w=0.6, wRed=0.8), obs=obsfn.obs_v1),
    'g_pure': wf.Spec('g_pure', G15, dict(w=1.0, wRed=1.0), obs=obsfn.obs_v1),
    'g_nohome': wf.Spec('g_nohome', [k for k in G15 if k not in ('Pg0', 'qg', 'rhoG')] + ['w', 'wRed'], dict(w=0.6, wRed=0.8, Pg0=1e-12, qg=0.0), obs=obsfn.obs_v1),
}
for k in which:
    wf.walk(d, specs[k], aux_fn=xgaux)
