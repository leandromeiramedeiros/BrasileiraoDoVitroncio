"""Engine v3 experiments (clean parameterisation: phi = 1, variance reset to P0 each season).
usage: exp4.py name [name ...]   -- each spec may declare a parent whose per-season optimum is used as a warm start.
"""
import sys, pickle, datetime as dt
import numpy as np, lib, core3
d = lib.load()
CHR = core3.channel_preds(d)
B = ['qm', 'P0', 'rho', 'dPO', 'dPD', 'kMu', 'kH', 'Pg0', 'qg', 'rhoG']
day0 = dt.date(2006, 1, 1)
covid = (d.day >= (dt.date(2020, 3, 15) - day0).days) & (d.day < (dt.date(2021, 10, 1) - day0).days)
S = {}
def add(name, free, fixed=None, parent=None, **kw):
    S[name] = (core3.Spec(name, free, fixed, **kw), parent)
add('B0', B)
add('B0c', B + ['cod'], dict(cod=0.3), 'B0')
add('B0w', B + ['wLead2', 'wTrail2'], dict(wLead2=0.7, wTrail2=0.7), 'B0')
add('B0cw', B + ['cod', 'wLead2', 'wTrail2'], dict(cod=0.3, wLead2=0.7, wTrail2=0.7), 'B0c')
add('A1', B + ['c_sot'], dict(c_sot=0.2), 'B0')
add('A1c', B + ['cod', 'c_sot'], dict(cod=0.3, c_sot=0.2), 'B0c')
add('A2c', B + ['cod', 'c_sot', 'c_sh'], dict(cod=0.3, c_sot=0.2, c_sh=0.1), 'A1c')
add('A3c', B + ['cod', 'c_sot', 'c_sh', 'c_co'], dict(cod=0.3, c_sot=0.2, c_sh=0.1, c_co=0.05), 'A2c')
add('A1cw', B + ['cod', 'c_sot', 'wLead2', 'wTrail2'], dict(cod=0.3, c_sot=0.2, wLead2=0.7, wTrail2=0.7), 'A1c')
add('A2cw', B + ['cod', 'c_sot', 'c_sh', 'wLead2', 'wTrail2'], dict(cod=0.3, c_sot=0.2, c_sh=0.1, wLead2=0.7, wTrail2=0.7), 'A2c')
# ---- single-factor variations around the main candidate M = A2cw (filled in after batch 1)
M = B + ['cod', 'c_sot', 'c_sh', 'wLead2', 'wTrail2']
MF = dict(cod=0.3, c_sot=0.2, c_sh=0.1, wLead2=0.7, wTrail2=0.7)
add('M_qd', [k if k != 'qm' else 'qd' for k in M], dict(MF, qm=0.0, qd=2e-5), 'A2cw')
add('M_nocap', M + ['sS'], dict(MF, cap=0.0, sS=0.005), 'A2cw')
add('M_qd_nocap', [k if k != 'qm' else 'qd' for k in M] + ['sS'], dict(MF, qm=0.0, qd=2e-5, cap=0.0, sS=0.005), 'A2cw')
add('M_promo', M + ['kappa', 'Ppromo'], dict(MF, Ppromo=0.003), 'A2cw')
add('M_covid', M, dict(MF, kCov=0.02), 'A2cw', covid=covid)
add('M_nohome', [k for k in M if k not in ('Pg0', 'qg', 'rhoG')], dict(MF, Pg0=1e-12, qg=0.0), None)
add('M_wred', M + ['wRed'], dict(MF, wRed=0.8), 'A2cw')
add('M_wpen', M + ['wPen', 'wOwn'], dict(MF, wPen=0.8, wOwn=0.8), 'A2cw')
add('M_expo', M, dict(MF, expo=1.0), 'A2cw')
add('M_late', M + ['wLead', 'wTrail'], dict(MF, wLead=0.8, wTrail=0.8), 'A2cw')
add('M_phi', M + ['phi'], dict(MF), 'A2cw')
add('M_co', M + ['c_co'], dict(MF, c_co=0.05), 'A2cw')
# ---- tuning-noise replicates of M and regularised tuning
add('M_rep1', M, MF, None)                      # no warm start from the parent chain
add('M_rep2', M, dict(MF, cod=0.6, c_sot=0.6, c_sh=0.6, wLead2=0.8, wTrail2=0.95, P0=0.011, qm=1e-5), None)   # different start
add('M_rw3', M, MF, None)                       # random-walk prior on hyper-parameters, gamma = 1e-3
add('M_rw4', M, MF, None)                       # gamma = 1e-4
M9 = ['P0', 'rho', 'dPO', 'dPD', 'Pg0', 'cod', 'c_sot', 'c_sh', 'wLead2']
add('M9', M9, dict(MF, qm=0.0, kMu=0.002, kH=0.0005, qg=0.0005, rhoG=0.2, wTrail2=1.0), None)
RW = {'M_rw3': 1e-3, 'M_rw4': 1e-4}
add('M_ll', M, MF, 'A2cw')
add('M_mix', M, MF, 'A2cw')
OBJ = {'M_ll': 'll', 'M_mix': 'mix'}
if __name__ == '__main__':
    for k in sys.argv[1:]:
        spec, parent = S[k]
        init = None
        if parent:
            try: init = pickle.load(open(lib.D + f'out_{parent}.pkl', 'rb'))['PAR']
            except Exception as e: print('no parent result', parent, e)
        core3.walk(d, spec, CHR, init=init, rw=RW.get(k, 0.0), objective=OBJ.get(k, 'pois'), steps=(np.full(len(spec.free), 0.2) if k == 'M_rep2' else None))
