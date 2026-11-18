import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL
def pois(name, feats, avail=K, l2=5.0, **kw):
    P,coefs,lh,la=lab.walk_forward_poisson(D, feats, avail, l2=l2, **kw)
    lab.score(D,P,'POIS '+name, sub=avail)
    print('      ', coefs[max(coefs)])
    return P
def ol(name, spec, avail=K, l2=5.0, **kw):
    P,coefs=lab.walk_forward(D, spec, avail, l2=l2, **kw)
    lab.score(D,P,'OLOG '+name, sub=avail)
    print('      ', coefs[max(coefs)])
    return P
if __name__=='__main__':
    pois('usage_in',['k_usage_in'])
    pois('usage_s',['k_usage_s'])
    pois('gashare',['k_gashare'])
    pois('exp',['k_exp'])
    pois('usage_in+ga',['k_usage_in','k_gashare'])
    pois('usage_in+ga+exp',['k_usage_in','k_gashare','k_exp'])
    pois('miss_pos',['k_miss_G','k_miss_D','k_miss_M','k_miss_F'])
    ol('usage_in+ga',E.d('k_usage_in','k_gashare'))
    ol('usage_in+ga+exp',E.d('k_usage_in','k_gashare','k_exp'))
    ol('usage_in+ga_in+exp',E.d('k_usage_in','k_gashare_in','k_exp'))
    ol('usage_in+ga+exp l2=20',E.d('k_usage_in','k_gashare','k_exp'),l2=20)
    ol('usage_in+ga+exp l2=1',E.d('k_usage_in','k_gashare','k_exp'),l2=1)
    ol('usage_in + u_usage10',E.d('k_usage_in','u_usage10'))
    ol('usage_s+ga+exp',E.d('k_usage_s','k_gashare','k_exp'))
