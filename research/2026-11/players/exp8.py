import sys, warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import experiments as E, lab
from exp3 import ol, pois
D=E.D; K=E.K_AVAIL; U=E.U_AVAIL; d=E.d
b4=['k_usage_in','k_gashare_in','k_exp','k_bench_usage']
what=sys.argv[1]
if what=='form':
    Po=ol('base4', d(*b4))
    Pp=pois('base4', b4)
    lab.score(D,(Po+Pp)/2,'AVG olog+pois base4', sub=K)
    Po5=ol('base4+sh90', d(*b4,'k_sh90'))
    Pp5=pois('base4+sh90', b4+['k_sh90'])
    lab.score(D,(Po5+Pp5)/2,'AVG olog+pois base4+sh90', sub=K)
    for l2 in (0.5, 20, 80):
        ol(f'base4 l2={l2}', d(*b4), l2=l2)
    ol('base4 first_train=2010', d(*b4), first_train=2010)
    ol('base4 train A only', d(*b4), train_lg=('A',))
    ol('base4 h/a separate', [(f,k) for f in b4 for k in ('h','a')])
if what=='gbm':
    import lightgbm as lgb
    feats=['k_usage','k_usage_s','k_gashare','k_shshare','k_exp','k_bench_usage','k_squad_usage','k_sh90','k_ga90','k_bench_n','k_chg','k_new','k_debut','k_rapm1x','k_bench_gashare','u_usage10','u_susp','n_season']
    X=np.column_stack([D[f+'_h'].to_numpy(float)-D[f+'_a'].to_numpy(float) for f in feats]+[D[f+s].to_numpy(float) for f in ('k_usage','k_squad_usage','k_exp','n_season') for s in ('_h','_a')])
    P0=D[['pH','pD','pA']].to_numpy(float); k=D.k.to_numpy(); season=D.season.to_numpy(); post=D.post.to_numpy()
    for params in (dict(num_leaves=4, learning_rate=0.02, n_estimators=150, min_child_samples=150, colsample_bytree=0.6, subsample=0.8, subsample_freq=1, reg_lambda=20),
                   dict(num_leaves=3, learning_rate=0.02, n_estimators=80, min_child_samples=250, colsample_bytree=0.5, subsample=0.8, subsample_freq=1, reg_lambda=50)):
        Pn=P0.copy()
        for Y in range(2013,2027):
            tr=K&post&(season<Y)&(season>=2008); te=K&(season==Y)
            if te.sum()==0: continue
            m=lgb.LGBMClassifier(objective='multiclass', verbose=-1, random_state=1, n_jobs=2, **params)
            m.fit(X[tr], k[tr], init_score=np.log(P0[tr]))
            raw=m.predict(X[te], raw_score=True)+np.log(P0[te])
            e=np.exp(raw-raw.max(1,keepdims=True)); Pn[te]=e/e.sum(1,keepdims=True)
        lab.score(D,Pn,f'GBM leaves={params["num_leaves"]} n={params["n_estimators"]}', sub=K)
