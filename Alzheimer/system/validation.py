"""Five-fold out-of-fold assessment; test data are never loaded here."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from pipeline import ROOT, MODELS, fit, predict
from evaluate import HORIZONS

def weights(train, test, h):
    t=train.days.to_numpy()
    cens=train.event.to_numpy()==0
    cut,count=np.unique(t[cens & (t<=h)],return_counts=True)
    surv=np.r_[1.,np.cumprod(1-count/(len(t)-np.searchsorted(np.sort(t),cut)))]
    event=(test.event.eq(1)&test.days.le(h)).to_numpy()
    control=test.days.gt(h).to_numpy()
    w=np.zeros(len(test))
    w[event]=1/surv[np.searchsorted(cut,test.days[event])]
    w[control]=1/surv[-1]
    return event,w

def summary(y,w,p):
    known=w>0
    auc=roc_auc_score(y[known],p[known],sample_weight=w[known]) if len(np.unique(y[known]))==2 else np.nan
    return np.array([auc,np.mean(w*(y-p)**2)])

def validate():
    out=ROOT/'runs'
    df=pd.read_csv(out/'train_landmarks.csv').query('days>0').dropna(subset=MODELS['M3']).copy()
    assert df.RID.is_unique
    strata=np.where(df.event.eq(0),-1,np.searchsorted(HORIZONS,df.days))
    for fold,(_,idx) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(df,strata)):
        df.loc[df.index[idx],'fold']=fold
    age=(pd.to_datetime(df.t0)-pd.to_datetime(df.hippo_date)).dt.days
    coverage=[]; chunks=[]
    for window in (180,365):
        group=df[age<=window]
        coverage.append([window,len(group),group.event.sum(),age.loc[group.index].median()])
        for fold in range(5):
            a=group[group.fold!=fold]; b=group[group.fold==fold]
            for name,cols in MODELS.items():
                m=fit(a,cols)
                for h in HORIZONS:
                    y,w=weights(a,b,h)
                    chunks.append(pd.DataFrame(dict(RID=b.RID,fold=fold,window=window,model=name,h=h,y=y,w=w,p=predict(m,b,h))))
    long=pd.concat(chunks,ignore_index=True)
    long.to_csv(out/'validation_oof.csv',index=False)
    rows=[]
    for scope,view in [('eligible',long),('shared',long[long.RID.isin(df.loc[age<=180,'RID'])])]:
        for k,g in view.groupby(['window','model','h']):
            auc,bs=summary(g.y.to_numpy(),g.w.to_numpy(),g.p.to_numpy())
            rows.append([scope,*k,len(g),auc,bs,g.p.mean(),(g.y*g.w).mean()])
    pd.DataFrame(rows,columns=['scope','window','model','h','n','AUC','Brier','predicted','observed_IPCW']).to_csv(out/'validation.csv',index=False)
    pd.DataFrame(coverage,columns=['window','n','events','median_age_days']).to_csv(out/'windows.csv',index=False)

def differences(train,test,models):
    rng=np.random.default_rng(42)
    rows=[]
    for h in HORIZONS:
        y,w=weights(train,test,h)
        p=[predict(models[n]['model'],test,h) for n in ['M2','M3']]
        def delta(i):return summary(y[i],w[i],p[1][i])-summary(y[i],w[i],p[0][i])
        estimate=delta(np.arange(len(y)))
        draws=np.array([delta(rng.integers(len(y),size=len(y))) for _ in range(1000)])
        for j,key in enumerate(['AUC','Brier']):
            valid=draws[:,j][np.isfinite(draws[:,j])]
            limits=np.quantile(valid,[.025,.975]) if len(valid) else [np.nan,np.nan]
            rows.append([h,key,estimate[j],*limits,len(valid)])
    return pd.DataFrame(rows,columns=['days','metric','M3_minus_M2','low95','high95','valid_replicates'])
if __name__=='__main__':validate()
