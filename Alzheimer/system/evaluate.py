"""Read held-out tables only for evaluation. Never refit on these records."""
import argparse
import json
from sklearn.metrics import roc_auc_score
from pipeline import ROOT, Path, pd, np, cohort, predict

HORIZONS = [183,365,548,730]
def metrics(model, test, train, h):
    t,e = train.days.to_numpy(),train.event.to_numpy()
    cut = np.unique(t[(e==0)&(t<=h)])
    surv = np.cumprod([1-((t==u)&(e==0)).sum()/(t>=u).sum() for u in cut])
    def G(value,side):
        i=np.searchsorted(cut,value,side=side)
        return np.r_[1.,surv][i]
    score = predict(model,test,h)
    t,e = test.days.to_numpy(),test.event.to_numpy()
    cases=(e==1)&(t<=h)
    controls=t>h
    known=cases|controls
    info=dict(horizon=h,n=len(t),cases=int(cases.sum()),controls=int(controls.sum()),
              censored_early=int((~known).sum()))
    if not cases.any() or not controls.any() or G(h,'right')<=0 or h>model['times'][-1]:
        return info
    weights=np.zeros(len(t))
    weights[cases]=1/G(t[cases],'left')
    weights[controls]=1/G(h,'right')
    info.update(auc_ipcw=float(roc_auc_score(cases[known],score[known],sample_weight=weights[known])),
                brier_ipcw=float((weights*(cases-score)**2).mean()),
                mean_prediction=float(score.mean()),observed_ipcw=float(weights[cases].sum()/len(t)),
                tp=int(((score>=.5)&cases).sum()),fn=int(((score<.5)&cases).sum()),
                tn=int(((score<.5)&controls).sum()),fp=int(((score>=.5)&controls).sum()))
    return info

def evaluate(testdir):
    test,_=cohort(testdir)
    train=pd.read_csv(ROOT/'runs/train_landmarks.csv')
    raw_train=set()
    from pipeline import read
    for name in ['MMSE','PTDEMOG','DXSUM','FAQ','UCSFFSX7']:
        raw_train.update(read(ROOT/'data/train',name).RID)
    for name in ['MMSE','PTDEMOG','DXSUM','FAQ','UCSFFSX7']:
        if raw_train.intersection(read(testdir,name).RID):raise ValueError('RID overlap')
    results=[]
    paired=[]
    models=json.loads((ROOT/"runs/models.json").read_text())
    secondary=json.loads((ROOT/"runs/secondary_models.json").read_text())
    for name,entry in models.items():
        model=entry['model']
        if model is None:continue
        a=train[train.days>0].dropna(subset=model['columns'])
        b=test[test.days>0].dropna(subset=model['columns'])
        if len(b):
            for h in HORIZONS: results.append(dict(model=name,**metrics(secondary[name]["model"],b,a,h)))
        common=b.dropna(subset=list(json.loads((ROOT/'runs/models.json').read_text())['M3']['model']['columns']))
        if len(common):
            for h in HORIZONS: paired.append(dict(model=name,**metrics(model,common,a.dropna(subset=models["M3"]["model"]["columns"]),h)))
    from validation import differences
    if len(common): differences(a.dropna(subset=models["M3"]["model"]["columns"]),common,models).to_csv(ROOT/"runs/deltas.csv",index=False)
    pd.DataFrame(paired).to_csv(ROOT/'runs/paired.csv',index=False)
    df=pd.DataFrame(results)
    df.to_csv(ROOT/'runs/evaluation.csv',index=False)
    return df
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('test_directory',type=Path)
    print(evaluate(p.parse_args().test_directory).to_string(index=False))
