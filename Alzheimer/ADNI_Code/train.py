import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from cohort import build, SETS

H = 183

def probability(model, X):
    return np.array([1-f(H) for f in model.predict_survival_function(X)])

def scores(ytrain,ytest,p):
    auc = cumulative_dynamic_auc(ytrain,ytest,p,[H])[0][0]
    brier = brier_score(ytrain,ytest,(1-p)[:,None],[H])[1][0]
    return auc,brier

def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    data,audit = build(source)
    audit.to_csv(output/"landmarks.csv",index=False)
    if data.empty:
        raise ValueError("No eligible landmarks")
    data.to_csv(output/"cohort.csv",index=False)
    data = data[data.days.gt(0)].dropna(subset=SETS["M3"]).reset_index(drop=True)
    strata = (data.event.eq(1)&data.days.le(H)).astype(int)
    tr,te = train_test_split(np.arange(len(data)),stratify=strata,random_state=42,test_size=.25)
    data.assign(split=np.where(data.index.isin(te),"test","development")).to_csv(output/"split.csv",index=False)
    y = Surv.from_arrays(data.event.astype(bool),data.days)
    pred = data.iloc[te][["rid","days","event"]].copy()
    metrics,models = [],{}
    folds = list(StratifiedKFold(4,shuffle=True,random_state=42).split(tr,strata.iloc[tr]))
    for name,cols in SETS.items():
        X = data[cols]
        def fit(ids,alpha):
            model = make_pipeline(StandardScaler(),CoxPHSurvivalAnalysis(alpha=alpha))
            return model.fit(X.iloc[ids],y[ids])
        losses = []
        for alpha in (0.1,1.,10.):
            values = []
            for a,b in folds:
                model = fit(tr[a],alpha)
                p = probability(model,X.iloc[tr[b]])
                values.append(brier_score(y[tr[a]],y[tr[b]],(1-p)[:,None],[H])[1][0])
            losses.append((np.mean(values),-alpha))
        alpha = -min(losses)[1]
        models[name] = fit(tr,alpha)
        pred[name] = probability(models[name],X.iloc[te])
        metrics.append([name,alpha,*scores(y[tr],y[te],pred[name].to_numpy())])
    pd.DataFrame(metrics,columns=["model","alpha","auc183","brier183"]).to_csv(output/"metrics.csv",index=False)
    pred.to_csv(output/"predictions.csv",index=False)
    joblib.dump(models,output/"models.joblib")
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(500):
        idx = rng.integers(len(te),size=len(te))
        try:
            pair = [scores(y[tr],y[te][idx],pred[n].to_numpy()[idx])[0] for n in ["M2","M3"]]
            differences.append(pair[1]-pair[0])
        except ValueError:
            differences.append(np.nan)
    pd.Series(differences,name="delta_auc_M3_M2").to_csv(output/"bootstrap.csv",index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    run(args.source,args.output)
