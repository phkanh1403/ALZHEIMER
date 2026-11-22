"""Evaluate missing-data branches on a shared person-level holdout."""
import sys
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from cohort import build,SETS
from train import H,probability,scores


def run(source,out):
    df,_=build(source)
    df=df[df.days.gt(0)].copy()
    train,test=train_test_split(df,test_size=.25,random_state=42,
        stratify=df.event.eq(1)&df.days.le(H))
    out.mkdir(parents=True,exist_ok=True)
    models={}
    report=[]
    for n,cols in SETS.items():
        a=train.dropna(subset=cols); b=test.dropna(subset=cols)
        item=dict(model=n,development=len(a),eligible_test=len(b),total_test=len(test))
        try:
            y=Surv.from_arrays(a.event.astype(bool),a.days)
            model=make_pipeline(StandardScaler(),CoxPHSurvivalAnalysis(alpha=1)).fit(a[cols],y)
            models[n]=model
            if not len(b):
                raise ValueError("Thiếu hồ sơ test")
            p=probability(model,b[cols])
            item["auc183"],item["brier183"]=scores(y,Surv.from_arrays(b.event.astype(bool),b.days),p)
        except (ValueError,ArithmeticError) as exc:
            item["reason"]=str(exc)
        report.append(item)
    pd.DataFrame(report).to_csv(out/"coverage.csv",index=False)
    pd.Series([next((n for n in ("M3","M2","M1","MC","M0") if n in models and
        row[SETS[n]].notna().all()),"unavailable") for _,row in test.iterrows()],
        index=test.rid,name="branch").to_csv(out/"routing.csv")
    joblib.dump(models,out/"branches.joblib")

if __name__=="__main__":
    run(Path(sys.argv[1]),Path(sys.argv[2]))
