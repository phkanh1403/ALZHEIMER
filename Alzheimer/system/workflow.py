import hashlib
import os
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from sqlalchemy import select
from cohort import SETS
from storage import history, entries,engine,record
from train import probability

folder=Path(os.getenv("ADNI_MODEL_DIR","runs/demo"))

@st.cache_resource
def load_models(path,modified):
    return joblib.load(path),hashlib.sha256(path.read_bytes()).hexdigest()

def run_score(row):
    path=folder/"branches.joblib"
    if not path.is_file():
        raise ValueError(f"Chưa có {path}. Chạy branches.py theo README.")
    models,digest=load_models(path,path.stat().st_mtime_ns)
    for name in ("M3","M2","M1","MC","M0"):
        x=row[SETS[name]]
        if name in models and x.notna().all(axis=None):
            return dict(model=name,features=x.iloc[0].to_dict(),probability=float(probability(models[name],x)[0]),
                        model_sha256=digest,source=str(folder))
    return None

def select_patient():
    with engine.connect() as c:
        ids=c.scalars(select(entries.c.patient).distinct().order_by(entries.c.patient)).all()
    return st.selectbox("Chọn hồ sơ",ids) if ids else None

def ingest(dest,batch):
    visits=pd.read_csv(dest/"visits.csv",dtype={"rid":str})
    features=dest/"features_m06.csv"
    rows=pd.read_csv(features,dtype={"rid":str}).set_index("rid") if features.exists() else pd.DataFrame()
    scored=0
    for rid,group in visits.groupby("rid"):
        patient="ADNI-"+rid
        existing={json.dumps(r["payload"],sort_keys=True,ensure_ascii=False) for r in history(patient)}
        def save(kind,p):
            if json.dumps(p,sort_keys=True,ensure_ascii=False) not in existing:
                record(patient,kind,p)
        for v in json.loads(group.to_json(orient="records")):
            save("visit",dict(v,normalized_dir=str(dest.resolve()),batch=batch,mmse=None,faq=None,hippo=None,entor=None))
        if rid in rows.index:
            result=run_score(rows.loc[[rid]])
            if result:
                save("prediction",dict(result,t0=str(rows.loc[rid,"t0"])[:10],horizon=183,batch=batch))
                scored+=1
    return visits.rid.nunique(),scored
