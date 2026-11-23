from datetime import date
from collections import defaultdict
import numpy as np
import pandas as pd

BASE = ["age","female","education","mmse","faq"]
MRI = ["hippo","entor"]
RATES = ["mmse_rate","faq_rate"]
BRANCHES = {"M3":BASE+MRI+RATES+["hippo_rate","entor_rate"],"M2":BASE+MRI+RATES,
            "M1":BASE+MRI,"MC":BASE+RATES,"M0":BASE}
def days(a,b):
    return (date.fromisoformat(b)-date.fromisoformat(a)).days

def reconcile(rows):
    groups=defaultdict(list)
    issues=[]
    for row in rows:
        groups[(row["rid"],row["date"],row["kind"])].append(row)
    kept=[]
    for key,group in sorted(groups.items()):
        signatures={(str(r["value"]),r["version"],r["uid"],r["available"]) for r in group}
        if len(group)>1:
            issues.append(dict(rid=key[0],date=key[1],kind=key[2],reason="duplicate" if len(signatures)==1 else "conflict",
                               sources=" | ".join(r["source"] for r in group)))
        if len(signatures)==1:
            kept.append(group[0])
        elif key[2]=="dx":
            kept.append(dict(group[0],value="Unknown"))
    return kept,issues

def features(person,t0):
    x={}; evidence={}
    for kind in ["birthyear","age","female","education","mmse","faq",*MRI]:
        series=[r for r in person if r["kind"]==kind and r["available"]<=t0 and r["date"]<=t0]
        if not series: continue
        latest=series[-1]
        if kind not in ("birthyear","female","education") and days(latest["date"],t0)>45: continue
        if kind=="birthyear":
            age=int(t0[:4])-float(latest["value"])
            if 18<=age<=120: x["age"]=age
        else: x[kind]=float(latest["value"])
        evidence[kind]=latest
        if kind in ["mmse","faq",*MRI]:
            for prev in reversed(series[:-1]):
                dt=days(prev["date"],latest["date"])
                if dt>=90 and prev["version"]==latest["version"]:
                    x[kind+"_rate"]=(float(latest["value"])-float(prev["value"]))*365.25/dt
                    evidence[kind+"_previous"]=prev
                    break
    return x,evidence

def landmarks(clean):
    people=defaultdict(list)
    for r in clean: people[r["rid"]].append(r)
    result=[]
    for rid,person in people.items():
        person.sort(key=lambda r:r["date"])
        visits=[r for r in person if r["kind"]=="dx"]
        first=None
        mci=next((v for v in visits if v["value"]=="MCI"),None)
        m06=None
        if mci:
            target=(pd.Timestamp(mci["date"])+pd.DateOffset(months=6)).date().isoformat()
            dated=[v for v in visits if v["date"]!="0001-01-01" and abs(days(v["date"],target))<=30]
            if dated: m06=min(dated,key=lambda v:abs(days(v["date"],target)))["date"]
        for i,v in enumerate(visits):
            if v["value"] in ("Dementia","Unknown"): break
            if v["value"]!="MCI": continue
            t0=v["date"]
            if first is None: first=t0
            end=t0; event=0
            for f in visits[i+1:]:
                if f["value"]=="Unknown": break
                end=f["date"]
                if f["value"]=="Dementia":
                    event=1
                    break
            x,_=features(person,t0)
            result.append(dict(**x,rid=rid,t0=t0,first=t0==first,m06=t0==m06,days=days(t0,end),event=event))
    return pd.DataFrame(result).reindex(columns=["rid","t0","first","m06","days","event",*BRANCHES["M3"]])
