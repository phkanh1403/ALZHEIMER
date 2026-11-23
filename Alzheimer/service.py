import json
import numpy as np
import pandas as pd
from cohort import reconcile,landmarks,features,days
from etl import normalize,digest
from survival import probability
import repository as repo

CONTRACT="landmark-first-recorded-dementia-v1"
def snapshot(frames,raw,rules,parent=None):
    rows,issues=normalize(frames,rules)
    batch=digest(raw)
    for r in rows: r["source"]=batch[:12]+":"+r["source"]
    prior=repo.get("data",parent) if parent else {}
    batches=prior.get("batches",{})
    mapping_hash=digest(json.dumps(rules,sort_keys=True).encode())
    if batch in batches:
        if batches[batch]["mapping_hash"]!=mapping_hash: raise ValueError("Mapping đã đổi. Hãy tạo snapshot độc lập từ ZIP.")
        return parent
    batches[batch]=dict(mapping_hash=mapping_hash,mapping=rules,counts={k:len(v) for k,v in frames.items()})
    return store_data(prior.get("rows",[])+rows,prior.get("issues",[])+issues,batches,parent)
def store_data(rows,issues,batches,parent):
    phases={r["phase"] for r in rows if r["phase"]!="manual"}
    if "SIM" in phases and len(phases)>1: raise ValueError("Không trộn demo và ADNI")
    domain="SIM" if "SIM" in phases else "ADNI"
    clean,conflicts=reconcile(rows)
    ids={}; reverse={}
    for r in clean:
        if r.get("ptid"):
            ids.setdefault(r["rid"],set()).add(r["ptid"]); reverse.setdefault(r["ptid"],set()).add(r["rid"])
    bad={rid for rid,v in ids.items() if len(v)>1}
    for values in reverse.values():
        if len(values)>1: bad.update(values)
    clean=[r for r in clean if r["rid"] not in bad]
    conflicts.extend(dict(rid=k,reason="RID/PTID ambiguous") for k in sorted(bad))
    cohort=landmarks(clean)
    summary=dict(people=len({r["rid"] for r in clean}),visits=sum(r["kind"]=="dx" for r in clean),
                 landmarks=len(cohort),followup=int((cohort.days>0).sum()),events=int(cohort.loc[cohort.event.eq(1),"rid"].nunique()))
    return repo.save("data",f'{summary["people"]} RID · {summary["visits"]} lượt',dict(contract=CONTRACT,domain=domain,rows=rows,clean=clean,
                     issues=issues,conflicts=conflicts,batches=batches,parent=parent,summary=summary),True)
def score(bundle,person,t0,target,first):
    if bundle.get("contract")!=CONTRACT: raise ValueError("Phiên bản đặc trưng không tương thích")
    h=days(t0,target)
    if h<=0: raise ValueError("Ngày đích phải sau ngày gốc")
    person=sorted(person,key=lambda r:r["date"])
    dx=[r for r in person if r["kind"]=="dx" and r["date"]<=t0]
    if not dx or dx[-1]["date"]!=t0 or dx[-1]["value"]!="MCI" or any(r["value"] in ("Dementia","Unknown") for r in dx):
        raise ValueError("Ngày gốc không thuộc tập nguy cơ MCI")
    x,source=features(person,t0)
    reasons=[]
    for name,m in bundle["models"].items():
        missing=set(m["cols"])-x.keys()
        if missing: reasons.append(name+" thiếu "+", ".join(sorted(missing))); continue
        if name=="KM" and not first: continue
        if h not in m["supported"] and not any(a<h<b for a,b in m.get("intervals",[])):
            reasons.append(name+" ngoài miền thời gian"); continue
        p=float(probability(m,pd.DataFrame([x]),h)[0])
        return dict(branch=name,probability=p,t0=t0,target=target,h=h,features={k:x[k] for k in m["cols"]},evidence=source)
    raise ValueError("Chưa có nhánh phù hợp. "+" | ".join(reasons))
