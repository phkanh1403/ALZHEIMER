"""Piecewise exponential likelihood, person splits, IPCW validation; JSON model."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score
from cohort import BRANCHES

EDGES=[90,183,365,730,1095]
def exposure(t, edges):
    return np.clip(np.asarray(t)[:,None]-np.r_[0,edges[:-1]],0,np.diff([0,*edges]))
def fit(df,cols,edges):
    x=df[cols].to_numpy(float)
    mean=x.mean(0); sd=x.std(0); sd[sd<1e-8]=1
    z=(x-mean)/sd
    e=exposure(df.days,edges)
    k=np.searchsorted(edges,df.days)
    hit=df.event.to_numpy(bool)&(k<len(edges))
    counts=np.bincount(k[hit],minlength=len(edges))
    if min(counts)<1: raise ValueError("Khoảng train thiếu biến cố")
    def objective(w):
        alpha=w[:len(edges)]; beta=w[len(edges):]
        eta=alpha[None,:]+(z@beta)[:,None]
        mu=e*np.exp(eta)
        loss=mu.sum()-counts@alpha-(z[hit]@beta).sum()+0.5*(beta@beta)
        grad=np.r_[mu.sum(0)-counts,z.T@mu.sum(1)-z[hit].sum(0)+beta]
        return loss,grad
    opt=minimize(objective,np.r_[np.log(counts/e.sum(0)),np.zeros(len(cols))],jac=True,
                 method="L-BFGS-B",bounds=[(-20,0)]*len(edges)+[(-4,4)]*len(cols))
    if not opt.success: raise ValueError("Tối ưu không hội tụ")
    return dict(type="piecewise",cols=cols,mean=mean.tolist(),sd=sd.tolist(),edges=edges,
                alpha=opt.x[:len(edges)].tolist(),beta=opt.x[len(edges):].tolist())

def km(df,censor=False):
    t=df.days.to_numpy(); flag=df.event.to_numpy()==(0 if censor else 1)
    times,counts=np.unique(t[flag],return_counts=True)
    risk=len(t)-np.searchsorted(np.sort(t),times)
    return times.tolist(),np.cumprod(1-counts/risk).tolist()
def step(curve,t,left=False):
    idx=np.searchsorted(curve[0],t,side="left" if left else "right")
    return np.r_[1.,curve[1]][idx]
def probability(m,df,h):
    if m["type"]=="km": return np.full(len(df),1-step(m["curve"],h))
    x=(df[m["cols"]].to_numpy(float)-m["mean"])/m["sd"]
    base=exposure([h],m["edges"])[0]@np.exp(m["alpha"])
    return -np.expm1(-base*np.exp(x@m["beta"]))
def evaluate(m,df,h,curve):
    if df.empty: return {}
    t=df.days.to_numpy(); d=df.event.to_numpy(bool)
    cases=d&(t<=h); controls=(t>h)|((t==h)&~d)
    gc=step(curve,h); ge=step(curve,t[cases],left=True)
    if (gc<=0 and (t>h).any()) or any(ge<=0): return {}
    w=np.zeros(len(df)); w[cases]=1/ge; w[t>h]=1/gc if gc>0 else 0
    boundary=controls&(t==h)
    if boundary.any(): w[boundary]=1/step(curve,h,left=True)
    p=probability(m,df,h)
    known=w>0
    auc=None
    if cases.any() and controls.any():
        auc=float(roc_auc_score(cases[known],p[known],sample_weight=w[known]))
    if not known.any(): return {}
    observed=float(np.average(cases,weights=w))
    # Cluster bootstrap on fixed predictions and development censoring distribution.
    by=pd.DataFrame(dict(rid=df.rid.to_numpy(),a=w*cases,b=w,c=p)).groupby("rid").agg(
        a=("a","sum"),b=("b","sum"),c=("c","sum"),n=("c","size")).to_numpy()
    draws=np.random.default_rng(283).integers(len(by),size=(150,len(by)))
    tot=by[draws].sum(axis=1); tot=tot[tot[:,1]>0]
    diff=tot[:,2]/tot[:,3]-tot[:,0]/tot[:,1]
    low,high=np.quantile(diff,[.025,.975]) if len(diff) else (-1.,1.)
    flagged=p>=0.20
    return dict(gap_ci_low=float(low),gap_ci_high=float(high),threshold=.2,
                tp=int(sum(flagged&cases)),fp=int(sum(flagged&controls)),
                tn=int(sum(~flagged&controls)),fn=int(sum(~flagged&cases)),n=len(t),people=df.rid.nunique(),events=df.loc[cases,"rid"].nunique(),
                risk=df.loc[t>=h,"rid"].nunique(),unknown=int((~known).sum()),auc=auc,
                brier=float(np.mean(w*(cases.astype(float)-p)**2)),pred=float(p.mean()),observed=observed,
                gap=abs(float(p.mean())-observed))
def train(df,seed,policy):
    ids=np.array(sorted(df.rid.unique()))
    if len(ids)<10: raise ValueError("Cần thêm người để chia ba tập độc lập")
    np.random.default_rng(seed).shuffle(ids)
    a,b=int(.6*len(ids)),int(.8*len(ids))
    splits={v:s for part,s in [(ids[:a],"train"),(ids[a:b],"validation"),(ids[b:],"test")] for v in part}
    df=df[df.days>0].copy(); df["split"]=df.rid.map(splits)
    models={}; report=[]; fitted={}
    for name,cols in {**BRANCHES,"KM": []}.items():
        group=df.dropna(subset=cols) if cols else df[df["first"]]
        tr,va,te=[group[group.split==s] for s in ("train","validation","test")]
        if tr.empty: continue
        edges=[]; last=0
        for h in EDGES:
            if tr.loc[tr.event.eq(1)&tr.days.between(last+1,h),"rid"].nunique()>=policy["events"]:
                edges.append(h); last=h
        if not edges: continue
        try:
            m=fit(tr,cols,edges) if cols else dict(type="km",cols=[],curve=km(tr))
        except ValueError as exc:
            report.append(dict(branch=name,reason=str(exc))); continue
        curve=km(tr,censor=True); supported=[]; fitted[name]=(m,curve)
        for h in edges:
            val=evaluate(m,va,h,curve)
            passed=bool(val and val["events"]>=policy["events"] and val["risk"]>=policy["risk"] and max(abs(val["gap_ci_low"]),abs(val["gap_ci_high"]))<=policy["gap"])
            for split,stats in [("validation",val),("test",evaluate(m,te,h,curve))]:
                report.append(dict(branch=name,h=h,split=split,passed_validation=passed,**stats))
            if passed: supported.append(h)
        if supported:
            intervals=[]
            for lo,hi in zip(edges[:-1],edges[1:]):
                if lo in supported and hi in supported:
                    mid=evaluate(m,va,(lo+hi)//2,curve)
                    if mid and max(abs(mid["gap_ci_low"]),abs(mid["gap_ci_high"]))<=policy["gap"] and mid["risk"]>=policy["risk"] and mid["events"]>=policy["events"]:
                        intervals.append([lo,hi])
            models[name]=dict(m,supported=supported,intervals=intervals)
    # Paired evaluations on identical held-out landmarks.
    test=df[df.split=="test"]
    for label,names,common in [
        ("M06/H183",["M0","M1","M2","M3"],test[test.m06].dropna(subset=BRANCHES["M3"])),
        ("MRI lặp",["M2","M3"],test.dropna(subset=BRANCHES["M3"]))]:
        for name in names:
            if name in fitted and len(common):
                m,g=fitted[name]
                if 183<=max(m["edges"]):
                    report.append(dict(branch=name,h=183,split="test ghép cặp · "+label,**evaluate(m,common,183,g)))
    return dict(models=models,metrics=report,splits=splits,policy=policy,seed=seed)
