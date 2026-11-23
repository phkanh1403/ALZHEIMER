"""Integration and boundary checks against artificial input only."""
from datetime import date,timedelta
from tempfile import TemporaryDirectory
from copy import deepcopy
import os
from demo import generate
import repository as repo
from etl import read_package,normalize
from cohort import landmarks,features,reconcile
from service import snapshot,score,CONTRACT
from survival import train
def main():
    with TemporaryDirectory() as temp:
        os.environ["ADNI_HOME"]=temp
        raw,rules=generate()
        frames=read_package(raw)
        sid=snapshot(frames,raw,rules)
        assert snapshot(frames,raw,rules,sid)==sid
        data=repo.get("data",sid)
        d=landmarks(data["clean"])
        model=train(d,42,dict(events=5,risk=10,gap=.15))
        model.update(contract=CONTRACT,snapshot=sid,domain="SIM")
        assert "M0" in model["models"] and len(model["splits"])==320
        key=repo.save("model","demo",model,True)
        assert repo.current("model")==key
        row=d.iloc[0]; t=date.fromisoformat(row.t0)
        person=[r for r in data["clean"] if r["rid"]==row.rid]
        h=model["models"]["M0"]["supported"][0]
        receipt=score(model,person,row.t0,str(t+timedelta(days=h)),True)
        assert 0<=receipt["probability"]<=1
        truncated=[r for r in person if r["date"]<=row.t0]
        assert score(model,truncated,row.t0,str(t+timedelta(days=h)),True)==receipt
        mri=dict(person[0],kind="hippo",date=row.t0,available=str(t+timedelta(days=2)),value=1234)
        assert "hippo" not in features([mri],row.t0)[0]
        ambiguous=dict(next(r for r in person if r["kind"]=="dx"),value="Unknown")
        clean,issues=reconcile(person+[ambiguous])
        assert issues and landmarks(clean).empty
        bad=deepcopy(rules); bad["phases"]["SIM"]["dx_field"]=None
        normalized,issues=normalize(frames,bad)
        assert issues and all(r["value"]=="Unknown" for r in normalized if r["kind"]=="dx")
        for horizon in (-1,0,10000):
            try: score(model,person,row.t0,str(t+timedelta(days=horizon)),True)
            except ValueError: pass
            else: raise AssertionError("Invalid horizon accepted")
        print("PASS",data["summary"],"branches",list(model["models"]),"sample",receipt["probability"])
if __name__=="__main__": main()
