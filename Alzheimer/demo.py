"""Create synthetic ADNI-shaped input, never clinical data."""
import csv
import io
import json
import zipfile
from datetime import date,timedelta
from pathlib import Path
import numpy as np
from etl import PATHS

def generate():
    rng=np.random.default_rng(731)
    tables={k:[] for k in ["DXSUM","MMSE","FAQ","PTDEMOG","DATADIC"]}
    for i in range(320):
        base=date(2018,1,1)+timedelta(days=int(rng.integers(0,1200)))
        age=int(rng.integers(58,89)); severity=rng.normal()
        event=rng.exponential(800*np.exp(-0.65*severity))
        follow=int(rng.integers(250,1550))
        common=dict(RID=f"SIM-{i:04d}",PHASE="SIM")
        tables["PTDEMOG"].append(dict(**common,VISDATE=str(base),PTDOBYY=base.year-age,PTGENDER=str(i%2),PTEDUCAT=int(rng.integers(8,21))))
        t=0
        while t<=follow:
            when=str(base+timedelta(days=t))
            tables["DXSUM"].append(dict(**common,EXAMDATE=when,DIAGNOSIS="D" if t>=event else "M"))
            tables["MMSE"].append(dict(**common,VISDATE=when,DONE="Y",MMSCORE=int(np.clip(26-2*severity-t/600+rng.normal(),0,30))))
            tables["FAQ"].append(dict(**common,VISDATE=when,FAQTOTAL=int(np.clip(5+2*severity+t/400+rng.normal(),0,30))))
            if t>=event: break
            t+=int(rng.integers(30,155))
    for code,meaning in [("D","Dementia"),("M","MCI")]:
        tables["DATADIC"].append(dict(PHASE="SIM",TBLNAME="DXSUM",FLDNAME="DIAGNOSIS",CODE=code,TEXT=meaning,UNITS=""))
    rule=dict(reviewed=True,phases=dict(SIM=dict(dx_date="EXAMDATE",dx_field="DIAGNOSIS",diagnoses={"D":"Dementia","M":"MCI"},female={"0":0,"1":1},mmse_done=["Y"])))
    buf=io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for k,rows in tables.items():
            f=io.StringIO(); w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            z.writestr(PATHS[k],f.getvalue())
    return buf.getvalue(),rule
if __name__=="__main__":
    raw,m=generate()
    Path("demo_sim.zip").write_bytes(raw)
    Path("demo_mapping.json").write_text(json.dumps(m,indent=2),encoding="utf-8")
