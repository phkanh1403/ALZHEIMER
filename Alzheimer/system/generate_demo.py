from pathlib import Path
import numpy as np
import pandas as pd

def generate():
    rng = np.random.default_rng(42)
    visits, measures = [], []
    for i in range(400):
        rid = f"DEMO-{i:03}"
        age = rng.integers(60,86)
        female, education = rng.integers(2), rng.integers(8,21)
        base = pd.Timestamp("2024-01-12")
        t0 = base + pd.DateOffset(months=6)
        change = rng.normal()
        days = int(min(360,max(1,rng.exponential(360*np.exp(-0.55*change)))))
        event = days < 360 and rng.random() < .8
        for date,dx in [(base,"MCI"),(t0,"MCI"),(t0+pd.Timedelta(days=days),"Dementia" if event else "MCI")]:
            visits.append([rid,date,dx,age,female,education])
        for var,start,delta in [
            ("mmse",rng.integers(24,30),rng.integers(-3,2)),
            ("faq",rng.integers(2,9),rng.integers(-1,4)),
            ("hippo",rng.normal(6500,450),-120-70*change),
            ("entor",rng.normal(2.8,.25),rng.normal(-.05,.03))]:
            for date,value in [(base,start),(t0,start+delta)]:
                measures.append([rid,date,var,value,1,"demo-v1"])
    path = Path(__file__).parent/"demo"
    path.mkdir(exist_ok=True)
    for name,records,columns in [
        ("visits",visits,"rid date dx age female education"),
        ("measures",measures,"rid date variable value qc version")]:
        pd.DataFrame(records,columns=columns.split()).to_csv(path/f"{name}.csv",index=False)

if __name__ == "__main__":
    generate()
