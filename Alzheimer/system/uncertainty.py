"""Bootstrap participants with replacement within each training cohort."""
import json
from pipeline import ROOT, MODELS, pd, np, fit, predict
from evaluate import HORIZONS
def build():
    rng=np.random.default_rng(42)
    df=pd.read_csv(ROOT/'runs/train_landmarks.csv')
    output={}
    for name,cols in MODELS.items():
        group=df[df.days>0].dropna(subset=MODELS["M3"])
        draws=[]
        for i in range(200):
            sample=group.iloc[rng.integers(len(group),size=len(group))]
            m=fit(sample,cols)
            m['baseline']=[m['baseline'][np.searchsorted(m['times'],h,side='right')-1] if min(m['times'])<=h else 0 for h in HORIZONS]
            m['times']=HORIZONS
            draws.append(m)
        output[name]=draws
    (ROOT/'runs/bootstrap.json').write_text(json.dumps(output))
if __name__=='__main__':build()
