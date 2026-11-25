import json
from pipeline import ROOT, MODELS, cohort, fit

def train():
    df, issues = cohort(ROOT/'data/train')
    out = ROOT/'runs'
    out.mkdir(exist_ok=True)
    df.to_csv(out/'train_landmarks.csv',index=False)
    issues.to_csv(out/'conflicting_dates.csv',index=False)
    shared=df[df.days>0].dropna(subset=MODELS["M3"])
    secondary={}
    models = {}
    for name, cols in MODELS.items():
        data = df[df.days>0].dropna(subset=cols)
        secondary[name]=dict(n=len(data),model=fit(data,cols))
        data=shared
        n, events = len(data), int(data.event.sum())
        # Conservative feasibility floor, not evidence of clinical validity.
        model = fit(data,cols) if events>=max(10,len(cols)) and n-events>=10 else None
        models[name] = dict(n=n,events=events,model=model)
        print(name,n,events,'fitted' if model else 'insufficient')
    (out/'secondary_models.json').write_text(json.dumps(secondary))
    tmp = out/'models.tmp'
    tmp.write_text(json.dumps(models,allow_nan=False),encoding='utf8')
    tmp.replace(out/'models.json')
if __name__=='__main__':train()
