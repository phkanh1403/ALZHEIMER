"""ADNI landmark cohort, as-of feature extraction and ridge Cox regression."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
MODELS = {
    'M0': ['age', 'sex', 'education', 'mmse', 'faq'],
    'M1': ['hippo', 'entor'],
    'M2': ['mmse_rate', 'faq_rate'],
    'M3': ['hippo_rate', 'entor_rate']
}
for prev, name in zip(['M0','M1','M2'], ['M1','M2','M3']):
    MODELS[name] = MODELS[prev] + MODELS[name]

def read(root, name):
    files = list(Path(root).rglob(name + '.csv'))
    if len(files) != 1:
        raise ValueError(f'Expected one {name}.csv, found {len(files)}')
    return pd.read_csv(files[0], encoding='latin1', low_memory=False)

def diagnoses(dx):
    dx = dx.copy()
    dx['day'] = pd.to_datetime(dx.EXAMDATE, errors='coerce')
    dx['screen'] = (dx.VISCODE.fillna('').str.lower().str.startswith('sc') |
                    dx.VISCODE2.fillna('').str.lower().str.startswith('sc'))
    valid = dx[dx.DIAGNOSIS.isin([1,2,3]) & dx.day.notna()]
    keys = ['RID','day']
    conflicts = valid.groupby(keys).DIAGNOSIS.nunique()
    conflicts = conflicts[conflicts > 1].reset_index()[keys]
    visits = valid[~valid.RID.isin(conflicts.RID)].groupby(keys).agg(
        dx=('DIAGNOSIS','first'), screen=('screen','all')).reset_index()
    rows = []
    for rid, history in visits.groupby('RID'):
        mci = history[(history.dx == 2) & ~history.screen]
        if len(mci) < 2:
            continue
        t = mci.day.iloc[1]
        if ((history.day <= t) & (history.dx == 3)).any():
            continue
        future = history[history.day > t]
        events = future[future.dx == 3]
        end = events.day.min() if len(events) else history.day.max()
        rows.append((rid,t,(end-t).days,int(len(events)>0)))
    return pd.DataFrame(rows, columns=['RID','t0','days','event']), conflicts

def dated(df, column):
    return df.assign(day=pd.to_datetime(df[column],errors='coerce')).dropna(subset=['day'])

def measurements(root):
    series = {}
    for table, col, name in [('MMSE','MMSCORE','mmse'),('FAQ','FAQTOTAL','faq')]:
        d = dated(read(root,table),'VISDATE')
        val = pd.to_numeric(d[col],errors='coerce')
        good = val.between(0,30)
        if 'DONE' in d:
            good &= ~d.DONE.eq(0)
        if 'HAS_QC_ERROR' in d:
            good &= ~d.HAS_QC_ERROR.astype(str).str.lower().isin(['1','1.0','true','yes'])
        series[name] = d[good].assign(value=val[good], version='clinical')
    m = dated(read(root,'UCSFFSX7'),'EXAMDATE')
    cols = ['ST29SV','ST88SV','ST24TA','ST83TA']
    m[cols] = m[cols].apply(pd.to_numeric, errors='coerce')
    m = m[~m.OVERALLQC.isin(['Fail','Hippocampus Only']) & ~m.HIPPOQC.eq('Fail') & ~m.TEMPQC.eq('Fail')
          & m[cols].gt(0).all(axis=1)].copy()
    m['available'] = pd.to_datetime(m.RUNDATE, errors='coerce')
    m['version'] = m.FSVER
    series['hippo'] = m.assign(value=m.ST29SV + m.ST88SV)
    series['entor'] = m.assign(value=(m.ST24TA + m.ST83TA)/2)
    return {name:{rid:g for rid,g in df.groupby('RID')} for name,df in series.items()}

def cohort(root):
    rows, conflicts = diagnoses(read(root,'DXSUM'))
    demo = dated(read(root,'PTDEMOG'),'VISDATE')
    people = dict(tuple(demo.groupby('RID')))
    streams = measurements(root)
    values = []
    for r in rows.itertuples():
        result = {}
        if r.RID in people:
            g = people[r.RID]
            g = g[g.day <= r.t0].sort_values('day')
            if len(g):
                for col,key,lo,hi in [('PTEDUCAT','education',0,36),('PTGENDER','sex',1,2)]:
                    s = pd.to_numeric(g[col],errors='coerce')
                    s = s[s.between(lo,hi)]
                    if len(s): result[key] = float(s.iloc[-1])
                years = pd.to_numeric(g.PTDOBYY.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0],errors='coerce')
                years = years.fillna(pd.to_numeric(g.PTDOB.astype(str).str.extract(r'((?:18|19|20)\d{2})')[0],errors='coerce')).dropna().unique()
                if len(years)==1 and 18 <= r.t0.year-years[0] <= 115:
                    result['age'] = r.t0.year-years[0]
        for name, groups in streams.items():
            g = groups.get(r.RID)
            if g is None: continue
            g = g[g.day <= r.t0]
            # Retrospective acquisition-date analysis; QC unknown is retained.
            if g.empty: continue
            g = g[['day','value','version']].drop_duplicates()
            g = g[~g.day.duplicated(keep=False)].sort_values('day')
            if g.empty: continue
            last = g.iloc[-1]
            if (r.t0-last.day).days > (365 if name in ('hippo','entor') else 45): continue
            result[name] = last.value
            result[name+'_date'] = str(last.day.date())
            old = g[(g.day <= last.day-pd.Timedelta(days=90)) & (g.version == last.version)]
            if len(old):
                first = old.iloc[-1]
                result[name+'_previous'] = str(first.day.date())
                result[name+'_rate'] = 365.25*(last.value-first.value)/(last.day-first.day).days
        values.append(result)
    features = pd.DataFrame(values,index=rows.index)
    for col in MODELS['M3']:
        if col not in features: features[col] = np.nan
    return pd.concat([rows, features],axis=1), conflicts

def fit(frame, cols, penalty=1.0):
    X = frame[cols].to_numpy(float)
    mean, scale = X.mean(0), X.std(0)
    scale[scale==0] = 1
    X = (X-mean)/scale
    t, e = frame.days.to_numpy(), frame.event.to_numpy(bool)
    times, counts = np.unique(t[e],return_counts=True)
    risk = (t[None,:]>=times[:,None]).astype(float)
    def objective(beta):
        z = X@beta
        shift = z.max()
        w = np.exp(z-shift)
        den = risk@w
        loss = counts@(np.log(den)+shift)-z[e].sum()+penalty*(beta@beta)/2
        grad = X.T@(w*((counts/den)@risk))-X[e].sum(0)+penalty*beta
        return loss, grad
    opt = minimize(objective,np.zeros(len(cols)),jac=True,method='L-BFGS-B')
    if not opt.success: raise RuntimeError(opt.message)
    hazards = counts/(risk@np.exp(X@opt.x))
    return dict(columns=cols,mean=mean.tolist(),scale=scale.tolist(),beta=opt.x.tolist(),
                times=times.tolist(),baseline=np.cumsum(hazards).tolist())

def predict(model, frame, horizon):
    z = (frame[model['columns']].to_numpy(float)-model['mean'])/model['scale']
    idx = np.searchsorted(model['times'],horizon,side='right')-1
    base = model['baseline'][idx] if idx>=0 else 0.
    return -np.expm1(-base*np.exp(z@model['beta']))
