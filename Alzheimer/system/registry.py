"""SQLite patient store.

- Prediction inputs come from raw_points() at EVERY MCI visit, including the latest one without a later visit.
- Record key: (source, RID, visit date). Re-importing a participant with a new visit adds it;
  the same (RID, date) is updated.
- Each record stores where values come from: MRI date, test dates, trend time span.
"""
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pipeline import MODELS, MIN_SPAN, Sources, eligible_points, raw_points, apply_params, blup_slope

FIELDS = MODELS['M3']
PROVENANCE = ['mri_date', 'mri_from', 'mri_span_days', 'mri_age_days', 'scanner', 'mmse_date', 'adas13_date',
              'adas13_from', 'faq_from', 'has_followup']
BOUNDS = {
    'tau': (0.0, 20.0), 'age': (18.0, 115.0), 'sex': (1.0, 2.0), 'education': (0.0, 36.0),
    'mmse': (0.0, 30.0), 'faq': (0.0, 30.0), 'adas13': (0.0, 85.0), 'ravlt_delay': (0.0, 15.0),
    'hippo_adj': (1000.0, 15000.0), 'entor': (0.5, 6.0), 'field3t': (0.0, 1.0),
    'adas13_slope': (-30.0, 30.0), 'faq_slope': (-30.0, 30.0),
    'hippo_slope': (-2000.0, 2000.0), 'entor_slope': (-2.0, 2.0),
}
REQUIRED = ['DXSUM', 'MMSE', 'FAQ', 'PTDEMOG', 'UCSFFSX7', 'ADAS', 'NEUROBAT']
OPTIONAL = ['FOXLABBSI', 'All_Subjects_Structural_MRI_Images']


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NaT:
        return None
    if isinstance(v, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(v).date())
    if isinstance(v, (np.integer, np.floating)):
        return float(v)
    return v


def records(frame, prefix, latest_only=True):
    if frame.empty:
        return []
    f = frame.sort_values('t0')
    if latest_only:
        f = f.drop_duplicates('RID', keep='last')
    out = []
    for _, r in f.iterrows():
        feats = {k: _clean(r.get(k)) for k in FIELDS + PROVENANCE}
        day = str(pd.Timestamp(r.t0).date())
        label = 'ADNI import' if prefix == 'ZIP' else 'ADNI cohort'
        out.append((f'{prefix}:{int(r.RID)}:{day}', f'RID {int(r.RID)} ({label})', day, json.dumps(feats, ensure_ascii=False)))
    return out


def upsert(db, rows):
    """Insert or update by (source, RID, visit date); returns (added, updated)."""
    added = updated = 0
    for row in rows:
        exists = db.execute('SELECT 1 FROM profiles WHERE id=?', (row[0],)).fetchone()
        db.execute('INSERT INTO profiles(id,name,day,features) VALUES(?,?,?,?) '
                   'ON CONFLICT(id) DO UPDATE SET name=excluded.name, day=excluded.day, features=excluded.features', row)
        updated += bool(exists)
        added += not exists
    return added, updated


def initialize(db, inputs):
    db.execute('CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY)')
    with db:
        if db.execute("INSERT OR IGNORE INTO migrations VALUES('seed_train_v5')").rowcount:
            db.execute("DELETE FROM profiles WHERE id LIKE 'Train:%'")
            upsert(db, records(inputs, 'Train'))


def inputs_from_folder(folder, prep):
    src = Sources(folder)
    return apply_params(raw_points(src, eligible_points(src.vis)), prep)


def import_zip(db, file, prep, train_ids):
    with tempfile.TemporaryDirectory() as tmp, ZipFile(file) as archive:
        folder = Path(tmp)
        for table in REQUIRED + OPTIONAL:
            matches = [i for i in archive.infolist() if Path(i.filename).name == table + '.csv']
            if table in OPTIONAL and not matches:
                continue
            if len(matches) != 1 or matches[0].file_size > 300_000_000:
                raise ValueError('Missing, duplicated or oversized file: ' + table + '.csv')
            with archive.open(matches[0]) as src, (folder / (table + '.csv')).open('wb') as dest:
                shutil.copyfileobj(src, dest)
        frame = inputs_from_folder(folder, prep)
        rids = set(frame.RID)
        frame = frame[~frame.RID.isin(train_ids)]
        with db:
            added, updated = upsert(db, records(frame, 'ZIP'))
        return dict(added=added, updated=updated, rids=len(rids), skipped_train=len(rids & set(train_ids)))


def _series_slope(dates, values, params):
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    o = np.argsort(d.values)
    d, v = d.iloc[o].reset_index(drop=True), np.asarray(values, float)[o]
    span = int((d.iloc[-1] - d.iloc[0]).days)
    if span < MIN_SPAN:
        return None, span, d.iloc[0]
    t = ((d - d.iloc[0]).dt.days / 365.25).to_numpy()
    return blup_slope(t, v, params), span, d.iloc[0]


def manual_features(prep, visit, mci_years, age, sex, education, mmse, faq, adas13, ravlt, mri_date=None, hv=None, icv=None,
                    entor=None, field='3T', cog_history=None, mri_history=None):
    """Model inputs for a patient entered by hand. Trends use the frozen training parameters (BLUP)."""
    v = pd.Timestamp(visit)
    b, m = prep['icv']['b'], prep['icv']['icv_mean']
    f = dict(tau=float(mci_years), tau2=float(mci_years) ** 2, age=float(age), sex=1.0 if sex == 'Male' else 2.0,
             education=float(education), mmse=float(mmse), faq=float(faq), adas13=float(adas13), ravlt_delay=float(ravlt),
             mmse_date=str(v.date()), adas13_date=str(v.date()))
    if hv and icv and entor:
        md = pd.Timestamp(mri_date) if mri_date else v
        f.update(hippo_adj=float(hv - b * (icv - m)), entor=float(entor), field3t=1.0 if field == '3T' else 0.0,
                 mri_date=str(md.date()), mri_age_days=int((v - md).days))
    if cog_history is not None and len(cog_history):
        ch = cog_history.dropna(subset=['Date'])
        for col, key, cur in [('ADAS-Cog13', 'adas13', adas13), ('FAQ', 'faq', faq)]:
            s = ch.dropna(subset=[col])
            if len(s):
                slope, span, start = _series_slope(list(s.Date) + [v], list(s[col]) + [cur], prep[key])
                if slope is not None:
                    f[f'{key}_slope'], f[f'{key}_from'] = slope, str(start.date())
    cols = ['Date', 'Hippocampal volume (mm³)', 'Intracranial volume (mm³)', 'Entorhinal thickness (mm)']
    if mri_history is not None and len(mri_history) and 'hippo_adj' in f:
        mh = mri_history.dropna(subset=cols)
        if len(mh):
            dates = list(mh.Date) + [pd.Timestamp(f['mri_date'])]
            hadj = list(mh[cols[1]].astype(float) - b * (mh[cols[2]].astype(float) - m)) + [f['hippo_adj']]
            slope, span, start = _series_slope(dates, hadj, prep['hippo_adj'])
            f['mri_span_days'], f['mri_from'] = span, str(start.date())
            if slope is not None:
                f['hippo_slope'] = slope
                f['entor_slope'] = _series_slope(dates, list(mh[cols[3]].astype(float)) + [f['entor']], prep['entor'])[0]
    return f
