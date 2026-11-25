"""Kho hồ sơ SQLite ver5.

- Đầu vào dự báo dựng bằng raw_points() trên MỌI lượt MCI, kể cả lượt mới nhất chưa có lần khám sau.
- Khóa hồ sơ: (nguồn, RID, ngày mốc). Nhập lại cùng RID với lượt mới sẽ thêm mốc mới;
  cùng (RID, ngày) sẽ được cập nhật.
- Mỗi hồ sơ lưu nguồn gốc số đo: ngày MRI, ngày điểm nhận thức, khoảng thời gian tính slope.
"""
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pipeline import MODELS, Sources, eligible_points, raw_points, apply_params

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
        out.append((f'{prefix}:{int(r.RID)}:{day}', f'{prefix} · RID {int(r.RID)}', day, json.dumps(feats, ensure_ascii=False)))
    return out


def upsert(db, rows):
    """Thêm mới hoặc cập nhật theo khóa (nguồn, RID, ngày mốc); trả về (thêm, cập nhật)."""
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
                raise ValueError('Thiếu, trùng hoặc vượt giới hạn kích thước: ' + table + '.csv')
            with archive.open(matches[0]) as src, (folder / (table + '.csv')).open('wb') as dest:
                shutil.copyfileobj(src, dest)
        frame = inputs_from_folder(folder, prep)
        rids = set(frame.RID)
        frame = frame[~frame.RID.isin(train_ids)]
        with db:
            added, updated = upsert(db, records(frame, 'ZIP'))
        return dict(added=added, updated=updated, rids=len(rids), skipped_train=len(rids & set(train_ids)))
