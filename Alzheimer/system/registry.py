"""Kho hồ sơ SQLite: hồ sơ train, hồ sơ nhập tay và hồ sơ nhập từ ZIP ADNI."""
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pipeline import MODELS, LABELS, build, read, has_table

FIELDS = MODELS['M3']
EXTRA = ['mri_span_days', 'field3t', 'mri_age_days']
BOUNDS = {
    'tau': (0.0, 20.0), 'tau2': (0.0, 400.0), 'age': (18.0, 115.0), 'sex': (1.0, 2.0), 'education': (0.0, 36.0),
    'mmse': (0.0, 30.0), 'faq': (0.0, 30.0), 'adas13': (0.0, 85.0), 'ravlt_delay': (0.0, 15.0),
    'hippo_adj': (1000.0, 15000.0), 'entor': (0.5, 6.0), 'field3t': (0.0, 1.0),
    'adas13_slope': (-30.0, 30.0), 'faq_slope': (-30.0, 30.0),
    'hippo_slope': (-2000.0, 2000.0), 'entor_slope': (-2.0, 2.0),
}
REQUIRED = ['DXSUM', 'MMSE', 'FAQ', 'PTDEMOG', 'UCSFFSX7', 'ADAS', 'NEUROBAT']
OPTIONAL = ['FOXLABBSI']


def latest_records(frame, prefix):
    """Mỗi RID lấy mốc gần nhất (thông tin mới nhất trước lần tái khám cuối)."""
    if frame.empty:
        return
    last = frame.sort_values('t0').drop_duplicates('RID', keep='last')
    for r in last.itertuples():
        feats = {k: (None if pd.isna(getattr(r, k)) else float(getattr(r, k))) for k in FIELDS + EXTRA}
        yield (f'{prefix}:{r.RID}', f'{prefix} · RID {r.RID}', str(pd.Timestamp(r.t0).date()), json.dumps(feats))


def save(db, rows):
    start = db.total_changes
    db.executemany('INSERT OR IGNORE INTO profiles(id,name,day,features) VALUES(?,?,?,?)', rows)
    return db.total_changes - start


def initialize(db, frame):
    db.execute('CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY)')
    with db:
        if db.execute("INSERT OR IGNORE INTO migrations VALUES('seed_train_v2')").rowcount:
            save(db, latest_records(frame, 'Train'))


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
        frame, _ = build(folder, prep)
        ids = set(read(folder, 'DXSUM').RID.dropna().astype(int))
        frame = frame[~frame.RID.isin(train_ids)]
        with db:
            added = save(db, latest_records(frame, 'ZIP'))
        return added, len(ids), len(ids & set(train_ids))
