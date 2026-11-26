"""Import patients from an ADNI data export (ZIP) into the patient store.

Prediction inputs come from data.raw_points() at every MCI visit, including the latest one without a later visit.
Record id: (source, RID, visit date). Re-importing a participant with a new visit adds it; the same (RID, date) is
updated. Each record also stores where its values come from (MRI date, test dates, trend time span).
"""
import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .config import MODELS, OPTIONAL_TABLES, REQUIRED_TABLES
from .data import Sources, apply_params, eligible_points, raw_points
from .store import ProfileRow

FIELDS = MODELS['M3']
PROVENANCE = ['mri_date', 'mri_from', 'mri_span_days', 'mri_age_days', 'scanner', 'mmse_date', 'adas13_date',
              'adas13_from', 'faq_from', 'has_followup']
MAX_TABLE_BYTES = 300_000_000
SOURCE_LABELS = {'ZIP': 'ADNI import'}


def _json_value(v):
    """Convert pandas/numpy scalars to JSON values (missing -> None, dates -> ISO day)."""
    if v is None or v is pd.NaT or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, pd.Timestamp | np.datetime64):
        return str(pd.Timestamp(v).date())
    if isinstance(v, np.integer | np.floating):
        return float(v)
    return v


def profile_rows(frame, source, latest_only=True):
    """Store rows for the visits in `frame` (by default only each participant's latest visit)."""
    if frame.empty:
        return []
    f = frame.sort_values('t0')
    if latest_only:
        f = f.drop_duplicates('RID', keep='last')
    label = SOURCE_LABELS.get(source, 'ADNI cohort')
    rows = []
    for _, r in f.iterrows():
        features = {k: _json_value(r.get(k)) for k in FIELDS + PROVENANCE}
        day = str(pd.Timestamp(r.t0).date())
        rows.append(ProfileRow(f'{source}:{int(r.RID)}:{day}', f'RID {int(r.RID)} ({label})', day,
                               json.dumps(features, ensure_ascii=False)))
    return rows


def inputs_from_folder(folder, prep):
    src = Sources(folder)
    return apply_params(raw_points(src, eligible_points(src.vis)), prep)


def _extract_tables(archive, folder):
    """Copy the needed CSV files (found anywhere in the ZIP) into folder; file names only, never ZIP paths."""
    for table in REQUIRED_TABLES + OPTIONAL_TABLES:
        matches = [i for i in archive.infolist() if Path(i.filename).name == table + '.csv']
        if table in OPTIONAL_TABLES and not matches:
            continue
        if len(matches) != 1 or matches[0].file_size > MAX_TABLE_BYTES:
            raise ValueError(f'Missing, duplicated or oversized file: {table}.csv')
        with archive.open(matches[0]) as src, (folder / f'{table}.csv').open('wb') as dest:
            shutil.copyfileobj(src, dest)


def import_zip(store, file, prep, train_ids):
    """Import every participant's latest MCI visit; participants in the training data are skipped."""
    with tempfile.TemporaryDirectory() as tmp, ZipFile(file) as archive:
        folder = Path(tmp)
        _extract_tables(archive, folder)
        frame = inputs_from_folder(folder, prep)
    rids = set(frame.RID)
    added, updated = store.upsert(profile_rows(frame[~frame.RID.isin(train_ids)], 'ZIP'))
    return dict(added=added, updated=updated, rids=len(rids), skipped_train=len(rids & set(train_ids)))
