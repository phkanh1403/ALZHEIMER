"""Patient store (SQLite) and ADNI ZIP import."""
import io
import json
import sqlite3
import zipfile

import pandas as pd
import pytest

from mci_risk.adni_import import import_zip, profile_rows
from mci_risk.config import OPTIONAL_TABLES, REQUIRED_TABLES, TRAIN_DIR
from mci_risk.data import read_table
from mci_risk.store import PatientStore, ProfileRow
from tests.conftest import needs_data


@pytest.fixture
def store(tmp_path):
    return PatientStore(tmp_path / 'profiles.sqlite')


def test_reimport_adds_new_visits_and_updates_existing_ones(store):
    frame = pd.DataFrame([dict(RID=1, t0=pd.Timestamp('2020-01-01')), dict(RID=1, t0=pd.Timestamp('2021-01-01'))])
    assert store.upsert(profile_rows(frame.iloc[:1], 'ZIP')) == (1, 0)
    assert store.upsert(profile_rows(frame, 'ZIP')) == (1, 0)
    assert store.upsert(profile_rows(frame, 'ZIP')) == (0, 1)
    assert list(store.all().id) == ['ZIP:1:2020-01-01', 'ZIP:1:2021-01-01']


def test_star_and_delete(store):
    store.upsert([ProfileRow('Manual:a', 'A', '2026-01-01', '{}'), ProfileRow('Manual:b', 'B', '2026-01-01', '{}')])
    store.toggle_star('Manual:b')
    assert store.all().id.iloc[0] == 'Manual:b'            # starred first
    store.delete('Manual:a')
    assert list(store.all().id) == ['Manual:b']


def test_old_training_cohort_copy_is_removed_once(tmp_path):
    path = tmp_path / 'old.sqlite'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, '
                   'star INTEGER DEFAULT 0)')
        db.executemany('INSERT INTO profiles(id, name, day, features) VALUES(?, ?, ?, ?)',
                       [('Train:1:2020-01-01', 'RID 1', '2020-01-01', '{}'), ('Manual:x', 'X', '2026-01-01', '{}')])
    store = PatientStore(path)
    assert list(store.all().id) == ['Manual:x']
    store.upsert([ProfileRow('Train:2:2020-01-01', 'kept now', '2020-01-01', '{}')])
    assert len(PatientStore(path).all()) == 2                 # the migration runs only once


def test_profile_rows_store_json_safe_values():
    frame = pd.DataFrame([dict(RID=7, t0=pd.Timestamp('2020-05-01'), mmse=27, mri_date=pd.Timestamp('2020-04-01'),
                               hippo_adj=float('nan'))])
    row = profile_rows(frame, 'ZIP')[0]
    features = json.loads(row.features)
    assert row.id == 'ZIP:7:2020-05-01' and features['mmse'] == 27.0
    assert features['mri_date'] == '2020-04-01' and features['hippo_adj'] is None


def _export(rids):
    """A small ADNI export (ZIP in memory) with the given participants, tables in nested folders."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as z:
        for table in REQUIRED_TABLES + OPTIONAL_TABLES[:1]:
            d = read_table(TRAIN_DIR, table)
            z.writestr(f'export/{table.lower()}/{table}.csv', d[d.RID.isin(rids)].to_csv(index=False))
    buffer.seek(0)
    return buffer


@needs_data
def test_zip_import_skips_training_participants(store, prep, points):
    rids = sorted(points.RID.unique())[:12]
    result = import_zip(store, _export(rids), prep, train_ids=set(rids[:4]))
    assert result['skipped_train'] == 4 and result['added'] == len(store.all()) > 0
    assert not store.all().id.str.contains('|'.join(f'ZIP:{r}:' for r in rids[:4])).any()


@needs_data
def test_zip_without_a_required_table_is_rejected(store, prep):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('DXSUM.csv', 'RID\n1\n')
    with pytest.raises(ValueError, match=r'MMSE\.csv'):
        import_zip(store, buffer, prep, train_ids=set())
