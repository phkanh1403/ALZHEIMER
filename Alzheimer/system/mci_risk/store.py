"""SQLite store of the clinician's patients.

Each record is one patient visit: id, display name, visit day, JSON model inputs (with provenance) and a star flag.
Every call opens and closes its own connection, so the store is safe to share between Streamlit sessions.
"""
import sqlite3
from contextlib import closing, contextmanager
from pathlib import Path
from typing import NamedTuple

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles(id TEXT PRIMARY KEY, name TEXT, day TEXT, features TEXT, star INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY);
"""
# Earlier versions copied the whole training cohort into this table but never showed it; remove those rows once.
MIGRATIONS = [('drop_training_cohort_copy', "DELETE FROM profiles WHERE id LIKE 'Train:%'")]


class ProfileRow(NamedTuple):
    id: str
    name: str
    day: str
    features: str   # JSON


class PatientStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._transaction() as db:
            db.executescript(SCHEMA)
            for name, sql in MIGRATIONS:
                if db.execute('INSERT OR IGNORE INTO migrations VALUES(?)', (name,)).rowcount:
                    db.execute(sql)

    @contextmanager
    def _transaction(self):
        with closing(sqlite3.connect(self.path)) as db, db:
            yield db

    def all(self):
        with closing(sqlite3.connect(self.path)) as db:
            return pd.read_sql('SELECT * FROM profiles ORDER BY star DESC, name, day', db)

    def upsert(self, rows):
        """Insert or update by id (source:RID:visit date for imports); returns (added, updated)."""
        added = updated = 0
        with self._transaction() as db:
            for row in rows:
                exists = db.execute('SELECT 1 FROM profiles WHERE id=?', (row[0],)).fetchone() is not None
                db.execute('INSERT INTO profiles(id, name, day, features) VALUES(?, ?, ?, ?) '
                           'ON CONFLICT(id) DO UPDATE SET name=excluded.name, day=excluded.day, '
                           'features=excluded.features', tuple(row))
                updated += exists
                added += not exists
        return added, updated

    def toggle_star(self, profile_id):
        with self._transaction() as db:
            db.execute('UPDATE profiles SET star = 1 - star WHERE id=?', (profile_id,))

    def delete(self, profile_id):
        with self._transaction() as db:
            db.execute('DELETE FROM profiles WHERE id=?', (profile_id,))
