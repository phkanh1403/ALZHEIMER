"""Append-only encounter revisions and prediction snapshots."""
import json
import os
from sqlalchemy import create_engine, MetaData, Table, Column, String, Text, select
from uuid import uuid4
from datetime import datetime, timezone

engine = create_engine(os.getenv("DATABASE_URL", "sqlite:///clinical_demo.sqlite3"))
meta = MetaData()
entries = Table("clinical_history", meta, Column("id", String(32), primary_key=True),
                Column("patient", String(100), nullable=False, index=True),
                Column("kind", String(20), nullable=False),
                Column("created", String(40), nullable=False), Column("payload", Text, nullable=False))
meta.create_all(engine)


def record(patient, kind, payload):
    if not patient.strip():
        raise ValueError("Nhập mã hồ sơ.")
    with engine.begin() as connection:
        connection.execute(entries.insert().values(id=uuid4().hex, patient=patient.strip(),
            kind=kind, created=datetime.now(timezone.utc).isoformat(), payload=json.dumps(payload,
            ensure_ascii=False, allow_nan=False, default=str)))


def history(patient=None):
    query = select(entries).order_by(entries.c.created, entries.c.id)
    if patient is not None:
        query = query.where(entries.c.patient == patient.strip())
    with engine.connect() as connection:
        return [dict(row, payload=json.loads(row["payload"])) for row in connection.execute(query).mappings()]
