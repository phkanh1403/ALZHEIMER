import json
import os
import sqlite3
import zlib
from contextlib import closing
from pathlib import Path
from datetime import datetime, timezone
from etl import digest

def pack(obj):
    return zlib.compress(json.dumps(obj,ensure_ascii=False,allow_nan=False).encode())
def unpack(data):
    return json.loads(zlib.decompress(data))
def db():
    root=Path(os.environ.get("ADNI_HOME","adni_local")); root.mkdir(exist_ok=True,parents=True)
    con=sqlite3.connect(root/"research.sqlite",timeout=60)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS items(kind TEXT,id TEXT,label TEXT,created TEXT,payload BLOB, PRIMARY KEY(kind,id));
    CREATE TABLE IF NOT EXISTS active(slot TEXT PRIMARY KEY,id TEXT);
    """)
    return con
def save(kind,label,obj,activate=False):
    payload=pack(obj); key=digest(payload)
    with closing(db()) as c, c:
        c.execute("INSERT OR IGNORE INTO items VALUES(?,?,?,?,?)",(kind,key,label,datetime.now(timezone.utc).isoformat(),payload))
        if activate: c.execute("INSERT OR REPLACE INTO active VALUES(?,?)",(kind,key))
    return key
def select(kind,key):
    with closing(db()) as c,c:
        if not c.execute("SELECT 1 FROM items WHERE kind=? AND id=?",(kind,key)).fetchone(): raise ValueError("Không tìm thấy phiên bản")
        c.execute("INSERT OR REPLACE INTO active VALUES(?,?)",(kind,key))
def current(kind):
    with closing(db()) as c:
        r=c.execute("SELECT id FROM active WHERE slot=?",(kind,)).fetchone()
        return r[0] if r else None
def get(kind,key):
    with closing(db()) as c:
        r=c.execute("SELECT payload FROM items WHERE kind=? AND id=?",(kind,key)).fetchone()
        if not r: raise ValueError("Thiếu phiên bản")
        return unpack(r[0])
def listing(kind):
    with closing(db()) as c:
        return c.execute("SELECT id,label,created FROM items WHERE kind=? ORDER BY created DESC",(kind,)).fetchall()
