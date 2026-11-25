import json
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile
from pipeline import MODELS, cohort, pd

FIELDS=MODELS['M3']
LABELS=dict(zip(FIELDS,['Tuổi (năm)','Giới ADNI (1 nam, 2 nữ)','Học vấn (năm)','MMSE (điểm)','FAQ (điểm)','Tổng hồi hải mã (mm³)','Độ dày nội khứu (mm)','Tốc độ MMSE (điểm/năm)','Tốc độ FAQ (điểm/năm)','Tốc độ hồi hải mã (mm³/năm)','Tốc độ nội khứu (mm/năm)']))
BOUNDS=dict(zip(FIELDS,[(18,115),(1,2),(0,36),(0,30),(0,30),(0.1,15000.),(0.01,6.),(-122.,122.),(-122.,122.),(-5000.,5000.),(-5.,5.)]))
TABLES=['DXSUM','MMSE','FAQ','PTDEMOG','UCSFFSX7']

def records(frame,ids,prefix):
    by_id=frame.set_index('RID')
    for rid in sorted(ids):
        features=dict.fromkeys(FIELDS)
        day=''
        if rid in by_id.index:
            row=by_id.loc[rid]
            day=str(pd.Timestamp(row.t0).date())
            features.update({k:float(row[k]) for k in FIELDS if pd.notna(row[k])})
        yield (f'{prefix}:{rid}',f'{prefix} · RID {rid}',day,json.dumps(features))

def roster(folder):
    ids=set()
    for table in TABLES:
        path=next(folder.rglob(table+'.csv'))
        ids.update(pd.read_csv(path,usecols=['RID']).RID.astype(int))
    return ids

def save(db,rows):
    start=db.total_changes
    db.executemany('INSERT OR IGNORE INTO profiles(id,name,day,features) VALUES(?,?,?,?)',rows)
    return db.total_changes-start

def initialize(db,root,frame):
    db.execute('CREATE TABLE IF NOT EXISTS migrations(name TEXT PRIMARY KEY)')
    with db:
        if db.execute("INSERT OR IGNORE INTO migrations VALUES('seed_train')").rowcount:
            save(db,records(frame,roster(root/'data/train'),'Train'))

def import_zip(db,file):
    with tempfile.TemporaryDirectory() as tmp, ZipFile(file) as archive:
        folder=Path(tmp)
        for table in TABLES:
            filename=table+'.csv'
            matches=[i for i in archive.infolist() if Path(i.filename).name==filename]
            if len(matches)!=1 or matches[0].file_size>200_000_000:
                raise ValueError('Thiếu, trùng hoặc vượt giới hạn kích thước: '+filename)
            with archive.open(matches[0]) as src,(folder/filename).open('wb') as dest:
                shutil.copyfileobj(src,dest)
        frame,_=cohort(folder)
        ids=roster(folder)
        with db:
            known={int(r[0].split(':')[1]) for r in db.execute("SELECT id FROM profiles WHERE id LIKE 'Train:%'")}
            return save(db,records(frame,ids-known,'ZIP')),len(ids)
