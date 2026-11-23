"""Reproduce the frozen RID partition from the extracted original ZIP."""
import argparse
import csv
import shutil
from sklearn.model_selection import train_test_split
from pipeline import Path, read, diagnoses, pd

def split(source, train, test):
    files = sorted(source.rglob('*.csv'))
    roster = set()
    for f in files:
        if f.stem != 'DATADIC':
            roster.update(read(source,f.stem).RID.astype(int))
    landmarks,_ = diagnoses(read(source,'DXSUM'))
    strata = {rid:'other' for rid in roster}
    for r in landmarks.itertuples():
        strata[r.RID] = ('event' if r.event else 'censored') if r.days else 'no_followup'
    ids = sorted(roster)
    a,b = train_test_split(ids,test_size=.2,random_state=42,stratify=[strata[i] for i in ids])
    held = set(b)
    for dest in (train,test):
        if dest.exists(): raise FileExistsError(dest)
    for f in files:
        paths = [dest/f.relative_to(source) for dest in (train,test)]
        for path in paths: path.parent.mkdir(parents=True,exist_ok=True)
        if f.stem=='DATADIC':
            for path in paths: shutil.copyfile(f,path)
            continue
        with f.open(encoding='latin1',newline='') as src, paths[0].open('w',encoding='latin1',newline='') as left, paths[1].open('w',encoding='latin1',newline='') as right:
            reader=csv.reader(src)
            header=next(reader)
            writers=[csv.writer(left),csv.writer(right)]
            for writer in writers: writer.writerow(header)
            for row in reader: writers[int(int(row[header.index('RID')]) in held)].writerow(row)
    print('Train',len(a),'test',len(b))
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['source','train','test']:parser.add_argument(key,type=Path)
    args=parser.parse_args()
    split(args.source,args.train,args.test)
