"""Split the original ADNI dataset into 80% training / 20% test by participant (RID).

The split is frozen in data/partition.csv (4,020 training and 1,006 test participants).
Participants not listed in partition.csv (present only in auxiliary tables)
are assigned by a fixed hash: about 20% go to test.

Usage:
    python split_dataset.py <original_data_folder> <train_folder> <test_folder>
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_partition(path=ROOT / 'data' / 'partition.csv'):
    with open(path, newline='', encoding='utf-8') as f:
        return {int(r['RID']): r['set'] for r in csv.DictReader(f)}


def assign(rid, partition):
    if rid in partition:
        return partition[rid]
    return 'test' if (rid * 2654435761) % 100 < 20 else 'train'


def rid_of(row, header):
    if 'RID' in header:
        value = row[header.index('RID')].strip()
        return int(float(value)) if value else None
    if 'subject_id' in header:  # All_Subjects_Structural_MRI_Images.csv: PTID like 002_S_0295
        m = re.search(r'_S_(\d+)$', row[header.index('subject_id')].strip())
        return int(m.group(1)) if m else None
    raise KeyError('No RID or subject_id column found')


def split(source, train, test):
    partition = load_partition()
    for dest in (train, test):
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f'{dest} already exists and is not empty')
    counts = {'train': set(), 'test': set()}
    for f in sorted(source.rglob('*.csv')):
        rel = f.relative_to(source)
        outs = {k: d / rel for k, d in (('train', train), ('test', test))}
        for p in outs.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        if f.stem == 'DATADIC':
            for p in outs.values():
                shutil.copyfile(f, p)
            continue
        with f.open(encoding='latin1', newline='') as src, \
                outs['train'].open('w', encoding='latin1', newline='') as a, \
                outs['test'].open('w', encoding='latin1', newline='') as b:
            reader = csv.reader(src)
            header = next(reader)
            writers = {'train': csv.writer(a), 'test': csv.writer(b)}
            for w in writers.values():
                w.writerow(header)
            for row in reader:
                rid = rid_of(row, header)
                if rid is None:
                    continue
                side = assign(rid, partition)
                counts[side].add(rid)
                writers[side].writerow(row)
    overlap = counts['train'] & counts['test']
    if overlap:
        raise RuntimeError(f'{len(overlap)} participants appear on both sides')
    print(f"Train: {len(counts['train'])} RID · Test: {len(counts['test'])} RID")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'train', 'test']:
        p.add_argument(key, type=Path)
    a = p.parse_args()
    split(a.source, a.train, a.test)
