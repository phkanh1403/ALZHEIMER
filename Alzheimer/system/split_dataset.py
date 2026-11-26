"""Split the original ADNI dataset into 80% training / 20% test by participant (RID).

The split is frozen in data/partition.csv (4,020 training and 1,006 test participants).
Participants not listed in partition.csv (present only in auxiliary tables) are assigned by a fixed
multiplicative hash: about 20% go to test.

Usage:
    python split_dataset.py <original_data_folder> <train_folder> <test_folder>
"""
import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

from mci_risk.config import PARTITION_FILE

TEST_PERCENT = 20
_HASH_MULTIPLIER = 2654435761      # Knuth's multiplicative hash constant
_COPY_WHOLE = {'DATADIC'}          # dictionaries without participant rows go to both sides
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def load_partition(path=PARTITION_FILE):
    with open(path, newline='', encoding='utf-8') as f:
        return {int(r['RID']): r['set'] for r in csv.DictReader(f)}


def assign(rid, partition):
    if rid in partition:
        return partition[rid]
    return 'test' if (rid * _HASH_MULTIPLIER) % 100 < TEST_PERCENT else 'train'


def rid_of(row, header):
    if 'RID' in header:
        value = row[header.index('RID')].strip()
        return int(float(value)) if value else None
    if 'subject_id' in header:  # All_Subjects_Structural_MRI_Images.csv: PTID like 002_S_0295
        m = re.search(r'_S_(\d+)$', row[header.index('subject_id')].strip())
        return int(m.group(1)) if m else None
    raise KeyError('No RID or subject_id column found')


def _split_table(path, outs, partition, seen):
    with path.open(encoding='latin1', newline='') as src, \
            outs['train'].open('w', encoding='latin1', newline='') as train, \
            outs['test'].open('w', encoding='latin1', newline='') as test:
        reader = csv.reader(src)
        header = next(reader)
        writers = {'train': csv.writer(train), 'test': csv.writer(test)}
        for w in writers.values():
            w.writerow(header)
        for row in reader:
            rid = rid_of(row, header)
            if rid is None:
                continue
            side = assign(rid, partition)
            seen[side].add(rid)
            writers[side].writerow(row)


def split(source, train, test):
    partition = load_partition()
    for dest in (train, test):
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f'{dest} already exists and is not empty')
    seen = {'train': set(), 'test': set()}
    for path in sorted(source.rglob('*.csv')):
        rel = path.relative_to(source)
        outs = {'train': train / rel, 'test': test / rel}
        for p in outs.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        if path.stem in _COPY_WHOLE:
            for p in outs.values():
                shutil.copyfile(path, p)
        else:
            _split_table(path, outs, partition, seen)
    overlap = seen['train'] & seen['test']
    if overlap:
        raise RuntimeError(f'{len(overlap)} participants appear on both sides')
    print(f"Train: {len(seen['train'])} RID · Test: {len(seen['test'])} RID")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for key in ['source', 'train', 'test']:
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    split(args.source, args.train, args.test)


if __name__ == '__main__':
    main()
