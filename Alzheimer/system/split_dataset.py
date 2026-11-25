"""Chia bộ dữ liệu ADNI gốc thành train 80% / test 20% theo RID.

Phân hoạch được đóng băng trong data/partition.csv (giữ nguyên 4.020 RID train và
1.006 RID test của phiên bản v1). RID không có trong partition.csv (chỉ xuất hiện ở
các bảng phụ) được gán bằng một hàm băm cố định: khoảng 20% vào test.

Cách dùng:
    python split_dataset.py <thư_mục_dữ_liệu_gốc> <thư_mục_train> <thư_mục_test>
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
    with open(path, newline='') as f:
        return {int(r['RID']): r['set'] for r in csv.DictReader(f)}


def assign(rid, partition):
    if rid in partition:
        return partition[rid]
    return 'test' if (rid * 2654435761) % 100 < 20 else 'train'


def rid_of(row, header):
    if 'RID' in header:
        value = row[header.index('RID')].strip()
        return int(float(value)) if value else None
    if 'subject_id' in header:  # All_Subjects_Structural_MRI_Images.csv: PTID dạng 002_S_0295
        m = re.search(r'_S_(\d+)$', row[header.index('subject_id')].strip())
        return int(m.group(1)) if m else None
    raise KeyError('Không tìm thấy cột RID hoặc subject_id')


def split(source, train, test):
    partition = load_partition()
    for dest in (train, test):
        if dest.exists() and any(dest.iterdir()):
            raise FileExistsError(f'{dest} đã tồn tại và không rỗng')
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
        raise RuntimeError(f'{len(overlap)} RID xuất hiện ở cả hai phía')
    print(f"Train: {len(counts['train'])} RID · Test: {len(counts['test'])} RID")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for key in ['source', 'train', 'test']:
        p.add_argument(key, type=Path)
    a = p.parse_args()
    split(a.source, a.train, a.test)
