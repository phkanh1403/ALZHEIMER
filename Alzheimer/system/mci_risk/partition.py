"""The frozen 80/20 split by participant: which side (train/test) a RID belongs to.

Listed RIDs follow data/partition.csv; any other RID (present only in auxiliary tables) is assigned by a fixed
multiplicative hash, so the same RID always lands on the same side. Needs no ADNI data, so the app can tell
whether an imported participant was part of the training data even in the distributed release.
"""
import csv
from functools import cache

from .config import PARTITION_FILE

TEST_PERCENT = 20
_HASH_MULTIPLIER = 2654435761      # Knuth's multiplicative hash constant


@cache
def load_partition(path=PARTITION_FILE):
    with open(path, newline='', encoding='utf-8') as f:
        return {int(r['RID']): r['set'] for r in csv.DictReader(f)}


def side(rid, partition=None):
    """'train' or 'test'."""
    partition = load_partition() if partition is None else partition
    rid = int(rid)
    if rid in partition:
        return partition[rid]
    return 'test' if (rid * _HASH_MULTIPLIER) % 100 < TEST_PERCENT else 'train'


def in_training(rid):
    return side(rid) == 'train'