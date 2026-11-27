"""What may leave the project: building a distributable copy without participant-level ADNI data.

The ADNI Data Use Agreement does not allow redistributing ADNI data. The release therefore keeps the code, the
fitted models and aggregated results, and drops every table with one row per participant or visit. A final scan
refuses to build if any remaining table still has a participant identifier column.
"""
import fnmatch
import gzip
import json
import zipfile
from pathlib import Path

# participant-level files (ADNI data or rows derived from them)
RESTRICTED = [
    'data/train/*', 'data/test/*',
    'runs/train_landmarks.csv', 'runs/train_inputs.csv', 'runs/conflicts.csv',
    'runs/cv/*_oof.csv*',
    'instance/*',
]
# local build and tool files
EXCLUDED = ['*/__pycache__/*', '__pycache__/*', '.venv/*', '.pytest_cache/*', '.ruff_cache/*', 'dist/*', '*.tmp',
            '*.pyc', 'docs/*.aux', 'docs/*.log', 'docs/*.out', 'docs/build/*']
# identifiers that mark a participant-level table
ID_COLUMNS = {'RID', 'PTID', 'subject_id', 'IMAGEUID', 'image_id'}
# the frozen split lists participant numbers only (no ADNI measurements) and is needed to reproduce it
ALLOWED_ID_FILES = {'data/partition.csv'}


class RestrictedDataError(RuntimeError):
    """A file selected for the release still contains participant-level data."""


def _matches(rel, patterns):
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def release_files(root):
    """Relative paths (POSIX) of the files that go into the release, sorted."""
    root = Path(root)
    out = []
    for path in root.rglob('*'):
        rel = path.relative_to(root).as_posix()
        if path.is_file() and not _matches(rel, RESTRICTED + EXCLUDED):
            out.append(rel)
    return sorted(out)


def _csv_header(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='latin1') as f:
        return {c.strip().strip('"') for c in f.readline().split(',')}


def _json_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _json_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_keys(v)


def identifier_columns(path):
    """Participant identifier columns/keys found in a CSV (optionally gzipped) or JSON file."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(('.csv', '.csv.gz')):
        return sorted(_csv_header(path) & ID_COLUMNS)
    if name.endswith('.json'):
        return sorted(set(_json_keys(json.loads(path.read_text(encoding='utf-8')))) & ID_COLUMNS)
    return []


def check(root, files):
    """Raise RestrictedDataError when any file to be released holds participant identifiers."""
    problems = {}
    for rel in files:
        ids = [] if rel in ALLOWED_ID_FILES else identifier_columns(Path(root) / rel)
        if ids:
            problems[rel] = ids
    if problems:
        listing = '; '.join(f'{rel}: {", ".join(ids)}' for rel, ids in problems.items())
        raise RestrictedDataError(f'participant-level data found, release not built: {listing}')


def build(root, target, prefix):
    """Write the release ZIP (files under `prefix/`); returns the list of released files."""
    files = release_files(root)
    check(root, files)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(Path(root) / rel, f'{prefix}/{rel}')
    return files
