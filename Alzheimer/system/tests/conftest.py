"""Shared fixtures. Tests that need the ADNI training data or the results in runs/ are skipped when absent."""
import pytest

from mci_risk.config import RUNS_DIR, TRAIN_DIR
from mci_risk.data import Sources, attach_labels, eligible_points
from mci_risk.files import load_json

HAS_DATA = (TRAIN_DIR / 'Clinical_Data' / 'DXSUM.csv').exists()
HAS_RUNS = (RUNS_DIR / 'preprocessing.json').exists()
HAS_COHORT = (RUNS_DIR / 'train_inputs.csv').exists()     # participant-level files: local copies only
needs_data = pytest.mark.skipif(not HAS_DATA, reason='ADNI training data not in data/train')
needs_runs = pytest.mark.skipif(not HAS_RUNS, reason='run train.py first')
needs_cohort = pytest.mark.skipif(not HAS_COHORT, reason='participant-level files are not distributed')


@pytest.fixture(scope='session')
def src():
    if not HAS_DATA:
        pytest.skip('ADNI training data not in data/train')
    return Sources(TRAIN_DIR)


@pytest.fixture(scope='session')
def points(src):
    return eligible_points(src.vis)


@pytest.fixture(scope='session')
def labelled(src, points):
    return attach_labels(points, src.vis)


@pytest.fixture(scope='session')
def prep():
    if not HAS_RUNS:
        pytest.skip('run train.py first')
    return load_json(RUNS_DIR / 'preprocessing.json')
