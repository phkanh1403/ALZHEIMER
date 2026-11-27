"""Every app page renders without errors, also when some results are missing (Streamlit AppTest)."""
import pytest
from streamlit.testing.v1 import AppTest

from mci_risk.config import ROOT
from mci_risk.ui import common
from tests.conftest import needs_cohort, needs_runs

PAGES = ['Patient risk', 'Add patient', 'Patient list', 'Model performance', 'Study cohort']
pytestmark = needs_runs


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'DB_PATH', tmp_path / 'profiles.sqlite')
    common.patient_store.clear()
    yield AppTest.from_file(str(ROOT / 'app.py'), default_timeout=120)
    common.patient_store.clear()


def go(at, page):
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.mark.parametrize('page', PAGES)
def test_page_renders(app, page):
    go(app.run(), page)


@needs_cohort
def test_cohort_participant_risk(app):
    at = go(app.run(), 'Patient risk')
    at.radio(key='patient_source').set_value('ADNI study cohort').run()
    assert not at.exception
    assert any('Risk of dementia diagnosis' in m.value for m in at.markdown)
    assert any('estimated risk of a dementia diagnosis' in m.value for m in at.markdown)   # overview paragraph
    at.expander[-1].open = True
    assert not at.exception


def test_release_mode_without_participant_files(app, monkeypatch, tmp_path):
    """With only distributable results (no ADNI rows), prediction pages work and the cohort page explains why."""
    runs = tmp_path / 'runs'
    (runs / 'cv').mkdir(parents=True)
    for p in common.RUNS_DIR.glob('*'):
        if p.is_file() and p.name not in ('train_inputs.csv', 'train_landmarks.csv', 'conflicts.csv'):
            (runs / p.name).symlink_to(p)
    for p in (common.RUNS_DIR / 'cv').glob('*'):
        (runs / 'cv' / p.name).symlink_to(p)
    monkeypatch.setattr(common, 'RUNS_DIR', runs)
    common.run_csv.clear()
    common.run_json.clear()
    try:
        at = go(app.run(), 'Study cohort')
        assert any('Data Use Agreement' in i.value for i in at.info)
        at = go(at, 'Patient risk')
        assert [r.options for r in at.radio if r.key == 'patient_source'] == [['My patients']]
        go(at, 'Model performance')
    finally:
        common.run_csv.clear()
        common.run_json.clear()


def test_add_patient_then_view_risk(app):
    at = go(app.run(), 'Add patient')
    at.text_input[0].input('Test patient')
    at.checkbox[0].check()
    at.button[0].click().run()
    assert not at.exception and at.success, [e.value for e in at.error]
    at = go(at, 'Patient risk')
    assert any('Risk of dementia diagnosis' in m.value for m in at.markdown)


def test_performance_page_without_test_results(app, monkeypatch, tmp_path):
    runs = tmp_path / 'runs'
    (runs / 'cv').mkdir(parents=True)
    for p in common.RUNS_DIR.glob('*'):
        if p.is_file() and not p.name.startswith(('test_', 'summary')):
            (runs / p.name).symlink_to(p)
    for p in (common.RUNS_DIR / 'cv').glob('*'):
        (runs / 'cv' / p.name).symlink_to(p)
    monkeypatch.setattr(common, 'RUNS_DIR', runs)
    common.run_csv.clear()
    common.run_json.clear()
    try:
        at = go(app.run(), 'Model performance')
        assert any('evaluate.py' in i.value for i in at.info)
    finally:
        common.run_csv.clear()
        common.run_json.clear()
