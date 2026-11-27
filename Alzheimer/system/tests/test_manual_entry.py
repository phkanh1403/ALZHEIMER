"""Hand-entered patients: trends from history tables, and rejection of entries the models were not trained on."""
import pandas as pd
import pytest

from mci_risk.config import MODELS
from mci_risk.manual_entry import InvalidEntry, empty_cognitive_history, empty_mri_history, manual_features

VISIT = dict(visit='2026-01-20', mci_years=2, age=74, sex='Female', education=16, mmse=26, faq=4, adas13=19, ravlt=3)
MRI = dict(mri_date='2026-01-10', hv=6400.0, icv=1.5e6, entor=3.1, field='3T')


def cog(dates, adas=None, faq=None):
    n = len(dates)
    return pd.DataFrame({'Date': pd.to_datetime(dates), 'ADAS-Cog13': adas or [14.0] * n, 'FAQ': faq or [1.0] * n})


def mri(dates, field='3T'):
    n = len(dates)
    return pd.DataFrame({'Date': pd.to_datetime(dates), 'Hippocampal volume (mm³)': [6600.0] * n,
                         'Intracranial volume (mm³)': [1.5e6] * n, 'Entorhinal thickness (mm)': [3.3] * n,
                         'Field strength': [field] * n})


def test_history_tables_give_all_m3_inputs(prep):
    f = manual_features(prep, **VISIT, **MRI, cog_history=cog(['2024-01-10', '2025-01-12'], [14.0, 16.0], [1.0, 2.0]),
                        mri_history=mri(['2024-01-15']))
    assert all(f.get(c) is not None for c in MODELS['M3'])
    assert f['mri_span_days'] >= 300 and f['hippo_slope'] < 0


def test_short_mri_history_gives_no_atrophy_trend(prep):
    f = manual_features(prep, **VISIT, **MRI, mri_history=mri(['2025-10-01']))
    assert 'hippo_slope' not in f and f['mri_span_days'] < 300


def test_empty_editor_tables_are_ignored(prep):
    f = manual_features(prep, **VISIT, **MRI, cog_history=empty_cognitive_history(), mri_history=empty_mri_history())
    assert 'adas13_slope' not in f and 'hippo_slope' not in f


def test_without_mri_only_clinical_inputs(prep):
    f = manual_features(prep, **VISIT)
    assert 'hippo_adj' not in f and f['mmse'] == 26


@pytest.mark.parametrize(('changes', 'message'), [
    (dict(cog_history=cog(['2027-06-01'])), 'before the visit'),            # history after the visit
    (dict(cog_history=cog(['2026-01-20'])), 'before the visit'),            # history on the visit day
    (dict(mri_date='2023-01-10'), 'days older than the visit'),             # MRI older than the training window
    (dict(mri_date='2026-06-01'), 'after the visit'),                       # MRI after the visit
    (dict(mri_history=mri(['2024-01-15'], field='1.5T')), 'field strength'),
    (dict(mri_history=mri(['2026-02-01'])), 'before the current MRI'),
    (dict(hv=None), 'together'),                                            # incomplete current MRI
    (dict(mmse=31), 'outside the range'),
    (dict(cog_history=pd.DataFrame({'Date': [pd.NaT], 'ADAS-Cog13': [14.0], 'FAQ': [None]})), 'needs a date'),
])
def test_entries_the_models_were_not_trained_on_are_rejected(prep, changes, message):
    with pytest.raises(InvalidEntry, match=message):
        manual_features(prep, **{**VISIT, **MRI, **changes})


def test_mri_history_without_current_mri_is_rejected(prep):
    with pytest.raises(InvalidEntry, match='need a current MRI'):
        manual_features(prep, **VISIT, mri_history=mri(['2024-01-15']))
