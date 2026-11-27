"""Unit tests of the plain-language descriptions shown on the patient page."""
import numpy as np
import pandas as pd
import pytest

from mci_risk.narrative import (
    ModelEstimate,
    PatientResult,
    comparison_note,
    drivers_caption,
    patient_summary,
    top_factors,
    trajectory_caption,
)
from mci_risk.risk import RISK_BINS, risk_distribution, share_below

LABELS = {'adas13': 'ADAS-Cog13 score', 'faq': 'FAQ score (0-30)', 'ravlt_delay': 'Delayed recall', 'age': 'Age'}
BANDS = {'low': 'Lower risk', 'mid': 'Intermediate risk', 'high': 'Higher risk'}


@pytest.fixture
def contrib():
    return pd.Series({'adas13': 0.6, 'faq': 0.3, 'ravlt_delay': -0.4, 'age': -0.1})


def test_top_factors_by_direction(contrib):
    assert top_factors(contrib, LABELS) == (['ADAS-Cog13 score', 'FAQ score'], ['Delayed recall', 'Age'])


def test_patient_summary_reports_numbers_band_and_drivers(contrib):
    r = PatientResult('RID 7', '2012-05-01', 'Clinical', '2 years', 0.34, 0.26, 0.43, 'high', 'Higher risk', 0.78)
    text = patient_summary(r, contrib, LABELS, 0.15, 0.30)
    for part in ('**34%**', '95% range 26%–43%', 'higher risk', '30% or more', 'higher than 78%',
                 'ADAS-Cog13 score and FAQ score', 'band is uncertain', 'Model: Clinical'):
        assert part in text


def test_patient_summary_without_range(contrib):
    r = PatientResult('Ann', '2024-01-02', 'Clinical', '1 year', 0.08, None, None, 'low', 'Lower risk')
    text = patient_summary(r, contrib, LABELS, 0.15, 0.30)
    assert '95% range' not in text and 'below 15%' in text and 'uncertain' not in text


def test_drivers_caption_names_largest_effects(contrib):
    text = drivers_caption(contrib, LABELS, 4)
    assert 'upwards comes from ADAS-Cog13 score' in text and 'downwards from Delayed recall' in text


def test_trajectory_caption_trend():
    assert 'rose from 10% to 30%' in trajectory_caption([0.10, 0.2, 0.30], '2 years')
    assert 'stayed about the same' in trajectory_caption([0.20, 0.22], '2 years')


def _estimates(risks, bands, spread=0.02):
    labels = ['Clinical', '+ MRI', '+ cognitive trend', '+ atrophy trend']
    return [ModelEstimate(lab, r, r - spread, r + spread, b) for lab, r, b in zip(labels, risks, bands, strict=True)]


def test_comparison_note_largest_step_and_band_disagreement():
    text = comparison_note(_estimates([0.12, 0.20, 0.21, 0.22], ['low', 'mid', 'mid', 'mid']), BANDS)
    assert 'moving to **+ MRI**' in text and 'raises the estimate by 8 percentage points' in text
    assert 'disagree on the band' in text and 'overlap' not in text


def test_comparison_note_small_changes_and_overlap():
    text = comparison_note(_estimates([0.20, 0.205, 0.21, 0.20], ['mid'] * 4, spread=0.05), BANDS)
    assert 'less than 2 percentage points' in text and 'All four models place' in text and 'overlap' in text


def test_risk_distribution_is_aggregated_and_ranks_correctly():
    model = dict(kind='pwexp_interval', columns=[], mean=[], scale=[], beta=[],
                 log_hazard=list(np.log(np.full(9, 1e-3))))
    dist = risk_distribution(model, pd.DataFrame(index=range(50)), [365])['365']
    assert set(dist) == {'n', 'quantiles', 'counts'} and dist['n'] == 50
    assert sum(dist['counts']) == 50 and len(dist['counts']) == len(RISK_BINS) - 1
    assert share_below(dist, 0.0) == 0.0 and share_below(dist, 1.0) == 1.0
