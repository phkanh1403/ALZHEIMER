"""Unit tests of outcome status, Turnbull estimation, net benefit and the decision rule."""
import numpy as np
import pandas as pd
import pytest

from mci_risk.config import DELTA_AUC, NB_MARGIN
from mci_risk.cv import folds_by_rid, pair_label
from mci_risk.metrics import categories, decide, nb_key, net_benefit, status, turnbull, turnbull_risk
from mci_risk.model import fit_cox, fit_interval

INF = np.inf


def test_categories():
    L = np.array([0, 800, 200, 100])
    R = np.array([300, INF, 900, INF])
    assert list(categories(L, R, 730)) == ['case_certain', 'control_certain', 'ambiguous', 'censored_early']


@pytest.fixture(scope='module')
def tiny():
    frame = pd.DataFrame(dict(RID=range(6), tau=0.5, age=[70, 72, 75, 68, 80, 77], mmse=[27, 26, 25, 28, 24, 26],
                              adas13=[15, 18, 20, 12, 25, 19], faq=[2, 3, 5, 1, 8, 4],
                              L=[0, 800, 500, 100, 182, 365], R=[300, INF, 900, INF, 547, INF]))
    event = np.isfinite(frame.R.to_numpy())
    cm = fit_cox(frame, ['age'], np.where(event, frame.R, frame.L), ~event)
    return frame, fit_interval(frame, ['age']), cm


@pytest.mark.parametrize(('mode', 'is_case'), [('L', 1.0), ('mid', 1.0), ('R', 0.0)])
def test_status_places_crossing_onsets_by_mode(tiny, mode, is_case):
    frame, ref, cm = tiny
    y, w, cat = status(frame, 730, ref, np.random.default_rng(0), cm, mode=mode)
    assert cat[2] == 'ambiguous' and y[2] == is_case     # interval (500, 900]: midpoint 700 <= 730
    assert y[0] == 1 and y[1] == 0 and w[3] == 0          # certain case, certain control, censored before h
    assert (w[[0, 1, 2]] >= 1).all()


def test_status_rejects_unknown_mode(tiny):
    frame, ref, cm = tiny
    with pytest.raises(ValueError, match='mode'):
        status(frame, 730, ref, np.random.default_rng(0), cm, mode='middle')


def test_turnbull_equals_empirical_distribution_for_narrow_intervals():
    onsets = np.array([100, 200, 300, 400])
    _, _, s = turnbull(onsets - 1, onsets)
    assert np.allclose(s, 0.25)
    assert turnbull_risk(onsets - 1, onsets, 250) == pytest.approx((0.5, 0.5, 0.5))


def test_turnbull_risk_brackets_uncertain_mass():
    lower, upper, mid = turnbull_risk(np.array([0.0, 0.0]), np.array([500.0, 1000.0]), 730)
    assert lower <= mid <= upper


def test_net_benefit_by_hand():
    y, w, p = np.array([1, 0, 1, 0]), np.ones(4), np.array([0.9, 0.8, 0.1, 0.05])
    # flagged: one case and one non-case; NB = (1 - 1 * 0.2 / 0.8) / 4
    assert net_benefit(y, w, p, 0.2) == pytest.approx((1 - 0.25) / 4)


def test_nb_key():
    assert [nb_key(t) for t in (0.05, 0.15, 0.3)] == ['nb_5', 'nb_15', 'nb_30']


def test_decision_rule():
    assert decide(2 * DELTA_AUC, 0.001, 0.05, 0.01, 0.001, 0.02) == 'Added value'
    assert decide(0.0, -DELTA_AUC / 2, DELTA_AUC / 2, 0.0, -0.01, NB_MARGIN) == 'Equivalent'
    assert decide(0.01, -0.03, 0.05, 0.0, -0.01, 0.02) == 'Inconclusive'


def test_folds_split_participants_evenly_and_reproducibly():
    folds = folds_by_rid(range(103), 5, seed=1)
    sizes = np.bincount(list(folds.values()))
    assert set(folds) == set(range(103)) and sizes.max() - sizes.min() <= 1
    assert folds == folds_by_rid(range(103), 5, seed=1) != folds_by_rid(range(103), 5, seed=2)


def test_pair_label():
    assert pair_label('M3', 'M2') == 'M3 − M2'
